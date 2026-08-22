"""
FastAPI 应用入口。

挂载路由、注册异常处理器、配置中间件。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api import agent_runs, approvals, cases, console, documents, eval_runs, health, protocols
from app.api.devmate import router as devmate_router
from app.api.devmate import webhook_router as devmate_webhook_router
from app.api.errors import app_error_handler, generic_error_handler
from app.config import get_settings
from app.core.exceptions import AppError

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """为每个请求分配唯一 request_id。"""

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期：full 模式下初始化数据库和外部服务。"""
    settings = get_settings()

    if settings.app_mode == "full":
        from app.services.security.oidc import validate_full_mode_oidc

        validate_full_mode_oidc()
        from app.db.session import init_db

        await init_db()
        logger.info("full_mode_initialized")

    checkpointer_manager = None
    if settings.graph_checkpointer_backend == "postgres":
        from app.api.dependencies import get_container

        checkpointer_manager = get_container().graph_checkpointer
        await checkpointer_manager.setup()
        logger.info("graph_checkpointer_initialized")

    try:
        yield
    finally:
        if checkpointer_manager is not None:
            await checkpointer_manager.teardown()
            logger.info("graph_checkpointer_shutdown")

        if settings.app_mode == "full":
            from app.db.session import close_db

            await close_db()
            logger.info("full_mode_shutdown")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    app = FastAPI(
        title="EnterpriseMind Agent Harness RAG",
        version="0.2.0",
        description="企业制度型长流程 Agent Runtime 与执行治理 API",
        lifespan=lifespan,
    )

    # ── 中间件 ──
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def oidc_full_mode(request: Request, call_next):  # type: ignore[no-untyped-def]
        settings = get_settings()
        if settings.app_mode != "full" or request.url.path in {"/health", "/actuator/health"}:
            return await call_next(request)
        authorization = request.headers.get("authorization", "")
        scheme, _, bearer = authorization.partition(" ")
        if scheme.lower() != "bearer" or not bearer:
            return JSONResponse(status_code=401, content={"detail": "missing bearer token"})
        from app.services.security.oidc import reset_claims, set_claims, verify_bearer

        try:
            claims = await asyncio.to_thread(verify_bearer, bearer)
        except (AppError, ValueError) as exc:
            return JSONResponse(status_code=401, content={"detail": str(exc)})
        context_token = set_claims(claims)
        try:
            return await call_next(request)
        finally:
            reset_claims(context_token)

    # ── 异常处理器 ──
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, generic_error_handler)

    # ── 路由 ──
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(agent_runs.router)
    app.include_router(approvals.router)
    app.include_router(eval_runs.router)
    app.include_router(console.router)
    app.include_router(cases.router)
    app.include_router(protocols.router)
    app.include_router(devmate_router)
    app.include_router(devmate_webhook_router)

    return app


app = create_app()
