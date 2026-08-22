"""全模式 MCP 写工具必须落入真实数据库，而不是 Fake server。"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.models  # noqa: F401
from app.models.base import Base
from app.models.runtime import HrTicketRecord
from app.schemas.user import UserContext
from app.services.mcp.sqlalchemy_server import SqlAlchemyMcpServer


@pytest_asyncio.fixture()
async def session_factory():  # type: ignore[no-untyped-def]
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_full_mcp_ticket_is_persisted_and_tenant_scoped(session_factory) -> None:
    server = SqlAlchemyMcpServer(session_factory)
    user = UserContext(
        user_id="user_hr",
        tenant_id="tenant_a",
        department_ids=["dept_hr"],
        role="hr",
        permissions=["hr.ticket.write"],
    )

    result = await server.call_tool(
        "create_hr_ticket",
        {"title": "办理入职", "description": "为新员工建立工单"},
        user,
    )

    async with session_factory() as session:
        stored = await session.scalar(
            select(HrTicketRecord).where(HrTicketRecord.id == result["ticket_id"])
        )
    assert stored is not None
    assert stored.tenant_id == "tenant_a"
    assert stored.created_by == "user_hr"
    assert stored.title == "办理入职"


def test_full_runtime_selects_persistent_mcp_server() -> None:
    from app.api.dependencies import _build_mcp_server

    class Settings:
        app_mode = "full"

    server_type = type(_build_mcp_server(Settings(), object()))
    assert server_type.__name__ == "SqlAlchemyMcpServer"
