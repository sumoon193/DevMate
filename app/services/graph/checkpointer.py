"""
LangGraph checkpointer 工厂与生命周期管理。

settings 驱动选择后端：
- memory（默认/测试）：进程内 MemorySaver，无需 setup/teardown。
- postgres：AsyncPostgresSaver + psycopg 异步连接池，checkpoint 落盘到
  PostgreSQL，使 waiting_approval 的跨天长流程在进程重启后仍可 resume。
  连接池的打开/关闭挂在 FastAPI lifespan 上（见 app/main.py）。
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class GraphCheckpointerManager:
    """
    Checkpointer 生命周期管理器。

    memory 后端 setup/teardown 均为 no-op；postgres 后端在 setup 时
    打开连接池并初始化 checkpoint 表，teardown 时关闭连接池。
    setup/teardown 均幂等，可安全重复调用。
    """

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver,
        backend: str = "memory",
        pool: Any | None = None,
    ) -> None:
        """
        初始化管理器。

        Args:
            checkpointer: 编译 graph 时注入的 checkpointer 实例
            backend: 后端名称（memory / postgres）
            pool: postgres 后端的 psycopg 异步连接池；memory 后端为 None
        """
        self._checkpointer = checkpointer
        self._backend = backend
        self._pool = pool
        self._ready = False

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        """编译 graph 时使用的 checkpointer 实例。"""
        return self._checkpointer

    @property
    def backend(self) -> str:
        """当前后端名称（memory / postgres）。"""
        return self._backend

    async def setup(self) -> None:
        """打开底层连接池并初始化 checkpoint 表（memory 后端为 no-op）。"""
        if self._pool is None or self._ready:
            return
        await self._pool.open()
        await self._checkpointer.setup()
        self._ready = True
        logger.info("graph_checkpointer_ready", extra={"backend": self._backend})

    async def teardown(self) -> None:
        """关闭底层连接池（memory 后端为 no-op）。"""
        if self._pool is None or not self._ready:
            return
        await self._pool.close()
        self._ready = False
        logger.info("graph_checkpointer_closed", extra={"backend": self._backend})


def create_checkpointer_manager(settings: object) -> GraphCheckpointerManager:
    """
    根据配置构建 checkpointer 管理器。

    graph_checkpointer_backend=postgres 且有连接串时使用 AsyncPostgresSaver
    （连接串优先取 graph_checkpointer_postgres_url，为空则回退 postgres_url）；
    其余情况一律回退进程内 MemorySaver，测试环境无需 Postgres。

    Args:
        settings: 项目 Settings（或提供同名字段的兼容对象）

    注意：postgres 分支必须在事件循环内调用（AsyncPostgresSaver 构造时
    绑定当前 running loop），FastAPI lifespan / 请求上下文均满足该条件。

    Returns:
        GraphCheckpointerManager；postgres 后端须在异步上下文中调用 setup()。

    Raises:
        RuntimeError: 配置了 postgres 后端但缺少 langgraph-checkpoint-postgres
            依赖，或在无事件循环的同步上下文中构建 postgres checkpointer。
    """
    backend = getattr(settings, "graph_checkpointer_backend", "memory")
    if getattr(settings, "app_mode", "fallback") == "full" and backend != "postgres":
        raise ValidationError("APP_MODE=full requires graph_checkpointer_backend=postgres")
    conn_string = getattr(settings, "graph_checkpointer_postgres_url", "") or getattr(
        settings, "postgres_url", ""
    )

    if backend == "postgres" and conn_string:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool
        except ImportError as exc:
            raise RuntimeError(
                "graph_checkpointer_backend=postgres 需要安装 "
                "langgraph-checkpoint-postgres（含 psycopg / psycopg-pool）依赖，"
                "请执行 uv sync 或 pip install langgraph-checkpoint-postgres"
            ) from exc

        # open=False：连接池在 FastAPI lifespan 的 setup() 中才真正打开，
        # 构造阶段不产生任何网络 IO。
        pool = AsyncConnectionPool(
            conninfo=conn_string,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )
        try:
            saver = AsyncPostgresSaver(pool)
        except RuntimeError as exc:
            # AsyncPostgresSaver 构造时调用 asyncio.get_running_loop()，
            # 在纯同步上下文中会抛 "no running event loop"。
            raise RuntimeError(
                "postgres checkpointer 必须在事件循环内构建"
                "（如 FastAPI lifespan），不能在纯同步上下文中调用"
            ) from exc
        logger.info("graph_checkpointer_selected", extra={"backend": "postgres"})
        return GraphCheckpointerManager(
            checkpointer=saver,
            backend="postgres",
            pool=pool,
        )

    from langgraph.checkpoint.memory import MemorySaver

    logger.info("graph_checkpointer_selected", extra={"backend": "memory"})
    return GraphCheckpointerManager(checkpointer=MemorySaver(), backend="memory")
