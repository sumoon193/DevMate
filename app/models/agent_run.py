"""
Agent Run 表模型。
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IDMixin, TimestampMixin
from app.schemas.enums import RunStatus

if TYPE_CHECKING:
    from app.models.agent_step import AgentStep
    from app.models.approval import ApprovalRequest
    from app.models.tool_call import ToolCall


class AgentRun(Base, IDMixin, TimestampMixin):
    """
    Agent Run 表。

    存储 Agent 执行的完整生命周期。
    """

    __tablename__ = "agent_runs"

    user_id: Mapped[str] = mapped_column(String(64), comment="用户 ID")
    thread_id: Mapped[str] = mapped_column(String(64), comment="LangGraph thread ID")
    original_query: Mapped[str] = mapped_column(Text, comment="原始查询")
    status: Mapped[str] = mapped_column(
        String(32), default=RunStatus.CREATED.value, comment="当前状态"
    )
    result: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, default=None, comment="最终结果（答案、citations 等）"
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None, comment="完成时间（UTC）"
    )

    # 关系
    steps: Mapped[list[AgentStep]] = relationship(
        "AgentStep", back_populates="run", order_by="AgentStep.created_at"
    )
    tool_calls: Mapped[list[ToolCall]] = relationship("ToolCall", back_populates="run")
    approval_requests: Mapped[list[ApprovalRequest]] = relationship(
        "ApprovalRequest", back_populates="run"
    )
