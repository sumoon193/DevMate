"""
工具调用表模型。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IDMixin, TimestampMixin
from app.schemas.enums import ToolCallStatus

if TYPE_CHECKING:
    from app.models.agent_run import AgentRun
    from app.models.approval import ApprovalRequest


class ToolCall(Base, IDMixin, TimestampMixin):
    """
    工具调用表。

    记录单次工具调用的参数、结果和状态。
    """

    __tablename__ = "tool_calls"

    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("agent_runs.id"), comment="所属 Run ID"
    )
    tool_name: Mapped[str] = mapped_column(String(128), comment="工具名称")
    parameters: Mapped[dict] = mapped_column(JSON, comment="调用参数")
    result: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, default=None, comment="执行结果"
    )
    status: Mapped[str] = mapped_column(
        String(32), default=ToolCallStatus.PENDING.value, comment="调用状态"
    )
    approval_required: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否需要审批")

    # 关系
    run: Mapped[AgentRun] = relationship(back_populates="tool_calls")
    approval_request: Mapped[ApprovalRequest | None] = relationship(
        back_populates="tool_call", uselist=False
    )


class ToolDefinition(Base, TimestampMixin):
    """
    工具定义表。

    存储工具元数据和风险等级。
    """

    __tablename__ = "tool_definitions"

    name: Mapped[str] = mapped_column(String(128), primary_key=True, comment="工具名称（主键）")
    description: Mapped[str] = mapped_column(Text, comment="工具描述")
    permission_scope: Mapped[str] = mapped_column(String(128), comment="权限范围")
    risk_level: Mapped[str] = mapped_column(String(32), comment="风险等级")
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否需要审批")
    timeout_seconds: Mapped[int] = mapped_column(default=30, comment="超时时间（秒）")
    idempotent: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否幂等")
    parameters_schema: Mapped[dict] = mapped_column(JSON, default=dict, comment="参数 JSON Schema")
