"""
审批表模型。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.agent_run import AgentRun
from app.models.base import Base, IDMixin, TimestampMixin
from app.models.tool_call import ToolCall
from app.schemas.enums import ApprovalDecisionType, ApprovalStatus


class ApprovalRequest(Base, IDMixin, TimestampMixin):
    """
    审批请求表。

    存储工具审批的请求和决策。
    """

    __tablename__ = "approval_requests"

    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("agent_runs.id"), comment="所属 Run ID"
    )
    tool_call_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("tool_calls.id"), comment="工具调用 ID"
    )
    tool_name: Mapped[str] = mapped_column(String(128), comment="工具名称")
    parameters: Mapped[dict] = mapped_column(JSON, comment="工具参数")
    expected_effect: Mapped[str] = mapped_column(String(512), comment="预期影响描述")
    evidence: Mapped[list] = mapped_column(JSON, default=list, comment="相关证据")
    risk_level: Mapped[str] = mapped_column(String(32), comment="风险等级")
    options: Mapped[list[str]] = mapped_column(
        JSON,
        default=lambda: [
            ApprovalDecisionType.APPROVE.value,
            ApprovalDecisionType.EDIT.value,
            ApprovalDecisionType.REJECT.value,
        ],
        comment="审批选项",
    )
    status: Mapped[str] = mapped_column(
        String(32), default=ApprovalStatus.PENDING.value, comment="审批状态"
    )
    decision: Mapped[str | None] = mapped_column(
        String(32), nullable=True, default=None, comment="审批决策"
    )
    decided_by: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None, comment="审批人用户 ID"
    )
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None, comment="审批时间（UTC）"
    )
    revision: Mapped[int] = mapped_column(Integer, default=1, comment="审批修订号")
    subject_hash: Mapped[str] = mapped_column(String(64), default="", comment="审批对象哈希")
    requested_by: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="审批发起人"
    )
    requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="审批发起时间"
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="审批失效时间"
    )
    policy_version: Mapped[str] = mapped_column(String(128), default="", comment="绑定策略版本")
    execution_manifest_hash: Mapped[str] = mapped_column(
        String(128), default="", comment="绑定执行清单哈希"
    )
    supersedes_approval_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="被替代审批 ID"
    )
    revoked_by: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="撤销人")
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="撤销时间"
    )
    revoke_reason: Mapped[str | None] = mapped_column(Text, nullable=True, comment="撤销原因")

    # 关系
    run: Mapped[AgentRun] = relationship(back_populates="approval_requests")
    tool_call: Mapped[ToolCall] = relationship(back_populates="approval_request")
