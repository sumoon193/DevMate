"""
Agent Step 表模型。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import JSON, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IDMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.agent_run import AgentRun


class AgentStep(Base, IDMixin, TimestampMixin):
    """
    Agent Step 表。

    记录单个节点的输入输出、证据和 token 消耗。
    """

    __tablename__ = "agent_steps"

    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("agent_runs.id"), comment="所属 Run ID"
    )
    node_name: Mapped[str] = mapped_column(String(64), comment="节点名称")
    input_data: Mapped[dict] = mapped_column(JSON, default=dict, comment="输入数据")
    output_data: Mapped[dict] = mapped_column(JSON, default=dict, comment="输出数据")
    evidence: Mapped[list] = mapped_column(JSON, default=list, comment="相关证据")
    token_usage: Mapped[dict[str, int]] = mapped_column(
        JSON, default=dict, comment="Token 使用情况"
    )
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, comment="执行耗时（毫秒）")

    # 关系
    run: Mapped[AgentRun] = relationship("AgentRun", back_populates="steps")
