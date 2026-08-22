"""
评测表模型。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IDMixin, TimestampMixin


class EvalCase(Base, IDMixin, TimestampMixin):
    """
    评测用例表。

    存储问题-答案对，用于 RAGAS 评测。
    """

    __tablename__ = "eval_cases"

    question: Mapped[str] = mapped_column(Text, comment="问题")
    answer: Mapped[str] = mapped_column(Text, comment="标准答案")
    contexts: Mapped[list[str]] = mapped_column(JSON, comment="相关上下文列表")
    ground_truth_docs: Mapped[list[str]] = mapped_column(
        JSON, default=list, comment="相关文档名称列表"
    )
    ground_truth_sections: Mapped[list[str]] = mapped_column(
        JSON, default=list, comment="相关章节列表"
    )
    expected_tools: Mapped[list[str]] = mapped_column(
        JSON, default=list, comment="预期调用的工具列表"
    )
    requires_approval: Mapped[bool] = mapped_column(default=False, comment="是否需要审批")


class EvalRun(Base, IDMixin, TimestampMixin):
    """
    评测运行表。

    存储一次评测任务的元数据和结果。
    """

    __tablename__ = "eval_runs"

    dataset_name: Mapped[str] = mapped_column(String(255), comment="数据集名称")
    metrics: Mapped[dict[str, float]] = mapped_column(JSON, default=dict, comment="评测指标")
    status: Mapped[str] = mapped_column(String(32), default="pending", comment="运行状态")
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None, comment="开始时间（UTC）"
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None, comment="完成时间（UTC）"
    )
