"""
入库任务表模型。

持久化文档入库过程的每个阶段状态。
"""

from __future__ import annotations

from sqlalchemy import JSON, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IDMixin, TimestampMixin


class IngestionTaskRecord(Base, IDMixin, TimestampMixin):
    """
    入库任务表。

    记录文档从上传到入库完成的完整生命周期。
    """

    __tablename__ = "ingestion_tasks"

    document_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="关联的文档 ID",
    )
    filename: Mapped[str] = mapped_column(
        String(255),
        comment="原始文件名",
    )
    mime_type: Mapped[str] = mapped_column(
        String(128),
        comment="MIME 类型",
    )
    storage_key: Mapped[str] = mapped_column(
        String(512),
        default="",
        comment="MinIO 对象键",
    )
    current_stage: Mapped[str] = mapped_column(
        String(32),
        default="queued",
        comment="当前阶段",
    )
    progress: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        comment="进度 0.0 ~ 1.0",
    )
    total_chunks: Mapped[int] = mapped_column(
        default=0,
        comment="总分块数",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        comment="错误信息",
    )
    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        default=None,
        comment="错误码",
    )
    stages_json: Mapped[list] = mapped_column(
        JSON,
        default=list,
        comment="阶段执行记录 JSON",
    )
