"""
文档表模型。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IDMixin, TimestampMixin
from app.schemas.enums import DocumentStatus, Visibility

if TYPE_CHECKING:
    from app.models.chunk import DocumentChunk


class Document(Base, IDMixin, TimestampMixin):
    """
    文档表。

    存储上传文档的元数据和 ACL 信息。
    """

    __tablename__ = "documents"

    title: Mapped[str] = mapped_column(String(255), comment="文档标题")
    file_path: Mapped[str] = mapped_column(String(512), comment="MinIO 对象键")
    mime_type: Mapped[str] = mapped_column(String(128), comment="MIME 类型")
    status: Mapped[str] = mapped_column(
        String(32), default=DocumentStatus.QUEUED.value, comment="入库状态"
    )
    tenant_id: Mapped[str] = mapped_column(String(64), comment="租户 ID")
    department_id: Mapped[str] = mapped_column(String(64), comment="部门 ID")
    visibility: Mapped[str] = mapped_column(
        String(32), default=Visibility.DEPARTMENT.value, comment="可见性级别"
    )
    metadata_: Mapped[dict] = mapped_column(JSON, default=dict, comment="扩展元数据")

    # 关系
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", order_by="DocumentChunk.id"
    )
