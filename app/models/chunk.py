"""
分块表模型。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import JSON, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IDMixin, TimestampMixin
from app.schemas.enums import Visibility

if TYPE_CHECKING:
    from app.models.document import Document


class DocumentChunk(Base, IDMixin, TimestampMixin):
    """
    文档分块表。

    存储文档解析后的分块，支持 Parent-Child 关系。
    """

    __tablename__ = "document_chunks"

    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id"), comment="所属文档 ID"
    )
    chunk_text: Mapped[str] = mapped_column(Text, comment="原始分块文本")
    context_prefix: Mapped[str] = mapped_column(Text, default="", comment="生成的上下文前缀")
    full_text: Mapped[str] = mapped_column(
        Text, default="", comment="context_prefix + chunk_text（用于 embedding）"
    )
    parent_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("document_chunks.id"),
        nullable=True,
        default=None,
        comment="Parent chunk ID",
    )
    chunk_type: Mapped[str] = mapped_column(
        String(16), default="child", comment="分块类型：parent 或 child"
    )
    heading_path: Mapped[str] = mapped_column(String(512), default="", comment="标题路径")
    page_numbers: Mapped[list[int]] = mapped_column(JSON, default=list, comment="所在页码列表")
    token_count: Mapped[int] = mapped_column(Integer, default=0, comment="分块 token 数量")
    # ACL 元数据
    tenant_id: Mapped[str] = mapped_column(String(64), comment="租户 ID")
    department_id: Mapped[str] = mapped_column(String(64), comment="部门 ID")
    visibility: Mapped[Visibility] = mapped_column(Enum(Visibility), comment="可见性级别")
    acl_metadata: Mapped[dict] = mapped_column(JSON, default=dict, comment="扩展 ACL 元数据")

    # 关系
    document: Mapped[Document] = relationship(back_populates="chunks")
    parent: Mapped[DocumentChunk | None] = relationship(
        back_populates="children", remote_side="DocumentChunk.id"
    )
    children: Mapped[list[DocumentChunk]] = relationship(back_populates="parent")
