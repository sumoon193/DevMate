"""不可变 DocumentVersion registry 的 deterministic fallback。"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import NotFoundError
from app.models.runtime import DocumentVersionRecord
from app.schemas.document import DocumentVersion


class DocumentVersionRegistry(Protocol):
    """不可变文档版本的持久化边界。"""

    async def register(self, *, document_id: str, content: bytes) -> DocumentVersion: ...

    async def get(self, version_id: str) -> DocumentVersion: ...

    async def get_active(self, document_id: str) -> DocumentVersion: ...


class InMemoryDocumentVersionRegistry:
    """保存历史版本，并在新内容完整注册后切换 active version。"""

    def __init__(self) -> None:
        self._versions: dict[str, DocumentVersion] = {}
        self._by_document: dict[str, list[str]] = {}

    async def register(self, *, document_id: str, content: bytes) -> DocumentVersion:
        """注册不可变内容版本并原子切换 active 标记。"""
        content_hash = hashlib.sha256(content).hexdigest()
        version_ids = self._by_document.setdefault(document_id, [])
        for version_id in version_ids:
            existing = self._versions[version_id]
            if existing.content_hash == content_hash:
                return existing.model_copy(deep=True)

        superseded_id = version_ids[-1] if version_ids else None
        if superseded_id is not None:
            self._versions[superseded_id].is_active = False
        version = DocumentVersion(
            id=f"docver_{uuid.uuid4().hex[:12]}",
            document_id=document_id,
            version=len(version_ids) + 1,
            content_hash=content_hash,
            is_active=True,
            supersedes_version_id=superseded_id,
            created_at=datetime.now(UTC),
        )
        self._versions[version.id] = version
        version_ids.append(version.id)
        return version.model_copy(deep=True)

    async def get(self, version_id: str) -> DocumentVersion:
        """按 ID 查询文档版本。"""
        version = self._versions.get(version_id)
        if version is None:
            raise NotFoundError(f"Document version not found: {version_id}")
        return version.model_copy(deep=True)

    async def get_active(self, document_id: str) -> DocumentVersion:
        """查询当前 active 文档版本。"""
        version_ids = self._by_document.get(document_id, [])
        for version_id in reversed(version_ids):
            version = self._versions[version_id]
            if version.is_active:
                return version.model_copy(deep=True)
        raise NotFoundError(f"Active document version not found: {document_id}")


class SqlAlchemyDocumentVersionRegistry:
    """PostgreSQL/SQLAlchemy 文档版本 registry。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def register(self, *, document_id: str, content: bytes) -> DocumentVersion:
        """幂等注册内容，并在事务内切换同文档 active version。"""
        content_hash = hashlib.sha256(content).hexdigest()
        async with self._sessions() as session, session.begin():
            existing = await session.scalar(
                select(DocumentVersionRecord).where(
                    DocumentVersionRecord.document_id == document_id,
                    DocumentVersionRecord.content_hash == content_hash,
                )
            )
            if existing is not None:
                if not existing.is_active:
                    await session.execute(
                        update(DocumentVersionRecord)
                        .where(DocumentVersionRecord.document_id == document_id)
                        .values(is_active=False)
                    )
                    existing.is_active = True
                return self._from_record(existing)

            versions = list(
                (
                    await session.scalars(
                        select(DocumentVersionRecord)
                        .where(DocumentVersionRecord.document_id == document_id)
                        .order_by(DocumentVersionRecord.version)
                        .with_for_update()
                    )
                ).all()
            )
            for version in versions:
                version.is_active = False
            record = DocumentVersionRecord(
                id=f"docver_{uuid.uuid4().hex[:12]}",
                document_id=document_id,
                version=len(versions) + 1,
                content_hash=content_hash,
                is_active=True,
                supersedes_version_id=versions[-1].id if versions else None,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            session.add(record)
        return self._from_record(record)

    async def get(self, version_id: str) -> DocumentVersion:
        async with self._sessions() as session:
            record = await session.get(DocumentVersionRecord, version_id)
            if record is None:
                raise NotFoundError(f"Document version not found: {version_id}")
            return self._from_record(record)

    async def get_active(self, document_id: str) -> DocumentVersion:
        async with self._sessions() as session:
            record = await session.scalar(
                select(DocumentVersionRecord).where(
                    DocumentVersionRecord.document_id == document_id,
                    DocumentVersionRecord.is_active.is_(True),
                )
            )
            if record is None:
                raise NotFoundError(f"Active document version not found: {document_id}")
            return self._from_record(record)

    @staticmethod
    def _from_record(record: DocumentVersionRecord) -> DocumentVersion:
        created_at = record.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return DocumentVersion(
            id=record.id,
            document_id=record.document_id,
            version=record.version,
            content_hash=record.content_hash,
            is_active=record.is_active,
            supersedes_version_id=record.supersedes_version_id,
            created_at=created_at,
        )
