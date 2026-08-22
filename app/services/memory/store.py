"""Episodic Memory Store 的 ACL-aware deterministic fallback。"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import NotFoundError, PermissionError, ValidationError
from app.models.runtime import EpisodicMemoryRecordORM
from app.schemas.enums import MemoryStatus
from app.schemas.memory import EpisodicMemoryRecord
from app.services.runtime.clock import Clock, SystemClock
from app.services.security.prompt_guard import PromptGuard


def _normalize_content(content: str) -> str:
    normalized = " ".join(content.split())
    if not normalized:
        raise ValidationError("Episodic memory content is required")
    return normalized


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.casefold().encode("utf-8")).hexdigest()


def _as_utc(value: datetime | None) -> datetime | None:
    """统一数据库驱动返回的 naive/aware 时间，避免 SQLite 测试与 PG 行为分叉。"""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _merge_provenance(existing: list[str], incoming: list[str]) -> list[str]:
    return list(dict.fromkeys([*existing, *incoming]))


def _validate_memory_input(
    provenance_event_ids: list[str],
    ttl_seconds: int | None,
    importance_score: float,
) -> None:
    if not provenance_event_ids:
        raise ValidationError("Episodic memory requires provenance events")
    if ttl_seconds is not None and ttl_seconds <= 0:
        raise ValidationError("Episodic memory TTL must be positive")
    if not 0.0 <= importance_score <= 1.0:
        raise ValidationError("Episodic memory importance_score must be between 0 and 1")


class MemorySemanticIndex(Protocol):
    """按租户隔离的长期记忆语义索引边界。"""

    async def upsert(self, *, memory_id: str, tenant_id: str, content: str) -> None: ...

    async def search(
        self, *, tenant_id: str, query: str, limit: int
    ) -> list[tuple[str, float]]: ...

    async def delete(self, *, memory_id: str, tenant_id: str) -> None: ...


class EpisodicMemoryStore(Protocol):
    """带 ACL、provenance 和删除语义的长期记忆边界。"""

    async def remember(
        self,
        *,
        tenant_id: str,
        case_id: str,
        memory_key: str,
        content: str,
        provenance_event_ids: list[str],
        ttl_seconds: int | None = None,
        importance_score: float = 0.5,
    ) -> EpisodicMemoryRecord: ...

    async def search(
        self,
        *,
        tenant_id: str,
        query: str,
    ) -> list[EpisodicMemoryRecord]: ...

    async def forget(
        self,
        memory_id: str,
        *,
        tenant_id: str,
    ) -> EpisodicMemoryRecord: ...

    async def get(
        self,
        memory_id: str,
        *,
        tenant_id: str,
    ) -> EpisodicMemoryRecord: ...

    async def list_records(
        self,
        *,
        tenant_id: str,
    ) -> list[EpisodicMemoryRecord]: ...


class InMemoryEpisodicMemoryStore:
    """隔离租户、保留 provenance 并支持 quarantine/forget 的内存实现。"""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        prompt_guard: PromptGuard | None = None,
        semantic_index: MemorySemanticIndex | None = None,
    ) -> None:
        self._clock = clock or SystemClock()
        self._prompt_guard = prompt_guard or PromptGuard()
        self._semantic_index = semantic_index
        self._records: dict[str, EpisodicMemoryRecord] = {}

    async def remember(
        self,
        *,
        tenant_id: str,
        case_id: str,
        memory_key: str,
        content: str,
        provenance_event_ids: list[str],
        ttl_seconds: int | None = None,
        importance_score: float = 0.5,
    ) -> EpisodicMemoryRecord:
        """写入带来源的经验，注入内容只进入 quarantine。"""
        _validate_memory_input(provenance_event_ids, ttl_seconds, importance_score)
        is_injection, reason = self._prompt_guard.detect_injection(content)
        now = self._clock.now()
        normalized = _normalize_content(content)
        content_hash = _content_hash(normalized)
        for existing in self._records.values():
            self._expire_if_needed(existing, now)
            if (
                self._semantic_index is not None
                and existing.status in {MemoryStatus.EXPIRED, MemoryStatus.DELETED}
            ):
                await self._semantic_index.delete(
                    memory_id=existing.id,
                    tenant_id=existing.tenant_id,
                )
            if (
                existing.tenant_id == tenant_id
                and existing.memory_key == memory_key
                and existing.content_hash == content_hash
                and existing.status == MemoryStatus.ACTIVE
                and not is_injection
            ):
                existing.provenance_event_ids = _merge_provenance(
                    existing.provenance_event_ids, provenance_event_ids
                )
                existing.importance_score = max(existing.importance_score, importance_score)
                existing.updated_at = now
                return existing.model_copy(deep=True)
        record = EpisodicMemoryRecord(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            tenant_id=tenant_id,
            case_id=case_id,
            memory_key=memory_key,
            content=normalized,
            content_hash=content_hash,
            provenance_event_ids=provenance_event_ids,
            importance_score=importance_score,
            status=(MemoryStatus.QUARANTINED if is_injection else MemoryStatus.ACTIVE),
            poisoning_reason=reason or None,
            created_at=now,
            updated_at=now,
            expires_at=(now + timedelta(seconds=ttl_seconds) if ttl_seconds is not None else None),
        )
        self._records[record.id] = record
        if self._semantic_index is not None and record.status == MemoryStatus.ACTIVE:
            await self._semantic_index.upsert(
                memory_id=record.id,
                tenant_id=tenant_id,
                content=record.content,
            )
        return record.model_copy(deep=True)

    async def search(
        self,
        *,
        tenant_id: str,
        query: str,
    ) -> list[EpisodicMemoryRecord]:
        """融合词面、语义和重要性，只召回当前租户有效记忆。"""
        now = self._clock.now()
        query_terms = [term for term in query.lower().split() if term]
        semantic_scores: dict[str, float] = {}
        if self._semantic_index is not None:
            semantic_scores = dict(
                await self._semantic_index.search(tenant_id=tenant_id, query=query, limit=50)
            )
        ranked: list[tuple[float, EpisodicMemoryRecord]] = []
        for record in self._records.values():
            self._expire_if_needed(record, now)
            if (
                self._semantic_index is not None
                and record.status in {MemoryStatus.EXPIRED, MemoryStatus.DELETED}
            ):
                await self._semantic_index.delete(
                    memory_id=record.id,
                    tenant_id=record.tenant_id,
                )
            if record.tenant_id != tenant_id or record.status != MemoryStatus.ACTIVE:
                continue
            content = record.content.lower()
            lexical = 1.0 if query.lower() in content else 0.0
            if query_terms:
                lexical = max(
                    lexical,
                    sum(term in content for term in query_terms) / len(query_terms),
                )
            semantic = semantic_scores.get(record.id, 0.0)
            if lexical <= 0.0 and semantic <= 0.0:
                continue
            score = 0.7 * max(lexical, semantic) + 0.3 * record.importance_score
            record.access_count += 1
            record.last_accessed_at = now
            record.updated_at = now
            ranked.append((score, record))
        ranked.sort(key=lambda item: (-item[0], -item[1].importance_score, item[1].id))
        return [record.model_copy(deep=True) for _, record in ranked]

    async def forget(
        self,
        memory_id: str,
        *,
        tenant_id: str,
    ) -> EpisodicMemoryRecord:
        """逻辑删除记忆并保留审计记录。"""
        record = self._authorized(memory_id, tenant_id)
        record.status = MemoryStatus.DELETED
        record.updated_at = self._clock.now()
        if self._semantic_index is not None:
            await self._semantic_index.delete(memory_id=memory_id, tenant_id=tenant_id)
        return record.model_copy(deep=True)

    async def get(
        self,
        memory_id: str,
        *,
        tenant_id: str,
    ) -> EpisodicMemoryRecord:
        """按租户查询单条记忆，包括已删除状态。"""
        record = self._authorized(memory_id, tenant_id)
        self._expire_if_needed(record, self._clock.now())
        if (
            self._semantic_index is not None
            and record.status in {MemoryStatus.EXPIRED, MemoryStatus.DELETED}
        ):
            await self._semantic_index.delete(memory_id=memory_id, tenant_id=tenant_id)
        return record.model_copy(deep=True)

    async def list_records(
        self,
        *,
        tenant_id: str,
    ) -> list[EpisodicMemoryRecord]:
        """列出当前租户记忆，包含隔离、过期和已删除状态。"""
        now = self._clock.now()
        records: list[EpisodicMemoryRecord] = []
        for record in self._records.values():
            if record.tenant_id != tenant_id:
                continue
            expired = self._expire_if_needed(record, now)
            if expired and self._semantic_index is not None:
                await self._semantic_index.delete(
                    memory_id=record.id,
                    tenant_id=tenant_id,
                )
            records.append(record.model_copy(deep=True))
        return sorted(records, key=lambda record: (record.created_at, record.id), reverse=True)

    @staticmethod
    def _expire_if_needed(record: EpisodicMemoryRecord, now: datetime) -> bool:
        if (
            record.status == MemoryStatus.ACTIVE
            and record.expires_at is not None
            and record.expires_at <= now
        ):
            record.status = MemoryStatus.EXPIRED
            record.updated_at = now
            return True
        return False

    def _authorized(self, memory_id: str, tenant_id: str) -> EpisodicMemoryRecord:
        record = self._records.get(memory_id)
        if record is None:
            raise NotFoundError(f"Memory not found: {memory_id}")
        if record.tenant_id != tenant_id:
            raise PermissionError(f"Memory tenant mismatch: {memory_id}")
        return record


class SqlAlchemyEpisodicMemoryStore:
    """PostgreSQL/SQLAlchemy episodic memory adapter。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        clock: Clock | None = None,
        prompt_guard: PromptGuard | None = None,
        semantic_index: MemorySemanticIndex | None = None,
    ) -> None:
        self._sessions = session_factory
        self._clock = clock or SystemClock()
        self._prompt_guard = prompt_guard or PromptGuard()
        self._semantic_index = semantic_index

    async def remember(
        self,
        *,
        tenant_id: str,
        case_id: str,
        memory_key: str,
        content: str,
        provenance_event_ids: list[str],
        ttl_seconds: int | None = None,
        importance_score: float = 0.5,
    ) -> EpisodicMemoryRecord:
        _validate_memory_input(provenance_event_ids, ttl_seconds, importance_score)
        is_injection, reason = self._prompt_guard.detect_injection(content)
        now = self._clock.now()
        normalized = _normalize_content(content)
        content_hash = _content_hash(normalized)
        stored: EpisodicMemoryRecordORM
        created = False
        expired_memory_id: str | None = None
        async with self._sessions() as session, session.begin():
            existing = None
            if not is_injection:
                existing = await session.scalar(
                    select(EpisodicMemoryRecordORM).where(
                        EpisodicMemoryRecordORM.tenant_id == tenant_id,
                        EpisodicMemoryRecordORM.memory_key == memory_key,
                        EpisodicMemoryRecordORM.content_hash == content_hash,
                        EpisodicMemoryRecordORM.status == MemoryStatus.ACTIVE.value,
                    )
                )
            existing_expires_at = _as_utc(existing.expires_at) if existing is not None else None
            if existing is not None and (existing_expires_at is None or existing_expires_at > now):
                existing.provenance_event_ids = _merge_provenance(
                    list(existing.provenance_event_ids), provenance_event_ids
                )
                existing.importance_score = max(existing.importance_score, importance_score)
                existing.updated_at = now
                stored = existing
            else:
                if existing is not None:
                    existing.status = MemoryStatus.EXPIRED.value
                    existing.updated_at = now
                    expired_memory_id = existing.id
                stored = EpisodicMemoryRecordORM(
                    id=f"mem_{uuid.uuid4().hex[:12]}",
                    tenant_id=tenant_id,
                    case_id=case_id,
                    memory_key=memory_key,
                    content=normalized,
                    content_hash=content_hash,
                    provenance_event_ids=provenance_event_ids,
                    importance_score=importance_score,
                    status=(
                        MemoryStatus.QUARANTINED.value
                        if is_injection
                        else MemoryStatus.ACTIVE.value
                    ),
                    poisoning_reason=reason or None,
                    created_at=now,
                    updated_at=now,
                    expires_at=(
                        now + timedelta(seconds=ttl_seconds) if ttl_seconds is not None else None
                    ),
                )
                session.add(stored)
                created = True
        result = self._from_record(stored)
        if expired_memory_id is not None and self._semantic_index is not None:
            await self._semantic_index.delete(
                memory_id=expired_memory_id,
                tenant_id=tenant_id,
            )
        if created and self._semantic_index is not None and result.status == MemoryStatus.ACTIVE:
            await self._semantic_index.upsert(
                memory_id=result.id,
                tenant_id=tenant_id,
                content=result.content,
            )
        return result

    async def search(
        self,
        *,
        tenant_id: str,
        query: str,
    ) -> list[EpisodicMemoryRecord]:
        now = self._clock.now()
        expired_memory_ids: list[str] = []
        async with self._sessions() as session, session.begin():
            records = list(
                (
                    await session.scalars(
                        select(EpisodicMemoryRecordORM).where(
                            EpisodicMemoryRecordORM.tenant_id == tenant_id,
                            EpisodicMemoryRecordORM.status.in_(
                                [
                                    MemoryStatus.ACTIVE.value,
                                    MemoryStatus.EXPIRED.value,
                                    MemoryStatus.DELETED.value,
                                ]
                            ),
                        )
                    )
                ).all()
            )
            for record in records:
                expires_at = _as_utc(record.expires_at)
                if (
                    record.status == MemoryStatus.ACTIVE.value
                    and expires_at is not None
                    and expires_at <= now
                ):
                    record.status = MemoryStatus.EXPIRED.value
                    record.updated_at = now
                if record.status in {
                    MemoryStatus.EXPIRED.value,
                    MemoryStatus.DELETED.value,
                }:
                    expired_memory_ids.append(record.id)
        if self._semantic_index is not None:
            for memory_id in expired_memory_ids:
                await self._semantic_index.delete(
                    memory_id=memory_id,
                    tenant_id=tenant_id,
                )
        query_terms = [term for term in query.lower().split() if term]
        semantic_scores: dict[str, float] = {}
        if self._semantic_index is not None:
            semantic_scores = dict(
                await self._semantic_index.search(tenant_id=tenant_id, query=query, limit=50)
            )
        ranked: list[tuple[float, EpisodicMemoryRecordORM]] = []
        for record in records:
            if record.status != MemoryStatus.ACTIVE.value:
                continue
            content = record.content.lower()
            lexical = 1.0 if query.lower() in content else 0.0
            if query_terms:
                lexical = max(
                    lexical,
                    sum(term in content for term in query_terms) / len(query_terms),
                )
            semantic = semantic_scores.get(record.id, 0.0)
            if lexical <= 0.0 and semantic <= 0.0:
                continue
            score = 0.7 * max(lexical, semantic) + 0.3 * record.importance_score
            record.access_count += 1
            record.last_accessed_at = now
            record.updated_at = now
            ranked.append((score, record))
        ranked.sort(key=lambda item: (-item[0], -item[1].importance_score, item[1].id))
        if ranked:
            access_state = {
                record.id: (record.access_count, record.last_accessed_at, record.updated_at)
                for _, record in ranked
            }
            async with self._sessions() as session, session.begin():
                persisted = (
                    await session.scalars(
                        select(EpisodicMemoryRecordORM).where(
                            EpisodicMemoryRecordORM.id.in_(access_state)
                        )
                    )
                ).all()
                for record in persisted:
                    count, last_accessed_at, updated_at = access_state[record.id]
                    record.access_count = count
                    record.last_accessed_at = last_accessed_at
                    record.updated_at = updated_at
        return [self._from_record(record) for _, record in ranked]

    async def forget(
        self,
        memory_id: str,
        *,
        tenant_id: str,
    ) -> EpisodicMemoryRecord:
        async with self._sessions() as session, session.begin():
            record = await self._authorized(session, memory_id, tenant_id)
            record.status = MemoryStatus.DELETED.value
            record.updated_at = self._clock.now()
        result = self._from_record(record)
        if self._semantic_index is not None:
            await self._semantic_index.delete(memory_id=memory_id, tenant_id=tenant_id)
        return result

    async def get(
        self,
        memory_id: str,
        *,
        tenant_id: str,
    ) -> EpisodicMemoryRecord:
        async with self._sessions() as session:
            async with session.begin():
                record = await self._authorized(session, memory_id, tenant_id)
                now = self._clock.now()
                expires_at = _as_utc(record.expires_at)
                if (
                    record.status == MemoryStatus.ACTIVE.value
                    and expires_at is not None
                    and expires_at <= now
                ):
                    record.status = MemoryStatus.EXPIRED.value
                    record.updated_at = now
                result = self._from_record(record)
        if (
            self._semantic_index is not None
            and result.status in {MemoryStatus.EXPIRED, MemoryStatus.DELETED}
        ):
            await self._semantic_index.delete(memory_id=memory_id, tenant_id=tenant_id)
        return result

    async def list_records(
        self,
        *,
        tenant_id: str,
    ) -> list[EpisodicMemoryRecord]:
        """从持久化存储读取租户记忆，并同步 TTL 过期状态。"""
        now = self._clock.now()
        expired_ids: list[str] = []
        async with self._sessions() as session, session.begin():
            records = list(
                (
                    await session.scalars(
                        select(EpisodicMemoryRecordORM)
                        .where(EpisodicMemoryRecordORM.tenant_id == tenant_id)
                        .order_by(EpisodicMemoryRecordORM.created_at.desc())
                    )
                ).all()
            )
            for record in records:
                expires_at = _as_utc(record.expires_at)
                if (
                    record.status == MemoryStatus.ACTIVE.value
                    and expires_at is not None
                    and expires_at <= now
                ):
                    record.status = MemoryStatus.EXPIRED.value
                    record.updated_at = now
                    expired_ids.append(record.id)
        if self._semantic_index is not None:
            for memory_id in expired_ids:
                await self._semantic_index.delete(memory_id=memory_id, tenant_id=tenant_id)
        return [self._from_record(record) for record in records]

    @staticmethod
    async def _authorized(
        session: AsyncSession,
        memory_id: str,
        tenant_id: str,
    ) -> EpisodicMemoryRecordORM:
        record = await session.get(EpisodicMemoryRecordORM, memory_id)
        if record is None:
            raise NotFoundError(f"Memory not found: {memory_id}")
        if record.tenant_id != tenant_id:
            raise PermissionError(f"Memory tenant mismatch: {memory_id}")
        return record

    @staticmethod
    def _from_record(record: EpisodicMemoryRecordORM) -> EpisodicMemoryRecord:
        created_at = record.created_at
        updated_at = record.updated_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        expires_at = _as_utc(record.expires_at)
        last_accessed_at = _as_utc(record.last_accessed_at)
        return EpisodicMemoryRecord(
            id=record.id,
            tenant_id=record.tenant_id,
            case_id=record.case_id,
            memory_key=record.memory_key,
            content=record.content,
            content_hash=record.content_hash,
            provenance_event_ids=list(record.provenance_event_ids),
            importance_score=record.importance_score,
            access_count=record.access_count,
            last_accessed_at=last_accessed_at,
            status=MemoryStatus(record.status),
            poisoning_reason=record.poisoning_reason,
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
        )
