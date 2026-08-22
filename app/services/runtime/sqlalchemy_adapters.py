"""PostgreSQL/SQLAlchemy Runtime 持久化 adapters。"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import NotFoundError, ValidationError
from app.models.runtime import (
    CaseRecord,
    DurableTimerRecord,
    OutboxRecord,
    RuntimeAggregateRecord,
    RuntimeEventRecord,
    RuntimeLeaseRecord,
    SideEffectLedgerRecord,
)
from app.schemas.enums import CaseStatus, SideEffectStatus, TimerStatus
from app.schemas.runtime import (
    DurableTimer,
    ExecutionManifest,
    HRCase,
    OutboxMessage,
    RunEventEnvelope,
    RunLease,
    SideEffectRecord,
)
from app.services.observability.runtime_metrics import RuntimeMetrics
from app.services.runtime.event_store import InMemoryEventStore


def _utc(value: datetime) -> datetime:
    """SQLite 会丢失时区；统一恢复为 UTC aware datetime。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _event_from_record(record: RuntimeEventRecord) -> RunEventEnvelope:
    return RunEventEnvelope(
        id=record.id,
        aggregate_id=record.aggregate_id,
        aggregate_type=record.aggregate_type,
        sequence=record.sequence,
        event_type=record.event_type,
        payload=dict(record.payload),
        command_id=record.command_id,
        actor_id=record.actor_id,
        created_at=_utc(record.created_at),
        schema_version=record.schema_version,
        prev_hash=record.prev_hash,
        event_hash=record.event_hash,
    )


def _outbox_from_record(record: OutboxRecord) -> OutboxMessage:
    return OutboxMessage(
        id=record.id,
        event_id=record.event_id,
        aggregate_id=record.aggregate_id,
        sequence=record.sequence,
        topic=record.topic,
        payload=dict(record.payload),
        created_at=_utc(record.created_at),
        published_at=_utc(record.published_at) if record.published_at else None,
        claimed_by=record.claimed_by,
        claimed_at=_utc(record.claimed_at) if record.claimed_at else None,
        delivery_attempts=record.delivery_attempts,
    )


class SqlAlchemyEventStore:
    """用事务、行锁和唯一约束实现 Event Store + Outbox。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        metrics: RuntimeMetrics | None = None,
    ) -> None:
        self._sessions = session_factory
        self._metrics = metrics

    async def append(
        self,
        *,
        aggregate_id: str,
        aggregate_type: str,
        event_type: str,
        payload: dict[str, Any],
        command_id: str,
        expected_version: int,
        actor_id: str,
    ) -> RunEventEnvelope:
        """在同一事务内追加 event、推进 aggregate version 并写 outbox。"""
        fingerprint = InMemoryEventStore._command_fingerprint(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            event_type=event_type,
            payload=payload,
            actor_id=actor_id,
        )
        async with self._sessions() as session, session.begin():
            prior = await session.scalar(
                select(RuntimeEventRecord).where(RuntimeEventRecord.command_id == command_id)
            )
            if prior is not None:
                prior_fingerprint = InMemoryEventStore._command_fingerprint(
                    aggregate_id=prior.aggregate_id,
                    aggregate_type=prior.aggregate_type,
                    event_type=prior.event_type,
                    payload=dict(prior.payload),
                    actor_id=prior.actor_id,
                )
                if prior_fingerprint != fingerprint:
                    raise ValidationError(f"Command id reused with different content: {command_id}")
                return _event_from_record(prior)

            aggregate = await session.scalar(
                select(RuntimeAggregateRecord)
                .where(RuntimeAggregateRecord.id == aggregate_id)
                .with_for_update()
            )
            now = datetime.now(UTC)
            if aggregate is None:
                if expected_version != 0:
                    raise ValidationError(
                        f"Event stream version conflict: expected {expected_version}, actual 0"
                    )
                aggregate = RuntimeAggregateRecord(
                    id=aggregate_id,
                    aggregate_type=aggregate_type,
                    version=0,
                    updated_at=now,
                )
                session.add(aggregate)
                await session.flush()
            if aggregate.aggregate_type != aggregate_type:
                raise ValidationError(f"Aggregate type mismatch: {aggregate_id}")
            if aggregate.version != expected_version:
                raise ValidationError(
                    f"Event stream version conflict: expected {expected_version}, "
                    f"actual {aggregate.version}"
                )

            previous = None
            if aggregate.version:
                previous = await session.scalar(
                    select(RuntimeEventRecord).where(
                        RuntimeEventRecord.aggregate_id == aggregate_id,
                        RuntimeEventRecord.sequence == aggregate.version,
                    )
                )
            sequence = aggregate.version + 1
            event_id = f"evt_{uuid.uuid4().hex[:16]}"
            previous_hash = previous.event_hash if previous is not None else ""
            event_hash = InMemoryEventStore._compute_hash(
                event_id=event_id,
                aggregate_id=aggregate_id,
                aggregate_type=aggregate_type,
                sequence=sequence,
                event_type=event_type,
                payload=payload,
                command_id=command_id,
                actor_id=actor_id,
                created_at=now,
                schema_version=1,
                prev_hash=previous_hash,
            )
            event = RunEventEnvelope(
                id=event_id,
                aggregate_id=aggregate_id,
                aggregate_type=aggregate_type,
                sequence=sequence,
                event_type=event_type,
                payload=payload,
                command_id=command_id,
                actor_id=actor_id,
                created_at=now,
                schema_version=1,
                prev_hash=previous_hash,
                event_hash=event_hash,
            )
            session.add(RuntimeEventRecord(**event.model_dump(mode="python")))
            outbox_id = f"outbox_{uuid.uuid4().hex[:16]}"
            session.add(
                OutboxRecord(
                    id=outbox_id,
                    event_id=event.id,
                    aggregate_id=aggregate_id,
                    sequence=sequence,
                    topic=event_type,
                    payload=event.model_dump(mode="json"),
                    created_at=now,
                    delivery_attempts=0,
                )
            )
            aggregate.version = sequence
            aggregate.updated_at = now

        if self._metrics is not None:
            self._metrics.increment("runtime.events.total")
            self._metrics.set_gauge(
                "runtime.outbox.backlog",
                len(await self.pending_outbox()),
            )
        return event

    async def load_stream(self, aggregate_id: str) -> list[RunEventEnvelope]:
        async with self._sessions() as session:
            records = list(
                (
                    await session.scalars(
                        select(RuntimeEventRecord)
                        .where(RuntimeEventRecord.aggregate_id == aggregate_id)
                        .order_by(RuntimeEventRecord.sequence)
                    )
                ).all()
            )
        return [_event_from_record(record) for record in records]

    async def verify_chain(self, aggregate_id: str) -> bool:
        events = await self.load_stream(aggregate_id)
        previous_hash = ""
        for sequence, event in enumerate(events, start=1):
            if event.sequence != sequence or event.prev_hash != previous_hash:
                return False
            expected = InMemoryEventStore._compute_hash(
                event_id=event.id,
                aggregate_id=event.aggregate_id,
                aggregate_type=event.aggregate_type,
                sequence=event.sequence,
                event_type=event.event_type,
                payload=event.payload,
                command_id=event.command_id,
                actor_id=event.actor_id,
                created_at=event.created_at,
                schema_version=event.schema_version,
                prev_hash=event.prev_hash,
            )
            if expected != event.event_hash:
                return False
            previous_hash = event.event_hash
        return True

    async def pending_outbox(self, *, limit: int = 100) -> list[OutboxMessage]:
        async with self._sessions() as session:
            records = list(
                (
                    await session.scalars(
                        select(OutboxRecord)
                        .where(OutboxRecord.published_at.is_(None))
                        .order_by(OutboxRecord.created_at, OutboxRecord.id)
                        .limit(limit)
                    )
                ).all()
            )
        return [_outbox_from_record(record) for record in records]

    async def claim_outbox(
        self,
        *,
        owner_id: str,
        limit: int,
        claim_ttl_seconds: int = 30,
    ) -> list[OutboxMessage]:
        now = datetime.now(UTC)
        stale_before = now - timedelta(seconds=claim_ttl_seconds)
        async with self._sessions() as session, session.begin():
            records = list(
                (
                    await session.scalars(
                        select(OutboxRecord)
                        .where(
                            OutboxRecord.published_at.is_(None),
                            or_(
                                OutboxRecord.claimed_at.is_(None),
                                OutboxRecord.claimed_at <= stale_before,
                            ),
                        )
                        .order_by(OutboxRecord.created_at, OutboxRecord.id)
                        .limit(limit)
                        .with_for_update(skip_locked=True)
                    )
                ).all()
            )
            for record in records:
                record.claimed_by = owner_id
                record.claimed_at = now
                record.delivery_attempts += 1
        return [_outbox_from_record(record) for record in records]

    async def mark_outbox_published(self, outbox_id: str, *, owner_id: str) -> None:
        async with self._sessions() as session, session.begin():
            record = await session.scalar(
                select(OutboxRecord).where(OutboxRecord.id == outbox_id).with_for_update()
            )
            if record is None:
                raise NotFoundError(f"Outbox message not found: {outbox_id}")
            if record.claimed_by not in {None, owner_id}:
                raise ValidationError(f"Outbox message is claimed by another owner: {outbox_id}")
            if record.published_at is None:
                record.published_at = datetime.now(UTC)
        if self._metrics is not None:
            self._metrics.increment("runtime.outbox.published")
            self._metrics.set_gauge(
                "runtime.outbox.backlog",
                len(await self.pending_outbox()),
            )


class SqlAlchemyCaseProjectionStore:
    """查询优化的 Case projection，按 version 幂等 upsert。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def get(self, case_id: str) -> HRCase | None:
        async with self._sessions() as session:
            record = await session.get(CaseRecord, case_id)
            if record is None:
                return None
            return HRCase(
                id=record.id,
                title=record.title,
                tenant_id=record.tenant_id,
                subject_user_id=record.subject_user_id,
                status=CaseStatus(record.status),
                version=record.version,
                execution_manifest=ExecutionManifest.model_validate(record.execution_manifest),
                policy_versions=dict(record.policy_versions),
                working_memory=dict(record.working_memory),
                active_run_id=record.active_run_id,
                created_at=_utc(record.created_at),
                updated_at=_utc(record.updated_at),
            )

    async def upsert(self, case: HRCase) -> None:
        async with self._sessions() as session, session.begin():
            record = await session.scalar(
                select(CaseRecord).where(CaseRecord.id == case.id).with_for_update()
            )
            if record is None:
                session.add(
                    CaseRecord(
                        id=case.id,
                        title=case.title,
                        tenant_id=case.tenant_id,
                        subject_user_id=case.subject_user_id,
                        status=case.status.value,
                        version=case.version,
                        active_run_id=case.active_run_id,
                        execution_manifest=case.execution_manifest.model_dump(mode="json"),
                        policy_versions=case.policy_versions,
                        working_memory=case.working_memory,
                        created_at=case.created_at,
                        updated_at=case.updated_at,
                    )
                )
                return
            if record.version >= case.version:
                return
            record.title = case.title
            record.status = case.status.value
            record.version = case.version
            record.active_run_id = case.active_run_id
            record.execution_manifest = case.execution_manifest.model_dump(mode="json")
            record.policy_versions = case.policy_versions
            record.working_memory = case.working_memory
            record.updated_at = case.updated_at

    async def list(self, *, limit: int = 100) -> list[HRCase]:
        """按更新时间倒序读取运维队列。"""
        async with self._sessions() as session:
            ids = list(
                (
                    await session.scalars(
                        select(CaseRecord.id)
                        .order_by(CaseRecord.updated_at.desc(), CaseRecord.id.desc())
                        .limit(limit)
                    )
                ).all()
            )
        cases: list[HRCase] = []
        for case_id in ids:
            case = await self.get(case_id)
            if case is not None:
                cases.append(case)
        return cases


class SqlAlchemyLeaseStore:
    """使用行锁和 fencing token 的持久 lease。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def acquire(
        self,
        resource_id: str,
        owner_id: str,
        *,
        ttl_seconds: int,
        now: datetime | None = None,
    ) -> RunLease:
        acquired_at = now or datetime.now(UTC)
        async with self._sessions() as session, session.begin():
            record = await session.scalar(
                select(RuntimeLeaseRecord)
                .where(RuntimeLeaseRecord.resource_id == resource_id)
                .with_for_update()
            )
            if record is not None:
                expires_at = _utc(record.expires_at)
                if expires_at > acquired_at and record.owner_id != owner_id:
                    raise ValidationError(
                        f"Resource already leased: {resource_id} by {record.owner_id}"
                    )
                token = record.fencing_token + 1
                record.owner_id = owner_id
                record.acquired_at = acquired_at
                record.expires_at = acquired_at + timedelta(seconds=ttl_seconds)
                record.fencing_token = token
            else:
                token = 1
                record = RuntimeLeaseRecord(
                    resource_id=resource_id,
                    owner_id=owner_id,
                    acquired_at=acquired_at,
                    expires_at=acquired_at + timedelta(seconds=ttl_seconds),
                    fencing_token=token,
                )
                session.add(record)
        return RunLease(
            resource_id=resource_id,
            owner_id=owner_id,
            acquired_at=acquired_at,
            expires_at=acquired_at + timedelta(seconds=ttl_seconds),
            fencing_token=token,
        )

    async def release(
        self,
        resource_id: str,
        owner_id: str,
        fencing_token: int,
    ) -> None:
        async with self._sessions() as session, session.begin():
            record = await session.get(RuntimeLeaseRecord, resource_id)
            if record is None:
                return
            if record.owner_id != owner_id or record.fencing_token != fencing_token:
                raise ValidationError(f"Lease owner or fencing token mismatch: {resource_id}")
            await session.delete(record)


class SqlAlchemySideEffectLedger:
    """持久化副作用 reservation 与不确定结果。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def reserve(
        self,
        *,
        idempotency_key: str,
        tool_name: str,
        subject_hash: str,
    ) -> SideEffectRecord:
        async with self._sessions() as session, session.begin():
            record = await session.scalar(
                select(SideEffectLedgerRecord)
                .where(SideEffectLedgerRecord.idempotency_key == idempotency_key)
                .with_for_update()
            )
            if record is not None:
                if record.tool_name != tool_name or record.subject_hash != subject_hash:
                    raise ValidationError(
                        f"Idempotency key reused with different side effect: {idempotency_key}"
                    )
                return self._side_effect(record)
            now = datetime.now(UTC)
            record = SideEffectLedgerRecord(
                id=f"effect_{uuid.uuid4().hex[:12]}",
                idempotency_key=idempotency_key,
                tool_name=tool_name,
                subject_hash=subject_hash,
                status=SideEffectStatus.RESERVED.value,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
        return self._side_effect(record)

    async def mark_succeeded(
        self,
        record_id: str,
        result: dict[str, object],
    ) -> SideEffectRecord:
        return await self._update_side_effect(
            record_id,
            status=SideEffectStatus.SUCCEEDED,
            result=result,
            error=None,
        )

    async def mark_unknown(self, record_id: str, error: str) -> SideEffectRecord:
        return await self._update_side_effect(
            record_id,
            status=SideEffectStatus.UNKNOWN,
            result=None,
            error=error,
        )

    async def list_records(self) -> list[SideEffectRecord]:
        async with self._sessions() as session:
            records = list((await session.scalars(select(SideEffectLedgerRecord))).all())
        return [self._side_effect(record) for record in records]

    async def _update_side_effect(
        self,
        record_id: str,
        *,
        status: SideEffectStatus,
        result: dict[str, object] | None,
        error: str | None,
    ) -> SideEffectRecord:
        async with self._sessions() as session, session.begin():
            record = await session.get(SideEffectLedgerRecord, record_id)
            if record is None:
                raise NotFoundError(f"Side effect record not found: {record_id}")
            record.status = status.value
            record.result = result
            record.error = error
            record.updated_at = datetime.now(UTC)
        return self._side_effect(record)

    @staticmethod
    def _side_effect(record: SideEffectLedgerRecord) -> SideEffectRecord:
        return SideEffectRecord(
            id=record.id,
            idempotency_key=record.idempotency_key,
            tool_name=record.tool_name,
            subject_hash=record.subject_hash,
            status=SideEffectStatus(record.status),
            result=dict(record.result) if record.result is not None else None,
            error=record.error,
            created_at=_utc(record.created_at),
            updated_at=_utc(record.updated_at),
        )


class SqlAlchemyTimerStore:
    """使用 skip-locked claim 的 durable timer store。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def schedule(
        self,
        *,
        case_id: str,
        timer_type: str,
        due_at: datetime,
        payload: dict[str, object],
        idempotency_key: str,
    ) -> DurableTimer:
        if due_at.tzinfo is None:
            raise ValidationError("Timer due_at must be timezone-aware")
        async with self._sessions() as session, session.begin():
            record = await session.scalar(
                select(DurableTimerRecord).where(
                    DurableTimerRecord.idempotency_key == idempotency_key
                )
            )
            if record is None:
                now = datetime.now(UTC)
                record = DurableTimerRecord(
                    id=f"timer_{uuid.uuid4().hex[:12]}",
                    case_id=case_id,
                    timer_type=timer_type,
                    due_at=due_at,
                    payload=payload,
                    idempotency_key=idempotency_key,
                    status=TimerStatus.SCHEDULED.value,
                    created_at=now,
                    updated_at=now,
                )
                session.add(record)
        return self._timer(record)

    async def claim_due(
        self,
        *,
        owner_id: str,
        limit: int,
        now: datetime | None = None,
    ) -> list[DurableTimer]:
        claimed_at = now or datetime.now(UTC)
        async with self._sessions() as session, session.begin():
            records = list(
                (
                    await session.scalars(
                        select(DurableTimerRecord)
                        .where(
                            DurableTimerRecord.status == TimerStatus.SCHEDULED.value,
                            DurableTimerRecord.due_at <= claimed_at,
                        )
                        .order_by(DurableTimerRecord.due_at, DurableTimerRecord.id)
                        .limit(limit)
                        .with_for_update(skip_locked=True)
                    )
                ).all()
            )
            for record in records:
                record.status = TimerStatus.CLAIMED.value
                record.claimed_by = owner_id
                record.claimed_at = claimed_at
                record.updated_at = claimed_at
        return [self._timer(record) for record in records]

    async def mark_fired(
        self,
        timer_id: str,
        *,
        owner_id: str,
        now: datetime | None = None,
    ) -> DurableTimer:
        fired_at = now or datetime.now(UTC)
        async with self._sessions() as session, session.begin():
            record = await session.get(DurableTimerRecord, timer_id)
            if record is None:
                raise NotFoundError(f"Timer not found: {timer_id}")
            if record.status != TimerStatus.CLAIMED.value or record.claimed_by != owner_id:
                raise ValidationError(f"Timer is not claimed by {owner_id}: {timer_id}")
            record.status = TimerStatus.FIRED.value
            record.fired_at = fired_at
            record.updated_at = fired_at
        return self._timer(record)

    async def get(self, timer_id: str) -> DurableTimer:
        async with self._sessions() as session:
            record = await session.get(DurableTimerRecord, timer_id)
            if record is None:
                raise NotFoundError(f"Timer not found: {timer_id}")
            return self._timer(record)

    @staticmethod
    def _timer(record: DurableTimerRecord) -> DurableTimer:
        return DurableTimer(
            id=record.id,
            case_id=record.case_id,
            timer_type=record.timer_type,
            due_at=_utc(record.due_at),
            payload=dict(record.payload),
            idempotency_key=record.idempotency_key,
            status=TimerStatus(record.status),
            claimed_by=record.claimed_by,
            claimed_at=_utc(record.claimed_at) if record.claimed_at else None,
            fired_at=_utc(record.fired_at) if record.fired_at else None,
            created_at=_utc(record.created_at),
        )
