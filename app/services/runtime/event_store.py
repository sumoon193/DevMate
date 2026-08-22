"""append-only Event Store 及 deterministic in-memory 实现。"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.exceptions import NotFoundError, ValidationError
from app.schemas.runtime import OutboxMessage, RunEventEnvelope
from app.services.observability.runtime_metrics import RuntimeMetrics


class InMemoryEventStore:
    """用于 fallback、单元测试和 projection rebuild 的内存事件存储。"""

    def __init__(self, metrics: RuntimeMetrics | None = None) -> None:
        self._streams: dict[str, list[RunEventEnvelope]] = {}
        self._commands: dict[str, tuple[str, RunEventEnvelope]] = {}
        self._outbox: dict[str, OutboxMessage] = {}
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
        """以乐观并发版本向聚合流追加事件。"""
        command_fingerprint = self._command_fingerprint(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            event_type=event_type,
            payload=payload,
            actor_id=actor_id,
        )
        prior_command = self._commands.get(command_id)
        if prior_command is not None:
            prior_fingerprint, prior_event = prior_command
            if prior_fingerprint != command_fingerprint:
                raise ValidationError(f"Command id reused with different content: {command_id}")
            return prior_event

        stream = self._streams.setdefault(aggregate_id, [])
        if len(stream) != expected_version:
            raise ValidationError(
                f"Event stream version conflict: expected {expected_version}, actual {len(stream)}"
            )

        sequence = len(stream) + 1
        prev_hash = stream[-1].event_hash if stream else ""
        created_at = datetime.now(UTC)
        event_id = f"evt_{uuid.uuid4().hex[:16]}"
        event_hash = self._compute_hash(
            event_id=event_id,
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            sequence=sequence,
            event_type=event_type,
            payload=payload,
            command_id=command_id,
            actor_id=actor_id,
            created_at=created_at,
            schema_version=1,
            prev_hash=prev_hash,
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
            created_at=created_at,
            schema_version=1,
            prev_hash=prev_hash,
            event_hash=event_hash,
        )
        stream.append(event)
        self._commands[command_id] = (command_fingerprint, event)
        outbox_id = f"outbox_{uuid.uuid4().hex[:16]}"
        self._outbox[outbox_id] = OutboxMessage(
            id=outbox_id,
            event_id=event.id,
            aggregate_id=event.aggregate_id,
            sequence=event.sequence,
            topic=event.event_type,
            payload=event.model_dump(mode="json"),
            created_at=created_at,
        )
        if self._metrics is not None:
            self._metrics.increment("runtime.events.total")
            self._metrics.set_gauge(
                "runtime.outbox.backlog",
                len(self._pending_outbox()),
            )
        return event

    async def load_stream(self, aggregate_id: str) -> list[RunEventEnvelope]:
        """返回聚合事件流的副本。"""
        return list(self._streams.get(aggregate_id, []))

    async def verify_chain(self, aggregate_id: str) -> bool:
        """验证指定聚合的事件顺序和防篡改哈希链。"""
        previous_hash = ""
        for sequence, event in enumerate(self._streams.get(aggregate_id, []), start=1):
            if event.sequence != sequence or event.prev_hash != previous_hash:
                return False
            expected_hash = self._compute_hash(
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
            if event.event_hash != expected_hash:
                return False
            previous_hash = event.event_hash
        return True

    async def pending_outbox(self, *, limit: int = 100) -> list[OutboxMessage]:
        """返回尚未确认投递的 outbox message。"""
        return self._pending_outbox()[:limit]

    async def claim_outbox(
        self,
        *,
        owner_id: str,
        limit: int,
        claim_ttl_seconds: int = 30,
    ) -> list[OutboxMessage]:
        """claim 待投递消息，并允许回收超时 claim。"""
        now = datetime.now(UTC)
        stale_before = now - timedelta(seconds=claim_ttl_seconds)
        claimed: list[OutboxMessage] = []
        for message in self._outbox.values():
            if message.published_at is not None:
                continue
            if message.claimed_at is not None and message.claimed_at > stale_before:
                continue
            message.claimed_by = owner_id
            message.claimed_at = now
            message.delivery_attempts += 1
            claimed.append(message.model_copy(deep=True))
            if len(claimed) >= limit:
                break
        return claimed

    async def mark_outbox_published(self, outbox_id: str, *, owner_id: str) -> None:
        """由 claim owner 幂等确认 outbox message 已投递。"""
        message = self._outbox.get(outbox_id)
        if message is None:
            raise NotFoundError(f"Outbox message not found: {outbox_id}")
        if message.claimed_by not in {None, owner_id}:
            raise ValidationError(f"Outbox message is claimed by another owner: {outbox_id}")
        if message.published_at is None:
            message.published_at = datetime.now(UTC)
            if self._metrics is not None:
                self._metrics.increment("runtime.outbox.published")
                self._metrics.set_gauge(
                    "runtime.outbox.backlog",
                    len(self._pending_outbox()),
                )

    def _pending_outbox(self) -> list[OutboxMessage]:
        """返回全部未发布 outbox 的内部副本。"""
        return [
            message.model_copy(deep=True)
            for message in self._outbox.values()
            if message.published_at is None
        ]

    @staticmethod
    def _command_fingerprint(
        *,
        aggregate_id: str,
        aggregate_type: str,
        event_type: str,
        payload: dict[str, Any],
        actor_id: str,
    ) -> str:
        """计算命令语义指纹，防止同一 command_id 被复用于不同动作。"""
        canonical = json.dumps(
            {
                "aggregate_id": aggregate_id,
                "aggregate_type": aggregate_type,
                "event_type": event_type,
                "payload": payload,
                "actor_id": actor_id,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_hash(
        *,
        event_id: str,
        aggregate_id: str,
        aggregate_type: str,
        sequence: int,
        event_type: str,
        payload: dict[str, Any],
        command_id: str,
        actor_id: str,
        created_at: datetime,
        schema_version: int,
        prev_hash: str,
    ) -> str:
        """对标准化事件内容计算 SHA-256。"""
        canonical = json.dumps(
            {
                "id": event_id,
                "aggregate_id": aggregate_id,
                "aggregate_type": aggregate_type,
                "sequence": sequence,
                "event_type": event_type,
                "payload": payload,
                "command_id": command_id,
                "actor_id": actor_id,
                "created_at": created_at.isoformat(),
                "schema_version": schema_version,
                "prev_hash": prev_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
