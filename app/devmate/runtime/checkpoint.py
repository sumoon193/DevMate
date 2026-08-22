"""Checkpoint：事件与 Outbox 的同事务提交入口。

合同：``CheckpointPort.execute(input: DM03Input) -> DM03Result``。
版本冲突时事件存储与 Outbox 都不写入；checkpoint_id 作为幂等键，重放
同一 Checkpoint 返回同一结果而不产生重复写入。
"""

from __future__ import annotations

from typing import Protocol

from app.devmate.runtime.event_store import ConcurrentVersionError, EventStore
from app.devmate.runtime.models import (
    CaseRecord,
    Clock,
    DM03Input,
    DM03Result,
    FixedClock,
    OutboxMessage,
    StoredEvent,
)
from app.devmate.runtime.outbox import Outbox
from app.devmate.runtime.projection import Projection


class CheckpointPort(Protocol):
    def execute(self, input_: DM03Input) -> DM03Result: ...


class TransactionalCheckpoint:
    """以版本校验为单一门控，原子提交事件与 Outbox 消息。"""

    def __init__(
        self,
        *,
        store: EventStore,
        outbox: Outbox,
        projection: Projection,
        clock: Clock | None = None,
    ) -> None:
        self.store = store
        self.outbox = outbox
        self.projection = projection
        self.clock = clock or FixedClock()
        self._processed: dict[str, DM03Result] = {}
        self._cases: dict[str, CaseRecord] = {}

    def execute(self, input_: DM03Input) -> DM03Result:
        if input_.checkpoint_id in self._processed:
            return self._processed[input_.checkpoint_id]

        stream = self.store.load_stream(input_.aggregate_id)
        if len(stream) != input_.expected_version:
            raise ConcurrentVersionError(
                f"expected version {input_.expected_version}, current {len(stream)}"
            )

        now = self.clock.now()
        new_version = len(stream) + 1
        event = StoredEvent(
            event_id=f"evt-{input_.checkpoint_id}",
            aggregate_id=input_.aggregate_id,
            aggregate_type=input_.aggregate_type,
            version=new_version,
            event_type=input_.event_type,
            payload=dict(input_.payload),
            checkpoint_id=input_.checkpoint_id,
            actor_id=input_.actor_id,
            created_at=now,
        )
        messages = [
            OutboxMessage(
                outbox_id=f"out-{input_.checkpoint_id}-{topic}",
                checkpoint_id=input_.checkpoint_id,
                aggregate_id=input_.aggregate_id,
                event_type=input_.event_type,
                topic=topic,
                payload=dict(input_.payload),
                created_at=now,
            )
            for topic in input_.outbox_topics
        ]

        # 版本校验通过：先落事件，再落 Outbox，两者要么都写要么都不写。
        self.store.append(event, expected_version=input_.expected_version)
        for message in messages:
            self.outbox.enqueue(message)

        prior = self._cases.get(input_.aggregate_id)
        self._cases[input_.aggregate_id] = CaseRecord(
            case_id=input_.aggregate_id,
            version=new_version,
            checkpoint_id=input_.checkpoint_id,
            status=input_.status,
            created_at=prior.created_at if prior else now,
            updated_at=now,
            audit_source=input_.audit_source or input_.actor_id,
            payload=dict(input_.payload),
        )

        result = DM03Result(
            checkpoint_id=input_.checkpoint_id,
            new_version=new_version,
            outbox_ids=tuple(message.outbox_id for message in messages),
            projection=self.projection.rebuild(self.store.all_events()),
        )
        self._processed[input_.checkpoint_id] = result
        return result

    def case_records(self) -> dict[str, CaseRecord]:
        return dict(self._cases)
