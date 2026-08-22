"""devmate Runtime：Event Store、Projection 与 Outbox 同事务 Checkpoint。

Runtime 候选通过 CheckpointPort 依赖本层，不引用 HR/RAG 领域接口。
"""

from __future__ import annotations

from app.devmate.runtime.checkpoint import (
    CheckpointPort,
    TransactionalCheckpoint,
)
from app.devmate.runtime.event_store import ConcurrentVersionError, EventStore
from app.devmate.runtime.models import (
    CaseRecord,
    DM03Input,
    DM03Result,
    OutboxMessage,
    StoredEvent,
)
from app.devmate.runtime.outbox import Outbox
from app.devmate.runtime.projection import Projection

__all__ = [
    "CaseRecord",
    "CheckpointPort",
    "ConcurrentVersionError",
    "DM03Input",
    "DM03Result",
    "EventStore",
    "Outbox",
    "OutboxMessage",
    "Projection",
    "StoredEvent",
    "TransactionalCheckpoint",
]
