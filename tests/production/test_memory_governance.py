"""长期记忆的 TTL、去重、排序、语义召回和删除传播合同。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.schemas.enums import MemoryStatus
from app.services.memory.store import InMemoryEpisodicMemoryStore
from app.services.runtime.clock import FakeClock


class RecordingSemanticIndex:
    def __init__(self) -> None:
        self.records: dict[str, tuple[str, str]] = {}
        self.matches: list[tuple[str, float]] = []
        self.deleted: list[tuple[str, str]] = []

    async def upsert(self, *, memory_id: str, tenant_id: str, content: str) -> None:
        self.records[memory_id] = (tenant_id, content)

    async def search(self, *, tenant_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        return self.matches[:limit]

    async def delete(self, *, memory_id: str, tenant_id: str) -> None:
        self.deleted.append((memory_id, tenant_id))


@pytest.mark.asyncio
async def test_expired_memory_is_not_recalled_and_is_marked_expired() -> None:
    clock = FakeClock(datetime(2026, 8, 6, tzinfo=UTC))
    store = InMemoryEpisodicMemoryStore(clock=clock)
    record = await store.remember(
        tenant_id="tenant_a",
        case_id="case_1",
        memory_key="policy.exception",
        content="该例外仅在本周有效。",
        provenance_event_ids=["evt_1"],
        ttl_seconds=60,
    )

    clock.advance(seconds=61)

    assert await store.search(tenant_id="tenant_a", query="例外") == []
    assert (await store.get(record.id, tenant_id="tenant_a")).status == MemoryStatus.EXPIRED


@pytest.mark.asyncio
async def test_duplicate_memory_merges_provenance_and_keeps_one_record() -> None:
    store = InMemoryEpisodicMemoryStore()
    first = await store.remember(
        tenant_id="tenant_a",
        case_id="case_1",
        memory_key="onboarding.material",
        content="员工缺少学历证明。",
        provenance_event_ids=["evt_1"],
        importance_score=0.4,
    )
    second = await store.remember(
        tenant_id="tenant_a",
        case_id="case_2",
        memory_key="onboarding.material",
        content="  员工缺少学历证明。  ",
        provenance_event_ids=["evt_2"],
        importance_score=0.9,
    )

    assert second.id == first.id
    assert second.provenance_event_ids == ["evt_1", "evt_2"]
    assert second.importance_score == 0.9
    assert len(await store.search(tenant_id="tenant_a", query="学历证明")) == 1


@pytest.mark.asyncio
async def test_search_ranks_higher_importance_before_lower_importance() -> None:
    store = InMemoryEpisodicMemoryStore()
    low = await store.remember(
        tenant_id="tenant_a",
        case_id="case_1",
        memory_key="policy.low",
        content="报销需要发票。",
        provenance_event_ids=["evt_1"],
        importance_score=0.2,
    )
    high = await store.remember(
        tenant_id="tenant_a",
        case_id="case_2",
        memory_key="policy.high",
        content="报销需要发票并经过主管审批。",
        provenance_event_ids=["evt_2"],
        importance_score=0.9,
    )

    results = await store.search(tenant_id="tenant_a", query="报销 发票")

    assert [item.id for item in results] == [high.id, low.id]


@pytest.mark.asyncio
async def test_semantic_index_can_recall_non_lexical_match() -> None:
    index = RecordingSemanticIndex()
    store = InMemoryEpisodicMemoryStore(semantic_index=index)
    record = await store.remember(
        tenant_id="tenant_a",
        case_id="case_1",
        memory_key="policy.travel",
        content="出差交通工具应按职级标准预订。",
        provenance_event_ids=["evt_1"],
    )
    index.matches = [(record.id, 0.95)]

    results = await store.search(tenant_id="tenant_a", query="差旅怎么坐车")

    assert [item.id for item in results] == [record.id]


@pytest.mark.asyncio
async def test_forget_propagates_to_semantic_index() -> None:
    index = RecordingSemanticIndex()
    store = InMemoryEpisodicMemoryStore(semantic_index=index)
    record = await store.remember(
        tenant_id="tenant_a",
        case_id="case_1",
        memory_key="policy.delete",
        content="待删除记忆。",
        provenance_event_ids=["evt_1"],
    )

    await store.forget(record.id, tenant_id="tenant_a")

    assert index.deleted == [(record.id, "tenant_a")]


@pytest.mark.asyncio
async def test_expired_memory_is_deleted_from_semantic_index() -> None:
    clock = FakeClock(datetime(2026, 8, 6, tzinfo=UTC))
    index = RecordingSemanticIndex()
    store = InMemoryEpisodicMemoryStore(clock=clock, semantic_index=index)
    record = await store.remember(
        tenant_id="tenant_a",
        case_id="case_1",
        memory_key="policy.temporary",
        content="temporary policy",
        provenance_event_ids=["evt_1"],
        ttl_seconds=60,
    )

    clock.advance(seconds=61)
    await store.search(tenant_id="tenant_a", query="temporary")

    assert index.deleted == [(record.id, "tenant_a")]


@pytest.mark.asyncio
async def test_expired_memory_retries_semantic_delete_after_transient_failure() -> None:
    class FlakyDeletionIndex(RecordingSemanticIndex):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        async def delete(self, *, memory_id: str, tenant_id: str) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("transient semantic index failure")
            await super().delete(memory_id=memory_id, tenant_id=tenant_id)

    clock = FakeClock(datetime(2026, 8, 6, tzinfo=UTC))
    index = FlakyDeletionIndex()
    store = InMemoryEpisodicMemoryStore(clock=clock, semantic_index=index)
    record = await store.remember(
        tenant_id="tenant_a",
        case_id="case_1",
        memory_key="policy.retry-delete",
        content="temporary retry policy",
        provenance_event_ids=["evt_1"],
        ttl_seconds=60,
    )

    clock.advance(seconds=61)
    with pytest.raises(RuntimeError, match="transient semantic index failure"):
        await store.get(record.id, tenant_id="tenant_a")

    expired = await store.get(record.id, tenant_id="tenant_a")
    assert expired.status == MemoryStatus.EXPIRED
    assert index.attempts == 2
    assert index.deleted == [(record.id, "tenant_a")]
