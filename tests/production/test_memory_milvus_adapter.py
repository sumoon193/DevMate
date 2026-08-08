"""Milvus 长期记忆语义索引必须下推租户过滤并传播删除。"""

from __future__ import annotations

import pytest

from app.services.memory.milvus_index import MilvusMemorySemanticIndex


class FakeSchema:
    def __init__(self) -> None:
        self.fields: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def add_field(self, *args: object, **kwargs: object) -> None:
        self.fields.append((args, kwargs))


class FakeIndexParams:
    def add_index(self, **_kwargs: object) -> None:
        return None


class FakeMilvusClient:
    def __init__(self) -> None:
        self.schema = FakeSchema()
        self.upserts: list[list[dict[str, object]]] = []
        self.search_filter = ""
        self.delete_filter = ""

    def has_collection(self, _name: str) -> bool:
        return False

    def create_schema(self, **_kwargs: object) -> FakeSchema:
        return self.schema

    def prepare_index_params(self) -> FakeIndexParams:
        return FakeIndexParams()

    def create_collection(self, **_kwargs: object) -> None:
        return None

    def load_collection(self, _name: str) -> None:
        return None

    def upsert(self, *, collection_name: str, data: list[dict[str, object]]) -> None:
        assert collection_name == "episodic_memories_v1"
        self.upserts.append(data)

    def flush(self, _name: str) -> None:
        return None

    def search(self, **kwargs: object) -> list[list[dict[str, object]]]:
        self.search_filter = str(kwargs["filter"])
        return [[{"id": "mem_1", "distance": 0.91, "entity": {"memory_id": "mem_1"}}]]

    def delete(self, *, collection_name: str, filter: str) -> None:
        assert collection_name == "episodic_memories_v1"
        self.delete_filter = filter


class FakeEmbedder:
    dimension = 2

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        assert texts
        return [[0.1, 0.2] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        assert query
        return [0.1, 0.2]


@pytest.mark.asyncio
async def test_milvus_memory_index_upsert_search_and_delete_are_tenant_scoped() -> None:
    client = FakeMilvusClient()
    index = MilvusMemorySemanticIndex(
        embedder=FakeEmbedder(),
        dimension=2,
        client=client,
    )

    await index.upsert(memory_id="mem_1", tenant_id="tenant_a", content="差旅标准")
    matches = await index.search(tenant_id="tenant_a", query="出差交通", limit=5)
    await index.delete(memory_id="mem_1", tenant_id="tenant_a")

    assert client.upserts[0][0]["tenant_id"] == "tenant_a"
    assert matches == [("mem_1", 0.91)]
    assert client.search_filter == 'tenant_id == "tenant_a"'
    assert client.delete_filter == 'memory_id == "mem_1" and tenant_id == "tenant_a"'


def test_milvus_memory_index_retries_transient_client_startup_failure() -> None:
    client = FakeMilvusClient()
    attempts = 0
    delays: list[float] = []

    def create_client(*, uri: str) -> FakeMilvusClient:
        nonlocal attempts
        assert uri == "http://milvus:19530"
        attempts += 1
        if attempts < 3:
            raise RuntimeError("Milvus Proxy is not ready yet")
        return client

    MilvusMemorySemanticIndex(
        embedder=FakeEmbedder(),
        dimension=2,
        host="milvus",
        client_factory=create_client,
        connection_attempts=3,
        retry_delay_seconds=0.25,
        sleep=delays.append,
    )

    assert attempts == 3
    assert delays == [0.25, 0.25]
