"""Milvus 文档向量写入必须使用非空稳定 chunk ID。"""

from __future__ import annotations

import pytest

from app.schemas.chunk import ChunkCreate
from app.schemas.enums import Visibility
from app.services.retrieval.store.milvus_vector import MilvusVectorStore


class FakeMilvusClient:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def insert(self, *, collection_name: str, data: list[dict[str, object]]) -> None:
        self.records = data

    def flush(self, *, collection_name: str) -> None:
        return None


@pytest.mark.asyncio
async def test_milvus_generates_stable_id_when_chunk_id_is_empty() -> None:
    store = MilvusVectorStore.__new__(MilvusVectorStore)
    store._dim = 2
    store._client = FakeMilvusClient()
    chunk = ChunkCreate(
        document_id="doc_1",
        chunk_text="员工入职需要身份证明。",
        tenant_id="tenant_a",
        department_id="dept_a",
        visibility=Visibility.DEPARTMENT,
    )

    await store.add_chunks([chunk], [[0.1, 0.2]])

    assert str(store._client.records[0]["chunk_id"]).startswith("chunk_")
