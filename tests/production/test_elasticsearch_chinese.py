"""Elasticsearch BM25 索引必须显式支持中文分词。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.exceptions import ExternalServiceError
from app.schemas.chunk import ChunkCreate
from app.schemas.enums import Visibility
from app.services.retrieval.store.es_bm25 import INDEX_NAME, ElasticsearchBM25Store


class FakeIndices:
    def __init__(self) -> None:
        self.created: dict[str, object] | None = None

    async def exists(self, *, index: str) -> bool:
        assert index == INDEX_NAME
        return False

    async def create(self, *, index: str, body: dict[str, object]) -> None:
        assert index == INDEX_NAME
        self.created = body


class FakeElasticsearch:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.operations: list[dict[str, object]] = []
        self.bulk_has_errors = False

    async def bulk(self, *, operations: list[dict[str, object]], refresh: bool) -> SimpleNamespace:
        assert refresh is True
        self.operations = operations
        return SimpleNamespace(
            body={
                "errors": self.bulk_has_errors,
                "items": [
                    {
                        "index": {
                            "status": 400,
                            "error": {"reason": "invalid document"},
                        }
                    }
                ],
            }
        )


@pytest.mark.asyncio
async def test_index_mapping_uses_chinese_ngram_analyzer() -> None:
    store = ElasticsearchBM25Store.__new__(ElasticsearchBM25Store)
    store._index = INDEX_NAME
    store._es = FakeElasticsearch()

    await store.ensure_index()

    mapping = store._es.indices.created
    assert mapping is not None
    assert "analysis" in mapping["settings"]
    properties = mapping["mappings"]["properties"]
    assert properties["chunk_text"]["analyzer"] == "zh_ngram"
    assert properties["context_prefix"]["analyzer"] == "zh_ngram"


def _chunk() -> ChunkCreate:
    return ChunkCreate(
        document_id="doc_1",
        chunk_text="员工转正需提交转正申请表。",
        tenant_id="tenant_a",
        department_id="dept_a",
        visibility=Visibility.DEPARTMENT,
    )


@pytest.mark.asyncio
async def test_bulk_generates_stable_id_when_chunk_id_is_empty() -> None:
    store = ElasticsearchBM25Store.__new__(ElasticsearchBM25Store)
    store._index = INDEX_NAME
    store._es = FakeElasticsearch()

    await store.add_chunks([_chunk()])

    document_id = store._es.operations[0]["index"]["_id"]
    assert str(document_id).startswith("chunk_")


@pytest.mark.asyncio
async def test_bulk_item_error_is_not_silently_ignored() -> None:
    store = ElasticsearchBM25Store.__new__(ElasticsearchBM25Store)
    store._index = INDEX_NAME
    store._es = FakeElasticsearch()
    store._es.bulk_has_errors = True

    with pytest.raises(ExternalServiceError, match="invalid document"):
        await store.add_chunks([_chunk()])
