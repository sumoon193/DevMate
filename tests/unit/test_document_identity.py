"""版本化文档与稳定 chunk identity 测试。"""

from __future__ import annotations

import pytest

from app.core.exceptions import ValidationError
from app.schemas.chunk import ChunkCreate, Citation
from app.schemas.enums import Visibility
from app.services.ingestion.document_versions import InMemoryDocumentVersionRegistry
from app.services.ingestion.identity import stable_chunk_id
from app.services.retrieval.evidence_freshness import EvidenceFreshnessValidator
from app.services.retrieval.store.base import ACLFilter
from app.services.retrieval.store.memory_bm25 import InMemoryBM25Store
from app.services.retrieval.store.memory_vector import InMemoryVectorStore


def test_stable_chunk_id_is_reproducible_and_version_sensitive() -> None:
    """相同输入产生相同 ID，文档版本变化必须产生不同 ID。"""
    first = stable_chunk_id(
        document_id="doc_001",
        document_version="v1",
        heading_path="入职制度 > 材料",
        ordinal=1,
        chunk_text="员工应提交身份证明。",
    )
    replayed = stable_chunk_id(
        document_id="doc_001",
        document_version="v1",
        heading_path="入职制度 > 材料",
        ordinal=1,
        chunk_text="员工应提交身份证明。",
    )
    updated = stable_chunk_id(
        document_id="doc_001",
        document_version="v2",
        heading_path="入职制度 > 材料",
        ordinal=1,
        chunk_text="员工应提交身份证明。",
    )

    assert first == replayed
    assert first.startswith("chunk_")
    assert updated != first


@pytest.mark.asyncio
async def test_vector_and_bm25_share_versioned_stable_chunk_id() -> None:
    """同一 chunk 在不同索引中必须共享稳定 identity。"""
    chunk_id = stable_chunk_id(
        document_id="doc_001",
        document_version="v2",
        heading_path="入职制度 > 材料",
        ordinal=1,
        chunk_text="员工应提交身份证明。",
    )
    chunk = ChunkCreate(
        id=chunk_id,
        document_id="doc_001",
        document_version="v2",
        chunk_text="员工应提交身份证明。",
        full_text="员工应提交身份证明。",
        heading_path="入职制度 > 材料",
        tenant_id="tenant_a",
        department_id="dept_hr",
        visibility=Visibility.DEPARTMENT,
    )
    vector = InMemoryVectorStore()
    bm25 = InMemoryBM25Store()
    await vector.add_chunks([chunk], [[1.0, 0.0]])
    await bm25.add_chunks([chunk])
    acl = ACLFilter(
        tenant_id="tenant_a",
        department_ids=["dept_hr"],
        allowed_visibility=[Visibility.DEPARTMENT],
    )

    vector_hits = await vector.search([1.0, 0.0], acl, top_k=1)
    bm25_hits = await bm25.search("身份证明", acl, top_k=1)

    assert vector_hits[0].chunk_id == bm25_hits[0].chunk_id == chunk_id
    assert vector_hits[0].document_version == bm25_hits[0].document_version == "v2"


def test_evidence_freshness_rejects_superseded_document_version() -> None:
    """审批或回答引用旧制度版本时必须触发重新检索。"""
    citation = Citation(
        id=1,
        chunk_id="chunk_old",
        document_id="doc_001",
        document_version="v1",
        document_name="入职制度",
        section="材料",
        page=1,
        chunk_text="旧制度内容",
        score=0.9,
        rerank_score=0.9,
    )
    validator = EvidenceFreshnessValidator()

    with pytest.raises(ValidationError, match="stale evidence"):
        validator.validate([citation], active_versions={"doc_001": "v2"})


@pytest.mark.asyncio
async def test_document_version_registry_activates_new_content_atomically() -> None:
    """新内容激活后旧版本保留但不再作为 active evidence。"""
    registry = InMemoryDocumentVersionRegistry()

    first = await registry.register(document_id="doc_001", content=b"policy v1")
    second = await registry.register(document_id="doc_001", content=b"policy v2")

    assert first.version == 1
    assert second.version == 2
    assert (await registry.get_active("doc_001")).id == second.id
    assert (await registry.get(first.id)).is_active is False
