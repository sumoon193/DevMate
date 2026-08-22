"""
文档入库流水线测试。

模块 03 规范要求的 5 个测试用例。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.schemas.chunk import ChunkCreate
from app.schemas.enums import Visibility
from app.services.chunker.hybrid import HybridChunker
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.ingestion.task import IngestionStage, IngestionTask
from app.services.parser.markdown_parser import MarkdownParser
from app.services.parser.plain_parser import PlainTextParser
from app.services.parser.registry import ParserRegistry
from app.services.retrieval.embedding.mock_embedding import MockEmbedder
from app.services.retrieval.store.base import ACLFilter
from app.services.retrieval.store.memory_bm25 import InMemoryBM25Store
from app.services.retrieval.store.memory_vector import InMemoryVectorStore

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def parser_registry() -> ParserRegistry:
    registry = ParserRegistry()
    registry.register(MarkdownParser())
    registry.register(PlainTextParser())
    return registry


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def bm25_store() -> InMemoryBM25Store:
    return InMemoryBM25Store()


@pytest.fixture
def pipeline(
    parser_registry: ParserRegistry, vector_store: InMemoryVectorStore
) -> IngestionPipeline:
    return IngestionPipeline(
        parser_registry=parser_registry,
        chunker=HybridChunker(),
        embedder=MockEmbedder(),
        vector_store=vector_store,
    )


def _md_content() -> bytes:
    """标准 Markdown 测试文档。"""
    text = """# Employee Onboarding Policy

## Chapter 1: Required Materials

New employees must submit the following materials:

1. ID card copy
2. Education certificate
3. Resignation letter from previous employer

## Chapter 2: Probation Period

The probation period is 3 months. It can be extended to 6 months for special positions.
"""
    return text.encode("utf-8")


def _make_task(document_id: str = "doc_test_001", filename: str = "test.md") -> IngestionTask:
    """创建测试用 IngestionTask。"""
    return IngestionTask(
        document_id=document_id,
        filename=filename,
        mime_type="text/markdown",
    )


class BatchLimitedEmbedder:
    """测试用 embedder：模拟真实 embedding API 的批量上限。"""

    dimension = 3

    def __init__(self, max_batch_size: int = 10) -> None:
        self.max_batch_size = max_batch_size
        self.calls: list[list[str]] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) > self.max_batch_size:
            raise ValueError(f"batch size must be <= {self.max_batch_size}")
        self.calls.append(texts)
        return [[float(index), 0.0, 0.0] for index, _ in enumerate(texts)]

    async def embed_query(self, query: str) -> list[float]:
        return (await self.embed_documents([query]))[0]


def _chunk(index: int) -> ChunkCreate:
    return ChunkCreate(
        document_id="doc_batch_001",
        chunk_text=f"chunk text {index}",
        full_text=f"context chunk text {index}",
        tenant_id="tenant_hr",
        department_id="dept_001",
        visibility=Visibility.DEPARTMENT,
    )


# ── 1. test_upload_creates_document_and_task ─────────────────────────


@pytest.mark.asyncio
async def test_upload_creates_document_and_task(pipeline: IngestionPipeline) -> None:
    """上传文档应创建 document 和 IngestionTask，完成后状态为 ready。"""
    task = _make_task()
    content = _md_content()

    result = await pipeline.run(
        task=task,
        file_content=content,
        tenant_id="tenant_hr",
        department_id="dept_001",
    )

    # 任务应有 document_id
    assert result.document_id == "doc_test_001"
    # 状态应为 ready
    assert result.current_stage == IngestionStage.READY
    # 应有分块
    assert result.total_chunks > 0
    # 应记录了多个阶段
    assert len(result.stages) >= 5
    # progress 应为 1.0
    assert result.progress == 1.0


# ── 2. test_upload_rejects_unsupported_extension ─────────────────────


def test_upload_rejects_unsupported_extension() -> None:
    """不支持的文件扩展名应返回 None。"""
    assert IngestionPipeline.detect_mime_type("image.png") is None
    assert IngestionPipeline.detect_mime_type("video.mp4") is None
    assert IngestionPipeline.detect_mime_type("archive.zip") is None
    # 支持的类型应返回正确 MIME
    assert IngestionPipeline.detect_mime_type("readme.md") == "text/markdown"
    assert IngestionPipeline.detect_mime_type("notes.txt") == "text/plain"
    assert IngestionPipeline.detect_mime_type("report.pdf") == "application/pdf"


# ── 3. test_ingestion_status_updates_stage_progress ──────────────────


@pytest.mark.asyncio
async def test_ingestion_status_updates_stage_progress(pipeline: IngestionPipeline) -> None:
    """入库过程中 stage 和 progress 应逐步更新。"""
    task = _make_task()
    content = _md_content()

    result = await pipeline.run(
        task=task,
        file_content=content,
        tenant_id="tenant_hr",
        department_id="dept_001",
    )

    # 每个阶段应有记录
    stage_names = [s.stage for s in result.stages]
    assert IngestionStage.PARSING in stage_names
    assert IngestionStage.CHUNKING in stage_names
    assert IngestionStage.INDEXING in stage_names
    assert IngestionStage.READY in stage_names

    # 每个阶段应有完成时间
    for record in result.stages:
        assert record.completed_at is not None
        assert record.duration_ms >= 0

    # 最终 progress 应为 1.0
    assert result.progress == 1.0

    # to_status_dict 应包含所有必要字段
    status = result.to_status_dict()
    assert "task_id" in status
    assert "document_id" in status
    assert "status" in status
    assert "progress" in status


# ── 4. test_failed_task_records_error_message ────────────────────────


@pytest.mark.asyncio
async def test_failed_task_records_error_message(parser_registry: ParserRegistry) -> None:
    """入库失败应记录 error_message 和 error_code。"""
    # 用一个会失败的 pipeline（空 parser_registry）
    empty_registry = ParserRegistry()
    # 不注册任何 parser，get_parser 会抛 NotFoundError

    failed_pipeline = IngestionPipeline(
        parser_registry=empty_registry,
    )

    task = _make_task()
    content = _md_content()

    result = await failed_pipeline.run(
        task=task,
        file_content=content,
        tenant_id="tenant_hr",
        department_id="dept_001",
    )

    # 状态应为 failed
    assert result.current_stage == IngestionStage.FAILED
    # 应有错误信息
    assert result.error_message is not None
    assert len(result.error_message) > 0
    # 应有错误码
    assert result.error_code == "ingestion_error"

    # to_status_dict 应包含 error 字段
    status = result.to_status_dict()
    assert status["error"] is not None


# ── 5. test_retry_single_stage_does_not_duplicate_chunks ─────────────


@pytest.mark.asyncio
async def test_retry_single_stage_does_not_duplicate_chunks(
    pipeline: IngestionPipeline,
    vector_store: InMemoryVectorStore,
) -> None:
    """重试入库不应产生重复 chunks。"""
    content = _md_content()

    # 第一次入库
    task1 = _make_task(document_id="doc_retry_001")
    result1 = await pipeline.run(
        task=task1,
        file_content=content,
        tenant_id="tenant_hr",
        department_id="dept_001",
    )
    first_chunk_count = result1.total_chunks
    assert first_chunk_count > 0

    # 第二次入库（同一 document_id，模拟重试）
    task2 = _make_task(document_id="doc_retry_001")
    result2 = await pipeline.run(
        task=task2,
        file_content=content,
        tenant_id="tenant_hr",
        department_id="dept_001",
    )

    # chunk 数量应相同（幂等）
    assert result2.total_chunks == first_chunk_count
    # 第二次也应成功
    assert result2.current_stage == IngestionStage.READY

    query_embedding = await MockEmbedder().embed_query("Employee Onboarding Policy")
    indexed_results = await vector_store.search(
        query_embedding=query_embedding,
        acl_filter=ACLFilter(
            tenant_id="tenant_hr",
            department_ids=["dept_001"],
            allowed_visibility=list(Visibility),
        ),
        top_k=100,
    )
    indexed_doc_results = [
        result for result in indexed_results if result.document_id == "doc_retry_001"
    ]
    assert len(indexed_doc_results) == first_chunk_count


@pytest.mark.asyncio
async def test_pipeline_indexes_bm25_store_when_configured(
    parser_registry: ParserRegistry,
    vector_store: InMemoryVectorStore,
    bm25_store: InMemoryBM25Store,
) -> None:
    """full mode 入库流水线应同时写入向量索引和 BM25 索引。"""
    pipeline = IngestionPipeline(
        parser_registry=parser_registry,
        chunker=HybridChunker(),
        embedder=MockEmbedder(),
        vector_store=vector_store,
        bm25_store=bm25_store,
    )
    task = _make_task(document_id="doc_bm25_001")

    result = await pipeline.run(
        task=task,
        file_content=_md_content(),
        tenant_id="tenant_hr",
        department_id="dept_001",
    )

    assert result.current_stage == IngestionStage.READY
    results = await bm25_store.search(
        query="probation period",
        acl_filter=ACLFilter(
            tenant_id="tenant_hr",
            department_ids=["dept_001"],
            allowed_visibility=list(Visibility),
        ),
        top_k=10,
    )
    assert any(item.document_id == "doc_bm25_001" for item in results)


@pytest.mark.asyncio
async def test_pipeline_batches_embedding_requests(parser_registry: ParserRegistry) -> None:
    """Embedding 阶段应按批次调用真实模型，避免超过供应商批量上限。"""
    embedder = BatchLimitedEmbedder(max_batch_size=10)
    pipeline = IngestionPipeline(
        parser_registry=parser_registry,
        embedder=embedder,
        embedding_batch_size=10,
    )
    task = _make_task(document_id="doc_batch_001")
    chunks = [_chunk(index) for index in range(23)]

    await pipeline._stage_embed(task, chunks)

    assert [len(call) for call in embedder.calls] == [10, 10, 3]
    assert all(chunk.acl_metadata["_embedding"] for chunk in chunks)


@pytest.mark.asyncio
async def test_document_pipeline_builder_uses_container_ai_and_indexes(
    vector_store: InMemoryVectorStore,
    bm25_store: InMemoryBM25Store,
) -> None:
    """API 文档端点的 pipeline builder 应能复用 full mode 容器依赖。"""
    from app.api.documents import _build_pipeline

    container = SimpleNamespace(
        embedder=MockEmbedder(),
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    pipeline = _build_pipeline(container)
    task = _make_task(document_id="doc_api_builder_001")

    result = await pipeline.run(
        task=task,
        file_content=_md_content(),
        tenant_id="tenant_hr",
        department_id="dept_001",
    )

    assert result.current_stage == IngestionStage.READY
    bm25_results = await bm25_store.search(
        query="required materials",
        acl_filter=ACLFilter(
            tenant_id="tenant_hr",
            department_ids=["dept_001"],
            allowed_visibility=list(Visibility),
        ),
        top_k=10,
    )
    assert any(item.document_id == "doc_api_builder_001" for item in bm25_results)
