"""
同步入库流水线（V1 fallback 模式）。

串联 parse → clean → chunk → contextualize → embed → index → mark_ready。
不依赖 Celery 或 Docker，使用已有 in-memory service。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import anyio

from app.schemas.chunk import ChunkCreate
from app.services.chunker.hybrid import HybridChunker
from app.services.ingestion.identity import stable_chunk_id
from app.services.ingestion.task import IngestionStage, IngestionTask
from app.services.parser.base import ParsedDocument
from app.services.parser.registry import ParserRegistry
from app.services.retrieval.embedding.base import Embedder
from app.services.retrieval.embedding.mock_embedding import MockEmbedder
from app.services.retrieval.store.base import BM25Store, VectorStore
from app.services.retrieval.store.memory_vector import InMemoryVectorStore

logger = logging.getLogger(__name__)

# 支持的文件扩展名 → MIME 类型映射
_EXTENSION_MAP: dict[str, str] = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".markdown": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


class IngestionPipeline:
    """
    文档入库流水线。

    V1 使用同步执行模式（直接 await 每个阶段），
    后续接入 Celery 后改为 task chain。
    """

    def __init__(
        self,
        parser_registry: ParserRegistry,
        chunker: HybridChunker | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        embedding_batch_size: int = 10,
    ) -> None:
        self._parser_registry = parser_registry
        self._chunker = chunker or HybridChunker()
        self._embedder = embedder or MockEmbedder()
        self._vector_store = vector_store or InMemoryVectorStore()
        self._bm25_store = bm25_store
        self._embedding_batch_size = max(1, embedding_batch_size)
        # 已入库的 chunk（内存存储，幂等检查用）
        self._indexed_chunks: dict[str, list[str]] = {}  # doc_id -> [chunk_id]

    async def run(
        self,
        task: IngestionTask,
        file_content: bytes,
        tenant_id: str,
        department_id: str,
        visibility: str = "department",
    ) -> IngestionTask:
        """
        执行完整入库流水线。

        Args:
            task: 入库任务
            file_content: 文件内容
            tenant_id: 租户 ID
            department_id: 部门 ID
            visibility: 可见性

        Returns:
            更新后的任务状态
        """
        doc_metadata = {
            "tenant_id": tenant_id,
            "department_id": department_id,
            "visibility": visibility,
        }

        try:
            # 阶段 1: 解析
            parsed_doc = await self._stage_parse(task, file_content, doc_metadata)

            # 阶段 2: 清洗
            cleaned_doc = self._stage_clean(task, parsed_doc)

            # 阶段 3: 分块
            chunks = await self._stage_chunk(task, cleaned_doc)

            # 阶段 4: 上下文化
            chunks = self._stage_contextualize(task, chunks)

            # 阶段 5: Embedding
            await self._stage_embed(task, chunks)

            # 阶段 6: 索引
            await self._stage_index(task, chunks)

            # 阶段 7: 标记就绪
            self._stage_mark_ready(task, len(chunks))

        except Exception as e:
            logger.error(
                "pipeline_failed",
                extra={"task_id": task.id, "stage": task.current_stage, "error": str(e)},
            )
            task.fail_stage(task.current_stage, str(e))

        return task

    async def _stage_parse(
        self,
        task: IngestionTask,
        file_content: bytes,
        doc_metadata: dict,
    ) -> ParsedDocument:
        """阶段 1: 解析文件。"""
        task.start_stage(IngestionStage.PARSING)

        parser = self._parser_registry.get_parser(task.mime_type)

        # 先写临时文件供 parser 读取
        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=self._mime_to_ext(task.mime_type),
            delete=False,
            mode="wb",
        ) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            parsed_doc = await parser.parse(
                file_path=tmp_path,
                document_id=task.document_id,
                metadata=doc_metadata,
            )
        finally:
            await anyio.Path(tmp_path).unlink(missing_ok=True)

        task.complete_stage(IngestionStage.PARSING)
        return parsed_doc

    def _stage_clean(
        self,
        task: IngestionTask,
        parsed_doc: ParsedDocument,
    ) -> ParsedDocument:
        """阶段 2: 清洗文本。"""
        task.start_stage(IngestionStage.CLEANING)

        # V1 简单清洗：去除多余空白
        cleaned_blocks = []
        for block in parsed_doc.blocks:
            cleaned_text = re.sub(r"\n{3,}", "\n\n", block.text.strip())
            if cleaned_text:
                cleaned_blocks.append(block.model_copy(update={"text": cleaned_text}))

        cleaned_doc = parsed_doc.model_copy(update={"blocks": cleaned_blocks})
        task.complete_stage(IngestionStage.CLEANING)
        return cleaned_doc

    async def _stage_chunk(
        self,
        task: IngestionTask,
        parsed_doc: ParsedDocument,
    ) -> list[ChunkCreate]:
        """阶段 3: 分块。"""
        task.start_stage(IngestionStage.CHUNKING)
        chunks = await self._chunker.chunk(parsed_doc)
        chunks = [
            chunk.model_copy(
                update={
                    "id": stable_chunk_id(
                        document_id=task.document_id,
                        document_version=task.document_version,
                        heading_path=chunk.heading_path,
                        ordinal=index,
                        chunk_text=chunk.chunk_text,
                    ),
                    "document_version": task.document_version,
                }
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        task.complete_stage(IngestionStage.CHUNKING, chunk_count=len(chunks))
        return chunks

    def _stage_contextualize(
        self,
        task: IngestionTask,
        chunks: list[ChunkCreate],
    ) -> list[ChunkCreate]:
        """阶段 4: 上下文化（已在 chunker 中完成）。"""
        task.start_stage(IngestionStage.CONTEXTUALIZING)
        # HybridChunker 已包含 Contextual Enrichment
        task.complete_stage(IngestionStage.CONTEXTUALIZING)
        return chunks

    async def _stage_embed(
        self,
        task: IngestionTask,
        chunks: list[ChunkCreate],
    ) -> None:
        """阶段 5: 向量化。"""
        task.start_stage(IngestionStage.EMBEDDING)

        embeddings: list[list[float]] = []
        texts = [c.full_text for c in chunks]
        for start in range(0, len(texts), self._embedding_batch_size):
            batch = texts[start : start + self._embedding_batch_size]
            embeddings.extend(await self._embedder.embed_documents(batch))

        # 暂存 embedding 到 chunk metadata（后续 index 阶段使用）
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if not chunk.acl_metadata:
                chunk.acl_metadata = {}
            chunk.acl_metadata["_embedding"] = embedding

        task.complete_stage(IngestionStage.EMBEDDING)

    async def _stage_index(
        self,
        task: IngestionTask,
        chunks: list[ChunkCreate],
    ) -> None:
        """阶段 6: 写入索引。"""
        task.start_stage(IngestionStage.INDEXING)

        doc_id = task.document_id

        # 幂等检查：先清除已有 chunks
        if doc_id in self._indexed_chunks:
            old_count = len(self._indexed_chunks[doc_id])
            logger.info(
                "reindexing_document",
                extra={"doc_id": doc_id, "old_chunk_count": old_count},
            )
            await self._vector_store.delete_by_document(doc_id)
            if self._bm25_store is not None:
                await self._bm25_store.delete_by_document(doc_id)

        # 收集 embeddings
        embeddings: list[list[float]] = []
        for chunk in chunks:
            emb = chunk.acl_metadata.get("_embedding", []) if chunk.acl_metadata else []
            embeddings.append(emb)

        await self._vector_store.add_chunks(chunks, embeddings)
        if self._bm25_store is not None:
            await self._bm25_store.add_chunks(chunks)

        self._indexed_chunks[doc_id] = [chunk.id for chunk in chunks]
        task.complete_stage(IngestionStage.INDEXING, chunk_count=len(chunks))

    def _stage_mark_ready(self, task: IngestionTask, chunk_count: int) -> None:
        """阶段 7: 标记就绪。"""
        task.start_stage(IngestionStage.READY)
        task.complete_stage(IngestionStage.READY, chunk_count=chunk_count)
        logger.info(
            "document_ready",
            extra={
                "task_id": task.id,
                "document_id": task.document_id,
                "total_chunks": chunk_count,
            },
        )

    @staticmethod
    def _mime_to_ext(mime_type: str) -> str:
        """MIME 类型 → 文件扩展名。"""
        for ext, mime in _EXTENSION_MAP.items():
            if mime == mime_type:
                return ext
        return ".txt"

    @staticmethod
    def detect_mime_type(filename: str) -> str | None:
        """根据文件名检测 MIME 类型，不支持则返回 None。"""
        ext = Path(filename).suffix.lower()
        return _EXTENSION_MAP.get(ext)

    @property
    def supported_extensions(self) -> set[str]:
        """支持的文件扩展名。"""
        return set(_EXTENSION_MAP.keys())
