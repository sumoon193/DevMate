"""
Hybrid Retriever。

编排完整检索流程：ACL -> Dense -> Sparse -> RRF -> Rerank -> Evidence。
"""

from __future__ import annotations

import logging

from app.core.exceptions import NotFoundError
from app.schemas.chunk import ChunkCreate, EvidenceBundle
from app.schemas.enums import Visibility
from app.schemas.retrieval import RetrievalResult
from app.services.ingestion.document_versions import DocumentVersionRegistry
from app.services.retrieval.embedding.base import Embedder
from app.services.retrieval.evidence_builder import EvidenceBuilder
from app.services.retrieval.fusion.rrf import RRFFuser
from app.services.retrieval.reranker.base import Reranker
from app.services.retrieval.store.base import ACLFilter, BM25Store, VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    混合检索器。

    编排完整的检索流程：
    1. 构建 ACL filter
    2. Dense search（向量检索）
    3. Sparse search（BM25 检索）
    4. RRF fusion
    5. Rerank
    6. 构建 EvidenceBundle
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        reranker: Reranker,
        document_version_registry: DocumentVersionRegistry | None = None,
    ) -> None:
        """
        初始化混合检索器。

        Args:
            embedder: Embedding 实现
            vector_store: 向量存储
            bm25_store: BM25 存储
            reranker: Reranker 实现
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._reranker = reranker
        self._document_versions = document_version_registry
        self._rrf = RRFFuser()
        self._evidence_builder = EvidenceBuilder()

        logger.info("hybrid_retriever_initialized")

    async def index_chunks(self, chunks: list[ChunkCreate]) -> None:
        """
        索引 chunks。

        将 chunks 添加到向量存储和 BM25 存储。

        Args:
            chunks: 分块列表
        """
        logger.info("indexing_chunks", extra={"count": len(chunks)})

        # Embed chunks
        texts = [c.full_text or c.chunk_text for c in chunks]
        embeddings = await self._embedder.embed_documents(texts)

        # 添加到向量存储
        await self._vector_store.add_chunks(chunks, embeddings)

        # 添加到 BM25 存储
        await self._bm25_store.add_chunks(chunks)

        logger.info("indexing_complete", extra={"count": len(chunks)})

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        department_ids: list[str],
        allowed_visibility: list[Visibility] | None = None,
        top_k: int = 10,
    ) -> EvidenceBundle:
        """
        执行混合检索。

        Args:
            query: 查询文本
            tenant_id: 租户 ID
            department_ids: 部门 ID 列表
            allowed_visibility: 允许的可见性级别列表（默认所有级别）
            top_k: 返回结果数量

        Returns:
            证据包
        """
        logger.info(
            "hybrid_retrieval", extra={"query": query[:50], "tenant_id": tenant_id, "top_k": top_k}
        )

        # 构建 ACL filter
        if allowed_visibility is None:
            allowed_visibility = list(Visibility)

        acl_filter = ACLFilter(
            tenant_id=tenant_id,
            department_ids=department_ids,
            allowed_visibility=allowed_visibility,
        )

        # Step 1: Dense search
        query_embedding = await self._embedder.embed_query(query)
        dense_results = await self._vector_store.search(
            query_embedding=query_embedding,
            acl_filter=acl_filter,
            top_k=top_k * 2,  # 取更多结果用于融合
        )

        logger.info("dense_search_done", extra={"count": len(dense_results)})

        # Step 2: Sparse search（BM25）
        sparse_results = await self._bm25_store.search(
            query=query, acl_filter=acl_filter, top_k=top_k * 2
        )

        if self._document_versions is not None:
            dense_results = await self._active_version_only(dense_results)
            sparse_results = await self._active_version_only(sparse_results)

        logger.info("sparse_search_done", extra={"count": len(sparse_results)})

        # Step 3: RRF fusion
        fused_results = self._rrf.fuse(dense_results, sparse_results, top_k=top_k * 2)

        logger.info("rrf_fusion_done", extra={"count": len(fused_results)})

        # Step 4: Rerank
        reranked_results = await self._reranker.rerank(
            query=query, results=fused_results, top_k=top_k
        )

        logger.info("reranking_done", extra={"count": len(reranked_results)})

        # Step 5: 构建 EvidenceBundle
        evidence = self._evidence_builder.build(results=reranked_results, query=query)

        logger.info(
            "hybrid_retrieval_complete",
            extra={
                "citation_count": evidence.total_count,
                "query_coverage": evidence.query_coverage_score,
            },
        )

        return evidence

    async def _active_version_only(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """只保留 registry 中当前 active 的不可变文档版本。"""
        active_ids: dict[str, str | None] = {}
        filtered: list[RetrievalResult] = []
        for result in results:
            if result.document_id not in active_ids:
                try:
                    active = await self._document_versions.get_active(result.document_id)
                    active_ids[result.document_id] = active.id
                except NotFoundError:
                    active_ids[result.document_id] = None
            if active_ids[result.document_id] == result.document_version:
                filtered.append(result)
        return filtered

    async def delete_document(self, document_id: str) -> None:
        """
        删除指定文档的所有索引。

        Args:
            document_id: 文档 ID
        """
        logger.info("deleting_document", extra={"document_id": document_id})

        await self._vector_store.delete_by_document(document_id)
        await self._bm25_store.delete_by_document(document_id)

        logger.info("document_deleted", extra={"document_id": document_id})
