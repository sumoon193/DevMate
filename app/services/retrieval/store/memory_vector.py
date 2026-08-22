"""
In-Memory Vector Store。

使用纯 Python list 实现向量存储和余弦相似度检索。
"""

from __future__ import annotations

import logging

from app.schemas.chunk import ChunkCreate
from app.schemas.enums import Visibility
from app.schemas.retrieval import RetrievalResult
from app.services.ingestion.identity import stable_chunk_id
from app.services.retrieval.store.base import ACLFilter

logger = logging.getLogger(__name__)


class InMemoryVectorStore:
    """
    In-Memory 向量存储。

    使用 Python list 存储向量，支持余弦相似度检索和 ACL 过滤。
    """

    def __init__(self) -> None:
        self._chunks: list[ChunkCreate] = []
        self._embeddings: list[list[float]] = []
        self._chunk_ids: list[str] = []  # 用于删除

    async def add_chunks(self, chunks: list[ChunkCreate], embeddings: list[list[float]]) -> None:
        """
        添加 chunks 到向量存储。

        Args:
            chunks: 分块列表
            embeddings: 对应的向量列表
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        logger.info("adding_chunks_to_vector_store", extra={"count": len(chunks)})

        for ordinal, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True), start=1):
            stored_chunk = chunk
            if not chunk.id:
                stored_chunk = chunk.model_copy(
                    update={
                        "id": stable_chunk_id(
                            document_id=chunk.document_id,
                            document_version=chunk.document_version,
                            heading_path=chunk.heading_path,
                            ordinal=ordinal,
                            chunk_text=chunk.chunk_text,
                        )
                    }
                )
            self._chunks.append(stored_chunk)
            self._embeddings.append(embedding)
            self._chunk_ids.append(stored_chunk.document_id)

    async def search(
        self, query_embedding: list[float], acl_filter: ACLFilter, top_k: int = 10
    ) -> list[RetrievalResult]:
        """
        向量检索。

        Args:
            query_embedding: 查询向量
            acl_filter: ACL 过滤器
            top_k: 返回结果数量

        Returns:
            检索结果列表（按相似度降序）
        """
        logger.info("vector_search", extra={"top_k": top_k, "total_chunks": len(self._chunks)})

        # 计算所有 chunk 的相似度
        scored_chunks: list[tuple[float, ChunkCreate]] = []

        for chunk, embedding in zip(self._chunks, self._embeddings, strict=True):
            # ACL 过滤
            if not self._check_acl(chunk, acl_filter):
                continue

            # 计算余弦相似度
            similarity = self._cosine_similarity(query_embedding, embedding)
            scored_chunks.append((similarity, chunk))

        # 按相似度降序排序
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # 取 top_k
        top_chunks = scored_chunks[:top_k]

        # 转换为 RetrievalResult
        results: list[RetrievalResult] = []
        for score, chunk in top_chunks:
            # 将余弦相似度 [-1, 1] 归一化到 [0, 1]
            normalized_score = (score + 1.0) / 2.0
            normalized_score = min(max(normalized_score, 0.0), 1.0)

            result = RetrievalResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_version=chunk.document_version,
                chunk_text=chunk.chunk_text,
                context_prefix=chunk.context_prefix,
                score=normalized_score,
                rerank_score=0.0,  # 后续由 reranker 填充
                raw_score=score,
                document_name="",
                section="",
                page=chunk.page_numbers[0] if chunk.page_numbers else 0,
                heading_path=chunk.heading_path,
                tenant_id=chunk.tenant_id,
                department_id=chunk.department_id,
                visibility=chunk.visibility,
            )
            results.append(result)

        logger.info("vector_search_complete", extra={"returned": len(results)})

        return results

    async def delete_by_document(self, document_id: str) -> None:
        """
        删除指定文档的所有 chunks。

        Args:
            document_id: 文档 ID
        """
        logger.info("deleting_chunks_by_document", extra={"document_id": document_id})

        # 找到需要删除的索引
        indices_to_remove = [i for i, cid in enumerate(self._chunk_ids) if cid == document_id]

        # 从后往前删除（避免索引偏移）
        for i in reversed(indices_to_remove):
            del self._chunks[i]
            del self._embeddings[i]
            del self._chunk_ids[i]

        logger.info(
            "deleted_chunks", extra={"document_id": document_id, "count": len(indices_to_remove)}
        )

    def _check_acl(self, chunk: ChunkCreate, acl_filter: ACLFilter) -> bool:
        """
        检查 chunk 是否满足 ACL 过滤条件。

        Args:
            chunk: 分块
            acl_filter: ACL 过滤器

        Returns:
            是否满足过滤条件
        """
        # 检查租户
        if chunk.tenant_id != acl_filter.tenant_id:
            return False

        # 检查可见性
        if chunk.visibility not in acl_filter.allowed_visibility:
            return False

        # public 表示同租户公开，不受部门过滤限制
        if chunk.visibility == Visibility.PUBLIC:
            return True

        # 检查部门（chunk 的部门必须在允许的部门列表中）
        if chunk.department_id not in acl_filter.department_ids:
            return False

        return True

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """
        计算余弦相似度。

        Args:
            vec_a: 向量 A
            vec_b: 向量 B

        Returns:
            余弦相似度（-1 到 1）
        """
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must have the same length")

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
