"""
Milvus 向量存储适配器。

实现 VectorStore 协议，替代 InMemoryVectorStore。
"""

from __future__ import annotations

import json
import logging

from pymilvus import DataType, MilvusClient

from app.schemas.chunk import ChunkCreate
from app.schemas.enums import Visibility
from app.schemas.retrieval import RetrievalResult
from app.services.ingestion.identity import stable_chunk_id
from app.services.retrieval.store.base import ACLFilter

logger = logging.getLogger(__name__)

COLLECTION_NAME = "document_chunks_v2"


class MilvusVectorStore:
    """
    Milvus 向量存储。

    实现 VectorStore 协议，用于 full 模式。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        dim: int = 1024,
    ) -> None:
        self._dim = dim
        self._client = MilvusClient(uri=f"http://{host}:{port}")
        self._get_or_create_collection()
        logger.info(
            "milvus_vector_store_init",
            extra={"host": host, "port": port, "dim": dim},
        )

    def _get_or_create_collection(self) -> None:
        """获取或创建 Milvus collection。"""
        if self._client.has_collection(COLLECTION_NAME):
            self._client.load_collection(COLLECTION_NAME)
            return

        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="HR document chunks",
        )
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("document_id", DataType.VARCHAR, max_length=64)
        schema.add_field("document_version", DataType.VARCHAR, max_length=64)
        schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
        schema.add_field("department_id", DataType.VARCHAR, max_length=64)
        schema.add_field("visibility", DataType.VARCHAR, max_length=32)
        schema.add_field("chunk_text", DataType.VARCHAR, max_length=8192)
        schema.add_field("context_prefix", DataType.VARCHAR, max_length=2048)
        schema.add_field("heading_path", DataType.VARCHAR, max_length=512)
        schema.add_field("page", DataType.INT64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        self._client.load_collection(COLLECTION_NAME)
        logger.info(
            "milvus_collection_created",
            extra={"collection_name": COLLECTION_NAME},
        )

    async def add_chunks(
        self,
        chunks: list[ChunkCreate],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        records: list[dict[str, object]] = []
        for i, chunk in enumerate(chunks):
            stored_chunk = chunk
            if not chunk.id:
                stored_chunk = chunk.model_copy(
                    update={
                        "id": stable_chunk_id(
                            document_id=chunk.document_id,
                            document_version=chunk.document_version,
                            heading_path=chunk.heading_path,
                            ordinal=i + 1,
                            chunk_text=chunk.chunk_text,
                        )
                    }
                )
            records.append(
                {
                    "chunk_id": stored_chunk.id,
                    "document_id": stored_chunk.document_id,
                    "document_version": stored_chunk.document_version,
                    "tenant_id": stored_chunk.tenant_id,
                    "department_id": stored_chunk.department_id,
                    "visibility": stored_chunk.visibility.value
                    if hasattr(stored_chunk.visibility, "value")
                    else str(stored_chunk.visibility),
                    "chunk_text": stored_chunk.chunk_text[:8192],
                    "context_prefix": (stored_chunk.context_prefix or "")[:2048],
                    "heading_path": (stored_chunk.heading_path or "")[:512],
                    "page": stored_chunk.page_numbers[0] if stored_chunk.page_numbers else 0,
                    "embedding": embeddings[i],
                }
            )

        self._client.insert(collection_name=COLLECTION_NAME, data=records)
        self._client.flush(collection_name=COLLECTION_NAME)
        logger.info("milvus_chunks_inserted", extra={"count": len(chunks)})

    async def search(
        self,
        query_embedding: list[float],
        acl_filter: ACLFilter,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        # ACL 必须下推到 Milvus 查询表达式，避免无权限向量先进入候选集。
        search_limit = top_k * 5
        expr = self._build_acl_expr(acl_filter)

        results = self._client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=search_limit,
            filter=expr,
            output_fields=[
                "chunk_id",
                "document_id",
                "document_version",
                "tenant_id",
                "department_id",
                "visibility",
                "chunk_text",
                "context_prefix",
                "heading_path",
                "page",
            ],
        )

        output: list[RetrievalResult] = []
        for hit in results[0]:
            if isinstance(hit, dict):
                entity = hit.get("entity", {})
                distance = hit.get("distance", hit.get("score"))
            else:
                entity = hit.entity
                distance = hit.distance
            # ACL 过滤
            vis = entity.get("visibility")
            if entity.get("tenant_id") != acl_filter.tenant_id:
                continue
            vis_enum = Visibility(vis) if vis else Visibility.DEPARTMENT
            if vis_enum not in acl_filter.allowed_visibility:
                continue
            if (
                vis_enum != Visibility.PUBLIC
                and entity.get("department_id") not in acl_filter.department_ids
            ):
                continue

            # pymilvus 3.0 COSINE: distance = cosine_similarity（越大越相似）
            score = distance if distance is not None else 0.0
            score = max(0.0, min(score, 1.0))

            output.append(
                RetrievalResult(
                    chunk_id=entity.get("chunk_id", ""),
                    document_id=entity.get("document_id", ""),
                    document_version=entity.get("document_version", "v1"),
                    chunk_text=entity.get("chunk_text", ""),
                    context_prefix=entity.get("context_prefix", ""),
                    score=score,
                    rerank_score=0.0,
                    raw_score=score,
                    document_name="",
                    section="",
                    page=entity.get("page", 0),
                    heading_path=entity.get("heading_path", ""),
                    tenant_id=entity.get("tenant_id", ""),
                    department_id=entity.get("department_id", ""),
                    visibility=vis_enum,
                )
            )
            if len(output) >= top_k:
                break

        logger.info("milvus_search_done", extra={"returned": len(output)})
        return output

    def _build_acl_expr(self, acl_filter: ACLFilter) -> str:
        """构造 Milvus ACL 表达式，在向量召回前完成租户、部门和可见性过滤。"""
        tenant_expr = f"tenant_id == {self._quote_expr_value(acl_filter.tenant_id)}"
        visibility_values = [
            visibility.value if hasattr(visibility, "value") else str(visibility)
            for visibility in acl_filter.allowed_visibility
        ]

        if not visibility_values:
            return f"{tenant_expr} and visibility in []"

        visibility_expr = (
            "visibility in ["
            + ", ".join(self._quote_expr_value(value) for value in visibility_values)
            + "]"
        )

        public_expr = f"visibility == {self._quote_expr_value(Visibility.PUBLIC.value)}"
        if acl_filter.department_ids:
            department_expr = (
                "department_id in ["
                + ", ".join(
                    self._quote_expr_value(dept_id) for dept_id in acl_filter.department_ids
                )
                + "]"
            )
            access_expr = f"({public_expr} or {department_expr})"
        else:
            access_expr = public_expr

        return f"{tenant_expr} and {visibility_expr} and {access_expr}"

    def _quote_expr_value(self, value: str) -> str:
        """安全引用 Milvus 表达式字符串值。"""
        return json.dumps(value, ensure_ascii=False)

    async def delete_by_document(self, document_id: str) -> None:
        self._client.delete(
            collection_name=COLLECTION_NAME,
            filter=f"document_id == {self._quote_expr_value(document_id)}",
        )
        self._client.flush(collection_name=COLLECTION_NAME)
        logger.info("milvus_document_deleted", extra={"document_id": document_id})
