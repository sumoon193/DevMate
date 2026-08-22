"""基于 Qwen Embedding 与 Milvus 的租户隔离长期记忆语义索引。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from pymilvus import DataType, MilvusClient

from app.services.retrieval.embedding.base import Embedder

COLLECTION_NAME = "episodic_memories_v1"


class MilvusMemorySemanticIndex:
    """只保存记忆语义副本，权威状态仍由 PostgreSQL 记忆表持有。"""

    def __init__(
        self,
        *,
        embedder: Embedder,
        dimension: int,
        host: str = "localhost",
        port: int = 19530,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
        connection_attempts: int = 6,
        retry_delay_seconds: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._embedder = embedder
        self._dimension = dimension
        self._client = client or self._connect_client(
            uri=f"http://{host}:{port}",
            client_factory=client_factory or MilvusClient,
            connection_attempts=connection_attempts,
            retry_delay_seconds=retry_delay_seconds,
            sleep=sleep,
        )
        self._get_or_create_collection()

    @staticmethod
    def _connect_client(
        *,
        uri: str,
        client_factory: Callable[..., Any],
        connection_attempts: int,
        retry_delay_seconds: float,
        sleep: Callable[[float], None],
    ) -> Any:
        if connection_attempts < 1:
            raise ValueError("connection_attempts must be positive")
        for attempt in range(connection_attempts):
            try:
                return client_factory(uri=uri)
            except Exception:
                if attempt == connection_attempts - 1:
                    raise
                sleep(retry_delay_seconds)
        raise AssertionError("unreachable")

    def _get_or_create_collection(self) -> None:
        if self._client.has_collection(COLLECTION_NAME):
            self._client.load_collection(COLLECTION_NAME)
            return
        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="Tenant-scoped episodic memories",
        )
        schema.add_field("memory_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
        schema.add_field("content", DataType.VARCHAR, max_length=8192)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dimension)
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 128},
        )
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        self._client.load_collection(COLLECTION_NAME)

    async def upsert(self, *, memory_id: str, tenant_id: str, content: str) -> None:
        vector = (await self._embedder.embed_documents([content]))[0]
        self._client.upsert(
            collection_name=COLLECTION_NAME,
            data=[
                {
                    "memory_id": memory_id,
                    "tenant_id": tenant_id,
                    "content": content[:8192],
                    "embedding": vector,
                }
            ],
        )
        self._client.flush(COLLECTION_NAME)

    async def search(self, *, tenant_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        vector = await self._embedder.embed_query(query)
        results = self._client.search(
            collection_name=COLLECTION_NAME,
            data=[vector],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit,
            filter=f"tenant_id == {json.dumps(tenant_id, ensure_ascii=False)}",
            output_fields=["memory_id", "tenant_id"],
        )
        matches: list[tuple[str, float]] = []
        for hit in results[0] if results else []:
            if isinstance(hit, dict):
                entity = hit.get("entity", {})
                memory_id = entity.get("memory_id") or hit.get("id")
                score = hit.get("distance", hit.get("score", 0.0))
            else:
                entity = hit.entity
                memory_id = entity.get("memory_id")
                score = hit.distance
            if isinstance(memory_id, str):
                matches.append((memory_id, max(0.0, min(float(score), 1.0))))
        return matches

    async def delete(self, *, memory_id: str, tenant_id: str) -> None:
        self._client.delete(
            collection_name=COLLECTION_NAME,
            filter=(
                f"memory_id == {json.dumps(memory_id)} and tenant_id == {json.dumps(tenant_id)}"
            ),
        )
        self._client.flush(COLLECTION_NAME)
