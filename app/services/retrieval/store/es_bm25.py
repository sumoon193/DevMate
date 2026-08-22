"""
Elasticsearch BM25 存储适配器。

实现 BM25Store 协议，替代 InMemoryBM25Store。
"""

from __future__ import annotations

import logging

from elasticsearch import AsyncElasticsearch

from app.core.exceptions import ExternalServiceError
from app.schemas.chunk import ChunkCreate
from app.schemas.enums import Visibility
from app.schemas.retrieval import RetrievalResult
from app.services.ingestion.identity import stable_chunk_id
from app.services.retrieval.store.base import ACLFilter

logger = logging.getLogger(__name__)

INDEX_NAME = "document_chunks_v3"


class ElasticsearchBM25Store:
    """
    Elasticsearch BM25 存储。

    实现 BM25Store 协议，用于 full 模式。
    """

    def __init__(self, es_url: str = "http://localhost:9200") -> None:
        self._es = AsyncElasticsearch(es_url)
        self._index = INDEX_NAME
        logger.info("es_bm25_store_init", extra={"url": es_url})

    async def ensure_index(self) -> None:
        """确保索引存在，不存在则创建。"""
        exists = await self._es.indices.exists(index=self._index)
        if exists:
            return

        mapping = {
            "settings": {
                "analysis": {
                    "tokenizer": {
                        "zh_ngram_tokenizer": {
                            "type": "ngram",
                            "min_gram": 1,
                            "max_gram": 2,
                        }
                    },
                    "analyzer": {
                        "zh_ngram": {
                            "type": "custom",
                            "tokenizer": "zh_ngram_tokenizer",
                            "filter": ["lowercase"],
                        }
                    },
                }
            },
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "document_version": {"type": "keyword"},
                    "chunk_text": {
                        "type": "text",
                        "analyzer": "zh_ngram",
                        "search_analyzer": "zh_ngram",
                    },
                    "context_prefix": {
                        "type": "text",
                        "analyzer": "zh_ngram",
                        "search_analyzer": "zh_ngram",
                    },
                    "heading_path": {
                        "type": "text",
                        "analyzer": "zh_ngram",
                        "search_analyzer": "zh_ngram",
                    },
                    "page": {"type": "integer"},
                    "tenant_id": {"type": "keyword"},
                    "department_id": {"type": "keyword"},
                    "visibility": {"type": "keyword"},
                }
            },
        }
        await self._es.indices.create(index=self._index, body=mapping)
        logger.info("es_index_created", extra={"index": self._index})

    async def add_chunks(self, chunks: list[ChunkCreate]) -> None:
        await self.ensure_index()

        actions = []
        for ordinal, chunk in enumerate(chunks, start=1):
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
            actions.append({"index": {"_index": self._index, "_id": stored_chunk.id}})
            actions.append(
                {
                    "document_id": stored_chunk.document_id,
                    "document_version": stored_chunk.document_version,
                    "chunk_text": stored_chunk.chunk_text,
                    "context_prefix": stored_chunk.context_prefix or "",
                    "heading_path": stored_chunk.heading_path or "",
                    "page": stored_chunk.page_numbers[0] if stored_chunk.page_numbers else 0,
                    "tenant_id": stored_chunk.tenant_id,
                    "department_id": stored_chunk.department_id,
                    "visibility": stored_chunk.visibility.value
                    if hasattr(stored_chunk.visibility, "value")
                    else str(stored_chunk.visibility),
                }
            )

        response = await self._es.bulk(operations=actions, refresh=True)
        body = response.body if hasattr(response, "body") else response
        if body.get("errors"):
            for item in body.get("items", []):
                operation = item.get("index", {})
                error = operation.get("error")
                if error:
                    reason = error.get("reason", "unknown bulk indexing error")
                    raise ExternalServiceError(f"Elasticsearch chunk indexing failed: {reason}")
            raise ExternalServiceError("Elasticsearch chunk indexing failed")
        logger.info("es_chunks_indexed", extra={"count": len(chunks)})

    async def search(
        self,
        query: str,
        acl_filter: ACLFilter,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        await self.ensure_index()

        # 构建 ACL filter —— tenant + visibility 必须精确匹配
        must_filters = [
            {"term": {"tenant_id": acl_filter.tenant_id}},
            {
                "terms": {
                    "visibility": [
                        v.value if hasattr(v, "value") else str(v)
                        for v in acl_filter.allowed_visibility
                    ]
                }
            },
        ]

        # 非 PUBLIC 的文档还需匹配 department
        # 使用 should + minimum_should_match 实现：
        # PUBLIC 不限部门，非 PUBLIC 需要 department 在列表中
        should_clauses = [
            {"term": {"visibility": Visibility.PUBLIC.value}},
            {"terms": {"department_id": acl_filter.department_ids}},
        ]

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": must_filters
                    + [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["chunk_text^3", "context_prefix^1", "heading_path^1"],
                            }
                        }
                    ],
                    "should": should_clauses,
                    "minimum_should_match": 1,
                }
            },
        }

        resp = await self._es.search(index=self._index, body=body)

        results: list[RetrievalResult] = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            score = hit["_score"] or 0.0
            # 归一化 BM25 分数到 [0, 1]
            normalized = min(score / (score + 1), 1.0)

            vis_str = src.get("visibility", "department")
            vis_enum = (
                Visibility(vis_str)
                if vis_str in [v.value for v in Visibility]
                else Visibility.DEPARTMENT
            )

            results.append(
                RetrievalResult(
                    chunk_id=hit["_id"],
                    document_id=src.get("document_id", ""),
                    document_version=src.get("document_version", "v1"),
                    chunk_text=src.get("chunk_text", ""),
                    context_prefix=src.get("context_prefix", ""),
                    score=normalized,
                    rerank_score=0.0,
                    raw_score=score,
                    document_name="",
                    section="",
                    page=src.get("page", 0),
                    heading_path=src.get("heading_path", ""),
                    tenant_id=src.get("tenant_id", ""),
                    department_id=src.get("department_id", ""),
                    visibility=vis_enum,
                )
            )

        logger.info("es_bm25_search_done", extra={"returned": len(results)})
        return results

    async def delete_by_document(self, document_id: str) -> None:
        await self.ensure_index()
        await self._es.delete_by_query(
            index=self._index,
            body={"query": {"term": {"document_id": document_id}}},
            refresh=True,
        )
        logger.info("es_document_deleted", extra={"document_id": document_id})

    async def close(self) -> None:
        """关闭 ES 客户端连接。"""
        await self._es.close()
