"""入库 worker 执行逻辑。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from app.config import Settings, get_settings
from app.services.chunker.hybrid import HybridChunker
from app.services.ingestion.job import IngestionJobPayload
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.ingestion.store import (
    IngestionTaskStore,
    InMemoryIngestionTaskStore,
    RedisIngestionTaskStore,
)
from app.services.ingestion.task import IngestionTask
from app.services.parser.markdown_parser import MarkdownParser
from app.services.parser.plain_parser import PlainTextParser
from app.services.parser.registry import ParserRegistry
from app.services.storage.local_storage import LocalFileStorage
from app.services.storage.protocol import StorageBackend

logger = logging.getLogger(__name__)

PipelineFactory = Callable[[], IngestionPipeline]


async def run_ingestion_job(
    *,
    payload: IngestionJobPayload,
    task_store: IngestionTaskStore,
    storage: StorageBackend,
    pipeline_factory: PipelineFactory,
) -> IngestionTask:
    """执行单个入库 job，并把最终任务状态写回 store。"""
    task = payload.task
    try:
        file_content = storage.get_object(task.storage_key)
        pipeline = pipeline_factory()
        result = await pipeline.run(
            task=task,
            file_content=file_content,
            tenant_id=payload.tenant_id,
            department_id=payload.department_id,
            visibility=payload.visibility,
        )
    except Exception as exc:
        logger.exception("ingestion_worker_failed", extra={"task_id": task.id})
        task.fail_stage(task.current_stage, str(exc))
        result = task

    task_store.save(result)
    return result


async def run_ingestion_job_from_payload(payload: dict) -> IngestionTask:
    """从 Celery JSON payload 执行入库 job。"""
    job_payload = IngestionJobPayload.model_validate(payload)
    settings = get_settings()
    return await run_ingestion_job(
        payload=job_payload,
        task_store=build_task_store(settings),
        storage=build_storage(settings),
        pipeline_factory=lambda: build_pipeline(settings),
    )


def run_ingestion_job_sync(payload: dict) -> dict:
    """Celery task 的同步入口。"""
    result = asyncio.run(run_ingestion_job_from_payload(payload))
    return result.model_dump(mode="json")


def build_task_store(settings: Settings | None = None) -> IngestionTaskStore:
    """按配置构建任务状态存储。"""
    settings = settings or get_settings()
    if settings.ingestion_task_store == "redis":
        return RedisIngestionTaskStore(settings.redis_url)
    return InMemoryIngestionTaskStore()


def build_storage(settings: Settings | None = None) -> StorageBackend:
    """按配置构建对象存储。"""
    settings = settings or get_settings()
    if settings.app_mode == "full":
        from app.services.storage.minio_storage import MinIOStorage

        return MinIOStorage(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
        )
    return LocalFileStorage()


def build_pipeline(settings: Settings | None = None) -> IngestionPipeline:
    """按配置构建 worker 入库流水线。"""
    settings = settings or get_settings()
    registry = ParserRegistry()
    registry.register(MarkdownParser())
    registry.register(PlainTextParser())

    # 注册 Office/PDF parser（全模式共用）
    from app.services.parser.office_parser import OfficeParser

    registry.register(OfficeParser())

    if settings.app_mode == "full":
        from app.api.dependencies import _build_ai_adapters
        from app.services.retrieval.store.es_bm25 import ElasticsearchBM25Store
        from app.services.retrieval.store.milvus_vector import MilvusVectorStore

        _, embedder, _ = _build_ai_adapters(settings)
        return IngestionPipeline(
            parser_registry=registry,
            chunker=HybridChunker(),
            embedder=embedder,
            vector_store=MilvusVectorStore(
                host=settings.milvus_host,
                port=settings.milvus_port,
                dim=settings.embedding_dim,
            ),
            bm25_store=ElasticsearchBM25Store(es_url=settings.es_url),
            embedding_batch_size=settings.embedding_batch_size,
        )

    return IngestionPipeline(
        parser_registry=registry,
        chunker=HybridChunker(),
        embedding_batch_size=settings.embedding_batch_size,
    )
