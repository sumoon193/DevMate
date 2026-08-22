"""
Celery 阶段级入库 tasks。

每个阶段独立为 Celery task，支持单阶段 retry。
用 chain() 串联：parse → chunk → embed → index → finalize。

失败时：
- 每个 task 自动重试 max_retries=2 次
- 最终失败后更新 Redis 中的 task 状态为 failed
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from celery import chain

from app.services.ingestion.celery_app import celery_app

logger = logging.getLogger(__name__)

_worker_loop: asyncio.AbstractEventLoop | None = None
_worker_loop_pid: int | None = None

# 最大重试次数
MAX_RETRIES = 2


@celery_app.task(name="system.ping")
def system_ping(nonce: str) -> dict[str, str]:
    """真实 broker/worker/result-backend 往返探针。"""
    return {"status": "ok", "nonce": nonce}


def _run_async(coro):
    """在同步 Celery worker 中执行 async 函数。"""
    global _worker_loop, _worker_loop_pid

    process_id = os.getpid()
    if (
        _worker_loop is None
        or _worker_loop.is_closed()
        or _worker_loop_pid != process_id
    ):
        if _worker_loop is not None and not _worker_loop.is_closed():
            _worker_loop.close()
        _worker_loop = asyncio.new_event_loop()
        _worker_loop_pid = process_id
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop.run_until_complete(coro)


def _save_task_state(task_id: str, task_data: dict) -> None:
    """把 task 状态写回 task store，并在 full mode 同步 PostgreSQL 快照。"""
    from app.services.ingestion.task import IngestionTask
    from app.services.ingestion.worker import build_task_store

    try:
        task = IngestionTask.model_validate(task_data)
    except Exception:
        logger.warning("save_task_state_invalid", extra={"task_id": task_id}, exc_info=True)
        return

    try:
        store = build_task_store()
        store.save(task)
    except Exception:
        logger.warning("save_task_state_store_failed", extra={"task_id": task_id}, exc_info=True)

    _save_task_state_to_postgres(task)


def _save_task_state_to_postgres(task: Any) -> None:
    """full mode 下同步 PostgreSQL，避免 API 状态端点读到旧 queued 快照。"""
    try:
        from app.config import get_settings

        if get_settings().app_mode != "full":
            return

        async def update_snapshot() -> None:
            from app.db import crud as db
            from app.db.session import get_session_factory, init_db

            try:
                factory = get_session_factory()
            except RuntimeError:
                # Celery starts in a separate process from FastAPI's lifespan.
                await init_db()
                factory = get_session_factory()
            async with factory() as session:
                await db.update_ingestion_task(
                    session,
                    task.id,
                    current_stage=task.current_stage.value,
                    progress=task.progress,
                    total_chunks=task.total_chunks,
                    error_message=task.error_message,
                    error_code=task.error_code,
                    stages_json=[stage.model_dump(mode="json") for stage in task.stages],
                )

        _run_async(update_snapshot())
    except Exception:
        logger.warning(
            "save_task_state_postgres_failed",
            extra={"task_id": getattr(task, "id", "")},
            exc_info=True,
        )


def _build_pipeline():
    """构建入库 pipeline（每个 task 独立构建，避免跨进程共享）。"""
    from app.services.ingestion.worker import build_pipeline

    return build_pipeline()


def _build_storage():
    """构建存储后端。"""
    from app.services.ingestion.worker import build_storage

    return build_storage()


# ── Stage 1: 解析 ─────────────────────────────────────────────────────


@celery_app.task(
    name="ingestion.stage_parse",
    bind=True,
    max_retries=MAX_RETRIES,
    default_retry_delay=5,
)
def stage_parse(self, payload: dict) -> dict:
    """
    阶段 1：解析文件。

    从 MinIO 读取原文 → 解析 → 清洗。
    返回: {task, parsed_blocks_json, doc_metadata}
    """
    from app.services.ingestion.job import IngestionJobPayload
    from app.services.ingestion.task import IngestionStage

    job = IngestionJobPayload.model_validate(payload)
    task = job.task
    task_id = task.id

    try:
        logger.info("stage_parse_start", extra={"task_id": task_id})

        storage = _build_storage()
        file_content = storage.get_object(task.storage_key)

        pipeline = _build_pipeline()

        # 解析
        parsed_doc = _run_async(
            pipeline._stage_parse(
                task,
                file_content,
                {
                    "tenant_id": job.tenant_id,
                    "department_id": job.department_id,
                    "visibility": job.visibility,
                },
            )
        )

        # 清洗
        cleaned_doc = pipeline._stage_clean(task, parsed_doc)

        _save_task_state(task_id, task.model_dump(mode="json"))

        return {
            "task": task.model_dump(mode="json"),
            "parsed_doc_json": cleaned_doc.model_dump(mode="json"),
            "job_payload": payload,
        }

    except Exception as exc:
        logger.error("stage_parse_failed", extra={"task_id": task_id, "error": str(exc)})
        task.fail_stage(IngestionStage.PARSING, str(exc))
        _save_task_state(task_id, task.model_dump(mode="json"))
        raise self.retry(exc=exc) from exc


# ── Stage 2: 分块 ─────────────────────────────────────────────────────


@celery_app.task(
    name="ingestion.stage_chunk",
    bind=True,
    max_retries=MAX_RETRIES,
    default_retry_delay=5,
)
def stage_chunk(self, prev_result: dict) -> dict:
    """
    阶段 2：分块。

    从上一步的解析结果生成 chunks。
    返回: {task, chunks_json, ...}
    """
    from app.services.ingestion.task import IngestionStage
    from app.services.parser.base import ParsedDocument

    task_data = prev_result["task"]
    task_id = task_data["id"]

    try:
        logger.info("stage_chunk_start", extra={"task_id": task_id})

        from app.services.ingestion.task import IngestionTask

        task = IngestionTask.model_validate(task_data)

        parsed_doc = ParsedDocument.model_validate(prev_result["parsed_doc_json"])

        pipeline = _build_pipeline()
        chunks = _run_async(pipeline._stage_chunk(task, parsed_doc))
        chunks = pipeline._stage_contextualize(task, chunks)

        _save_task_state(task_id, task.model_dump(mode="json"))

        # 序列化 chunks
        chunks_json = [c.model_dump(mode="json") for c in chunks]

        prev_result["task"] = task.model_dump(mode="json")
        prev_result["chunks_json"] = chunks_json
        prev_result.pop("parsed_doc_json", None)
        return prev_result

    except Exception as exc:
        logger.error("stage_chunk_failed", extra={"task_id": task_id, "error": str(exc)})
        from app.services.ingestion.task import IngestionTask

        task = IngestionTask.model_validate(task_data)
        task.fail_stage(IngestionStage.CHUNKING, str(exc))
        _save_task_state(task_id, task.model_dump(mode="json"))
        raise self.retry(exc=exc) from exc


# ── Stage 3: Embedding ────────────────────────────────────────────────


@celery_app.task(
    name="ingestion.stage_embed",
    bind=True,
    max_retries=MAX_RETRIES,
    default_retry_delay=10,
)
def stage_embed(self, prev_result: dict) -> dict:
    """
    阶段 3：向量化。

    为每个 chunk 生成 embedding 向量。
    返回: {task, chunks_json, embeddings_json, ...}
    """
    from app.services.ingestion.task import IngestionStage, IngestionTask
    from app.services.ingestion.worker import build_pipeline

    task_data = prev_result["task"]
    task_id = task_data["id"]

    try:
        logger.info("stage_embed_start", extra={"task_id": task_id})

        task = IngestionTask.model_validate(task_data)

        pipeline = build_pipeline()

        # 重建 chunks
        from app.schemas.chunk import ChunkCreate

        chunks = [ChunkCreate.model_validate(c) for c in prev_result["chunks_json"]]

        _run_async(pipeline._stage_embed(task, chunks))

        # 收集 embeddings
        embeddings = []
        for chunk in chunks:
            emb = chunk.acl_metadata.get("_embedding", []) if chunk.acl_metadata else []
            embeddings.append(emb)

        _save_task_state(task_id, task.model_dump(mode="json"))

        prev_result["task"] = task.model_dump(mode="json")
        prev_result["embeddings_json"] = embeddings
        return prev_result

    except Exception as exc:
        logger.error("stage_embed_failed", extra={"task_id": task_id, "error": str(exc)})
        task = IngestionTask.model_validate(task_data)
        task.fail_stage(IngestionStage.EMBEDDING, str(exc))
        _save_task_state(task_id, task.model_dump(mode="json"))
        raise self.retry(exc=exc) from exc


# ── Stage 4: 索引 ─────────────────────────────────────────────────────


@celery_app.task(
    name="ingestion.stage_index",
    bind=True,
    max_retries=MAX_RETRIES,
    default_retry_delay=10,
)
def stage_index(self, prev_result: dict) -> dict:
    """
    阶段 4：写入 Milvus + ES 索引。
    """
    from app.services.ingestion.task import IngestionStage, IngestionTask

    task_data = prev_result["task"]
    task_id = task_data["id"]

    try:
        logger.info("stage_index_start", extra={"task_id": task_id})

        task = IngestionTask.model_validate(task_data)

        pipeline = _build_pipeline()

        from app.schemas.chunk import ChunkCreate

        chunks = [ChunkCreate.model_validate(c) for c in prev_result["chunks_json"]]
        embeddings = prev_result["embeddings_json"]

        # 注入 embeddings 到 chunk metadata
        for chunk, emb in zip(chunks, embeddings, strict=True):
            if not chunk.acl_metadata:
                chunk.acl_metadata = {}
            chunk.acl_metadata["_embedding"] = emb

        _run_async(pipeline._stage_index(task, chunks))

        _save_task_state(task_id, task.model_dump(mode="json"))

        prev_result["task"] = task.model_dump(mode="json")
        return prev_result

    except Exception as exc:
        logger.error("stage_index_failed", extra={"task_id": task_id, "error": str(exc)})
        task = IngestionTask.model_validate(task_data)
        task.fail_stage(IngestionStage.INDEXING, str(exc))
        _save_task_state(task_id, task.model_dump(mode="json"))
        raise self.retry(exc=exc) from exc


# ── Stage 5: 完成 ─────────────────────────────────────────────────────


@celery_app.task(
    name="ingestion.stage_finalize",
    bind=True,
    max_retries=0,
)
def stage_finalize(self, prev_result: dict) -> dict:
    """
    阶段 5：标记入库完成。
    """
    from app.services.ingestion.task import IngestionStage, IngestionTask

    task_data = prev_result["task"]
    task_id = task_data["id"]

    task = IngestionTask.model_validate(task_data)
    chunk_count = len(prev_result.get("chunks_json", []))
    task.start_stage(IngestionStage.READY)
    task.complete_stage(IngestionStage.READY, chunk_count=chunk_count)

    _save_task_state(task_id, task.model_dump(mode="json"))

    logger.info(
        "ingestion_complete",
        extra={"task_id": task_id, "total_chunks": chunk_count},
    )

    # 清理不需要传递的中间数据
    prev_result.pop("chunks_json", None)
    prev_result.pop("embeddings_json", None)
    prev_result["task"] = task.model_dump(mode="json")
    return prev_result


# ── 组合入口 ───────────────────────────────────────────────────────────


def dispatch_ingestion_chain(payload: dict) -> None:
    """
    调度完整的入库 task chain。

    使用 Celery chain() 串联各阶段，前一阶段的输出自动传递给下一阶段。
    """
    workflow = chain(
        stage_parse.s(payload),
        stage_chunk.s(),
        stage_embed.s(),
        stage_index.s(),
        stage_finalize.s(),
    )
    workflow.apply_async()


# ── 兼容旧接口 ─────────────────────────────────────────────────────────


@celery_app.task(name="ingestion.run_document_ingestion")
def run_ingestion_task(payload: dict) -> dict:
    """
    旧版单 task 入口（保持向后兼容）。

    新版本请使用 dispatch_ingestion_chain()。
    """
    from app.services.ingestion.worker import run_ingestion_job_sync

    return run_ingestion_job_sync(payload)
