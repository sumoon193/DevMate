"""
文档上传端点。

接收文件上传，创建入库任务，并按配置选择同步入库或 Celery 异步入库。
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencies import ServiceContainer, get_container
from app.api.schemas import DocumentCreateResponse, IngestionStatusResponse
from app.config import get_settings
from app.core.exceptions import NotFoundError, ValidationError
from app.schemas.enums import DocumentStatus
from app.services.ingestion.dispatcher import IngestionDispatcher
from app.services.ingestion.job import IngestionJobPayload
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.ingestion.store import IngestionTaskStore, InMemoryIngestionTaskStore
from app.services.ingestion.task import IngestionTask
from app.services.ingestion.worker import run_ingestion_job
from app.services.parser.markdown_parser import MarkdownParser
from app.services.parser.plain_parser import PlainTextParser
from app.services.parser.registry import ParserRegistry

router = APIRouter(tags=["documents"])

_fallback_task_store = InMemoryIngestionTaskStore()
_SUPPORTED_UPLOAD_EXTENSIONS = (
    ".md",
    ".txt",
    ".markdown",
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
)
_SUPPORTED_UPLOAD_TEXT = ", ".join(_SUPPORTED_UPLOAD_EXTENSIONS)


async def _persist_document_and_task(
    doc_id: str,
    filename: str,
    mime_type: str,
    storage_key: str,
    task: IngestionTask,
    tenant_id: str,
    department_id: str,
    visibility: str,
) -> None:
    """full 模式：将文档元数据和入库任务持久化到 PostgreSQL。"""
    from app.db import crud as db
    from app.db.session import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        # 保存文档元数据（Document 表由 ORM 基础设施管理，这里直接用 crud）
        from app.models.document import Document

        doc = Document(
            id=doc_id,
            title=filename,
            file_path=storage_key,
            mime_type=mime_type,
            status="queued",
            tenant_id=tenant_id,
            department_id=department_id,
            visibility=visibility,
        )
        session.add(doc)
        await session.commit()

        # 保存入库任务
        await db.save_ingestion_task(
            session,
            task_id=task.id,
            document_id=doc_id,
            filename=filename,
            mime_type=mime_type,
            storage_key=storage_key,
        )


async def _sync_persisted_task_snapshot(task: IngestionTask) -> None:
    """full 模式：将最新入库任务状态同步到 PostgreSQL 快照。"""
    from app.db import crud as db
    from app.db.session import get_session_factory

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


def reset_documents_store() -> None:
    """重置任务存储（测试用）。"""
    _fallback_task_store.clear()


def _build_pipeline(container: ServiceContainer | None = None) -> IngestionPipeline:
    """构建 V1 入库流水线。"""
    registry = ParserRegistry()
    registry.register(MarkdownParser())
    registry.register(PlainTextParser())
    from app.services.parser.office_parser import OfficeParser

    registry.register(OfficeParser())
    return IngestionPipeline(
        parser_registry=registry,
        embedder=getattr(container, "embedder", None),
        vector_store=getattr(container, "vector_store", None),
        bm25_store=getattr(container, "bm25_store", None),
        embedding_batch_size=get_settings().embedding_batch_size,
    )


def _get_task_store(container: ServiceContainer) -> IngestionTaskStore:
    """获取当前容器的入库任务状态存储。"""
    return getattr(container, "ingestion_task_store", _fallback_task_store)


def _build_dispatcher(container: ServiceContainer) -> IngestionDispatcher:
    """构建入库调度器。"""
    settings = get_settings()
    task_store = _get_task_store(container)

    async def pipeline_runner(payload: IngestionJobPayload) -> IngestionTask:
        return await run_ingestion_job(
            payload=payload,
            task_store=task_store,
            storage=container.storage,
            pipeline_factory=lambda: _build_pipeline(container),
        )

    return IngestionDispatcher(
        execution_mode=settings.ingestion_execution_mode,
        task_store=task_store,
        pipeline_runner=pipeline_runner,
    )


def _build_storage_key(tenant_id: str, document_id: str, filename: str) -> str:
    """构建对象存储 key，并丢弃上传文件名中的路径部分。"""
    safe_filename = Path(filename).name or "document.txt"
    return f"{tenant_id}/{document_id}/{safe_filename}"


@router.post("/documents", response_model=DocumentCreateResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(description=f"上传文件（{_SUPPORTED_UPLOAD_TEXT}）"),
    tenant_id: str = Form(description="租户 ID"),
    department_id: str = Form(description="部门 ID"),
    visibility: str = Form(default="department", description="可见性"),
    container: ServiceContainer = Depends(get_container),
) -> DocumentCreateResponse:
    settings = get_settings()
    if settings.app_mode == "full":
        from app.services.security.oidc import current_tenant_id

        if tenant_id != current_tenant_id():
            from app.core.exceptions import PermissionError

            raise PermissionError("Document tenant does not match authenticated tenant")
    # 检测文件类型
    filename = file.filename or "unknown.txt"
    mime_type = IngestionPipeline.detect_mime_type(filename)

    if mime_type is None:
        raise ValidationError(f"不支持的文件类型: {filename}。当前支持: {_SUPPORTED_UPLOAD_TEXT}")

    # 读取文件内容
    content = await file.read()
    if not content:
        raise ValidationError("上传文件为空")

    # 创建文档、保存原文并创建任务
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    storage_key = _build_storage_key(tenant_id, doc_id, filename)
    container.storage.put_object(storage_key, content, content_type=mime_type)

    task = IngestionTask(
        document_id=doc_id,
        filename=filename,
        mime_type=mime_type,
        storage_key=storage_key,
    )

    # full 模式：持久化文档和任务到 PostgreSQL
    if settings.app_mode == "full":
        await _persist_document_and_task(
            doc_id=doc_id,
            filename=filename,
            mime_type=mime_type,
            storage_key=storage_key,
            task=task,
            tenant_id=tenant_id,
            department_id=department_id,
            visibility=visibility,
        )

    document_version = await container.document_version_registry.register(
        document_id=doc_id,
        content=content,
    )
    task.document_version = document_version.id

    dispatcher = _build_dispatcher(container)
    task = await dispatcher.dispatch(
        task=task,
        tenant_id=tenant_id,
        department_id=department_id,
        visibility=visibility,
    )
    if settings.app_mode == "full":
        await _sync_persisted_task_snapshot(task)

    if task.current_stage == "ready":
        status = DocumentStatus.READY
    elif task.current_stage == "failed":
        status = DocumentStatus.FAILED
    else:
        status = DocumentStatus.QUEUED

    return DocumentCreateResponse(
        id=doc_id,
        document_version=document_version.id,
        task_id=task.id,
        status=status,
        message=_build_upload_message(task),
    )


@router.get("/ingestions/{task_id}", response_model=IngestionStatusResponse)
async def get_ingestion_status(
    task_id: str,
    container: ServiceContainer = Depends(get_container),
) -> IngestionStatusResponse:
    settings = get_settings()

    # full 模式：优先从 PostgreSQL 读
    if settings.app_mode == "full":
        from app.db import crud as db
        from app.db.session import get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            record = await db.get_ingestion_task(session, task_id)
            if record:
                return IngestionStatusResponse(
                    task_id=record.id,
                    document_id=record.document_id,
                    status=record.current_stage,
                    progress=record.progress,
                    error_message=record.error_message,
                )

    # fallback：从内存 store 读
    task = _get_task_store(container).get(task_id)
    if not task:
        raise NotFoundError(f"任务不存在: {task_id}")

    return IngestionStatusResponse(
        task_id=task.id,
        document_id=task.document_id,
        status=task.current_stage,
        progress=task.progress,
        error_message=task.error_message,
    )


def _build_upload_message(task: IngestionTask) -> str:
    """构建上传响应消息。"""
    if task.current_stage == "ready":
        return f"文档入库完成，共 {task.total_chunks} 个分块"
    if task.current_stage == "failed":
        return f"入库失败: {task.error_message}"
    return "文档已接收，正在异步入库"
