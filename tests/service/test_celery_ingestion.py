"""Celery 文档入库调度测试。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.services.chunker.hybrid import HybridChunker
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.ingestion.task import IngestionStage, IngestionTask
from app.services.parser.markdown_parser import MarkdownParser
from app.services.parser.plain_parser import PlainTextParser
from app.services.parser.registry import ParserRegistry
from app.services.retrieval.embedding.mock_embedding import MockEmbedder
from app.services.retrieval.store.memory_vector import InMemoryVectorStore
from app.services.storage.local_storage import LocalFileStorage


def _task(document_id: str = "doc_celery_001") -> IngestionTask:
    return IngestionTask(
        document_id=document_id,
        filename="hr-policy.md",
        mime_type="text/markdown",
        storage_key=f"tenant_001/{document_id}/hr-policy.md",
    )


def _content() -> bytes:
    return (
        "# HR 入职与转正制度\n\n"
        "## 入职材料\n\n"
        "新员工入职当天需提交身份证明、学历证明、离职证明，并签署劳动合同。\n\n"
        "## 转正流程\n\n"
        "试用期满前五个工作日，员工提交转正申请，主管完成评估，HR 归档。"
    ).encode()


def _pipeline_factory(
    vector_store: InMemoryVectorStore | None = None,
) -> Callable[[], IngestionPipeline]:
    def build() -> IngestionPipeline:
        registry = ParserRegistry()
        registry.register(MarkdownParser())
        registry.register(PlainTextParser())
        return IngestionPipeline(
            parser_registry=registry,
            chunker=HybridChunker(),
            embedder=MockEmbedder(),
            vector_store=vector_store or InMemoryVectorStore(),
        )

    return build


@pytest.mark.asyncio
async def test_dispatcher_runs_sync_pipeline_and_saves_ready_task() -> None:
    """sync 模式应直接执行入库流水线并保存 ready 状态。"""
    from app.services.ingestion.dispatcher import IngestionDispatcher
    from app.services.ingestion.job import IngestionJobPayload
    from app.services.ingestion.store import InMemoryIngestionTaskStore

    store = InMemoryIngestionTaskStore()

    async def runner(payload: IngestionJobPayload) -> IngestionTask:
        task = payload.task
        task.start_stage(IngestionStage.READY)
        task.complete_stage(IngestionStage.READY, chunk_count=2)
        return task

    dispatcher = IngestionDispatcher(
        execution_mode="sync",
        task_store=store,
        pipeline_runner=runner,
    )

    result = await dispatcher.dispatch(
        task=_task(),
        tenant_id="tenant_001",
        department_id="dept_hr",
        visibility="department",
    )

    saved = store.get(result.id)
    assert result.current_stage == IngestionStage.READY
    assert result.total_chunks == 2
    assert saved is not None
    assert saved.current_stage == IngestionStage.READY


@pytest.mark.asyncio
async def test_dispatcher_queues_celery_job_without_running_pipeline() -> None:
    """celery 模式应保存 queued 任务并只发送 Celery payload。"""
    from app.services.ingestion.dispatcher import IngestionDispatcher
    from app.services.ingestion.job import IngestionJobPayload
    from app.services.ingestion.store import InMemoryIngestionTaskStore

    store = InMemoryIngestionTaskStore()
    enqueued: list[dict[str, Any]] = []

    async def runner(payload: IngestionJobPayload) -> IngestionTask:
        raise AssertionError("celery queue mode must not run the pipeline inline")

    def enqueue(payload: dict[str, Any]) -> None:
        enqueued.append(payload)

    task = _task()
    dispatcher = IngestionDispatcher(
        execution_mode="celery",
        task_store=store,
        pipeline_runner=runner,
        celery_enqueue=enqueue,
    )

    result = await dispatcher.dispatch(
        task=task,
        tenant_id="tenant_001",
        department_id="dept_hr",
        visibility="department",
    )

    saved = store.get(task.id)
    assert result.current_stage == IngestionStage.QUEUED
    assert saved is not None
    assert saved.current_stage == IngestionStage.QUEUED
    assert len(enqueued) == 1
    assert enqueued[0]["task"]["id"] == task.id
    assert enqueued[0]["task"]["storage_key"] == task.storage_key


@pytest.mark.asyncio
async def test_run_ingestion_job_reads_storage_and_updates_task_store(tmp_path: Path) -> None:
    """worker job 应从 storage 读取原文，执行 pipeline，并写回任务状态。"""
    from app.services.ingestion.job import IngestionJobPayload
    from app.services.ingestion.store import InMemoryIngestionTaskStore
    from app.services.ingestion.worker import run_ingestion_job

    storage = LocalFileStorage(tmp_path)
    vector_store = InMemoryVectorStore()
    task = _task("doc_worker_001")
    storage.put_object(task.storage_key, _content(), content_type="text/markdown")

    store = InMemoryIngestionTaskStore()
    store.save(task)
    payload = IngestionJobPayload(
        task=task,
        tenant_id="tenant_001",
        department_id="dept_hr",
        visibility="department",
    )

    result = await run_ingestion_job(
        payload=payload,
        task_store=store,
        storage=storage,
        pipeline_factory=_pipeline_factory(vector_store),
    )

    saved = store.get(task.id)
    assert result.current_stage == IngestionStage.READY
    assert result.total_chunks > 0
    assert saved is not None
    assert saved.current_stage == IngestionStage.READY


def test_stage_parse_records_single_completed_parse_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Celery 分阶段 parse 不应额外留下未完成的 parsing 记录。"""
    from app.services.ingestion import celery_tasks
    from app.services.ingestion.job import IngestionJobPayload

    storage = LocalFileStorage(tmp_path)
    task = _task("doc_stage_parse_001")
    storage.put_object(task.storage_key, _content(), content_type="text/markdown")

    monkeypatch.setattr(celery_tasks, "_build_storage", lambda: storage)
    monkeypatch.setattr(celery_tasks, "_build_pipeline", _pipeline_factory())
    monkeypatch.setattr(celery_tasks, "_save_task_state", lambda *_args, **_kwargs: None)

    payload = IngestionJobPayload(
        task=task,
        tenant_id="tenant_001",
        department_id="dept_hr",
        visibility="department",
    ).model_dump(mode="json")

    result = celery_tasks.stage_parse(payload)
    updated_task = IngestionTask.model_validate(result["task"])
    parse_records = [
        record for record in updated_task.stages if record.stage == IngestionStage.PARSING
    ]

    assert len(parse_records) == 1
    assert parse_records[0].completed_at is not None


def test_stage_chain_reaches_ready_without_duplicate_open_stage_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Celery 分阶段链路应能串完，并且每个成功阶段只有一条完成记录。"""
    from app.services.ingestion import celery_tasks
    from app.services.ingestion.job import IngestionJobPayload

    storage = LocalFileStorage(tmp_path)
    task = _task("doc_stage_chain_001")
    storage.put_object(task.storage_key, _content(), content_type="text/markdown")

    monkeypatch.setattr(celery_tasks, "_build_storage", lambda: storage)
    monkeypatch.setattr(celery_tasks, "_build_pipeline", _pipeline_factory())
    monkeypatch.setattr(celery_tasks, "_save_task_state", lambda *_args, **_kwargs: None)

    payload = IngestionJobPayload(
        task=task,
        tenant_id="tenant_001",
        department_id="dept_hr",
        visibility="department",
    ).model_dump(mode="json")

    result = celery_tasks.stage_parse(payload)
    result = celery_tasks.stage_chunk(result)
    result = celery_tasks.stage_embed(result)
    result = celery_tasks.stage_index(result)
    result = celery_tasks.stage_finalize(result)

    updated_task = IngestionTask.model_validate(result["task"])

    assert updated_task.current_stage == IngestionStage.READY
    assert updated_task.total_chunks > 0
    for stage in (
        IngestionStage.PARSING,
        IngestionStage.CLEANING,
        IngestionStage.CHUNKING,
        IngestionStage.CONTEXTUALIZING,
        IngestionStage.EMBEDDING,
        IngestionStage.INDEXING,
        IngestionStage.READY,
    ):
        records = [record for record in updated_task.stages if record.stage == stage]
        assert len(records) == 1
        assert records[0].completed_at is not None


def test_save_task_state_updates_postgres_snapshot_in_full_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """full mode 下 Celery worker 状态应同步到 PostgreSQL，供 API 状态端点读取。"""
    from app import config as config_module
    from app.db import crud as db_module
    from app.db import session as session_module
    from app.services.ingestion import celery_tasks
    from app.services.ingestion import worker as worker_module

    saved_to_store: list[IngestionTask] = []
    db_updates: list[tuple[str, dict[str, Any]]] = []

    class FakeStore:
        def save(self, task: IngestionTask) -> None:
            saved_to_store.append(task)

    class FakeSessionContext:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *_args: object) -> None:
            return None

    class FakeSessionFactory:
        def __call__(self) -> FakeSessionContext:
            return FakeSessionContext()

    async def fake_update_ingestion_task(
        _session: object,
        task_id: str,
        **values: Any,
    ) -> None:
        db_updates.append((task_id, values))

    task = _task("doc_full_celery_001")
    task.start_stage(IngestionStage.READY)
    task.complete_stage(IngestionStage.READY, chunk_count=2)

    monkeypatch.setattr(worker_module, "build_task_store", lambda *_args, **_kwargs: FakeStore())
    monkeypatch.setattr(config_module, "get_settings", lambda: SimpleNamespace(app_mode="full"))
    monkeypatch.setattr(session_module, "get_session_factory", lambda: FakeSessionFactory())
    monkeypatch.setattr(db_module, "update_ingestion_task", fake_update_ingestion_task)

    celery_tasks._save_task_state(task.id, task.model_dump(mode="json"))

    assert [saved.id for saved in saved_to_store] == [task.id]
    assert db_updates == [
        (
            task.id,
            {
                "current_stage": IngestionStage.READY.value,
                "progress": 1.0,
                "total_chunks": 2,
                "error_message": None,
                "error_code": None,
                "stages_json": [stage.model_dump(mode="json") for stage in task.stages],
            },
        )
    ]


def test_save_task_state_initializes_worker_database_before_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh Celery worker must initialize its own database session factory."""
    from app import config as config_module
    from app.db import crud as db_module
    from app.db import session as session_module
    from app.services.ingestion import celery_tasks
    from app.services.ingestion import worker as worker_module

    initialized: list[bool] = []
    db_updates: list[tuple[str, dict[str, Any]]] = []

    class FakeStore:
        def save(self, _task: IngestionTask) -> None:
            return None

    class FakeSessionContext:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *_args: object) -> None:
            return None

    class FakeSessionFactory:
        def __call__(self) -> FakeSessionContext:
            return FakeSessionContext()

    factory_attempts = 0

    def get_factory_after_initialization() -> FakeSessionFactory:
        nonlocal factory_attempts
        factory_attempts += 1
        if factory_attempts == 1:
            raise RuntimeError("database is not initialized")
        return FakeSessionFactory()

    async def fake_init_db() -> None:
        initialized.append(True)

    async def fake_update_ingestion_task(
        _session: object,
        task_id: str,
        **values: Any,
    ) -> None:
        db_updates.append((task_id, values))

    task = _task("doc_worker_database_init_001")
    task.start_stage(IngestionStage.READY)
    task.complete_stage(IngestionStage.READY, chunk_count=1)

    monkeypatch.setattr(worker_module, "build_task_store", lambda *_args, **_kwargs: FakeStore())
    monkeypatch.setattr(config_module, "get_settings", lambda: SimpleNamespace(app_mode="full"))
    monkeypatch.setattr(session_module, "get_session_factory", get_factory_after_initialization)
    monkeypatch.setattr(session_module, "init_db", fake_init_db)
    monkeypatch.setattr(db_module, "update_ingestion_task", fake_update_ingestion_task)

    celery_tasks._save_task_state(task.id, task.model_dump(mode="json"))

    assert initialized == [True]
    assert len(db_updates) == 1


def test_worker_async_runner_reuses_one_event_loop_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent async resources must not cross a closed asyncio.run loop."""
    import asyncio

    from app.services.ingestion import celery_tasks

    monkeypatch.setattr(celery_tasks, "_worker_loop", None, raising=False)
    monkeypatch.setattr(celery_tasks, "_worker_loop_pid", None, raising=False)

    async def current_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    first = celery_tasks._run_async(current_loop())
    second = celery_tasks._run_async(current_loop())

    assert first is second


@pytest.mark.asyncio
async def test_update_ingestion_task_allows_clearing_error_fields() -> None:
    """入库任务从失败重试到成功时，应能把 PostgreSQL 错误字段清成 NULL。"""
    from app.db import crud as db

    class FakeSession:
        def __init__(self) -> None:
            self.statements: list[Any] = []
            self.commits = 0

        async def execute(self, statement: Any) -> None:
            self.statements.append(statement)

        async def commit(self) -> None:
            self.commits += 1

    session = FakeSession()

    await db.update_ingestion_task(
        session,  # type: ignore[arg-type]
        "ing_clear_error_001",
        error_message=None,
        error_code=None,
    )

    assert session.commits == 1
    assert len(session.statements) == 1
    updated_keys = {column.key for column in session.statements[0]._values}
    assert {"error_message", "error_code"}.issubset(updated_keys)


@pytest.mark.asyncio
async def test_document_upload_sync_path_updates_postgres_task_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """full + sync 入库完成后，API 状态端点优先读取的 PostgreSQL 快照必须更新。"""
    from app.api import documents
    from app.db import crud as db_module
    from app.db import session as session_module

    db_updates: list[tuple[str, dict[str, Any]]] = []

    class FakeSessionContext:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *_args: object) -> None:
            return None

    class FakeSessionFactory:
        def __call__(self) -> FakeSessionContext:
            return FakeSessionContext()

    async def fake_update_ingestion_task(
        _session: object,
        task_id: str,
        **values: Any,
    ) -> None:
        db_updates.append((task_id, values))

    task = _task("doc_full_sync_001")
    task.start_stage(IngestionStage.READY)
    task.complete_stage(IngestionStage.READY, chunk_count=3)

    monkeypatch.setattr(session_module, "get_session_factory", lambda: FakeSessionFactory())
    monkeypatch.setattr(db_module, "update_ingestion_task", fake_update_ingestion_task)

    await documents._sync_persisted_task_snapshot(task)

    assert db_updates == [
        (
            task.id,
            {
                "current_stage": IngestionStage.READY.value,
                "progress": 1.0,
                "total_chunks": 3,
                "error_message": None,
                "error_code": None,
                "stages_json": [stage.model_dump(mode="json") for stage in task.stages],
            },
        )
    ]
