"""
集成测试：验证所有真实外部服务适配器。

每个测试直接与 Docker 容器中的服务交互。
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio  # noqa: F401 — 保证 pytest_asyncio fixture 注册

from app.schemas.chunk import ChunkCreate
from app.schemas.enums import RunStatus, Visibility
from app.schemas.user import UserContext
from app.services.retrieval.store.base import ACLFilter

# ── PostgreSQL ────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.integration
async def test_postgres_save_and_get_run(db_session):
    """验证 PostgreSQL 能保存和读取 AgentRun。"""
    from app.db import crud as db
    from app.schemas.enums import RunStatus

    run_id = f"run_{uuid.uuid4().hex[:8]}"

    await db.save_run(
        db_session,
        run_id=run_id,
        user_id="user_test",
        thread_id="thread_test",
        original_query="测试查询",
        status=RunStatus.CREATED,
    )

    loaded = await db.get_run(db_session, run_id)
    assert loaded is not None
    assert loaded.id == run_id
    assert loaded.original_query == "测试查询"
    assert loaded.status == RunStatus.CREATED


@pytest.mark.asyncio
@pytest.mark.integration
async def test_postgres_update_run_status(db_session):
    """验证 PostgreSQL 能更新 Run 状态。"""
    from app.db import crud as db

    run_id = f"run_{uuid.uuid4().hex[:8]}"

    await db.save_run(
        db_session,
        run_id=run_id,
        user_id="user_test",
        thread_id="thread_test",
        original_query="测试查询",
    )

    await db.update_run_status(db_session, run_id, RunStatus.COMPLETED, {"answer": "测试答案"})

    loaded = await db.get_run(db_session, run_id)
    assert loaded.status == RunStatus.COMPLETED
    assert loaded.result == {"answer": "测试答案"}
    assert loaded.completed_at is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_postgres_save_step(db_session):
    """验证 PostgreSQL 能保存 AgentStep。"""
    from app.db import crud as db

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    await db.save_run(db_session, run_id=run_id, user_id="u", thread_id="t", original_query="q")

    await db.save_step(
        db_session,
        run_id=run_id,
        node_name="intent",
        input_data={"query": "test"},
        output_data={"intent": "policy_question"},
    )

    steps = await db.get_steps(db_session, run_id)
    assert len(steps) == 1
    assert steps[0].node_name == "intent"


# ── MinIO ─────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_minio_put_and_get(minio_storage, unique_id):
    """验证 MinIO 能存储和读取对象。"""
    key = f"test/{unique_id}/hello.txt"
    data = "你好，企业智慧！".encode()

    minio_storage.put_object(key, data, content_type="text/plain")
    assert minio_storage.object_exists(key)

    loaded = minio_storage.get_object(key)
    assert loaded == data

    minio_storage.delete_object(key)
    assert not minio_storage.object_exists(key)


@pytest.mark.integration
def test_minio_object_not_found(minio_storage):
    """验证 MinIO 不存在的对象抛出 NotFoundError。"""
    from app.core.exceptions import NotFoundError

    with pytest.raises(NotFoundError):
        minio_storage.get_object("nonexistent/key.txt")


# ── Milvus ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.integration
async def test_milvus_add_and_search(milvus_store, unique_id):
    """验证 Milvus 能插入和检索向量。"""
    acl = ACLFilter(
        tenant_id="tenant_test",
        department_ids=["dept_001"],
        allowed_visibility=[Visibility.PUBLIC, Visibility.DEPARTMENT],
    )

    chunks = [
        ChunkCreate(
            document_id=f"doc_{unique_id}",
            chunk_text="新员工入职需提交身份证复印件",
            context_prefix="来自入职制度",
            full_text="",
            parent_id=None,
            chunk_type="child",
            heading_path="入职制度 > 材料清单",
            page_numbers=[1],
            token_count=20,
            tenant_id="tenant_test",
            department_id="dept_001",
            visibility=Visibility.DEPARTMENT,
            acl_metadata={},
        )
    ]
    # 使用固定的随机向量
    embeddings = [[0.1] * 1024]

    await milvus_store.add_chunks(chunks, embeddings)

    # 使用相同方向的向量搜索
    query_embedding = [0.1] * 1024
    results = await milvus_store.search(query_embedding, acl, top_k=5)

    assert len(results) >= 1, (
        f"Milvus search returned 0 results (collection has {len(chunks)} chunks)"
    )
    assert any(r.document_id == f"doc_{unique_id}" for r in results), (
        f"Expected doc_{unique_id} in results, got: {[r.document_id for r in results]}"
    )


# ── Elasticsearch ─────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.integration
async def test_es_add_and_search(es_store, unique_id):
    """验证 Elasticsearch 能索引和检索文档。"""
    await es_store.ensure_index()

    acl = ACLFilter(
        tenant_id="tenant_test",
        department_ids=["dept_001"],
        allowed_visibility=[Visibility.PUBLIC, Visibility.DEPARTMENT],
    )

    chunks = [
        ChunkCreate(
            document_id=f"doc_{unique_id}",
            chunk_text="员工转正需提交转正申请表和工作总结",
            context_prefix="来自转正制度",
            full_text="",
            parent_id=None,
            chunk_type="child",
            heading_path="转正制度 > 申请流程",
            page_numbers=[2],
            token_count=20,
            tenant_id="tenant_test",
            department_id="dept_001",
            visibility=Visibility.DEPARTMENT,
            acl_metadata={},
        )
    ]

    await es_store.add_chunks(chunks)

    results = await es_store.search("转正申请", acl, top_k=5)

    assert len(results) >= 1
    assert any(r.document_id == f"doc_{unique_id}" for r in results)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_es_delete_by_document(es_store, unique_id):
    """验证 Elasticsearch 能按文档 ID 删除。"""
    doc_id = f"doc_del_{unique_id}"

    await es_store.add_chunks(
        [
            ChunkCreate(
                document_id=doc_id,
                chunk_text="待删除的测试文档",
                context_prefix="",
                full_text="",
                parent_id=None,
                chunk_type="child",
                heading_path="",
                page_numbers=[1],
                token_count=5,
                tenant_id="tenant_test",
                department_id="dept_001",
                visibility=Visibility.DEPARTMENT,
                acl_metadata={},
            )
        ]
    )

    await es_store.delete_by_document(doc_id)

    acl = ACLFilter(
        tenant_id="tenant_test",
        department_ids=["dept_001"],
        allowed_visibility=[Visibility.PUBLIC, Visibility.DEPARTMENT],
    )
    results = await es_store.search("待删除的测试文档", acl, top_k=5)
    assert not any(r.document_id == doc_id for r in results)


# ── Redis ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.integration
async def test_redis_rate_limiter(redis_client):
    """验证 Redis 速率限制器。"""
    from app.services.security.redis_rate_limiter import RedisRateLimiter

    limiter = RedisRateLimiter(
        redis_url="redis://localhost:6379/0", max_requests=3, window_seconds=60
    )

    user_id = f"user_{uuid.uuid4().hex[:8]}"

    # 前 3 次应该通过
    assert await limiter.check(user_id, "test_action") is True
    assert await limiter.check(user_id, "test_action") is True
    assert await limiter.check(user_id, "test_action") is True

    # 第 4 次应该被限制
    assert await limiter.check(user_id, "test_action") is False

    remaining = await limiter.get_remaining(user_id, "test_action")
    assert remaining == 0

    # 重置后应该恢复
    await limiter.reset(user_id, "test_action")
    remaining = await limiter.get_remaining(user_id, "test_action")
    assert remaining == 3

    await limiter.close()


# ── 端到端：AgentRunManager + PostgreSQL ──────────────────────────────


@pytest.mark.asyncio
@pytest.mark.integration
async def test_run_manager_with_postgres(db_session, settings):
    """验证 AgentRunManager 使用 PostgreSQL 持久化。"""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.services.agent.approval_manager import ApprovalManager
    from app.services.agent.run_manager import AgentRunManager
    from app.services.agent.step_logger import StepLogger
    from app.services.agent.tool_executor import ToolExecutor
    from app.services.agent.tool_registry import ToolRegistry
    from app.services.security.acl_validator import ACLValidator

    url = settings.postgres_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(url, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    step_logger = StepLogger()
    approval_mgr = ApprovalManager(step_logger=step_logger)
    registry = ToolRegistry()
    acl = ACLValidator()
    tool_executor = ToolExecutor(
        registry=registry,
        approval_manager=approval_mgr,
        step_logger=step_logger,
        acl_validator=acl,
    )

    run_mgr = AgentRunManager(
        tool_executor=tool_executor,
        approval_manager=approval_mgr,
        step_logger=step_logger,
        session_factory=factory,
    )

    user = UserContext(
        user_id="user_int",
        tenant_id="tenant_hr",
        department_ids=["dept_001"],
        role="hr",
        permissions=[],
    )

    run = await run_mgr.create_run("入职需要什么材料？", user)
    assert run.status == RunStatus.CREATED

    from app.db import crud as db

    async with factory() as session:
        loaded = await db.get_run(session, run.id)
        assert loaded is not None
        assert loaded.original_query == "入职需要什么材料？"

    await run_mgr.start_run(run.id)
    await run_mgr.complete_run(run.id, {"answer": "测试答案"})

    async with factory() as session:
        loaded = await db.get_run(session, run.id)
        assert loaded.status == RunStatus.COMPLETED

    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_run_manager_persists_tool_call_before_approval_snapshot(db_session, settings):
    """审批快照落库前必须先落库对应 ToolCall，避免 PostgreSQL 外键失败。"""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.models.approval import ApprovalRequest as ApprovalORM
    from app.models.tool_call import ToolCall as ToolCallORM
    from app.schemas.approval import ApprovalDecision
    from app.schemas.enums import (
        ApprovalDecisionType,
        ApprovalStatus,
        ToolCallStatus,
        ToolRiskLevel,
    )
    from app.schemas.tool import ToolDefinition
    from app.services.agent.approval_manager import ApprovalManager
    from app.services.agent.run_manager import AgentRunManager
    from app.services.agent.step_logger import StepLogger
    from app.services.agent.tool_executor import ToolExecutor
    from app.services.agent.tool_registry import ToolRegistry
    from app.services.agent.tools.mock_ticket import MockTicketHandler
    from app.services.security.acl_validator import ACLValidator

    url = settings.postgres_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(url, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    step_logger = StepLogger()
    approval_mgr = ApprovalManager(step_logger=step_logger)
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="create_hr_ticket",
            description="创建模拟 HR 工单",
            permission_scope="hr.ticket.write",
            risk_level=ToolRiskLevel.WRITE,
            requires_approval=True,
            timeout_seconds=10,
            idempotent=True,
            parameters_schema={"title": {"type": "string"}, "description": {"type": "string"}},
        ),
        MockTicketHandler(),
    )
    tool_executor = ToolExecutor(
        registry=registry,
        approval_manager=approval_mgr,
        step_logger=step_logger,
        acl_validator=ACLValidator(),
    )
    run_mgr = AgentRunManager(
        tool_executor=tool_executor,
        approval_manager=approval_mgr,
        step_logger=step_logger,
        session_factory=factory,
    )
    user = UserContext(
        user_id="user_int",
        tenant_id="tenant_hr",
        department_ids=["dept_001"],
        role="hr",
        permissions=["hr.ticket.write"],
    )

    run = await run_mgr.create_run("帮我创建入职办理工单", user)
    await run_mgr.start_run(run.id)
    pending_call = await run_mgr.execute_tool(
        run_id=run.id,
        tool_name="create_hr_ticket",
        parameters={
            "title": "新员工入职办理",
            "description": "准备入职材料与系统账号",
            "priority": "medium",
            "category": "入职",
        },
        user_context=user,
    )
    approvals = await run_mgr.get_run_approvals(run.id)
    approval = approvals[0]

    await run_mgr.apply_approval_decision(
        run_id=run.id,
        approval_id=approval.id,
        approval_decision=ApprovalDecision(decision=ApprovalDecisionType.APPROVE),
        user_context=user,
    )
    executed_call = await run_mgr.execute_approved_tool(run.id, approval.id, user)
    await run_mgr.complete_run(run.id, {"answer": "已创建工单"})

    async with factory() as session:
        stored_tool_call = await session.scalar(
            select(ToolCallORM).where(ToolCallORM.id == pending_call.id)
        )
        stored_approval = await session.scalar(
            select(ApprovalORM).where(ApprovalORM.id == approval.id)
        )

    assert stored_tool_call is not None
    assert stored_tool_call.status == ToolCallStatus.COMPLETED
    assert stored_tool_call.result == executed_call.result
    assert stored_approval is not None
    assert stored_approval.tool_call_id == stored_tool_call.id
    assert stored_approval.status == ApprovalStatus.APPROVED
    assert stored_approval.decision == ApprovalDecisionType.APPROVE
    assert stored_approval.decided_by == user.user_id

    await engine.dispose()
