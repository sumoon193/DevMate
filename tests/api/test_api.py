"""
API 集成测试。

覆盖模块 13 规范要求的 6 个核心测试用例，加额外覆盖。
使用 FastAPI TestClient，不依赖真实网络。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import reset_container
from app.main import create_app


@pytest.fixture()
def app():
    """每次测试创建新 app 和新容器，保证状态隔离。"""
    from app.api.documents import reset_documents_store

    reset_container()
    reset_documents_store()
    application = create_app()
    yield application
    reset_container()
    reset_documents_store()


@pytest.fixture()
def client(app) -> TestClient:
    return TestClient(app)


def _create_run_and_get_pending_approval(
    client: TestClient,
    query: str = "新员工入职到转正要办哪些事项？",
) -> tuple[str, str]:
    """创建标准 demo Run，并返回待审批请求 ID。"""
    create_resp = client.post(
        "/agent-runs",
        json={
            "query": query,
            "user_id": "user_001",
        },
    )
    assert create_resp.status_code == 201
    run_id = create_resp.json()["id"]

    detail_resp = client.get(f"/agent-runs/{run_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    pending = [approval for approval in detail["approvals"] if approval["status"] == "pending"]
    assert len(pending) == 1
    return run_id, pending[0]["id"]


# ── 1. test_create_agent_run_returns_run_id ─────────────────────────


def test_create_agent_run_returns_run_id(client: TestClient) -> None:
    """创建 Agent Run 应返回有效的 run_id。"""
    resp = client.post(
        "/agent-runs",
        json={
            "query": "入职需要哪些材料？",
            "user_id": "user_001",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"].startswith("run_")
    assert data["thread_id"].startswith("thread_")
    assert data["status"] == "awaiting_approval"


# ── 2. test_get_agent_run_returns_steps_evidence_approvals ──────────


def test_get_agent_run_returns_steps_evidence_approvals(client: TestClient) -> None:
    """查询 Agent Run 应返回 steps、tool_calls 和 approvals 字段。"""
    # 先创建一个 Run
    create_resp = client.post(
        "/agent-runs",
        json={
            "query": "试用期多久？",
            "user_id": "user_001",
        },
    )
    run_id = create_resp.json()["id"]

    # 查询详情
    resp = client.get(f"/agent-runs/{run_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["id"] == run_id
    assert "steps" in data
    assert isinstance(data["steps"], list)
    assert "tool_calls" in data
    assert isinstance(data["tool_calls"], list)
    assert "approvals" in data
    assert isinstance(data["approvals"], list)
    assert data["status"] == "awaiting_approval"
    assert len(data["steps"]) >= 5
    assert [approval["status"] for approval in data["approvals"]] == ["pending"]
    assert data["result"]["approval_required"] is True
    assert len(data["result"]["citations"]) >= 2
    assert data["result"]["plan"]["steps"] == [
        "policy_search",
        "hr_checklist",
        "create_hr_ticket",
    ]


# ── 3. test_submit_approval_resumes_run ─────────────────────────────


def test_submit_approval_resumes_run(client: TestClient) -> None:
    """提交审批后应更新 approval 状态。"""
    run_id, approval_id = _create_run_and_get_pending_approval(client, "帮我提交请假工单")

    # 提交审批（通过）
    resp = client.post(
        f"/agent-runs/{run_id}/approvals/{approval_id}",
        json={"decision": "approve"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["approval_id"] == approval_id
    assert data["status"] == "approved"
    assert data["decision"] == "approve"

    detail_resp = client.get(f"/agent-runs/{run_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["status"] == "completed"
    assert detail["result"]["approval_required"] is False
    assert detail["result"]["tool_result"]["ticket_id"].startswith("TK-")


def test_get_agent_run_returns_decided_approvals(client: TestClient) -> None:
    """Agent Run 详情应保留已审批记录，便于审计回放。"""
    run_id, approval_id = _create_run_and_get_pending_approval(client, "帮我提交请假工单")

    resp = client.post(
        f"/agent-runs/{run_id}/approvals/{approval_id}",
        json={"decision": "approve"},
    )
    assert resp.status_code == 200

    detail_resp = client.get(f"/agent-runs/{run_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert [approval["id"] for approval in detail["approvals"]] == [approval_id]
    assert detail["approvals"][0]["status"] == "approved"


# ── 4. test_sse_stream_emits_ordered_events ─────────────────────────


def test_sse_stream_emits_ordered_events(client: TestClient) -> None:
    """SSE 流应返回 text/event-stream 格式，包含有序事件。"""
    # 创建 Run
    create_resp = client.post(
        "/agent-runs",
        json={
            "query": "报销流程是什么？",
            "user_id": "user_001",
        },
    )
    run_id = create_resp.json()["id"]

    # 获取 SSE 流
    with client.stream("GET", f"/agent-runs/{run_id}/stream") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        body = ""
        for chunk in resp.iter_text():
            body += chunk

    # 应包含 run_started 和 run_status 事件
    assert "run_started" in body
    assert "run_status" in body


# ── 5. test_api_error_shape_is_stable ───────────────────────────────


def test_api_error_shape_is_stable(client: TestClient) -> None:
    """API 错误响应必须返回统一格式 { error: { code, message, request_id } }。"""
    # 访问不存在的 Run
    resp = client.get("/agent-runs/run_nonexistent")
    assert resp.status_code == 404

    data = resp.json()
    assert "error" in data
    error = data["error"]
    assert "code" in error
    assert "message" in error
    assert "request_id" in error
    assert "details" in error
    assert error["code"] == "not_found"


# ── 6. test_route_does_not_expose_internal_exception ────────────────


def test_route_does_not_expose_internal_exception(client: TestClient) -> None:
    """错误响应不得包含内部堆栈信息。"""
    resp = client.get("/agent-runs/run_nonexistent")
    data = resp.json()

    error_str = str(data).lower()
    # 不应包含 Python 堆栈关键词
    assert "traceback" not in error_str
    assert 'file "' not in error_str
    assert ".py" not in error_str


# ── 额外覆盖 ────────────────────────────────────────────────────────


def test_health_endpoint(client: TestClient) -> None:
    """健康检查端点应返回 ok。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_create_document_returns_id(client: TestClient) -> None:
    """文档上传应返回文档 ID。"""
    content = b"# Onboarding\n\nWelcome to the company."
    resp = client.post(
        "/documents",
        files={"file": ("onboarding.md", content, "text/markdown")},
        data={"tenant_id": "tenant_001", "department_id": "dept_hr"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"].startswith("doc_")
    assert data["document_version"].startswith("docver_")
    assert data["task_id"].startswith("ing_")
    assert data["status"] == "ready"


def test_create_document_returns_queued_when_celery_mode_enabled(monkeypatch) -> None:
    """启用 Celery 入库时，上传接口应快速返回 queued。"""
    from app.api.documents import reset_documents_store
    from app.config import get_settings
    from app.services.ingestion import dispatcher as dispatcher_module

    enqueued: list[dict] = []
    monkeypatch.setenv("INGESTION_EXECUTION_MODE", "celery")
    monkeypatch.setenv("INGESTION_TASK_STORE", "memory")
    monkeypatch.setattr(
        dispatcher_module,
        "_default_celery_enqueue",
        lambda payload: enqueued.append(payload),
    )
    get_settings.cache_clear()
    reset_container()
    reset_documents_store()

    try:
        client = TestClient(create_app())
        content = "# Celery Upload\n\n异步入库测试。".encode()
        resp = client.post(
            "/documents",
            files={"file": ("celery-upload.md", content, "text/markdown")},
            data={"tenant_id": "tenant_001", "department_id": "dept_hr"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "queued"
        assert len(enqueued) == 1

        status_resp = client.get(f"/ingestions/{data['task_id']}")
        assert status_resp.status_code == 200
        status = status_resp.json()
        assert status["status"] == "queued"
        assert status["progress"] == 0.0
    finally:
        get_settings.cache_clear()
        reset_container()
        reset_documents_store()


def test_ingestion_status_returns_task_info(client: TestClient) -> None:
    """查询入库任务应返回任务状态。"""
    content = b"# Reimbursement\n\nReimbursement policy."
    doc_resp = client.post(
        "/documents",
        files={"file": ("reimbursement.md", content, "text/markdown")},
        data={"tenant_id": "tenant_001", "department_id": "dept_hr"},
    )
    doc_data = doc_resp.json()

    task_id = doc_data["task_id"]

    resp = client.get(f"/ingestions/{task_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == task_id
    assert data["document_id"] == doc_data["id"]
    assert data["status"] == "ready"


def test_get_nonexistent_run_returns_404(client: TestClient) -> None:
    """查询不存在的 Run 应返回 404。"""
    resp = client.get("/agent-runs/run_notexist")
    assert resp.status_code == 404


def test_approval_with_edit_decision(client: TestClient) -> None:
    """审批决策为 edit 时应更新参数。"""
    run_id, approval_id = _create_run_and_get_pending_approval(client, "帮我提交工单")

    resp = client.post(
        f"/agent-runs/{run_id}/approvals/{approval_id}",
        json={
            "decision": "edit",
            "edited_parameters": {
                "title": "修改后标题",
                "description": "调整后的工单说明",
                "priority": "high",
                "category": "转正",
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "approved"
    assert data["decision"] == "edit"

    detail_resp = client.get(f"/agent-runs/{run_id}")
    assert detail_resp.status_code == 200
    tool_result = detail_resp.json()["result"]["tool_result"]
    assert tool_result["title"] == "修改后标题"
    assert tool_result["priority"] == "high"
    assert tool_result["category"] == "转正"


def test_request_id_in_response_headers(client: TestClient) -> None:
    """所有响应应包含 X-Request-ID 头。"""
    resp = client.get("/health")
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) > 0


def test_langgraph_agent_run_engine_can_resume_approval(monkeypatch) -> None:
    """AGENT_RUN_ENGINE=langgraph 时，API 应走真实 LangGraph 编排并可审批恢复。"""
    from app.api.documents import reset_documents_store
    from app.config import get_settings

    monkeypatch.setenv("APP_MODE", "fallback")
    monkeypatch.setenv("AGENT_RUN_ENGINE", "langgraph")
    get_settings.cache_clear()
    reset_container()
    reset_documents_store()

    try:
        graph_client = TestClient(create_app())
        create_resp = graph_client.post(
            "/agent-runs",
            json={"query": "帮我创建入职工单", "user_id": "user_001"},
        )
        assert create_resp.status_code == 201
        created = create_resp.json()
        assert created["status"] == "awaiting_approval"

        detail_resp = graph_client.get(f"/agent-runs/{created['id']}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["result"] is None
        pending = [approval for approval in detail["approvals"] if approval["status"] == "pending"]
        assert len(pending) == 1

        approve_resp = graph_client.post(
            f"/agent-runs/{created['id']}/approvals/{pending[0]['id']}",
            json={"decision": "approve"},
        )
        assert approve_resp.status_code == 200
        assert approve_resp.json()["status"] == "approved"

        completed_resp = graph_client.get(f"/agent-runs/{created['id']}")
        assert completed_resp.status_code == 200
        completed = completed_resp.json()
        assert completed["status"] == "completed"
        assert completed["tool_calls"][0]["status"] == "completed"
        assert completed["result"]["answer"]
    finally:
        get_settings.cache_clear()
        reset_container()
        reset_documents_store()


def test_demo_run_completes_automatically_when_policy_mode(monkeypatch) -> None:
    """APPROVAL_MODE=policy 时，demo 链路自动批准写入工具，Run 全程无人值守完成。"""
    from app.api.documents import reset_documents_store
    from app.config import get_settings

    monkeypatch.setenv("APP_MODE", "fallback")
    monkeypatch.setenv("AGENT_RUN_ENGINE", "demo")
    monkeypatch.setenv("APPROVAL_MODE", "policy")
    get_settings.cache_clear()
    reset_container()
    reset_documents_store()

    try:
        policy_client = TestClient(create_app())
        create_resp = policy_client.post(
            "/agent-runs",
            json={"query": "帮我创建入职工单", "user_id": "user_001"},
        )
        assert create_resp.status_code == 201
        created = create_resp.json()
        assert created["status"] == "completed"

        detail_resp = policy_client.get(f"/agent-runs/{created['id']}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["result"]["approval_required"] is False
        assert detail["result"]["tool_result"]["ticket_id"].startswith("TK-")

        # 审批记录仍保留：decided_by=policy_engine，可审计回放
        approvals = detail["approvals"]
        assert len(approvals) == 1
        assert approvals[0]["status"] == "approved"
        assert approvals[0]["decided_by"] == "policy_engine"
    finally:
        get_settings.cache_clear()
        reset_container()
        reset_documents_store()
