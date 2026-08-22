"""长期 Case、事件游标和 SSE API。"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.api.dependencies import ServiceContainer, get_container
from app.api.schemas import (
    CaseApprovalRequest,
    CaseCreateRequest,
    CaseEventPage,
    CaseMessageRequest,
    CasePolicyRefreshRequest,
    CaseResponse,
    CaseWorkflowStartRequest,
)
from app.schemas.runtime import ExecutionManifest
from app.schemas.user import UserContext

router = APIRouter(prefix="/cases", tags=["cases"])


def _execution_manifest(container: ServiceContainer) -> ExecutionManifest:
    """从当前配置构造可回放 ExecutionManifest。"""
    settings = container.settings
    return ExecutionManifest(
        model_provider="qwen" if settings.qwen_api_key else "fake",
        model_name=settings.qwen_chat_model if settings.qwen_api_key else "deterministic-answer",
        model_version="configured",
        prompt_version="grounded-answer-v1",
        skill_versions={"hr_onboarding": "1.0.0"},
        tool_schema_versions={"create_hr_ticket": "1"},
        policy_version="hr-policy-2026-01",
        retrieval_version="hybrid-rrf-rerank-v1",
        context_strategy_version="write-select-compress-isolate-v1",
        code_version="0.2.0",
    )


@router.post("", response_model=CaseResponse, status_code=201)
async def create_case(
    body: CaseCreateRequest,
    container: ServiceContainer = Depends(get_container),
) -> CaseResponse:
    """创建跨轮次、跨天业务 Case。"""
    return await container.case_service.create_case(
        title=body.title,
        tenant_id=body.tenant_id,
        subject_user_id=body.subject_user_id,
        actor_id=body.actor_id,
        command_id=body.command_id,
        execution_manifest=_execution_manifest(container),
    )


@router.get("", response_model=list[CaseResponse])
async def list_cases(
    limit: int = Query(default=100, ge=1, le=500),
    container: ServiceContainer = Depends(get_container),
) -> list[CaseResponse]:
    """读取 Case 运维队列 projection。"""
    return await container.case_service.list_cases(limit=limit)


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: str,
    container: ServiceContainer = Depends(get_container),
) -> CaseResponse:
    """查询 Case projection。"""
    return await container.case_service.get_case(case_id)


@router.post("/{case_id}/messages", response_model=CaseResponse)
async def add_case_message(
    case_id: str,
    body: CaseMessageRequest,
    container: ServiceContainer = Depends(get_container),
) -> CaseResponse:
    """通过 expected_version 追加 Case 消息。"""
    return await container.case_service.add_message(
        case_id=case_id,
        message=body.message,
        actor_id=body.actor_id,
        command_id=body.command_id,
        expected_version=body.expected_version,
    )


def _case_user(actor_id: str, tenant_id: str) -> UserContext:
    """构造 Reference Application 的确定性 HR 操作上下文。"""
    return UserContext(
        user_id=actor_id,
        tenant_id=tenant_id,
        department_ids=["dept_hr"],
        role="hr",
        permissions=["hr.document.read", "hr.ticket.write"],
    )


@router.post("/{case_id}/start", response_model=CaseResponse)
async def start_case_workflow(
    case_id: str,
    body: CaseWorkflowStartRequest,
    container: ServiceContainer = Depends(get_container),
) -> CaseResponse:
    """启动入职到转正标准 Case，并在写操作审批点暂停。"""
    owner_id = f"api_{uuid.uuid4().hex[:12]}"
    lease = await container.lease_store.acquire(
        case_id,
        owner_id,
        ttl_seconds=30,
    )
    try:
        case = await container.case_service.get_case(case_id)
        return await container.onboarding_workflow.start(
            case_id=case_id,
            user_context=_case_user(body.actor_id, case.tenant_id),
            expected_version=body.expected_version,
            command_id=body.command_id,
        )
    finally:
        await container.lease_store.release(
            case_id,
            owner_id,
            lease.fencing_token,
        )


@router.post("/{case_id}/approvals/{approval_id}", response_model=CaseResponse)
async def decide_case_approval(
    case_id: str,
    approval_id: str,
    body: CaseApprovalRequest,
    container: ServiceContainer = Depends(get_container),
) -> CaseResponse:
    """审批并恢复 Case checkpoint，批准后执行幂等写工具。"""
    owner_id = f"api_{uuid.uuid4().hex[:12]}"
    lease = await container.lease_store.acquire(
        case_id,
        owner_id,
        ttl_seconds=30,
    )
    try:
        case = await container.case_service.get_case(case_id)
        return await container.onboarding_workflow.decide_approval(
            case_id=case_id,
            approval_id=approval_id,
            decision=body.decision,
            decided_by=body.actor_id,
            user_context=_case_user(body.actor_id, case.tenant_id),
            expected_version=body.expected_version,
            command_id=body.command_id,
            edited_parameters=body.edited_parameters,
        )
    finally:
        await container.lease_store.release(
            case_id,
            owner_id,
            lease.fencing_token,
        )


@router.post("/{case_id}/policies/refresh", response_model=CaseResponse)
async def refresh_case_policy(
    case_id: str,
    body: CasePolicyRefreshRequest,
    container: ServiceContainer = Depends(get_container),
) -> CaseResponse:
    """检测制度更新，重新研究 evidence、修订计划并生成新审批。"""
    owner_id = f"api_{uuid.uuid4().hex[:12]}"
    lease = await container.lease_store.acquire(
        case_id,
        owner_id,
        ttl_seconds=30,
    )
    try:
        case = await container.case_service.get_case(case_id)
        return await container.onboarding_workflow.refresh_policy(
            case_id=case_id,
            policy_version=body.policy_version,
            user_context=_case_user(body.actor_id, case.tenant_id),
            expected_version=body.expected_version,
            command_id=body.command_id,
        )
    finally:
        await container.lease_store.release(
            case_id,
            owner_id,
            lease.fencing_token,
        )


@router.get("/{case_id}/events", response_model=CaseEventPage)
async def get_case_events(
    case_id: str,
    after_sequence: int = Query(default=0, ge=0),
    container: ServiceContainer = Depends(get_container),
) -> CaseEventPage:
    """读取游标之后的持久事件。"""
    await container.case_service.get_case(case_id)
    stream = await container.event_store.load_stream(case_id)
    events = [event for event in stream if event.sequence > after_sequence]
    return CaseEventPage(
        case_id=case_id,
        after_sequence=after_sequence,
        items=[event.model_dump(mode="json") for event in events],
        next_sequence=(events[-1].sequence if events else after_sequence),
    )


@router.get("/{case_id}/stream")
async def stream_case_events(
    case_id: str,
    after_sequence: int = Query(default=0, ge=0),
    container: ServiceContainer = Depends(get_container),
) -> StreamingResponse:
    """按 Event Store sequence 输出可断线恢复的 SSE。"""
    await container.case_service.get_case(case_id)
    stream = await container.event_store.load_stream(case_id)
    events = [event for event in stream if event.sequence > after_sequence]

    async def generate() -> AsyncGenerator[str, None]:
        for event in events:
            data = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
            yield f"id: {event.sequence}\nevent: {event.event_type}\ndata: {data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
