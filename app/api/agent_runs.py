"""
Agent Run 端点。

包括创建 Run、查询详情、SSE 事件流。
- full 模式：走真实 LangGraph 工作流（Milvus+ES 检索 → Qwen 生成）
- fallback 模式：走确定性 demo 链路
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import ServiceContainer, get_container
from app.api.schemas import AgentRunCreateRequest, AgentRunCreateResponse, AgentRunDetail
from app.schemas.agent import AgentPlan, AgentRunResponse
from app.schemas.chunk import Citation, EvidenceBundle
from app.schemas.user import UserContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent-runs", tags=["agent-runs"])


def _build_user_context(user_id: str) -> UserContext:
    """从 user_id 构建 UserContext（V1 简化实现）。"""
    return UserContext(
        user_id=user_id,
        tenant_id="tenant_001",
        department_ids=["dept_hr"],
        role="employee",
        permissions=[
            "hr.document.read",
            "hr.profile.read",
            "hr.checklist.read",
            "hr.chat.write",
            "hr.ticket.write",
        ],
    )


@router.post("", response_model=AgentRunCreateResponse, status_code=201)
async def create_agent_run(
    body: AgentRunCreateRequest,
    container: ServiceContainer = Depends(get_container),
) -> AgentRunCreateResponse:
    user_context = _build_user_context(body.user_id)

    if getattr(container.settings, "agent_run_engine", "demo") == "langgraph":
        run = await container.graph_runner.run_to_checkpoint(body.query, user_context)
    else:
        # V1 前端验收默认走确定性 demo 链路；显式设置 AGENT_RUN_ENGINE=langgraph
        # 时切到真实 LangGraph 编排。
        run = await container.run_manager.create_run(body.query, user_context)
        run = await _run_fallback_demo(container, run.id, body.query, user_context)
    return AgentRunCreateResponse(
        id=run.id,
        thread_id=run.thread_id,
        status=run.status,
    )


@router.get("/{run_id}", response_model=AgentRunDetail)
async def get_agent_run(
    run_id: str,
    container: ServiceContainer = Depends(get_container),
) -> AgentRunDetail:
    run = await container.run_manager.get_run(run_id)
    steps = await container.run_manager.get_run_steps(run_id)
    approvals = await container.run_manager.get_run_approvals(run_id)
    timeline = container.timeline_builder.build(run, steps, approvals)

    return AgentRunDetail(
        id=run.id,
        user_id=run.user_id,
        thread_id=run.thread_id,
        original_query=run.original_query,
        status=run.status,
        steps=[s.model_dump(mode="json") for s in steps],
        tool_calls=[tc.model_dump(mode="json") for tc in run.tool_calls],
        approvals=[a.model_dump(mode="json") for a in approvals],
        timeline=[event.model_dump(mode="json") for event in timeline],
        result=run.result,
        created_at=run.created_at,
        completed_at=run.completed_at,
    )


@router.get("/{run_id}/stream")
async def stream_agent_run(
    run_id: str,
    container: ServiceContainer = Depends(get_container),
) -> StreamingResponse:
    """SSE 事件流：查询 Run 当前状态并输出。"""
    from app.services.graph.sse import (
        SSEEventType,
        create_run_started_event,
        create_sse_event,
        create_step_completed_event,
    )

    run = await container.run_manager.get_run(run_id)
    steps = await container.run_manager.get_run_steps(run_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        yield create_run_started_event(run_id, run.original_query, run.thread_id)

        for step in steps:
            yield create_step_completed_event(
                run_id,
                step.node_name,
                step.output_data,
            )

        yield create_sse_event(
            SSEEventType.RUN_STATUS,
            {"status": run.status.value},
            run_id,
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _run_fallback_demo(
    container: ServiceContainer,
    run_id: str,
    query: str,
    user_context: UserContext,
) -> AgentRunResponse:
    """执行模块 11 前端演示需要的确定性 fallback 链路。"""
    run_manager = container.run_manager

    await run_manager.start_run(run_id)

    evidence = _build_demo_evidence()
    await run_manager.retrieve_evidence(run_id, evidence)

    plan = AgentPlan(
        id=f"plan_{run_id[-8:]}",
        run_id=run_id,
        steps=["policy_search", "hr_checklist", "create_hr_ticket"],
        current_step_index=0,
    )
    await run_manager.create_plan(run_id, plan)

    await run_manager.execute_tool(
        run_id=run_id,
        tool_name="policy_search",
        parameters={"query": query, "top_k": 2},
        user_context=user_context,
    )
    await run_manager.execute_tool(
        run_id=run_id,
        tool_name="hr_checklist",
        parameters={"scenario": "入职"},
        user_context=user_context,
    )
    await run_manager.execute_tool(
        run_id=run_id,
        tool_name="create_hr_ticket",
        parameters=_build_ticket_parameters(query),
        user_context=user_context,
    )

    # 策略化自动审批：命中规则时直接执行并完成，全程无人值守。
    auto_tool_call = await run_manager.auto_approve_and_execute(run_id, user_context)
    if auto_tool_call is not None:
        await run_manager.complete_run(
            run_id=run_id,
            result={
                "answer": (
                    "新员工入职到转正通常包括材料提交、合同签署、账号开通、"
                    "入职培训、试用期目标确认、转正评估和 HR 归档。"
                    "系统已按审批策略自动批准并创建模拟 HR 工单。"
                ),
                "citations": [
                    citation.model_dump(mode="json") for citation in evidence.evidence_list
                ],
                "confidence": evidence.query_coverage_score,
                "plan": {
                    "id": plan.id,
                    "steps": plan.steps,
                    "requires_approval": ["create_hr_ticket"],
                },
                "tool_result": auto_tool_call.result,
                "approval_required": False,
            },
        )
        return await run_manager.get_run(run_id)

    run = await run_manager.get_run(run_id)
    run.result = {
        "answer": (
            "新员工入职到转正通常包括材料提交、合同签署、账号开通、"
            "入职培训、试用期目标确认、转正评估和 HR 归档。"
            "系统已准备创建模拟 HR 工单，需人工审批后才会执行。"
        ),
        "citations": [citation.model_dump(mode="json") for citation in evidence.evidence_list],
        "confidence": evidence.query_coverage_score,
        "plan": {
            "id": plan.id,
            "steps": plan.steps,
            "requires_approval": ["create_hr_ticket"],
        },
        "tool_result": None,
        "approval_required": True,
    }
    return run


def _build_demo_evidence() -> EvidenceBundle:
    """构建前端标准 demo 的确定性证据。"""
    citations = [
        Citation(
            id=1,
            document_name="员工入职与转正管理制度",
            section="第二章 入职办理",
            page=3,
            chunk_text="新员工入职需提交身份证明、学历证明、离职证明，并在入职当天签署劳动合同。",
            score=0.92,
            rerank_score=0.95,
        ),
        Citation(
            id=2,
            document_name="员工入职与转正管理制度",
            section="第三章 试用期与转正",
            page=5,
            chunk_text="试用期内需完成目标确认、主管评估、转正申请和 HR 复核归档。",
            score=0.88,
            rerank_score=0.91,
        ),
    ]
    return EvidenceBundle(
        evidence_list=citations,
        total_count=len(citations),
        query_coverage_score=0.9,
    )


def _build_ticket_parameters(query: str) -> dict[str, Any]:
    """构建创建模拟 HR 工单的确定性参数。"""
    return {
        "title": "新员工入职到转正办理工单",
        "description": query,
        "priority": "medium",
        "category": "入职",
    }
