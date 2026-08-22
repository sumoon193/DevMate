"""入职到转正 Reference Application 的 Case 工作流测试。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.schemas.enums import ApprovalDecisionType, CaseStatus, ToolCallStatus
from app.schemas.runtime import ExecutionManifest
from app.schemas.user import UserContext
from app.services.a2a.policy_research import InProcessA2AClient, PolicyResearchA2AAgent
from app.services.agent.approval_manager import ApprovalManager
from app.services.agent.step_logger import StepLogger
from app.services.agent.tool_executor import ToolExecutor
from app.services.agent.tool_registry import ToolRegistry
from app.services.context.compactor import ContextCompactor
from app.services.mcp.adapter import McpApprovalBridge, McpToolAdapter, McpToolDiscovery
from app.services.mcp.fake_server import FakeMcpServer
from app.services.memory.store import InMemoryEpisodicMemoryStore
from app.services.runtime.case_service import CaseService
from app.services.runtime.clock import FakeClock
from app.services.runtime.event_store import InMemoryEventStore
from app.services.runtime.onboarding_workflow import OnboardingCaseWorkflow
from app.services.runtime.side_effects import InMemorySideEffectLedger
from app.services.runtime.timer_coordinator import TimerCoordinator
from app.services.runtime.timers import InMemoryTimerStore
from app.services.skills.registry import SkillRegistry


def _manifest() -> ExecutionManifest:
    return ExecutionManifest(
        model_provider="fake",
        model_name="deterministic-answer",
        model_version="1",
        prompt_version="answer-v1",
        skill_versions={"hr_onboarding": "1.0.0"},
        tool_schema_versions={"create_hr_ticket": "1"},
        policy_version="hr-policy-2026-01",
        retrieval_version="hybrid-v1",
        context_strategy_version="context-v1",
        code_version="test",
    )


def _user() -> UserContext:
    return UserContext(
        user_id="user_hr",
        tenant_id="tenant_a",
        department_ids=["dept_hr"],
        role="hr",
        permissions=["hr.document.read", "hr.ticket.write"],
    )


def _workflow(
    *,
    clock: FakeClock,
    events: InMemoryEventStore | None = None,
) -> tuple[
    OnboardingCaseWorkflow,
    CaseService,
    InMemoryEventStore,
    ApprovalManager,
    FakeMcpServer,
]:
    events = events or InMemoryEventStore()
    cases = CaseService(event_store=events)
    steps = StepLogger()
    approvals = ApprovalManager(steps, clock=clock)
    ledger = InMemorySideEffectLedger(clock=clock)
    registry = ToolRegistry()
    fake_mcp = FakeMcpServer()
    executor = ToolExecutor(
        registry=registry,
        approval_manager=approvals,
        step_logger=steps,
        side_effect_ledger=ledger,
    )
    mcp = McpToolAdapter(McpToolDiscovery(fake_mcp), registry, executor)
    mcp.register_discovered_tools()
    skills = SkillRegistry(
        allowed_source_prefixes=["repo://skills/"],
        activation_threshold=0.9,
        clock=clock,
    )
    draft = skills.register(
        name="hr_onboarding",
        version="1.0.0",
        content="先研究制度，再生成计划，写操作必须审批。",
        source_uri="repo://skills/hr_onboarding/1.0.0",
        allowed_tools=["create_hr_ticket"],
        required_permissions=["hr.document.read", "hr.ticket.write"],
    )
    skills.activate(draft.id, eval_score=0.98)
    timers = InMemoryTimerStore(clock=clock)
    workflow = OnboardingCaseWorkflow(
        case_service=cases,
        event_store=events,
        skill_registry=skills,
        memory_store=InMemoryEpisodicMemoryStore(clock=clock),
        context_compactor=ContextCompactor(),
        a2a_client=InProcessA2AClient(PolicyResearchA2AAgent()),
        mcp_adapter=mcp,
        mcp_approval_bridge=McpApprovalBridge(executor, approvals),
        approval_manager=approvals,
        timer_coordinator=TimerCoordinator(case_service=cases, timer_store=timers),
        clock=clock,
    )
    return workflow, cases, events, approvals, fake_mcp


@pytest.mark.asyncio
async def test_onboarding_case_waits_for_bound_approval_before_write() -> None:
    """标准 Case 应先形成证据和计划，再在写工具前暂停。"""
    clock = FakeClock(datetime(2026, 7, 13, tzinfo=UTC))
    workflow, cases, events, approvals, fake_mcp = _workflow(clock=clock)
    case = await cases.create_case(
        title="新员工入职到转正",
        tenant_id="tenant_a",
        subject_user_id="user_employee",
        actor_id="user_hr",
        command_id="cmd_case_create",
        execution_manifest=_manifest(),
    )

    started = await workflow.start(
        case_id=case.id,
        user_context=_user(),
        expected_version=case.version,
        command_id="cmd_workflow_start",
    )

    assert started.status == CaseStatus.WAITING_APPROVAL
    assert started.active_run_id is not None
    assert started.working_memory["plan"]["steps"][0]["owner"] == "HR"
    assert started.working_memory["evidence"][0]["document_version"] == "v1"
    approval_id = started.working_memory["approvals"][-1]["approval_id"]
    approval = approvals.get_request(approval_id)
    assert approval.policy_version == "hr-policy-2026-01"
    assert approval.execution_manifest_hash
    assert approval.evidence[0]["document_version"] == "v1"
    assert fake_mcp.call_count("create_hr_ticket") == 0
    assert [event.event_type for event in await events.load_stream(case.id)] == [
        "case.created",
        "run.started",
        "skill.loaded",
        "a2a.task.completed",
        "evidence.retrieved",
        "plan.created",
        "tool.call_prepared",
        "approval.requested",
        "context.compacted",
    ]


@pytest.mark.asyncio
async def test_approved_case_executes_once_and_schedules_probation_timer() -> None:
    """跨天审批恢复只能创建一次工单，并持久调度转正提醒。"""
    clock = FakeClock(datetime(2026, 7, 13, tzinfo=UTC))
    workflow, cases, _, _, fake_mcp = _workflow(clock=clock)
    case = await cases.create_case(
        title="新员工入职到转正",
        tenant_id="tenant_a",
        subject_user_id="user_employee",
        actor_id="user_hr",
        command_id="cmd_case_create",
        execution_manifest=_manifest(),
    )
    started = await workflow.start(
        case_id=case.id,
        user_context=_user(),
        expected_version=case.version,
        command_id="cmd_workflow_start",
    )
    approval_id = started.working_memory["approvals"][-1]["approval_id"]

    resumed = await workflow.decide_approval(
        case_id=case.id,
        approval_id=approval_id,
        decision=ApprovalDecisionType.APPROVE,
        decided_by="user_manager",
        user_context=_user(),
        expected_version=started.version,
        command_id="cmd_approve_ticket",
    )
    replayed = await workflow.decide_approval(
        case_id=case.id,
        approval_id=approval_id,
        decision=ApprovalDecisionType.APPROVE,
        decided_by="user_manager",
        user_context=_user(),
        expected_version=started.version,
        command_id="cmd_approve_ticket",
    )

    assert resumed.status == CaseStatus.WAITING_TIMER
    assert replayed.version == resumed.version
    assert resumed.working_memory["tool_results"][-1]["status"] == ToolCallStatus.COMPLETED
    assert resumed.working_memory["memories"][-1]["memory_key"] == "onboarding.ticket_created"
    assert resumed.working_memory["timers"][-1]["timer_type"] == "probation.review_due"
    assert fake_mcp.call_count("create_hr_ticket") == 1


@pytest.mark.asyncio
async def test_approval_rehydrates_from_event_store_after_service_restart() -> None:
    """第二天审批时应从持久事件恢复完整授权对象并继续执行。"""
    clock = FakeClock(datetime(2026, 7, 13, tzinfo=UTC))
    workflow, cases, events, _, _ = _workflow(clock=clock)
    case = await cases.create_case(
        title="新员工入职到转正",
        tenant_id="tenant_a",
        subject_user_id="user_employee",
        actor_id="user_hr",
        command_id="cmd_case_create",
        execution_manifest=_manifest(),
    )
    started = await workflow.start(
        case_id=case.id,
        user_context=_user(),
        expected_version=case.version,
        command_id="cmd_workflow_start",
    )
    approval_id = started.working_memory["approvals"][-1]["approval_id"]

    restarted_workflow, _, _, restarted_approvals, restarted_mcp = _workflow(
        clock=clock,
        events=events,
    )
    resumed = await restarted_workflow.decide_approval(
        case_id=case.id,
        approval_id=approval_id,
        decision=ApprovalDecisionType.APPROVE,
        decided_by="user_manager",
        user_context=_user(),
        expected_version=started.version,
        command_id="cmd_approval_after_restart",
    )

    restored = restarted_approvals.get_request(approval_id)
    assert restored.evidence[0]["document_version"] == "v1"
    assert resumed.status == CaseStatus.WAITING_TIMER
    assert restarted_mcp.call_count("create_hr_ticket") == 1


@pytest.mark.asyncio
async def test_approval_command_resumes_after_decision_event_crash() -> None:
    """审批事件落库后进程崩溃，重试应补齐工具、记忆与 timer。"""
    clock = FakeClock(datetime(2026, 7, 13, tzinfo=UTC))
    workflow, cases, events, approvals, _ = _workflow(clock=clock)
    case = await cases.create_case(
        title="新员工入职到转正",
        tenant_id="tenant_a",
        subject_user_id="user_employee",
        actor_id="user_hr",
        command_id="cmd_case_create",
        execution_manifest=_manifest(),
    )
    started = await workflow.start(
        case_id=case.id,
        user_context=_user(),
        expected_version=case.version,
        command_id="cmd_workflow_start",
    )
    approval_id = started.working_memory["approvals"][-1]["approval_id"]
    effective = approvals.approve(approval_id, "user_manager")
    await cases.record_event(
        case_id=case.id,
        event_type="approval.decided",
        payload={
            "approval_id": approval_id,
            "effective_approval_id": effective.id,
            "decision": ApprovalDecisionType.APPROVE.value,
            "status": effective.status.value,
            "decided_by": "user_manager",
            "revision": effective.revision,
            "approval": effective.model_dump(mode="json"),
        },
        actor_id="user_manager",
        command_id="cmd_crash_recovery:approval",
        expected_version=started.version,
    )

    restarted, _, _, _, restarted_mcp = _workflow(clock=clock, events=events)
    resumed = await restarted.decide_approval(
        case_id=case.id,
        approval_id=approval_id,
        decision=ApprovalDecisionType.APPROVE,
        decided_by="user_manager",
        user_context=_user(),
        expected_version=started.version,
        command_id="cmd_crash_recovery",
    )

    stream = await events.load_stream(case.id)
    assert resumed.status == CaseStatus.WAITING_TIMER
    assert restarted_mcp.call_count("create_hr_ticket") == 1
    assert (
        len([event for event in stream if event.command_id == "cmd_crash_recovery:approval"]) == 1
    )
    assert len([event for event in stream if event.command_id == "cmd_crash_recovery:tool"]) == 1
    assert len([event for event in stream if event.command_id == "cmd_crash_recovery:memory"]) == 1
    assert len([event for event in stream if event.command_id == "cmd_crash_recovery:timer"]) == 1


@pytest.mark.asyncio
async def test_policy_update_refreshes_evidence_revises_plan_and_requests_new_approval() -> None:
    """制度更新应使旧证据失效，并通过只读 A2A 产生新计划和审批。"""
    clock = FakeClock(datetime(2026, 7, 13, tzinfo=UTC))
    workflow, cases, events, approvals, _ = _workflow(clock=clock)
    case = await cases.create_case(
        title="新员工入职到转正",
        tenant_id="tenant_a",
        subject_user_id="user_employee",
        actor_id="user_hr",
        command_id="cmd_case_create",
        execution_manifest=_manifest(),
    )
    started = await workflow.start(
        case_id=case.id,
        user_context=_user(),
        expected_version=case.version,
        command_id="cmd_workflow_start",
    )
    first_approval_id = started.working_memory["approvals"][-1]["approval_id"]
    waiting_timer = await workflow.decide_approval(
        case_id=case.id,
        approval_id=first_approval_id,
        decision=ApprovalDecisionType.APPROVE,
        decided_by="user_manager",
        user_context=_user(),
        expected_version=started.version,
        command_id="cmd_approve_initial",
    )

    refreshed = await workflow.refresh_policy(
        case_id=case.id,
        policy_version="v2",
        user_context=_user(),
        expected_version=waiting_timer.version,
        command_id="cmd_refresh_policy_v2",
    )

    assert refreshed.status == CaseStatus.WAITING_APPROVAL
    assert refreshed.policy_versions["hr_policy"] == "v2"
    assert refreshed.working_memory["evidence"][0]["document_version"] == "v2"
    assert refreshed.working_memory["plan"]["revision_reason"] == "policy_version_changed"
    second_approval_id = refreshed.working_memory["approvals"][-1]["approval_id"]
    assert second_approval_id != first_approval_id
    assert approvals.get_request(second_approval_id).policy_version == "v2"
    event_types = [event.event_type for event in await events.load_stream(case.id)]
    assert "policy.stale_detected" in event_types
    assert "plan.revised" in event_types


@pytest.mark.asyncio
async def test_workflow_resumes_missing_steps_after_partial_command_crash() -> None:
    """首个事件已提交后重试同一命令，应继续而不是冲突或提前返回。"""
    clock = FakeClock(datetime(2026, 7, 13, tzinfo=UTC))
    workflow, cases, events, _, _ = _workflow(clock=clock)
    case = await cases.create_case(
        title="新员工入职到转正",
        tenant_id="tenant_a",
        subject_user_id="user_employee",
        actor_id="user_hr",
        command_id="cmd_case_create",
        execution_manifest=_manifest(),
    )
    partial = await cases.record_event(
        case_id=case.id,
        event_type="run.started",
        payload={"run_id": "run_partial", "workflow": "onboarding_to_regularization"},
        actor_id="user_hr",
        command_id="cmd_partial_start:run",
        expected_version=case.version,
    )

    resumed = await workflow.start(
        case_id=case.id,
        user_context=_user(),
        expected_version=case.version,
        command_id="cmd_partial_start",
    )

    assert partial.version == 2
    assert resumed.status == CaseStatus.WAITING_APPROVAL
    assert resumed.active_run_id == "run_partial"
    stream = await events.load_stream(case.id)
    assert len([event for event in stream if event.command_id == "cmd_partial_start:run"]) == 1
    assert stream[-1].command_id == "cmd_partial_start:context"
