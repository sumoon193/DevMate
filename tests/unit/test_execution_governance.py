"""审批、副作用账本与持久化定时器治理测试。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.exceptions import ValidationError
from app.schemas.enums import ApprovalStatus, SideEffectStatus, TimerStatus, ToolRiskLevel
from app.schemas.tool import ToolCall
from app.schemas.user import UserContext
from app.services.agent.approval_manager import ApprovalManager
from app.services.agent.step_logger import StepLogger
from app.services.runtime.clock import FakeClock
from app.services.runtime.side_effects import InMemorySideEffectLedger
from app.services.runtime.timers import InMemoryTimerStore


def _user() -> UserContext:
    return UserContext(
        user_id="user_hr",
        tenant_id="tenant_a",
        department_ids=["dept_hr"],
        role="hr",
        permissions=["hr.ticket.write"],
    )


def _tool_call() -> ToolCall:
    return ToolCall(
        id="tool_001",
        run_id="run_001",
        tool_name="create_hr_ticket",
        parameters={"title": "创建入职工单"},
        approval_required=True,
    )


def test_approval_expires_and_rejects_late_decision() -> None:
    """超过有效期的审批不能再批准。"""
    clock = FakeClock(datetime(2026, 7, 13, tzinfo=UTC))
    manager = ApprovalManager(StepLogger(), clock=clock, default_ttl_seconds=60)
    request = manager.create_request(
        run_id="run_001",
        tool_call=_tool_call(),
        tool_name="create_hr_ticket",
        parameters={"title": "创建入职工单"},
        risk_level=ToolRiskLevel.WRITE,
        user_context=_user(),
        evidence=[{"chunk_id": "chunk_v1", "document_version": "v1"}],
        policy_version="hr-policy-v1",
        execution_manifest_hash="manifest-v1",
    )

    clock.advance(seconds=61)
    with pytest.raises(ValidationError, match="expired"):
        manager.approve(request.id, "user_manager")
    assert manager.get_request(request.id).status == ApprovalStatus.EXPIRED


def test_approval_subject_hash_detects_parameter_drift() -> None:
    """审批后执行参数变化时 subject hash 校验必须失败。"""
    manager = ApprovalManager(StepLogger())
    request = manager.create_request(
        run_id="run_001",
        tool_call=_tool_call(),
        tool_name="create_hr_ticket",
        parameters={"title": "创建入职工单"},
        risk_level=ToolRiskLevel.WRITE,
        user_context=_user(),
        evidence=[{"chunk_id": "chunk_v1", "document_version": "v1"}],
        policy_version="hr-policy-v1",
        execution_manifest_hash="manifest-v1",
    )
    manager.approve(request.id, "user_manager")

    with pytest.raises(ValidationError, match="subject hash"):
        manager.validate_for_execution(
            request.id,
            tool_name="create_hr_ticket",
            parameters={"title": "被修改的工单"},
            evidence=[{"chunk_id": "chunk_v1", "document_version": "v1"}],
            policy_version="hr-policy-v1",
            execution_manifest_hash="manifest-v1",
        )


def test_admin_approval_requires_different_decider() -> None:
    """管理级工具必须执行 maker-checker 分离。"""
    manager = ApprovalManager(StepLogger())
    request = manager.create_request(
        run_id="run_001",
        tool_call=_tool_call(),
        tool_name="grant_admin_access",
        parameters={"user_id": "user_employee"},
        risk_level=ToolRiskLevel.ADMIN,
        user_context=_user(),
    )

    with pytest.raises(ValidationError, match="maker-checker"):
        manager.approve(request.id, "user_hr")


@pytest.mark.asyncio
async def test_side_effect_ledger_returns_cached_success_for_duplicate_key() -> None:
    """相同幂等键重放时应返回已成功的副作用结果。"""
    ledger = InMemorySideEffectLedger()
    reserved = await ledger.reserve(
        idempotency_key="effect_case_001_ticket",
        tool_name="create_hr_ticket",
        subject_hash="subject-v1",
    )
    await ledger.mark_succeeded(reserved.id, {"ticket_id": "HR-001"})

    replayed = await ledger.reserve(
        idempotency_key="effect_case_001_ticket",
        tool_name="create_hr_ticket",
        subject_hash="subject-v1",
    )

    assert replayed.status == SideEffectStatus.SUCCEEDED
    assert replayed.result == {"ticket_id": "HR-001"}
    assert len(await ledger.list_records()) == 1


@pytest.mark.asyncio
async def test_due_timer_can_only_be_claimed_once() -> None:
    """两个 scheduler 竞争时只有一个能 claim 到期 timer。"""
    clock = FakeClock(datetime(2026, 7, 13, tzinfo=UTC))
    timers = InMemoryTimerStore(clock=clock)
    timer = await timers.schedule(
        case_id="case_001",
        timer_type="probation.review_due",
        due_at=datetime(2026, 7, 14, tzinfo=UTC),
        payload={"employee_id": "user_employee"},
        idempotency_key="case_001:probation-review",
    )

    clock.advance(seconds=86_401)
    claimed_a = await timers.claim_due(owner_id="scheduler_a", limit=10)
    claimed_b = await timers.claim_due(owner_id="scheduler_b", limit=10)

    assert [item.id for item in claimed_a] == [timer.id]
    assert claimed_b == []
    await timers.mark_fired(timer.id, owner_id="scheduler_a")
    assert (await timers.get(timer.id)).status == TimerStatus.FIRED
