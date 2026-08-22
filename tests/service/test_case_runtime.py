"""长期 HRCase 与运行时治理 service 测试。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.exceptions import ValidationError
from app.schemas.enums import CaseStatus
from app.schemas.runtime import ExecutionManifest
from app.services.runtime.case_service import CaseService
from app.services.runtime.clock import FakeClock
from app.services.runtime.event_store import InMemoryEventStore
from app.services.runtime.lease import InMemoryLeaseStore
from app.services.runtime.timer_coordinator import TimerCoordinator
from app.services.runtime.timers import InMemoryTimerStore


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


@pytest.mark.asyncio
async def test_create_case_records_manifest_and_open_projection() -> None:
    """创建 Case 应产生带执行清单的可查询 projection。"""
    service = CaseService(event_store=InMemoryEventStore())

    case = await service.create_case(
        title="新员工入职到转正",
        tenant_id="tenant_a",
        subject_user_id="user_employee",
        actor_id="user_hr",
        command_id="cmd_case_create",
        execution_manifest=_manifest(),
    )

    assert case.id.startswith("case_")
    assert case.status == CaseStatus.OPEN
    assert case.version == 1
    assert case.execution_manifest.policy_version == "hr-policy-2026-01"


@pytest.mark.asyncio
async def test_case_message_updates_working_memory_and_rebuilds_equally() -> None:
    """跨轮信息应进入 working memory，且事件重放结果一致。"""
    store = InMemoryEventStore()
    service = CaseService(event_store=store)
    case = await service.create_case(
        title="新员工入职到转正",
        tenant_id="tenant_a",
        subject_user_id="user_employee",
        actor_id="user_hr",
        command_id="cmd_case_create",
        execution_manifest=_manifest(),
    )

    updated = await service.add_message(
        case_id=case.id,
        message="员工已提交身份证明和学历材料",
        actor_id="user_employee",
        command_id="cmd_message_001",
        expected_version=1,
    )
    rebuilt = await service.rebuild(case.id)

    assert updated.version == 2
    assert updated.working_memory["messages"][-1]["content"] == "员工已提交身份证明和学历材料"
    assert rebuilt.model_dump(mode="json") == updated.model_dump(mode="json")


@pytest.mark.asyncio
async def test_run_lease_blocks_competitor_until_fake_clock_expires() -> None:
    """未过期 lease 应阻止并发 owner，过期后允许接管。"""
    clock = FakeClock(datetime(2026, 7, 13, tzinfo=UTC))
    leases = InMemoryLeaseStore(clock=clock)
    first = await leases.acquire("case_001", "worker_a", ttl_seconds=30)

    with pytest.raises(ValidationError, match="already leased"):
        await leases.acquire("case_001", "worker_b", ttl_seconds=30)

    clock.advance(seconds=31)
    second = await leases.acquire("case_001", "worker_b", ttl_seconds=30)
    assert second.fencing_token == first.fencing_token + 1


@pytest.mark.asyncio
async def test_timer_wakes_case_after_cross_day_restart_safe_event() -> None:
    """到期 timer 应通过事件唤醒 Case，并保持 replay 一致。"""
    clock = FakeClock(datetime(2026, 7, 13, tzinfo=UTC))
    event_store = InMemoryEventStore()
    cases = CaseService(event_store=event_store)
    timers = InMemoryTimerStore(clock=clock)
    coordinator = TimerCoordinator(case_service=cases, timer_store=timers)
    case = await cases.create_case(
        title="新员工入职到转正",
        tenant_id="tenant_a",
        subject_user_id="user_employee",
        actor_id="user_hr",
        command_id="cmd_case_create",
        execution_manifest=_manifest(),
    )

    scheduled = await coordinator.schedule(
        case_id=case.id,
        timer_type="probation.review_due",
        due_at=datetime(2026, 7, 14, tzinfo=UTC),
        payload={"employee_id": "user_employee"},
        actor_id="user_hr",
        command_id="cmd_schedule_review",
        expected_version=1,
    )
    assert scheduled.case.status == CaseStatus.WAITING_TIMER

    clock.advance(seconds=86_401)
    fired = await coordinator.fire_due(owner_id="scheduler_a", limit=10)

    assert len(fired) == 1
    assert fired[0].status.value == "fired"
    assert (await cases.get_case(case.id)).status == CaseStatus.OPEN
    assert (await cases.rebuild(case.id)).model_dump(mode="json") == (
        await cases.get_case(case.id)
    ).model_dump(mode="json")
