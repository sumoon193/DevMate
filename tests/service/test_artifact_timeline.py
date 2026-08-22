"""
Artifact Timeline 服务测试。

Timeline 是从 run artifacts 派生的复盘视图，不作为新的事实来源。
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.schemas.agent import AgentRunResponse, AgentStep
from app.schemas.approval import ApprovalRequest
from app.schemas.enums import ApprovalStatus, RunStatus, ToolCallStatus, ToolRiskLevel
from app.schemas.harness import TimelineEventType
from app.schemas.tool import ToolCall
from app.services.agent.artifact_timeline import ArtifactTimelineBuilder


def _step(node_name: str, output_data: dict) -> AgentStep:
    return AgentStep(
        id=f"step_{node_name}",
        run_id="run_timeline_001",
        node_name=node_name,
        input_data={"query": "入职流程"},
        output_data=output_data,
        evidence=[],
        token_usage={},
        duration_ms=1,
        created_at=datetime(2026, 7, 4, tzinfo=UTC),
    )


def test_timeline_builds_ordered_events_from_run_artifacts() -> None:
    """一次成功 run 应能串起 evidence、plan、approval、tool 和 answer。"""
    run = AgentRunResponse(
        id="run_timeline_001",
        user_id="user_001",
        thread_id="thread_001",
        original_query="新员工入职到转正要办哪些事项？",
        status=RunStatus.COMPLETED,
        steps=[],
        tool_calls=[
            ToolCall(
                id="tool_001",
                run_id="run_timeline_001",
                tool_name="create_hr_ticket",
                parameters={"title": "入职工单"},
                result={"ticket_id": "TK-001"},
                status=ToolCallStatus.COMPLETED,
                approval_required=True,
            )
        ],
        result={"answer": "已创建工单", "citations": [{"id": 1}]},
        created_at=datetime(2026, 7, 4, tzinfo=UTC),
        completed_at=datetime(2026, 7, 4, tzinfo=UTC),
    )
    steps = [
        _step("run_created", {"run_id": run.id}),
        _step("evidence_retrieved", {"citation_count": 2}),
        _step("plan_created", {"plan_id": "plan_001"}),
        _step("tool_approval_requested", {"approval_request_id": "appr_001"}),
        _step("approval_approved", {"approval_id": "appr_001"}),
        _step("tool_executed_after_approval", {"tool_call_id": "tool_001"}),
        _step("run_completed", {"result": run.result}),
    ]
    approvals = [
        ApprovalRequest(
            id="appr_001",
            run_id=run.id,
            tool_call_id="tool_001",
            tool_name="create_hr_ticket",
            parameters={"title": "入职工单"},
            expected_effect="创建 HR 工单",
            evidence=[{"citation_id": 1}],
            risk_level=ToolRiskLevel.WRITE,
            status=ApprovalStatus.APPROVED,
            decision=None,
            decided_by="admin",
            decided_at=datetime(2026, 7, 4, tzinfo=UTC),
        )
    ]

    timeline = ArtifactTimelineBuilder().build(run, steps, approvals)

    event_types = [event.event_type for event in timeline]
    assert event_types == [
        TimelineEventType.RUN_CREATED,
        TimelineEventType.EVIDENCE_RETRIEVED,
        TimelineEventType.PLAN_GENERATED,
        TimelineEventType.APPROVAL_REQUESTED,
        TimelineEventType.APPROVAL_DECIDED,
        TimelineEventType.TOOL_EXECUTED,
        TimelineEventType.ANSWER_GENERATED,
    ]
    assert timeline[3].approval_status == "pending"
    assert timeline[4].approval_status == "approved"
    assert timeline[5].risk_level == "write"


def test_timeline_redacts_pii_from_summaries() -> None:
    """Timeline 对外展示前必须脱敏摘要中的 PII。"""
    run = AgentRunResponse(
        id="run_timeline_002",
        user_id="user_001",
        thread_id="thread_001",
        original_query="手机号 13812345678",
        status=RunStatus.RUNNING,
        steps=[],
        tool_calls=[],
        result=None,
        created_at=datetime(2026, 7, 4, tzinfo=UTC),
        completed_at=None,
    )
    steps = [
        _step("run_created", {"message": "用户手机号 13812345678"}),
    ]

    timeline = ArtifactTimelineBuilder().build(run, steps, [])

    assert "13812345678" not in timeline[0].output_summary
    assert "***手机号***" in timeline[0].output_summary
