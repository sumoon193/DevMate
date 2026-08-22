"""
Agent Run Artifact Timeline。

Timeline 从 steps、tool calls、approvals 和 result 派生，只用于复盘视图。
"""

from __future__ import annotations

import json
from typing import Any

from app.schemas.agent import AgentRunResponse, AgentStep
from app.schemas.approval import ApprovalRequest
from app.schemas.harness import TimelineEvent, TimelineEventType
from app.schemas.tool import ToolCall
from app.services.security.pii_redactor import PIIRedactor


class ArtifactTimelineBuilder:
    """从 Agent Run artifacts 构建时间线。"""

    _NODE_EVENT_MAP: dict[str, TimelineEventType] = {
        "run_created": TimelineEventType.RUN_CREATED,
        "evidence_retrieved": TimelineEventType.EVIDENCE_RETRIEVED,
        "plan_created": TimelineEventType.PLAN_GENERATED,
        "tool_approval_requested": TimelineEventType.APPROVAL_REQUESTED,
        "approval_approved": TimelineEventType.APPROVAL_DECIDED,
        "approval_rejected": TimelineEventType.APPROVAL_DECIDED,
        "approval_edited": TimelineEventType.APPROVAL_DECIDED,
        "tool_executed": TimelineEventType.TOOL_EXECUTED,
        "tool_executed_after_approval": TimelineEventType.TOOL_EXECUTED,
        "reflection_created": TimelineEventType.REFLECTION_CREATED,
        "repair_action_created": TimelineEventType.REPAIR_ACTION_CREATED,
        "run_completed": TimelineEventType.ANSWER_GENERATED,
    }

    def __init__(self, pii_redactor: PIIRedactor | None = None) -> None:
        self._pii_redactor = pii_redactor or PIIRedactor()

    def build(
        self,
        run: AgentRunResponse,
        steps: list[AgentStep],
        approvals: list[ApprovalRequest],
    ) -> list[TimelineEvent]:
        """
        构建按 step 顺序排列的 timeline。

        Args:
            run: Agent Run 响应体
            steps: 已记录步骤
            approvals: 审批记录

        Returns:
            TimelineEvent 列表
        """
        approval_by_id = {approval.id: approval for approval in approvals}
        approval_by_tool_call = {approval.tool_call_id: approval for approval in approvals}
        tool_call_by_id = {
            tool_call_id: tool_call
            for tool_call in run.tool_calls
            if (tool_call_id := self._tool_call_id(tool_call)) is not None
        }

        events: list[TimelineEvent] = []
        for index, step in enumerate(steps, start=1):
            event_type = self._NODE_EVENT_MAP.get(step.node_name)
            if event_type is None:
                continue

            approval = self._find_approval(step, approval_by_id, approval_by_tool_call)
            tool_call = self._find_tool_call(step, tool_call_by_id)
            events.append(
                TimelineEvent(
                    id=f"timeline_{index:04d}",
                    run_id=run.id,
                    event_type=event_type,
                    timestamp=step.created_at,
                    stage=self._stage_for_event(event_type),
                    input_summary=self._summarize(step.input_data),
                    output_summary=self._summarize(step.output_data),
                    risk_level=self._risk_level(approval, tool_call),
                    citation_ids=self._extract_citation_ids(step.output_data, step.evidence),
                    trace_span_id=self._extract_trace_span_id(step.output_data),
                    approval_status=self._approval_status(event_type, approval),
                    metadata={
                        "step_id": step.id,
                        "node_name": step.node_name,
                    },
                )
            )

        return events

    def _find_approval(
        self,
        step: AgentStep,
        approval_by_id: dict[str, ApprovalRequest],
        approval_by_tool_call: dict[str, ApprovalRequest],
    ) -> ApprovalRequest | None:
        """根据 step 输出定位审批记录。"""
        approval_id = (
            step.output_data.get("approval_request_id")
            or step.output_data.get("approval_id")
            or step.input_data.get("approval_id")
        )
        if isinstance(approval_id, str) and approval_id in approval_by_id:
            return approval_by_id[approval_id]

        tool_call_id = step.output_data.get("tool_call_id")
        if isinstance(tool_call_id, str):
            return approval_by_tool_call.get(tool_call_id)
        return None

    def _find_tool_call(
        self,
        step: AgentStep,
        tool_call_by_id: dict[str, ToolCall | dict[str, Any]],
    ) -> ToolCall | dict[str, Any] | None:
        """根据 step 输出定位工具调用。"""
        tool_call_id = step.output_data.get("tool_call_id")
        if isinstance(tool_call_id, str):
            return tool_call_by_id.get(tool_call_id)
        return None

    def _stage_for_event(self, event_type: TimelineEventType) -> str:
        """将事件类型映射到复盘阶段。"""
        if event_type in {
            TimelineEventType.RUN_CREATED,
            TimelineEventType.PLAN_GENERATED,
        }:
            return "plan"
        if event_type in {
            TimelineEventType.EVIDENCE_RETRIEVED,
            TimelineEventType.TOOL_EXECUTED,
            TimelineEventType.APPROVAL_REQUESTED,
            TimelineEventType.APPROVAL_DECIDED,
        }:
            return "act"
        if event_type == TimelineEventType.REFLECTION_CREATED:
            return "reflect"
        if event_type == TimelineEventType.REPAIR_ACTION_CREATED:
            return "repair"
        return "answer"

    def _summarize(self, payload: dict[str, Any]) -> str:
        """生成可展示摘要并脱敏。"""
        if not payload:
            return ""
        text = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
        if len(text) > 500:
            text = f"{text[:497]}..."
        return self._pii_redactor.redact(text)

    def _risk_level(
        self,
        approval: ApprovalRequest | None,
        tool_call: ToolCall | dict[str, Any] | None,
    ) -> str | None:
        """推断风险等级。"""
        if approval is not None:
            return approval.risk_level.value
        if isinstance(tool_call, ToolCall) and tool_call.approval_required:
            return "write"
        if isinstance(tool_call, dict) and tool_call.get("approval_required") is True:
            return "write"
        return None

    def _approval_status(
        self,
        event_type: TimelineEventType,
        approval: ApprovalRequest | None,
    ) -> str | None:
        """推断审批状态。"""
        if event_type == TimelineEventType.APPROVAL_REQUESTED:
            return "pending"
        if approval is not None:
            return approval.status.value
        return None

    def _extract_citation_ids(
        self,
        output_data: dict[str, Any],
        evidence: list[dict],
    ) -> list[int]:
        """提取关联 citation ID。"""
        citation_ids: list[int] = []
        citations = output_data.get("citations")
        if isinstance(citations, list):
            citation_ids.extend(self._ids_from_items(citations))

        result = output_data.get("result")
        if isinstance(result, dict) and isinstance(result.get("citations"), list):
            citation_ids.extend(self._ids_from_items(result["citations"]))

        citation_ids.extend(self._ids_from_items(evidence))
        return sorted(set(citation_ids))

    def _ids_from_items(self, items: list[dict]) -> list[int]:
        """从字典列表中提取 id/citation_id。"""
        ids: list[int] = []
        for item in items:
            raw_id = item.get("id", item.get("citation_id"))
            if isinstance(raw_id, int):
                ids.append(raw_id)
        return ids

    def _extract_trace_span_id(self, output_data: dict[str, Any]) -> str | None:
        """提取 trace span ID。"""
        trace_span_id = output_data.get("trace_span_id")
        return trace_span_id if isinstance(trace_span_id, str) else None

    def _tool_call_id(self, tool_call: ToolCall | dict[str, Any]) -> str | None:
        """兼容 ToolCall 对象和历史 dict 形态。"""
        if isinstance(tool_call, ToolCall):
            return tool_call.id
        raw_id = tool_call.get("id")
        return raw_id if isinstance(raw_id, str) else None
