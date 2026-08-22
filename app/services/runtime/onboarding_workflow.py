"""员工入职到转正 Reference Application 的长期 Case 编排。"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import timedelta
from typing import Any

from app.core.exceptions import NotFoundError, PermissionError, ValidationError
from app.schemas.approval import ApprovalRequest
from app.schemas.enums import ApprovalDecisionType, ApprovalStatus, ToolRiskLevel
from app.schemas.runtime import HRCase, RunEventEnvelope
from app.schemas.tool import ToolCall
from app.schemas.user import UserContext
from app.services.a2a.policy_research import InProcessA2AClient
from app.services.agent.approval_manager import ApprovalManager
from app.services.context.compactor import ContextCompactor
from app.services.mcp.adapter import McpApprovalBridge, McpToolAdapter
from app.services.memory.store import EpisodicMemoryStore
from app.services.observability.context import TraceContext
from app.services.observability.runtime_metrics import RuntimeMetrics
from app.services.observability.span import Span, SpanStatus, SpanType
from app.services.observability.tracer import Tracer
from app.services.runtime.case_service import CaseService
from app.services.runtime.clock import Clock, SystemClock
from app.services.runtime.interfaces import EventStore
from app.services.runtime.timer_coordinator import TimerCoordinator
from app.services.skills.registry import SkillRegistry


class OnboardingCaseWorkflow:
    """把制度研究、计划、审批、写工具和跨天 timer 组合为治理主链路。"""

    def __init__(
        self,
        *,
        case_service: CaseService,
        event_store: EventStore,
        skill_registry: SkillRegistry,
        memory_store: EpisodicMemoryStore,
        context_compactor: ContextCompactor,
        a2a_client: InProcessA2AClient,
        mcp_adapter: McpToolAdapter,
        mcp_approval_bridge: McpApprovalBridge,
        approval_manager: ApprovalManager,
        timer_coordinator: TimerCoordinator,
        clock: Clock | None = None,
        metrics: RuntimeMetrics | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._cases = case_service
        self._events = event_store
        self._skills = skill_registry
        self._memories = memory_store
        self._compactor = context_compactor
        self._a2a = a2a_client
        self._mcp = mcp_adapter
        self._mcp_approval = mcp_approval_bridge
        self._approvals = approval_manager
        self._timers = timer_coordinator
        self._clock = clock or SystemClock()
        self._metrics = metrics
        self._tracer = tracer

    async def start(
        self,
        *,
        case_id: str,
        user_context: UserContext,
        expected_version: int,
        command_id: str,
    ) -> HRCase:
        """启动标准入职 Case，并在首个写操作前暂停等待审批。"""
        if await self._find_command(case_id, f"{command_id}:context") is not None:
            return await self._cases.get_case(case_id)

        case = await self._cases.get_case(case_id)
        persisted_run = await self._find_command(case_id, f"{command_id}:run")
        if persisted_run is None:
            self._validate_start(case, user_context, expected_version)
            run_id = f"run_{uuid.uuid4().hex[:12]}"
        else:
            if case.tenant_id != user_context.tenant_id:
                raise PermissionError(f"Case tenant mismatch: {case.id}")
            run_id = str(persisted_run.payload["run_id"])
        skill = self._skills.resolve("hr_onboarding")
        if skill is None:
            raise ValidationError("Active hr_onboarding Skill is required")
        missing_permissions = set(skill.required_permissions) - set(user_context.permissions)
        if missing_permissions:
            raise PermissionError(
                f"Skill permissions missing: {', '.join(sorted(missing_permissions))}"
            )

        trace_context = None
        root_span = None
        if self._tracer is not None:
            trace_context = self._tracer.start_trace(
                run_id,
                user_context.user_id,
                user_context.tenant_id,
                case_id=case_id,
            )
            root_span = self._tracer.start_span(
                trace_context,
                SpanType.AGENT_RUN,
                "onboarding_case_start",
                attributes={"workflow": "onboarding_to_regularization"},
            )
        if self._metrics is not None:
            self._metrics.increment("runtime.cases.started")
        case = await self._record(
            case,
            event_type="run.started",
            payload={"run_id": run_id, "workflow": "onboarding_to_regularization"},
            actor_id=user_context.user_id,
            command_id=f"{command_id}:run",
        )
        case = await self._record(
            case,
            event_type="skill.loaded",
            payload={
                "skill_id": skill.id,
                "name": skill.name,
                "version": skill.version,
                "checksum": skill.checksum,
            },
            actor_id=user_context.user_id,
            command_id=f"{command_id}:skill",
        )

        task = await self._a2a.send_message(
            context_id=case_id,
            text="研究当前入职与转正制度、适用范围和所需材料",
            user_context=user_context,
        )
        artifact = task.artifacts[0]
        if self._metrics is not None:
            self._metrics.increment("runtime.protocol.a2a.success")
        citations = list(artifact.content.get("citations", []))
        case = await self._record(
            case,
            event_type="a2a.task.completed",
            payload={
                "task_id": task.id,
                "artifact_id": artifact.id,
                "name": artifact.name,
                "uri": artifact.uri,
                "read_only": True,
            },
            actor_id="agent:policy-research",
            command_id=f"{command_id}:a2a",
        )
        case = await self._record(
            case,
            event_type="evidence.retrieved",
            payload={
                "citations": citations,
                "citation_ids": [item.get("chunk_id") for item in citations],
                "policy_version": case.execution_manifest.policy_version,
                "freshness": "active",
            },
            actor_id="agent:policy-research",
            command_id=f"{command_id}:evidence",
        )
        case = await self._record(
            case,
            event_type="plan.created",
            payload={
                "plan_id": f"plan_{uuid.uuid4().hex[:12]}",
                "steps": [
                    {"id": "collect_materials", "owner": "HR", "status": "ready"},
                    {"id": "provision_accounts", "owner": "IT", "status": "ready"},
                    {
                        "id": "confirm_probation_goals",
                        "owner": "Manager",
                        "status": "ready",
                    },
                    {
                        "id": "schedule_regularization",
                        "owner": "Harness",
                        "status": "ready",
                    },
                ],
                "evidence_version": case.execution_manifest.policy_version,
            },
            actor_id="agent:harness",
            command_id=f"{command_id}:plan",
        )

        case, _ = await self._prepare_ticket_approval(
            case=case,
            run_id=run_id,
            parameters={
                "title": "新员工入职到转正办理工单",
                "description": f"为 {case.subject_user_id} 创建跨部门入职办理 Case",
            },
            citations=citations,
            policy_version=case.execution_manifest.policy_version,
            execution_manifest_hash=self._manifest_hash(case),
            user_context=user_context,
            command_id=command_id,
        )

        snapshot = self._compactor.compact(
            case_id=case_id,
            events=await self._events.load_stream(case_id),
            summarizer_version="deterministic-structured-v1",
            selector_version="safe-prefix-v1",
        )
        completed = await self._record(
            case,
            event_type="context.compacted",
            payload=snapshot.model_dump(mode="json"),
            actor_id="agent:harness",
            command_id=f"{command_id}:context",
        )
        if self._metrics is not None and not snapshot.invariant_check_passed:
            self._metrics.increment("runtime.context.invariant_failures")
        if self._tracer is not None and trace_context is not None and root_span is not None:
            last_event = (await self._events.load_stream(case_id))[-1]
            trace_context.event_id = last_event.id
            root_span.set_attribute("event_id", last_event.id)
            root_span.set_attribute("case_version", completed.version)
            self._tracer.end_span(root_span, SpanStatus.OK)
            self._tracer.export_trace(trace_context.trace_id)
        return completed

    async def decide_approval(
        self,
        *,
        case_id: str,
        approval_id: str,
        decision: ApprovalDecisionType,
        decided_by: str,
        user_context: UserContext,
        expected_version: int,
        command_id: str,
        edited_parameters: dict[str, Any] | None = None,
    ) -> HRCase:
        """记录人工决策；批准后恢复 checkpoint 语义并执行一次副作用。"""
        persisted_decision = await self._find_command(
            case_id,
            f"{command_id}:approval",
        )
        case = await self._cases.get_case(case_id)
        if case.tenant_id != user_context.tenant_id:
            raise PermissionError(f"Case tenant mismatch: {case.id}")

        approval = await self._restore_approval(case_id, approval_id)
        if approval.run_id != case.active_run_id:
            raise ValidationError(f"Approval does not belong to Case: {approval_id}")

        decision_was_persisted = persisted_decision is not None
        if persisted_decision is not None:
            if persisted_decision.payload.get("approval_id") != approval_id:
                raise ValidationError(f"Command {command_id} is bound to another approval")
            if persisted_decision.payload.get("decision") != decision.value:
                raise ValidationError(f"Command {command_id} is bound to another decision")
            if persisted_decision.payload.get("decided_by") != decided_by:
                raise ValidationError(f"Command {command_id} is bound to another decision maker")
            raw_effective = persisted_decision.payload.get("approval")
            if not isinstance(raw_effective, dict):
                raise ValidationError("Persisted approval decision is incomplete")
            effective = ApprovalRequest.model_validate(raw_effective)
            if decision == ApprovalDecisionType.EDIT:
                if not edited_parameters:
                    raise ValidationError("edited_parameters is required when decision is edit")
                if effective.parameters != edited_parameters:
                    raise ValidationError(
                        f"Command {command_id} is bound to other edited parameters"
                    )
            self._approvals.restore_request(effective)
        else:
            if case.version != expected_version:
                raise ValidationError(
                    f"Case version conflict: expected {expected_version}, actual {case.version}"
                )
            approval_projection = next(
                (
                    item
                    for item in case.working_memory.get("approvals", [])
                    if item.get("approval_id") == approval_id
                ),
                None,
            )
            if approval_projection is None:
                raise ValidationError(f"Approval does not belong to Case: {approval_id}")
            if approval_projection.get("status") != "pending":
                raise ValidationError(
                    f"Approval request {approval_id} is not pending "
                    f"({approval_projection.get('status')})"
                )

            if decision == ApprovalDecisionType.APPROVE:
                effective = self._approvals.approve(approval_id, decided_by)
            elif decision == ApprovalDecisionType.EDIT:
                if not edited_parameters:
                    raise ValidationError("edited_parameters is required when decision is edit")
                effective = self._approvals.edit_and_approve(
                    approval_id,
                    edited_parameters,
                    decided_by,
                )
            else:
                effective = self._approvals.reject(approval_id, decided_by)

            case = await self._record(
                case,
                event_type="approval.decided",
                payload={
                    "approval_id": approval_id,
                    "effective_approval_id": effective.id,
                    "decision": decision.value,
                    "status": effective.status.value,
                    "decided_by": decided_by,
                    "revision": effective.revision,
                    "approval": effective.model_dump(mode="json"),
                },
                actor_id=decided_by,
                command_id=f"{command_id}:approval",
            )

        if effective.run_id != case.active_run_id:
            raise ValidationError(f"Approval does not belong to Case: {approval_id}")
        if effective.status != ApprovalStatus.REJECTED:
            persisted_timer = await self._find_command(
                case_id,
                f"{command_id}:timer",
            )
            if persisted_timer is not None:
                return await self._cases.get_case(case_id)

        trace_context = None
        approval_span = None
        if self._tracer is not None:
            trace_context = self._tracer.start_trace(
                approval.run_id,
                user_context.user_id,
                user_context.tenant_id,
                case_id=case_id,
            )
            approval_span = self._tracer.start_span(
                trace_context,
                SpanType.APPROVAL_WAIT,
                "case_approval_resume",
                attributes={"approval_id": approval_id},
            )
        if self._metrics is not None and not decision_was_persisted:
            self._metrics.increment("runtime.approvals.decided")
            self._metrics.increment("runtime.human_interventions.total")
            self._metrics.set_gauge("runtime.approvals.stuck", 0)
            if approval.requested_at is not None:
                self._metrics.observe(
                    "runtime.approvals.wait_seconds",
                    max(0.0, (self._clock.now() - approval.requested_at).total_seconds()),
                )
        if effective.status == ApprovalStatus.REJECTED:
            self._finish_approval_trace(
                trace_context,
                approval_span,
                case_id=case_id,
                event_id=(await self._events.load_stream(case_id))[-1].id,
            )
            return case

        tool_event = await self._find_command(case_id, f"{command_id}:tool")
        if tool_event is None:
            tool_call = await self._mcp_approval.execute_approved(
                run_id=effective.run_id,
                approval_id=effective.id,
                user_context=user_context,
            )
            case = await self._record(
                case,
                event_type="tool.executed",
                payload={
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.tool_name,
                    "status": tool_call.status.value,
                    "result": tool_call.result,
                    "approval_id": effective.id,
                },
                actor_id="agent:harness",
                command_id=f"{command_id}:tool",
            )
            tool_event = await self._find_command(case_id, f"{command_id}:tool")
            if self._metrics is not None:
                self._metrics.increment("runtime.side_effects.succeeded")
        if tool_event is None:
            raise ValidationError("Tool execution event was not persisted")

        memory_event = await self._find_command(case_id, f"{command_id}:memory")
        if memory_event is None:
            memory = await self._memories.remember(
                tenant_id=case.tenant_id,
                case_id=case.id,
                memory_key="onboarding.ticket_created",
                content=("入职工单已在人工审批后创建，后续应按 durable timer 跟进转正。"),
                provenance_event_ids=[tool_event.id],
            )
            case = await self._record(
                case,
                event_type="memory.stored",
                payload={
                    "memory_id": memory.id,
                    "memory_key": memory.memory_key,
                    "status": memory.status.value,
                    "provenance_event_ids": memory.provenance_event_ids,
                },
                actor_id="agent:harness",
                command_id=f"{command_id}:memory",
            )
            if self._metrics is not None:
                self._metrics.increment("runtime.memories.stored")

        scheduled = await self._timers.schedule(
            case_id=case_id,
            timer_type="probation.review_due",
            due_at=self._clock.now() + timedelta(days=80),
            payload={"employee_id": case.subject_user_id},
            actor_id="agent:harness",
            command_id=f"{command_id}:timer",
            expected_version=case.version,
        )
        if self._metrics is not None:
            self._metrics.increment("runtime.timers.scheduled")
        self._finish_approval_trace(
            trace_context,
            approval_span,
            case_id=case_id,
            event_id=(await self._events.load_stream(case_id))[-1].id,
        )
        return scheduled.case

    async def refresh_policy(
        self,
        *,
        case_id: str,
        policy_version: str,
        user_context: UserContext,
        expected_version: int,
        command_id: str,
    ) -> HRCase:
        """制度变化后重新研究 evidence、修订计划并生成新审批。"""
        if await self._find_command(case_id, f"{command_id}:context") is not None:
            return await self._cases.get_case(case_id)
        case = await self._cases.get_case(case_id)
        persisted_run = await self._find_command(case_id, f"{command_id}:run")
        persisted_stale = await self._find_command(case_id, f"{command_id}:stale")
        if persisted_run is None:
            self._validate_start(case, user_context, expected_version)
            old_version = case.policy_versions.get(
                "hr_policy",
                case.execution_manifest.policy_version,
            )
            if old_version == policy_version:
                raise ValidationError(f"Policy version is already active: {policy_version}")
            run_id = f"run_{uuid.uuid4().hex[:12]}"
        else:
            if case.tenant_id != user_context.tenant_id:
                raise PermissionError(f"Case tenant mismatch: {case.id}")
            run_id = str(persisted_run.payload["run_id"])
            old_version = str(
                persisted_stale.payload["previous_policy_version"]
                if persisted_stale is not None
                else case.execution_manifest.policy_version
            )
        case = await self._record(
            case,
            event_type="run.started",
            payload={
                "run_id": run_id,
                "workflow": "policy_refresh_and_replan",
                "policy_version": policy_version,
            },
            actor_id=user_context.user_id,
            command_id=f"{command_id}:run",
        )
        case = await self._record(
            case,
            event_type="policy.stale_detected",
            payload={
                "previous_policy_version": old_version,
                "active_policy_version": policy_version,
                "reason": "document_version_changed",
            },
            actor_id="agent:harness",
            command_id=f"{command_id}:stale",
        )
        task = await self._a2a.send_message(
            context_id=case_id,
            text="研究更新后的入职与转正制度、适用范围和所需材料",
            user_context=user_context,
            policy_version=policy_version,
        )
        artifact = task.artifacts[0]
        citations = list(artifact.content.get("citations", []))
        case = await self._record(
            case,
            event_type="a2a.task.completed",
            payload={
                "task_id": task.id,
                "artifact_id": artifact.id,
                "name": artifact.name,
                "uri": artifact.uri,
                "read_only": True,
                "policy_version": policy_version,
            },
            actor_id="agent:policy-research",
            command_id=f"{command_id}:a2a",
        )
        case = await self._record(
            case,
            event_type="evidence.retrieved",
            payload={
                "citations": citations,
                "citation_ids": [item.get("chunk_id") for item in citations],
                "policy_version": policy_version,
                "freshness": "refreshed",
            },
            actor_id="agent:policy-research",
            command_id=f"{command_id}:evidence",
        )
        case = await self._record(
            case,
            event_type="policy.refreshed",
            payload={
                "policy_version": policy_version,
                "supersedes": old_version,
                "artifact_id": artifact.id,
            },
            actor_id="agent:harness",
            command_id=f"{command_id}:policy",
        )
        case = await self._record(
            case,
            event_type="plan.revised",
            payload={
                "plan_id": f"plan_{uuid.uuid4().hex[:12]}",
                "revision_reason": "policy_version_changed",
                "evidence_version": policy_version,
                "steps": [
                    {"id": "review_changed_policy", "owner": "HR", "status": "ready"},
                    {"id": "reconfirm_probation_goals", "owner": "Manager", "status": "ready"},
                    {"id": "create_change_ticket", "owner": "Harness", "status": "ready"},
                ],
            },
            actor_id="agent:harness",
            command_id=f"{command_id}:plan",
        )
        case, _ = await self._prepare_ticket_approval(
            case=case,
            run_id=run_id,
            parameters={
                "title": "转正制度变更复核工单",
                "description": (
                    f"制度由 {old_version} 更新为 {policy_version}，"
                    f"需复核 {case.subject_user_id} 的转正计划"
                ),
            },
            citations=citations,
            policy_version=policy_version,
            execution_manifest_hash=self._manifest_hash_for_policy(
                case,
                policy_version,
            ),
            user_context=user_context,
            command_id=command_id,
        )
        if self._metrics is not None:
            self._metrics.increment("runtime.evidence.stale")
            self._metrics.increment("runtime.protocol.a2a.success")
        snapshot = self._compactor.compact(
            case_id=case_id,
            events=await self._events.load_stream(case_id),
            summarizer_version="deterministic-structured-v1",
            selector_version="safe-prefix-v1",
        )
        return await self._record(
            case,
            event_type="context.compacted",
            payload=snapshot.model_dump(mode="json"),
            actor_id="agent:harness",
            command_id=f"{command_id}:context",
        )

    async def _prepare_ticket_approval(
        self,
        *,
        case: HRCase,
        run_id: str,
        parameters: dict[str, Any],
        citations: list[dict[str, Any]],
        policy_version: str,
        execution_manifest_hash: str,
        user_context: UserContext,
        command_id: str,
    ) -> tuple[HRCase, ApprovalRequest]:
        """幂等恢复或创建写工具预览和绑定审批。"""
        persisted_approval = await self._find_command(
            case.id,
            f"{command_id}:approval",
        )
        if persisted_approval is not None:
            raw = persisted_approval.payload.get("approval")
            if not isinstance(raw, dict):
                raise ValidationError("Persisted approval payload is incomplete")
            approval = ApprovalRequest.model_validate(raw)
            self._approvals.restore_request(approval)
            return await self._cases.get_case(case.id), approval

        persisted_tool = await self._find_command(case.id, f"{command_id}:tool")
        if persisted_tool is not None:
            tool_call = ToolCall(
                id=str(persisted_tool.payload["tool_call_id"]),
                run_id=run_id,
                tool_name=str(persisted_tool.payload["tool_name"]),
                parameters=dict(persisted_tool.payload["parameters"]),
                status=str(persisted_tool.payload["status"]),
                approval_required=True,
            )
            approval = self._approvals.create_request(
                run_id=run_id,
                tool_call=tool_call,
                tool_name=tool_call.tool_name,
                parameters=tool_call.parameters,
                risk_level=ToolRiskLevel.WRITE,
                user_context=user_context,
                evidence=citations,
                policy_version=policy_version,
                execution_manifest_hash=execution_manifest_hash,
            )
            case = await self._cases.get_case(case.id)
        else:
            tool_call = await self._mcp.call(
                run_id=run_id,
                tool_name="create_hr_ticket",
                parameters=parameters,
                user_context=user_context,
                approval_evidence=citations,
                policy_version=policy_version,
                execution_manifest_hash=execution_manifest_hash,
            )
            approval = self._approvals.get_pending_requests(run_id)[-1]
            case = await self._record(
                case,
                event_type="tool.call_prepared",
                payload={
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.tool_name,
                    "parameters": tool_call.parameters,
                    "status": tool_call.status.value,
                    "requires_approval": True,
                },
                actor_id="agent:harness",
                command_id=f"{command_id}:tool",
            )
            if self._metrics is not None:
                self._metrics.increment("runtime.protocol.mcp.success")

        case = await self._record(
            case,
            event_type="approval.requested",
            payload={
                "approval_id": approval.id,
                "tool_call_id": tool_call.id,
                "tool_name": approval.tool_name,
                "subject_hash": approval.subject_hash,
                "revision": approval.revision,
                "expires_at": approval.expires_at.isoformat() if approval.expires_at else None,
                "policy_version": approval.policy_version,
                "approval": approval.model_dump(mode="json"),
            },
            actor_id=user_context.user_id,
            command_id=f"{command_id}:approval",
        )
        if self._metrics is not None:
            self._metrics.increment("runtime.approvals.requested")
            self._metrics.set_gauge("runtime.approvals.stuck", 1)
        return case, approval

    def _finish_approval_trace(
        self,
        trace_context: TraceContext | None,
        approval_span: Span | None,
        *,
        case_id: str,
        event_id: str,
    ) -> None:
        """完成带 Case/Event 关联字段的审批恢复 span。"""
        if self._tracer is None or trace_context is None or approval_span is None:
            return
        trace_context.event_id = event_id
        approval_span.set_attribute("case_id", case_id)
        approval_span.set_attribute("event_id", event_id)
        self._tracer.end_span(approval_span, SpanStatus.OK)
        self._tracer.export_trace(trace_context.trace_id)

    def _validate_start(
        self,
        case: HRCase,
        user_context: UserContext,
        expected_version: int,
    ) -> None:
        if case.version != expected_version:
            raise ValidationError(
                f"Case version conflict: expected {expected_version}, actual {case.version}"
            )
        if case.tenant_id != user_context.tenant_id:
            raise PermissionError(f"Case tenant mismatch: {case.id}")

    async def _record(
        self,
        case: HRCase,
        *,
        event_type: str,
        payload: dict[str, Any],
        actor_id: str,
        command_id: str,
    ) -> HRCase:
        if await self._find_command(case.id, command_id) is not None:
            return await self._cases.get_case(case.id)
        return await self._cases.record_event(
            case_id=case.id,
            event_type=event_type,
            payload=payload,
            actor_id=actor_id,
            command_id=command_id,
            expected_version=case.version,
        )

    async def _find_command(
        self,
        case_id: str,
        command_id: str,
    ) -> RunEventEnvelope | None:
        events = await self._events.load_stream(case_id)
        return next(
            (event for event in events if event.command_id == command_id),
            None,
        )

    async def _restore_approval(
        self,
        case_id: str,
        approval_id: str,
    ) -> ApprovalRequest:
        """从内存或持久事件恢复完整审批对象。"""
        try:
            return self._approvals.get_request(approval_id)
        except NotFoundError as exc:
            events = await self._events.load_stream(case_id)
            for event in reversed(events):
                raw = event.payload.get("approval")
                if not isinstance(raw, dict):
                    continue
                restored = ApprovalRequest.model_validate(raw)
                if restored.id == approval_id:
                    return self._approvals.restore_request(restored)
            raise NotFoundError(
                f"Approval request not found in Case events: {approval_id}"
            ) from exc

    @staticmethod
    def _manifest_hash(case: HRCase) -> str:
        canonical = json.dumps(
            case.execution_manifest.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _manifest_hash_for_policy(case: HRCase, policy_version: str) -> str:
        manifest = case.execution_manifest.model_copy(update={"policy_version": policy_version})
        canonical = json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
