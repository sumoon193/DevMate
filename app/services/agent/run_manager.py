"""
Agent Run Manager。

管理 Agent Run 的完整生命周期。
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import NotFoundError, ValidationError
from app.schemas.agent import AgentPlan, AgentRunResponse, AgentStep
from app.schemas.approval import ApprovalDecision, ApprovalRequest
from app.schemas.chunk import EvidenceBundle
from app.schemas.enums import ApprovalDecisionType, ApprovalStatus, RunStatus, ToolCallStatus
from app.schemas.tool import ToolCall
from app.schemas.user import UserContext
from app.services.agent.approval_manager import ApprovalManager
from app.services.agent.approval_policy import (
    POLICY_ENGINE_APPROVER,
    ApprovalPolicy,
    NoopApprovalPolicy,
)
from app.services.agent.state_machine import AgentStateMachine
from app.services.agent.step_logger import StepLogger
from app.services.agent.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)


class AgentRunManager:
    """
    Agent Run 生命周期管理器。

    管理 Agent Run 的创建、执行、审批和完成。
    当 session_factory 不为 None 时，同时将数据持久化到 PostgreSQL。
    """

    def __init__(
        self,
        tool_executor: ToolExecutor,
        approval_manager: ApprovalManager,
        step_logger: StepLogger,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ) -> None:
        self._tool_executor = tool_executor
        self._approval_manager = approval_manager
        self._step_logger = step_logger
        self._state_machine = AgentStateMachine()
        self._approval_policy = approval_policy or NoopApprovalPolicy()
        self._runs: dict[str, AgentRunResponse] = {}  # run_id -> run
        self._session_factory = session_factory

    async def _persist_run_snapshot(self, run_id: str) -> None:
        """将内存中的 Run 及其 Steps/Approvals 快照写入数据库。"""
        if self._session_factory is None:
            return
        from app.db import crud as db

        run = self._runs.get(run_id)
        if run is None:
            return

        async with self._session_factory() as session:
            existing = await db.get_run(session, run_id)
            if existing is None:
                await db.save_run(
                    session,
                    run_id=run.id,
                    user_id=run.user_id,
                    thread_id=run.thread_id,
                    original_query=run.original_query,
                    status=run.status,
                )
            else:
                await db.update_run_status(session, run_id, run.status, run.result)

            # 持久化 steps
            for step in self._step_logger.get_steps(run_id):
                await db.save_step(
                    session,
                    run_id=step.run_id,
                    node_name=step.node_name,
                    input_data=step.input_data,
                    output_data=step.output_data,
                    evidence=step.evidence,
                    duration_ms=step.duration_ms,
                )

            # 持久化 approvals
            for tool_call in run.tool_calls:
                await db.upsert_tool_call(
                    session,
                    tool_call_id=tool_call.id,
                    run_id=tool_call.run_id,
                    tool_name=tool_call.tool_name,
                    parameters=tool_call.parameters,
                    status=tool_call.status,
                    approval_required=tool_call.approval_required,
                    result_data=tool_call.result,
                )

            for approval in self._approval_manager.get_all_requests(run_id):
                await db.upsert_approval(
                    session,
                    approval_id=approval.id,
                    run_id=approval.run_id,
                    tool_call_id=approval.tool_call_id,
                    tool_name=approval.tool_name,
                    parameters=approval.parameters,
                    expected_effect=approval.expected_effect,
                    evidence=approval.evidence,
                    risk_level=approval.risk_level.value
                    if hasattr(approval.risk_level, "value")
                    else str(approval.risk_level),
                    status=approval.status,
                    decision=approval.decision,
                    decided_by=approval.decided_by,
                    decided_at=approval.decided_at,
                    options=approval.options,
                    revision=approval.revision,
                    subject_hash=approval.subject_hash,
                    requested_by=approval.requested_by,
                    requested_at=approval.requested_at,
                    expires_at=approval.expires_at,
                    policy_version=approval.policy_version,
                    execution_manifest_hash=approval.execution_manifest_hash,
                    supersedes_approval_id=approval.supersedes_approval_id,
                    revoked_by=approval.revoked_by,
                    revoked_at=approval.revoked_at,
                    revoke_reason=approval.revoke_reason,
                )

    async def create_run(self, query: str, user_context: UserContext) -> AgentRunResponse:
        """
        创建新的 Agent Run。

        Args:
            query: 用户查询
            user_context: 用户上下文

        Returns:
            创建的 Agent Run
        """
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        thread_id = f"thread_{uuid.uuid4().hex[:12]}"

        run = AgentRunResponse(
            id=run_id,
            user_id=user_context.user_id,
            thread_id=thread_id,
            original_query=query,
            status=RunStatus.CREATED,
            steps=[],
            tool_calls=[],
            result=None,
            created_at=datetime.now(UTC),
            completed_at=None,
        )

        self._runs[run_id] = run

        # 记录步骤
        self._step_logger.log_step(
            run_id=run_id,
            node_name="run_created",
            input_data={"query": query, "user_id": user_context.user_id},
            output_data={"run_id": run_id, "thread_id": thread_id},
        )

        logger.info("run_created", extra={"run_id": run_id, "user_id": user_context.user_id})

        await self._persist_run_snapshot(run_id)
        return run

    async def start_run(self, run_id: str) -> AgentRunResponse:
        """
        启动 Agent Run。

        Args:
            run_id: Run ID

        Returns:
            更新后的 Agent Run
        """
        run = self._get_run(run_id)

        # 验证状态流转
        self._state_machine.validate_transition(run.status, RunStatus.RUNNING)

        run.status = RunStatus.RUNNING

        # 记录步骤
        self._step_logger.log_step(
            run_id=run_id,
            node_name="run_started",
            input_data={"run_id": run_id},
            output_data={"status": RunStatus.RUNNING.value},
        )

        logger.info("run_started", extra={"run_id": run_id})

        return run

    async def retrieve_evidence(self, run_id: str, evidence: EvidenceBundle) -> AgentRunResponse:
        """
        检索证据。

        Args:
            run_id: Run ID
            evidence: 证据包

        Returns:
            更新后的 Agent Run
        """
        run = self._get_run(run_id)

        # 验证状态流转
        self._state_machine.validate_transition(run.status, RunStatus.RETRIEVING_EVIDENCE)

        run.status = RunStatus.RETRIEVING_EVIDENCE

        # 记录步骤
        self._step_logger.log_step(
            run_id=run_id,
            node_name="evidence_retrieved",
            input_data={"run_id": run_id},
            output_data={
                "citation_count": evidence.total_count,
                "query_coverage": evidence.query_coverage_score,
            },
            evidence=[c.model_dump() for c in evidence.evidence_list],
        )

        logger.info(
            "evidence_retrieved", extra={"run_id": run_id, "citation_count": evidence.total_count}
        )

        return run

    async def create_plan(self, run_id: str, plan: AgentPlan) -> AgentRunResponse:
        """
        创建执行计划。

        Args:
            run_id: Run ID
            plan: 执行计划

        Returns:
            更新后的 Agent Run
        """
        run = self._get_run(run_id)

        # 验证状态流转
        self._state_machine.validate_transition(run.status, RunStatus.PLANNING)

        run.status = RunStatus.PLANNING

        # 记录步骤
        self._step_logger.log_step(
            run_id=run_id,
            node_name="plan_created",
            input_data={"run_id": run_id},
            output_data={
                "plan_id": plan.id,
                "steps": plan.steps,
                "current_step_index": plan.current_step_index,
            },
        )

        logger.info(
            "plan_created",
            extra={"run_id": run_id, "plan_id": plan.id, "step_count": len(plan.steps)},
        )

        return run

    async def execute_tool(
        self, run_id: str, tool_name: str, parameters: dict[str, Any], user_context: UserContext
    ) -> ToolCall:
        """
        执行工具。

        Args:
            run_id: Run ID
            tool_name: 工具名称
            parameters: 工具参数
            user_context: 用户上下文

        Returns:
            工具调用记录
        """
        run = self._get_run(run_id)

        # 执行工具
        tool_call = await self._tool_executor.execute(
            run_id=run_id, tool_name=tool_name, parameters=parameters, user_context=user_context
        )
        self._upsert_tool_call(run, tool_call)

        # 如果需要审批，更新 Run 状态
        if tool_call.approval_required and tool_call.status == ToolCallStatus.PENDING:
            run.status = RunStatus.AWAITING_APPROVAL

        return tool_call

    async def apply_approval_decision(
        self,
        run_id: str,
        approval_id: str,
        approval_decision: ApprovalDecision,
        user_context: UserContext,
    ) -> ApprovalRequest:
        """
        应用人工审批决策。

        Args:
            run_id: Run ID
            approval_id: 审批请求 ID
            approval_decision: 审批决策
            user_context: 用户上下文

        Returns:
            更新后的审批请求
        """
        request = self._approval_manager.get_request(approval_id)
        if request.run_id != run_id:
            raise ValidationError(f"Approval request {approval_id} does not belong to run {run_id}")

        if request.status != ApprovalStatus.PENDING:
            return request

        if approval_decision.decision == ApprovalDecisionType.APPROVE:
            return self._approval_manager.approve(approval_id, user_context.user_id)

        if approval_decision.decision == ApprovalDecisionType.EDIT:
            edited_parameters = approval_decision.edited_parameters
            if not edited_parameters:
                raise ValidationError("edited_parameters is required when decision is edit")
            return self._approval_manager.edit_and_approve(
                approval_id=approval_id,
                edited_parameters=edited_parameters,
                decided_by=user_context.user_id,
            )

        return self._approval_manager.reject(approval_id, user_context.user_id)

    def maybe_auto_approve(
        self,
        run_id: str,
        approval_id: str,
        user_context: UserContext,
    ) -> ApprovalDecision | None:
        """
        策略命中时自动审批。

        命中规则时立即以 policy_engine 身份批准并记录审计步骤，
        返回合成决策；未命中返回 None，走人工审批。

        Args:
            run_id: Run ID
            approval_id: 审批请求 ID
            user_context: 用户上下文

        Returns:
            自动决策，或 None（转人工）
        """
        request = self._approval_manager.get_request(approval_id)
        if request.status != ApprovalStatus.PENDING:
            return None

        decision = self._approval_policy.evaluate(
            tool_name=request.tool_name,
            parameters=request.parameters,
            risk_level=request.risk_level,
            user_context=user_context,
        )
        if decision is None:
            return None

        self._approval_manager.approve(approval_id, POLICY_ENGINE_APPROVER)
        self._step_logger.log_step(
            run_id=run_id,
            node_name="approval_auto_approved",
            input_data={
                "approval_id": approval_id,
                "tool_name": request.tool_name,
                "risk_level": request.risk_level.value,
            },
            output_data={
                "decision": decision.value,
                "decided_by": POLICY_ENGINE_APPROVER,
            },
        )

        logger.info(
            "approval_auto_approved",
            extra={
                "run_id": run_id,
                "approval_id": approval_id,
                "tool_name": request.tool_name,
            },
        )

        return ApprovalDecision(decision=decision)

    async def auto_approve_and_execute(
        self,
        run_id: str,
        user_context: UserContext,
    ) -> ToolCall | None:
        """
        demo 链路：策略命中时自动审批并执行写入工具。

        用于确定性 fallback demo 的无人值守场景；未命中返回 None。

        Args:
            run_id: Run ID
            user_context: 用户上下文

        Returns:
            已执行的 ToolCall；策略未命中时为 None
        """
        pending = self._approval_manager.get_pending_requests(run_id)
        if not pending:
            return None

        approval = pending[0]
        decision = self.maybe_auto_approve(run_id, approval.id, user_context)
        if decision is None:
            return None

        return await self.execute_approved_tool(run_id, approval.id, user_context)

    async def execute_approved_tool(
        self, run_id: str, approval_id: str, user_context: UserContext
    ) -> ToolCall:
        """
        执行已审批通过的工具。

        Args:
            run_id: Run ID
            approval_id: 审批请求 ID
            user_context: 用户上下文

        Returns:
            工具调用结果
        """
        run = self._get_run(run_id)
        approval_request = self._approval_manager.get_request(approval_id)
        if approval_request.run_id != run_id:
            raise ValidationError(f"Approval request {approval_id} does not belong to run {run_id}")
        if approval_request.status != ApprovalStatus.APPROVED:
            raise ValidationError(f"Approval request {approval_id} is not approved")

        if run.status == RunStatus.AWAITING_APPROVAL:
            self._state_machine.validate_transition(run.status, RunStatus.RESUMED)
            run.status = RunStatus.RESUMED

        tool_call = await self._tool_executor.execute_after_approval(
            run_id=run_id,
            tool_call_id=approval_request.tool_call_id,
            approval_id=approval_id,
            user_context=user_context,
        )
        self._upsert_tool_call(run, tool_call)

        self._step_logger.log_step(
            run_id=run_id,
            node_name="run_resumed_after_approval",
            input_data={"run_id": run_id, "approval_id": approval_id},
            output_data={"tool_call_id": tool_call.id, "status": tool_call.status.value},
        )

        logger.info("approved_tool_executed", extra={"run_id": run_id, "approval_id": approval_id})

        return tool_call

    async def mark_resumed_without_tool(self, run_id: str, reason: str) -> AgentRunResponse:
        """
        在不执行工具的情况下恢复 Run。

        用于审批拒绝路径，让后续 answer/finalize 可以正常完成。
        """
        run = self._get_run(run_id)
        if run.status == RunStatus.AWAITING_APPROVAL:
            self._state_machine.validate_transition(run.status, RunStatus.RESUMED)
            run.status = RunStatus.RESUMED

        self._step_logger.log_step(
            run_id=run_id,
            node_name="run_resumed_without_tool",
            input_data={"run_id": run_id},
            output_data={"reason": reason},
        )

        return run

    async def resume_after_approval(
        self, run_id: str, approval_id: str, user_context: UserContext
    ) -> AgentRunResponse:
        """
        审批后恢复执行。

        Args:
            run_id: Run ID
            approval_id: 审批请求 ID
            user_context: 用户上下文

        Returns:
            更新后的 Agent Run
        """
        run = self._get_run(run_id)

        # 验证状态流转
        self._state_machine.validate_transition(run.status, RunStatus.RESUMED)

        await self.execute_approved_tool(run_id, approval_id, user_context)

        return run

    async def complete_run(self, run_id: str, result: dict[str, Any]) -> AgentRunResponse:
        """
        完成 Agent Run。

        Args:
            run_id: Run ID
            result: 最终结果

        Returns:
            更新后的 Agent Run
        """
        run = self._get_run(run_id)

        # 验证状态流转
        self._state_machine.validate_transition(run.status, RunStatus.COMPLETED)

        run.status = RunStatus.COMPLETED
        run.result = result
        run.completed_at = datetime.now(UTC)

        # 记录步骤
        self._step_logger.log_step(
            run_id=run_id,
            node_name="run_completed",
            input_data={"run_id": run_id},
            output_data={"result": result},
        )

        logger.info("run_completed", extra={"run_id": run_id})

        await self._persist_run_snapshot(run_id)
        return run

    async def fail_run(self, run_id: str, error: str) -> AgentRunResponse:
        """
        标记 Agent Run 失败。

        Args:
            run_id: Run ID
            error: 错误信息

        Returns:
            更新后的 Agent Run
        """
        run = self._get_run(run_id)

        # 验证状态流转
        self._state_machine.validate_transition(run.status, RunStatus.FAILED)

        run.status = RunStatus.FAILED
        run.result = {"error": error}
        run.completed_at = datetime.now(UTC)

        # 记录步骤
        self._step_logger.log_step(
            run_id=run_id,
            node_name="run_failed",
            input_data={"run_id": run_id},
            output_data={"error": error},
        )

        logger.error("run_failed", extra={"run_id": run_id, "error": error})

        await self._persist_run_snapshot(run_id)
        return run

    async def cancel_run(self, run_id: str) -> AgentRunResponse:
        """
        取消 Agent Run。

        Args:
            run_id: Run ID

        Returns:
            更新后的 Agent Run
        """
        run = self._get_run(run_id)

        # 验证状态流转
        self._state_machine.validate_transition(run.status, RunStatus.CANCELLED)

        run.status = RunStatus.CANCELLED
        run.completed_at = datetime.now(UTC)

        # 记录步骤
        self._step_logger.log_step(
            run_id=run_id,
            node_name="run_cancelled",
            input_data={"run_id": run_id},
            output_data={"status": RunStatus.CANCELLED.value},
        )

        logger.info("run_cancelled", extra={"run_id": run_id})

        await self._persist_run_snapshot(run_id)
        return run

    def _get_run(self, run_id: str) -> AgentRunResponse:
        """获取 Agent Run（内部方法）。"""
        if run_id not in self._runs:
            raise NotFoundError(f"Agent Run not found: {run_id}")
        return self._runs[run_id]

    async def get_run(self, run_id: str) -> AgentRunResponse:
        """
        获取 Agent Run。

        优先从内存缓存读取；full 模式下若内存没有则从数据库加载。
        """
        if run_id in self._runs:
            return self._runs[run_id]

        if self._session_factory is not None:
            from app.db import crud as db

            async with self._session_factory() as session:
                orm_run = await db.get_run(session, run_id)
                if orm_run is not None:
                    # 从 ORM 构建 Pydantic schema
                    run = AgentRunResponse(
                        id=orm_run.id,
                        user_id=orm_run.user_id,
                        thread_id=orm_run.thread_id,
                        original_query=orm_run.original_query,
                        status=orm_run.status,
                        steps=[],
                        tool_calls=[],
                        result=orm_run.result,
                        created_at=orm_run.created_at,
                        completed_at=orm_run.completed_at,
                    )
                    self._runs[run_id] = run
                    return run

        raise NotFoundError(f"Agent Run not found: {run_id}")

    async def get_run_steps(self, run_id: str) -> list[AgentStep]:
        """
        获取 Agent Run 的所有步骤。

        Args:
            run_id: Run ID

        Returns:
            步骤列表
        """
        return self._step_logger.get_steps(run_id)

    async def get_pending_approvals(self, run_id: str) -> list[ApprovalRequest]:
        """
        获取 Agent Run 的待审批请求。

        Args:
            run_id: Run ID

        Returns:
            待审批请求列表
        """
        return self._approval_manager.get_pending_requests(run_id)

    async def get_run_approvals(self, run_id: str) -> list[ApprovalRequest]:
        """
        获取 Agent Run 的所有审批请求。

        Args:
            run_id: Run ID

        Returns:
            审批请求列表
        """
        self._get_run(run_id)
        return self._approval_manager.get_all_requests(run_id)

    def _get_run(self, run_id: str) -> AgentRunResponse:
        """获取 Agent Run（内部方法）。"""
        if run_id not in self._runs:
            raise NotFoundError(f"Agent Run not found: {run_id}")
        return self._runs[run_id]

    def _upsert_tool_call(self, run: AgentRunResponse, tool_call: ToolCall) -> None:
        """新增或更新 Run 中的工具调用记录。"""
        for index, existing_call in enumerate(run.tool_calls):
            if existing_call.id == tool_call.id:
                run.tool_calls[index] = tool_call
                return

        run.tool_calls.append(tool_call)
