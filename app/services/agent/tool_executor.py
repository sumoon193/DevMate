"""
Tool Executor。

执行工具，带权限检查和审批流程。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.core.exceptions import PermissionError as AppPermissionError
from app.schemas.enums import SideEffectStatus, ToolCallStatus
from app.schemas.tool import ToolCall
from app.schemas.user import UserContext
from app.services.agent.approval_manager import ApprovalManager
from app.services.agent.step_logger import StepLogger
from app.services.agent.tool_registry import ToolRegistry
from app.services.runtime.interfaces import SideEffectLedger
from app.services.runtime.side_effects import InMemorySideEffectLedger
from app.services.security.acl_validator import ACLValidator

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    工具执行器。

    执行工具，处理权限检查和审批流程。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        approval_manager: ApprovalManager,
        step_logger: StepLogger,
        acl_validator: ACLValidator | None = None,
        side_effect_ledger: SideEffectLedger | None = None,
    ) -> None:
        """
        初始化工具执行器。

        Args:
            registry: 工具注册表
            approval_manager: 审批管理器
            step_logger: 步骤记录器
            acl_validator: ACL 校验器
        """
        self._registry = registry
        self._approval_manager = approval_manager
        self._step_logger = step_logger
        self._acl_validator = acl_validator or ACLValidator()
        self._side_effect_ledger = side_effect_ledger or InMemorySideEffectLedger()

    async def execute(
        self,
        run_id: str,
        tool_name: str,
        parameters: dict[str, Any],
        user_context: UserContext,
        *,
        approval_evidence: list[dict[str, Any]] | None = None,
        policy_version: str = "",
        execution_manifest_hash: str = "",
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

        Raises:
            NotFoundError: 工具不存在
        """
        logger.info("executing_tool", extra={"run_id": run_id, "tool_name": tool_name})

        # 获取工具定义和处理器
        tool = self._registry.get_tool(tool_name)
        self._ensure_tool_permission(run_id, tool_name, tool.permission_scope, user_context)

        # 创建 ToolCall 记录
        tool_call_id = f"tool_{uuid.uuid4().hex[:12]}"

        # 检查是否需要审批
        if tool.requires_approval:
            # 生成 ApprovalRequest
            tool_call = ToolCall(
                id=tool_call_id,
                run_id=run_id,
                tool_name=tool_name,
                parameters=parameters,
                result=None,
                status=ToolCallStatus.PENDING,
                approval_required=True,
            )

            # 创建审批请求
            approval_request = self._approval_manager.create_request(
                run_id=run_id,
                tool_call=tool_call,
                tool_name=tool_name,
                parameters=parameters,
                risk_level=tool.risk_level,
                user_context=user_context,
                evidence=approval_evidence,
                policy_version=policy_version,
                execution_manifest_hash=execution_manifest_hash,
            )

            # 记录步骤
            self._step_logger.log_step(
                run_id=run_id,
                node_name="tool_approval_requested",
                input_data={"tool_name": tool_name, "parameters": parameters},
                output_data={
                    "tool_call_id": tool_call_id,
                    "approval_request_id": approval_request.id,
                    "requires_approval": True,
                },
            )

            logger.info(
                "tool_requires_approval",
                extra={
                    "run_id": run_id,
                    "tool_name": tool_name,
                    "approval_request_id": approval_request.id,
                },
            )

            return tool_call

        # 不需要审批，直接执行
        handler = self._registry.get_handler(tool_name)
        try:
            result = await handler.execute(parameters, user_context)

            tool_call = ToolCall(
                id=tool_call_id,
                run_id=run_id,
                tool_name=tool_name,
                parameters=parameters,
                result=result,
                status=ToolCallStatus.COMPLETED,
                approval_required=False,
            )

            # 记录步骤
            self._step_logger.log_step(
                run_id=run_id,
                node_name="tool_executed",
                input_data={"tool_name": tool_name, "parameters": parameters},
                output_data={"result": result, "tool_call_id": tool_call_id},
            )

            logger.info(
                "tool_executed_successfully",
                extra={"run_id": run_id, "tool_name": tool_name, "tool_call_id": tool_call_id},
            )

            return tool_call

        except Exception as e:
            # 执行失败
            tool_call = ToolCall(
                id=tool_call_id,
                run_id=run_id,
                tool_name=tool_name,
                parameters=parameters,
                result={"error": str(e)},
                status=ToolCallStatus.FAILED,
                approval_required=False,
            )

            # 记录步骤
            self._step_logger.log_step(
                run_id=run_id,
                node_name="tool_execution_failed",
                input_data={"tool_name": tool_name, "parameters": parameters},
                output_data={"error": str(e), "tool_call_id": tool_call_id},
            )

            logger.error(
                "tool_execution_failed",
                extra={"run_id": run_id, "tool_name": tool_name, "error": str(e)},
            )

            return tool_call

    async def execute_after_approval(
        self, run_id: str, tool_call_id: str, approval_id: str, user_context: UserContext
    ) -> ToolCall:
        """
        审批后执行工具。

        Args:
            run_id: Run ID
            tool_call_id: 工具调用 ID
            approval_id: 审批请求 ID
            user_context: 用户上下文

        Returns:
            工具调用记录
        """
        logger.info(
            "executing_tool_after_approval",
            extra={"run_id": run_id, "tool_call_id": tool_call_id, "approval_id": approval_id},
        )

        # 获取审批请求
        approval_request = self._approval_manager.get_request(approval_id)

        # 审批通过不等于越权执行，恢复时仍需校验工具权限
        tool = self._registry.get_tool(approval_request.tool_name)
        self._ensure_tool_permission(run_id, tool.name, tool.permission_scope, user_context)

        # 使用审批后的参数
        parameters = approval_request.parameters
        handler = self._registry.get_handler(approval_request.tool_name)
        self._approval_manager.validate_for_execution(
            approval_id,
            tool_name=approval_request.tool_name,
            parameters=parameters,
            evidence=approval_request.evidence,
            policy_version=approval_request.policy_version,
            execution_manifest_hash=approval_request.execution_manifest_hash,
        )
        effect_record = await self._side_effect_ledger.reserve(
            idempotency_key=f"{run_id}:{tool_call_id}",
            tool_name=approval_request.tool_name,
            subject_hash=approval_request.subject_hash,
        )
        if effect_record.status == SideEffectStatus.SUCCEEDED:
            return ToolCall(
                id=tool_call_id,
                run_id=run_id,
                tool_name=approval_request.tool_name,
                parameters=parameters,
                result=effect_record.result,
                status=ToolCallStatus.COMPLETED,
                approval_required=True,
            )
        if effect_record.status == SideEffectStatus.UNKNOWN:
            return ToolCall(
                id=tool_call_id,
                run_id=run_id,
                tool_name=approval_request.tool_name,
                parameters=parameters,
                result={"error": "side effect result is unknown; reconciliation required"},
                status=ToolCallStatus.FAILED,
                approval_required=True,
            )

        try:
            result = await handler.execute(parameters, user_context)
            await self._side_effect_ledger.mark_succeeded(effect_record.id, result)

            # 更新 ToolCall 状态
            tool_call = ToolCall(
                id=tool_call_id,
                run_id=run_id,
                tool_name=approval_request.tool_name,
                parameters=parameters,
                result=result,
                status=ToolCallStatus.COMPLETED,
                approval_required=True,
            )

            # 记录步骤
            self._step_logger.log_step(
                run_id=run_id,
                node_name="tool_executed_after_approval",
                input_data={
                    "tool_name": approval_request.tool_name,
                    "parameters": parameters,
                    "approval_id": approval_id,
                },
                output_data={"result": result, "tool_call_id": tool_call_id},
            )

            logger.info(
                "tool_executed_after_approval",
                extra={"run_id": run_id, "tool_name": approval_request.tool_name},
            )

            return tool_call

        except Exception as e:
            await self._side_effect_ledger.mark_unknown(effect_record.id, str(e))
            # 执行失败
            tool_call = ToolCall(
                id=tool_call_id,
                run_id=run_id,
                tool_name=approval_request.tool_name,
                parameters=parameters,
                result={"error": str(e)},
                status=ToolCallStatus.FAILED,
                approval_required=True,
            )

            logger.error(
                "tool_execution_after_approval_failed", extra={"run_id": run_id, "error": str(e)}
            )

            return tool_call

    def _ensure_tool_permission(
        self, run_id: str, tool_name: str, permission_scope: str, user_context: UserContext
    ) -> None:
        """校验用户是否具备工具所需权限。"""
        tool = self._registry.get_tool(tool_name)
        if self._acl_validator.validate_tool_permission(tool, user_context):
            return

        self._step_logger.log_step(
            run_id=run_id,
            node_name="tool_permission_denied",
            input_data={
                "tool_name": tool_name,
                "user_id": user_context.user_id,
                "required_scope": permission_scope,
            },
            output_data={"allowed": False},
        )
        raise AppPermissionError(
            f"User {user_context.user_id} lacks permission {permission_scope} for tool {tool_name}"
        )
