"""
MCP 风格工具 adapter。

所有工具调用仍经过 ToolRegistry、ToolExecutor 和 ApprovalManager。
"""

from __future__ import annotations

from typing import Any, Protocol

from app.core.exceptions import ValidationError
from app.schemas.tool import ToolCall, ToolDefinition
from app.schemas.user import UserContext
from app.services.agent.approval_manager import ApprovalManager
from app.services.agent.tool_executor import ToolExecutor
from app.services.agent.tool_registry import ToolRegistry


class McpServer(Protocol):
    """MCP 工具发现与执行边界；生产和离线实现必须显式装配。"""

    def list_tools(self) -> list[ToolDefinition]: ...

    async def call_tool(
        self,
        name: str,
        parameters: dict[str, Any],
        user_context: UserContext,
    ) -> dict[str, Any]: ...


class McpToolDiscovery:
    """从 MCP server 发现工具定义。"""

    def __init__(self, server: McpServer) -> None:
        self._server = server

    def discover_tools(self) -> list[ToolDefinition]:
        """发现 MCP 工具列表。"""
        return self._server.list_tools()

    @property
    def server(self) -> McpServer:
        """返回底层 MCP server。"""
        return self._server


class McpToolHandler:
    """ToolExecutor 可调用的 MCP 工具处理器。"""

    def __init__(self, server: McpServer, tool_name: str) -> None:
        self._server = server
        self._tool_name = tool_name

    async def execute(
        self,
        parameters: dict[str, Any],
        user_context: UserContext,
    ) -> dict[str, Any]:
        """调用 MCP server。"""
        return await self._server.call_tool(self._tool_name, parameters, user_context)


class McpResultNormalizer:
    """MCP 结果归一化。"""

    def normalize_error(self, message: str) -> dict[str, Any]:
        """归一化错误结果。"""
        return {"error": message}


class McpToolAdapter:
    """将 MCP 风格工具纳入现有 Harness 执行路径。"""

    def __init__(
        self,
        discovery: McpToolDiscovery,
        registry: ToolRegistry,
        tool_executor: ToolExecutor,
    ) -> None:
        self._discovery = discovery
        self._registry = registry
        self._tool_executor = tool_executor

    def discover_tools(self) -> list[ToolDefinition]:
        """发现工具列表。"""
        return self._discovery.discover_tools()

    def register_discovered_tools(self) -> None:
        """把 MCP 工具注册到 ToolRegistry。"""
        for tool in self.discover_tools():
            self._registry.register(
                tool,
                McpToolHandler(self._discovery.server, tool.name),
            )

    async def call(
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
        调用 MCP 工具。

        schema 校验通过后，所有调用交给 ToolExecutor 处理权限和审批。
        """
        tool = self._registry.get_tool(tool_name)
        self._validate_parameters(tool, parameters)
        return await self._tool_executor.execute(
            run_id=run_id,
            tool_name=tool_name,
            parameters=parameters,
            user_context=user_context,
            approval_evidence=approval_evidence,
            policy_version=policy_version,
            execution_manifest_hash=execution_manifest_hash,
        )

    def _validate_parameters(
        self,
        tool: ToolDefinition,
        parameters: dict[str, Any],
    ) -> None:
        """执行轻量 JSON Schema 校验。"""
        schema = tool.parameters_schema
        required = schema.get("required", [])
        if isinstance(required, list):
            for field_name in required:
                if isinstance(field_name, str) and field_name not in parameters:
                    raise ValidationError(f"Missing required MCP parameter: {field_name}")

        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return

        for field_name, field_schema in properties.items():
            if field_name not in parameters or not isinstance(field_schema, dict):
                continue
            expected_type = field_schema.get("type")
            if not self._matches_type(parameters[field_name], expected_type):
                raise ValidationError(
                    f"Invalid MCP parameter type for {field_name}: expected {expected_type}"
                )

    def _matches_type(self, value: object, expected_type: object) -> bool:
        """检查 JSON Schema 基础类型。"""
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "number":
            return isinstance(value, int | float) and not isinstance(value, bool)
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type == "object":
            return isinstance(value, dict)
        if expected_type == "array":
            return isinstance(value, list)
        return True


class McpApprovalBridge:
    """把 MCP 写工具恢复执行接入现有 approval gate。"""

    def __init__(
        self,
        tool_executor: ToolExecutor,
        approval_manager: ApprovalManager,
    ) -> None:
        self._tool_executor = tool_executor
        self._approval_manager = approval_manager

    async def execute_approved(
        self,
        run_id: str,
        approval_id: str,
        user_context: UserContext,
    ) -> ToolCall:
        """审批通过后执行 MCP 写工具。"""
        approval = self._approval_manager.get_request(approval_id)
        return await self._tool_executor.execute_after_approval(
            run_id=run_id,
            tool_call_id=approval.tool_call_id,
            approval_id=approval_id,
            user_context=user_context,
        )
