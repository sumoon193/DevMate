"""
本地 fake MCP server。

用于在不依赖远程 MCP 生态的情况下验证工具发现和调用治理。
"""

from __future__ import annotations

import hashlib
from typing import Any

from app.schemas.enums import ToolRiskLevel
from app.schemas.tool import ToolDefinition
from app.schemas.user import UserContext


class FakeMcpServer:
    """确定性的 fake MCP server。"""

    def __init__(self) -> None:
        self._calls: dict[str, int] = {}
        self._fail_next: dict[str, str] = {}
        self._tools = [
            ToolDefinition(
                name="list_hr_policy_documents",
                description="列出 HR 制度文档",
                permission_scope="hr.document.read",
                risk_level=ToolRiskLevel.READ,
                requires_approval=False,
                timeout_seconds=5,
                idempotent=True,
                parameters_schema={
                    "type": "object",
                    "required": [],
                    "properties": {
                        "department_id": {"type": "string"},
                    },
                },
            ),
            ToolDefinition(
                name="create_hr_ticket",
                description="通过 MCP 风格工具创建模拟 HR 工单",
                permission_scope="hr.ticket.write",
                risk_level=ToolRiskLevel.WRITE,
                requires_approval=True,
                timeout_seconds=10,
                idempotent=True,
                parameters_schema={
                    "type": "object",
                    "required": ["title", "description"],
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            ),
            ToolDefinition(
                name="summarize_agent_run_artifacts",
                description="汇总 Agent Run artifacts",
                permission_scope="agent.artifact.read",
                risk_level=ToolRiskLevel.READ,
                requires_approval=False,
                timeout_seconds=5,
                idempotent=True,
                parameters_schema={
                    "type": "object",
                    "required": ["run_id"],
                    "properties": {
                        "run_id": {"type": "string"},
                    },
                },
            ),
        ]

    def list_tools(self) -> list[ToolDefinition]:
        """返回 server 暴露的工具 schema。"""
        return self._tools.copy()

    async def call_tool(
        self,
        name: str,
        parameters: dict[str, Any],
        user_context: UserContext,
    ) -> dict[str, Any]:
        """执行 fake MCP 工具。"""
        self._calls[name] = self._calls.get(name, 0) + 1
        if name in self._fail_next:
            message = self._fail_next.pop(name)
            raise RuntimeError(message)

        if name == "list_hr_policy_documents":
            department_id = parameters.get("department_id", "dept_hr")
            return {
                "documents": [
                    {
                        "document_id": "doc_onboarding",
                        "title": "员工入职与转正管理制度",
                        "department_id": department_id,
                    },
                    {
                        "document_id": "doc_reimbursement",
                        "title": "财务报销制度",
                        "department_id": department_id,
                    },
                ]
            }

        if name == "create_hr_ticket":
            title = str(parameters["title"])
            digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:5].upper()
            return {
                "ticket_id": f"MCP-TK-{digest}",
                "title": title,
                "description": parameters["description"],
                "created_by": user_context.user_id,
            }

        if name == "summarize_agent_run_artifacts":
            return {
                "run_id": parameters["run_id"],
                "summary": "该 Agent Run 包含 evidence、plan、approval、tool result 和 trace。",
            }

        raise RuntimeError(f"unknown MCP tool: {name}")

    def fail_next(self, name: str, message: str) -> None:
        """让下一次指定工具调用失败。"""
        self._fail_next[name] = message

    def call_count(self, name: str) -> int:
        """返回指定工具被 fake server 调用的次数。"""
        return self._calls.get(name, 0)
