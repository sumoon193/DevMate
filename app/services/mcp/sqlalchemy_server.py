"""PostgreSQL/SQLAlchemy-backed MCP 工具服务。"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.runtime import HrTicketRecord
from app.schemas.enums import ToolRiskLevel
from app.schemas.tool import ToolDefinition
from app.schemas.user import UserContext


class SqlAlchemyMcpServer:
    """把 MCP 写工具的业务结果持久化到受租户隔离的数据库表。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    def list_tools(self) -> list[ToolDefinition]:
        return [
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
                    "properties": {"department_id": {"type": "string"}},
                },
            ),
            ToolDefinition(
                # 兼容既有工作流合同；实现已不再是 Fake，而是数据库持久化写入。
                name="create_hr_ticket",
                description="创建并持久化内部 HR 工单",
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
                        "priority": {"type": "string"},
                        "category": {"type": "string"},
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
                    "properties": {"run_id": {"type": "string"}},
                },
            ),
        ]

    async def call_tool(
        self,
        name: str,
        parameters: dict[str, Any],
        user_context: UserContext,
    ) -> dict[str, Any]:
        if name == "list_hr_policy_documents":
            department_id = str(parameters.get("department_id", "dept_hr"))
            return {
                "documents": [
                    {
                        "document_id": "doc_onboarding",
                        "title": "员工入职与转正管理制度",
                        "department_id": department_id,
                    }
                ]
            }
        if name == "create_hr_ticket":
            record = HrTicketRecord(
                id=f"TK-{uuid.uuid4().hex[:12].upper()}",
                tenant_id=user_context.tenant_id,
                title=str(parameters["title"]),
                description=str(parameters["description"]),
                priority=str(parameters.get("priority", "medium")),
                category=str(parameters.get("category", "其他")),
                status="created",
                created_by=user_context.user_id,
            )
            async with self._sessions() as session, session.begin():
                session.add(record)
            return {
                "ticket_id": record.id,
                "tenant_id": record.tenant_id,
                "title": record.title,
                "description": record.description,
                "priority": record.priority,
                "category": record.category,
                "status": record.status,
                "created_by": record.created_by,
            }
        if name == "summarize_agent_run_artifacts":
            return {
                "run_id": str(parameters["run_id"]),
                "summary": "该 Agent Run 包含 evidence、plan、approval、tool result 和 trace。",
            }
        raise RuntimeError(f"unknown MCP tool: {name}")
