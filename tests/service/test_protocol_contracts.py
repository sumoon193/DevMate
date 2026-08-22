"""MCP 2025-11-25 与 A2A read-only 协议边界测试。"""

from __future__ import annotations

import pytest

from app.core.exceptions import ValidationError
from app.schemas.protocol import JsonRpcRequest
from app.schemas.user import UserContext
from app.services.a2a.policy_research import (
    InProcessA2AClient,
    PolicyResearchA2AAgent,
)
from app.services.agent.approval_manager import ApprovalManager
from app.services.agent.step_logger import StepLogger
from app.services.agent.tool_executor import ToolExecutor
from app.services.agent.tool_registry import ToolRegistry
from app.services.mcp.adapter import McpToolAdapter, McpToolDiscovery
from app.services.mcp.fake_server import FakeMcpServer
from app.services.mcp.protocol_server import LocalMcpProtocolServer


def _user() -> UserContext:
    return UserContext(
        user_id="user_hr",
        tenant_id="tenant_a",
        department_ids=["dept_hr"],
        role="hr",
        permissions=["hr.document.read", "hr.ticket.write"],
    )


def _mcp_protocol_server() -> tuple[LocalMcpProtocolServer, FakeMcpServer, ApprovalManager]:
    fake = FakeMcpServer()
    registry = ToolRegistry()
    steps = StepLogger()
    approvals = ApprovalManager(steps)
    executor = ToolExecutor(registry, approvals, steps)
    adapter = McpToolAdapter(McpToolDiscovery(fake), registry, executor)
    adapter.register_discovered_tools()
    server = LocalMcpProtocolServer(
        tool_adapter=adapter,
        resources={
            "policy://hr/onboarding": {
                "name": "员工入职制度",
                "mimeType": "text/markdown",
                "text": "新员工应提交身份证明。",
            }
        },
        prompts={"plan_hr_case": "基于制度证据生成长期 Case 计划，并标注写操作审批。"},
    )
    return server, fake, approvals


@pytest.mark.asyncio
async def test_mcp_initialize_exposes_tools_resources_and_prompts() -> None:
    """MCP 初始化必须声明最新协议版本和三类能力。"""
    server, _, _ = _mcp_protocol_server()

    response = await server.handle(
        JsonRpcRequest(id=1, method="initialize", params={}),
        run_id="run_mcp_protocol",
        user_context=_user(),
    )

    assert response.result["protocolVersion"] == "2025-11-25"
    assert set(response.result["capabilities"]) == {"tools", "resources", "prompts"}


@pytest.mark.asyncio
async def test_mcp_write_tool_returns_structured_pending_approval() -> None:
    """MCP write call 必须经过 Harness，未审批前不能调用 server。"""
    server, fake, approvals = _mcp_protocol_server()

    response = await server.handle(
        JsonRpcRequest(
            id=2,
            method="tools/call",
            params={
                "name": "create_hr_ticket",
                "arguments": {"title": "入职工单", "description": "新员工入职"},
            },
        ),
        run_id="run_mcp_protocol",
        user_context=_user(),
    )

    assert response.result["structuredContent"]["status"] == "pending"
    assert response.result["structuredContent"]["approvalRequired"] is True
    assert fake.call_count("create_hr_ticket") == 0
    assert len(approvals.get_pending_requests("run_mcp_protocol")) == 1


@pytest.mark.asyncio
async def test_a2a_policy_research_returns_evidence_artifact_and_rejects_write_goal() -> None:
    """A2A peer 只能完成制度研究任务，不能获得写入能力。"""
    client = InProcessA2AClient(PolicyResearchA2AAgent())

    card = await client.get_agent_card()
    task = await client.send_message(
        context_id="case_001",
        text="研究当前入职制度需要提交哪些材料",
        user_context=_user(),
    )

    assert card.name == "HR Policy Research Agent"
    assert card.capabilities["writeActions"] is False
    assert task.status == "completed"
    assert task.artifacts[0].metadata["citation_count"] >= 1

    with pytest.raises(ValidationError, match="read-only"):
        await client.send_message(
            context_id="case_001",
            text="创建一个 HR 工单并修改员工状态",
            user_context=_user(),
        )
