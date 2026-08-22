"""
MCP 风格本地 adapter 测试。

确保 MCP 工具发现、schema 校验、调用、审批仍经过 Harness 治理。
"""

from __future__ import annotations

import pytest

from app.core.exceptions import ValidationError
from app.schemas.enums import ToolCallStatus
from app.schemas.user import UserContext
from app.services.agent.approval_manager import ApprovalManager
from app.services.agent.step_logger import StepLogger
from app.services.agent.tool_executor import ToolExecutor
from app.services.agent.tool_registry import ToolRegistry
from app.services.mcp.adapter import McpApprovalBridge, McpToolAdapter, McpToolDiscovery
from app.services.mcp.fake_server import FakeMcpServer


@pytest.fixture
def user_context() -> UserContext:
    """具备 HR 读写权限的用户。"""
    return UserContext(
        user_id="user_001",
        tenant_id="tenant_001",
        department_ids=["dept_hr"],
        role="employee",
        permissions=[
            "hr.document.read",
            "hr.ticket.write",
            "agent.artifact.read",
        ],
    )


@pytest.fixture
def mcp_stack() -> tuple[
    FakeMcpServer,
    McpToolAdapter,
    McpApprovalBridge,
    ApprovalManager,
]:
    """构建 fake MCP server + Harness adapter。"""
    server = FakeMcpServer()
    discovery = McpToolDiscovery(server)
    registry = ToolRegistry()
    step_logger = StepLogger()
    approval_manager = ApprovalManager(step_logger)
    tool_executor = ToolExecutor(registry, approval_manager, step_logger)
    adapter = McpToolAdapter(discovery, registry, tool_executor)
    adapter.register_discovered_tools()
    bridge = McpApprovalBridge(tool_executor, approval_manager)
    return server, adapter, bridge, approval_manager


def test_fake_mcp_server_discovers_tool_schemas(
    mcp_stack: tuple[FakeMcpServer, McpToolAdapter, McpApprovalBridge, ApprovalManager],
) -> None:
    """fake MCP server 应返回三类第一阶段工具。"""
    _, adapter, _, _ = mcp_stack

    tools = adapter.discover_tools()

    assert [tool.name for tool in tools] == [
        "list_hr_policy_documents",
        "create_hr_ticket",
        "summarize_agent_run_artifacts",
    ]
    assert tools[1].requires_approval is True


@pytest.mark.asyncio
async def test_schema_mismatch_is_rejected_before_server_call(
    mcp_stack: tuple[FakeMcpServer, McpToolAdapter, McpApprovalBridge, ApprovalManager],
    user_context: UserContext,
) -> None:
    """schema 不匹配时应拒绝调用，且 fake server 不收到请求。"""
    server, adapter, _, _ = mcp_stack

    with pytest.raises(ValidationError):
        await adapter.call(
            run_id="run_mcp_001",
            tool_name="create_hr_ticket",
            parameters={"title": 123, "description": "入职"},
            user_context=user_context,
        )

    assert server.call_count("create_hr_ticket") == 0


@pytest.mark.asyncio
async def test_write_tool_is_blocked_by_approval_gate(
    mcp_stack: tuple[FakeMcpServer, McpToolAdapter, McpApprovalBridge, ApprovalManager],
    user_context: UserContext,
) -> None:
    """写工具未审批前只能创建 approval preview，不能调用 fake MCP server。"""
    server, adapter, _, approval_manager = mcp_stack

    tool_call = await adapter.call(
        run_id="run_mcp_002",
        tool_name="create_hr_ticket",
        parameters={"title": "入职工单", "description": "新员工入职"},
        user_context=user_context,
    )

    assert tool_call.status == ToolCallStatus.PENDING
    assert tool_call.approval_required is True
    assert server.call_count("create_hr_ticket") == 0
    assert len(approval_manager.get_pending_requests("run_mcp_002")) == 1


@pytest.mark.asyncio
async def test_approved_write_tool_calls_fake_server_once(
    mcp_stack: tuple[FakeMcpServer, McpToolAdapter, McpApprovalBridge, ApprovalManager],
    user_context: UserContext,
) -> None:
    """审批通过后由 bridge 恢复执行，fake server 只收到一次写调用。"""
    server, adapter, bridge, approval_manager = mcp_stack
    pending_call = await adapter.call(
        run_id="run_mcp_003",
        tool_name="create_hr_ticket",
        parameters={"title": "入职工单", "description": "新员工入职"},
        user_context=user_context,
    )
    approval = approval_manager.get_pending_requests("run_mcp_003")[0]
    approval_manager.approve(approval.id, "admin")

    executed = await bridge.execute_approved(
        run_id="run_mcp_003",
        approval_id=approval.id,
        user_context=user_context,
    )

    assert pending_call.id == executed.id
    assert executed.status == ToolCallStatus.COMPLETED
    assert executed.result is not None
    assert executed.result["ticket_id"].startswith("MCP-TK-")
    assert server.call_count("create_hr_ticket") == 1


@pytest.mark.asyncio
async def test_server_failure_is_normalized_as_failed_tool_call(
    mcp_stack: tuple[FakeMcpServer, McpToolAdapter, McpApprovalBridge, ApprovalManager],
    user_context: UserContext,
) -> None:
    """server 失败应被 ToolExecutor 归一化为 failed tool call。"""
    server, adapter, _, _ = mcp_stack
    server.fail_next("list_hr_policy_documents", "server timeout")

    tool_call = await adapter.call(
        run_id="run_mcp_004",
        tool_name="list_hr_policy_documents",
        parameters={"department_id": "dept_hr"},
        user_context=user_context,
    )

    assert tool_call.status == ToolCallStatus.FAILED
    assert tool_call.result == {"error": "server timeout"}
