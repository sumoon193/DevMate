"""
Agent Harness 测试。

按模块规范要求的 6 个测试：
1. test_write_tool_requires_approval_before_execution
2. test_read_tool_executes_without_approval
3. test_approval_resume_keeps_same_run_id
4. test_reject_records_audit_step
5. test_invalid_status_transition_is_rejected
6. test_tool_registry_refuses_unregistered_tool
"""

from __future__ import annotations

import pytest

from app.core.exceptions import NotFoundError, PermissionError, ValidationError
from app.schemas.enums import RunStatus, ToolRiskLevel
from app.schemas.tool import ToolDefinition
from app.schemas.user import UserContext
from app.services.agent.approval_manager import ApprovalManager
from app.services.agent.run_manager import AgentRunManager
from app.services.agent.state_machine import AgentStateMachine
from app.services.agent.step_logger import StepLogger
from app.services.agent.tool_executor import ToolExecutor
from app.services.agent.tool_registry import ToolRegistry
from app.services.agent.tools.clarification import ClarificationHandler
from app.services.agent.tools.hr_checklist import HRChecklistHandler
from app.services.agent.tools.mock_ticket import MockTicketHandler
from app.services.agent.tools.policy_search import PolicySearchHandler
from app.services.agent.tools.user_profile import UserProfileHandler


@pytest.fixture
def user_context() -> UserContext:
    """用户上下文。"""
    return UserContext(
        user_id="user_001",
        tenant_id="tenant_hr",
        department_ids=["dept_001"],
        role="hr",
        permissions=["hr.document.read", "hr.ticket.write"],
    )


@pytest.fixture
def step_logger() -> StepLogger:
    """Step Logger。"""
    return StepLogger()


@pytest.fixture
def approval_manager(step_logger: StepLogger) -> ApprovalManager:
    """Approval Manager。"""
    return ApprovalManager(step_logger)


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Tool Registry（注册所有 V1 工具）。"""
    registry = ToolRegistry()

    # 注册读取型工具
    registry.register(
        ToolDefinition(
            name="policy_search",
            description="检索制度证据",
            permission_scope="hr.document.read",
            risk_level=ToolRiskLevel.READ,
            requires_approval=False,
            timeout_seconds=10,
            idempotent=True,
            parameters_schema={"query": {"type": "string"}},
        ),
        PolicySearchHandler(),
    )

    registry.register(
        ToolDefinition(
            name="get_user_profile",
            description="查询用户档案",
            permission_scope="hr.user.read",
            risk_level=ToolRiskLevel.READ,
            requires_approval=False,
            timeout_seconds=5,
            idempotent=True,
            parameters_schema={"user_id": {"type": "string"}},
        ),
        UserProfileHandler(),
    )

    registry.register(
        ToolDefinition(
            name="generate_hr_checklist",
            description="生成 HR 清单",
            permission_scope="hr.checklist.read",
            risk_level=ToolRiskLevel.READ,
            requires_approval=False,
            timeout_seconds=5,
            idempotent=True,
            parameters_schema={"scenario": {"type": "string"}},
        ),
        HRChecklistHandler(),
    )

    registry.register(
        ToolDefinition(
            name="ask_clarification",
            description="生成澄清问题",
            permission_scope="hr.question.read",
            risk_level=ToolRiskLevel.READ,
            requires_approval=False,
            timeout_seconds=5,
            idempotent=True,
            parameters_schema={"question": {"type": "string"}},
        ),
        ClarificationHandler(),
    )

    # 注册写入型工具
    registry.register(
        ToolDefinition(
            name="create_hr_ticket",
            description="创建模拟 HR 工单",
            permission_scope="hr.ticket.write",
            risk_level=ToolRiskLevel.WRITE,
            requires_approval=True,
            timeout_seconds=10,
            idempotent=True,
            parameters_schema={"title": {"type": "string"}, "description": {"type": "string"}},
        ),
        MockTicketHandler(),
    )

    return registry


@pytest.fixture
def tool_executor(
    tool_registry: ToolRegistry, approval_manager: ApprovalManager, step_logger: StepLogger
) -> ToolExecutor:
    """Tool Executor。"""
    return ToolExecutor(tool_registry, approval_manager, step_logger)


@pytest.fixture
def run_manager(
    tool_executor: ToolExecutor, approval_manager: ApprovalManager, step_logger: StepLogger
) -> AgentRunManager:
    """Agent Run Manager。"""
    return AgentRunManager(tool_executor, approval_manager, step_logger)


class TestToolRegistry:
    """Tool Registry 测试。"""

    def test_tool_registry_refuses_unregistered_tool(self, tool_registry: ToolRegistry):
        """测试 6：Tool Registry 拒绝未注册的工具。"""
        with pytest.raises(NotFoundError):
            tool_registry.get_tool("non_existent_tool")

        with pytest.raises(NotFoundError):
            tool_registry.get_handler("non_existent_tool")

        assert tool_registry.has_tool("non_existent_tool") is False

    def test_tool_registry_lists_tools(self, tool_registry: ToolRegistry):
        """测试 Tool Registry 列出所有工具。"""
        tools = tool_registry.list_tools()
        assert len(tools) == 5

        read_tools = tool_registry.get_read_tools()
        assert len(read_tools) == 4

        write_tools = tool_registry.get_write_tools()
        assert len(write_tools) == 1


class TestToolExecution:
    """工具执行测试。"""

    @pytest.mark.asyncio
    async def test_read_tool_executes_without_approval(
        self, run_manager: AgentRunManager, user_context: UserContext
    ):
        """测试 2：读取型工具无需审批即可执行。"""
        # 创建 Run
        run = await run_manager.create_run("测试查询", user_context)
        await run_manager.start_run(run.id)

        # 执行读取型工具
        tool_call = await run_manager.execute_tool(
            run_id=run.id,
            tool_name="policy_search",
            parameters={"query": "入职材料"},
            user_context=user_context,
        )

        # 验证
        assert tool_call.approval_required is False
        assert tool_call.status == "completed"
        assert tool_call.result is not None
        assert "citations" in tool_call.result

    @pytest.mark.asyncio
    async def test_run_records_tool_call_history(
        self, run_manager: AgentRunManager, user_context: UserContext
    ):
        """测试：Agent Run 应记录已触发的工具调用，供 API 详情展示。"""
        run = await run_manager.create_run("测试查询", user_context)
        await run_manager.start_run(run.id)

        tool_call = await run_manager.execute_tool(
            run_id=run.id,
            tool_name="policy_search",
            parameters={"query": "入职材料"},
            user_context=user_context,
        )

        updated_run = await run_manager.get_run(run.id)
        assert len(updated_run.tool_calls) == 1
        assert updated_run.tool_calls[0].id == tool_call.id
        assert updated_run.tool_calls[0].tool_name == "policy_search"

    @pytest.mark.asyncio
    async def test_write_tool_requires_approval_before_execution(
        self, run_manager: AgentRunManager, user_context: UserContext
    ):
        """测试 1：写入型工具必须先审批后执行。"""
        # 创建 Run
        run = await run_manager.create_run("创建工单", user_context)
        await run_manager.start_run(run.id)

        # 执行写入型工具
        tool_call = await run_manager.execute_tool(
            run_id=run.id,
            tool_name="create_hr_ticket",
            parameters={"title": "入职申请", "description": "新员工入职"},
            user_context=user_context,
        )

        # 验证
        assert tool_call.approval_required is True
        assert tool_call.status == "pending"
        assert tool_call.result is None

        # 验证 Run 状态变为 AWAITING_APPROVAL
        updated_run = await run_manager.get_run(run.id)
        assert updated_run.status == RunStatus.AWAITING_APPROVAL

    @pytest.mark.asyncio
    async def test_tool_execution_requires_permission_scope(self, run_manager: AgentRunManager):
        """测试：工具执行路径必须校验用户权限，未授权时不能创建审批。"""
        unauthorized_user = UserContext(
            user_id="user_no_write",
            tenant_id="tenant_hr",
            department_ids=["dept_001"],
            role="employee",
            permissions=["hr.document.read"],
        )
        run = await run_manager.create_run("创建工单", unauthorized_user)
        await run_manager.start_run(run.id)

        with pytest.raises(PermissionError):
            await run_manager.execute_tool(
                run_id=run.id,
                tool_name="create_hr_ticket",
                parameters={"title": "入职申请", "description": "新员工入职"},
                user_context=unauthorized_user,
            )

        pending_approvals = await run_manager.get_pending_approvals(run.id)
        assert pending_approvals == []

        updated_run = await run_manager.get_run(run.id)
        assert updated_run.status == RunStatus.RUNNING


class TestApprovalFlow:
    """审批流程测试。"""

    @pytest.mark.asyncio
    async def test_approval_resume_keeps_same_run_id(
        self,
        run_manager: AgentRunManager,
        approval_manager: ApprovalManager,
        user_context: UserContext,
    ):
        """测试 3：审批后恢复保持相同的 Run ID。"""
        # 创建 Run
        run = await run_manager.create_run("创建工单", user_context)
        await run_manager.start_run(run.id)

        # 执行写入型工具
        await run_manager.execute_tool(
            run_id=run.id,
            tool_name="create_hr_ticket",
            parameters={"title": "入职申请", "description": "新员工入职"},
            user_context=user_context,
        )

        # 获取待审批请求
        pending_approvals = await run_manager.get_pending_approvals(run.id)
        assert len(pending_approvals) == 1

        approval_id = pending_approvals[0].id

        # 审批通过
        approval_manager.approve(approval_id, "admin")

        # 恢复执行
        resumed_run = await run_manager.resume_after_approval(
            run_id=run.id, approval_id=approval_id, user_context=user_context
        )

        # 验证 Run ID 不变
        assert resumed_run.id == run.id

    @pytest.mark.asyncio
    async def test_reject_records_audit_step(
        self,
        run_manager: AgentRunManager,
        approval_manager: ApprovalManager,
        step_logger: StepLogger,
        user_context: UserContext,
    ):
        """测试 4：拒绝后记录审计步骤。"""
        # 创建 Run
        run = await run_manager.create_run("创建工单", user_context)
        await run_manager.start_run(run.id)

        # 执行写入型工具
        await run_manager.execute_tool(
            run_id=run.id,
            tool_name="create_hr_ticket",
            parameters={"title": "入职申请", "description": "新员工入职"},
            user_context=user_context,
        )

        # 获取待审批请求
        pending_approvals = await run_manager.get_pending_approvals(run.id)
        approval_id = pending_approvals[0].id

        # 拒绝审批
        approval_manager.reject(approval_id, "admin")

        # 验证审计步骤
        steps = step_logger.get_steps(run.id)
        rejection_steps = [s for s in steps if s.node_name == "approval_rejected"]
        assert len(rejection_steps) == 1

        step = rejection_steps[0]
        assert step.output_data["decision"] == "reject"
        assert step.output_data["decided_by"] == "admin"

    @pytest.mark.asyncio
    async def test_approve_records_audit_step(
        self,
        run_manager: AgentRunManager,
        approval_manager: ApprovalManager,
        step_logger: StepLogger,
        user_context: UserContext,
    ):
        """测试：审批通过也必须记录审计步骤。"""
        run = await run_manager.create_run("创建工单", user_context)
        await run_manager.start_run(run.id)

        await run_manager.execute_tool(
            run_id=run.id,
            tool_name="create_hr_ticket",
            parameters={"title": "入职申请", "description": "新员工入职"},
            user_context=user_context,
        )

        pending_approvals = await run_manager.get_pending_approvals(run.id)
        approval_id = pending_approvals[0].id

        approval_manager.approve(approval_id, "admin")

        steps = step_logger.get_steps(run.id)
        approval_steps = [s for s in steps if s.node_name == "approval_approved"]
        assert len(approval_steps) == 1

        step = approval_steps[0]
        assert step.output_data["decision"] == "approve"
        assert step.output_data["decided_by"] == "admin"


class TestStateMachine:
    """状态机测试。"""

    def test_invalid_status_transition_is_rejected(self):
        """测试 5：非法状态流转被拒绝。"""
        state_machine = AgentStateMachine()

        # 合法的流转
        assert state_machine.validate_transition(RunStatus.CREATED, RunStatus.RUNNING) is True

        # 非法的流转
        with pytest.raises(ValidationError):
            state_machine.validate_transition(RunStatus.CREATED, RunStatus.COMPLETED)

        with pytest.raises(ValidationError):
            state_machine.validate_transition(RunStatus.COMPLETED, RunStatus.RUNNING)

    def test_allowed_transitions(self):
        """测试获取允许的状态流转。"""
        state_machine = AgentStateMachine()

        allowed = state_machine.get_allowed_transitions(RunStatus.CREATED)
        assert allowed == [RunStatus.RUNNING]

        allowed = state_machine.get_allowed_transitions(RunStatus.PLANNING)
        assert RunStatus.AWAITING_APPROVAL in allowed
        assert RunStatus.COMPLETED in allowed

    def test_terminal_states(self):
        """测试终态检测。"""
        state_machine = AgentStateMachine()

        assert state_machine.is_terminal(RunStatus.COMPLETED) is True
        assert state_machine.is_terminal(RunStatus.FAILED) is True
        assert state_machine.is_terminal(RunStatus.CANCELLED) is True

        assert state_machine.is_terminal(RunStatus.CREATED) is False
        assert state_machine.is_terminal(RunStatus.RUNNING) is False


class TestAgentRunLifecycle:
    """Agent Run 生命周期测试。"""

    @pytest.mark.asyncio
    async def test_run_lifecycle_complete(
        self, run_manager: AgentRunManager, user_context: UserContext
    ):
        """测试完整的 Run 生命周期。"""
        # 创建 Run
        run = await run_manager.create_run("新员工入职需要提交哪些材料？", user_context)
        assert run.status == RunStatus.CREATED

        # 启动 Run
        run = await run_manager.start_run(run.id)
        assert run.status == RunStatus.RUNNING

        # 模拟检索证据
        from app.schemas.chunk import Citation, EvidenceBundle

        evidence = EvidenceBundle(
            evidence_list=[
                Citation(
                    id=1,
                    document_name="员工入职制度",
                    section="第二章",
                    page=3,
                    chunk_text="入职材料",
                    score=0.9,
                    rerank_score=0.95,
                )
            ],
            total_count=1,
            query_coverage_score=0.9,
        )
        run = await run_manager.retrieve_evidence(run.id, evidence)
        assert run.status == RunStatus.RETRIEVING_EVIDENCE

        # 创建计划
        from app.schemas.agent import AgentPlan

        plan = AgentPlan(
            id="plan_001",
            run_id=run.id,
            steps=["policy_search", "generate_hr_checklist"],
            current_step_index=0,
        )
        run = await run_manager.create_plan(run.id, plan)
        assert run.status == RunStatus.PLANNING

        # 执行读取型工具
        tool_call = await run_manager.execute_tool(
            run_id=run.id,
            tool_name="policy_search",
            parameters={"query": "入职材料"},
            user_context=user_context,
        )
        assert tool_call.status == "completed"

        # 完成 Run
        result = {
            "answer": "新员工入职需要提交身份证复印件、学历证明和离职证明。",
            "citations": tool_call.result.get("citations", []),
        }
        run = await run_manager.complete_run(run.id, result)
        assert run.status == RunStatus.COMPLETED
        assert run.result == result

    @pytest.mark.asyncio
    async def test_run_with_approval_flow(
        self,
        run_manager: AgentRunManager,
        approval_manager: ApprovalManager,
        user_context: UserContext,
    ):
        """测试带审批流程的 Run。"""
        # 创建 Run
        run = await run_manager.create_run("帮我创建入职工单", user_context)
        await run_manager.start_run(run.id)

        # 执行写入型工具（需要审批）
        tool_call = await run_manager.execute_tool(
            run_id=run.id,
            tool_name="create_hr_ticket",
            parameters={"title": "入职申请", "description": "新员工入职"},
            user_context=user_context,
        )
        assert tool_call.approval_required is True

        # 验证 Run 状态
        run = await run_manager.get_run(run.id)
        assert run.status == RunStatus.AWAITING_APPROVAL

        # 审批通过
        pending_approvals = await run_manager.get_pending_approvals(run.id)
        approval_manager.approve(pending_approvals[0].id, "admin")

        # 恢复执行
        run = await run_manager.resume_after_approval(
            run_id=run.id, approval_id=pending_approvals[0].id, user_context=user_context
        )
        assert run.status == RunStatus.RESUMED

        # 完成 Run
        run = await run_manager.complete_run(run.id, {"message": "工单已创建"})
        assert run.status == RunStatus.COMPLETED
