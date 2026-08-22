"""
审批策略与自动审批测试。

覆盖：
1. 策略规则（WRITE 自动 / ADMIN 人工 / manual 全人工）
2. RunManager.maybe_auto_approve / auto_approve_and_execute
3. LangGraph 链路策略命中时跳过 interrupt 全程自动执行
"""

from __future__ import annotations

import pytest

from app.schemas.approval import ApprovalDecision
from app.schemas.enums import (
    ApprovalDecisionType,
    ApprovalStatus,
    RunStatus,
    ToolCallStatus,
    ToolRiskLevel,
)
from app.schemas.tool import ToolDefinition
from app.schemas.user import UserContext
from app.services.agent.approval_manager import ApprovalManager
from app.services.agent.approval_policy import (
    POLICY_ENGINE_APPROVER,
    NoopApprovalPolicy,
    RuleBasedApprovalPolicy,
    build_approval_policy,
)
from app.services.agent.run_manager import AgentRunManager
from app.services.agent.step_logger import StepLogger
from app.services.agent.tool_executor import ToolExecutor
from app.services.agent.tool_registry import ToolRegistry
from app.services.agent.tools.mock_ticket import MockTicketHandler
from app.services.agent.tools.policy_search import PolicySearchHandler
from app.services.graph.graph import create_agent_graph
from app.services.graph.runner import GraphRunner
from app.services.graph.sse import SSEEventType
from app.services.retrieval.embedding.mock_embedding import MockEmbedder
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.reranker.mock_reranker import MockReranker
from app.services.retrieval.store.memory_bm25 import InMemoryBM25Store
from app.services.retrieval.store.memory_vector import InMemoryVectorStore

TICKET_PARAMS = {
    "title": "新员工入职工单",
    "description": "帮我创建入职工单",
    "priority": "medium",
    "category": "入职",
}


def _parse_sse_event(event: str) -> dict:
    """解析 SSE 字符串中的 JSON payload。"""
    assert event.startswith("data: ")
    import json

    return json.loads(event.split("data: ", 1)[1])


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
    """Tool Registry（读取 + 写入工具）。"""
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="policy_search",
            description="检索制度证据",
            permission_scope="hr.document.read",
            risk_level=ToolRiskLevel.READ,
            requires_approval=False,
        ),
        PolicySearchHandler(),
    )

    registry.register(
        ToolDefinition(
            name="create_hr_ticket",
            description="创建模拟 HR 工单",
            permission_scope="hr.ticket.write",
            risk_level=ToolRiskLevel.WRITE,
            requires_approval=True,
        ),
        MockTicketHandler(),
    )

    return registry


@pytest.fixture
def tool_executor(
    tool_registry: ToolRegistry,
    approval_manager: ApprovalManager,
    step_logger: StepLogger,
) -> ToolExecutor:
    """Tool Executor。"""
    return ToolExecutor(tool_registry, approval_manager, step_logger)


@pytest.fixture
def auto_policy() -> RuleBasedApprovalPolicy:
    """policy 模式策略：写入自动、管理级人工。"""
    return RuleBasedApprovalPolicy(allow_writes=True, allow_admin=False)


@pytest.fixture
def run_manager(
    tool_executor: ToolExecutor,
    approval_manager: ApprovalManager,
    step_logger: StepLogger,
    auto_policy: RuleBasedApprovalPolicy,
) -> AgentRunManager:
    """注入自动审批策略的 Run Manager。"""
    return AgentRunManager(
        tool_executor,
        approval_manager,
        step_logger,
        approval_policy=auto_policy,
    )


@pytest.fixture
def run_manager_manual(
    tool_executor: ToolExecutor,
    approval_manager: ApprovalManager,
    step_logger: StepLogger,
) -> AgentRunManager:
    """manual 模式 Run Manager：从不自动审批。"""
    return AgentRunManager(
        tool_executor,
        approval_manager,
        step_logger,
        approval_policy=NoopApprovalPolicy(),
    )


@pytest.fixture
def hybrid_retriever() -> HybridRetriever:
    """Hybrid Retriever。"""
    embedder = MockEmbedder(dimension=64)
    vector_store = InMemoryVectorStore()
    bm25_store = InMemoryBM25Store()
    reranker = MockReranker()
    return HybridRetriever(embedder, vector_store, bm25_store, reranker)


@pytest.fixture
def graph(run_manager: AgentRunManager, hybrid_retriever: HybridRetriever):
    """Agent Graph。"""
    return create_agent_graph(run_manager, hybrid_retriever)


@pytest.fixture
def graph_runner(
    graph,
    run_manager: AgentRunManager,
    step_logger: StepLogger,
) -> GraphRunner:
    """Graph Runner。"""
    return GraphRunner(graph, run_manager, step_logger)


async def _create_run_with_pending_approval(
    run_manager: AgentRunManager,
    user_context: UserContext,
) -> tuple[str, str]:
    """创建 Run 并触发写入型工具审批，返回 (run_id, approval_id)。"""
    run = await run_manager.create_run("帮我创建入职工单", user_context)
    await run_manager.start_run(run.id)

    tool_call = await run_manager.execute_tool(
        run_id=run.id,
        tool_name="create_hr_ticket",
        parameters=TICKET_PARAMS,
        user_context=user_context,
    )
    assert tool_call.approval_required

    pending = await run_manager.get_pending_approvals(run.id)
    assert pending, "应当存在待审批请求"
    return run.id, pending[0].id


class TestRuleBasedApprovalPolicy:
    """策略规则测试。"""

    def test_approves_write_when_allowed(self, user_context: UserContext):
        """WRITE 且 allow_writes 时自动 APPROVE。"""
        policy = RuleBasedApprovalPolicy(allow_writes=True)
        decision = policy.evaluate(
            tool_name="create_hr_ticket",
            parameters=TICKET_PARAMS,
            risk_level=ToolRiskLevel.WRITE,
            user_context=user_context,
        )
        assert decision == ApprovalDecisionType.APPROVE

    def test_holds_write_when_disallowed(self, user_context: UserContext):
        """WRITE 但 allow_writes=False 时转人工。"""
        policy = RuleBasedApprovalPolicy(allow_writes=False)
        decision = policy.evaluate(
            tool_name="create_hr_ticket",
            parameters=TICKET_PARAMS,
            risk_level=ToolRiskLevel.WRITE,
            user_context=user_context,
        )
        assert decision is None

    def test_holds_admin_for_human_by_default(self, user_context: UserContext):
        """ADMIN 默认转人工（maker-checker 保护）。"""
        policy = RuleBasedApprovalPolicy(allow_writes=True)
        decision = policy.evaluate(
            tool_name="some_admin_tool",
            parameters={},
            risk_level=ToolRiskLevel.ADMIN,
            user_context=user_context,
        )
        assert decision is None

    def test_approves_admin_when_allow_admin(self, user_context: UserContext):
        """沙箱场景 allow_admin=True 时 ADMIN 也自动。"""
        policy = RuleBasedApprovalPolicy(allow_writes=True, allow_admin=True)
        decision = policy.evaluate(
            tool_name="some_admin_tool",
            parameters={},
            risk_level=ToolRiskLevel.ADMIN,
            user_context=user_context,
        )
        assert decision == ApprovalDecisionType.APPROVE

    def test_read_never_auto_approves(self, user_context: UserContext):
        """READ 不产生审批请求，策略返回 None。"""
        policy = RuleBasedApprovalPolicy(allow_writes=True, allow_admin=True)
        decision = policy.evaluate(
            tool_name="policy_search",
            parameters={},
            risk_level=ToolRiskLevel.READ,
            user_context=user_context,
        )
        assert decision is None


class TestNoopApprovalPolicy:
    """manual 模式策略测试。"""

    def test_never_auto_approves(self, user_context: UserContext):
        """Noop 策略对任何风险等级都不自动审批。"""
        policy = NoopApprovalPolicy()
        for risk in (ToolRiskLevel.READ, ToolRiskLevel.WRITE, ToolRiskLevel.ADMIN):
            assert (
                policy.evaluate(
                    tool_name="tool",
                    parameters={},
                    risk_level=risk,
                    user_context=user_context,
                )
                is None
            )


class TestBuildApprovalPolicy:
    """配置到策略的工厂映射测试。"""

    def test_manual_maps_to_noop(self):
        """manual 模式映射到 Noop（等价原人工流程）。"""
        assert isinstance(build_approval_policy("manual"), NoopApprovalPolicy)

    def test_policy_maps_to_rule_write_auto_admin_human(self, user_context: UserContext):
        """policy 模式：写入自动、管理级人工。"""
        policy = build_approval_policy("policy")
        assert isinstance(policy, RuleBasedApprovalPolicy)
        assert (
            policy.evaluate(
                tool_name="create_hr_ticket",
                parameters=TICKET_PARAMS,
                risk_level=ToolRiskLevel.WRITE,
                user_context=user_context,
            )
            == ApprovalDecisionType.APPROVE
        )
        assert (
            policy.evaluate(
                tool_name="admin_tool",
                parameters={},
                risk_level=ToolRiskLevel.ADMIN,
                user_context=user_context,
            )
            is None
        )

    def test_auto_maps_to_rule_with_admin_flag(self):
        """auto 模式通过 allow_admin 参数控制管理级自动。"""
        assert isinstance(
            build_approval_policy("auto", allow_admin=True),
            RuleBasedApprovalPolicy,
        )


class TestRunManagerAutoApprove:
    """RunManager 自动审批路径测试。"""

    @pytest.mark.asyncio
    async def test_auto_approve_marks_policy_engine_and_audit(
        self,
        run_manager: AgentRunManager,
        approval_manager: ApprovalManager,
        step_logger: StepLogger,
        user_context: UserContext,
    ):
        """策略命中时以 policy_engine 身份批准并写入审计步骤。"""
        run_id, approval_id = await _create_run_with_pending_approval(run_manager, user_context)

        decision = run_manager.maybe_auto_approve(run_id, approval_id, user_context)

        assert decision is not None
        assert decision.decision == ApprovalDecisionType.APPROVE

        request = approval_manager.get_request(approval_id)
        assert request.status == ApprovalStatus.APPROVED
        assert request.decided_by == POLICY_ENGINE_APPROVER
        assert request.decided_at is not None

        steps = step_logger.get_steps(run_id)
        assert any(s.node_name == "approval_auto_approved" for s in steps)

    @pytest.mark.asyncio
    async def test_auto_approve_returns_none_when_policy_misses(
        self,
        run_manager_manual: AgentRunManager,
        approval_manager: ApprovalManager,
        user_context: UserContext,
    ):
        """manual 模式策略未命中，返回 None 且保持 PENDING。"""
        run_id, approval_id = await _create_run_with_pending_approval(
            run_manager_manual, user_context
        )

        decision = run_manager_manual.maybe_auto_approve(run_id, approval_id, user_context)

        assert decision is None
        request = approval_manager.get_request(approval_id)
        assert request.status == ApprovalStatus.PENDING

    @pytest.mark.asyncio
    async def test_auto_approve_ignores_already_decided(
        self,
        run_manager: AgentRunManager,
        approval_manager: ApprovalManager,
        user_context: UserContext,
    ):
        """已人工批准的请求不再自动审批（幂等）。"""
        run_id, approval_id = await _create_run_with_pending_approval(run_manager, user_context)

        await run_manager.apply_approval_decision(
            run_id=run_id,
            approval_id=approval_id,
            approval_decision=ApprovalDecision(
                decision=ApprovalDecisionType.APPROVE,
                edited_parameters=None,
            ),
            user_context=user_context,
        )
        request = approval_manager.get_request(approval_id)
        assert request.decided_by == user_context.user_id

        decision = run_manager.maybe_auto_approve(run_id, approval_id, user_context)
        assert decision is None
        assert approval_manager.get_request(approval_id).decided_by == user_context.user_id

    @pytest.mark.asyncio
    async def test_auto_approve_and_execute_executes_tool_once(
        self,
        run_manager: AgentRunManager,
        user_context: UserContext,
    ):
        """demo 链路：策略命中时自动审批并执行写入工具。"""
        run_id, _ = await _create_run_with_pending_approval(run_manager, user_context)

        executed = await run_manager.auto_approve_and_execute(run_id, user_context)

        assert executed is not None
        assert executed.status == ToolCallStatus.COMPLETED
        assert executed.result is not None
        assert executed.result.get("ticket_id", "").startswith("TK-")

        run = await run_manager.get_run(run_id)
        assert run.status == RunStatus.RESUMED


class TestGraphAutoApprove:
    """LangGraph 链路自动审批测试。"""

    @pytest.mark.asyncio
    async def test_graph_auto_approves_without_interrupt(
        self,
        graph_runner: GraphRunner,
        run_manager: AgentRunManager,
        approval_manager: ApprovalManager,
        user_context: UserContext,
    ):
        """策略命中时 Graph 不 interrupt，直接执行写入工具并完成。"""
        events = []
        async for event in graph_runner.run(
            query="帮我创建入职工单",
            user_context=user_context,
        ):
            events.append(event)

        # 不应发出 approval_required（即没有中断等待人工）
        parsed_events = [_parse_sse_event(e) for e in events]
        approval_events = [
            e for e in parsed_events if e["type"] == SSEEventType.APPROVAL_REQUIRED.value
        ]
        assert approval_events == []

        # 写入工具应被执行
        tool_events = [e for e in parsed_events if e["type"] == SSEEventType.TOOL_EXECUTED.value]
        assert len(tool_events) == 1
        assert tool_events[0]["data"]["tool_name"] == "create_hr_ticket"

        # 审批记录为 policy_engine，Run 正常完成
        started = [e for e in parsed_events if e["type"] == SSEEventType.RUN_STARTED.value]
        run_id = started[0]["run_id"]

        approvals = await run_manager.get_run_approvals(run_id)
        assert len(approvals) == 1
        assert approvals[0].status == ApprovalStatus.APPROVED
        assert approvals[0].decided_by == POLICY_ENGINE_APPROVER

        run = await run_manager.get_run(run_id)
        assert run.status == RunStatus.COMPLETED
        assert any(
            tc.tool_name == "create_hr_ticket" and tc.status == ToolCallStatus.COMPLETED
            for tc in run.tool_calls
        )

    @pytest.mark.asyncio
    async def test_graph_resume_respects_human_decision_over_policy(
        self,
        tool_executor: ToolExecutor,
        approval_manager: ApprovalManager,
        step_logger: StepLogger,
        hybrid_retriever: HybridRetriever,
        user_context: UserContext,
    ):
        """resume 恢复路径：人工决策优先，策略不得覆盖已提交的审批结果。"""

        class FlipPolicy:
            """首次调用转人工（产生 interrupt），之后策略已命中但不应被咨询。"""

            def __init__(self) -> None:
                self.calls = 0

            def evaluate(self, **kwargs: object) -> ApprovalDecisionType | None:
                self.calls += 1
                return None if self.calls == 1 else ApprovalDecisionType.APPROVE

        policy = FlipPolicy()
        run_manager = AgentRunManager(
            tool_executor,
            approval_manager,
            step_logger,
            approval_policy=policy,  # type: ignore[arg-type]
        )
        graph = create_agent_graph(run_manager, hybrid_retriever)
        runner = GraphRunner(graph, run_manager, step_logger)

        thread_id = None
        events = []
        async for event in runner.run(
            query="帮我创建入职工单",
            user_context=user_context,
        ):
            events.append(event)
            if SSEEventType.RUN_STARTED.value in event:
                thread_id = _parse_sse_event(event)["data"]["thread_id"]

        # 首次运行策略未命中，进入 interrupt 等待人工
        approval_events = [
            e
            for e in (_parse_sse_event(ev) for ev in events)
            if e["type"] == SSEEventType.APPROVAL_REQUIRED.value
        ]
        assert approval_events, "策略首次未命中时应触发 interrupt"
        assert policy.calls == 1
        run_id = approval_events[0]["run_id"]

        # 人工提交拒绝；resume 重跑节点时策略不应被再次咨询
        resume_events = []
        async for event in runner.resume(
            thread_id=thread_id,
            approval_decision=ApprovalDecision(
                decision=ApprovalDecisionType.REJECT,
                edited_parameters=None,
            ),
            user_context=user_context,
        ):
            resume_events.append(event)

        assert len(resume_events) > 0
        assert policy.calls == 1, "恢复路径上守卫应阻止策略被再次咨询"

        approvals = await run_manager.get_run_approvals(run_id)
        assert approvals[0].status == ApprovalStatus.REJECTED
        run = await run_manager.get_run(run_id)
        assert not any(tc.status == ToolCallStatus.COMPLETED for tc in run.tool_calls)
