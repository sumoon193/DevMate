"""
LangGraph 测试。

按模块规范要求的 5 个测试：
1. test_graph_interrupts_before_write_tool
2. test_graph_resumes_with_same_thread_id
3. test_graph_reject_path_does_not_execute_tool
4. test_graph_low_confidence_routes_to_clarification
5. test_sse_events_follow_step_order
"""

from __future__ import annotations

import json

import pytest

from app.schemas.approval import ApprovalDecision
from app.schemas.enums import (
    ApprovalDecisionType,
    ApprovalStatus,
    RunStatus,
    ToolRiskLevel,
)
from app.schemas.tool import ToolDefinition
from app.schemas.user import UserContext
from app.services.agent.approval_manager import ApprovalManager
from app.services.agent.run_manager import AgentRunManager
from app.services.agent.step_logger import StepLogger
from app.services.agent.tool_executor import ToolExecutor
from app.services.agent.tool_registry import ToolRegistry
from app.services.agent.tools.mock_ticket import MockTicketHandler
from app.services.agent.tools.policy_search import PolicySearchHandler
from app.services.graph.graph import create_agent_graph
from app.services.graph.runner import GraphRunner
from app.services.graph.sse import SSEEventType
from app.services.observability.span import SpanType
from app.services.observability.tracer import Tracer
from app.services.retrieval.embedding.mock_embedding import MockEmbedder
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.reranker.mock_reranker import MockReranker
from app.services.retrieval.store.memory_bm25 import InMemoryBM25Store
from app.services.retrieval.store.memory_vector import InMemoryVectorStore


def parse_sse_event(event: str) -> dict:
    """解析 SSE 字符串中的 JSON payload。"""
    assert event.startswith("data: ")
    return json.loads(event.split("data: ", 1)[1])


@pytest.fixture
def user_context() -> UserContext:
    """用户上下文。"""
    return UserContext(
        user_id="user_001",
        tenant_id="tenant_hr",
        department_ids=["dept_001", "dept_002"],
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
    """Tool Registry。"""
    registry = ToolRegistry()

    # 注册读取型工具
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

    # 注册写入型工具
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
def graph_runner(graph, run_manager: AgentRunManager, step_logger: StepLogger) -> GraphRunner:
    """Graph Runner。"""
    return GraphRunner(graph, run_manager, step_logger)


class TestGraphInterrupt:
    """Graph 中断测试。"""

    def test_full_mode_graph_uses_compile_compatible_checkpointer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        run_manager: AgentRunManager,
        hybrid_retriever: HybridRetriever,
    ):
        """full mode 下 graph compile 不能传入未进入的 async context manager。"""
        from app.config import get_settings

        monkeypatch.setenv("APP_MODE", "full")
        get_settings.cache_clear()
        try:
            graph = create_agent_graph(run_manager, hybrid_retriever)
        finally:
            get_settings.cache_clear()

        assert graph is not None

    def test_graph_compiles_with_injected_checkpointer(
        self,
        run_manager: AgentRunManager,
        hybrid_retriever: HybridRetriever,
    ):
        """create_agent_graph 使用外部注入的 checkpointer 编译。"""
        from langgraph.checkpoint.memory import MemorySaver

        saver = MemorySaver()
        graph = create_agent_graph(run_manager, hybrid_retriever, checkpointer=saver)

        assert graph.checkpointer is saver

    @pytest.mark.asyncio
    async def test_graph_interrupts_before_write_tool(
        self, graph_runner: GraphRunner, user_context: UserContext
    ):
        """测试 1：Graph 在写入型工具前中断。"""
        # 执行 graph（触发写入型工具）
        events = []
        async for event in graph_runner.run(query="帮我创建入职工单", user_context=user_context):
            events.append(event)

        # 验证包含 approval_required 事件
        approval_events = [e for e in events if SSEEventType.APPROVAL_REQUIRED.value in e]
        assert len(approval_events) > 0, "Should have approval_required event"

        # 验证包含 create_hr_ticket
        for event in approval_events:
            assert "create_hr_ticket" in event

    @pytest.mark.asyncio
    async def test_graph_interrupt_payload_is_json_serializable(
        self, graph_runner: GraphRunner, user_context: UserContext
    ):
        """验证审批中断 payload 可 JSON 序列化且包含必需字段。"""
        events = []
        async for event in graph_runner.run(query="帮我创建入职工单", user_context=user_context):
            events.append(event)

        approval_payloads = [
            parse_sse_event(event)["data"]
            for event in events
            if SSEEventType.APPROVAL_REQUIRED.value in event
        ]

        assert approval_payloads
        payload = approval_payloads[0]
        json.dumps(payload, ensure_ascii=False)
        assert payload["run_id"].startswith("run_")
        assert payload["approval_id"].startswith("appr_")
        assert payload["tool_name"] == "create_hr_ticket"
        assert payload["risk_level"] == "write"
        assert payload["allowed_decisions"] == ["approve", "edit", "reject"]

    @pytest.mark.asyncio
    async def test_graph_runner_can_continue_existing_api_run(
        self,
        graph_runner: GraphRunner,
        run_manager: AgentRunManager,
        user_context: UserContext,
    ):
        """GraphRunner 接管 API 已创建的 run 时，不应另建第二个 run。"""
        run = await run_manager.create_run("帮我创建入职工单", user_context)

        events = []
        async for event in graph_runner.run_existing(
            run_id=run.id,
            query=run.original_query,
            user_context=user_context,
        ):
            events.append(event)

        started = [
            parse_sse_event(event) for event in events if SSEEventType.RUN_STARTED.value in event
        ]
        approvals = [
            parse_sse_event(event)["data"]
            for event in events
            if SSEEventType.APPROVAL_REQUIRED.value in event
        ]

        assert started[0]["run_id"] == run.id
        assert started[0]["data"]["thread_id"] == run.thread_id
        assert approvals[0]["run_id"] == run.id
        assert approvals[0]["tool_name"] == "create_hr_ticket"

        stored = await run_manager.get_run(run.id)
        assert stored.status == RunStatus.AWAITING_APPROVAL


class TestGraphResume:
    """Graph 恢复测试。"""

    @pytest.mark.asyncio
    async def test_graph_resumes_with_same_thread_id(
        self, graph_runner: GraphRunner, user_context: UserContext
    ):
        """测试 2：Graph 使用相同的 thread_id 恢复。"""
        # 第一次执行（会中断）
        thread_id = None
        async for event in graph_runner.run(query="帮我创建入职工单", user_context=user_context):
            if "run_started" in event:
                # 提取 thread_id
                import json

                data = json.loads(event.split("data: ")[1])
                thread_id = data.get("data", {}).get("thread_id")

        # 验证 thread_id
        assert thread_id is not None

        # 恢复执行（approve）
        approval_decision = ApprovalDecision(
            decision=ApprovalDecisionType.APPROVE, edited_parameters=None
        )

        resume_events = []
        async for event in graph_runner.resume(
            thread_id=thread_id, approval_decision=approval_decision, user_context=user_context
        ):
            resume_events.append(event)

        # 验证恢复执行
        assert len(resume_events) > 0

    @pytest.mark.asyncio
    async def test_graph_approve_path_executes_tool_once(
        self, graph_runner: GraphRunner, user_context: UserContext
    ):
        """验证 approve 路径只执行一次写入型工具。"""
        thread_id = None
        async for event in graph_runner.run(query="帮我创建入职工单", user_context=user_context):
            if "run_started" in event:
                thread_id = parse_sse_event(event)["data"]["thread_id"]

        approval_decision = ApprovalDecision(
            decision=ApprovalDecisionType.APPROVE, edited_parameters=None
        )

        events = []
        async for event in graph_runner.resume(
            thread_id=thread_id, approval_decision=approval_decision, user_context=user_context
        ):
            events.append(event)

        tool_events = [
            parse_sse_event(event) for event in events if SSEEventType.TOOL_EXECUTED.value in event
        ]
        assert len(tool_events) == 1
        assert tool_events[0]["data"]["tool_name"] == "create_hr_ticket"

    @pytest.mark.asyncio
    async def test_graph_edit_path_uses_edited_parameters(
        self, graph_runner: GraphRunner, user_context: UserContext
    ):
        """验证 edit 路径使用编辑后的工具参数。"""
        thread_id = None
        async for event in graph_runner.run(query="帮我创建入职工单", user_context=user_context):
            if "run_started" in event:
                thread_id = parse_sse_event(event)["data"]["thread_id"]

        approval_decision = ApprovalDecision(
            decision=ApprovalDecisionType.EDIT,
            edited_parameters={
                "title": "编辑后的入职工单",
                "description": "由审批人修改",
                "priority": "high",
                "category": "入职",
            },
        )

        events = []
        async for event in graph_runner.resume(
            thread_id=thread_id, approval_decision=approval_decision, user_context=user_context
        ):
            events.append(event)

        tool_events = [
            parse_sse_event(event) for event in events if SSEEventType.TOOL_EXECUTED.value in event
        ]
        assert len(tool_events) == 1
        assert tool_events[0]["data"]["result"]["title"] == "编辑后的入职工单"


class TestGraphReject:
    """Graph 拒绝测试。"""

    @pytest.mark.asyncio
    async def test_graph_reject_path_does_not_execute_tool(
        self, graph_runner: GraphRunner, user_context: UserContext
    ):
        """测试 3：拒绝路径不执行工具。"""
        # 第一次执行（会中断）
        thread_id = None
        async for event in graph_runner.run(query="帮我创建入职工单", user_context=user_context):
            if "run_started" in event:
                import json

                data = json.loads(event.split("data: ")[1])
                thread_id = data.get("data", {}).get("thread_id")

        # 恢复执行（reject）
        approval_decision = ApprovalDecision(
            decision=ApprovalDecisionType.REJECT, edited_parameters=None
        )

        events = []
        async for event in graph_runner.resume(
            thread_id=thread_id, approval_decision=approval_decision, user_context=user_context
        ):
            events.append(event)

        # 验证没有 tool_executed 事件
        tool_executed_events = [e for e in events if SSEEventType.TOOL_EXECUTED.value in e]
        assert len(tool_executed_events) == 0, "Should not have tool_executed event after reject"

    @pytest.mark.asyncio
    async def test_graph_reject_path_marks_approval_rejected(
        self,
        graph_runner: GraphRunner,
        approval_manager: ApprovalManager,
        user_context: UserContext,
    ):
        """验证 reject 路径会标记审批请求为 rejected。"""
        thread_id = None
        approval_id = None
        async for event in graph_runner.run(query="帮我创建入职工单", user_context=user_context):
            if "run_started" in event:
                thread_id = parse_sse_event(event)["data"]["thread_id"]
            if SSEEventType.APPROVAL_REQUIRED.value in event:
                approval_id = parse_sse_event(event)["data"]["approval_id"]

        approval_decision = ApprovalDecision(
            decision=ApprovalDecisionType.REJECT, edited_parameters=None
        )

        async for _ in graph_runner.resume(
            thread_id=thread_id, approval_decision=approval_decision, user_context=user_context
        ):
            pass

        request = approval_manager.get_request(approval_id)
        assert request.status == ApprovalStatus.REJECTED
        assert request.decision == ApprovalDecisionType.REJECT


class TestGraphClarification:
    """Graph 澄清测试。"""

    @pytest.mark.asyncio
    async def test_graph_low_confidence_routes_to_clarification(
        self, graph_runner: GraphRunner, user_context: UserContext
    ):
        """测试 4：低置信度路由到澄清。"""
        # 执行一个模糊的查询
        events = []
        async for event in graph_runner.run(query="那个事情怎么办", user_context=user_context):
            events.append(event)

        # 验证有 answer_ready 事件
        answer_events = [e for e in events if SSEEventType.ANSWER_READY.value in e]
        assert len(answer_events) > 0


class TestSSEEvents:
    """SSE 事件测试。"""

    @pytest.mark.asyncio
    async def test_sse_events_follow_step_order(
        self, graph_runner: GraphRunner, user_context: UserContext
    ):
        """测试 5：SSE 事件按步骤顺序输出。"""
        events = []
        async for event in graph_runner.run(
            query="新员工入职需要提交哪些材料？", user_context=user_context
        ):
            events.append(event)

        # 验证事件顺序
        event_types = []
        for event in events:
            if event.startswith("data: "):
                import json

                data = json.loads(event.split("data: ")[1])
                event_types.append(data.get("type"))

        # 验证基本顺序
        assert SSEEventType.RUN_STARTED.value in event_types
        assert SSEEventType.ANSWER_READY.value in event_types

        # 验证 step_started 在 step_completed 之前
        open_steps = 0
        for etype in event_types:
            if etype == SSEEventType.STEP_STARTED.value:
                open_steps += 1
            if etype == SSEEventType.STEP_COMPLETED.value:
                assert open_steps > 0
                open_steps -= 1
        assert open_steps == 0


class TestGraphObservability:
    """Graph trace 导出测试。"""

    @pytest.mark.asyncio
    async def test_graph_runner_exports_node_level_spans(
        self,
        graph,
        run_manager: AgentRunManager,
        step_logger: StepLogger,
        user_context: UserContext,
    ):
        """GraphRunner 应为 run 和关键节点导出 span，供 Phoenix/OTel 展示。"""

        class CollectingExporter:
            def __init__(self) -> None:
                self.exports = []

            def export(self, spans, context) -> None:
                self.exports.append((spans, context))

        exporter = CollectingExporter()
        traced_runner = GraphRunner(
            graph=graph,
            run_manager=run_manager,
            step_logger=step_logger,
            tracer=Tracer(exporter=exporter),
        )

        async for _ in traced_runner.run("帮我创建入职工单", user_context):
            pass

        assert exporter.exports
        spans, context = exporter.exports[0]
        names = {span.name for span in spans}
        assert context.run_id.startswith("run_")
        assert "agent_run" in names
        assert "intent" in names
        assert "approval_gate" in names
        assert any(span.span_type == SpanType.APPROVAL_WAIT for span in spans)
