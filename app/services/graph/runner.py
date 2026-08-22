"""
Graph Runner。

执行 Agent Graph 的入口。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

from langgraph.graph import StateGraph
from langgraph.types import Command

from app.schemas.agent import AgentRunResponse
from app.schemas.approval import ApprovalDecision
from app.schemas.user import UserContext
from app.services.agent.run_manager import AgentRunManager
from app.services.agent.step_logger import StepLogger
from app.services.graph.sse import (
    create_answer_ready_event,
    create_approval_required_event,
    create_evidence_found_event,
    create_run_failed_event,
    create_run_started_event,
    create_step_completed_event,
    create_step_started_event,
    create_tool_executed_event,
)
from app.services.graph.state import AgentGraphState
from app.services.observability.context import TraceContext
from app.services.observability.span import Span, SpanStatus, SpanType
from app.services.observability.tracer import Tracer

logger = logging.getLogger(__name__)


class GraphRunner:
    """
    Graph 执行器。

    执行 Agent Graph，支持 run 和 resume。
    """

    def __init__(
        self,
        graph: StateGraph,
        run_manager: AgentRunManager,
        step_logger: StepLogger,
        tracer: Tracer | None = None,
    ) -> None:
        """
        初始化 Graph Runner。

        Args:
            graph: 编译后的 StateGraph
            run_manager: Agent Run Manager
            step_logger: Step Logger
        """
        self._graph = graph
        self._run_manager = run_manager
        self._step_logger = step_logger
        self._tracer = tracer
        self._thread_runs: dict[str, str] = {}

    async def run(
        self, query: str, user_context: UserContext, thread_id: str | None = None
    ) -> AsyncGenerator[str, None]:
        """
        执行 Agent Graph。

        Args:
            query: 用户查询
            user_context: 用户上下文
            thread_id: 线程 ID（可选，用于 resume）

        Yields:
            SSE 事件字符串
        """
        # 创建 Run
        run = await self._run_manager.create_run(query, user_context)
        run_id = run.id

        if thread_id is None:
            thread_id = run.thread_id
        self._thread_runs[thread_id] = run_id

        logger.info("graph_run_started", extra={"run_id": run_id, "thread_id": thread_id})
        trace_context, root_span = self._start_trace(
            run_id=run_id,
            thread_id=thread_id,
            query=query,
            user_context=user_context,
        )

        # 发送 run_started 事件
        yield create_run_started_event(run_id, query, thread_id)

        # 准备初始状态
        initial_state: AgentGraphState = {
            "run_id": run_id,
            "thread_id": thread_id,
            "user": user_context,
            "question": query,
            "intent": None,
            "rewritten_queries": [],
            "evidence": None,
            "plan": None,
            "pending_tool_call": None,
            "pending_approval_id": None,
            "approval_decision": None,
            "tool_results": [],
            "answer": None,
            "errors": [],
        }

        # 配置
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # 执行 graph
            async for event in self._graph.astream(initial_state, config=config):
                # 处理事件
                async for sse in self._process_event(event, run_id, trace_context):
                    yield sse

        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            logger.error(
                "graph_run_failed",
                extra={"run_id": run_id, "error": str(e), "traceback": error_detail},
            )
            self._record_trace_error(root_span, e)
            yield create_run_failed_event(run_id, str(e))
            await self._run_manager.fail_run(run_id, str(e))
        finally:
            self._finish_trace(trace_context, root_span)

    async def run_existing(
        self, run_id: str, query: str, user_context: UserContext, thread_id: str | None = None
    ) -> AsyncGenerator[str, None]:
        """
        使用已经创建好的 Agent Run 执行 Graph。

        API 层先创建 run 再交给 Graph 时必须走这个入口，否则会生成第二个
        run_id，导致审批、SSE 和详情查询分裂。
        """
        run = await self._run_manager.get_run(run_id)
        if thread_id is None:
            thread_id = run.thread_id
        self._thread_runs[thread_id] = run_id

        logger.info("graph_existing_run_started", extra={"run_id": run_id, "thread_id": thread_id})
        trace_context, root_span = self._start_trace(
            run_id=run_id,
            thread_id=thread_id,
            query=query,
            user_context=user_context,
        )
        yield create_run_started_event(run_id, query, thread_id)

        initial_state: AgentGraphState = {
            "run_id": run_id,
            "thread_id": thread_id,
            "user": user_context,
            "question": query,
            "intent": None,
            "rewritten_queries": [],
            "evidence": None,
            "plan": None,
            "pending_tool_call": None,
            "pending_approval_id": None,
            "approval_decision": None,
            "tool_results": [],
            "answer": None,
            "errors": [],
        }
        config = {"configurable": {"thread_id": thread_id}}

        try:
            async for event in self._graph.astream(initial_state, config=config):
                async for sse in self._process_event(event, run_id, trace_context):
                    yield sse
        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            logger.error(
                "graph_existing_run_failed",
                extra={"run_id": run_id, "error": str(e), "traceback": error_detail},
            )
            self._record_trace_error(root_span, e)
            yield create_run_failed_event(run_id, str(e))
            await self._run_manager.fail_run(run_id, str(e))
        finally:
            self._finish_trace(trace_context, root_span)

    async def run_to_checkpoint(
        self, query: str, user_context: UserContext, thread_id: str | None = None
    ) -> AgentRunResponse:
        """执行 Graph 到完成或 interrupt，并返回当前 run 快照。"""
        run_id = ""
        async for event in self.run(query=query, user_context=user_context, thread_id=thread_id):
            if not run_id and event.startswith("data: "):
                try:
                    payload = json.loads(event.split("data: ", 1)[1])
                    data = payload.get("data", {})
                    run_id = data.get("run_id") or payload.get("run_id") or ""
                except json.JSONDecodeError:
                    logger.debug("graph_event_parse_failed", extra={"event": event[:120]})

        if not run_id:
            raise RuntimeError("Graph run did not emit a run_id")
        return await self._run_manager.get_run(run_id)

    async def resume(
        self, thread_id: str, approval_decision: ApprovalDecision, user_context: UserContext
    ) -> AsyncGenerator[str, None]:
        """
        恢复执行（审批后）。

        Args:
            thread_id: 线程 ID
            approval_decision: 审批决策
            user_context: 用户上下文

        Yields:
            SSE 事件字符串
        """
        logger.info("graph_resume_started", extra={"thread_id": thread_id})

        # 从 checkpointer 恢复状态
        config = {"configurable": {"thread_id": thread_id}}
        resume_command = Command(resume=approval_decision.model_dump(mode="json"))
        run_id = self._thread_runs.get(thread_id, "")

        try:
            # 恢复执行
            async for event in self._graph.astream(resume_command, config=config, subgraphs=False):
                # 从事件中提取 run_id
                event_run_id = self._extract_run_id(event) or run_id

                # 处理事件
                async for sse in self._process_event(event, event_run_id):
                    yield sse

        except Exception as e:
            logger.error("graph_resume_failed", extra={"thread_id": thread_id, "error": str(e)})
            yield create_run_failed_event("", str(e))

    async def _process_event(
        self,
        event: dict[str, Any],
        run_id: str,
        trace_context: TraceContext | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        处理 graph 事件。

        Args:
            event: graph 事件
            run_id: Run ID

        Yields:
            SSE 事件字符串
        """
        # 遍历事件中的节点
        for node_name, node_output in event.items():
            if node_name == "__end__":
                continue
            if node_name == "__interrupt__":
                span = self._start_node_span(trace_context, "approval_gate", run_id, node_output)
                try:
                    for payload in self._extract_interrupt_payloads(node_output):
                        yield create_approval_required_event(
                            run_id=payload.get("run_id", run_id),
                            tool_name=payload.get("tool_name", ""),
                            parameters=payload.get("tool_args", {}),
                            risk_level=payload.get("risk_level", "write"),
                            approval_id=payload.get("approval_id", ""),
                            evidence_summary=payload.get("evidence_summary", []),
                            allowed_decisions=payload.get("allowed_decisions", []),
                        )
                finally:
                    self._finish_node_span(span, node_output)
                continue

            span = self._start_node_span(trace_context, node_name, run_id, node_output)

            # 发送 step_started 事件
            try:
                yield create_step_started_event(run_id, node_name)

                # 根据节点类型发送特定事件
                if node_name == "retrieve" and node_output.get("evidence"):
                    evidence = node_output["evidence"]
                    yield create_evidence_found_event(
                        run_id,
                        evidence.get("total_count", 0),
                        evidence.get("query_coverage_score", 0.0),
                    )

                elif (
                    node_name == "approval_gate"
                    and node_output.get("pending_tool_call")
                    and not node_output.get("approval_decision")
                ):
                    tool_call = node_output["pending_tool_call"]
                    yield create_approval_required_event(
                        run_id, tool_call.tool_name, tool_call.parameters, "write"
                    )

                elif node_name == "tool_execute" and node_output.get("tool_results"):
                    for tool_call in node_output["tool_results"]:
                        yield create_tool_executed_event(
                            run_id, tool_call.tool_name, tool_call.result or {}
                        )

                elif node_name == "answer" and node_output.get("answer"):
                    answer = node_output["answer"]
                    yield create_answer_ready_event(
                        run_id,
                        answer.get("answer", ""),
                        answer.get("citations", []),
                        answer.get("confidence", 0.0),
                    )

                # 发送 step_completed 事件（需要将不可序列化的对象转换为 dict）
                serializable_output = self._make_serializable(node_output)
                yield create_step_completed_event(run_id, node_name, serializable_output)
            finally:
                self._finish_node_span(span, node_output)

    def _start_trace(
        self,
        run_id: str,
        thread_id: str,
        query: str,
        user_context: UserContext,
    ) -> tuple[TraceContext | None, Span | None]:
        """启动 Graph run 级 trace。"""
        if self._tracer is None:
            return None, None

        context = self._tracer.start_trace(
            run_id=run_id,
            user_id=user_context.user_id,
            tenant_id=user_context.tenant_id,
        )
        root_span = self._tracer.start_span(
            context=context,
            span_type=SpanType.AGENT_RUN,
            name="agent_run",
            attributes={
                "thread_id": thread_id,
                "question": query[:500],
            },
        )
        return context, root_span

    def _finish_trace(self, context: TraceContext | None, root_span: Span | None) -> None:
        """结束并导出 Graph trace；导出失败不能影响主链路。"""
        if self._tracer is None or context is None:
            return

        if root_span is not None and root_span.end_time is None:
            status = root_span.status if root_span.status != SpanStatus.UNSET else SpanStatus.OK
            self._tracer.end_span(root_span, status)
        self._tracer.export_trace(context.trace_id)

    def _record_trace_error(self, root_span: Span | None, error: Exception) -> None:
        """将 Graph 级错误写入 root span。"""
        if self._tracer is not None and root_span is not None:
            self._tracer.record_error(root_span, error)

    def _start_node_span(
        self,
        context: TraceContext | None,
        node_name: str,
        run_id: str,
        node_output: Any,
    ) -> Span | None:
        """为单个 LangGraph 节点启动 span。"""
        if self._tracer is None or context is None:
            return None

        root_span = next((span for span in context.spans if span.name == "agent_run"), None)
        attributes: dict[str, Any] = {
            "run_id": run_id,
            "node_name": node_name,
        }
        if isinstance(node_output, dict):
            if node_output.get("evidence"):
                evidence = node_output["evidence"]
                if isinstance(evidence, dict):
                    attributes["citation_count"] = evidence.get("total_count", 0)
                    attributes["query_coverage"] = evidence.get("query_coverage_score", 0.0)
            if node_output.get("pending_approval_id"):
                attributes["approval_id"] = node_output["pending_approval_id"]
            if node_output.get("errors"):
                attributes["error_count"] = len(node_output["errors"])

        return self._tracer.start_span(
            context=context,
            span_type=self._span_type_for_node(node_name),
            name=node_name,
            parent=root_span,
            attributes=attributes,
        )

    def _finish_node_span(self, span: Span | None, node_output: Any) -> None:
        """结束单个节点 span。"""
        if self._tracer is None or span is None:
            return

        status = SpanStatus.OK
        if isinstance(node_output, dict) and node_output.get("errors"):
            status = SpanStatus.ERROR
        self._tracer.end_span(span, status)

    def _span_type_for_node(self, node_name: str) -> SpanType:
        """将 LangGraph 节点映射为可观测 span 类型。"""
        return {
            "retrieve": SpanType.RETRIEVAL_SEARCH,
            "answer": SpanType.LLM_CALL,
            "tool_execute": SpanType.TOOL_CALL,
            "approval_gate": SpanType.APPROVAL_WAIT,
        }.get(node_name, SpanType.AGENT_STEP)

    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可 JSON 序列化的格式。"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "model_dump"):
            # Pydantic model
            return obj.model_dump(mode="json")
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, "__dict__"):
            # 普通对象
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        else:
            return obj

    def _extract_run_id(self, event: dict[str, Any]) -> str:
        """从事件中提取 run_id。"""
        # 尝试从节点输出中提取
        for node_output in event.values():
            if isinstance(node_output, dict) and "run_id" in node_output:
                return node_output["run_id"]
        return ""

    def _extract_interrupt_payloads(self, node_output: Any) -> list[dict[str, Any]]:
        """从 LangGraph interrupt 事件中提取 JSON payload。"""
        payloads: list[dict[str, Any]] = []
        interrupt_items = node_output
        if not isinstance(interrupt_items, (list, tuple)):
            interrupt_items = [interrupt_items]

        for item in interrupt_items:
            value = getattr(item, "value", item)
            if isinstance(value, dict):
                payloads.append(value)

        return payloads
