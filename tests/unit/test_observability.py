"""
Observability 测试。

按模块规范要求的 5 个测试：
1. test_trace_id_created_for_agent_run
2. test_retrieval_span_records_scores
3. test_tool_span_records_approval_id
4. test_sensitive_fields_are_redacted
5. test_observability_failure_does_not_fail_request
"""

from __future__ import annotations

import pytest

from app.services.observability.exporters.log_exporter import LogExporter
from app.services.observability.span import Span, SpanStatus, SpanType
from app.services.observability.tracer import Tracer


@pytest.fixture
def tracer() -> Tracer:
    """Tracer 实例。"""
    exporter = LogExporter()
    return Tracer(exporter=exporter)


class TestTraceCreation:
    """Trace 创建测试。"""

    def test_trace_id_created_for_agent_run(self, tracer: Tracer):
        """测试 1：为 Agent Run 创建 trace_id。"""
        # 开始 Trace
        context = tracer.start_trace(run_id="run_001", user_id="user_001", tenant_id="tenant_hr")

        # 验证 trace_id
        assert context.trace_id is not None
        assert context.trace_id.startswith("trace_")
        assert len(context.trace_id) > 10

        # 验证上下文信息
        assert context.run_id == "run_001"
        assert context.user_id == "user_001"
        assert context.tenant_id == "tenant_hr"

    def test_trace_id_unique(self, tracer: Tracer):
        """测试 trace_id 唯一性。"""
        context1 = tracer.start_trace("run_001", "user_001", "tenant_hr")
        context2 = tracer.start_trace("run_002", "user_002", "tenant_hr")

        assert context1.trace_id != context2.trace_id

    def test_case_trace_correlates_case_run_and_event(self, tracer: Tracer):
        """长期 Case span 应同时携带 case_id、run_id 与 event_id。"""
        context = tracer.start_trace(
            "run_001",
            "user_001",
            "tenant_hr",
            case_id="case_001",
            event_id="evt_001",
        )
        span = tracer.start_span(context, SpanType.AGENT_STEP, "case_event")

        assert span.attributes["case_id"] == "case_001"
        assert span.attributes["run_id"] == "run_001"
        assert span.attributes["event_id"] == "evt_001"


class TestSpanRecording:
    """Span 记录测试。"""

    def test_retrieval_span_records_scores(self, tracer: Tracer):
        """测试 2：检索 Span 记录分数。"""
        # 开始 Trace
        context = tracer.start_trace("run_001", "user_001", "tenant_hr")

        # 开始检索 Span
        span = tracer.start_span(
            context=context, span_type=SpanType.RETRIEVAL_SEARCH, name="hybrid_search"
        )

        # 设置属性
        span.set_attribute("query", "入职材料")
        span.set_attribute("citation_count", 5)
        span.set_attribute("query_coverage", 0.85)

        # 结束 Span
        tracer.end_span(span, SpanStatus.OK)

        # 验证
        assert span.attributes.get("citation_count") == 5
        assert span.attributes.get("query_coverage") == 0.85
        assert span.status == SpanStatus.OK
        assert span.end_time is not None

    def test_tool_span_records_approval_id(self, tracer: Tracer):
        """测试 3：工具 Span 记录 approval_id。"""
        # 开始 Trace
        context = tracer.start_trace("run_001", "user_001", "tenant_hr")

        # 开始工具 Span
        span = tracer.start_span(
            context=context,
            span_type=SpanType.TOOL_CALL,
            name="create_hr_ticket",
            attributes={"tool_name": "create_hr_ticket"},
        )

        # 设置 approval_id
        span.set_attribute("approval_id", "appr_001")

        # 结束 Span
        tracer.end_span(span, SpanStatus.OK)

        # 验证
        assert span.attributes.get("approval_id") == "appr_001"

    def test_span_hierarchy(self, tracer: Tracer):
        """测试 Span 层次结构。"""
        # 开始 Trace
        context = tracer.start_trace("run_001", "user_001", "tenant_hr")

        # 创建根 Span
        root_span = tracer.start_span(
            context=context, span_type=SpanType.AGENT_RUN, name="agent_run"
        )

        # 创建子 Span
        child_span = tracer.start_span(
            context=context, span_type=SpanType.RETRIEVAL_SEARCH, name="retrieval", parent=root_span
        )

        # 验证父子关系
        assert child_span.parent_id == root_span.span_id
        assert root_span.parent_id is None

        # 验证上下文中的 Span
        assert len(context.spans) == 2
        assert context.get_child_spans(root_span.span_id) == [child_span]


class TestSensitiveFieldRedaction:
    """敏感字段脱敏测试。"""

    def test_sensitive_fields_are_redacted(self):
        """测试 4：敏感字段被脱敏。"""
        exporter = LogExporter()

        # 创建带有敏感字段的 Span
        span = Span(
            span_id="span_001", trace_id="trace_001", span_type=SpanType.LLM_CALL, name="llm_call"
        )
        span.set_attributes(
            {
                "api_key": "sk-1234567890",
                "password": "secret123",
                "authorization": "Bearer token123",
                "query": "普通查询",
            }
        )

        # 脱敏
        sanitized = exporter._sanitize_attributes(span.attributes)

        # 验证
        assert sanitized["api_key"] == "***REDACTED***"
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["authorization"] == "***REDACTED***"
        assert sanitized["query"] == "普通查询"

    def test_export_sanitizes_attributes(self, tracer: Tracer):
        """测试导出时脱敏属性。"""
        # 开始 Trace
        context = tracer.start_trace("run_001", "user_001", "tenant_hr")

        # 创建带有敏感字段的 Span
        span = tracer.start_span(context=context, span_type=SpanType.LLM_CALL, name="llm_call")
        span.set_attribute("api_key", "sk-1234567890")

        # 结束 Span
        tracer.end_span(span, SpanStatus.OK)

        # 验证导出不会抛出异常
        tracer.export_trace(context.trace_id)


class TestObservabilityFailure:
    """可观测性失败测试。"""

    def test_observability_failure_does_not_fail_request(self, tracer: Tracer):
        """测试 5：可观测性失败不影响主链路。"""
        # 开始 Trace
        context = tracer.start_trace("run_001", "user_001", "tenant_hr")

        # 创建 Span
        tracer.start_span(context=context, span_type=SpanType.AGENT_RUN, name="agent_run")

        # 模拟导出失败
        class FailingExporter:
            def export(self, spans, context):
                raise Exception("Export failed")

        failing_tracer = Tracer(exporter=FailingExporter())
        failing_tracer._contexts = tracer._contexts

        # 导出失败不应该抛出异常
        try:
            failing_tracer.export_trace(context.trace_id)
        except Exception:
            pytest.fail("Observability failure should not propagate")

    def test_tracer_without_exporter(self):
        """测试没有导出器时的行为。"""
        tracer = Tracer()  # 没有导出器

        context = tracer.start_trace("run_001", "user_001", "tenant_hr")
        span = tracer.start_span(context=context, span_type=SpanType.AGENT_RUN, name="agent_run")
        tracer.end_span(span, SpanStatus.OK)

        # 导出不应该抛出异常
        tracer.export_trace(context.trace_id)


class TestSpanEvents:
    """Span 事件测试。"""

    def test_span_add_event(self):
        """测试添加 Span 事件。"""
        span = Span(
            span_id="span_001", trace_id="trace_001", span_type=SpanType.AGENT_RUN, name="agent_run"
        )

        # 添加事件
        event = span.add_event("evidence_found", {"count": 5})

        assert len(span.events) == 1
        assert event.name == "evidence_found"
        assert event.attributes["count"] == 5

    def test_span_record_error(self):
        """测试记录 Span 错误。"""
        span = Span(
            span_id="span_001", trace_id="trace_001", span_type=SpanType.TOOL_CALL, name="tool_call"
        )

        # 记录错误
        error = ValueError("Invalid parameter")
        span.record_error(error)

        assert span.status == SpanStatus.ERROR
        assert span.attributes.get("error.type") == "ValueError"
        assert span.attributes.get("error.message") == "Invalid parameter"
        assert len(span.events) == 1
        assert span.events[0].name == "exception"


class TestTraceContext:
    """Trace 上下文测试。"""

    def test_trace_context_to_dict(self, tracer: Tracer):
        """测试 Trace 上下文转字典。"""
        context = tracer.start_trace("run_001", "user_001", "tenant_hr")

        context_dict = context.to_dict()

        assert context_dict["trace_id"] == context.trace_id
        assert context_dict["run_id"] == "run_001"
        assert context_dict["user_id"] == "user_001"
        assert context_dict["tenant_id"] == "tenant_hr"
        assert context_dict["span_count"] == 0

    def test_trace_context_get_span(self, tracer: Tracer):
        """测试获取 Span。"""
        context = tracer.start_trace("run_001", "user_001", "tenant_hr")

        span = tracer.start_span(context=context, span_type=SpanType.AGENT_RUN, name="agent_run")

        # 获取存在的 Span
        found = context.get_span(span.span_id)
        assert found is not None
        assert found.span_id == span.span_id

        # 获取不存在的 Span
        not_found = context.get_span("non_existent")
        assert not_found is None
