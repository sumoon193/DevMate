"""
Trace 管理器。

管理 Trace 的创建、Span 的生命周期和导出。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Protocol

from app.services.observability.context import TraceContext
from app.services.observability.span import Span, SpanStatus, SpanType

logger = logging.getLogger(__name__)


class TraceExporter(Protocol):
    """Trace 导出器接口。"""

    def export(self, spans: list[Span], context: TraceContext) -> None:
        """导出 Span。"""
        ...


class Tracer:
    """
    Trace 管理器。

    管理 Trace 和 Span 的生命周期。
    """

    def __init__(self, exporter: TraceExporter | None = None) -> None:
        """
        初始化 Tracer。

        Args:
            exporter: Trace 导出器（可选）
        """
        self._exporter = exporter
        self._contexts: dict[str, TraceContext] = {}  # trace_id -> context

    def start_trace(
        self,
        run_id: str,
        user_id: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        event_id: str | None = None,
    ) -> TraceContext:
        """
        开始一个新的 Trace。

        Args:
            run_id: Run ID
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            Trace 上下文
        """
        trace_id = f"trace_{uuid.uuid4().hex[:16]}"

        context = TraceContext(
            trace_id=trace_id,
            run_id=run_id,
            user_id=user_id,
            tenant_id=tenant_id,
            case_id=case_id,
            event_id=event_id,
        )

        self._contexts[trace_id] = context

        logger.info(
            "trace_started", extra={"trace_id": trace_id, "run_id": run_id, "user_id": user_id}
        )

        return context

    def start_span(
        self,
        context: TraceContext,
        span_type: SpanType,
        name: str,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """
        开始一个新的 Span。

        Args:
            context: Trace 上下文
            span_type: Span 类型
            name: Span 名称
            parent: 父 Span（可选）
            attributes: 属性（可选）

        Returns:
            新的 Span
        """
        span_id = f"span_{uuid.uuid4().hex[:12]}"

        # 构建默认属性
        default_attrs = {
            "run_id": context.run_id,
            "user_id": context.user_id,
            "tenant_id": context.tenant_id,
        }
        if context.case_id is not None:
            default_attrs["case_id"] = context.case_id
        if context.event_id is not None:
            default_attrs["event_id"] = context.event_id
        if attributes:
            default_attrs.update(attributes)

        span = Span(
            span_id=span_id,
            trace_id=context.trace_id,
            span_type=span_type,
            name=name,
            parent_id=parent.span_id if parent else None,
            attributes=default_attrs,
        )

        context.add_span(span)

        logger.debug(
            "span_started",
            extra={
                "trace_id": context.trace_id,
                "span_id": span_id,
                "span_type": span_type.value,
                "span_name": name,
            },
        )

        return span

    def end_span(self, span: Span, status: SpanStatus = SpanStatus.OK) -> None:
        """
        结束一个 Span。

        Args:
            span: 要结束的 Span
            status: 最终状态
        """
        span.set_status(status)
        span.end()

        logger.debug(
            "span_ended",
            extra={
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "status": status.value,
                "duration_ms": span.duration_ms,
            },
        )

    def record_error(self, span: Span, error: Exception) -> None:
        """
        记录错误。

        Args:
            span: 发生错误的 Span
            error: 异常
        """
        span.record_error(error)

        logger.warning(
            "span_error",
            extra={
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )

    def get_context(self, trace_id: str) -> TraceContext | None:
        """获取 Trace 上下文。"""
        return self._contexts.get(trace_id)

    def export_trace(self, trace_id: str) -> None:
        """
        导出 Trace。

        Args:
            trace_id: Trace ID
        """
        context = self._contexts.get(trace_id)
        if not context:
            return

        if self._exporter:
            try:
                self._exporter.export(context.spans, context)
            except Exception as e:
                logger.error("trace_export_failed", extra={"trace_id": trace_id, "error": str(e)})

    def get_all_traces(self) -> list[TraceContext]:
        """获取所有 Trace 上下文。"""
        return list(self._contexts.values())

    def clear(self) -> None:
        """清空所有 Trace（用于测试）。"""
        self._contexts.clear()
