"""
日志导出器。

使用标准库 logging 输出 Trace 信息（fallback 模式）。
"""

from __future__ import annotations

import logging

from app.services.observability.context import TraceContext
from app.services.observability.span import Span, SpanStatus

logger = logging.getLogger(__name__)


class LogExporter:
    """
    日志导出器。

    使用标准库 logging 输出 Trace 信息，用于 fallback 模式。
    """

    def __init__(self, log_level: int = logging.INFO) -> None:
        """
        初始化日志导出器。

        Args:
            log_level: 日志级别
        """
        self._log_level = log_level

    def export(self, spans: list[Span], context: TraceContext) -> None:
        """
        导出 Span 到日志。

        Args:
            spans: Span 列表
            context: Trace 上下文
        """
        # 导出 Trace 摘要
        logger.log(
            self._log_level,
            "trace_exported",
            extra={
                "trace_id": context.trace_id,
                "run_id": context.run_id,
                "user_id": context.user_id,
                "span_count": len(spans),
            },
        )

        # 导出每个 Span
        for span in spans:
            self._export_span(span)

    def _export_span(self, span: Span) -> None:
        """导出单个 Span。"""
        # 脱敏敏感字段
        attributes = self._sanitize_attributes(span.attributes)

        logger.log(
            self._log_level,
            "span_exported",
            extra={
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "span_type": span.span_type.value,
                "span_name": span.name,
                "status": span.status.value,
                "duration_ms": span.duration_ms,
                "attributes": attributes,
            },
        )

    def _sanitize_attributes(self, attributes: dict) -> dict:
        """
        脱敏敏感字段。

        Args:
            attributes: 原始属性

        Returns:
            脱敏后的属性
        """
        sensitive_keys = {"api_key", "token", "password", "secret", "authorization", "credential"}

        sanitized = {}
        for key, value in attributes.items():
            if any(s in key.lower() for s in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value

        return sanitized

    def export_summary(self, context: TraceContext) -> dict:
        """
        导出 Trace 摘要。

        Args:
            context: Trace 上下文

        Returns:
            摘要字典
        """
        spans = context.get_all_spans()

        # 统计信息
        total_duration = sum(s.duration_ms for s in spans)
        error_count = sum(1 for s in spans if s.status == SpanStatus.ERROR)

        return {
            "trace_id": context.trace_id,
            "run_id": context.run_id,
            "user_id": context.user_id,
            "span_count": len(spans),
            "total_duration_ms": total_duration,
            "error_count": error_count,
            "created_at": context.created_at.isoformat(),
        }
