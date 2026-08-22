"""
Trace 上下文。

管理 Trace 的上下文信息。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.services.observability.span import Span


class TraceContext:
    """
    Trace 上下文。

    管理一次 Agent Run 的完整 Trace 信息。
    """

    def __init__(
        self,
        trace_id: str,
        run_id: str,
        user_id: str,
        tenant_id: str,
        case_id: str | None = None,
        event_id: str | None = None,
    ) -> None:
        self.trace_id = trace_id
        self.run_id = run_id
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.case_id = case_id
        self.event_id = event_id
        self.spans: list[Span] = []
        self.current_span: Span | None = None
        self.created_at: datetime = datetime.now(UTC)

    def add_span(self, span: Span) -> None:
        """添加 Span 到上下文。"""
        self.spans.append(span)
        self.current_span = span

    def get_span(self, span_id: str) -> Span | None:
        """获取 Span。"""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_root_span(self) -> Span | None:
        """获取根 Span（没有 parent 的 Span）。"""
        for span in self.spans:
            if span.parent_id is None:
                return span
        return None

    def get_child_spans(self, parent_id: str) -> list[Span]:
        """获取子 Span。"""
        return [s for s in self.spans if s.parent_id == parent_id]

    def get_all_spans(self) -> list[Span]:
        """获取所有 Span。"""
        return self.spans.copy()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "case_id": self.case_id,
            "event_id": self.event_id,
            "span_count": len(self.spans),
            "created_at": self.created_at.isoformat(),
        }
