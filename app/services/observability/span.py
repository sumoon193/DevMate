"""
Span 定义。

定义 Trace 中的 Span 数据结构。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class SpanType(StrEnum):
    """
    Span 类型。

    按照模块规范定义的 Span 类型。
    """

    AGENT_RUN = "agent.run"
    AGENT_STEP = "agent.step"
    RETRIEVAL_SEARCH = "retrieval.search"
    RETRIEVAL_RERANK = "retrieval.rerank"
    LLM_CALL = "llm.call"
    EMBEDDING_CALL = "embedding.call"
    TOOL_CALL = "tool.call"
    APPROVAL_WAIT = "approval.wait"
    GUARDRAIL_CHECK = "guardrail.check"
    EVAL_RUN = "eval.run"


class SpanStatus(StrEnum):
    """Span 状态。"""

    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


class SpanEvent:
    """
    Span 事件。

    记录 Span 内部的事件。
    """

    def __init__(
        self, name: str, attributes: dict[str, Any] | None = None, timestamp: datetime | None = None
    ) -> None:
        self.name = name
        self.attributes = attributes or {}
        self.timestamp = timestamp or datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "attributes": self.attributes,
            "timestamp": self.timestamp.isoformat(),
        }


class Span:
    """
    Span。

    Trace 中的基本单元，记录一次操作的详细信息。
    """

    def __init__(
        self,
        span_id: str,
        trace_id: str,
        span_type: SpanType,
        name: str,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.span_id = span_id
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.span_type = span_type
        self.name = name
        self.attributes: dict[str, Any] = attributes or {}
        self.start_time: datetime = datetime.now(UTC)
        self.end_time: datetime | None = None
        self.status: SpanStatus = SpanStatus.UNSET
        self.events: list[SpanEvent] = []
        self.duration_ms: float = 0.0

    def set_attribute(self, key: str, value: Any) -> None:
        """设置属性。"""
        self.attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """批量设置属性。"""
        self.attributes.update(attributes)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> SpanEvent:
        """添加事件。"""
        event = SpanEvent(name, attributes)
        self.events.append(event)
        return event

    def set_status(self, status: SpanStatus) -> None:
        """设置状态。"""
        self.status = status

    def record_error(self, error: Exception) -> None:
        """记录错误。"""
        self.status = SpanStatus.ERROR
        self.set_attributes({"error.type": type(error).__name__, "error.message": str(error)})
        self.add_event(
            "exception", {"exception.type": type(error).__name__, "exception.message": str(error)}
        )

    def end(self) -> None:
        """结束 Span。"""
        self.end_time = datetime.now(UTC)
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "span_type": self.span_type.value,
            "name": self.name,
            "attributes": self.attributes,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "events": [e.to_dict() for e in self.events],
        }
