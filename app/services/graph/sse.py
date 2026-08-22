"""
SSE 事件生成。

定义 SSE 事件类型和生成逻辑。
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class SSEEventType(StrEnum):
    """
    SSE 事件类型。

    按照模块规范定义的事件类型。
    """

    RUN_STARTED = "run_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    EVIDENCE_FOUND = "evidence_found"
    APPROVAL_REQUIRED = "approval_required"
    TOOL_EXECUTED = "tool_executed"
    ANSWER_READY = "answer_ready"
    RUN_STATUS = "run_status"
    RUN_FAILED = "run_failed"


def create_sse_event(event_type: SSEEventType, data: dict[str, Any], run_id: str = "") -> str:
    """
    创建 SSE 事件。

    Args:
        event_type: 事件类型
        data: 事件数据
        run_id: Run ID

    Returns:
        SSE 事件字符串（data: {...}\n\n）
    """
    event = {
        "type": event_type.value,
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "data": data,
    }

    # 转换为 JSON 字符串
    event_json = json.dumps(event, ensure_ascii=False)

    logger.debug("sse_event_created", extra={"event_type": event_type.value, "run_id": run_id})

    return f"data: {event_json}\n\n"


def create_run_started_event(run_id: str, question: str, thread_id: str = "") -> str:
    """创建 run_started 事件。"""
    return create_sse_event(
        SSEEventType.RUN_STARTED, {"question": question, "thread_id": thread_id}, run_id
    )


def create_step_started_event(run_id: str, step_name: str) -> str:
    """创建 step_started 事件。"""
    return create_sse_event(SSEEventType.STEP_STARTED, {"step_name": step_name}, run_id)


def create_step_completed_event(run_id: str, step_name: str, result: dict | None = None) -> str:
    """创建 step_completed 事件。"""
    return create_sse_event(
        SSEEventType.STEP_COMPLETED, {"step_name": step_name, "result": result}, run_id
    )


def create_evidence_found_event(run_id: str, citation_count: int, query_coverage: float) -> str:
    """创建 evidence_found 事件。"""
    return create_sse_event(
        SSEEventType.EVIDENCE_FOUND,
        {"citation_count": citation_count, "query_coverage": query_coverage},
        run_id,
    )


def create_approval_required_event(
    run_id: str,
    tool_name: str,
    parameters: dict[str, Any],
    risk_level: str,
    approval_id: str = "",
    evidence_summary: list[dict[str, Any]] | None = None,
    allowed_decisions: list[str] | None = None,
) -> str:
    """创建 approval_required 事件。"""
    return create_sse_event(
        SSEEventType.APPROVAL_REQUIRED,
        {
            "run_id": run_id,
            "approval_id": approval_id,
            "tool_name": tool_name,
            "tool_args": parameters,
            "parameters": parameters,
            "risk_level": risk_level,
            "evidence_summary": evidence_summary or [],
            "allowed_decisions": allowed_decisions or ["approve", "edit", "reject"],
        },
        run_id,
    )


def create_tool_executed_event(run_id: str, tool_name: str, result: dict[str, Any]) -> str:
    """创建 tool_executed 事件。"""
    return create_sse_event(
        SSEEventType.TOOL_EXECUTED, {"tool_name": tool_name, "result": result}, run_id
    )


def create_answer_ready_event(
    run_id: str, answer: str, citations: list[dict], confidence: float
) -> str:
    """创建 answer_ready 事件。"""
    return create_sse_event(
        SSEEventType.ANSWER_READY,
        {"answer": answer, "citations": citations, "confidence": confidence},
        run_id,
    )


def create_run_failed_event(run_id: str, error: str) -> str:
    """创建 run_failed 事件。"""
    return create_sse_event(SSEEventType.RUN_FAILED, {"error": error}, run_id)
