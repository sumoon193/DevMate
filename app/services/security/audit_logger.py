"""
安全审计日志。

记录安全相关事件，用于审计和追踪。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


class SecurityEventType:
    """安全事件类型常量。"""

    ACCESS_DENIED = "access_denied"
    PERMISSION_VIOLATION = "permission_violation"
    PROMPT_INJECTION = "prompt_injection"
    PII_DETECTED = "pii_detected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TOOL_EXECUTION_BLOCKED = "tool_execution_blocked"
    AUTHENTICATION_FAILURE = "authentication_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class AuditLogger:
    """
    安全审计日志记录器。

    记录所有安全相关事件。
    """

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def log_security_event(
        self, event_type: str, user_id: str, details: dict[str, Any], severity: str = "warning"
    ) -> None:
        """
        记录安全事件。

        Args:
            event_type: 事件类型
            user_id: 用户 ID
            details: 事件详情
            severity: 严重程度（info, warning, error）
        """
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "severity": severity,
            "details": details,
        }

        self._events.append(event)

        # 使用标准库 logging 记录
        log_method = getattr(logger, severity, logger.warning)
        log_method(
            f"security_event:{event_type}",
            extra={"user_id": user_id, "event_type": event_type, "severity": severity, **details},
        )

    def log_access_denied(self, user_id: str, resource: str, reason: str) -> None:
        """记录访问拒绝事件。"""
        self.log_security_event(
            SecurityEventType.ACCESS_DENIED,
            user_id,
            {"resource": resource, "reason": reason},
            severity="warning",
        )

    def log_prompt_injection(self, user_id: str, text_preview: str, reason: str) -> None:
        """记录 Prompt Injection 检测事件。"""
        self.log_security_event(
            SecurityEventType.PROMPT_INJECTION,
            user_id,
            {"text_preview": text_preview[:100], "reason": reason},
            severity="error",
        )

    def log_rate_limit_exceeded(self, user_id: str, action: str) -> None:
        """记录速率限制超限事件。"""
        self.log_security_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED, user_id, {"action": action}, severity="warning"
        )

    def log_tool_blocked(self, user_id: str, tool_name: str, reason: str) -> None:
        """记录工具执行阻止事件。"""
        self.log_security_event(
            SecurityEventType.TOOL_EXECUTION_BLOCKED,
            user_id,
            {"tool_name": tool_name, "reason": reason},
            severity="warning",
        )

    def get_events(
        self, user_id: str | None = None, event_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        获取安全事件。

        Args:
            user_id: 过滤用户 ID（可选）
            event_type: 过滤事件类型（可选）
            limit: 返回数量限制

        Returns:
            安全事件列表
        """
        events = self._events

        if user_id:
            events = [e for e in events if e["user_id"] == user_id]

        if event_type:
            events = [e for e in events if e["event_type"] == event_type]

        return events[-limit:]

    def get_event_count(self, user_id: str | None = None, event_type: str | None = None) -> int:
        """
        获取安全事件数量。

        Args:
            user_id: 过滤用户 ID（可选）
            event_type: 过滤事件类型（可选）

        Returns:
            事件数量
        """
        events = self._events

        if user_id:
            events = [e for e in events if e["user_id"] == user_id]

        if event_type:
            events = [e for e in events if e["event_type"] == event_type]

        return len(events)
