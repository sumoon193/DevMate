"""可替换时间源，确保定时与过期测试可重复。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol

from app.core.exceptions import ValidationError


class Clock(Protocol):
    """运行时 UTC 时钟接口。"""

    def now(self) -> datetime:
        """返回当前 UTC 时间。"""
        ...


class SystemClock:
    """生产运行使用的系统时钟。"""

    def now(self) -> datetime:
        """返回当前 UTC 时间。"""
        return datetime.now(UTC)


class FakeClock:
    """可显式推进的 deterministic fake clock。"""

    def __init__(self, current: datetime) -> None:
        if current.tzinfo is None:
            raise ValidationError("FakeClock requires a timezone-aware datetime")
        self._current = current.astimezone(UTC)

    def now(self) -> datetime:
        """返回 fake 当前时间。"""
        return self._current

    def advance(self, *, seconds: float) -> datetime:
        """推进 fake 时间并返回新时间。"""
        self._current += timedelta(seconds=seconds)
        return self._current
