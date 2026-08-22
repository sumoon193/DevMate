"""devmate Case 状态机与存储：typed command 推进合法状态。"""

from __future__ import annotations

from app.devmate.cases.command import CaseCommand, DM04Input, DM04Result
from app.devmate.cases.state import LEGAL_TRANSITIONS, CaseStatus, IllegalTransitionError
from app.devmate.cases.store import (
    CaseNotFoundError,
    CaseRecord,
    CaseStore,
    DuplicateCaseError,
    TimelineEvent,
)

__all__ = [
    "LEGAL_TRANSITIONS",
    "CaseCommand",
    "CaseNotFoundError",
    "CaseRecord",
    "CaseStatus",
    "CaseStore",
    "DM04Input",
    "DM04Result",
    "DuplicateCaseError",
    "IllegalTransitionError",
    "TimelineEvent",
]
