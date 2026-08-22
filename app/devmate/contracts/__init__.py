"""devmate 运行时端口契约包。"""

from app.devmate.contracts.commands import DM02Input, DM02Result, RuntimeEvent
from app.devmate.contracts.ports import (
    CaseRecord,
    CaseStorePort,
    ClockPort,
    EventStreamPort,
)
from app.devmate.contracts.state import (
    LEGAL_TRANSITIONS,
    CaseStatus,
    IllegalTransitionError,
)

__all__ = [
    "LEGAL_TRANSITIONS",
    "CaseRecord",
    "CaseStatus",
    "CaseStorePort",
    "ClockPort",
    "DM02Input",
    "DM02Result",
    "EventStreamPort",
    "IllegalTransitionError",
    "RuntimeEvent",
]
