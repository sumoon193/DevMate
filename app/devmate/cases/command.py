"""devmate Case 状态推进的 typed command。

合同：``CaseCommand.execute(input: DM04Input) -> DM04Result``。
状态推进必须经过合法状态机；command_id 幂等，重放返回同一结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.devmate.cases.state import LEGAL_TRANSITIONS, CaseStatus, IllegalTransitionError
from app.devmate.cases.store import CaseNotFoundError, CaseStore


@dataclass(frozen=True)
class DM04Input:
    case_id: str
    command_id: str
    event_type: str
    actor_id: str
    target_status: CaseStatus
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DM04Result:
    case_id: str
    status: CaseStatus
    command_id: str
    state_event: str
    audit_info: dict[str, str]


class CaseCommand:
    def __init__(self, store: CaseStore) -> None:
        self._store = store
        self._results: dict[tuple[str, str], DM04Result] = {}

    def execute(self, input_: DM04Input) -> DM04Result:
        key = (input_.case_id, input_.command_id)
        if key in self._results:
            return self._results[key]
        record = self._store.get(input_.case_id)
        if record is None:
            raise CaseNotFoundError(input_.case_id)
        if input_.target_status not in LEGAL_TRANSITIONS[record.status]:
            raise IllegalTransitionError(
                f"illegal transition {record.status.value} -> {input_.target_status.value}"
            )
        new_record = self._store.advance(
            case_id=input_.case_id,
            target_status=input_.target_status,
            command_id=input_.command_id,
            event_type=input_.event_type,
            actor_id=input_.actor_id,
            payload=input_.payload,
        )
        result = DM04Result(
            case_id=input_.case_id,
            status=new_record.status,
            command_id=input_.command_id,
            state_event=(
                f"case {input_.case_id} {record.status.value} -> {new_record.status.value}"
            ),
            audit_info={
                "event_type": input_.event_type,
                "actor_id": input_.actor_id,
            },
        )
        self._results[key] = result
        return result
