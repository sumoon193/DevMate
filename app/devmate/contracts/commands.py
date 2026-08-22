"""devmate 运行时 typed command 入口。

合同：``RuntimeEvent.execute(input: DM02Input) -> DM02Result``。
Runtime 候选只通过本入口与端口契约推进 Case 状态，不直接依赖 HR/RAG
领域接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.devmate.contracts.state import (
    LEGAL_TRANSITIONS,
    CaseStatus,
    IllegalTransitionError,
)


@dataclass(frozen=True)
class DM02Input:
    runtime_id: str
    event_type: str
    payload: dict[str, Any]
    current_status: CaseStatus = CaseStatus.CREATED
    target_status: CaseStatus = CaseStatus.RUNNING


@dataclass(frozen=True)
class DM02Result:
    runtime_id: str
    status: CaseStatus
    state_event: str
    audit_info: dict[str, str]


class RuntimeEvent:
    """Runtime 候选的 typed command。"""

    @staticmethod
    def execute(input_: DM02Input) -> DM02Result:
        if input_.target_status not in LEGAL_TRANSITIONS[input_.current_status]:
            raise IllegalTransitionError(
                f"illegal transition {input_.current_status.value} -> {input_.target_status.value}"
            )
        return DM02Result(
            runtime_id=input_.runtime_id,
            status=input_.target_status,
            state_event=(
                f"runtime {input_.runtime_id} {input_.current_status.value}"
                f" -> {input_.target_status.value}"
            ),
            audit_info={
                "event_type": input_.event_type,
                "payload_keys": ",".join(sorted(input_.payload)),
            },
        )
