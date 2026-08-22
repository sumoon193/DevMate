"""DevMate DM-10 审批 capability 与 revision 失败测试。

合同：``CaseCommand.execute(input: DM10Input) -> DM10Result``。
审批绑定 patch、evidence、命令、主体与过期时间；过期后稳定拒绝。
"""

from __future__ import annotations

import pytest

from app.devmate.approval import (
    ApprovalExpiredError,
    ApprovalNotFoundError,
    ApprovalRequest,
    ApprovalStore,
    CaseCommand,
    DM10Input,
    DM10Result,
    InvalidDecisionError,
)


def _store() -> ApprovalStore:
    return ApprovalStore()


def _request(store: ApprovalStore, *, approval_id: str = "ap-1") -> ApprovalRequest:
    request = ApprovalRequest(
        approval_id=approval_id,
        case_id="case-1",
        patch_id="patch-abc",
        evidence_ids=("ev-1", "ev-2"),
        command="apply patch-abc",
        requested_by="bot-1",
    )
    store.request(request)
    return request


def _input(
    *,
    approval_id: str = "ap-1",
    decision: str = "approve",
    decided_by: str = "u-9",
    decided_at: str = "2026-08-04T12:00:00Z",
) -> DM10Input:
    return DM10Input(
        approval_id=approval_id,
        decision=decision,
        decided_by=decided_by,
        decided_at=decided_at,
    )


def test_case_command_has_typed_entry() -> None:
    store = _store()
    _request(store)

    result = CaseCommand(store).execute(_input())

    assert isinstance(result, DM10Result)
    assert result.status == "approved"


def test_approval_binds_patch_evidence_command_principal_and_expiry() -> None:
    store = _store()
    _request(store)

    result = CaseCommand(store).execute(_input(decided_by="u-9"))

    assert result.patch_id == "patch-abc"
    assert result.evidence_ids == ("ev-1", "ev-2")
    assert result.command == "apply patch-abc"
    assert result.decided_by == "u-9"
    assert result.expires_at


def test_reject_sets_status_rejected() -> None:
    store = _store()
    _request(store)

    result = CaseCommand(store).execute(_input(decision="reject"))

    assert result.status == "rejected"
    assert store.get("ap-1").status == "rejected"


def test_duplicate_decision_is_idempotent() -> None:
    store = _store()
    _request(store)
    command = CaseCommand(store)

    first = command.execute(_input(decision="approve"))
    second = command.execute(_input(decision="approve"))

    assert first == second
    assert store.get("ap-1").status == "approved"


def test_expired_approval_is_rejected() -> None:
    store = _store()
    _request(store)

    with pytest.raises(ApprovalExpiredError):
        CaseCommand(store).execute(_input(decided_at="2026-08-05T00:00:00Z"))


def test_unknown_approval_is_rejected() -> None:
    store = _store()

    with pytest.raises(ApprovalNotFoundError):
        CaseCommand(store).execute(_input(approval_id="missing"))


def test_invalid_decision_is_rejected() -> None:
    store = _store()
    _request(store)

    with pytest.raises(InvalidDecisionError):
        CaseCommand(store).execute(_input(decision="maybe"))


def test_approval_status_tracked_in_store() -> None:
    store = _store()
    _request(store)

    CaseCommand(store).execute(_input())

    assert store.get("ap-1").decided_by == "u-9"
    assert store.get("ap-1").status == "approved"
