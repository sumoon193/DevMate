"""DevMate DM-12 并发、租约与崩溃恢复失败测试。

合同：``CheckpointPort.execute(input: DM12Input) -> DM12Result``。
并发 resume 不重复副作用；过期 owner 不再写入，可被新 owner 重新获取。
"""

from __future__ import annotations

import pytest

from app.devmate.recovery import (
    DM12Input,
    DM12Result,
    LeaseConflictError,
    LeaseExpiredError,
    NotLeaseOwnerError,
    RecoveryCheckpoint,
)


def _input(
    *,
    case_id: str = "case-1",
    owner: str = "worker-A",
    action: str = "acquire",
    side_effect_id: str = "",
    now: int = 0,
) -> DM12Input:
    return DM12Input(
        case_id=case_id,
        owner=owner,
        action=action,
        side_effect_id=side_effect_id,
        now=now,
    )


def test_checkpoint_port_has_typed_entry() -> None:
    result = RecoveryCheckpoint().execute(_input())

    assert isinstance(result, DM12Result)
    assert result.lease is not None
    assert result.lease.owner == "worker-A"


def test_acquire_creates_lease_with_expiry() -> None:
    result = RecoveryCheckpoint(ttl=30).execute(_input(now=10))

    assert result.lease.expires_at == 40
    assert result.duplicate is False


def test_active_lease_conflicts_with_other_owner() -> None:
    checkpoint = RecoveryCheckpoint()
    checkpoint.execute(_input(owner="worker-A", now=0))

    with pytest.raises(LeaseConflictError):
        checkpoint.execute(_input(owner="worker-B", now=5))


def test_expired_lease_can_be_reacquired() -> None:
    checkpoint = RecoveryCheckpoint(ttl=30)
    checkpoint.execute(_input(owner="worker-A", now=0))

    result = checkpoint.execute(_input(owner="worker-B", now=31))

    assert result.lease.owner == "worker-B"
    assert result.lease.version == 2


def test_expired_owner_resume_is_rejected() -> None:
    checkpoint = RecoveryCheckpoint(ttl=30)
    checkpoint.execute(_input(owner="worker-A", now=0))

    with pytest.raises(LeaseExpiredError):
        checkpoint.execute(_input(action="resume", side_effect_id="eff-1", now=31))


def test_concurrent_resume_no_duplicate_side_effect() -> None:
    checkpoint = RecoveryCheckpoint(ttl=30)
    checkpoint.execute(_input(owner="worker-A", now=0))

    first = checkpoint.execute(_input(action="resume", side_effect_id="eff-1", now=5))
    second = checkpoint.execute(_input(action="resume", side_effect_id="eff-1", now=6))

    assert first.duplicate is False
    assert second.duplicate is True


def test_non_owner_resume_is_rejected() -> None:
    checkpoint = RecoveryCheckpoint(ttl=30)
    checkpoint.execute(_input(owner="worker-A", now=0))

    with pytest.raises(NotLeaseOwnerError):
        checkpoint.execute(_input(action="resume", owner="worker-B", side_effect_id="eff-1", now=5))


def test_release_clears_lease() -> None:
    checkpoint = RecoveryCheckpoint(ttl=30)
    checkpoint.execute(_input(owner="worker-A", now=0))

    released = checkpoint.execute(_input(action="release", owner="worker-A", now=5))
    reacquired = checkpoint.execute(_input(owner="worker-B", now=6))

    assert released.lease is None
    assert reacquired.lease.owner == "worker-B"
