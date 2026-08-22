"""租约 Checkpoint：并发 resume 不重复副作用，过期 owner 拒绝写入。

合同：``CheckpointPort.execute(input: DM12Input) -> DM12Result``。
"""

from __future__ import annotations

from typing import Protocol

from app.devmate.recovery.store import RecoveryStore
from app.devmate.recovery.types import DM12Input, DM12Result


class LeaseExpiredError(RuntimeError):
    def __init__(self, case_id: str) -> None:
        super().__init__(f"lease expired for {case_id}")
        self.case_id = case_id


class NotLeaseOwnerError(RuntimeError):
    def __init__(self, case_id: str, owner: str) -> None:
        super().__init__(f"{owner} does not hold the lease for {case_id}")
        self.case_id = case_id
        self.owner = owner


class InvalidActionError(ValueError):
    def __init__(self, action: str) -> None:
        super().__init__(f"invalid action: {action}")
        self.action = action


class CheckpointPort(Protocol):
    def execute(self, input_: DM12Input) -> DM12Result: ...


class RecoveryCheckpoint:
    def __init__(self, store: RecoveryStore | None = None, ttl: int = 30) -> None:
        self._store = store or RecoveryStore()
        self._ttl = ttl

    def execute(self, input_: DM12Input) -> DM12Result:
        if input_.action == "acquire":
            lease = self._store.acquire(input_.case_id, input_.owner, input_.now, self._ttl)
            return DM12Result(
                case_id=input_.case_id,
                lease=lease,
                owner=input_.owner,
                side_effect_id=None,
                duplicate=False,
                audit={"action": "acquire"},
            )
        if input_.action == "resume":
            return self._resume(input_)
        if input_.action == "release":
            return self._release(input_)
        raise InvalidActionError(input_.action)

    def _resume(self, input_: DM12Input) -> DM12Result:
        lease = self._store.get_lease(input_.case_id)
        if lease is None or lease.expires_at < input_.now:
            raise LeaseExpiredError(input_.case_id)
        if lease.owner != input_.owner:
            raise NotLeaseOwnerError(input_.case_id, input_.owner)
        if self._store.applied(input_.case_id, input_.side_effect_id):
            return DM12Result(
                case_id=input_.case_id,
                lease=lease,
                owner=input_.owner,
                side_effect_id=input_.side_effect_id,
                duplicate=True,
                audit={"action": "resume", "duplicate": "true"},
            )
        self._store.mark_applied(input_.case_id, input_.side_effect_id)
        return DM12Result(
            case_id=input_.case_id,
            lease=lease,
            owner=input_.owner,
            side_effect_id=input_.side_effect_id,
            duplicate=False,
            audit={"action": "resume"},
        )

    def _release(self, input_: DM12Input) -> DM12Result:
        lease = self._store.get_lease(input_.case_id)
        if lease is None or lease.owner != input_.owner:
            raise NotLeaseOwnerError(input_.case_id, input_.owner)
        self._store.release(input_.case_id)
        return DM12Result(
            case_id=input_.case_id,
            lease=None,
            owner=input_.owner,
            side_effect_id=None,
            duplicate=False,
            audit={"action": "release"},
        )
