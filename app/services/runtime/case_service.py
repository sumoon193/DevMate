"""跨轮次 HRCase 应用服务。"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from app.core.exceptions import NotFoundError
from app.schemas.runtime import DurableTimer, ExecutionManifest, HRCase
from app.services.observability.runtime_metrics import RuntimeMetrics
from app.services.runtime.interfaces import CaseProjectionStore, EventStore
from app.services.runtime.projection import CaseProjector


class CaseService:
    """通过事件命令管理长期 Case，并维护同步 fallback projection。"""

    def __init__(
        self,
        event_store: EventStore,
        projection_store: CaseProjectionStore | None = None,
        metrics: RuntimeMetrics | None = None,
    ) -> None:
        self._event_store = event_store
        self._projection_store = projection_store
        self._metrics = metrics
        self._projector = CaseProjector()
        self._projections: dict[str, HRCase] = {}

    async def create_case(
        self,
        *,
        title: str,
        tenant_id: str,
        subject_user_id: str,
        actor_id: str,
        command_id: str,
        execution_manifest: ExecutionManifest,
    ) -> HRCase:
        """创建长期 Case 并记录 execution manifest。"""
        case_id = f"case_{uuid.uuid4().hex[:12]}"
        event = await self._event_store.append(
            aggregate_id=case_id,
            aggregate_type="hr_case",
            event_type="case.created",
            payload={
                "title": title,
                "tenant_id": tenant_id,
                "subject_user_id": subject_user_id,
                "execution_manifest": execution_manifest.model_dump(mode="json"),
                "policy_versions": {"hr_policy": execution_manifest.policy_version},
            },
            command_id=command_id,
            expected_version=0,
            actor_id=actor_id,
        )
        projection = self._projector.apply(None, event)
        self._projections[case_id] = projection
        await self._save_projection(projection)
        return projection.model_copy(deep=True)

    async def add_message(
        self,
        *,
        case_id: str,
        message: str,
        actor_id: str,
        command_id: str,
        expected_version: int,
    ) -> HRCase:
        """向 Case 追加跨轮次用户或操作员消息。"""
        current = await self._get_projection(case_id)
        event = await self._event_store.append(
            aggregate_id=case_id,
            aggregate_type="hr_case",
            event_type="case.message_added",
            payload={"message": message},
            command_id=command_id,
            expected_version=expected_version,
            actor_id=actor_id,
        )
        projection = self._projector.apply(current, event)
        self._projections[case_id] = projection
        await self._save_projection(projection)
        return projection.model_copy(deep=True)

    async def get_case(self, case_id: str) -> HRCase:
        """查询 Case projection。"""
        return (await self._get_projection(case_id)).model_copy(deep=True)

    async def list_cases(self, *, limit: int = 100) -> list[HRCase]:
        """按最近更新时间返回 Case 查询 projection。"""
        if self._projection_store is not None:
            return await self._projection_store.list(limit=limit)
        return sorted(
            (case.model_copy(deep=True) for case in self._projections.values()),
            key=lambda case: (case.updated_at, case.id),
            reverse=True,
        )[:limit]

    async def rebuild(self, case_id: str) -> HRCase:
        """从 Event Store 重建并替换在线 projection。"""
        events = await self._event_store.load_stream(case_id)
        if not events:
            raise NotFoundError(f"Case not found: {case_id}")
        projection = self._projector.rebuild(events)
        self._projections[case_id] = projection
        await self._save_projection(projection)
        if self._metrics is not None:
            self._metrics.increment("runtime.crash_recovery.success")
        return projection.model_copy(deep=True)

    async def record_timer_scheduled(
        self,
        *,
        timer: DurableTimer,
        actor_id: str,
        command_id: str,
        expected_version: int,
    ) -> HRCase:
        """记录 timer 调度并将 Case 投影为等待定时器。"""
        return await self._append_and_project(
            case_id=timer.case_id,
            event_type="timer.scheduled",
            payload={
                "timer_id": timer.id,
                "timer_type": timer.timer_type,
                "due_at": timer.due_at.isoformat(),
            },
            actor_id=actor_id,
            command_id=command_id,
            expected_version=expected_version,
        )

    async def record_timer_fired(
        self,
        *,
        timer: DurableTimer,
        actor_id: str,
        expected_version: int,
    ) -> HRCase:
        """记录 timer 到期唤醒事件。"""
        return await self._append_and_project(
            case_id=timer.case_id,
            event_type="timer.fired",
            payload={"timer_id": timer.id, "timer_type": timer.timer_type},
            actor_id=actor_id,
            command_id=f"timer:{timer.id}:fire",
            expected_version=expected_version,
        )

    async def record_event(
        self,
        *,
        case_id: str,
        event_type: str,
        payload: dict[str, Any],
        actor_id: str,
        command_id: str,
        expected_version: int,
    ) -> HRCase:
        """追加通用 Case 运行时事件并更新 projection。"""
        return await self._append_and_project(
            case_id=case_id,
            event_type=event_type,
            payload=payload,
            actor_id=actor_id,
            command_id=command_id,
            expected_version=expected_version,
        )

    async def _append_and_project(
        self,
        *,
        case_id: str,
        event_type: str,
        payload: dict[str, Any],
        actor_id: str,
        command_id: str,
        expected_version: int,
    ) -> HRCase:
        """追加 Case 事件并同步更新 fallback projection。"""
        current = await self._get_projection(case_id)
        event = await self._event_store.append(
            aggregate_id=case_id,
            aggregate_type="hr_case",
            event_type=event_type,
            payload=dict(payload),
            command_id=command_id,
            expected_version=expected_version,
            actor_id=actor_id,
        )
        projection = self._projector.apply(current, event)
        self._projections[case_id] = projection
        await self._save_projection(projection)
        return projection.model_copy(deep=True)

    async def _get_projection(self, case_id: str) -> HRCase:
        projection = self._projections.get(case_id)
        if projection is None and self._projection_store is not None:
            projection = await self._projection_store.get(case_id)
            if projection is not None:
                self._projections[case_id] = projection
        if projection is None:
            events = await self._event_store.load_stream(case_id)
            if events:
                projection = self._projector.rebuild(events)
                self._projections[case_id] = projection
                await self._save_projection(projection)
        if projection is None:
            raise NotFoundError(f"Case not found: {case_id}")
        return projection

    async def _save_projection(self, projection: HRCase) -> None:
        """幂等同步查询 projection；事件流仍是审计事实源。"""
        if self._projection_store is not None:
            await self._projection_store.upsert(projection)
            if self._metrics is not None:
                lag_ms = max(
                    0.0,
                    (datetime.now(UTC) - projection.updated_at).total_seconds() * 1000,
                )
                self._metrics.observe("runtime.projection.lag_ms", lag_ms)
