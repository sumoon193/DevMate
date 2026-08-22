"""
异步 CRUD 操作。

为 AgentRun、AgentStep、ApprovalRequest、ToolCall 提供数据库持久化。
仅在 app_mode=full 时使用。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_run import AgentRun
from app.models.agent_step import AgentStep
from app.models.approval import ApprovalRequest
from app.models.tool_call import ToolCall
from app.schemas.enums import (
    ApprovalDecisionType,
    ApprovalStatus,
    RunStatus,
    ToolCallStatus,
)

if TYPE_CHECKING:
    from app.models.ingestion_task import IngestionTaskRecord

logger = logging.getLogger(__name__)
_UNSET = object()


def _enum_value(value: Any) -> Any:
    """将 StrEnum 等枚举值转换为数据库中存储的字符串。"""
    if value is None:
        return None
    return value.value if hasattr(value, "value") else value


def _enum_values(values: list[Any] | None) -> list[Any] | None:
    """批量转换可能包含枚举的列表。"""
    if values is None:
        return None
    return [_enum_value(value) for value in values]


# ── AgentRun ──────────────────────────────────────────────────────────


async def save_run(
    session: AsyncSession,
    *,
    run_id: str,
    user_id: str,
    thread_id: str,
    original_query: str,
    status: RunStatus = RunStatus.CREATED,
) -> AgentRun:
    """创建新的 AgentRun 记录。"""
    run = AgentRun(
        id=run_id,
        user_id=user_id,
        thread_id=thread_id,
        original_query=original_query,
        status=_enum_value(status),
    )
    session.add(run)
    await session.commit()
    logger.info("db_run_saved", extra={"run_id": run_id})
    return run


async def get_run(session: AsyncSession, run_id: str) -> AgentRun | None:
    """按 ID 获取 AgentRun。"""
    result = await session.execute(select(AgentRun).where(AgentRun.id == run_id))
    return result.scalar_one_or_none()


async def update_run_status(
    session: AsyncSession,
    run_id: str,
    status: RunStatus,
    result_data: dict[str, Any] | None = None,
) -> None:
    """更新 Run 状态和结果。"""
    values: dict[str, Any] = {"status": _enum_value(status)}
    if result_data is not None:
        values["result"] = result_data
    if status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
        values["completed_at"] = datetime.now(UTC)

    await session.execute(update(AgentRun).where(AgentRun.id == run_id).values(**values))
    await session.commit()
    logger.info("db_run_status_updated", extra={"run_id": run_id, "status": _enum_value(status)})


# ── AgentStep ─────────────────────────────────────────────────────────


async def save_step(
    session: AsyncSession,
    *,
    run_id: str,
    node_name: str,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    evidence: list[Any] | None = None,
    duration_ms: int = 0,
    step_id: str | None = None,
) -> AgentStep:
    """保存一个 AgentStep。"""
    import uuid as _uuid

    step = AgentStep(
        id=step_id or f"step_{_uuid.uuid4().hex[:12]}",
        run_id=run_id,
        node_name=node_name,
        input_data=input_data or {},
        output_data=output_data or {},
        evidence=evidence or [],
        duration_ms=duration_ms,
    )
    session.add(step)
    await session.commit()
    return step


async def get_steps(session: AsyncSession, run_id: str) -> list[AgentStep]:
    """获取指定 Run 的所有步骤。"""
    result = await session.execute(
        select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.created_at)
    )
    return list(result.scalars().all())


# ── ApprovalRequest ───────────────────────────────────────────────────


async def save_approval(
    session: AsyncSession,
    *,
    approval_id: str,
    run_id: str,
    tool_call_id: str,
    tool_name: str,
    parameters: dict[str, Any],
    expected_effect: str,
    evidence: list[Any] | None = None,
    risk_level: str = "write",
    options: list[Any] | None = None,
    status: ApprovalStatus = ApprovalStatus.PENDING,
    decision: ApprovalDecisionType | None = None,
    decided_by: str | None = None,
    decided_at: datetime | None = None,
    revision: int = 1,
    subject_hash: str = "",
    requested_by: str | None = None,
    requested_at: datetime | None = None,
    expires_at: datetime | None = None,
    policy_version: str = "",
    execution_manifest_hash: str = "",
    supersedes_approval_id: str | None = None,
    revoked_by: str | None = None,
    revoked_at: datetime | None = None,
    revoke_reason: str | None = None,
) -> ApprovalRequest:
    """创建审批请求。"""
    if options is None:
        options = [
            ApprovalDecisionType.APPROVE.value,
            ApprovalDecisionType.EDIT.value,
            ApprovalDecisionType.REJECT.value,
        ]
    approval = ApprovalRequest(
        id=approval_id,
        run_id=run_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        parameters=parameters,
        expected_effect=expected_effect,
        evidence=evidence or [],
        risk_level=_enum_value(risk_level),
        options=_enum_values(options) or [],
        status=_enum_value(status),
        decision=_enum_value(decision),
        decided_by=decided_by,
        decided_at=decided_at,
        revision=revision,
        subject_hash=subject_hash,
        requested_by=requested_by,
        requested_at=requested_at,
        expires_at=expires_at,
        policy_version=policy_version,
        execution_manifest_hash=execution_manifest_hash,
        supersedes_approval_id=supersedes_approval_id,
        revoked_by=revoked_by,
        revoked_at=revoked_at,
        revoke_reason=revoke_reason,
    )
    session.add(approval)
    await session.commit()
    logger.info("db_approval_saved", extra={"approval_id": approval_id})
    return approval


async def upsert_approval(
    session: AsyncSession,
    *,
    approval_id: str,
    run_id: str,
    tool_call_id: str,
    tool_name: str,
    parameters: dict[str, Any],
    expected_effect: str,
    evidence: list[Any] | None = None,
    risk_level: str = "write",
    options: list[Any] | None = None,
    status: ApprovalStatus = ApprovalStatus.PENDING,
    decision: ApprovalDecisionType | None = None,
    decided_by: str | None = None,
    decided_at: datetime | None = None,
    revision: int = 1,
    subject_hash: str = "",
    requested_by: str | None = None,
    requested_at: datetime | None = None,
    expires_at: datetime | None = None,
    policy_version: str = "",
    execution_manifest_hash: str = "",
    supersedes_approval_id: str | None = None,
    revoked_by: str | None = None,
    revoked_at: datetime | None = None,
    revoke_reason: str | None = None,
) -> ApprovalRequest:
    """新增或更新审批请求快照。"""
    existing = await get_approval(session, approval_id)
    if existing is None:
        return await save_approval(
            session,
            approval_id=approval_id,
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            parameters=parameters,
            expected_effect=expected_effect,
            evidence=evidence,
            risk_level=risk_level,
            options=_enum_values(options),
            status=status,
            decision=decision,
            decided_by=decided_by,
            decided_at=decided_at,
            revision=revision,
            subject_hash=subject_hash,
            requested_by=requested_by,
            requested_at=requested_at,
            expires_at=expires_at,
            policy_version=policy_version,
            execution_manifest_hash=execution_manifest_hash,
            supersedes_approval_id=supersedes_approval_id,
            revoked_by=revoked_by,
            revoked_at=revoked_at,
            revoke_reason=revoke_reason,
        )

    values = {
        "run_id": run_id,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "parameters": parameters,
        "expected_effect": expected_effect,
        "evidence": evidence or [],
        "risk_level": _enum_value(risk_level),
        "options": _enum_values(options)
        or [
            ApprovalDecisionType.APPROVE.value,
            ApprovalDecisionType.EDIT.value,
            ApprovalDecisionType.REJECT.value,
        ],
        "status": _enum_value(status),
        "decision": _enum_value(decision),
        "decided_by": decided_by,
        "decided_at": decided_at,
        "revision": revision,
        "subject_hash": subject_hash,
        "requested_by": requested_by,
        "requested_at": requested_at,
        "expires_at": expires_at,
        "policy_version": policy_version,
        "execution_manifest_hash": execution_manifest_hash,
        "supersedes_approval_id": supersedes_approval_id,
        "revoked_by": revoked_by,
        "revoked_at": revoked_at,
        "revoke_reason": revoke_reason,
    }
    await session.execute(
        update(ApprovalRequest).where(ApprovalRequest.id == approval_id).values(**values)
    )
    await session.commit()
    logger.info("db_approval_upserted", extra={"approval_id": approval_id})
    refreshed = await get_approval(session, approval_id)
    if refreshed is None:
        raise RuntimeError(f"Approval disappeared after upsert: {approval_id}")
    return refreshed


async def get_approval(session: AsyncSession, approval_id: str) -> ApprovalRequest | None:
    """按 ID 获取审批请求。"""
    result = await session.execute(select(ApprovalRequest).where(ApprovalRequest.id == approval_id))
    return result.scalar_one_or_none()


async def update_approval(
    session: AsyncSession,
    approval_id: str,
    status: ApprovalStatus,
    decision: ApprovalDecisionType | None = None,
    decided_by: str | None = None,
) -> None:
    """更新审批状态和决策。"""
    values: dict[str, Any] = {"status": _enum_value(status)}
    if decision is not None:
        values["decision"] = _enum_value(decision)
    if decided_by is not None:
        values["decided_by"] = decided_by
    if status != ApprovalStatus.PENDING:
        values["decided_at"] = datetime.now(UTC)

    await session.execute(
        update(ApprovalRequest).where(ApprovalRequest.id == approval_id).values(**values)
    )
    await session.commit()
    logger.info(
        "db_approval_updated",
        extra={"approval_id": approval_id, "status": _enum_value(status)},
    )


async def get_pending_approvals(session: AsyncSession, run_id: str) -> list[ApprovalRequest]:
    """获取指定 Run 的所有待审批请求。"""
    result = await session.execute(
        select(ApprovalRequest).where(
            ApprovalRequest.run_id == run_id,
            ApprovalRequest.status == ApprovalStatus.PENDING,
        )
    )
    return list(result.scalars().all())


async def get_all_approvals(session: AsyncSession, run_id: str) -> list[ApprovalRequest]:
    """获取指定 Run 的所有审批请求（含已决定）。"""
    result = await session.execute(select(ApprovalRequest).where(ApprovalRequest.run_id == run_id))
    return list(result.scalars().all())


# ── ToolCall ──────────────────────────────────────────────────────────


async def save_tool_call(
    session: AsyncSession,
    *,
    tool_call_id: str,
    run_id: str,
    tool_name: str,
    parameters: dict[str, Any],
    status: ToolCallStatus = ToolCallStatus.PENDING,
    approval_required: bool = False,
    result_data: dict[str, Any] | None = None,
) -> ToolCall:
    """保存工具调用记录。"""
    tc = ToolCall(
        id=tool_call_id,
        run_id=run_id,
        tool_name=tool_name,
        parameters=parameters,
        status=_enum_value(status),
        approval_required=approval_required,
        result=result_data,
    )
    session.add(tc)
    await session.commit()
    logger.info("db_tool_call_saved", extra={"tool_call_id": tool_call_id})
    return tc


async def get_tool_call(session: AsyncSession, tool_call_id: str) -> ToolCall | None:
    """按 ID 获取工具调用记录。"""
    result = await session.execute(select(ToolCall).where(ToolCall.id == tool_call_id))
    return result.scalar_one_or_none()


async def upsert_tool_call(
    session: AsyncSession,
    *,
    tool_call_id: str,
    run_id: str,
    tool_name: str,
    parameters: dict[str, Any],
    status: ToolCallStatus = ToolCallStatus.PENDING,
    approval_required: bool = False,
    result_data: dict[str, Any] | None = None,
) -> ToolCall:
    """新增或更新工具调用快照。"""
    existing = await get_tool_call(session, tool_call_id)
    if existing is None:
        return await save_tool_call(
            session,
            tool_call_id=tool_call_id,
            run_id=run_id,
            tool_name=tool_name,
            parameters=parameters,
            status=status,
            approval_required=approval_required,
            result_data=result_data,
        )

    await session.execute(
        update(ToolCall)
        .where(ToolCall.id == tool_call_id)
        .values(
            run_id=run_id,
            tool_name=tool_name,
            parameters=parameters,
            status=_enum_value(status),
            approval_required=approval_required,
            result=result_data,
        )
    )
    await session.commit()
    logger.info("db_tool_call_upserted", extra={"tool_call_id": tool_call_id})
    refreshed = await get_tool_call(session, tool_call_id)
    if refreshed is None:
        raise RuntimeError(f"Tool call disappeared after upsert: {tool_call_id}")
    return refreshed


async def update_tool_call(
    session: AsyncSession,
    tool_call_id: str,
    status: ToolCallStatus,
    result_data: dict[str, Any] | None = None,
) -> None:
    """更新工具调用状态和结果。"""
    values: dict[str, Any] = {"status": _enum_value(status)}
    if result_data is not None:
        values["result"] = result_data
    await session.execute(update(ToolCall).where(ToolCall.id == tool_call_id).values(**values))
    await session.commit()


# ── IngestionTaskRecord ───────────────────────────────────────────────


async def save_ingestion_task(
    session: AsyncSession,
    *,
    task_id: str,
    document_id: str,
    filename: str,
    mime_type: str,
    storage_key: str = "",
    current_stage: str = "queued",
    progress: float = 0.0,
    total_chunks: int = 0,
    error_message: str | None = None,
    error_code: str | None = None,
    stages_json: list[Any] | None = None,
) -> IngestionTaskRecord:
    """创建入库任务记录。"""
    from app.models.ingestion_task import IngestionTaskRecord

    record = IngestionTaskRecord(
        id=task_id,
        document_id=document_id,
        filename=filename,
        mime_type=mime_type,
        storage_key=storage_key,
        current_stage=current_stage,
        progress=progress,
        total_chunks=total_chunks,
        error_message=error_message,
        error_code=error_code,
        stages_json=stages_json or [],
    )
    session.add(record)
    await session.commit()
    logger.info("db_ingestion_task_saved", extra={"task_id": task_id})
    return record


async def update_ingestion_task(
    session: AsyncSession,
    task_id: str,
    *,
    current_stage: str | None = None,
    progress: float | None = None,
    total_chunks: int | None = None,
    error_message: str | object | None = _UNSET,
    error_code: str | object | None = _UNSET,
    stages_json: list[Any] | None = None,
) -> None:
    """更新入库任务状态。"""
    from app.models.ingestion_task import IngestionTaskRecord

    values: dict[str, Any] = {}
    if current_stage is not None:
        values["current_stage"] = current_stage
    if progress is not None:
        values["progress"] = progress
    if total_chunks is not None:
        values["total_chunks"] = total_chunks
    if error_message is not _UNSET:
        values["error_message"] = error_message
    if error_code is not _UNSET:
        values["error_code"] = error_code
    if stages_json is not None:
        values["stages_json"] = stages_json

    if not values:
        return

    await session.execute(
        update(IngestionTaskRecord).where(IngestionTaskRecord.id == task_id).values(**values)
    )
    await session.commit()


async def get_ingestion_task(
    session: AsyncSession,
    task_id: str,
) -> IngestionTaskRecord | None:
    """按 ID 获取入库任务。"""
    from app.models.ingestion_task import IngestionTaskRecord

    result = await session.execute(
        select(IngestionTaskRecord).where(IngestionTaskRecord.id == task_id)
    )
    return result.scalar_one_or_none()
