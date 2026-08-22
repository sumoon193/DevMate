"""工业化 Agent Runtime ORM 模型测试。"""

from __future__ import annotations

import inspect as pyinspect

from sqlalchemy import create_engine, inspect

import app.models  # noqa: F401
from app.db.crud import upsert_approval
from app.models.base import Base


def test_runtime_metadata_creates_governance_tables() -> None:
    """SQLite fallback 应可创建全部 runtime 治理表。"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    tables = set(inspect(engine).get_table_names())
    assert {
        "cases",
        "runtime_aggregates",
        "runtime_events",
        "outbox_messages",
        "runtime_leases",
        "side_effect_ledger",
        "durable_timers",
        "document_versions",
        "context_snapshots",
        "episodic_memories",
        "skill_manifests",
    }.issubset(tables)


def test_runtime_event_and_side_effect_keys_are_unique() -> None:
    """数据库必须约束 command、sequence 与 side-effect idempotency key。"""
    event_constraints = {
        constraint.name
        for constraint in Base.metadata.tables["runtime_events"].constraints
        if constraint.name
    }
    side_effect_columns = Base.metadata.tables["side_effect_ledger"].columns

    assert "uq_runtime_event_aggregate_sequence" in event_constraints
    assert "uq_runtime_event_command" in event_constraints
    assert side_effect_columns["idempotency_key"].unique is True


def test_approval_model_persists_industrial_authorization_fields() -> None:
    """审批重启恢复必须保留版本、哈希、有效期和撤销审计。"""
    columns = set(Base.metadata.tables["approval_requests"].columns.keys())
    assert {
        "revision",
        "subject_hash",
        "requested_by",
        "requested_at",
        "expires_at",
        "policy_version",
        "execution_manifest_hash",
        "supersedes_approval_id",
        "revoked_by",
        "revoked_at",
        "revoke_reason",
    }.issubset(columns)


def test_approval_upsert_accepts_industrial_authorization_fields() -> None:
    """CRUD 边界必须无损接收审批治理字段。"""
    parameters = set(pyinspect.signature(upsert_approval).parameters)
    assert {
        "revision",
        "subject_hash",
        "requested_by",
        "requested_at",
        "expires_at",
        "policy_version",
        "execution_manifest_hash",
        "supersedes_approval_id",
        "revoked_by",
        "revoked_at",
        "revoke_reason",
    }.issubset(parameters)
