"""Agent Runtime、长期 Case、Memory 与 Skill ORM 模型。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IDMixin, TimestampMixin


class CaseRecord(Base, IDMixin, TimestampMixin):
    """长期业务 Case 查询 projection。"""

    __tablename__ = "cases"

    title: Mapped[str] = mapped_column(String(255))
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    subject_user_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    version: Mapped[int] = mapped_column(Integer, default=0)
    active_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    execution_manifest: Mapped[dict] = mapped_column(JSON, default=dict)
    policy_versions: Mapped[dict] = mapped_column(JSON, default=dict)
    working_memory: Mapped[dict] = mapped_column(JSON, default=dict)


class RuntimeAggregateRecord(Base):
    """用于事件追加乐观锁的聚合头。"""

    __tablename__ = "runtime_aggregates"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    aggregate_type: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RuntimeEventRecord(Base):
    """append-only 领域事件。"""

    __tablename__ = "runtime_events"
    __table_args__ = (
        UniqueConstraint(
            "aggregate_id",
            "sequence",
            name="uq_runtime_event_aggregate_sequence",
        ),
        UniqueConstraint("command_id", name="uq_runtime_event_command"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    aggregate_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("runtime_aggregates.id"),
        index=True,
    )
    aggregate_type: Mapped[str] = mapped_column(String(64))
    sequence: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(128), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    command_id: Mapped[str] = mapped_column(String(128))
    actor_id: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    prev_hash: Mapped[str] = mapped_column(String(64), default="")
    event_hash: Mapped[str] = mapped_column(String(64))


class OutboxRecord(Base):
    """与 runtime event 同事务写入的 outbox。"""

    __tablename__ = "outbox_messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("runtime_events.id"),
        unique=True,
    )
    aggregate_id: Mapped[str] = mapped_column(String(64), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    topic: Mapped[str] = mapped_column(String(128), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    claimed_by: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    delivery_attempts: Mapped[int] = mapped_column(Integer, default=0)


class RuntimeLeaseRecord(Base):
    """跨 worker 运行租约。"""

    __tablename__ = "runtime_leases"

    resource_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(64))
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    fencing_token: Mapped[int] = mapped_column(Integer)


class SideEffectLedgerRecord(Base, IDMixin, TimestampMixin):
    """外部副作用 effectively-once 账本。"""

    __tablename__ = "side_effect_ledger"

    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True)
    tool_name: Mapped[str] = mapped_column(String(128))
    subject_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), index=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class DurableTimerRecord(Base, IDMixin, TimestampMixin):
    """SLA、审批过期和长期 Case 唤醒 timer。"""

    __tablename__ = "durable_timers"

    case_id: Mapped[str] = mapped_column(String(64), ForeignKey("cases.id"), index=True)
    timer_type: Mapped[str] = mapped_column(String(128), index=True)
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    claimed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    fired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DocumentVersionRecord(Base, IDMixin, TimestampMixin):
    """不可变文档版本及 active 标记。"""

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version", name="uq_document_version_number"),
        UniqueConstraint("document_id", "content_hash", name="uq_document_content_hash"),
    )

    document_id: Mapped[str] = mapped_column(String(64), ForeignKey("documents.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    supersedes_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class ContextSnapshotRecord(Base, IDMixin, TimestampMixin):
    """结构化上下文压缩快照。"""

    __tablename__ = "context_snapshots"

    case_id: Mapped[str] = mapped_column(String(64), ForeignKey("cases.id"), index=True)
    source_sequence_start: Mapped[int] = mapped_column(Integer)
    source_sequence_end: Mapped[int] = mapped_column(Integer)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    pinned_event_ids: Mapped[list] = mapped_column(JSON, default=list)
    token_count_before: Mapped[int] = mapped_column(Integer)
    token_count_after: Mapped[int] = mapped_column(Integer)
    summarizer_version: Mapped[str] = mapped_column(String(64))
    selector_version: Mapped[str] = mapped_column(String(64))
    invariant_hash: Mapped[str] = mapped_column(String(64))
    invariant_check_passed: Mapped[bool] = mapped_column(Boolean)


class EpisodicMemoryRecordORM(Base, IDMixin, TimestampMixin):
    """带 tenant/provenance 的 episodic memory。"""

    __tablename__ = "episodic_memories"

    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    case_id: Mapped[str] = mapped_column(String(64), ForeignKey("cases.id"), index=True)
    memory_key: Mapped[str] = mapped_column(String(255), index=True)
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    provenance_event_ids: Mapped[list] = mapped_column(JSON, default=list)
    importance_score: Mapped[float] = mapped_column(Float, default=0.5)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(String(32), index=True)
    poisoning_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class HrTicketRecord(Base, IDMixin, TimestampMixin):
    """Reference Application 内部 HR 工单，供生产 MCP 写工具持久化。"""

    __tablename__ = "hr_tickets"

    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    priority: Mapped[str] = mapped_column(String(32), default="medium")
    category: Mapped[str] = mapped_column(String(64), default="其他")
    status: Mapped[str] = mapped_column(String(32), default="created", index=True)
    created_by: Mapped[str] = mapped_column(String(64), index=True)


class SkillManifestRecord(Base, IDMixin, TimestampMixin):
    """Skill 供应链和生命周期记录。"""

    __tablename__ = "skill_manifests"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_skill_name_version"),)

    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str] = mapped_column(String(64))
    content: Mapped[str] = mapped_column(Text)
    checksum: Mapped[str] = mapped_column(String(64))
    source_uri: Mapped[str] = mapped_column(String(512))
    allowed_tools: Mapped[list] = mapped_column(JSON, default=list)
    required_permissions: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), index=True)
    eval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    revoke_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
