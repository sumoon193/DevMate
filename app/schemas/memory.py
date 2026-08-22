"""Context Engineering 与长期记忆 Schema。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.enums import MemoryStatus


class ContextSnapshot(BaseModel):
    """可验证的结构化上下文压缩快照。"""

    id: str = Field(description="Snapshot ID，前缀 ctx_")
    case_id: str
    source_sequence_start: int = Field(ge=1)
    source_sequence_end: int = Field(ge=1)
    summary: dict[str, Any] = Field(default_factory=dict)
    pinned_event_ids: list[str] = Field(default_factory=list)
    token_count_before: int = Field(ge=0)
    token_count_after: int = Field(ge=0)
    summarizer_version: str
    selector_version: str
    invariant_hash: str
    invariant_check_passed: bool
    created_at: datetime


class EpisodicMemoryRecord(BaseModel):
    """带 provenance、ACL 与删除状态的 Case 经验记录。"""

    id: str = Field(description="Memory ID，前缀 mem_")
    tenant_id: str
    case_id: str
    memory_key: str
    content: str
    content_hash: str
    provenance_event_ids: list[str] = Field(min_length=1)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    last_accessed_at: datetime | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    poisoning_reason: str | None = None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
