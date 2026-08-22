"""
IngestionTask 状态模型。

跟踪文档入库的每个阶段：进度、耗时、错误。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class IngestionStage(StrEnum):
    """入库阶段。"""

    QUEUED = "queued"
    PARSING = "parsing"
    CLEANING = "cleaning"
    CHUNKING = "chunking"
    CONTEXTUALIZING = "contextualizing"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


# 阶段 → 进度百分比映射
_STAGE_PROGRESS: dict[IngestionStage, float] = {
    IngestionStage.QUEUED: 0.0,
    IngestionStage.PARSING: 0.1,
    IngestionStage.CLEANING: 0.2,
    IngestionStage.CHUNKING: 0.4,
    IngestionStage.CONTEXTUALIZING: 0.5,
    IngestionStage.EMBEDDING: 0.7,
    IngestionStage.INDEXING: 0.85,
    IngestionStage.READY: 1.0,
    IngestionStage.FAILED: -1.0,  # 失败时不设固定进度
}


class StageRecord(BaseModel):
    """单个阶段执行记录。"""

    stage: IngestionStage
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: int = 0
    success: bool = True
    error_message: str | None = None
    chunk_count: int = 0


class IngestionTask(BaseModel):
    """
    入库任务。

    跟踪文档从上传到入库完成的完整生命周期。
    """

    id: str = Field(default_factory=lambda: f"ing_{uuid.uuid4().hex[:12]}")
    document_id: str
    document_version: str = "v1"
    filename: str
    mime_type: str
    storage_key: str = ""
    current_stage: IngestionStage = IngestionStage.QUEUED
    progress: float = 0.0
    error_message: str | None = None
    error_code: str | None = None
    stages: list[StageRecord] = Field(default_factory=list)
    total_chunks: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def start_stage(self, stage: IngestionStage) -> None:
        """开始一个阶段。"""
        self.current_stage = stage
        self.progress = _STAGE_PROGRESS.get(stage, self.progress)
        self.updated_at = datetime.now(UTC)
        self.stages.append(
            StageRecord(
                stage=stage,
                started_at=datetime.now(UTC),
            )
        )

    def complete_stage(
        self,
        stage: IngestionStage,
        chunk_count: int = 0,
    ) -> None:
        """完成一个阶段。"""
        now = datetime.now(UTC)
        self.updated_at = now
        self.progress = _STAGE_PROGRESS.get(stage, self.progress)
        if chunk_count > 0:
            self.total_chunks = chunk_count

        # 更新对应的 stage record
        for record in reversed(self.stages):
            if record.stage == stage and record.completed_at is None:
                record.completed_at = now
                record.duration_ms = int((now - record.started_at).total_seconds() * 1000)
                record.chunk_count = chunk_count
                break

    def fail_stage(
        self,
        stage: IngestionStage,
        error_message: str,
        error_code: str = "ingestion_error",
    ) -> None:
        """标记阶段失败。"""
        now = datetime.now(UTC)
        self.current_stage = IngestionStage.FAILED
        self.error_message = error_message
        self.error_code = error_code
        self.updated_at = now

        for record in reversed(self.stages):
            if record.stage == stage and record.completed_at is None:
                record.completed_at = now
                record.duration_ms = int((now - record.started_at).total_seconds() * 1000)
                record.success = False
                record.error_message = error_message
                break

    @property
    def is_terminal(self) -> bool:
        """是否已结束（成功或失败）。"""
        return self.current_stage in (IngestionStage.READY, IngestionStage.FAILED)

    def to_status_dict(self) -> dict[str, Any]:
        """转为 API 状态响应格式。"""
        return {
            "task_id": self.id,
            "document_id": self.document_id,
            "status": self.current_stage,
            "current_stage": self.current_stage,
            "progress": self.progress,
            "total_chunks": self.total_chunks,
            "error": self.error_message,
            "error_code": self.error_code,
        }
