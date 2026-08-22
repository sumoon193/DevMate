"""遵循 write/select/compress/isolate 的结构化上下文压缩。"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime

from app.core.exceptions import ValidationError
from app.schemas.memory import ContextSnapshot
from app.schemas.runtime import RunEventEnvelope


class ContextCompactor:
    """仅压缩自包含事件前缀，并保留治理事件引用。"""

    _PINNED_EVENT_TYPES = {
        "approval.requested",
        "approval.decided",
        "approval.revoked",
        "tool.call_prepared",
        "tool.executed",
        "tool.failed",
        "policy.decision",
        "run.failed",
    }

    def compact(
        self,
        *,
        case_id: str,
        events: list[RunEventEnvelope],
        summarizer_version: str,
        selector_version: str,
    ) -> ContextSnapshot:
        """生成结构化摘要，原始 Event Store 不做删除。"""
        if not events:
            raise ValidationError("Cannot compact an empty event stream")
        safe_prefix = self._safe_prefix(events)
        if not safe_prefix:
            raise ValidationError("No self-contained event prefix is safe to compact")

        summary = {
            "messages": [
                str(event.payload["message"])
                for event in safe_prefix
                if event.event_type == "case.message_added" and "message" in event.payload
            ],
            "citation_ids": [
                citation_id
                for event in safe_prefix
                if event.event_type == "evidence.retrieved"
                for citation_id in event.payload.get("citation_ids", [])
            ],
            "failures": [
                event.payload
                for event in safe_prefix
                if event.event_type in {"tool.failed", "run.failed"}
            ],
        }
        pinned = [event.id for event in events if event.event_type in self._PINNED_EVENT_TYPES]
        before = self._estimate_tokens([event.model_dump(mode="json") for event in safe_prefix])
        after = self._estimate_tokens(summary)
        invariant_hash = hashlib.sha256(
            json.dumps(summary, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return ContextSnapshot(
            id=f"ctx_{uuid.uuid4().hex[:12]}",
            case_id=case_id,
            source_sequence_start=safe_prefix[0].sequence,
            source_sequence_end=safe_prefix[-1].sequence,
            summary=summary,
            pinned_event_ids=pinned,
            token_count_before=before,
            token_count_after=after,
            summarizer_version=summarizer_version,
            selector_version=selector_version,
            invariant_hash=invariant_hash,
            invariant_check_passed=self._validate_invariants(safe_prefix, summary),
            created_at=datetime.now(UTC),
        )

    @staticmethod
    def _safe_prefix(events: list[RunEventEnvelope]) -> list[RunEventEnvelope]:
        """未决审批之前的事件才构成安全自包含前缀。"""
        pending_approval_ids: set[str] = set()
        for event in events:
            if event.event_type == "approval.requested":
                approval_id = str(event.payload.get("approval_id", ""))
                if approval_id:
                    pending_approval_ids.add(approval_id)
            elif event.event_type in {"approval.decided", "approval.revoked"}:
                pending_approval_ids.discard(str(event.payload.get("approval_id", "")))

        if pending_approval_ids:
            boundary = next(
                index
                for index, event in enumerate(events)
                if event.event_type == "approval.requested"
                and str(event.payload.get("approval_id", "")) in pending_approval_ids
            )
            return events[:boundary]
        return list(events)

    @staticmethod
    def _validate_invariants(
        events: list[RunEventEnvelope],
        summary: dict[str, object],
    ) -> bool:
        expected_messages = [
            str(event.payload["message"])
            for event in events
            if event.event_type == "case.message_added" and "message" in event.payload
        ]
        expected_citations = [
            item
            for event in events
            if event.event_type == "evidence.retrieved"
            for item in event.payload.get("citation_ids", [])
        ]
        return (
            summary.get("messages") == expected_messages
            and summary.get("citation_ids") == expected_citations
        )

    @staticmethod
    def _estimate_tokens(value: object) -> int:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return max(1, (len(serialized) + 3) // 4)
