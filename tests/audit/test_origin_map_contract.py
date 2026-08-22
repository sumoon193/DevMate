"""DevMate W1-C1 origin map contract tests."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGIN_MAP_PATH = REPO_ROOT / "docs" / "audit" / "origin-map.jsonl"

REQUIRED_FIELDS = {
    "path",
    "source_commit",
    "license_status",
    "domain",
    "reuse_decision",
}
ALLOWED_DOMAINS = {"runtime_kernel", "hr", "rag", "unknown"}
ALLOWED_REUSE_DECISIONS = {"allowed", "isolate", "blocked", "review"}


def _read_origin_map() -> list[dict[str, object]]:
    with ORIGIN_MAP_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_origin_map_records_have_required_fields_and_valid_enums() -> None:
    records = _read_origin_map()

    assert records, "origin map must contain at least one audit record"
    for record in records:
        assert record.keys() >= REQUIRED_FIELDS
        assert record["domain"] in ALLOWED_DOMAINS
        assert record["reuse_decision"] in ALLOWED_REUSE_DECISIONS
        assert str(record["path"]).strip()
        assert str(record["source_commit"]).strip()
        assert str(record["license_status"]).strip()


def test_allowed_reuse_requires_evidence_and_confirmed_license() -> None:
    for record in _read_origin_map():
        if record["reuse_decision"] != "allowed":
            continue

        evidence_refs = record.get("evidence_refs")
        assert isinstance(evidence_refs, list) and evidence_refs
        assert record["license_status"] not in {"unknown", "unconfirmed"}


def test_unknown_license_cannot_be_marked_allowed() -> None:
    for record in _read_origin_map():
        if record["license_status"] in {"unknown", "unconfirmed"}:
            assert record["reuse_decision"] in {"blocked", "review"}
