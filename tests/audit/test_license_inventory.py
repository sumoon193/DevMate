"""DevMate license inventory gate tests."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGIN_MAP_PATH = REPO_ROOT / "docs" / "audit" / "origin-map.jsonl"
LICENSE_INVENTORY_PATH = REPO_ROOT / "docs" / "audit" / "license-inventory.md"


def _read_origin_map() -> list[dict[str, object]]:
    with ORIGIN_MAP_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_allowed_or_isolated_records_reference_license_evidence() -> None:
    for record in _read_origin_map():
        if record["reuse_decision"] not in {"allowed", "isolate"}:
            continue

        evidence_refs = record["evidence_refs"]
        assert isinstance(evidence_refs, list)
        assert any(str(reference).startswith("license:") for reference in evidence_refs), record[
            "path"
        ]


def test_license_inventory_records_repository_license_and_decision() -> None:
    inventory = LICENSE_INVENTORY_PATH.read_text(encoding="utf-8")

    assert "Apache-2.0" in inventory
    assert "LICENSE" in inventory
    assert "pyproject.toml" in inventory
    assert "uv.lock" in inventory
    assert "review" in inventory


def test_unconfirmed_license_is_never_allowed() -> None:
    for record in _read_origin_map():
        if record["license_status"] in {"unknown", "unconfirmed", "conflict", "review"}:
            assert record["reuse_decision"] != "allowed", record["path"]
