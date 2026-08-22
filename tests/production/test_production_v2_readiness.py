from __future__ import annotations

import pytest

from scripts.devmate.production_readiness import (
    REQUIRED_GATES,
    EvidenceError,
    evaluate_evidence,
    gate_commands,
)

CURRENT_COMMIT = "e15ef6472f8ee78c2c21912cf3798ddf4ad9c31a"


def _passed(gate: str, commit: str = CURRENT_COMMIT) -> dict[str, object]:
    return {
        "gate": gate,
        "status": "passed",
        "commit_sha": commit,
        "command": f"verify {gate}",
        "exit_code": 0,
        "timestamp": "2026-08-08T12:00:00+08:00",
        "dataset_version": "devmate-production-v2-test",
        "raw_result": f"reports/production-v2/{gate}.json",
    }


def test_required_gates_cover_strict_production_verification() -> None:
    assert REQUIRED_GATES == (
        "source",
        "build",
        "offline-tests",
        "postgres",
        "redis",
        "minio",
        "milvus",
        "elasticsearch",
        "celery",
        "qwen",
        "otel",
        "business-e2e",
        "evaluation",
        "load",
        "security",
        "recovery",
        "cold-start",
        "public-deployment",
        "stability",
    )


def test_missing_gate_stays_blocked() -> None:
    result = evaluate_evidence([_passed(gate) for gate in REQUIRED_GATES[:-1]], CURRENT_COMMIT)

    assert result.status == "deployment-ready"
    assert result.missing_gates == ("stability",)
    assert result.exit_code == 2


def test_mixed_commit_evidence_fails_closed() -> None:
    evidence = [_passed(gate) for gate in REQUIRED_GATES]
    evidence[-1]["commit_sha"] = "b" * 40

    with pytest.raises(EvidenceError, match="multiple commits"):
        evaluate_evidence(evidence, CURRENT_COMMIT)


def test_all_gates_pass_at_expected_commit() -> None:
    result = evaluate_evidence([_passed(gate) for gate in REQUIRED_GATES], CURRENT_COMMIT)

    assert result.status == "production-verified"
    assert result.exit_code == 0


def test_gate_commands_are_runnable_and_do_not_embed_secrets() -> None:
    commands = gate_commands()

    assert set(commands) == set(REQUIRED_GATES)
    assert "python -m pytest -q" in commands["offline-tests"]
    assert "live_smoke.py --component memory" in commands["postgres"]
    assert "npm --prefix frontend run test:e2e:live" in commands["business-e2e"]
    assert all("API_KEY=" not in command for command in commands.values())
