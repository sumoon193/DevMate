"""DevMate DM-08 修复计划与 patch 候选失败测试。

合同：``RuntimeEvent.execute(input: DM08Input) -> DM08Result``。
RepairPlan 只生成不可变 patch artifact：frozen、digest 稳定、无突变路径。
"""

from __future__ import annotations

import pytest

from app.devmate.repair import (
    DM08Input,
    DM08Result,
    EmptyPlanError,
    PatchArtifact,
    RuntimeEvent,
)


def _input(
    *,
    case_id: str = "case-1",
    findings: tuple[tuple[str, str], ...] = (("log_error", "boom"), ("log_warning", "slow")),
    target_root: str = "repo",
    base_sha: str = "abc123",
) -> DM08Input:
    return DM08Input(
        case_id=case_id,
        findings=findings,
        target_root=target_root,
        base_sha=base_sha,
    )


def test_runtime_event_has_typed_entry() -> None:
    result = RuntimeEvent().execute(_input())

    assert isinstance(result, DM08Result)
    assert result.plan_id
    assert result.artifacts


def test_each_finding_produces_a_patch() -> None:
    result = RuntimeEvent().execute(_input())

    assert len(result.steps) == 2
    assert len(result.artifacts) == 2
    assert {step.rule for step in result.steps} == {"log_error", "log_warning"}


def test_patch_artifact_carries_path_and_kind() -> None:
    result = RuntimeEvent().execute(_input())

    artifact = result.artifacts[0]
    assert artifact.path
    assert artifact.kind == "edit"
    assert artifact.content
    assert artifact.patch_id


def test_artifact_digest_is_stable() -> None:
    first = RuntimeEvent().execute(_input())
    second = RuntimeEvent().execute(_input())

    assert [a.digest for a in first.artifacts] == [a.digest for a in second.artifacts]
    assert first.immutable_signature == second.immutable_signature


def test_artifact_is_immutable() -> None:
    result = RuntimeEvent().execute(_input())

    with pytest.raises(AttributeError):
        result.artifacts[0].content = "changed"


def test_artifact_digest_matches_content() -> None:
    import hashlib

    result = RuntimeEvent().execute(_input())

    for artifact in result.artifacts:
        expected = hashlib.sha256(artifact.content.encode("utf-8")).hexdigest()
        assert artifact.digest == expected


def test_empty_findings_rejected() -> None:
    with pytest.raises(EmptyPlanError):
        RuntimeEvent().execute(_input(findings=()))


def test_plan_signature_derived_from_artifacts() -> None:
    result = RuntimeEvent().execute(_input())

    assert result.immutable_signature
    assert result.plan_id
    assert isinstance(result.artifacts[0], PatchArtifact)
