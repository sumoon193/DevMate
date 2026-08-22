"""DevMate DM-07 模型 Fake/Recorded typed diagnosis 失败测试。

合同：``CaseCommand.execute(input: DM07Input) -> DM07Result``。
模型原始输出经过 typed parser；模型不可用或输出非法时降级到
Fake/Recorded 确定性结果。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.devmate.models import (
    CaseCommand,
    DiagnosisParseError,
    DM07Input,
    DM07Result,
    InvalidModeError,
    ModelUnavailableError,
    TypedDiagnosis,
)

PROMPTS_ROOT = Path(__file__).resolve().parents[3] / "app" / "prompts" / "devmate"

RECORDED_RAW = "summary=real summary\nseverity=error\nrule=reported_rule\nconfidence=0.95\n"
INVALID_RAW = "not a diagnosis at all\n"


def _input(
    *,
    case_id: str = "case-1",
    mode: str = "fake",
    raw_output: str = "",
    recorded: dict[str, str] | None = None,
) -> DM07Input:
    return DM07Input(
        case_id=case_id,
        mode=mode,
        raw_output=raw_output,
        recorded=recorded,
    )


def test_case_command_has_typed_entry() -> None:
    result = CaseCommand().execute(_input())

    assert isinstance(result, DM07Result)
    assert isinstance(result.diagnosis, TypedDiagnosis)
    assert result.degraded is False


def test_fake_mode_produces_typed_diagnosis() -> None:
    result = CaseCommand().execute(_input(mode="fake"))

    assert result.mode == "fake"
    assert result.diagnosis.severity == "warning"
    assert result.diagnosis.rule == "fake_rule"
    assert result.diagnosis.summary


def test_recorded_mode_replays_fixed_output() -> None:
    result = CaseCommand().execute(_input(mode="recorded", recorded={"case-1": RECORDED_RAW}))

    assert result.mode == "recorded"
    assert result.degraded is False
    assert result.diagnosis.severity == "error"
    assert result.diagnosis.rule == "reported_rule"
    assert result.diagnosis.confidence == pytest.approx(0.95)


def test_recorded_missing_entry_degrades_to_fake() -> None:
    result = CaseCommand().execute(_input(mode="recorded", recorded={}))

    assert result.degraded is True
    assert result.mode == "fake"
    assert result.diagnosis.rule == "fake_rule"


def test_invalid_output_degrades_to_fake() -> None:
    result = CaseCommand().execute(_input(mode="recorded", recorded={"case-1": INVALID_RAW}))

    assert result.degraded is True
    assert result.diagnosis.rule == "fake_rule"


def test_real_mode_without_provider_is_blocked() -> None:
    with pytest.raises(ModelUnavailableError):
        CaseCommand().execute(_input(mode="real"))


def test_real_mode_uses_injected_provider() -> None:
    class Provider:
        def generate(self, *, case_id: str, prompt: str) -> str:
            assert case_id == "case-1"
            assert prompt
            return RECORDED_RAW

    result = CaseCommand(model_provider=Provider()).execute(_input(mode="real"))
    assert result.mode == "real"
    assert result.degraded is False
    assert result.diagnosis.rule == "reported_rule"


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(InvalidModeError):
        CaseCommand().execute(_input(mode="bogus"))


def test_parser_rejects_missing_required_field() -> None:
    from app.devmate.models.parser import parse_typed_diagnosis

    with pytest.raises(DiagnosisParseError):
        parse_typed_diagnosis("summary=only\n")


def test_prompt_template_exists() -> None:
    templates = sorted(PROMPTS_ROOT.glob("*.txt"))
    assert templates, "app/prompts/devmate must contain a prompt template"
    for template in templates:
        assert template.read_text(encoding="utf-8").strip()
