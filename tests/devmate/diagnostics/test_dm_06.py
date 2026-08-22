"""DevMate DM-06 确定性诊断 baseline 失败测试。

合同：``CheckpointPort.execute(input: DM06Input) -> DM06Result``。
固定日志与测试报告产生可重复、排序去重的 findings，signature 稳定。
"""

from __future__ import annotations

from app.devmate.diagnostics import (
    DiagnosticsCheckpoint,
    DM06Input,
    DM06Result,
)


def _input(
    *,
    log_text: str = "INFO started\nERROR boom\nWARN slow\n",
    report_text: str = "",
    source: str = "runner",
) -> DM06Input:
    return DM06Input(log_text=log_text, report_text=report_text, source=source)


def test_checkpoint_port_has_typed_entry() -> None:
    result = DiagnosticsCheckpoint().execute(_input())

    assert isinstance(result, DM06Result)
    assert result.baseline_id
    assert result.findings


def test_fixed_log_produces_repeatable_findings() -> None:
    checkpoint = DiagnosticsCheckpoint()

    first = checkpoint.execute(_input())
    second = checkpoint.execute(_input())

    assert first.signature == second.signature
    assert first.findings == second.findings


def test_error_and_warning_lines_are_found() -> None:
    result = DiagnosticsCheckpoint().execute(_input(log_text="INFO ok\nERROR boom\nWARN slow\n"))

    severities = {finding.severity for finding in result.findings}
    assert "error" in severities
    assert "warning" in severities
    assert len(result.findings) == 2


def test_test_report_failures_are_found() -> None:
    result = DiagnosticsCheckpoint().execute(_input(report_text="tests.py:12 FAILED test_a\n"))

    assert any(finding.rule == "report_failure" for finding in result.findings)


def test_findings_are_sorted_and_deduped() -> None:
    result = DiagnosticsCheckpoint().execute(
        _input(log_text="ERROR dup\nERROR dup\nWARN a\nWARN b\n")
    )

    messages = [finding.message for finding in result.findings]
    assert messages == sorted(messages)
    assert messages.count("dup") == 1


def test_empty_input_produces_no_findings() -> None:
    result = DiagnosticsCheckpoint().execute(_input(log_text="", report_text=""))

    assert result.findings == ()
    assert result.finding_count == 0


def test_finding_carries_source_and_line() -> None:
    result = DiagnosticsCheckpoint().execute(
        _input(log_text="line one\nERROR boom\n", source="runner")
    )

    finding = result.findings[0]
    assert finding.source == "runner"
    assert finding.line == 2
    assert finding.message == "boom"
    assert finding.finding_id
