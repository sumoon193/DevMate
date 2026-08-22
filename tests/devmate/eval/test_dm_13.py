"""DevMate DM-13 冻结评测、OTel、性能与成本失败测试。

合同：``CaseCommand.execute(input: DM13Input) -> DM13Result``。
诊断、Sandbox、审批与副作用具有可复核指标；评测报告冻结且可重复。
"""

from __future__ import annotations

import pytest

from app.devmate.eval import (
    CaseCommand,
    DM13Input,
    DM13Result,
    EvalReport,
)
from app.devmate.observability import MetricSample, MetricsRegistry


def _samples() -> tuple[MetricSample, ...]:
    return (
        MetricSample("diagnosis.latency_ms", 10.0, (), 0),
        MetricSample("diagnosis.latency_ms", 20.0, (), 1),
        MetricSample("diagnosis.cost", 0.5, (), 0),
        MetricSample("sandbox.latency_ms", 5.0, (), 0),
        MetricSample("approval.latency_ms", 8.0, (), 0),
        MetricSample("side_effect.latency_ms", 12.0, (), 0),
    )


def _input(
    *, case_id: str = "case-1", metrics: tuple[MetricSample, ...] | None = None
) -> DM13Input:
    return DM13Input(
        case_id=case_id,
        metrics=metrics if metrics is not None else _samples(),
    )


def test_case_command_has_typed_entry() -> None:
    result = CaseCommand().execute(_input())

    assert isinstance(result, DM13Result)
    assert isinstance(result.report, EvalReport)
    assert result.report.report_id


def test_report_covers_all_four_dimensions() -> None:
    report = CaseCommand().execute(_input()).report

    dimensions = [dimension.dimension for dimension in report.dimensions]
    assert dimensions == ["diagnosis", "sandbox", "approval", "side_effect"]


def test_counts_and_avg_latency_computed() -> None:
    report = CaseCommand().execute(_input()).report

    diagnosis = report.dimension("diagnosis")
    assert diagnosis.count == 2
    assert diagnosis.avg_latency_ms == pytest.approx(15.0)
    sandbox = report.dimension("sandbox")
    assert sandbox.count == 1
    assert sandbox.avg_latency_ms == pytest.approx(5.0)


def test_total_cost_computed() -> None:
    report = CaseCommand().execute(_input()).report

    diagnosis = report.dimension("diagnosis")
    assert diagnosis.total_cost == pytest.approx(0.5)
    assert report.dimension("approval").total_cost == pytest.approx(0.0)


def test_report_is_frozen_and_repeatable() -> None:
    first = CaseCommand().execute(_input()).report
    second = CaseCommand().execute(_input()).report

    assert first.signature == second.signature
    assert first == second
    with pytest.raises(AttributeError):
        first.dimensions = ()


def test_empty_metrics_produce_zero_report() -> None:
    report = CaseCommand().execute(_input(metrics=())).report

    for dimension in report.dimensions:
        assert dimension.count == 0
        assert dimension.avg_latency_ms == 0.0
        assert dimension.total_cost == 0.0


def test_metrics_registry_records_and_queries() -> None:
    registry = MetricsRegistry()
    registry.record("diagnosis.latency_ms", 5.0, {"case": "c1"}, at=1)
    registry.record("diagnosis.latency_ms", 7.0, {"case": "c1"}, at=2)

    assert registry.count("diagnosis.latency_ms") == 2
    assert registry.total("diagnosis.latency_ms") == pytest.approx(12.0)
    assert registry.query("diagnosis.latency_ms")[0].labels == (("case", "c1"),)
