"""devmate 可观测性：内存指标收集器（OTel 形态，无外部导出）。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSample:
    metric: str
    value: float
    labels: tuple[tuple[str, str], ...] = ()
    at: int = 0


class MetricsRegistry:
    """追加式指标存储；同一输入可重复查询，结果确定。"""

    def __init__(self) -> None:
        self._samples: list[MetricSample] = []

    def record(
        self,
        metric: str,
        value: float,
        labels: dict[str, str] | None = None,
        *,
        at: int = 0,
    ) -> None:
        normalized = tuple(sorted((labels or {}).items()))
        self._samples.append(MetricSample(metric, value, normalized, at))

    def query(self, metric: str) -> list[MetricSample]:
        return [sample for sample in self._samples if sample.metric == metric]

    def count(self, metric: str) -> int:
        return len(self.query(metric))

    def total(self, metric: str) -> float:
        return sum(sample.value for sample in self.query(metric))
