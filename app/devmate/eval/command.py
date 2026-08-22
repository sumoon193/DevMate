"""冻结评测 typed command。

合同：``CaseCommand.execute(input: DM13Input) -> DM13Result``。
从固定指标样本生成冻结、可复核的评测报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.devmate.eval.evaluator import Evaluator
from app.devmate.eval.report import EvalReport
from app.devmate.observability import MetricSample


@dataclass(frozen=True)
class DM13Input:
    case_id: str
    metrics: tuple[MetricSample, ...]
    report_id: str = "eval-1"


@dataclass(frozen=True)
class DM13Result:
    case_id: str
    report: EvalReport
    audit: dict[str, str] = field(default_factory=dict)


class CaseCommand:
    def __init__(self, evaluator: Evaluator | None = None) -> None:
        self._evaluator = evaluator or Evaluator()

    def execute(self, input_: DM13Input) -> DM13Result:
        report = self._evaluator.evaluate(input_.metrics)
        return DM13Result(
            case_id=input_.case_id,
            report=report,
            audit={"report_id": report.report_id, "signature": report.signature},
        )
