"""确定性诊断 Checkpoint：固定输入产生可重复 findings。

合同：``CheckpointPort.execute(input: DM06Input) -> DM06Result``。
findings 以 (rule, message) 去重并按 (message, severity, line) 排序，
signature 为规范化摘要，保证可复核。
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from app.devmate.diagnostics.analyzer import analyze
from app.devmate.diagnostics.models import DiagnosticFinding, DM06Input, DM06Result


class CheckpointPort(Protocol):
    def execute(self, input_: DM06Input) -> DM06Result: ...


class DiagnosticsCheckpoint:
    def __init__(self, analyzer=analyze) -> None:
        self._analyzer = analyzer

    def execute(self, input_: DM06Input) -> DM06Result:
        findings = self._analyzer(input_.log_text, input_.report_text, input_.source)

        seen: dict[tuple[str, str], DiagnosticFinding] = {}
        for finding in findings:
            seen.setdefault((finding.rule, finding.message), finding)

        deduped = list(seen.values())
        deduped.sort(key=lambda f: (f.message, f.severity, f.line))
        return DM06Result(
            baseline_id=input_.baseline_id,
            findings=tuple(deduped),
            finding_count=len(deduped),
            signature=_signature(deduped),
        )


def _signature(findings: list[DiagnosticFinding]) -> str:
    canonical = json.dumps(
        [[f.rule, f.severity, f.message, f.source, f.line] for f in findings],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
