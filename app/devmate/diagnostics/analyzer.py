"""确定性诊断规则：从固定日志与测试报告提取 findings。"""

from __future__ import annotations

import hashlib

from app.devmate.diagnostics.models import DiagnosticFinding

ERROR_PREFIXES = ("ERROR",)
WARNING_PREFIXES = ("WARN", "WARNING")


def analyze(
    log_text: str,
    report_text: str,
    source: str,
) -> list[DiagnosticFinding]:
    """无随机、无外部依赖地扫描日志与测试报告。"""
    findings: list[DiagnosticFinding] = []
    for line_no, raw in enumerate(log_text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith(ERROR_PREFIXES):
            findings.append(
                _finding(
                    rule="log_error",
                    severity="error",
                    message=_message(stripped),
                    source=source,
                    line=line_no,
                )
            )
        elif upper.startswith(WARNING_PREFIXES):
            findings.append(
                _finding(
                    rule="log_warning",
                    severity="warning",
                    message=_message(stripped),
                    source=source,
                    line=line_no,
                )
            )
    for line_no, raw in enumerate(report_text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        if "FAILED" in stripped.upper():
            findings.append(
                _finding(
                    rule="report_failure",
                    severity="error",
                    message=stripped,
                    source=source,
                    line=line_no,
                )
            )
    return findings


def _message(stripped: str) -> str:
    parts = stripped.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _finding(
    *,
    rule: str,
    severity: str,
    message: str,
    source: str,
    line: int,
) -> DiagnosticFinding:
    digest = hashlib.sha256(f"{rule}:{source}:{line}:{message}".encode()).hexdigest()[:8]
    return DiagnosticFinding(
        finding_id=f"{rule}-{digest}",
        severity=severity,
        rule=rule,
        message=message,
        source=source,
        line=line,
    )
