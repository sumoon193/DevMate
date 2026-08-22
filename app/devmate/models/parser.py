"""模型原始输出到 typed diagnosis 的确定性解析器。

解析 key=value 行；缺少必需字段或 severity 非法时稳定拒绝。
"""

from __future__ import annotations

import hashlib

from app.devmate.models.types import TypedDiagnosis

REQUIRED_FIELDS = ("summary", "severity", "rule")
VALID_SEVERITIES = {"critical", "error", "warning", "info"}


class DiagnosisParseError(ValueError):
    """模型输出无法解析为 typed diagnosis。"""


def parse_typed_diagnosis(raw: str) -> TypedDiagnosis:
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        fields[key.strip()] = value.strip()

    for required in REQUIRED_FIELDS:
        if not fields.get(required):
            raise DiagnosisParseError(f"missing required field: {required}")

    severity = fields["severity"]
    if severity not in VALID_SEVERITIES:
        raise DiagnosisParseError(f"invalid severity: {severity}")

    confidence = _parse_float(fields.get("confidence", "1.0"))
    evidence = tuple(part.strip() for part in fields.get("evidence", "").split(",") if part.strip())
    diagnosis_id = hashlib.sha256(
        f"{fields['summary']}:{severity}:{fields['rule']}".encode()
    ).hexdigest()[:12]
    return TypedDiagnosis(
        diagnosis_id=diagnosis_id,
        summary=fields["summary"],
        severity=severity,
        rule=fields["rule"],
        confidence=confidence,
        evidence=evidence,
    )


def _parse_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise DiagnosisParseError(f"invalid confidence: {value}") from exc
    if not 0.0 <= parsed <= 1.0:
        raise DiagnosisParseError(f"confidence out of range: {value}")
    return parsed
