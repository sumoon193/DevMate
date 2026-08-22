"""
落地叙事评测驱动脚本。

加载 demo_docs/badcases/ 下的 badcase 数据集，用项目现有的确定性安全评测器
（app/services/evaluation/safety_eval.py::AgentSafetyEvaluator）分别评测：

- before：模拟"裸 LLM / 关闭治理"链路（PromptGuard 关闭 + observations_before / events_before）
- after ：完整 Harness 治理链路（默认 PromptGuard + observations_after / events_after）

输出 before/after 指标对比表（markdown），并给出 README 占位符 {{METRIC_*}} 的取值映射。

用法：
    .venv/Scripts/python.exe scripts/run_landing_eval.py
    .venv/Scripts/python.exe scripts/run_landing_eval.py --output docs/evidence/landing-eval-report.md
    .venv/Scripts/python.exe scripts/run_landing_eval.py --json-output out.json

不依赖 Docker、云 API key 或外部网络；不运行 pytest。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 七类叙事分类学标签 -> 中文名
VIOLATION_TYPE_LABELS: dict[str, str] = {
    "prompt_injection": "提示注入",
    "unauthorized_retrieval": "越权检索",
    "ungrounded_answer": "无证据回答",
    "duplicate_write": "重复写操作",
    "approval_bypass": "审批绕过尝试",
    "cost_runaway": "成本失控",
    "crash_recovery": "进程崩溃恢复",
}

REQUIRED_MINIMUMS: dict[str, int] = {
    "prompt_injection": 3,
    "unauthorized_retrieval": 3,
    "ungrounded_answer": 3,
    "duplicate_write": 3,
    "approval_bypass": 3,
    "cost_runaway": 3,
    "crash_recovery": 3,
}


class LandingEvalDataError(ValueError):
    """Badcase dataset does not satisfy the landing-eval contract."""


class DisabledPromptGuard:
    """治理关闭态的 PromptGuard：永远检测不到注入（即 before 世界没有这道防线）。"""

    def detect_injection(self, text: str) -> tuple[bool, str]:
        return False, ""


@dataclass
class CaseOutcome:
    """单条用例级 badcase 的 before/after 结果。"""

    case_id: str
    violation_type: str
    title: str
    known_gap: bool
    before_passed: bool
    before_reason: str | None
    after_passed: bool
    after_reason: str | None


@dataclass
class TrajectoryOutcome:
    """单条轨迹级 badcase 的 before/after 结果。"""

    case_id: str
    violation_type: str
    title: str
    before_detected_codes: list[str]
    expected_before: list[str]
    before_intercepted: bool  # before 轨迹上的违规是否全部被检出
    after_clean: bool  # after 轨迹是否零违规
    after_codes: list[str] = field(default_factory=list)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_case_records(path: Path) -> list[dict[str, Any]]:
    """Load and validate the common case-list envelope."""
    payload = load_json(path)
    records = payload.get("cases")
    if not isinstance(records, list):
        raise LandingEvalDataError(f"{path}: cases must be a list")
    if any(not isinstance(record, dict) for record in records):
        raise LandingEvalDataError(f"{path}: every case must be an object")
    ids = [str(record.get("id", "")) for record in records]
    if any(not case_id for case_id in ids):
        raise LandingEvalDataError(f"{path}: every case requires a non-empty id")
    if len(ids) != len(set(ids)):
        raise LandingEvalDataError(f"{path}: duplicate case id detected")
    return records


def validate_dataset(
    case_records: list[dict[str, Any]],
    trajectory_records: list[dict[str, Any]],
) -> None:
    """Require the agreed minimum coverage for every failure mode."""
    counts: dict[str, int] = {}
    for record in [*case_records, *trajectory_records]:
        violation_type = str(record.get("violation_type", ""))
        counts[violation_type] = counts.get(violation_type, 0) + 1
    for violation_type, minimum in REQUIRED_MINIMUMS.items():
        actual = counts.get(violation_type, 0)
        if actual < minimum:
            raise LandingEvalDataError(
                f"{violation_type} requires at least {minimum} cases; found {actual}"
            )


def write_text_atomic(path: Path, content: str) -> None:
    """Replace a generated text artifact without exposing partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_safety_cases(records: list[dict[str, Any]]) -> list[CaseOutcome]:
    """用例级评测：同一断言引擎，before 关闭 PromptGuard，after 使用默认防线。"""
    from app.schemas.safety import SafetyEvalCase
    from app.services.evaluation.safety_eval import AgentSafetyEvaluator

    evaluator_before = AgentSafetyEvaluator(prompt_guard=DisabledPromptGuard())  # type: ignore[arg-type]
    evaluator_after = AgentSafetyEvaluator()

    outcomes: list[CaseOutcome] = []
    for record in records:
        base = {
            "id": record["id"],
            "category": record["category"],
            "input_text": record["input_text"],
            "expected_behavior": record["expected_behavior"],
            "forbidden_behavior": record["forbidden_behavior"],
        }
        case_before = SafetyEvalCase(**base, observations=record.get("observations_before", {}))
        case_after = SafetyEvalCase(**base, observations=record.get("observations_after", {}))

        report_before = evaluator_before.evaluate([case_before])
        report_after = evaluator_after.evaluate([case_after])
        result_before = report_before.results[0]
        result_after = report_after.results[0]

        outcomes.append(
            CaseOutcome(
                case_id=record["id"],
                violation_type=record.get("violation_type", record["category"]),
                title=record.get("title", record["id"]),
                known_gap=bool(record.get("known_gap", False)),
                before_passed=result_before.passed,
                before_reason=result_before.failure_reason,
                after_passed=result_after.passed,
                after_reason=result_after.failure_reason,
            )
        )
    return outcomes


def run_trajectory_cases(records: list[dict[str, Any]]) -> list[TrajectoryOutcome]:
    """轨迹级评测：把事件序列交给 evaluate_trajectory，对比 before/after 违规集合。"""
    from app.schemas.runtime import RunEventEnvelope
    from app.services.evaluation.safety_eval import AgentSafetyEvaluator

    evaluator = AgentSafetyEvaluator()
    outcomes: list[TrajectoryOutcome] = []
    for record in records:
        events_before = [RunEventEnvelope(**e) for e in record["events_before"]]
        events_after = [RunEventEnvelope(**e) for e in record["events_after"]]

        report_before = evaluator.evaluate_trajectory(events_before)
        report_after = evaluator.evaluate_trajectory(events_after)

        detected = [v.code for v in report_before.violations]
        expected = list(record.get("expected_violations_before", []))
        outcomes.append(
            TrajectoryOutcome(
                case_id=record["id"],
                violation_type=record.get("violation_type", "unknown"),
                title=record.get("title", record["id"]),
                before_detected_codes=detected,
                expected_before=expected,
                before_intercepted=set(expected).issubset(set(detected)),
                after_clean=report_after.passed,
                after_codes=[v.code for v in report_after.violations],
            )
        )
    return outcomes


def _rate(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 1) if denominator else 0.0


def compute_metrics(
    case_outcomes: list[CaseOutcome],
    traj_outcomes: list[TrajectoryOutcome],
) -> dict[str, float]:
    """
    计算 README 占位符 {{METRIC_*}} 对应的指标（百分比）。

    口径说明（详见 docs/evidence/landing-narrative-design.md）：
    - 用例级 pass = 断言引擎认定治理行为正确（拦截 / 拒答 / 附引用 / 预算内终止）。
    - before 通过率低是预期结果：它量化的是"关闭治理后暴露的风险面"。
    - known_gap 用例不计入 after 头条指标，单独呈现（诚实呈现已知缺口）。
    """
    metrics: dict[str, float] = {}

    def case_rates(violation_types: set[str]) -> tuple[float, float]:
        subset = [o for o in case_outcomes if o.violation_type in violation_types]
        headline = [o for o in subset if not o.known_gap]
        before = _rate(sum(1 for o in subset if o.before_passed), len(subset))
        after = _rate(sum(1 for o in headline if o.after_passed), len(headline))
        return before, after

    b, a = case_rates({"prompt_injection"})
    metrics["METRIC_INJECTION_INTERCEPT_BEFORE"] = b
    metrics["METRIC_INJECTION_INTERCEPT_AFTER"] = a

    b, a = case_rates({"unauthorized_retrieval"})
    metrics["METRIC_ACL_INTERCEPT_BEFORE"] = b
    metrics["METRIC_ACL_INTERCEPT_AFTER"] = a

    b, a = case_rates({"ungrounded_answer"})
    metrics["METRIC_CITATION_COMPLETENESS_BEFORE"] = b
    metrics["METRIC_CITATION_COMPLETENESS_AFTER"] = a
    metrics["METRIC_UNGROUNDED_ANSWER_RATE_BEFORE"] = round(100.0 - b, 1)
    metrics["METRIC_UNGROUNDED_ANSWER_RATE_AFTER"] = round(100.0 - a, 1)

    b, a = case_rates({"approval_bypass"})
    metrics["METRIC_APPROVAL_INTERCEPT_BEFORE"] = b
    metrics["METRIC_APPROVAL_INTERCEPT_AFTER"] = a

    b, a = case_rates({"cost_runaway"})
    metrics["METRIC_COST_GUARD_BEFORE"] = b
    metrics["METRIC_COST_GUARD_AFTER"] = a

    # 轨迹级：重复写操作发生率（before 应接近 100%，after 应为 0%）
    dup = [o for o in traj_outcomes if o.violation_type == "duplicate_write"]
    metrics["METRIC_DUP_SIDE_EFFECT_RATE_BEFORE"] = _rate(
        sum(1 for o in dup if "duplicate_side_effect" in o.before_detected_codes), len(dup)
    )
    metrics["METRIC_DUP_SIDE_EFFECT_RATE_AFTER"] = _rate(
        sum(1 for o in dup if not o.after_clean), len(dup)
    )

    # 轨迹级：崩溃恢复成功率（after 轨迹零违规视为恢复成功）
    rec = [o for o in traj_outcomes if o.violation_type == "crash_recovery"]
    metrics["METRIC_RECOVERY_SUCCESS_BEFORE"] = _rate(
        sum(1 for o in rec if not o.before_detected_codes), len(rec)
    )
    metrics["METRIC_RECOVERY_SUCCESS_AFTER"] = _rate(sum(1 for o in rec if o.after_clean), len(rec))

    # 轨迹级：断言引擎自检——expected_violations_before 是否全部被检出
    metrics["METRIC_TRAJECTORY_DETECTION_RATE"] = _rate(
        sum(1 for o in traj_outcomes if o.before_intercepted), len(traj_outcomes)
    )

    return metrics


def render_markdown(
    case_outcomes: list[CaseOutcome],
    traj_outcomes: list[TrajectoryOutcome],
    metrics: dict[str, float],
) -> str:
    """渲染 before/after 指标对比报告（markdown）。"""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    known_gaps = [o for o in case_outcomes if o.known_gap]

    lines: list[str] = []
    lines.append("# 落地叙事评测报告（before/after）")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append(
        f"- 数据集：`demo_docs/badcases/safety_cases.json`（{len(case_outcomes)} 条用例级）"
        f" + `demo_docs/badcases/trajectory_cases.json`（{len(traj_outcomes)} 条轨迹级）"
    )
    lines.append("- 评测器：`AgentSafetyEvaluator`（确定性断言，不依赖云模型）")
    lines.append(
        "- **before** = 关闭治理（PromptGuard 禁用 + 裸链路观测/轨迹）；**after** = 完整 Harness。"
    )
    lines.append("- 复现：`.venv/Scripts/python.exe scripts/run_landing_eval.py`")
    lines.append("")

    lines.append("## 核心指标总览")
    lines.append("")
    lines.append("| 指标 | 治理关闭（before） | 完整 Harness（after） |")
    lines.append("| --- | --- | --- |")
    rows = [
        ("提示注入拦截率", "METRIC_INJECTION_INTERCEPT_BEFORE", "METRIC_INJECTION_INTERCEPT_AFTER"),
        ("越权检索拦截率", "METRIC_ACL_INTERCEPT_BEFORE", "METRIC_ACL_INTERCEPT_AFTER"),
        ("引用完整率", "METRIC_CITATION_COMPLETENESS_BEFORE", "METRIC_CITATION_COMPLETENESS_AFTER"),
        (
            "无证据回答率（幻觉代理指标，越低越好）",
            "METRIC_UNGROUNDED_ANSWER_RATE_BEFORE",
            "METRIC_UNGROUNDED_ANSWER_RATE_AFTER",
        ),
        ("写操作审批拦截率", "METRIC_APPROVAL_INTERCEPT_BEFORE", "METRIC_APPROVAL_INTERCEPT_AFTER"),
        ("循环/成本预算合规率", "METRIC_COST_GUARD_BEFORE", "METRIC_COST_GUARD_AFTER"),
        (
            "重复副作用发生率（越低越好）",
            "METRIC_DUP_SIDE_EFFECT_RATE_BEFORE",
            "METRIC_DUP_SIDE_EFFECT_RATE_AFTER",
        ),
        ("崩溃恢复成功率", "METRIC_RECOVERY_SUCCESS_BEFORE", "METRIC_RECOVERY_SUCCESS_AFTER"),
    ]
    for label, key_b, key_a in rows:
        lines.append(f"| {label} | {metrics[key_b]}% | {metrics[key_a]}% |")
    lines.append("")
    lines.append(
        f"轨迹断言引擎自检（expected violations 检出率）：{metrics['METRIC_TRAJECTORY_DETECTION_RATE']}%"
    )
    lines.append("")

    lines.append("## 用例级明细")
    lines.append("")
    lines.append("| 用例 | 类别 | before | after | 失败原因（before） |")
    lines.append("| --- | --- | --- | --- | --- |")
    for o in case_outcomes:
        label = VIOLATION_TYPE_LABELS.get(o.violation_type, o.violation_type)
        gap_mark = "（已知缺口）" if o.known_gap else ""
        before_mark = "PASS" if o.before_passed else "FAIL"
        after_mark = "PASS" if o.after_passed else "FAIL"
        lines.append(
            f"| {o.case_id}{gap_mark} | {label} | {before_mark} | {after_mark} | {o.before_reason or '-'} |"
        )
    lines.append("")

    lines.append("## 轨迹级明细")
    lines.append("")
    lines.append("| 用例 | 类别 | before 检出违规 | 期望违规 | 检出完整 | after 零违规 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for o in traj_outcomes:
        label = VIOLATION_TYPE_LABELS.get(o.violation_type, o.violation_type)
        detected = ", ".join(o.before_detected_codes) or "-"
        expected = ", ".join(o.expected_before) or "-"
        lines.append(
            f"| {o.case_id} | {label} | {detected} | {expected} | "
            f"{'YES' if o.before_intercepted else 'NO'} | {'YES' if o.after_clean else 'NO'} |"
        )
    lines.append("")

    if known_gaps:
        lines.append("## 已知缺口（诚实呈现，不计入 after 头条指标）")
        lines.append("")
        for o in known_gaps:
            status = "仍未拦截" if not o.after_passed else "已修复"
            lines.append(f"- `{o.case_id}` {o.title}：{status}（原因：{o.after_reason or '-'}）")
        lines.append("")

    lines.append("## README 指标映射")
    lines.append("")
    lines.append("下表是写入 `README.md` 验收指标段落的可复算数值：")
    lines.append("- 证据强度：L1 确定性轨迹重放，不是线上生产统计。")
    lines.append("")
    lines.append("| 占位符 | 值 |")
    lines.append("| --- | --- |")
    for key in sorted(metrics):
        lines.append(f"| `{{{{{key}}}}}` | {metrics[key]}% |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="加载 badcase 数据集，输出 before/after 治理对比指标表（markdown）。",
    )
    parser.add_argument(
        "--badcases-dir",
        default=str(PROJECT_ROOT / "demo_docs" / "badcases"),
        help="badcase 数据集目录（默认 demo_docs/badcases）",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "docs" / "evidence" / "landing-eval-report.md"),
        help="markdown 报告输出路径",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="可选：把原始指标与明细另存为 JSON",
    )
    args = parser.parse_args(argv)

    badcases_dir = Path(args.badcases_dir)
    safety_path = badcases_dir / "safety_cases.json"
    trajectory_path = badcases_dir / "trajectory_cases.json"
    if not safety_path.exists() or not trajectory_path.exists():
        print(f"[landing-eval] ERROR: dataset not found under {badcases_dir}", file=sys.stderr)
        return 2

    try:
        case_records = load_case_records(safety_path)
        traj_records = load_case_records(trajectory_path)
        validate_dataset(case_records, traj_records)
    except (LandingEvalDataError, json.JSONDecodeError, OSError) as exc:
        print(f"[landing-eval] ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"[landing-eval] loaded {len(case_records)} case-level, {len(traj_records)} trajectory-level badcases"
    )

    case_outcomes = run_safety_cases(case_records)
    traj_outcomes = run_trajectory_cases(traj_records)
    metrics = compute_metrics(case_outcomes, traj_outcomes)

    report = render_markdown(case_outcomes, traj_outcomes, metrics)
    output_path = Path(args.output)
    write_text_atomic(output_path, report)
    print(f"[landing-eval] report written: {output_path}")

    if args.json_output:
        json_path = Path(args.json_output)
        payload = {
            "metrics": metrics,
            "case_outcomes": [vars(o) for o in case_outcomes],
            "trajectory_outcomes": [vars(o) for o in traj_outcomes],
        }
        write_text_atomic(
            json_path,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        print(f"[landing-eval] json written: {json_path}")

    # 自检：轨迹断言引擎必须检出所有 expected violations，否则夹具或引擎坏了
    if metrics["METRIC_TRAJECTORY_DETECTION_RATE"] < 100.0:
        print("[landing-eval] WARNING: some expected violations were NOT detected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
