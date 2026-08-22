"""扫描 Runtime Kernel 对 HR/RAG 领域的直接依赖，只读不改写代码。

输出按文件记录的领域耦合命中（规则、类别、行号、命中片段）与隔离决定，
作为 W1/W2 go/no-go 输入；命中不代表自动重写，未命中也不等于领域隔离达标。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

RUNTIME_TOKENS = (
    "runtime_kernel",
    "runtime/",
    "event_store",
    "outbox",
    "lease",
    "timer",
)

# (规则名, 类别, 正则)。类别对应 HR/RAG 依赖面：schema / prompt / api /
# config / reference_workflow / rag。
RULES: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "schema-hr-case",
        "schema",
        re.compile(r"\bHRCase\b|\bhr_case\b"),
    ),
    (
        "schema-hr-policy",
        "schema",
        re.compile(r"\bhr_policy\b|\bhr_policy_version\b"),
    ),
    (
        "workflow-onboarding",
        "reference_workflow",
        re.compile(
            r"\bOnboardingCaseWorkflow\b|\bhr_onboarding\b|"
            r"onboarding_case_start|onboarding_to_regularization"
        ),
    ),
    (
        "workflow-hr-domain",
        "reference_workflow",
        re.compile(r"\bexpense\b|\bleave\b|\bregularization\b"),
    ),
    (
        "a2a-policy-research",
        "reference_workflow",
        re.compile(r"InProcessA2AClient|policy_research"),
    ),
    (
        "prompt-layer",
        "prompt",
        re.compile(r"app\.prompts|ANSWER_PROMPT|prompts/"),
    ),
    (
        "api-layer",
        "api",
        re.compile(r"\bFastAPI\b|\bAPIRouter\b|from app\.api(\.|\b)"),
    ),
    (
        "config-hr",
        "config",
        re.compile(r"settings\.hr_|\bhr_rule\b"),
    ),
    (
        "rag-retrieval",
        "rag",
        re.compile(
            r"\bretrieval\b|\bEvidenceBundle\b|\bvector_store\b|\bbm25\b|"
            r"\bembedding\b|\brerank\b|\bchunker\b|\bchunk\b"
        ),
    ),
]

_SNIPPET_LIMIT = 80


def _run_git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _list_git_paths(repo: Path) -> list[str]:
    output = _run_git(
        repo,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--full-name",
    )
    return sorted(item.decode("utf-8") for item in output.split(b"\0") if item.strip())


def is_runtime_path(path: str) -> bool:
    normalized = path.lower().replace("\\", "/")
    return any(token in normalized for token in RUNTIME_TOKENS)


def _scan_source(source: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        for rule_name, category, pattern in RULES:
            match = pattern.search(line)
            if match is None:
                continue
            snippet = line.strip()[:_SNIPPET_LIMIT]
            hits.append(
                {
                    "rule": rule_name,
                    "category": category,
                    "line": line_number,
                    "match": snippet,
                }
            )
    return hits


def scan_domain_coupling(repo: Path) -> list[dict[str, Any]]:
    """返回 runtime 范围内每个 Python 文件的耦合命中与隔离决定。"""
    repo = repo.resolve()
    records: list[dict[str, Any]] = []

    for path in _list_git_paths(repo):
        if not is_runtime_path(path):
            continue
        if Path(path).suffix.lower() != ".py":
            continue
        source = (repo / Path(path)).read_text(
            encoding="utf-8",
            errors="replace",
        )
        hits = _scan_source(source)
        records.append(
            {
                "path": path,
                "conclusion": "blocked" if hits else "ok",
                "hits": hits,
            }
        )

    return sorted(records, key=lambda record: record["path"])


def repository_head(repo: Path) -> str:
    try:
        return _run_git(repo, "rev-parse", "HEAD").decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return "no-commit"


def _worktree_dirty(repo: Path) -> bool:
    return bool(_run_git(repo, "status", "--porcelain").strip())


def _render_report(
    records: list[dict[str, Any]],
    *,
    head: str,
    dirty: bool,
    command: str,
) -> str:
    blocked = [record for record in records if record["conclusion"] == "blocked"]
    ok = [record for record in records if record["conclusion"] == "ok"]

    lines: list[str] = []
    lines.append("# DevMate 领域耦合扫描")
    lines.append("")
    lines.append("> 归属：W1-C1 审计卡输出，供 go/no-go 决策使用；只报告耦合，不自动改写代码。")
    lines.append(f"> 基线 commit：`{head}`")
    lines.append(f"> 工作树：{'dirty' if dirty else 'clean'}")
    lines.append(f"> 命令：`{command}`")
    lines.append("")
    lines.append("## 范围")
    lines.append("")
    lines.append(
        "路径命中 `runtime_kernel` / `runtime/` / `event_store` / `outbox` / "
        "`lease` / `timer` 的 Python 文件（与 origin-map 的 runtime_kernel 分类一致）。"
    )
    lines.append("")
    lines.append("## 结果")
    lines.append("")
    lines.append(f"- 扫描文件：{len(records)}")
    lines.append(f"- 存在 HR/RAG 直接依赖（blocked）：{len(blocked)}")
    lines.append(f"- 干净（ok）：{len(ok)}")
    lines.append("")
    lines.append("## go/no-go 输入")
    lines.append("")
    lines.append(
        "Runtime 直接引用 HR/RAG schema / prompt / API / config / reference "
        "workflow 时，对应重构入口保持 `blocked`；是否隔离由学习者复核命中清单后进入 W2 决策。"
    )
    lines.append("")
    lines.append("## 命中清单")
    lines.append("")
    lines.append("| 文件 | 规则 | 类别 | 行 | 命中 |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for record in blocked:
        for hit in record["hits"]:
            snippet = hit["match"].replace("|", "│")
            lines.append(
                f"| {record['path']} | {hit['rule']} | {hit['category']} "
                f"| {hit['line']} | `{snippet}` |"
            )
    lines.append("")
    lines.append("## 干净文件")
    lines.append("")
    if ok:
        for record in ok:
            lines.append(f"- `{record['path']}`")
    else:
        lines.append("- （无）")
    return "\n".join(lines) + "\n"


def write_report(repo: Path, output: Path, *, command: str = "") -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    head = repository_head(repo)
    dirty = _worktree_dirty(repo)
    records = scan_domain_coupling(repo)
    if head != repository_head(repo):
        raise RuntimeError("repository HEAD changed during domain coupling scan; re-run")

    report = _render_report(
        records,
        head=head,
        dirty=dirty,
        command=command,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(report)
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    command = " ".join([sys.executable, *sys.argv])
    try:
        write_report(arguments.repo, arguments.output, command=command)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[domain-coupling] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
