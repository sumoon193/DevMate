"""生成只包含路径和 Git 元数据的 DevMate origin map。

生成器只读取 Git 枚举出的相对路径、最近提交和许可证交叉证据，不读取
``.env`` 等敏感文件正文，也不改写生产代码。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

EXCLUDED_PATHS = {"README-DEVMATE.md"}
SENSITIVE_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
}


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _run_git_bytes(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _resolve_within_repo(repo: Path, relative: str) -> Path:
    """解析仓库内相对路径，越出根目录时拒绝访问。"""
    root = repo.resolve()
    target = (root / Path(relative)).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"path escapes repository root: {relative}")
    return target


def _list_paths(repo: Path) -> list[str]:
    # -z 输出原始字节并用 NUL 分隔，避免 core.quotepath 对非 ASCII 路径的
    # octal 转义被当成路径分隔符，导致中文文件名损坏。
    output = _run_git_bytes(
        repo,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--full-name",
    )
    paths = {item.decode("utf-8") for item in output.split(b"\0") if item.strip()}
    return sorted(path for path in paths if path not in EXCLUDED_PATHS)


def _source_commit(repo: Path, path: str) -> str:
    output = _run_git(repo, "log", "-1", "--format=%H", "--", path).strip()
    return output or "untracked"


def _classify_domain(path: str) -> str:
    normalized = path.lower().replace("\\", "/")
    runtime_tokens = (
        "runtime_kernel",
        "runtime/",
        "event_store",
        "outbox",
        "lease",
        "timer",
    )
    if any(token in normalized for token in runtime_tokens):
        return "runtime_kernel"
    if any(token in normalized for token in ("hr", "onboarding", "expense", "leave")):
        return "hr"
    if any(token in normalized for token in ("rag", "retrieval", "embedding", "rerank", "chunk")):
        return "rag"
    return "unknown"


def _license_status(repo: Path, path: str, source_commit: str) -> str:
    if Path(path).name in SENSITIVE_FILE_NAMES:
        return "unknown"
    if source_commit == "untracked":
        return "review"
    return "confirmed" if (repo / "LICENSE").is_file() else "unknown"


def _reuse_decision(
    license_status: str,
    domain: str,
    source_commit: str,
) -> str:
    if license_status in {"unknown", "unconfirmed", "conflict"} or source_commit == "untracked":
        return "review"
    if domain in {"hr", "rag"}:
        return "isolate"
    if domain == "runtime_kernel":
        return "allowed"
    return "review"


def _coupling_tags(repo: Path, path: str, domain: str) -> list[str]:
    if Path(path).name in SENSITIVE_FILE_NAMES:
        return ["sensitive_path"]
    if Path(path).suffix.lower() != ".py":
        return [f"domain:{domain}"] if domain != "unknown" else []

    source = (
        _resolve_within_repo(repo, path)
        .read_text(
            encoding="utf-8",
            errors="replace",
        )
        .lower()
    )
    markers = ("hr", "rag", "prompt", "schema", "config", "api")
    return sorted({f"{marker}_reference" for marker in markers if marker in source})


def generate_origin_map(repo: Path) -> list[dict[str, Any]]:
    """返回逐文件来源事实；敏感路径只读取文件名，不读取正文。"""
    repo = repo.resolve()
    records: list[dict[str, Any]] = []

    for path in _list_paths(repo):
        source_commit = _source_commit(repo, path)
        domain = _classify_domain(path)
        license_status = _license_status(repo, path, source_commit)
        evidence_refs: list[str] = []
        if source_commit != "untracked":
            evidence_refs.append(f"git:{source_commit}")
        if license_status == "confirmed":
            evidence_refs.append("license:LICENSE")
        records.append(
            {
                "path": path,
                "source_commit": source_commit,
                "license_status": license_status,
                "domain": domain,
                "coupling_tags": _coupling_tags(repo, path, domain),
                "reuse_decision": _reuse_decision(
                    license_status,
                    domain,
                    source_commit,
                ),
                "evidence_refs": evidence_refs,
            }
        )

    return records


def repository_head(repo: Path) -> str:
    """返回当前 HEAD 提交；空仓库返回 no-commit。"""
    try:
        return _run_git(repo, "rev-parse", "HEAD").strip()
    except subprocess.CalledProcessError:
        return "no-commit"


def write_origin_map(repo: Path, output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # 生成期间 HEAD 变化说明仓库正在被修改，报告可能基于过期来源，直接失败。
    head_before = repository_head(repo)
    records = generate_origin_map(repo)
    head_after = repository_head(repo)
    if head_before != head_after:
        raise RuntimeError("repository HEAD changed during origin map generation; re-run")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
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
    try:
        write_origin_map(arguments.repo, arguments.output)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[audit-origin-map] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
