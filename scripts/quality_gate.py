"""
质量门禁脚本。

一条命令检查全部质量指标，适合本地开发和 CI 使用。

用法：
    python scripts/quality_gate.py          # 快速检查（unit + service + api）
    python scripts/quality_gate.py --full   # 完整检查（含覆盖率报告）
"""

from __future__ import annotations

import re
import subprocess
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYCACHE_PREFIX = PROJECT_ROOT / f".quality-pycache-{uuid.uuid4().hex[:8]}"


def run(cmd: list[str], label: str) -> tuple[bool, str]:
    """运行命令，返回 (是否成功, 输出)。"""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    output = result.stdout + result.stderr
    if result.returncode == 0:
        print("  ✅ 通过")
        # 打印关键行
        for line in output.strip().split("\n")[-5:]:
            print(f"  {line}")
    else:
        print(f"  ❌ 失败 (exit code {result.returncode})")
        for line in output.strip().split("\n")[-20:]:
            print(f"  {line}")

    return result.returncode == 0, output


def check_tests() -> bool:
    """检查 1：全量测试通过。"""
    ok, _ = run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        "检查 1：全量测试通过",
    )
    return ok


def check_tests_with_coverage() -> bool:
    """检查 1b：带覆盖率的测试。"""
    ok, output = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--cov=app",
            "--cov-report=term-missing",
            "--cov-fail-under=70",
        ],
        "检查 1b：覆盖率 ≥ 70%",
    )
    return ok


def check_unit_marker() -> bool:
    """检查 2：unit marker 可用。"""
    ok, _ = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "unit",
            "--co",
        ],
        "检查 2：unit marker 可用",
    )
    return ok


def check_service_marker() -> bool:
    """检查 3：service marker 可用。"""
    ok, _ = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "service",
            "--co",
        ],
        "检查 3：service marker 可用",
    )
    return ok


def check_api_marker() -> bool:
    """检查 4：api marker 可用。"""
    ok, _ = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "api",
            "--co",
        ],
        "检查 4：api marker 可用",
    )
    return ok


def check_no_print_statements() -> bool:
    """检查 5：app/ 目录下没有 print 语句（只允许 logging）。"""
    print(f"\n{'=' * 60}")
    print("  检查 5：禁止 print 语句")
    print(f"{'=' * 60}")

    app_dir = PROJECT_ROOT / "app"
    violations: list[str] = []

    for py_file in app_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        content = py_file.read_text(encoding="utf-8")
        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            # 跳过注释和字符串
            if stripped.startswith("#"):
                continue
            if re.search(r"\bprint\s*\(", stripped) and not stripped.startswith('"""'):
                violations.append(f"  {py_file.relative_to(PROJECT_ROOT)}:{i}: {stripped}")

    if violations:
        print(f"  ❌ 发现 {len(violations)} 处 print 语句：")
        for v in violations[:10]:
            print(f"  {v}")
        return False

    print("  ✅ 通过：无 print 语句")
    return True


def check_env_not_tracked() -> bool:
    """检查 6：.env 不在 git 跟踪中。"""
    print(f"\n{'=' * 60}")
    print("  检查 6：.env 不在 git 中")
    print(f"{'=' * 60}")

    result = subprocess.run(
        ["git", "ls-files", ".env"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.stdout.strip():
        print(f"  ❌ .env 被 git 跟踪：{result.stdout.strip()}")
        return False

    print("  ✅ 通过：.env 未被 git 跟踪")
    return True


def check_compilation() -> bool:
    """检查 7：所有 Python 文件可编译。"""
    ok, _ = run(
        [
            sys.executable,
            "-X",
            f"pycache_prefix={PYCACHE_PREFIX}",
            "-m",
            "compileall",
            "-q",
            "app",
        ],
        "检查 7：Python 编译检查",
    )
    return ok


def main() -> None:
    full_mode = "--full" in sys.argv

    print("=" * 60)
    print("  EnterpriseMind 质量门禁")
    print(f"  模式：{'完整' if full_mode else '快速'}")
    print("=" * 60)

    results: dict[str, bool] = {}

    results["编译检查"] = check_compilation()
    results["全量测试"] = check_tests()

    if full_mode:
        results["覆盖率 ≥ 70%"] = check_tests_with_coverage()

    results["unit marker"] = check_unit_marker()
    results["service marker"] = check_service_marker()
    results["api marker"] = check_api_marker()
    results["禁止 print"] = check_no_print_statements()
    results[".env 安全"] = check_env_not_tracked()

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    print("  汇总报告")
    print(f"{'=' * 60}")

    all_pass = True
    for check_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_pass = False

    print(f"\n{'=' * 60}")
    if all_pass:
        print("  🎉 全部质量门禁通过！")
    else:
        print("  ⚠️  存在未通过的检查项，请修复后再提交。")
    print(f"{'=' * 60}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
