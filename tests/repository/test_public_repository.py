import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_SECTIONS = (
    "## 项目简介与适用场景",
    "## 功能清单",
    "## 系统架构与核心流程",
    "## 技术栈与运行依赖",
    "## 目录结构说明",
    "## 环境要求",
    "## 本地快速启动",
    "## Docker 或中间件启动方式",
    "## 配置项和环境变量",
    "## 主要 API",
    "## 请求示例与返回结果",
    "## 离线测试",
    "## 真实服务验证",
    "## 常见问题与故障排查",
    "## 安全边界和生产注意事项",
    "## License",
)


def test_public_repository_has_no_internal_governance_files() -> None:
    forbidden = [
        ".agent-governance",
        "AGENTS.md",
        "CLAUDE.md",
        "tools/governance",
        "tests/governance",
        ".github/workflows/governance.yml",
    ]
    result = subprocess.run(
        ["git", "ls-files", "--", *forbidden], cwd=ROOT, check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == ""


def test_readme_is_public_facing_and_portable() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in readme
    for forbidden in (
        "D:\\Code\\",
        "D:\\py\\",
        "C:\\Users\\",
        ".agent-governance",
        "README-DEVMATE",
        "面试",
        "简历",
        "完成证明",
        "学习证明",
        "内部治理",
        "任务交接",
    ):
        assert forbidden not in readme
    for required in (
        "python -m uvicorn app.main:app",
        "docker compose --profile full up",
        "POST` | `/documents",
        "POST` | `/agent-runs",
        "GET` | `/memories",
        "OIDC_ISSUER_URL",
        "scripts\\devmate\\live_smoke.py",
        "Apache-2.0",
    ):
        assert required in readme
