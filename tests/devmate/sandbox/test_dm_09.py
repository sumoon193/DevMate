"""DevMate DM-09 Sandbox 隔离执行失败测试。

合同：``CheckpointPort.execute(input: DM09Input) -> DM09Result``。
候选 patch 的命令仅在资源受限 Sandbox 内执行；未声明命令被拒绝，
资源超限被拒绝，结果确定可复核。
"""

from __future__ import annotations

from pathlib import Path

from app.devmate.sandbox import (
    DM09Input,
    DM09Result,
    IsolatedSandbox,
    SandboxCommand,
)

SANDBOX_ROOT = Path(__file__).resolve().parents[3] / "app" / "devmate" / "sandbox"


def _input(
    *,
    case_id: str = "case-1",
    commands: tuple[SandboxCommand, ...] = (
        SandboxCommand(command="pytest", args=("tests",)),
        SandboxCommand(command="compileall", args=("app",)),
    ),
    cpu_limit: float = 1.0,
    memory_limit: int = 64,
) -> DM09Input:
    return DM09Input(
        case_id=case_id,
        commands=commands,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
    )


def test_checkpoint_port_has_typed_entry() -> None:
    result = IsolatedSandbox().execute(_input())

    assert isinstance(result, DM09Result)
    assert result.runs


def test_declared_commands_execute_inside_sandbox() -> None:
    result = IsolatedSandbox().execute(_input())

    assert all(run.exit_code == 0 for run in result.runs)
    assert result.allowed is True
    assert [run.command for run in result.runs] == ["pytest", "compileall"]


def test_undeclared_command_is_rejected() -> None:
    result = IsolatedSandbox().execute(
        _input(commands=(SandboxCommand(command="rm", declared=False),))
    )

    run = result.runs[0]
    assert run.exit_code != 0
    assert "undeclared" in run.stderr
    assert result.allowed is False


def test_resource_limit_is_enforced() -> None:
    result = IsolatedSandbox().execute(
        _input(
            commands=(SandboxCommand(command="heavy", cost_cpu=10.0, cost_memory=8),),
            cpu_limit=1.0,
        )
    )

    assert result.resource_exceeded is True
    assert result.runs[0].exit_code == 137
    assert result.allowed is False


def test_memory_limit_is_enforced() -> None:
    result = IsolatedSandbox().execute(
        _input(
            commands=(SandboxCommand(command="big", cost_cpu=0.1, cost_memory=500),),
            memory_limit=64,
        )
    )

    assert result.resource_exceeded is True
    assert result.runs[0].exit_code == 137


def test_result_is_deterministic() -> None:
    first = IsolatedSandbox().execute(_input())
    second = IsolatedSandbox().execute(_input())

    assert first == second


def test_sandbox_module_has_no_network_or_subprocess() -> None:
    sources = sorted(SANDBOX_ROOT.rglob("*.py"))
    assert sources
    for source_path in sources:
        source = source_path.read_text(encoding="utf-8")
        assert "subprocess" not in source
        assert "socket" not in source
        assert "requests" not in source
        assert "urllib" not in source
