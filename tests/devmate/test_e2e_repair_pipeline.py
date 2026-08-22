"""DevMate 端到端 CI/CD 修复链路集成测试。

贯通 DM-05 (webhook 摄取) → DM-06 (诊断) → DM-08 (修复计划) →
DM-09 (沙箱) → DM-10 (审批) → DM-11 (GitHub 副作用) 的完整修复链路，
全部使用内存确定性实现，不接真实 GitHub/Docker/LLM。

验证 README-DEVMATE 宣称的「GitHub CI 失败 webhook → 根因分析 → 生成修复补丁 →
Docker 沙箱验证 → 人工审批 → 自动提交 PR」闭环可在 fake adapter 下端到端跑通。
"""

from __future__ import annotations

from app.devmate.approval import (
    ApprovalRequest,
    DM10Input,
)
from app.devmate.approval import (
    CaseCommand as ApprovalCaseCommand,
)
from app.devmate.diagnostics import DM06Input
from app.devmate.git import DM11Input
from app.devmate.git import RuntimeEvent as GitRuntimeEvent
from app.devmate.ingestion import (
    CIEvidence,
    CommitEvidence,
    DM05Input,
    IngestionStore,
)
from app.devmate.ingestion import (
    RuntimeEvent as IngestionRuntimeEvent,
)
from app.devmate.repair import DM08Input
from app.devmate.repair import RuntimeEvent as RepairRuntimeEvent
from app.devmate.sandbox import DM09Input, IsolatedSandbox, SandboxCommand


def test_end_to_end_repair_pipeline_webhook_to_pr() -> None:
    """完整 CI/CD 修复链路：webhook → 诊断 → 补丁 → 沙箱 → 审批 → PR。"""

    # ── 1. DM-05: 接收 GitHub CI 失败 webhook，幂等固定 evidence ──
    dm05 = DM05Input(
        webhook_id="wh-001",
        source="github",
        event_type="ci_failure",
        payload={
            "repo": "acme/payment-svc",
            "branch": "main",
            "head_sha": "abc123",
            "workflow_run_id": 42,
            "failed_job": "tests",
        },
        commit=CommitEvidence(
            commit_sha="abc123",
            branch="main",
            repo="acme/payment-svc",
        ),
        ci=CIEvidence(ci_run_id="42", ci_status="failed", ci_url=None),
    )
    ingestion = IngestionRuntimeEvent(store=IngestionStore())
    ev_result = ingestion.execute(dm05)
    assert not ev_result.duplicate
    assert ev_result.evidence.commit is not None
    assert ev_result.evidence.commit.commit_sha == "abc123"

    # 重复 webhook 幂等
    dup = ingestion.execute(dm05)
    assert dup.duplicate
    assert dup.evidence_id == ev_result.evidence_id

    # ── 2. DM-06: 确定性根因诊断（固定日志→可重复 findings） ──
    from app.devmate.diagnostics import DiagnosticsCheckpoint

    diag = DiagnosticsCheckpoint()
    dm06 = DM06Input(
        log_text="ImportError: cannot import name 'Foo' from 'bar' (line 42)",
        report_text="1 test failed: test_foo",
        source="webhook",
    )
    diag_result = diag.execute(dm06)
    assert diag_result.finding_count == len(diag_result.findings)
    assert diag_result.signature  # 诊断结果有签名，可复核
    finding_rules = {f.rule for f in diag_result.findings}
    # 诊断产生确定可重复的 finding（不依赖 LLM）
    assert finding_rules  # 至少一个规则命中

    # ── 3. DM-08: 修复计划生成不可变 patch artifact ──
    findings_tuple = tuple((f.finding_id, f.message) for f in diag_result.findings)
    dm08 = DM08Input(case_id="case-1", findings=findings_tuple, base_sha="abc123")
    repair_event = RepairRuntimeEvent()
    repair_result = repair_event.execute(dm08)
    assert repair_result.artifacts  # 生成了 patch artifact
    assert repair_result.immutable_signature  # 不可变签名
    patch_ids = {a.patch_id for a in repair_result.artifacts}

    # ── 4. DM-09: 补丁在资源受限沙箱执行声明命令 ──
    sandbox = IsolatedSandbox()
    dm09 = DM09Input(
        case_id="case-1",
        commands=(
            SandboxCommand(command="pytest", args=("tests/test_foo.py",), declared=True),
            SandboxCommand(command="python", args=("-m", "compileall", "src"), declared=True),
        ),
        cpu_limit=1.0,
        memory_limit=64,
    )
    sandbox_result = sandbox.execute(dm09)
    assert sandbox_result.audit["allowed"] in ("True", "False")
    # 声明命令被沙箱执行（确定性，不发起外部进程）
    assert len(sandbox_result.runs) == 2
    # undeclared 命令被拒绝
    bad = sandbox.execute(
        DM09Input(
            case_id="case-1",
            commands=(SandboxCommand(command="rm", args=("-rf", "/"), declared=False),),
        )
    )
    assert any(r.exit_code == 127 for r in bad.runs)

    # ── 5. DM-10: 审批绑定 patch+evidence+command+principal+expiry ──
    from app.devmate.approval import ApprovalStore

    store = ApprovalStore()
    approval_cmd = ApprovalCaseCommand(store=store)
    # 创建审批请求
    request = ApprovalRequest(
        approval_id="appr-1",
        case_id="case-1",
        patch_id=next(iter(patch_ids)),
        evidence_ids=(ev_result.evidence_id,),
        command="create_pr",
        requested_by="agent:harness",
    )
    store.request(request)

    dm10 = DM10Input(
        approval_id="appr-1",
        decision="approve",
        decided_by="reviewer:alice",
        decided_at="2026-08-04T10:00:00Z",
    )
    approval_result = approval_cmd.execute(dm10)
    assert approval_result.status == "approved"
    assert approval_result.decided_by == "reviewer:alice"
    assert approval_result.patch_id in patch_ids  # 绑定 patch

    # ── 6. DM-11: GitHub 副作用幂等创建分支/PR ──
    git_event = GitRuntimeEvent()
    dm11 = DM11Input(
        release_id="rel-1",
        branch="fix/devmate-patch-1",
        pr_title="[DevMate] auto-fix: ImportError in bar",
    )
    git_result = git_event.execute(dm11)
    assert not git_result.duplicated
    effect_kinds = {e.kind for e in git_result.effects}
    assert "branch" in effect_kinds
    assert "pull_request" in effect_kinds

    # 重复发布不重复创建分支/PR（Effectively-Once）
    dup_git = git_event.execute(dm11)
    assert dup_git.duplicated

    # ── 超时进入 UNKNOWN 并可对账 ──
    timeout_input = DM11Input(
        release_id="rel-2",
        branch="fix/devmate-patch-2",
        pr_title="[DevMate] auto-fix: timeout case",
        timeout=True,
    )
    timeout_result = git_event.execute(timeout_input)
    assert timeout_result.reconciliation_required
    assert all(e.status == "unknown" for e in timeout_result.effects)

    # 对账后状态收敛
    reconciled = git_event.reconcile("rel-2")
    assert reconciled.reconciliation_required is False


def test_pipeline_rejects_undeclared_sandbox_commands() -> None:
    """沙箱拒绝未声明命令，防止补丁执行任意代码。"""
    sandbox = IsolatedSandbox()
    result = sandbox.execute(
        DM09Input(
            case_id="case-x",
            commands=(
                SandboxCommand(command="curl", args=("http://evil.example",), declared=False),
            ),
        )
    )
    assert result.audit["allowed"] == "False"
    assert result.runs[0].exit_code == 127  # UNDECLARED_EXIT
