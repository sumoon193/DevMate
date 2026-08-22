"""
评测服务单元测试。

模块 08 — 必测用例：
- test_eval_runner_reports_rag_and_agent_metrics
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.schemas.chunk import Citation, EvidenceBundle
from app.services.answer.grounded_answer import GroundedAnswerService
from app.services.evaluation.agent_metrics import compute_agent_metrics
from app.services.evaluation.eval_runner import EvalRunner
from app.services.evaluation.ragas_adapter import FakeRAGASMetrics

# ── Fixtures ────────────────────────────────────────────────────────


def _make_evidence(citations: list[Citation], coverage: float = 0.8) -> EvidenceBundle:
    return EvidenceBundle(
        evidence_list=citations,
        total_count=len(citations),
        query_coverage_score=coverage,
    )


def _citation(id: int, text: str) -> Citation:
    return Citation(
        id=id,
        document_name="测试文档",
        section="测试章节",
        page=1,
        chunk_text=text,
        score=0.8,
        rerank_score=0.85,
    )


@pytest.fixture
def answer_service() -> GroundedAnswerService:
    return GroundedAnswerService()


@pytest.fixture
def golden_dataset_path() -> Path:
    """指向项目默认 golden dataset。"""
    return Path(__file__).parent.parent / "fixtures" / "golden_dataset.jsonl"


# ── FakeRAGASMetrics 测试 ──────────────────────────────────────────


class TestFakeRAGASMetrics:
    @pytest.mark.asyncio
    async def test_returns_all_metrics(self) -> None:
        ragas = FakeRAGASMetrics()
        metrics = await ragas.compute(
            question="入职需要什么材料？",
            answer="入职需要身份证和学历证书。",
            contexts=["新员工入职需提交身份证复印件和学历证书。"],
            ground_truth="新员工入职需提交身份证复印件、学历证书和离职证明。",
        )
        assert "context_precision" in metrics
        assert "context_recall" in metrics
        assert "faithfulness" in metrics
        assert "answer_relevancy" in metrics
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    @pytest.mark.asyncio
    async def test_empty_contexts_returns_zeros(self) -> None:
        ragas = FakeRAGASMetrics()
        metrics = await ragas.compute(
            question="任何问题",
            answer="任何答案",
            contexts=[],
            ground_truth="标准答案",
        )
        assert metrics["context_precision"] == 0.0
        assert metrics["context_recall"] == 0.0

    @pytest.mark.asyncio
    async def test_deterministic_output(self) -> None:
        ragas = FakeRAGASMetrics()
        args = ("问题", "答案", ["上下文"], "标准答案")
        assert await ragas.compute(*args) == await ragas.compute(*args)


# ── Agent Metrics 测试 ─────────────────────────────────────────────


class TestAgentMetrics:
    def test_empty_cases_returns_zeros(self) -> None:
        result = compute_agent_metrics([])
        assert result.tool_call_accuracy == 0.0
        assert result.total_cases == 0

    def test_tool_call_accuracy_all_correct(self) -> None:
        cases = [
            {
                "expected_tools": ["policy_search"],
                "actual_tools": ["policy_search"],
                "goal_completed": True,
            },
            {"expected_tools": [], "actual_tools": [], "goal_completed": True},
        ]
        result = compute_agent_metrics(cases)
        assert result.tool_call_accuracy == 1.0
        assert result.agent_goal_completion_rate == 1.0

    def test_tool_call_accuracy_partial(self) -> None:
        cases = [
            {
                "expected_tools": ["policy_search"],
                "actual_tools": ["policy_search"],
                "goal_completed": True,
            },
            {
                "expected_tools": ["policy_search"],
                "actual_tools": ["wrong_tool"],
                "goal_completed": False,
            },
        ]
        result = compute_agent_metrics(cases)
        assert result.tool_call_accuracy == 0.5
        assert result.agent_goal_completion_rate == 0.5

    def test_approval_correctness(self) -> None:
        cases = [
            {
                "expected_tools": [],
                "actual_tools": [],
                "requires_approval": True,
                "approval_granted": True,
            },
            {
                "expected_tools": [],
                "actual_tools": [],
                "requires_approval": True,
                "approval_granted": False,
            },
        ]
        result = compute_agent_metrics(cases)
        assert result.approval_correctness == 0.5

    def test_refusal_correctness(self) -> None:
        cases = [
            {
                "expected_tools": [],
                "actual_tools": [],
                "expected_refusal": True,
                "actual_refused": True,
            },
            {
                "expected_tools": [],
                "actual_tools": [],
                "expected_refusal": True,
                "actual_refused": False,
            },
        ]
        result = compute_agent_metrics(cases)
        assert result.refusal_correctness == 0.5


# ── EvalRunner 测试 ────────────────────────────────────────────────


class TestEvalRunner:
    def test_load_golden_dataset(
        self,
        answer_service: GroundedAnswerService,
        golden_dataset_path: Path,
    ) -> None:
        runner = EvalRunner(answer_service=answer_service)
        cases = runner.load_golden_dataset(golden_dataset_path)
        assert len(cases) == 5
        assert cases[0].question == "新员工入职需要提交哪些材料？"

    def test_load_golden_dataset_missing_file(
        self,
        answer_service: GroundedAnswerService,
    ) -> None:
        runner = EvalRunner(answer_service=answer_service)
        cases = runner.load_golden_dataset("/nonexistent/path.jsonl")
        assert cases == []

    @pytest.mark.asyncio
    async def test_eval_runner_reports_rag_and_agent_metrics(
        self,
        answer_service: GroundedAnswerService,
        golden_dataset_path: Path,
    ) -> None:
        """必测：test_eval_runner_reports_rag_and_agent_metrics"""
        runner = EvalRunner(answer_service=answer_service)
        eval_run = await runner.run(dataset_path=golden_dataset_path)

        assert eval_run.status == "completed"
        assert eval_run.completed_at is not None
        assert eval_run.started_at is not None

        # RAG 指标存在
        for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            assert metric in eval_run.metrics
            assert 0.0 <= eval_run.metrics[metric] <= 1.0

        # Agent 指标存在
        for metric in [
            "tool_call_accuracy",
            "approval_correctness",
            "agent_goal_completion_rate",
            "refusal_correctness",
        ]:
            assert metric in eval_run.metrics

    @pytest.mark.asyncio
    async def test_eval_runner_with_custom_dataset(
        self,
        answer_service: GroundedAnswerService,
    ) -> None:
        # 创建临时 JSONL
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                json.dumps(
                    {
                        "id": "eval_custom_001",
                        "question": "测试问题",
                        "answer": "测试答案",
                        "contexts": ["测试上下文：包含相关信息"],
                        "ground_truth_docs": ["测试文档"],
                        "ground_truth_sections": ["测试章节"],
                        "expected_tools": [],
                        "requires_approval": False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            tmp_path = f.name

        runner = EvalRunner(answer_service=answer_service)
        eval_run = await runner.run(dataset_path=tmp_path)

        assert eval_run.status == "completed"
        assert len(eval_run.metrics) > 0

        import os

        os.unlink(tmp_path)
