"""
评测运行器。

加载 Golden Dataset，逐条运行 GroundedAnswerService，汇总 RAGAS 和 Agent 指标。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.schemas.chunk import Citation, EvidenceBundle
from app.schemas.eval import EvalCase, EvalResult, EvalRun
from app.services.answer.grounded_answer import GroundedAnswerService
from app.services.evaluation.agent_metrics import compute_agent_metrics
from app.services.evaluation.ragas_adapter import FakeRAGASMetrics, RAGASMetrics

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN_PATH = (
    Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "golden_dataset.jsonl"
)


class EvalRunner:
    """
    评测运行器。

    职责：
    1. 加载 Golden Dataset（JSONL 格式）
    2. 逐条构造 EvidenceBundle，调用 GroundedAnswerService
    3. 用 RAGAS adapter 计算 RAG 指标
    4. 汇总 Agent 自定义指标
    5. 输出 EvalRun（含每条 EvalResult）
    """

    def __init__(
        self,
        answer_service: GroundedAnswerService,
        ragas_metrics: RAGASMetrics | None = None,
    ) -> None:
        self._answer_service = answer_service
        self._ragas = ragas_metrics or FakeRAGASMetrics()

    def load_golden_dataset(self, path: Path | str | None = None) -> list[EvalCase]:
        """
        加载 Golden Dataset。

        Args:
            path: JSONL 文件路径，None 使用默认路径

        Returns:
            EvalCase 列表
        """
        file_path = Path(path) if path else DEFAULT_GOLDEN_PATH
        if not file_path.exists():
            logger.warning("golden_dataset_not_found", extra={"path": str(file_path)})
            return []

        cases: list[EvalCase] = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                cases.append(EvalCase(**data))

        logger.info("golden_dataset_loaded", extra={"count": len(cases), "path": str(file_path)})
        return cases

    async def run(
        self,
        dataset_path: Path | str | None = None,
    ) -> EvalRun:
        """
        运行完整评测。

        Args:
            dataset_path: Golden Dataset 路径

        Returns:
            EvalRun（含每条 EvalResult 和汇总指标）
        """
        run_id = f"eval_{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(UTC)

        eval_run = EvalRun(
            id=run_id,
            dataset_name="golden_v1",
            status="running",
            started_at=started_at,
        )

        cases = self.load_golden_dataset(dataset_path)
        if not cases:
            eval_run.status = "failed"
            eval_run.completed_at = datetime.now(UTC)
            return eval_run

        results: list[EvalResult] = []
        rag_scores: dict[str, list[float]] = {
            "context_precision": [],
            "context_recall": [],
            "faithfulness": [],
            "answer_relevancy": [],
        }
        agent_cases: list[dict[str, object]] = []

        for case in cases:
            # 构造 EvidenceBundle（从 EvalCase.contexts）
            evidence = self._build_evidence_from_case(case)

            # 调用 answer service
            answer_response = await self._answer_service.answer(
                question=case.question,
                evidence=evidence,
            )

            # 计算 RAG 指标
            rag = await self._ragas.compute(
                question=case.question,
                answer=answer_response.answer,
                contexts=case.contexts,
                ground_truth=case.answer,
            )

            for metric_name, value in rag.items():
                if metric_name in rag_scores:
                    rag_scores[metric_name].append(value)

            results.append(
                EvalResult(
                    case_id=case.id,
                    metrics=rag,
                )
            )

            # 构造 Agent 指标用例
            agent_cases.append(
                {
                    "case_id": case.id,
                    "expected_tools": case.expected_tools,
                    "actual_tools": [],  # V1 不跟踪实际工具
                    "requires_approval": case.requires_approval,
                    "approval_granted": True,
                    "expected_refusal": False,
                    "actual_refused": answer_response.refusal_reason is not None,
                    "goal_completed": answer_response.confidence > 0.3,
                }
            )

        # 汇总指标
        aggregated_metrics: dict[str, float] = {}
        for metric_name, values in rag_scores.items():
            aggregated_metrics[metric_name] = round(sum(values) / len(values), 3) if values else 0.0

        # Agent 指标
        agent_result = compute_agent_metrics(agent_cases)
        aggregated_metrics["tool_call_accuracy"] = agent_result.tool_call_accuracy
        aggregated_metrics["approval_correctness"] = agent_result.approval_correctness
        aggregated_metrics["agent_goal_completion_rate"] = agent_result.agent_goal_completion_rate
        aggregated_metrics["refusal_correctness"] = agent_result.refusal_correctness

        eval_run.metrics = aggregated_metrics
        eval_run.status = "completed"
        eval_run.completed_at = datetime.now(UTC)

        logger.info(
            "eval_run_complete",
            extra={
                "run_id": run_id,
                "case_count": len(cases),
                "metrics": aggregated_metrics,
            },
        )
        return eval_run

    def _build_evidence_from_case(self, case: EvalCase) -> EvidenceBundle:
        """
        从 EvalCase 构造 EvidenceBundle。

        为每条 context 生成一个 Citation。
        """
        citations: list[Citation] = []
        for idx, ctx_text in enumerate(case.contexts, start=1):
            citations.append(
                Citation(
                    id=idx,
                    document_name=case.ground_truth_docs[0]
                    if case.ground_truth_docs
                    else "unknown",
                    section=case.ground_truth_sections[idx - 1]
                    if idx - 1 < len(case.ground_truth_sections)
                    else "",
                    page=0,
                    chunk_text=ctx_text,
                    score=0.8,
                    rerank_score=0.8,
                )
            )

        return EvidenceBundle(
            evidence_list=citations,
            total_count=len(citations),
            query_coverage_score=0.8 if citations else 0.0,
        )
