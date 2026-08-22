"""
Grounded Answer 服务单元测试。

模块 08 — 必测用例：
- test_answer_uses_only_evidence
- test_answer_returns_structured_citations
- test_low_confidence_refuses_or_clarifies
- test_fact_check_rejects_unsupported_claim
"""

from __future__ import annotations

import pytest

from app.schemas.chat import AnswerResponse
from app.schemas.chunk import Citation, EvidenceBundle
from app.services.answer.citation_builder import CitationBuilder
from app.services.answer.fact_checker import FactChecker
from app.services.answer.grounded_answer import FakeAnswerGenerator, GroundedAnswerService
from app.services.answer.low_confidence import (
    LowConfidenceAction,
    LowConfidenceHandler,
)

# ── Fixtures ────────────────────────────────────────────────────────


def _make_citation(
    id: int,
    text: str,
    doc: str = "员工手册",
    section: str = "入职流程",
    score: float = 0.8,
    rerank: float = 0.85,
) -> Citation:
    return Citation(
        id=id,
        document_name=doc,
        section=section,
        page=1,
        chunk_text=text,
        score=score,
        rerank_score=rerank,
    )


def _make_evidence(citations: list[Citation], coverage: float = 0.8) -> EvidenceBundle:
    return EvidenceBundle(
        evidence_list=citations,
        total_count=len(citations),
        query_coverage_score=coverage,
    )


@pytest.fixture
def sample_evidence() -> EvidenceBundle:
    return _make_evidence(
        [
            _make_citation(1, "新员工入职需提交身份证复印件、学历证书、离职证明和一寸照片。"),
            _make_citation(2, "试用期为三个月，自入职之日起计算。", section="试用期管理"),
        ]
    )


@pytest.fixture
def empty_evidence() -> EvidenceBundle:
    return _make_evidence([], coverage=0.0)


@pytest.fixture
def low_score_evidence() -> EvidenceBundle:
    return _make_evidence(
        [_make_citation(1, "某条低质量证据", score=0.1, rerank=0.1)],
        coverage=0.1,
    )


@pytest.fixture
def answer_service() -> GroundedAnswerService:
    return GroundedAnswerService()


# ── CitationBuilder 测试 ─────────────────────────────────────────────


class TestCitationBuilder:
    def test_build_returns_sorted_by_rerank_score(self) -> None:
        builder = CitationBuilder()
        evidence = _make_evidence(
            [
                _make_citation(1, "低分证据", rerank=0.3),
                _make_citation(2, "高分证据", rerank=0.9),
                _make_citation(3, "中分证据", rerank=0.6),
            ]
        )
        citations = builder.build(evidence)
        assert len(citations) == 3
        assert citations[0].chunk_text == "高分证据"
        assert citations[1].chunk_text == "中分证据"
        assert citations[2].chunk_text == "低分证据"

    def test_build_renumbers_from_one(self) -> None:
        builder = CitationBuilder()
        evidence = _make_evidence(
            [
                _make_citation(10, "证据A", rerank=0.9),
                _make_citation(20, "证据B", rerank=0.7),
            ]
        )
        citations = builder.build(evidence)
        assert citations[0].id == 1
        assert citations[1].id == 2

    def test_build_respects_max_citations(self) -> None:
        builder = CitationBuilder()
        evidence = _make_evidence(
            [_make_citation(i, f"证据{i}", rerank=0.9 - i * 0.01) for i in range(1, 11)]
        )
        citations = builder.build(evidence, max_citations=3)
        assert len(citations) == 3

    def test_build_empty_evidence_returns_empty(self) -> None:
        builder = CitationBuilder()
        evidence = _make_evidence([])
        assert builder.build(evidence) == []

    def test_format_for_prompt_returns_dicts(self) -> None:
        builder = CitationBuilder()
        citations = [_make_citation(1, "测试文本")]
        formatted = builder.format_for_prompt(citations)
        assert len(formatted) == 1
        assert formatted[0]["text"] == "测试文本"
        assert formatted[0]["document_name"] == "员工手册"


# ── FactChecker 测试 ────────────────────────────────────────────────


class TestFactChecker:
    def test_fact_check_rejects_unsupported_claim(self) -> None:
        """必测：test_fact_check_rejects_unsupported_claim"""
        checker = FactChecker()
        citations = [_make_citation(1, "新员工入职需提交身份证复印件和学历证书。")]
        # 答案中包含完全不在 evidence 中的内容
        result = checker.check(
            answer="公司规定所有人都可以获得股票期权，并且年终奖为三个月工资。",
            citations=citations,
        )
        assert not result.is_supported
        assert len(result.unsupported_claims) > 0

    def test_fact_check_passes_supported_claim(self) -> None:
        checker = FactChecker()
        citations = [_make_citation(1, "新员工入职需提交身份证复印件和学历证书。")]
        result = checker.check(
            answer="新员工入职需要提交身份证复印件和学历证书。",
            citations=citations,
        )
        assert result.is_supported
        assert result.support_ratio > 0.5

    def test_fact_check_fails_with_no_citations(self) -> None:
        checker = FactChecker()
        result = checker.check(answer="任何答案", citations=[])
        assert not result.is_supported
        assert result.support_ratio == 0.0


# ── LowConfidenceHandler 测试 ───────────────────────────────────────


class TestLowConfidenceHandler:
    def test_low_confidence_refuses_when_no_evidence(self) -> None:
        """必测：test_low_confidence_refuses_or_clarifies（无证据场景）"""
        handler = LowConfidenceHandler()
        verdict = handler.evaluate(evidence=None, question="任何问题")
        assert verdict.action == LowConfidenceAction.REFUSE
        assert verdict.confidence == 0.0

    def test_low_confidence_refuses_empty_evidence_list(self) -> None:
        handler = LowConfidenceHandler()
        evidence = _make_evidence([], coverage=0.0)
        verdict = handler.evaluate(evidence=evidence, question="任何问题")
        assert verdict.action == LowConfidenceAction.REFUSE

    def test_low_confidence_recommends_human_for_low_score(self) -> None:
        handler = LowConfidenceHandler()
        evidence = _make_evidence(
            [_make_citation(1, "低质量证据", rerank=0.1)],
            coverage=0.1,
        )
        verdict = handler.evaluate(evidence=evidence, question="任何问题")
        assert verdict.action == LowConfidenceAction.RECOMMEND_HUMAN

    def test_low_confidence_clarifies_on_conflict(self) -> None:
        handler = LowConfidenceHandler()
        evidence = _make_evidence(
            [
                _make_citation(1, "高分证据", rerank=0.95),
                _make_citation(2, "低分证据", rerank=0.1),
            ]
        )
        verdict = handler.evaluate(evidence=evidence, question="有争议的问题")
        assert verdict.action == LowConfidenceAction.CLARIFY

    def test_proceeds_with_good_evidence(self) -> None:
        handler = LowConfidenceHandler()
        evidence = _make_evidence(
            [
                _make_citation(1, "高质量证据", rerank=0.9),
            ],
            coverage=0.9,
        )
        verdict = handler.evaluate(evidence=evidence, question="正常问题")
        assert verdict.action == LowConfidenceAction.PROCEED


# ── GroundedAnswerService 集成测试 ──────────────────────────────────


class TestGroundedAnswerService:
    @pytest.mark.asyncio
    async def test_answer_uses_only_evidence(
        self,
        answer_service: GroundedAnswerService,
        sample_evidence: EvidenceBundle,
    ) -> None:
        """必测：test_answer_uses_only_evidence"""
        response = await answer_service.answer(
            question="新员工入职需要哪些材料？",
            evidence=sample_evidence,
        )
        assert isinstance(response, AnswerResponse)
        assert response.refusal_reason is None
        # 答案应包含 evidence 中的关键词
        assert "入职" in response.answer or "身份证" in response.answer or "员工" in response.answer

    @pytest.mark.asyncio
    async def test_answer_returns_structured_citations(
        self,
        answer_service: GroundedAnswerService,
        sample_evidence: EvidenceBundle,
    ) -> None:
        """必测：test_answer_returns_structured_citations"""
        response = await answer_service.answer(
            question="新员工入职需要哪些材料？",
            evidence=sample_evidence,
        )
        assert len(response.citations) > 0
        first = response.citations[0]
        assert first.id == 1
        assert first.document_name != ""
        assert first.chunk_text != ""
        assert 0.0 <= first.score <= 1.0

    @pytest.mark.asyncio
    async def test_answer_refuses_with_no_evidence(
        self,
        answer_service: GroundedAnswerService,
    ) -> None:
        """必测：test_low_confidence_refuses_or_clarifies（在 answer 层验证）"""
        response = await answer_service.answer(
            question="完全无关的问题",
            evidence=None,
        )
        assert response.refusal_reason is not None
        assert response.confidence == 0.0
        assert response.citations == []

    @pytest.mark.asyncio
    async def test_answer_has_trace_id(
        self,
        answer_service: GroundedAnswerService,
        sample_evidence: EvidenceBundle,
    ) -> None:
        response = await answer_service.answer(
            question="测试问题",
            evidence=sample_evidence,
            trace_id="trace_test_001",
        )
        assert response.trace_id == "trace_test_001"

    @pytest.mark.asyncio
    async def test_answer_generates_trace_id_when_empty(
        self,
        answer_service: GroundedAnswerService,
        sample_evidence: EvidenceBundle,
    ) -> None:
        response = await answer_service.answer(
            question="测试问题",
            evidence=sample_evidence,
        )
        assert response.trace_id.startswith("trace_")

    @pytest.mark.asyncio
    async def test_fake_generator_produces_deterministic_output(self) -> None:
        gen = FakeAnswerGenerator()
        prompt = "## 证据\n[1] 员工手册：新员工入职需提交身份证\n\n## 用户问题\n入职材料\n\n## 回答约束\n1. 仅使用上述证据回答"
        result1 = await gen.generate(prompt)
        result2 = await gen.generate(prompt)
        assert result1 == result2
        assert "身份证" in result1

    @pytest.mark.asyncio
    async def test_answer_with_tool_results(
        self,
        answer_service: GroundedAnswerService,
    ) -> None:
        from app.schemas.enums import ToolCallStatus
        from app.schemas.tool import ToolCall

        tool_call = ToolCall(
            id="tc_001",
            run_id="run_001",
            tool_name="policy_search",
            parameters={"query": "入职"},
            result={"status": "ok", "message": "工单已创建"},
            status=ToolCallStatus.COMPLETED,
            approval_required=False,
        )
        response = await answer_service.answer(
            question="帮我创建入职工单",
            evidence=None,
            tool_results=[tool_call],
        )
        # 无 evidence → refused，但 tool_results 仍附在 response 上
        assert response.tool_results == [tool_call]
