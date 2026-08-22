"""
Grounded Answer Service。

核心服务：基于 evidence 生成可信回答。
协调 CitationBuilder、LowConfidenceHandler、FactChecker 和 AnswerGenerator。
"""

from __future__ import annotations

import logging
import uuid
from typing import Protocol

from app.prompts.answer_prompt import ANSWER_PROMPT_V1, REFUSAL_PROMPT_V1
from app.schemas.chat import AnswerResponse
from app.schemas.chunk import EvidenceBundle
from app.schemas.tool import ToolCall
from app.services.answer.citation_builder import CitationBuilder
from app.services.answer.fact_checker import FactChecker
from app.services.answer.low_confidence import (
    LowConfidenceAction,
    LowConfidenceHandler,
    LowConfidenceVerdict,
)

logger = logging.getLogger(__name__)


class AnswerGenerator(Protocol):
    """
    答案生成器协议。

    V1 实现：FakeAnswerGenerator（确定性，基于 evidence 拼接）。
    后续替换为真实 LLM 调用。
    """

    async def generate(self, prompt: str) -> str:
        """
        根据 prompt 生成回答文本。

        Args:
            prompt: 渲染后的完整 prompt

        Returns:
            生成的回答文本
        """
        ...


class FakeAnswerGenerator:
    """
    确定性 Fake 答案生成器。

    V1 阶段使用，不依赖真实 LLM API。
    直接从 evidence 拼接答案，便于测试和本地开发。
    """

    async def generate(self, prompt: str) -> str:
        """
        确定性生成：提取 prompt 中的证据内容作为答案。

        Args:
            prompt: 渲染后的完整 prompt

        Returns:
            基于证据拼接的确定性答案
        """
        # 从 prompt 中提取证据文本行
        lines = prompt.split("\n")
        evidence_lines: list[str] = []
        in_evidence = False

        for line in lines:
            if line.strip() == "## 证据":
                in_evidence = True
                continue
            if line.strip().startswith("## ") and in_evidence:
                break
            if in_evidence and line.strip() and line.strip() != "（无可用证据）":
                evidence_lines.append(line.strip())

        if evidence_lines:
            # 组装带引用标注的答案
            answer_parts: list[str] = ["根据公司制度，相关信息如下：\n"]
            for i, line in enumerate(evidence_lines, start=1):
                # 提取编号后的实际内容
                if line.startswith("["):
                    bracket_end = line.find("]")
                    if bracket_end != -1:
                        content = line[bracket_end + 1 :].strip()
                        answer_parts.append(f"[{i}] {content}")
                    else:
                        answer_parts.append(line)
                else:
                    answer_parts.append(f"[{i}] {line}")
            return "\n".join(answer_parts)

        return "根据现有资料，无法找到相关信息回答您的问题。建议联系 HR 部门获取帮助。"


class GroundedAnswerService:
    """
    Grounded Answer 核心服务。

    职责：
    1. 检查置信度（LowConfidenceHandler）
    2. 构建引用（CitationBuilder）
    3. 渲染 Prompt 并调用 LLM
    4. 事实核查（FactChecker）
    5. 组装 AnswerResponse
    """

    def __init__(
        self,
        answer_generator: AnswerGenerator | None = None,
        citation_builder: CitationBuilder | None = None,
        fact_checker: FactChecker | None = None,
        low_confidence_handler: LowConfidenceHandler | None = None,
    ) -> None:
        self._generator: AnswerGenerator = answer_generator or FakeAnswerGenerator()
        self._citation_builder = citation_builder or CitationBuilder()
        self._fact_checker = fact_checker or FactChecker()
        self._low_confidence = low_confidence_handler or LowConfidenceHandler()

    async def answer(
        self,
        question: str,
        evidence: EvidenceBundle | None,
        tool_results: list[ToolCall] | None = None,
        trace_id: str = "",
    ) -> AnswerResponse:
        """
        生成 Grounded Answer。

        Args:
            question: 用户问题
            evidence: 检索到的证据包
            tool_results: 工具执行结果（可选）
            trace_id: 追踪 ID

        Returns:
            AnswerResponse
        """
        if not trace_id:
            trace_id = f"trace_{uuid.uuid4().hex[:12]}"

        tool_results = tool_results or []

        # Step 1：低置信度检查
        verdict = self._low_confidence.evaluate(evidence, question)

        if verdict.action == LowConfidenceAction.REFUSE:
            logger.info("answer_refused", extra={"reason": verdict.reason, "trace_id": trace_id})
            refusal_text = await self._generate_refusal(question, verdict)
            return AnswerResponse(
                answer=refusal_text,
                citations=[],
                confidence=verdict.confidence,
                refusal_reason=verdict.reason,
                tool_results=tool_results,
                trace_id=trace_id,
            )

        if verdict.action == LowConfidenceAction.RECOMMEND_HUMAN:
            logger.info(
                "answer_recommend_human", extra={"reason": verdict.reason, "trace_id": trace_id}
            )
            return AnswerResponse(
                answer=f"该问题的证据质量较低，建议直接联系 HR 部门获取准确信息。\n\n原因：{verdict.reason}",
                citations=[],
                confidence=verdict.confidence,
                refusal_reason=verdict.reason,
                tool_results=tool_results,
                trace_id=trace_id,
            )

        if verdict.action == LowConfidenceAction.CLARIFY:
            logger.info("answer_clarify", extra={"reason": verdict.reason, "trace_id": trace_id})
            return AnswerResponse(
                answer=f"检索到的证据存在分歧，需要您澄清问题。\n\n{verdict.reason}\n\n请提供更多细节，例如具体部门、时间范围或场景。",
                citations=[],
                confidence=verdict.confidence,
                refusal_reason=verdict.reason,
                tool_results=tool_results,
                trace_id=trace_id,
            )

        # Step 2：构建引用
        citations = self._citation_builder.build(evidence)

        # Step 3：渲染 Prompt
        evidence_dicts = self._citation_builder.format_for_prompt(citations)
        tool_result_dicts = [
            {"tool_name": tr.tool_name, "result": str(tr.result) if tr.result else ""}
            for tr in tool_results
        ]

        prompt = ANSWER_PROMPT_V1.render(
            question=question,
            evidence=evidence_dicts,
            tool_results=tool_result_dicts,
        )

        # Step 4：调用 LLM
        answer_text = await self._generator.generate(prompt)

        # Step 5：事实核查
        fact_result = self._fact_checker.check(answer_text, citations)
        final_confidence = (
            verdict.confidence if fact_result.is_supported else verdict.confidence * 0.5
        )

        if not fact_result.is_supported and fact_result.support_ratio < 0.3:
            logger.warning(
                "fact_check_failed",
                extra={
                    "support_ratio": fact_result.support_ratio,
                    "unsupported_count": len(fact_result.unsupported_claims),
                    "trace_id": trace_id,
                },
            )

        logger.info(
            "answer_generated",
            extra={
                "trace_id": trace_id,
                "citation_count": len(citations),
                "confidence": round(final_confidence, 2),
                "fact_supported": fact_result.is_supported,
            },
        )

        return AnswerResponse(
            answer=answer_text,
            citations=citations,
            confidence=round(final_confidence, 3),
            refusal_reason=None,
            tool_results=tool_results,
            trace_id=trace_id,
        )

    async def _generate_refusal(
        self,
        question: str,
        verdict: LowConfidenceVerdict,
    ) -> str:
        """生成拒答回复。"""
        prompt = REFUSAL_PROMPT_V1.render(
            question=question,
            refusal_reason=verdict.reason,
        )
        return await self._generator.generate(prompt)
