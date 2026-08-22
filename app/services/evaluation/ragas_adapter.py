"""RAGAS 评测适配器。"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import Any, Protocol

from app.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


def _validated_score(name: str, value: object) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValidationError(f"Invalid real RAGAS score for {name}: {score}")
    return round(score, 3)


class RAGASMetrics(Protocol):
    """
    RAGAS 指标计算协议。

    后续替换为真实 RAGAS 库。
    """

    async def compute(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> dict[str, float]:
        """
        计算 RAG 指标。

        Args:
            question: 用户问题
            answer: 生成的答案
            contexts: 检索到的上下文列表
            ground_truth: 标准答案

        Returns:
            指标字典：context_precision, context_recall, faithfulness, answer_relevancy
        """
        ...


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """计算两段文本的关键词重叠率（简化版）。"""

    def _keywords(text: str) -> set[str]:
        cleaned = re.sub(r"[\s，。、；：“”‘’【】《》（）！？.,;:!?\(\)\[\]\{\}]", " ", text)
        words: set[str] = set()
        for w in re.findall(r"[a-zA-Z]{2,}", cleaned):
            words.add(w.lower())
        for seg in re.findall(r"[一-鿿]+", cleaned):
            for i in range(max(len(seg) - 1, 0)):
                words.add(seg[i : i + 2])
            if len(seg) == 1:
                words.add(seg)
        return words

    kw_a = _keywords(text_a)
    kw_b = _keywords(text_b)
    if not kw_a or not kw_b:
        return 0.0
    return len(kw_a & kw_b) / min(len(kw_a), len(kw_b))


class FakeRAGASMetrics:
    """
    确定性 fake RAGAS 指标计算。

    基于关键词重叠率近似 RAGAS 指标，用于 V1 单元测试和本地开发。
    """

    async def compute(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> dict[str, float]:
        """
        计算简化 RAG 指标。

        Args:
            question: 用户问题
            answer: 生成的答案
            contexts: 检索到的上下文列表
            ground_truth: 标准答案

        Returns:
            指标字典
        """
        # context_precision: 上下文与问题的相关性
        context_scores = [_keyword_overlap(ctx, question) for ctx in contexts]
        context_precision = sum(context_scores) / len(context_scores) if context_scores else 0.0

        # context_recall: 上下文对标准答案的覆盖度
        all_context = " ".join(contexts)
        context_recall = _keyword_overlap(all_context, ground_truth)

        # faithfulness: 答案是否忠实于上下文
        faithfulness = _keyword_overlap(answer, all_context)

        # answer_relevancy: 答案与问题的相关性
        answer_relevancy = _keyword_overlap(answer, question)

        metrics = {
            "context_precision": round(min(context_precision, 1.0), 3),
            "context_recall": round(min(context_recall, 1.0), 3),
            "faithfulness": round(min(faithfulness, 1.0), 3),
            "answer_relevancy": round(min(answer_relevancy, 1.0), 3),
        }

        logger.debug("ragas_metrics_computed", extra=metrics)
        return metrics


_ZH_CN_PROMPT_EXAMPLES: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
    "context_precision.prompt": [
        (
            {
                "question": "系统如何保证回答可追溯？",
                "context": "系统在回答中返回文档标题、片段编号和来源链接。",
                "answer": "系统通过返回文档来源和片段编号保证回答可追溯。",
            },
            {"reason": "上下文直接支持答案中的追溯机制。", "verdict": 1},
        ),
        (
            {
                "question": "系统如何保证回答可追溯？",
                "context": "Redis 用于缓存会话状态。",
                "answer": "系统通过返回文档来源和片段编号保证回答可追溯。",
            },
            {"reason": "上下文没有提供答案所述的引用信息。", "verdict": 0},
        ),
    ],
    "context_recall.prompt": [
        (
            {
                "question": "系统使用了哪些检索方式？",
                "context": "系统同时执行向量检索和关键词检索，并融合两路结果。",
                "answer": "系统使用向量检索和关键词检索。系统还使用图数据库检索。",
            },
            {
                "classifications": [
                    {
                        "statement": "系统使用向量检索和关键词检索。",
                        "reason": "上下文明确说明了两种检索方式。",
                        "attributed": 1,
                    },
                    {
                        "statement": "系统还使用图数据库检索。",
                        "reason": "上下文没有提到图数据库检索。",
                        "attributed": 0,
                    },
                ]
            },
        )
    ],
    "faithfulness.statement_generator_prompt": [
        (
            {
                "question": "系统如何处理文档？",
                "answer": "系统解析文档并切分为片段，然后为片段建立检索索引。",
            },
            {
                "statements": [
                    "系统解析文档。",
                    "系统将文档切分为片段。",
                    "系统为文档片段建立检索索引。",
                ]
            },
        )
    ],
    "faithfulness.nli_statement_prompt": [
        (
            {
                "context": "系统使用 PostgreSQL 保存任务状态，使用 Redis 缓存短期数据。",
                "statements": [
                    "系统使用 PostgreSQL 保存任务状态。",
                    "系统使用 Redis 保存永久审计记录。",
                ],
            },
            {
                "statements": [
                    {
                        "statement": "系统使用 PostgreSQL 保存任务状态。",
                        "reason": "上下文直接说明 PostgreSQL 的用途。",
                        "verdict": 1,
                    },
                    {
                        "statement": "系统使用 Redis 保存永久审计记录。",
                        "reason": "上下文只说明 Redis 缓存短期数据。",
                        "verdict": 0,
                    },
                ]
            },
        )
    ],
    "answer_relevancy.prompt": [
        (
            {"response": "系统通过来源链接和片段编号提供可追溯证据。"},
            {"question": "系统如何提供可追溯证据？", "noncommittal": 0},
        ),
        (
            {"response": "这个问题要视情况而定，我无法确定。"},
            {"question": "系统采用了哪种实现方式？", "noncommittal": 1},
        ),
    ],
}


class RealRAGASMetrics:
    """
    基于 ragas 库的真实 RAGAS 指标计算。

    需要安装 ragas。依赖缺失或计算失败时直接失败，禁止伪造真实指标。
    """

    def __init__(
        self,
        llm_model: str = "qwen-plus",
        api_key: str = "",
        *,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_model: str = "text-embedding-v4",
        language: str = "chinese",
        timeout_seconds: float = 300.0,
    ) -> None:
        self._llm_model = llm_model
        self._api_key = api_key
        self._base_url = base_url
        self._embedding_model = embedding_model
        self._language = language
        self._timeout_seconds = timeout_seconds
        self._metrics: dict[str, Any] | None = None
        self._metrics_lock = asyncio.Lock()

    async def compute(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> dict[str, float]:
        """使用 ragas 库计算真实指标；任何失败都向上游传播。"""
        return await self._compute_ragas(question, answer, contexts, ground_truth)

    async def _compute_ragas(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> dict[str, float]:
        """按 RAGAS 0.4 collections API 逐项计算，失败直接向上游传播。"""
        context_args = {
            "user_input": question,
            "retrieved_contexts": contexts,
            "reference": ground_truth,
        }
        async with asyncio.timeout(self._timeout_seconds):
            metrics = await self._get_metrics()
            (
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ) = await asyncio.gather(
                metrics["context_precision"].ascore(**context_args),
                metrics["context_recall"].ascore(**context_args),
                metrics["faithfulness"].ascore(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                ),
                metrics["answer_relevancy"].ascore(
                    user_input=question,
                    response=answer,
                ),
            )
        return {
            "context_precision": _validated_score("context_precision", context_precision.value),
            "context_recall": _validated_score("context_recall", context_recall.value),
            "faithfulness": _validated_score("faithfulness", faithfulness.value),
            "answer_relevancy": _validated_score("answer_relevancy", answer_relevancy.value),
        }

    async def _get_metrics(self) -> dict[str, Any]:
        """构建并缓存已适配目标语言的真实 RAGAS 指标。"""
        if self._metrics is not None:
            return self._metrics
        async with self._metrics_lock:
            if self._metrics is not None:
                return self._metrics
            metrics = self._build_metrics()
            self._localize_metric_prompts(metrics)
            self._metrics = metrics
            return metrics

    def _localize_metric_prompts(self, metrics: dict[str, Any]) -> None:
        """加载版本化中文示例，避免运行时翻译不稳定和请求级额外开销。"""
        if self._language == "english":
            return
        if self._language not in {"chinese", "zh-CN", "zh_cn"}:
            raise ValidationError(f"Unsupported RAGAS prompt language: {self._language}")

        for metric_name, metric in metrics.items():
            for attribute, prompt in vars(metric).items():
                key = f"{metric_name}.{attribute}"
                examples = _ZH_CN_PROMPT_EXAMPLES.get(key)
                if examples is None:
                    continue
                prompt.examples = [
                    (
                        prompt.input_model.model_validate(input_data),
                        prompt.output_model.model_validate(output_data),
                    )
                    for input_data, output_data in examples
                ]
                prompt.language = "chinese"
                prompt.instruction += (
                    " Input may be Simplified Chinese. Preserve the input meaning and "
                    "return valid JSON using the required schema."
                )

    def _build_metrics(self) -> dict[str, Any]:
        """创建 Qwen OpenAI-compatible client 与 RAGAS 0.4 指标实例。"""
        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout_seconds,
        )
        llm = llm_factory(
            self._llm_model,
            client=client,
            temperature=0,
            max_tokens=4096,
        )
        embeddings = OpenAIEmbeddings(client=client, model=self._embedding_model)
        return {
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        }
