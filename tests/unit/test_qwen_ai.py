"""Qwen 真实 AI adapter 的无网络单元测试。"""

from __future__ import annotations

from typing import Any

import pytest

from app.config import Settings
from app.core.exceptions import ExternalServiceError
from app.schemas.enums import Visibility
from app.schemas.retrieval import RetrievalResult


@pytest.mark.asyncio
async def test_qwen_answer_generator_posts_chat_completion_and_extracts_content() -> None:
    from app.services.ai.qwen import QwenAnswerGenerator

    calls: list[dict[str, Any]] = []

    async def fake_post_json(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {"choices": [{"message": {"content": "真实模型回答"}}]}

    generator = QwenAnswerGenerator(
        api_key="sk-test",
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_seconds=3.0,
        post_json=fake_post_json,
    )

    answer = await generator.generate("请基于证据回答")

    assert answer == "真实模型回答"
    assert calls[0]["url"].endswith("/chat/completions")
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert calls[0]["payload"] == {
        "model": "qwen-plus",
        "messages": [{"role": "user", "content": "请基于证据回答"}],
        "temperature": 0.2,
    }
    assert calls[0]["timeout_seconds"] == 3.0


@pytest.mark.asyncio
async def test_qwen_embedder_posts_embeddings_and_validates_dimension() -> None:
    from app.services.ai.qwen import QwenEmbedder

    calls: list[dict[str, Any]] = []

    async def fake_post_json(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        calls.append({"url": url, "headers": headers, "payload": payload})
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        return {
            "data": [
                {"index": index, "embedding": vectors[index]}
                for index in range(len(payload["input"]))
            ]
        }

    embedder = QwenEmbedder(
        api_key="sk-test",
        model="text-embedding-v4",
        dimension=3,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        post_json=fake_post_json,
    )

    embeddings = await embedder.embed_documents(["入职材料", "转正流程"])
    query_embedding = await embedder.embed_query("入职要办什么")

    assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert query_embedding == [0.1, 0.2, 0.3]
    assert calls[0]["url"].endswith("/embeddings")
    assert calls[0]["payload"] == {
        "model": "text-embedding-v4",
        "input": ["入职材料", "转正流程"],
        "dimensions": 3,
        "encoding_format": "float",
    }
    assert calls[1]["payload"]["input"] == ["入职要办什么"]


@pytest.mark.asyncio
async def test_qwen_embedder_rejects_unexpected_embedding_dimension() -> None:
    from app.services.ai.qwen import QwenEmbedder

    async def fake_post_json(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    embedder = QwenEmbedder(
        api_key="sk-test",
        model="text-embedding-v4",
        dimension=3,
        post_json=fake_post_json,
    )

    with pytest.raises(ExternalServiceError, match="dimension"):
        await embedder.embed_query("维度不匹配")


@pytest.mark.asyncio
async def test_qwen_reranker_maps_relevance_scores_back_to_results() -> None:
    from app.services.ai.qwen import QwenReranker

    calls: list[dict[str, Any]] = []

    async def fake_post_json(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        calls.append({"url": url, "headers": headers, "payload": payload})
        return {
            "results": [
                {"index": 1, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.62},
            ]
        }

    results = [
        _retrieval_result("chunk_1", "入职需要提交身份证明", 0.4),
        _retrieval_result("chunk_2", "转正需要主管评估", 0.5),
    ]
    reranker = QwenReranker(
        api_key="sk-test",
        model="qwen3-rerank",
        rerank_base_url="https://dashscope.aliyuncs.com/compatible-api/v1",
        post_json=fake_post_json,
    )

    reranked = await reranker.rerank("转正流程", results, top_k=2)

    assert [item.chunk_id for item in reranked] == ["chunk_2", "chunk_1"]
    assert [item.rerank_score for item in reranked] == [0.91, 0.62]
    assert calls[0]["url"].endswith("/reranks")
    assert calls[0]["payload"] == {
        "model": "qwen3-rerank",
        "query": "转正流程",
        "documents": ["入职需要提交身份证明", "转正需要主管评估"],
        "top_n": 2,
    }


def _retrieval_result(chunk_id: str, chunk_text: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        chunk_text=chunk_text,
        context_prefix="",
        score=score,
        rerank_score=score,
        raw_score=score,
        document_name="HR 制度",
        section="入职",
        page=1,
        heading_path="HR 制度 > 入职",
        tenant_id="tenant_001",
        department_id="dept_hr",
        visibility=Visibility.DEPARTMENT,
    )


def test_build_ai_adapters_uses_qwen_when_api_key_is_configured() -> None:
    from app.api.dependencies import _build_ai_adapters
    from app.services.ai.qwen import QwenAnswerGenerator, QwenEmbedder, QwenReranker

    settings = Settings(_env_file=None, qwen_api_key="sk-test")

    answer_generator, embedder, reranker = _build_ai_adapters(settings)

    assert isinstance(answer_generator, QwenAnswerGenerator)
    assert isinstance(embedder, QwenEmbedder)
    assert isinstance(reranker, QwenReranker)


def test_build_ai_adapters_keeps_fake_when_api_key_is_missing() -> None:
    from app.api.dependencies import _build_ai_adapters
    from app.services.answer.grounded_answer import FakeAnswerGenerator
    from app.services.retrieval.embedding.mock_embedding import MockEmbedder
    from app.services.retrieval.reranker.mock_reranker import MockReranker

    settings = Settings(_env_file=None, qwen_api_key="")

    answer_generator, embedder, reranker = _build_ai_adapters(settings)

    assert isinstance(answer_generator, FakeAnswerGenerator)
    assert isinstance(embedder, MockEmbedder)
    assert isinstance(reranker, MockReranker)
