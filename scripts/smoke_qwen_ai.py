"""验证 Qwen Chat / Embedding / Rerank 真实链路。

运行前确保本地 .env 已填写 QWEN_API_KEY。
该脚本不会打印 API key。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.schemas.enums import Visibility
from app.schemas.retrieval import RetrievalResult
from app.services.ai.qwen import QwenAnswerGenerator, QwenEmbedder, QwenReranker


async def main() -> None:
    settings = Settings(_env_file=None)
    if not settings.qwen_api_key:
        raise SystemExit("QWEN_API_KEY is empty; please fill local .env first.")

    answer_generator = QwenAnswerGenerator(
        api_key=settings.qwen_api_key,
        model=settings.qwen_chat_model,
        base_url=settings.qwen_api_base_url,
        timeout_seconds=settings.qwen_timeout_seconds,
    )
    embedder = QwenEmbedder(
        api_key=settings.qwen_api_key,
        model=settings.qwen_embedding_model,
        dimension=settings.embedding_dim,
        base_url=settings.qwen_api_base_url,
        timeout_seconds=settings.qwen_timeout_seconds,
    )
    reranker = QwenReranker(
        api_key=settings.qwen_api_key,
        model=settings.qwen_rerank_model,
        rerank_base_url=settings.qwen_rerank_base_url,
        timeout_seconds=settings.qwen_timeout_seconds,
    )

    answer = await answer_generator.generate("请只回复：OK")
    embedding = await embedder.embed_query("新员工入职材料")
    reranked = await reranker.rerank(
        "新员工入职需要什么材料？",
        [
            _result("chunk_1", "新员工入职需要提交身份证明、学历证明和离职证明。"),
            _result("chunk_2", "报销需要提交发票、审批单和付款记录。"),
        ],
        top_k=2,
    )

    print("qwen_chat=ok", f"chars={len(answer)}")
    print("qwen_embedding=ok", f"dimension={len(embedding)}")
    print(
        "qwen_rerank=ok",
        f"top_chunk={reranked[0].chunk_id}",
        f"score={reranked[0].rerank_score:.3f}",
    )


def _result(chunk_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        chunk_text=text,
        context_prefix="",
        score=0.5,
        rerank_score=0.5,
        raw_score=0.5,
        document_name="HR 制度",
        section="入职",
        page=1,
        heading_path="HR 制度 > 入职",
        tenant_id="tenant_001",
        department_id="dept_hr",
        visibility=Visibility.DEPARTMENT,
    )


if __name__ == "__main__":
    asyncio.run(main())
