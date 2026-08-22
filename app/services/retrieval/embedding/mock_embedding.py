"""
Mock Embedding。

使用基于 hash 的确定性随机向量，便于测试和本地开发。
"""

from __future__ import annotations

import hashlib
import logging
import random

logger = logging.getLogger(__name__)


class MockEmbedder:
    """
    Mock Embedding 实现。

    使用文本的 hash 值生成确定性随机向量，确保：
    - 同一文本 always 生成相同向量
    - 不同文本生成不同向量
    - 不依赖任何外部服务
    """

    def __init__(self, dimension: int = 128) -> None:
        """
        初始化 Mock Embedder。

        Args:
            dimension: 向量维度，默认 128
        """
        self.dimension = dimension
        logger.info("mock_embedder_initialized", extra={"dimension": dimension})

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量 embed 文档。

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        logger.debug("embedding_documents", extra={"count": len(texts)})

        embeddings = [self._text_to_vector(text) for text in texts]

        logger.debug("embedding_documents_complete", extra={"count": len(embeddings)})

        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed 查询文本。

        Args:
            query: 查询文本

        Returns:
            查询向量
        """
        logger.debug("embedding_query", extra={"query_length": len(query)})

        return self._text_to_vector(query)

    def _text_to_vector(self, text: str) -> list[float]:
        """
        将文本转换为向量。

        使用 hash 值作为随机种子，生成确定性随机向量。

        Args:
            text: 输入文本

        Returns:
            向量
        """
        # 使用 SHA256 hash 作为随机种子
        hash_hex = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seed = int(hash_hex[:8], 16)  # 取前 8 个 hex 字符作为种子

        # 使用确定性随机数生成器
        rng = random.Random(seed)  # noqa: S311 - deterministic offline test vector

        # 生成随机向量
        vector = [rng.gauss(0, 1) for _ in range(self.dimension)]

        # L2 归一化
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector
