"""
集成测试 fixtures。

依赖 Docker 容器（PostgreSQL / Redis / MinIO / Milvus / ES）。
运行方式：python -m pytest tests/integration/ -m integration -v
"""

from __future__ import annotations

import logging
import uuid

import pytest
import pytest_asyncio

from app.config import Settings

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def settings() -> Settings:
    """只连接仓库 Compose 的专用本地服务，禁止读取用户 .env。"""
    return Settings(
        _env_file=None,
        app_mode="full",
        postgres_url=("postgresql://enterprisemind:change_me_local@127.0.0.1:5432/enterprisemind"),
        redis_url="redis://127.0.0.1:6379/0",
        minio_endpoint="http://127.0.0.1:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket="enterprisemind-docs",
        milvus_host="127.0.0.1",
        milvus_port=19530,
        es_url="http://127.0.0.1:9201",
        qwen_api_key="",
    )


# ── PostgreSQL ────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_session(settings: Settings):
    """提供一个真实的 async 数据库 session，测试后回滚。"""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    url = settings.postgres_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(url, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    # 确保所有 ORM 模型被导入（否则 create_all 找不到表）
    import app.models  # noqa: F401
    from app.models.base import Base

    # 先清理残留表，再创建
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    async with factory() as session:
        yield session

    # 清理
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ── Redis ─────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def redis_client(settings: Settings):
    """提供真实的 Redis 客户端。"""
    import redis.asyncio as aioredis

    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    yield client
    # 清理测试 keys
    async for key in client.scan_iter("rate_limit:*"):
        await client.delete(key)
    await client.aclose()


# ── MinIO ─────────────────────────────────────────────────────────────


@pytest.fixture
def minio_storage(settings: Settings):
    """提供真实的 MinIO 存储实例。"""
    from app.services.storage.minio_storage import MinIOStorage

    return MinIOStorage(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
    )


# ── Milvus ────────────────────────────────────────────────────────────


@pytest.fixture
def milvus_store(settings: Settings):
    """提供真实的 Milvus 向量存储实例。"""
    from pymilvus import MilvusClient

    from app.services.retrieval.store.milvus_vector import (
        COLLECTION_NAME,
        MilvusVectorStore,
    )

    store = MilvusVectorStore(
        host=settings.milvus_host,
        port=settings.milvus_port,
        dim=settings.embedding_dim,
    )
    yield store

    # 清理：删除 collection
    try:
        client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
    except Exception as exc:
        logger.warning("milvus_test_cleanup_failed", exc_info=exc)


# ── Elasticsearch ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def es_store(settings: Settings):
    """提供真实的 ES BM25 存储实例。"""
    from app.services.retrieval.store.es_bm25 import INDEX_NAME, ElasticsearchBM25Store

    store = ElasticsearchBM25Store(es_url=settings.es_url)
    yield store

    # 清理：删除索引
    try:
        exists = await store._es.indices.exists(index=INDEX_NAME)
        if exists:
            await store._es.indices.delete(index=INDEX_NAME)
    except Exception as exc:
        logger.warning("elasticsearch_test_cleanup_failed", exc_info=exc)
    await store.close()


# ── 通用 ──────────────────────────────────────────────────────────────


@pytest.fixture
def unique_id() -> str:
    """生成唯一 ID 前缀，避免测试间冲突。"""
    return uuid.uuid4().hex[:8]
