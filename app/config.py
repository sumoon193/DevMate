"""
应用配置。

默认 fallback mode 不访问真实外部服务；full mode 只在健康检查中探测连接状态。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目运行配置。"""

    environment: Literal["dev", "test", "prod"] = Field(default="dev")
    app_mode: Literal["fallback", "full"] = Field(default="fallback")
    agent_run_engine: Literal["demo", "langgraph"] = Field(default="demo")
    # 审批模式：manual 全部转人工（默认，等价原流程）；policy 写入型自动、管理级人工；
    # auto 写入型自动，管理级是否自动由 approval_auto_allow_admin 决定（沙箱用）。
    approval_mode: Literal["manual", "policy", "auto"] = Field(default="manual")
    approval_auto_allow_admin: bool = Field(default=False)
    local_storage_dir: str = Field(default="runtime_storage")

    postgres_url: str = Field(
        default="postgresql://enterprisemind:change_me_local@localhost:5432/enterprisemind"
    )
    redis_url: str = Field(default="redis://localhost:6379/0")

    # LangGraph checkpointer：memory（默认，进程内）或 postgres（持久化，
    # 跨天审批流程在进程重启后仍可 resume）。postgres 连接串为空时回退 postgres_url。
    graph_checkpointer_backend: Literal["memory", "postgres"] = Field(default="memory")
    graph_checkpointer_postgres_url: str = Field(default="")

    milvus_host: str = Field(default="localhost")
    milvus_port: int = Field(default=19530)

    es_url: str = Field(default="http://localhost:9201")

    minio_endpoint: str = Field(default="http://localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_bucket: str = Field(default="enterprisemind-docs")

    ingestion_execution_mode: Literal["sync", "celery"] = Field(default="sync")
    ingestion_task_store: Literal["memory", "redis"] = Field(default="memory")
    celery_broker_url: str = Field(default="")
    celery_result_backend: str = Field(default="")
    celery_task_always_eager: bool = Field(default=False)

    qwen_api_key: str = Field(default="")
    qwen_chat_model: str = Field(default="qwen-plus")
    qwen_embedding_model: str = Field(default="text-embedding-v4")
    qwen_rerank_model: str = Field(default="qwen3-rerank")
    qwen_api_base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    qwen_rerank_base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-api/v1")
    qwen_timeout_seconds: float = Field(default=30.0)
    ragas_timeout_seconds: float = Field(default=300.0)
    ragas_language: str = Field(default="chinese")
    embedding_dim: int = Field(default=1024)
    embedding_batch_size: int = Field(default=10)

    phoenix_endpoint: str = Field(default="http://localhost:6006")
    health_probe_timeout_seconds: float = Field(default=0.8)

    oidc_issuer_url: str = Field(default="")
    oidc_jwks_url: str = Field(default="")
    oidc_audience: str = Field(default="devmate-web")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """获取缓存后的配置对象。"""
    return Settings()
