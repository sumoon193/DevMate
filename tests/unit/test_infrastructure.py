"""
基础设施配置与健康检查测试。

默认不依赖 Docker 或真实外部服务。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.core.health import INFRA_SERVICES, check_infrastructure_health
from app.main import create_app


def test_settings_loads_test_mode_defaults(monkeypatch) -> None:
    """Settings 默认使用 fallback，不访问真实服务。"""
    monkeypatch.delenv("QWEN_CHAT_MODEL", raising=False)
    settings = Settings(_env_file=None)

    assert settings.environment == "dev"
    assert settings.app_mode == "fallback"
    assert settings.postgres_url.startswith("postgresql://")
    assert settings.redis_url.startswith("redis://")
    assert settings.milvus_port == 19530
    assert settings.ingestion_execution_mode == "sync"
    assert settings.ingestion_task_store == "memory"
    assert settings.celery_broker_url == ""
    assert settings.qwen_chat_model == "qwen-plus"
    assert settings.qwen_api_base_url.startswith("https://dashscope.aliyuncs.com")
    assert settings.qwen_rerank_base_url.endswith("/compatible-api/v1")
    assert settings.embedding_batch_size == 10


def test_trace_exporter_matches_runtime_mode() -> None:
    """fallback 使用日志 trace，full 使用 OTLP/Phoenix trace exporter。"""
    from app.api.dependencies import _build_trace_exporter
    from app.services.observability.exporters.log_exporter import LogExporter
    from app.services.observability.exporters.otel_exporter import OTelTraceExporter

    fallback = _build_trace_exporter(Settings(_env_file=None, app_mode="fallback"))
    full = _build_trace_exporter(
        Settings(_env_file=None, app_mode="full", phoenix_endpoint="http://phoenix:6006")
    )

    assert isinstance(fallback, LogExporter)
    assert isinstance(full, OTelTraceExporter)
    assert full.endpoint == "http://phoenix:6006"
    assert full.strict is True


def test_health_returns_fallback_status_without_docker(monkeypatch) -> None:
    """fallback mode 下 health 不探测 Docker 服务。"""
    monkeypatch.setenv("APP_MODE", "fallback")
    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        resp = client.get("/health")
    finally:
        get_settings.cache_clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["mode"] == "fallback"
    assert set(data["services"]) == set(INFRA_SERVICES)
    assert {service["status"] for service in data["services"].values()} == {"skipped"}


def test_health_marks_down_service_when_connection_fails() -> None:
    """full mode 下连接失败应返回 down，而不是抛异常。"""
    settings = Settings(
        _env_file=None,
        app_mode="full",
        postgres_url="postgresql://localhost:1/enterprisemind",
        redis_url="redis://localhost:1/0",
        milvus_host="localhost",
        milvus_port=1,
        es_url="http://localhost:1",
        minio_endpoint="http://localhost:1",
        health_probe_timeout_seconds=0.05,
    )

    services = check_infrastructure_health(settings)

    assert set(services) == set(INFRA_SERVICES)
    assert {service["status"] for service in services.values()} == {"down"}
    assert all("error" in service for service in services.values())


def test_env_example_contains_required_keys() -> None:
    """.env.example 应包含 full mode 所需配置项且不包含真实密钥。"""
    content = Path(".env.example").read_text(encoding="utf-8")
    required_keys = [
        "ENVIRONMENT",
        "APP_MODE",
        "AGENT_RUN_ENGINE",
        "POSTGRES_URL",
        "REDIS_URL",
        "MILVUS_HOST",
        "MILVUS_PORT",
        "ES_URL",
        "MINIO_ENDPOINT",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_BUCKET",
        "INGESTION_EXECUTION_MODE",
        "INGESTION_TASK_STORE",
        "CELERY_BROKER_URL",
        "CELERY_RESULT_BACKEND",
        "CELERY_TASK_ALWAYS_EAGER",
        "QWEN_API_KEY",
        "QWEN_CHAT_MODEL",
        "QWEN_EMBEDDING_MODEL",
        "QWEN_RERANK_MODEL",
        "QWEN_API_BASE_URL",
        "QWEN_RERANK_BASE_URL",
        "QWEN_TIMEOUT_SECONDS",
        "RAGAS_TIMEOUT_SECONDS",
        "EMBEDDING_DIM",
        "EMBEDDING_BATCH_SIZE",
        "PHOENIX_ENDPOINT",
    ]

    for key in required_keys:
        assert f"{key}=" in content

    assert "sk-" not in content.lower()
