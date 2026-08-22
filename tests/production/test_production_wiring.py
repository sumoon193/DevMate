"""生产装配必须失败关闭，不能静默使用离线 Fake。"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.core.exceptions import ValidationError
from app.services.answer.grounded_answer import FakeAnswerGenerator
from app.services.evaluation.ragas_adapter import FakeRAGASMetrics, RealRAGASMetrics


def test_full_mode_rejects_missing_qwen_key() -> None:
    from app.api.dependencies import _build_ai_adapters

    settings = Settings(_env_file=None, app_mode="full", qwen_api_key="")

    with pytest.raises(ValidationError, match="QWEN_API_KEY"):
        _build_ai_adapters(settings)


def test_fallback_mode_keeps_explicit_offline_fake() -> None:
    from app.api.dependencies import _build_ai_adapters

    settings = Settings(_env_file=None, app_mode="fallback", qwen_api_key="")

    answer_generator, _, _ = _build_ai_adapters(settings)

    assert isinstance(answer_generator, FakeAnswerGenerator)


def test_full_mode_builds_real_ragas_adapter() -> None:
    from app.api.dependencies import _build_ragas_metrics

    settings = Settings(
        _env_file=None,
        app_mode="full",
        qwen_api_key="sk-test",
        qwen_chat_model="qwen-plus",
    )

    assert isinstance(_build_ragas_metrics(settings), RealRAGASMetrics)


def test_fallback_mode_builds_fake_ragas_adapter() -> None:
    from app.api.dependencies import _build_ragas_metrics

    settings = Settings(_env_file=None, app_mode="fallback")

    assert isinstance(_build_ragas_metrics(settings), FakeRAGASMetrics)


def test_full_runtime_wires_persistent_memory_to_semantic_index() -> None:
    from app.api.dependencies import ServiceContainer

    source = inspect.getsource(ServiceContainer._init_runtime_services)

    assert "semantic_index=_build_memory_semantic_index" in source


def test_full_tool_registry_does_not_wire_mock_ticket_handler() -> None:
    from app.api.dependencies import _build_tool_registry

    settings = Settings(_env_file=None, app_mode="full")
    registry = _build_tool_registry(settings, session_factory=object())

    handler = registry.get_handler("create_hr_ticket")
    assert type(handler).__name__ == "McpToolHandler"


def test_full_mode_rejects_in_memory_graph_checkpointer() -> None:
    from app.services.graph.checkpointer import create_checkpointer_manager

    settings = Settings(
        _env_file=None,
        app_mode="full",
        graph_checkpointer_backend="memory",
    )

    with pytest.raises(ValidationError, match="postgres"):
        create_checkpointer_manager(settings)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ImportError("ragas missing"), RuntimeError("provider failed")])
async def test_real_ragas_never_converts_failure_to_fake_metrics(error: Exception) -> None:
    adapter = RealRAGASMetrics(api_key="sk-test")

    async def fail(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise error

    adapter._compute_ragas = fail  # type: ignore[method-assign]

    with pytest.raises(type(error), match=str(error)):
        await adapter.compute("question", "answer", ["context"], "ground truth")


@pytest.mark.asyncio
async def test_real_ragas_uses_v04_metric_contracts() -> None:
    calls: dict[str, dict[str, object]] = {}

    class Metric:
        def __init__(self, name: str, value: float) -> None:
            self.name = name
            self.value = value

        async def ascore(self, **kwargs: object) -> SimpleNamespace:
            calls[self.name] = kwargs
            return SimpleNamespace(value=self.value)

    adapter = RealRAGASMetrics(api_key="sk-test")
    adapter._build_metrics = lambda: {  # type: ignore[method-assign]
        "context_precision": Metric("context_precision", 0.81),
        "context_recall": Metric("context_recall", 0.82),
        "faithfulness": Metric("faithfulness", 0.83),
        "answer_relevancy": Metric("answer_relevancy", 0.84),
    }

    result = await adapter.compute("question", "answer", ["context"], "ground truth")

    assert result == {
        "context_precision": 0.81,
        "context_recall": 0.82,
        "faithfulness": 0.83,
        "answer_relevancy": 0.84,
    }
    context_args = {
        "user_input": "question",
        "retrieved_contexts": ["context"],
        "reference": "ground truth",
    }
    assert calls["context_precision"] == context_args
    assert calls["context_recall"] == context_args
    assert calls["faithfulness"] == {
        "user_input": "question",
        "response": "answer",
        "retrieved_contexts": ["context"],
    }
    assert calls["answer_relevancy"] == {
        "user_input": "question",
        "response": "answer",
    }


def test_real_ragas_source_uses_current_collections_api() -> None:
    source = inspect.getsource(RealRAGASMetrics)

    assert "ragas.metrics.collections" in source
    assert "from ragas import evaluate" not in source


def test_real_ragas_builds_installed_v04_metrics_without_network() -> None:
    adapter = RealRAGASMetrics(
        api_key="not-a-real-key",
        embedding_model="text-embedding-v4",
    )

    metrics = adapter._build_metrics()

    assert {name: type(metric).__name__ for name, metric in metrics.items()} == {
        "context_precision": "ContextPrecision",
        "context_recall": "ContextRecall",
        "faithfulness": "Faithfulness",
        "answer_relevancy": "AnswerRelevancy",
    }
    assert metrics["faithfulness"].llm.model_args == {
        "temperature": 0,
        "top_p": 0.1,
        "max_tokens": 4096,
    }


@pytest.mark.asyncio
async def test_real_ragas_runs_independent_metrics_concurrently() -> None:
    active = 0
    max_active = 0

    class Metric:
        async def ascore(self, **_kwargs: object) -> SimpleNamespace:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return SimpleNamespace(value=0.8)

    adapter = RealRAGASMetrics(api_key="not-a-real-key", timeout_seconds=1)
    adapter._build_metrics = lambda: {  # type: ignore[method-assign]
        "context_precision": Metric(),
        "context_recall": Metric(),
        "faithfulness": Metric(),
        "answer_relevancy": Metric(),
    }

    await adapter.compute("q", "a", ["c"], "r")

    assert max_active == 4


def test_real_ragas_loads_versioned_chinese_prompts_without_network() -> None:
    adapter = RealRAGASMetrics(api_key="not-a-real-key", language="chinese")
    metrics = adapter._build_metrics()

    adapter._localize_metric_prompts(metrics)

    localized = []
    for metric in metrics.values():
        for prompt in vars(metric).values():
            if getattr(prompt, "language", None) == "chinese":
                localized.append(prompt)
    assert len(localized) == 5
    assert all(prompt.examples for prompt in localized)
    serialized = " ".join(str(example) for prompt in localized for example in prompt.examples)
    assert "系统" in serialized


def test_real_ragas_rejects_unknown_prompt_language() -> None:
    adapter = RealRAGASMetrics(api_key="not-a-real-key", language="klingon")

    with pytest.raises(ValidationError, match="Unsupported RAGAS prompt language"):
        adapter._localize_metric_prompts({})


@pytest.mark.asyncio
async def test_real_ragas_rejects_nan_or_out_of_range_scores() -> None:
    class Metric:
        def __init__(self, value: float) -> None:
            self.value = value

        async def ascore(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(value=self.value)

    adapter = RealRAGASMetrics(api_key="not-a-real-key", timeout_seconds=1)
    adapter._build_metrics = lambda: {  # type: ignore[method-assign]
        "context_precision": Metric(0.8),
        "context_recall": Metric(0.8),
        "faithfulness": Metric(float("nan")),
        "answer_relevancy": Metric(0.8),
    }

    with pytest.raises(ValidationError, match="faithfulness"):
        await adapter.compute("q", "a", ["c"], "r")


def test_compose_declares_api_worker_and_otel_runtime() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    for service in ("api:", "worker:", "phoenix:"):
        assert service in compose
    assert "APP_MODE: full" in compose
    assert "GRAPH_CHECKPOINTER_BACKEND: postgres" in compose
    assert "INGESTION_EXECUTION_MODE: celery" in compose
    assert "RAGAS_LANGUAGE:" in compose
    assert "celery" in compose
    worker_section = compose.split("  worker:", 1)[1]
    assert "healthcheck:" in worker_section
    assert "inspect ping" in worker_section
    assert "uv sync --frozen --no-dev --no-editable" in dockerfile
    assert "uv sync --frozen --no-dev --no-install-project" in dockerfile
    assert "pip install --no-cache-dir ." not in dockerfile


def test_celery_exposes_real_broker_roundtrip_task() -> None:
    source = Path("app/services/ingestion/celery_tasks.py").read_text(encoding="utf-8")

    assert 'name="system.ping"' in source
    assert "def system_ping" in source


def test_runtime_tool_names_do_not_expose_mock_semantics() -> None:
    sources = "\n".join(path.read_text(encoding="utf-8") for path in Path("app").rglob("*.py"))

    legacy_mock_name = "create_" + "mock_hr_ticket"
    assert legacy_mock_name not in sources
    assert "create_hr_ticket" in sources
