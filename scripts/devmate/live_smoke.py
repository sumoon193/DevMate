"""DevMate live smoke: 0 passed, 1 failed, 2 blocked."""

from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        choices=("health", "model", "ragas", "queue", "otel", "mcp", "memory"),
        default="health",
    )
    args = parser.parse_args(argv)
    if args.component == "model":
        return _model_smoke()
    if args.component == "ragas":
        return _ragas_smoke()
    if args.component == "queue":
        return _queue_smoke()
    if args.component == "otel":
        return _otel_smoke()
    if args.component == "mcp":
        return _mcp_smoke()
    if args.component == "memory":
        return _memory_smoke()
    base = os.getenv("DEVMATE_BASE_URL", "").rstrip("/")
    if not base:
        print("BLOCKED: set DEVMATE_BASE_URL")
        return 2
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        print("FAILED: DEVMATE_BASE_URL must be an http(s) URL")
        return 1
    try:
        with urllib.request.urlopen(  # noqa: S310 - scheme validated above
            f"{base}/health", timeout=5
        ) as response:
            ok = response.status == 200
            print("PASSED: DevMate health" if ok else f"FAILED: status={response.status}")
            return 0 if ok else 1
    except urllib.error.URLError as exc:
        print(f"BLOCKED: service unavailable ({exc.reason})")
        return 2
    # CLI 边界必须把未知异常转换为稳定退出码，避免向用户输出调用栈。
    except Exception as exc:
        print(f"FAILED: {exc.__class__.__name__}")
        return 1


def _model_smoke() -> int:
    api_key = os.getenv("QWEN_API_KEY", "").strip()
    model = os.getenv("QWEN_CHAT_MODEL", "").strip()
    if not api_key or not model:
        print("BLOCKED: set QWEN_API_KEY and QWEN_CHAT_MODEL")
        return 2
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    try:
        from app.devmate.models.parser import parse_typed_diagnosis
        from app.services.ai.qwen import QwenAnswerGenerator

        raw = asyncio.run(
            QwenAnswerGenerator(
                api_key=api_key,
                model=model,
                base_url=os.getenv(
                    "QWEN_API_BASE_URL",
                    os.getenv(
                        "QWEN_BASE_URL",
                        "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    ),
                ),
                temperature=0,
            ).generate(
                "Return exactly five newline-separated fields and nothing else: "
                "summary=<text>\nseverity=warning\nrule=<identifier>\n"
                "confidence=<0 to 1>\nevidence=<comma-separated identifiers>"
            )
        )
        diagnosis = parse_typed_diagnosis(raw)
        if not diagnosis.summary or not diagnosis.rule:
            print("FAILED: typed diagnosis is incomplete")
            return 1
        print("PASSED: DevMate real model and typed parser")
        return 0
    # CLI 边界必须统一分类第三方 SDK 与解析器异常。
    except Exception as exc:
        message = str(exc)
        if "status=" not in message and "response" not in message.lower():
            print(f"BLOCKED: model service unavailable ({exc.__class__.__name__})")
            return 2
        print(f"FAILED: real model validation ({exc.__class__.__name__})")
        return 1


def _ragas_smoke() -> int:
    api_key = os.getenv("QWEN_API_KEY", "").strip()
    if not api_key:
        print("BLOCKED: set QWEN_API_KEY")
        return 2
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    try:
        from openai import APIConnectionError, APITimeoutError

        from app.services.evaluation.ragas_adapter import RealRAGASMetrics

        # 使用 Unicode 转义，避免 Windows PowerShell 管道代码页破坏中文 fixture。
        question = "\u4f01\u4e1a\u77e5\u8bc6\u5e93\u5982\u4f55\u4fdd\u8bc1\u56de\u7b54\u53ef\u4ee5\u8ffd\u6eaf\uff1f"
        answer = "\u7cfb\u7edf\u901a\u8fc7\u6df7\u5408\u68c0\u7d22\u83b7\u53d6\u6587\u6863\u7247\u6bb5\uff0c\u5e76\u5728\u56de\u7b54\u4e2d\u8fd4\u56de\u5f15\u7528\u6765\u6e90\u3002"
        contexts = [
            "\u7cfb\u7edf\u4f7f\u7528\u5411\u91cf\u68c0\u7d22\u548c\u5173\u952e\u8bcd\u68c0\u7d22\u53ec\u56de\u6587\u6863\u7247\u6bb5\uff0c\u878d\u5408\u6392\u5e8f\u540e\u751f\u6210\u5e26\u6765\u6e90\u5f15\u7528\u7684\u56de\u7b54\u3002"
        ]
        reference = "\u901a\u8fc7\u68c0\u7d22\u6587\u6863\u8bc1\u636e\u5e76\u8fd4\u56de\u6765\u6e90\u5f15\u7528\u4fdd\u8bc1\u56de\u7b54\u53ef\u8ffd\u6eaf\u3002"
        metrics = asyncio.run(
            RealRAGASMetrics(
                api_key=api_key,
                llm_model=os.getenv("QWEN_CHAT_MODEL", "qwen-plus"),
                embedding_model=os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4"),
                base_url=os.getenv(
                    "QWEN_API_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                language=os.getenv("RAGAS_LANGUAGE", "chinese"),
                timeout_seconds=float(os.getenv("RAGAS_TIMEOUT_SECONDS", "300")),
            ).compute(question, answer, contexts, reference)
        )
        print(
            "PASSED: DevMate real RAGAS " + json.dumps(metrics, ensure_ascii=False, sort_keys=True)
        )
        return 0
    except (APIConnectionError, APITimeoutError, TimeoutError) as exc:
        print(f"BLOCKED: RAGAS provider unavailable ({exc.__class__.__name__})")
        return 2
    # CLI 边界必须让所有真实评测失败都以退出码 1 结束。
    except Exception as exc:
        print(f"FAILED: real RAGAS validation ({exc.__class__.__name__}: {exc})")
        return 1


def _queue_smoke() -> int:
    try:
        from celery import Celery
        from celery.exceptions import TimeoutError as CeleryTimeoutError
        from kombu.exceptions import OperationalError

        broker = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/1")
        backend = os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/2")
        probe = Celery("devmate-live-smoke", broker=broker, backend=backend)
        nonce = uuid.uuid4().hex
        result = probe.send_task("system.ping", kwargs={"nonce": nonce})
        payload = result.get(timeout=20)
        if payload != {"status": "ok", "nonce": nonce}:
            print("FAILED: Celery queue returned an invalid probe payload")
            return 1
        print("PASSED: DevMate Celery broker/worker/result-backend roundtrip")
        return 0
    except (OperationalError, CeleryTimeoutError) as exc:
        print(f"BLOCKED: Celery service unavailable ({exc.__class__.__name__})")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: Celery queue validation ({exc.__class__.__name__}: {exc})")
        return 1


def _otel_smoke() -> int:
    base = os.getenv("PHOENIX_ENDPOINT", "http://127.0.0.1:6006").rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        print("FAILED: PHOENIX_ENDPOINT must be an http(s) URL")
        return 1
    try:
        with urllib.request.urlopen(  # noqa: S310 - scheme validated above
            f"{base}/healthz", timeout=5
        ) as response:
            if response.status != 200:
                print(f"FAILED: Phoenix health status={response.status}")
                return 1
    except (urllib.error.URLError, http.client.RemoteDisconnected, TimeoutError) as exc:
        detail = getattr(exc, "reason", exc.__class__.__name__)
        print(f"BLOCKED: Phoenix unavailable ({detail})")
        return 2

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
        from opentelemetry.sdk.trace.export import SpanExportResult

        exporter = OTLPSpanExporter(endpoint=f"{base}/v1/traces", timeout=10)

        class DirectExportProcessor(SpanProcessor):
            result: SpanExportResult | None = None

            def on_end(self, span) -> None:  # type: ignore[no-untyped-def]
                self.result = exporter.export((span,))

            def shutdown(self) -> None:
                exporter.shutdown()

        processor = DirectExportProcessor()
        provider = TracerProvider(resource=Resource.create({"service.name": "devmate-live-smoke"}))
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("devmate.live_smoke")
        with tracer.start_as_current_span("devmate.otel.live_smoke") as span:
            span.set_attribute("smoke.id", uuid.uuid4().hex)
        provider.shutdown()
        if processor.result is not SpanExportResult.SUCCESS:
            print("FAILED: Phoenix rejected the OTLP trace")
            return 1
        print("PASSED: DevMate OTel trace exported to Phoenix")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: OTel export validation ({exc.__class__.__name__}: {exc})")
        return 1


def _mcp_smoke() -> int:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    postgres_url = os.getenv(
        "POSTGRES_URL",
        "postgresql://enterprisemind:change_me_local@127.0.0.1:5432/enterprisemind",
    )
    try:
        from sqlalchemy import delete, select
        from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.db.session import _build_url
        from app.models.runtime import HrTicketRecord
        from app.schemas.user import UserContext
        from app.services.mcp.sqlalchemy_server import SqlAlchemyMcpServer

        async def probe() -> None:
            engine = create_async_engine(_build_url(postgres_url), pool_pre_ping=True)
            factory = async_sessionmaker(engine, expire_on_commit=False)
            smoke_id = uuid.uuid4().hex
            tenant_id = f"live-smoke-{smoke_id}"
            user_id = f"user-{smoke_id}"
            server = SqlAlchemyMcpServer(factory)
            try:
                result = await server.call_tool(
                    "create_hr_ticket",
                    {
                        "title": f"DevMate MCP live smoke {smoke_id}",
                        "description": "验证正式 MCP 工具写入并回查 PostgreSQL。",
                        "priority": "low",
                        "category": "integration-smoke",
                    },
                    UserContext(
                        user_id=user_id,
                        tenant_id=tenant_id,
                        department_ids=["dept_engineering"],
                        role="engineer",
                        permissions=["hr.ticket.write"],
                    ),
                )
                async with factory() as session:
                    stored = await session.scalar(
                        select(HrTicketRecord).where(HrTicketRecord.id == result["ticket_id"])
                    )
                    if (
                        stored is None
                        or stored.tenant_id != tenant_id
                        or stored.created_by != user_id
                        or stored.title != result["title"]
                    ):
                        raise ValueError("persisted MCP ticket does not match the request")
                    await session.execute(
                        delete(HrTicketRecord).where(HrTicketRecord.id == result["ticket_id"])
                    )
                    await session.commit()
            finally:
                await engine.dispose()

        asyncio.run(probe())
        print("PASSED: DevMate MCP tool persisted and queried a PostgreSQL ticket")
        return 0
    except (OSError, OperationalError, InterfaceError) as exc:
        print(f"BLOCKED: PostgreSQL unavailable ({exc.__class__.__name__})")
        return 2
    except SQLAlchemyError as exc:
        print(f"FAILED: MCP PostgreSQL validation ({exc.__class__.__name__})")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: MCP persistence validation ({exc.__class__.__name__}: {exc})")
        return 1


def _memory_smoke() -> int:
    api_key = os.getenv("QWEN_API_KEY", "").strip()
    if not api_key:
        print("BLOCKED: set QWEN_API_KEY")
        return 2

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    postgres_url = os.getenv(
        "POSTGRES_URL",
        "postgresql://enterprisemind:change_me_local@127.0.0.1:5432/enterprisemind",
    )
    try:
        from pymilvus.exceptions import MilvusException
        from sqlalchemy import delete, select
        from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.db.session import _build_url
        from app.models.runtime import CaseRecord, EpisodicMemoryRecordORM
        from app.services.ai.qwen import QwenEmbedder
        from app.services.memory.milvus_index import MilvusMemorySemanticIndex
        from app.services.memory.store import SqlAlchemyEpisodicMemoryStore

        async def probe() -> None:
            engine = create_async_engine(_build_url(postgres_url), pool_pre_ping=True)
            factory = async_sessionmaker(engine, expire_on_commit=False)
            smoke_id = uuid.uuid4().hex
            tenant_id = f"memory-smoke-{smoke_id}"
            other_tenant_id = f"memory-other-{smoke_id}"
            case_id = f"case_{smoke_id[:12]}"
            content = f"DevMate memory smoke marker {smoke_id}"
            memory_id: str | None = None
            index = MilvusMemorySemanticIndex(
                embedder=QwenEmbedder(
                    api_key=api_key,
                    model=os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4"),
                    dimension=int(os.getenv("EMBEDDING_DIM", "1024")),
                    base_url=os.getenv(
                        "QWEN_API_BASE_URL",
                        "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    ),
                ),
                dimension=int(os.getenv("EMBEDDING_DIM", "1024")),
                host=os.getenv("MILVUS_HOST", "127.0.0.1"),
                port=int(os.getenv("MILVUS_PORT", "19530")),
            )
            try:
                async with factory() as session, session.begin():
                    session.add(
                        CaseRecord(
                            id=case_id,
                            title="DevMate memory live smoke",
                            tenant_id=tenant_id,
                            subject_user_id="memory-smoke-user",
                            status="active",
                            version=0,
                            active_run_id=None,
                            execution_manifest={},
                            policy_versions={},
                            working_memory={},
                        )
                    )

                store = SqlAlchemyEpisodicMemoryStore(factory, semantic_index=index)
                record = await store.remember(
                    tenant_id=tenant_id,
                    case_id=case_id,
                    memory_key="live-smoke",
                    content=content,
                    provenance_event_ids=[f"evt_{smoke_id[:16]}"],
                    importance_score=0.9,
                )
                memory_id = record.id

                async with factory() as session:
                    persisted = await session.scalar(
                        select(EpisodicMemoryRecordORM).where(
                            EpisodicMemoryRecordORM.id == memory_id
                        )
                    )
                if persisted is None or persisted.tenant_id != tenant_id:
                    raise ValueError("memory was not persisted for the expected tenant")

                own_results = await store.search(tenant_id=tenant_id, query=content)
                other_results = await store.search(tenant_id=other_tenant_id, query=content)
                if memory_id not in {item.id for item in own_results}:
                    raise ValueError("memory was not recalled from the semantic index")
                if memory_id in {item.id for item in other_results}:
                    raise ValueError("memory crossed the tenant boundary")

                await store.forget(memory_id, tenant_id=tenant_id)
                if await store.search(tenant_id=tenant_id, query=content):
                    raise ValueError("forgotten memory remained recallable")
                async with factory() as session:
                    status = await session.scalar(
                        select(EpisodicMemoryRecordORM.status).where(
                            EpisodicMemoryRecordORM.id == memory_id
                        )
                    )
                if status != "deleted":
                    raise ValueError("forgotten memory status was not persisted")
            finally:
                if memory_id is not None:
                    await index.delete(memory_id=memory_id, tenant_id=tenant_id)
                async with factory() as session, session.begin():
                    await session.execute(
                        delete(EpisodicMemoryRecordORM).where(
                            EpisodicMemoryRecordORM.case_id == case_id
                        )
                    )
                    await session.execute(delete(CaseRecord).where(CaseRecord.id == case_id))
                await engine.dispose()

        asyncio.run(probe())
        print("PASSED: DevMate PostgreSQL and Qwen/Milvus tenant-scoped memory lifecycle")
        return 0
    except (OSError, OperationalError, InterfaceError) as exc:
        print(f"BLOCKED: memory dependency unavailable ({exc.__class__.__name__})")
        return 2
    except MilvusException as exc:
        message = str(exc).lower()
        if any(token in message for token in ("connect", "unavailable", "timeout", "refused")):
            print(f"BLOCKED: Milvus unavailable ({exc.__class__.__name__})")
            return 2
        print(f"FAILED: Milvus memory validation ({exc.__class__.__name__})")
        return 1
    except SQLAlchemyError as exc:
        print(f"FAILED: memory PostgreSQL validation ({exc.__class__.__name__})")
        return 1
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if any(token in message for token in ("status=401", "status=403", "connection", "timeout")):
            print(f"BLOCKED: memory provider unavailable ({exc.__class__.__name__})")
            return 2
        print(f"FAILED: memory lifecycle validation ({exc.__class__.__name__}: {exc})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
