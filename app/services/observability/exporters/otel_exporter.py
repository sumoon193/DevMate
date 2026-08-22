"""
OpenTelemetry Trace 导出器。

将 trace/span 数据导出到 OTel Collector 或 Phoenix。
full 模式使用；fallback 模式使用 log_exporter。
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.observability.context import TraceContext
from app.services.observability.span import Span

logger = logging.getLogger(__name__)


class OTelTraceExporter:
    """
    OpenTelemetry Trace 导出器。

    通过 OTLP HTTP 协议将 trace 数据发送到 OTel Collector 或 Phoenix。
    如果连接失败，静默降级（不影响业务逻辑）。
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:6006",
        service_name: str = "enterprisemind",
        *,
        strict: bool = False,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._service_name = service_name
        self._strict = strict
        self._tracer: Any = None
        self._initialized = False

    @property
    def endpoint(self) -> str:
        """OTLP/Phoenix endpoint，用于健康检查和测试断言。"""
        return self._endpoint

    @property
    def strict(self) -> bool:
        """生产装配为 True，初始化或导出失败时禁止静默吞错。"""
        return self._strict

    def _ensure_init(self) -> None:
        """延迟初始化 OTel SDK（避免 import 时阻塞）。"""
        if self._initialized:
            return
        self._initialized = True

        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource.create({"service.name": self._service_name})
            provider = TracerProvider(resource=resource)

            exporter = OTLPSpanExporter(
                endpoint=f"{self._endpoint}/v1/traces",
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(self._service_name)

            logger.info("otel_exporter_initialized", extra={"endpoint": self._endpoint})

        except ImportError as exc:
            logger.warning("otel_sdk_not_installed_traces_disabled")
            if self._strict:
                raise RuntimeError("OTel SDK is required in full mode") from exc
        except Exception as e:
            logger.warning("otel_init_failed", extra={"error": str(e)})
            if self._strict:
                raise RuntimeError("OTel exporter initialization failed") from e

    def export(self, spans: list[Span], context: TraceContext) -> None:
        """将 spans 导出到 OTel Collector。"""
        self._ensure_init()

        if self._tracer is None:
            return  # OTel SDK 不可用，静默跳过

        try:
            for span_data in spans:
                attributes = {
                    "trace_id": context.trace_id,
                    "span_id": span_data.span_id,
                    "span_type": span_data.span_type.value
                    if hasattr(span_data.span_type, "value")
                    else str(span_data.span_type),
                    "status": span_data.status.value
                    if hasattr(span_data.status, "value")
                    else str(span_data.status),
                }
                for key, value in span_data.attributes.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        attributes[key] = value
                    else:
                        attributes[key] = str(value)

                otel_span = self._tracer.start_span(
                    name=span_data.name,
                    attributes=attributes,
                )
                for event in span_data.events:
                    otel_span.add_event(
                        name=event.name,
                        attributes=event.attributes,
                    )
                otel_span.end()

        except Exception as e:
            logger.warning("otel_export_failed", extra={"error": str(e)})
            if self._strict:
                raise RuntimeError("OTel trace export failed") from e
