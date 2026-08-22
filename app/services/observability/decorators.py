"""
Trace 装饰器。

使用装饰器包装业务函数，自动记录 Trace。
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from app.services.observability.context import TraceContext
from app.services.observability.span import SpanStatus, SpanType
from app.services.observability.tracer import Tracer

logger = logging.getLogger(__name__)


def trace_agent_run(tracer: Tracer) -> Callable:
    """
    装饰 Agent Run 函数。

    自动创建 trace 和 root span。

    Args:
        tracer: Trace 管理器

    Returns:
        装饰器
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 从参数中提取上下文
            run_id = kwargs.get("run_id", "unknown")
            user_id = kwargs.get("user_id", "unknown")
            tenant_id = kwargs.get("tenant_id", "unknown")

            # 开始 Trace
            context = tracer.start_trace(run_id, user_id, tenant_id)
            span = tracer.start_span(
                context=context, span_type=SpanType.AGENT_RUN, name=func.__name__
            )

            try:
                result = await func(*args, **kwargs)
                tracer.end_span(span, SpanStatus.OK)
                return result
            except Exception as e:
                tracer.record_error(span, e)
                tracer.end_span(span, SpanStatus.ERROR)
                raise
            finally:
                # 导出 Trace
                tracer.export_trace(context.trace_id)

        return wrapper

    return decorator


def trace_retrieval(tracer: Tracer, context: TraceContext) -> Callable:
    """
    装饰检索函数。

    Args:
        tracer: Trace 管理器
        context: Trace 上下文

    Returns:
        装饰器
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 获取父 Span
            parent = context.current_span

            span = tracer.start_span(
                context=context,
                span_type=SpanType.RETRIEVAL_SEARCH,
                name=func.__name__,
                parent=parent,
            )

            try:
                result = await func(*args, **kwargs)

                # 记录检索分数等属性
                if hasattr(result, "citation_count"):
                    span.set_attribute("citation_count", result.citation_count)
                if hasattr(result, "query_coverage"):
                    span.set_attribute("query_coverage", result.query_coverage)

                tracer.end_span(span, SpanStatus.OK)
                return result
            except Exception as e:
                tracer.record_error(span, e)
                tracer.end_span(span, SpanStatus.ERROR)
                raise

        return wrapper

    return decorator


def trace_tool_call(tracer: Tracer, context: TraceContext) -> Callable:
    """
    装饰工具调用函数。

    Args:
        tracer: Trace 管理器
        context: Trace 上下文

    Returns:
        装饰器
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tool_name = kwargs.get("tool_name", "unknown")

            # 获取父 Span
            parent = context.current_span

            span = tracer.start_span(
                context=context,
                span_type=SpanType.TOOL_CALL,
                name=tool_name,
                parent=parent,
                attributes={"tool_name": tool_name},
            )

            try:
                result = await func(*args, **kwargs)

                # 记录审批 ID
                if hasattr(result, "approval_id"):
                    span.set_attribute("approval_id", result.approval_id)

                tracer.end_span(span, SpanStatus.OK)
                return result
            except Exception as e:
                tracer.record_error(span, e)
                tracer.end_span(span, SpanStatus.ERROR)
                raise

        return wrapper

    return decorator


def trace_llm_call(tracer: Tracer, context: TraceContext) -> Callable:
    """
    装饰 LLM 调用函数。

    Args:
        tracer: Trace 管理器
        context: Trace 上下文

    Returns:
        装饰器
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            model_name = kwargs.get("model_name", "unknown")

            # 获取父 Span
            parent = context.current_span

            span = tracer.start_span(
                context=context,
                span_type=SpanType.LLM_CALL,
                name=func.__name__,
                parent=parent,
                attributes={"model_name": model_name},
            )

            try:
                result = await func(*args, **kwargs)

                # 记录 token 使用
                if hasattr(result, "token_input"):
                    span.set_attribute("token_input", result.token_input)
                if hasattr(result, "token_output"):
                    span.set_attribute("token_output", result.token_output)

                tracer.end_span(span, SpanStatus.OK)
                return result
            except Exception as e:
                tracer.record_error(span, e)
                tracer.end_span(span, SpanStatus.ERROR)
                raise

        return wrapper

    return decorator


def trace_guardrail_check(tracer: Tracer, context: TraceContext) -> Callable:
    """
    装饰安全检查函数。

    Args:
        tracer: Trace 管理器
        context: Trace 上下文

    Returns:
        装饰器
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 获取父 Span
            parent = context.current_span

            span = tracer.start_span(
                context=context,
                span_type=SpanType.GUARDRAIL_CHECK,
                name=func.__name__,
                parent=parent,
            )

            try:
                result = await func(*args, **kwargs)

                # 记录检查结果
                if hasattr(result, "is_safe"):
                    span.set_attribute("is_safe", result.is_safe)

                tracer.end_span(span, SpanStatus.OK)
                return result
            except Exception as e:
                tracer.record_error(span, e)
                tracer.end_span(span, SpanStatus.ERROR)
                raise

        return wrapper

    return decorator
