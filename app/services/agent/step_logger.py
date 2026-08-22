"""
Agent Step Logger。

记录每一步的执行，用于审计和调试。
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from app.schemas.agent import AgentStep

logger = logging.getLogger(__name__)


class StepLogger:
    """
    Agent Step 记录器。

    记录每一步的输入输出、证据和 token 消耗。
    """

    def __init__(self) -> None:
        self._steps: dict[str, list[AgentStep]] = {}  # run_id -> steps

    def log_step(
        self,
        run_id: str,
        node_name: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        evidence: list[dict] | None = None,
        token_usage: dict[str, int] | None = None,
        duration_ms: int = 0,
    ) -> AgentStep:
        """
        记录一个步骤。

        Args:
            run_id: Run ID
            node_name: 节点名称（如 intent, retrieve, generate）
            input_data: 输入数据
            output_data: 输出数据
            evidence: 相关证据（可选）
            token_usage: Token 使用情况（可选）
            duration_ms: 执行耗时（毫秒）

        Returns:
            创建的 AgentStep
        """
        step_id = f"step_{uuid.uuid4().hex[:12]}"

        step = AgentStep(
            id=step_id,
            run_id=run_id,
            node_name=node_name,
            input_data=input_data,
            output_data=output_data,
            evidence=evidence or [],
            token_usage=token_usage or {},
            duration_ms=duration_ms,
            created_at=datetime.now(UTC),
        )

        # 存储步骤
        if run_id not in self._steps:
            self._steps[run_id] = []
        self._steps[run_id].append(step)

        logger.info(
            "step_logged",
            extra={
                "run_id": run_id,
                "step_id": step_id,
                "node_name": node_name,
                "duration_ms": duration_ms,
            },
        )

        return step

    def get_steps(self, run_id: str) -> list[AgentStep]:
        """
        获取指定 Run 的所有步骤。

        Args:
            run_id: Run ID

        Returns:
            步骤列表
        """
        return self._steps.get(run_id, [])

    def get_step_count(self, run_id: str) -> int:
        """
        获取指定 Run 的步骤数量。

        Args:
            run_id: Run ID

        Returns:
            步骤数量
        """
        return len(self._steps.get(run_id, []))

    def clear(self, run_id: str) -> None:
        """
        清除指定 Run 的所有步骤（用于测试）。

        Args:
            run_id: Run ID
        """
        if run_id in self._steps:
            del self._steps[run_id]
