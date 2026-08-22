"""
Agent Harness 深化能力 Schema。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LoopStage(str, Enum):
    """Agent Loop 阶段。"""

    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    REFLECT = "reflect"
    REPAIR = "repair"


class LoopDecision(str, Enum):
    """Reflection 阶段产生的治理决策。"""

    CONTINUE = "continue"
    REPAIR = "repair"
    AWAIT_APPROVAL = "await_approval"
    REFUSE = "refuse"


class TimelineEventType(str, Enum):
    """Artifact Timeline 事件类型。"""

    RUN_CREATED = "run_created"
    EVIDENCE_RETRIEVED = "evidence_retrieved"
    PLAN_GENERATED = "plan_generated"
    TOOL_CALL_PREPARED = "tool_call_prepared"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
    TOOL_EXECUTED = "tool_executed"
    REFLECTION_CREATED = "reflection_created"
    REPAIR_ACTION_CREATED = "repair_action_created"
    ANSWER_GENERATED = "answer_generated"
    EVAL_COMPLETED = "eval_completed"


class LoopEvent(BaseModel):
    """可审计的 Loop 事件。"""

    id: str = Field(description="事件 ID")
    run_id: str = Field(description="所属 Agent Run")
    stage: LoopStage = Field(description="Loop 阶段")
    decision: LoopDecision = Field(description="治理决策")
    reasons: list[str] = Field(default_factory=list, description="决策原因")
    action: str | None = Field(default=None, description="建议或修复动作")
    previous_failure_reason: str | None = Field(
        default=None,
        description="前一次失败原因",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="创建时间",
    )


class RunBudget(BaseModel):
    """限制 Agent 循环、工具、token、成本和墙钟时间。"""

    max_iterations: int = Field(default=4, ge=1)
    max_repairs: int = Field(default=2, ge=0)
    max_tool_calls: int = Field(default=8, ge=0)
    max_tokens: int = Field(default=16_000, ge=0)
    max_cost: float = Field(default=10.0, ge=0.0)
    max_duration_seconds: int = Field(default=300, ge=1)


class LoopRunResult(BaseModel):
    """受治理循环的确定性执行结果。"""

    run_id: str
    status: str
    iterations_used: int = Field(ge=0)
    repairs_used: int = Field(ge=0)
    events: list[LoopEvent] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)


class TimelineEvent(BaseModel):
    """Agent Run Artifact Timeline 事件。"""

    id: str = Field(description="Timeline 事件 ID")
    run_id: str = Field(description="所属 Agent Run")
    event_type: TimelineEventType = Field(description="事件类型")
    timestamp: datetime = Field(description="事件时间")
    stage: str = Field(description="阶段名称")
    input_summary: str = Field(default="", description="输入摘要")
    output_summary: str = Field(default="", description="输出摘要")
    risk_level: str | None = Field(default=None, description="风险等级")
    citation_ids: list[int] = Field(default_factory=list, description="关联 citation")
    trace_span_id: str | None = Field(default=None, description="关联 trace span")
    approval_status: str | None = Field(default=None, description="审批状态")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")
