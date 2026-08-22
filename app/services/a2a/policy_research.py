"""只读 HR Policy Research A2A peer 与 in-process client。"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from app.core.exceptions import PermissionError, ValidationError
from app.schemas.protocol import A2AMessage, AgentCard, ArtifactRef, ProtocolTask
from app.schemas.user import UserContext


class PolicyResearchA2AAgent:
    """独立只读权限域中的制度研究 Agent reference implementation。"""

    _WRITE_MARKERS = (
        "创建",
        "修改",
        "删除",
        "审批",
        "提交工单",
        "create ",
        "update ",
        "delete ",
        "approve ",
    )

    def get_agent_card(self) -> AgentCard:
        """返回 A2A AgentCard。"""
        return AgentCard(
            name="HR Policy Research Agent",
            description="只读检索和比较 HR 制度，返回 evidence artifacts。",
            version="1.0.0",
            url="http://127.0.0.1:8000/a2a",
            capabilities={
                "streaming": False,
                "pushNotifications": False,
                "writeActions": False,
            },
            skills=[
                {
                    "id": "hr_policy_research",
                    "name": "HR Policy Research",
                    "description": "研究制度版本、适用范围和引用证据。",
                    "tags": ["hr", "policy", "evidence", "read-only"],
                }
            ],
        )

    async def send_message(
        self,
        *,
        context_id: str,
        text: str,
        user_context: UserContext,
        policy_version: str = "v1",
    ) -> ProtocolTask:
        """执行只读研究任务并返回 evidence artifact。"""
        if "hr.document.read" not in user_context.permissions:
            raise PermissionError("Policy Research Agent requires hr.document.read")
        lowered = text.lower()
        if any(marker in lowered for marker in self._WRITE_MARKERS):
            raise ValidationError("Policy Research Agent is read-only")

        now = datetime.now(UTC)
        citations = [
            {
                "document_id": "doc_hr_onboarding",
                "document_version": policy_version,
                "chunk_id": "chunk_policy_materials",
                "section": "入职材料",
                "quote": "新员工应提交身份证明、学历证明和离职证明。",
            }
        ]
        artifact = ArtifactRef(
            id=f"artifact_{uuid.uuid4().hex[:12]}",
            name="HR 制度研究结果",
            mime_type="application/json",
            uri=f"artifact://{context_id}/policy-research",
            content={"summary": "入职材料要求已完成只读研究。", "citations": citations},
            metadata={
                "citation_count": len(citations),
                "read_only": True,
                "tenant_id": user_context.tenant_id,
            },
        )
        return ProtocolTask(
            id=f"task_{uuid.uuid4().hex[:12]}",
            context_id=context_id,
            status="completed",
            messages=[
                A2AMessage(
                    id=f"msg_{uuid.uuid4().hex[:12]}",
                    role="user",
                    text=text,
                    created_at=now,
                )
            ],
            artifacts=[artifact],
            metadata={
                "agent": self.get_agent_card().name,
                "read_only": True,
                "policy_version": policy_version,
            },
        )


class InProcessA2AClient:
    """无需网络的 A2A client fake，与 HTTP adapter 共享协议对象。"""

    def __init__(self, agent: PolicyResearchA2AAgent) -> None:
        self._agent = agent

    async def get_agent_card(self) -> AgentCard:
        """读取 peer AgentCard。"""
        return self._agent.get_agent_card()

    async def send_message(
        self,
        *,
        context_id: str,
        text: str,
        user_context: UserContext,
        policy_version: str = "v1",
    ) -> ProtocolTask:
        """委托 goal-oriented 只读研究任务。"""
        return await self._agent.send_message(
            context_id=context_id,
            text=text,
            user_context=user_context,
            policy_version=policy_version,
        )
