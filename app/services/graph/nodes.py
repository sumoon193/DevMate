"""
LangGraph 节点函数。

实现 Graph 中各个节点的逻辑。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.types import interrupt

from app.schemas.agent import AgentPlan
from app.schemas.approval import ApprovalDecision
from app.schemas.enums import ApprovalDecisionType, ApprovalStatus, ToolCallStatus
from app.schemas.tool import ToolCall
from app.services.agent.run_manager import AgentRunManager
from app.services.graph.state import AgentGraphState
from app.services.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


async def intent_node(state: AgentGraphState, run_manager: AgentRunManager) -> dict[str, Any]:
    """
    意图分析节点。

    分析用户意图，决定后续流程。

    Args:
        state: 当前状态
        run_manager: Agent Run Manager

    Returns:
        状态更新
    """
    run_id = state["run_id"]
    question = state["question"]

    logger.info("intent_node", extra={"run_id": run_id, "question": question[:50]})

    # V1 简单规则判断意图
    intent = _classify_intent(question)

    # 更新 Run 状态
    await run_manager.start_run(run_id)

    return {
        "intent": intent,
        "rewritten_queries": [question],  # V1 不改写
    }


async def query_rewrite_node(
    state: AgentGraphState, run_manager: AgentRunManager
) -> dict[str, Any]:
    """
    查询改写节点。

    改写用户查询以提高检索效果。

    Args:
        state: 当前状态
        run_manager: Agent Run Manager

    Returns:
        状态更新
    """
    run_id = state["run_id"]
    question = state["question"]

    logger.info("query_rewrite_node", extra={"run_id": run_id})

    # V1 简单改写（实际上不改写）
    rewritten_queries = [question]

    return {"rewritten_queries": rewritten_queries}


async def retrieve_node(
    state: AgentGraphState, run_manager: AgentRunManager, hybrid_retriever: HybridRetriever
) -> dict[str, Any]:
    """
    检索节点。

    调用 HybridRetriever 检索证据。

    Args:
        state: 当前状态
        run_manager: Agent Run Manager
        hybrid_retriever: 混合检索器

    Returns:
        状态更新
    """
    run_id = state["run_id"]
    question = state["question"]
    user = state["user"]

    logger.info("retrieve_node", extra={"run_id": run_id})

    try:
        # 检索证据
        evidence = await hybrid_retriever.retrieve(
            query=question, tenant_id=user.tenant_id, department_ids=user.department_ids, top_k=5
        )

        # 更新 Run 状态
        await run_manager.retrieve_evidence(run_id, evidence)

        # 转换为可序列化的 dict
        evidence_dict = {
            "evidence_list": [
                {
                    "id": c.id,
                    "document_name": c.document_name,
                    "section": c.section,
                    "page": c.page,
                    "chunk_text": c.chunk_text[:200],
                    "score": c.score,
                    "rerank_score": c.rerank_score,
                }
                for c in evidence.evidence_list
            ],
            "total_count": evidence.total_count,
            "query_coverage_score": evidence.query_coverage_score,
        }

        return {"evidence": evidence_dict}

    except Exception as e:
        logger.error("retrieve_failed", extra={"run_id": run_id, "error": str(e)})
        return {"evidence": None, "errors": state.get("errors", []) + [f"检索失败: {e!s}"]}


async def evidence_score_node(
    state: AgentGraphState, run_manager: AgentRunManager
) -> dict[str, Any]:
    """
    证据评分节点。

    评估证据质量。

    Args:
        state: 当前状态
        run_manager: Agent Run Manager

    Returns:
        状态更新
    """
    run_id = state["run_id"]
    evidence = state.get("evidence")

    logger.info("evidence_score_node", extra={"run_id": run_id})

    # V1 简单评分（使用 evidence_bundle 的 query_coverage_score）
    if evidence:
        coverage = evidence.get("query_coverage_score", 0.0)
        count = evidence.get("total_count", 0)
        logger.info(
            "evidence_scored", extra={"run_id": run_id, "coverage": coverage, "count": count}
        )

    return {}  # 不修改状态


async def plan_node(state: AgentGraphState, run_manager: AgentRunManager) -> dict[str, Any]:
    """
    计划节点。

    生成执行计划。

    Args:
        state: 当前状态
        run_manager: Agent Run Manager

    Returns:
        状态更新
    """
    run_id = state["run_id"]
    intent = state.get("intent", "hr_query")

    logger.info("plan_node", extra={"run_id": run_id, "intent": intent})

    # V1 简单计划生成
    if intent == "tool_request":
        # 需要工具的请求
        plan = AgentPlan(
            id=f"plan_{uuid.uuid4().hex[:8]}",
            run_id=run_id,
            steps=["create_hr_ticket"],
            current_step_index=0,
        )
    else:
        # 普通查询
        plan = AgentPlan(
            id=f"plan_{uuid.uuid4().hex[:8]}",
            run_id=run_id,
            steps=["policy_search"],
            current_step_index=0,
        )

    # 更新 Run 状态
    await run_manager.create_plan(run_id, plan)

    return {"plan": plan}


async def approval_gate_node(
    state: AgentGraphState, run_manager: AgentRunManager
) -> dict[str, Any]:
    """
    审批门控节点。

    检查是否需要审批，如果需要则触发 interrupt。

    Args:
        state: 当前状态
        run_manager: Agent Run Manager

    Returns:
        状态更新
    """
    run_id = state["run_id"]
    plan = state.get("plan")
    user = state["user"]

    logger.info("approval_gate_node", extra={"run_id": run_id})

    if not plan or not plan.steps:
        return {"pending_tool_call": None}

    # 获取当前要执行的工具
    current_step = plan.steps[plan.current_step_index]
    tool_name = current_step
    parameters = _build_tool_parameters(tool_name, state["question"])

    pending_approvals = await run_manager.get_pending_approvals(run_id)
    existing_approval = next(
        (approval for approval in pending_approvals if approval.tool_name == tool_name), None
    )

    if existing_approval:
        tool_call = ToolCall(
            id=existing_approval.tool_call_id,
            run_id=run_id,
            tool_name=existing_approval.tool_name,
            parameters=existing_approval.parameters,
            result=None,
            status=ToolCallStatus.PENDING,
            approval_required=True,
        )
        approval_id = existing_approval.id
        parameters = existing_approval.parameters
    else:
        # 执行工具（写入型工具会创建审批请求，但不会真正执行）
        tool_call = await run_manager.execute_tool(
            run_id=run_id, tool_name=tool_name, parameters=parameters, user_context=user
        )

        if not tool_call.approval_required or tool_call.status != ToolCallStatus.PENDING:
            return {"pending_tool_call": None, "pending_approval_id": None}

        created_approvals = await run_manager.get_pending_approvals(run_id)
        approval_request = next(
            approval for approval in created_approvals if approval.tool_call_id == tool_call.id
        )
        approval_id = approval_request.id

    payload = _build_interrupt_payload(
        run_id=run_id,
        approval_id=approval_id,
        tool_name=tool_name,
        tool_args=parameters,
        evidence=state.get("evidence"),
    )

    # 策略化自动审批：仅对本次新建的审批生效；resume 恢复路径（existing_approval
    # 非空）始终走 interrupt 接收人工决策，避免策略覆盖人工提交的审批结果。
    auto_decision = (
        None
        if existing_approval is not None
        else run_manager.maybe_auto_approve(
            run_id=run_id,
            approval_id=approval_id,
            user_context=user,
        )
    )
    if auto_decision is not None:
        logger.info(
            "approval_auto_approved",
            extra={"run_id": run_id, "approval_id": approval_id, "tool_name": tool_name},
        )
        approval_decision = auto_decision
    else:
        logger.info("approval_required", extra={"run_id": run_id, "tool_name": tool_name})
        decision_payload = interrupt(payload)
        approval_decision = ApprovalDecision.model_validate(decision_payload)

    return {
        "pending_tool_call": tool_call,
        "pending_approval_id": approval_id,
        "approval_decision": approval_decision,
    }


async def tool_execute_node(state: AgentGraphState, run_manager: AgentRunManager) -> dict[str, Any]:
    """
    工具执行节点。

    执行工具（审批后）。

    Args:
        state: 当前状态
        run_manager: Agent Run Manager

    Returns:
        状态更新
    """
    run_id = state["run_id"]
    approval_decision = state.get("approval_decision")
    pending_tool_call = state.get("pending_tool_call")
    pending_approval_id = state.get("pending_approval_id")
    user = state["user"]

    logger.info("tool_execute_node", extra={"run_id": run_id})

    tool_results = state.get("tool_results", [])

    if pending_tool_call and approval_decision and pending_approval_id:
        approval_request = await run_manager.apply_approval_decision(
            run_id=run_id,
            approval_id=pending_approval_id,
            approval_decision=approval_decision,
            user_context=user,
        )

        if approval_request.status == ApprovalStatus.REJECTED:
            await run_manager.mark_resumed_without_tool(
                run_id, f"approval rejected for {approval_request.tool_name}"
            )
            return {
                "tool_results": tool_results,
                "pending_tool_call": None,
                "pending_approval_id": None,
                "approval_decision": approval_decision,
                "answer": {
                    "answer": f"已拒绝执行工具 {approval_request.tool_name}，不会创建工单。",
                    "citations": [],
                    "confidence": 1.0,
                },
            }

        executed_tool = await run_manager.execute_approved_tool(
            run_id=run_id, approval_id=approval_request.id, user_context=user
        )
        tool_results.append(executed_tool)

    return {
        "tool_results": tool_results,
        "pending_tool_call": None,
        "pending_approval_id": None,
        "approval_decision": None,
    }


async def answer_node(
    state: AgentGraphState,
    run_manager: AgentRunManager,
    answer_service: object | None = None,
) -> dict[str, Any]:
    """
    答案生成节点。

    当 answer_service 存在时使用真实 LLM 生成答案，
    否则回退到 V1 简单拼接。

    Args:
        state: 当前状态
        run_manager: Agent Run Manager
        answer_service: GroundedAnswerService（可选）

    Returns:
        状态更新
    """
    run_id = state["run_id"]
    question = state["question"]
    evidence = state.get("evidence")
    tool_results = state.get("tool_results", [])

    logger.info("answer_node", extra={"run_id": run_id})

    # 如果已有答案（审批拒绝路径），直接返回
    if state.get("answer"):
        return {"answer": state["answer"]}

    # 尝试使用真实 answer_service（GroundedAnswerService）
    if answer_service is not None and evidence and evidence.get("evidence_list"):
        try:
            from app.schemas.chunk import Citation, EvidenceBundle

            citations = [
                Citation(
                    id=c.get("id", i),
                    document_name=c.get("document_name", ""),
                    section=c.get("section", ""),
                    page=c.get("page", 0),
                    chunk_text=c.get("chunk_text", ""),
                    score=c.get("score", 0.0),
                    rerank_score=c.get("rerank_score", 0.0),
                )
                for i, c in enumerate(evidence["evidence_list"])
            ]
            evidence_bundle = EvidenceBundle(
                evidence_list=citations,
                total_count=evidence.get("total_count", len(citations)),
                query_coverage_score=evidence.get("query_coverage_score", 0.0),
            )

            result = await answer_service.answer(
                question=question,
                evidence=evidence_bundle,
            )

            answer = {
                "answer": result.answer,
                "citations": [c.model_dump(mode="json") for c in result.citations],
                "confidence": result.confidence,
            }
            return {"answer": answer}

        except Exception as e:
            logger.error("answer_service_failed", extra={"run_id": run_id, "error": str(e)})
            # 降级到简单拼接

    # V1 简单答案生成（fallback）
    if evidence and evidence.get("evidence_list"):
        evidence_list = evidence["evidence_list"]
        citations = [
            {
                "id": c.get("id"),
                "document_name": c.get("document_name"),
                "section": c.get("section"),
                "page": c.get("page"),
                "chunk_text": c.get("chunk_text", "")[:100],
            }
            for c in evidence_list[:3]
        ]

        first_text = evidence_list[0].get("chunk_text", "")[:200]
        coverage = evidence.get("query_coverage_score", 0.0)

        answer = {
            "answer": f"根据公司制度，关于「{question}」的回答：\n\n" + first_text + "...",
            "citations": citations,
            "confidence": coverage,
        }
    elif tool_results:
        answer = {
            "answer": f"根据工具执行结果，关于「{question}」：\n\n" + str(tool_results[0].result),
            "citations": [],
            "confidence": 0.8,
        }
    else:
        answer = {
            "answer": f"抱歉，关于「{question}」，我暂时无法找到足够的证据来回答。请提供更多上下文或联系 HR 部门。",
            "citations": [],
            "confidence": 0.0,
        }

    return {"answer": answer}


async def fact_check_node(state: AgentGraphState, run_manager: AgentRunManager) -> dict[str, Any]:
    """
    事实核查节点。

    验证答案的准确性。

    Args:
        state: 当前状态
        run_manager: Agent Run Manager

    Returns:
        状态更新
    """
    run_id = state["run_id"]
    answer = state.get("answer")

    logger.info("fact_check_node", extra={"run_id": run_id})

    # V1 简单检查（实际上不修改）
    if answer:
        confidence = answer.get("confidence", 0.0)
        if confidence < 0.3:
            logger.warning(
                "low_confidence_answer", extra={"run_id": run_id, "confidence": confidence}
            )

    return {}  # 不修改状态


async def finalize_node(state: AgentGraphState, run_manager: AgentRunManager) -> dict[str, Any]:
    """
    完成节点。

    完成 Agent Run。

    Args:
        state: 当前状态
        run_manager: Agent Run Manager

    Returns:
        状态更新
    """
    run_id = state["run_id"]
    answer = state.get("answer")
    errors = state.get("errors", [])

    logger.info("finalize_node", extra={"run_id": run_id})

    if errors:
        # 有错误，标记为失败
        await run_manager.fail_run(run_id, "; ".join(errors))
    elif answer:
        # 有答案，标记为完成
        await run_manager.complete_run(run_id, answer)
    else:
        # 无答案，标记为失败
        await run_manager.fail_run(run_id, "无法生成答案")

    return {}


def _classify_intent(question: str) -> str:
    """
    分类用户意图。

    V1 简单规则：
    - 包含"创建"、"申请"、"提交" -> tool_request
    - 包含"什么是"、"如何"、"怎么" -> hr_query
    - 其他 -> hr_query

    Args:
        question: 用户问题

    Returns:
        意图类型
    """
    # 工具请求关键词。避免把“需要提交哪些材料”这类制度问答误判为写入请求。
    tool_keywords = ["创建", "工单", "ticket", "帮我办理", "帮我申请", "我要申请"]
    if any(kw in question for kw in tool_keywords):
        return "tool_request"

    # 默认为 HR 查询
    return "hr_query"


def _build_tool_parameters(tool_name: str, question: str) -> dict[str, Any]:
    """根据工具名称和用户问题生成确定性的工具参数。"""
    if tool_name == "create_hr_ticket":
        return {
            "title": "新员工入职工单",
            "description": question,
            "priority": "medium",
            "category": "入职",
        }
    return {"query": question}


def _build_interrupt_payload(
    run_id: str,
    approval_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    evidence: Any,
) -> dict[str, Any]:
    """构造 JSON-serializable 的审批中断 payload。"""
    evidence_summary: list[dict[str, Any]] = []
    if isinstance(evidence, dict):
        for item in evidence.get("evidence_list", [])[:3]:
            if isinstance(item, dict):
                evidence_summary.append(
                    {
                        "document_name": item.get("document_name", ""),
                        "section": item.get("section", ""),
                        "page": item.get("page", 0),
                        "score": item.get("score", 0.0),
                    }
                )

    return {
        "run_id": run_id,
        "approval_id": approval_id,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "risk_level": "write",
        "evidence_summary": evidence_summary,
        "allowed_decisions": [
            ApprovalDecisionType.APPROVE.value,
            ApprovalDecisionType.EDIT.value,
            ApprovalDecisionType.REJECT.value,
        ],
    }
