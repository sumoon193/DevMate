"""devmate GitHub CI failure webhook 适配路由。

把 GitHub CI failure webhook payload 适配为 DM05Input（webhook 摄取），
创建 DevMate Case，并推进到 running 状态，触发诊断→补丁→沙箱→审批→PR 链路。

本路由只做协议适配和领域对象转换，不承载领域决策；所有状态推进经
CaseCommand/IngestionStore 等 typed 入口。不接真实 GitHub，webhook
payload 由调用方提供。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Header, HTTPException, Response
from pydantic import BaseModel, Field

from app.devmate.cases import (
    CaseCommand,
    CaseStatus,
    CaseStore,
    DM04Input,
    IllegalTransitionError,
)
from app.devmate.ingestion import (
    CIEvidence,
    CommitEvidence,
    DM05Input,
    IngestionStore,
    InvalidWebhookError,
)
from app.devmate.ingestion import (
    RuntimeEvent as IngestionRuntimeEvent,
)


class GitHubWebhookPayload(BaseModel):
    """GitHub CI failure webhook 的最小契约字段。"""

    webhook_id: str = Field(description="去重 key，GitHub delivery GUID 或等价幂等键")
    repo: str
    branch: str
    head_sha: str
    workflow_run_id: str
    ci_status: str = "failed"
    ci_url: str | None = None
    failed_job: str | None = None
    raw_log: str | None = Field(default=None, description="失败 job 的日志片段")
    test_report: str | None = Field(default=None, description="测试报告片段")


class WebhookAcceptedResponse(BaseModel):
    webhook_id: str
    evidence_id: str
    case_id: str
    duplicate: bool
    status: str


def create_webhook_router(
    *,
    case_store: CaseStore,
    ingestion_store: IngestionStore,
) -> APIRouter:
    router = APIRouter(prefix="/webhooks", tags=["devmate-ingestion"])

    @router.post("/github", response_model=WebhookAcceptedResponse, status_code=202)
    def receive_github_ci_failure(
        body: GitHubWebhookPayload,
        response: Response,
        x_request_id: str | None = Header(default=None),
    ) -> WebhookAcceptedResponse:
        request_id = x_request_id or f"req-{uuid.uuid4().hex[:12]}"
        response.headers["X-Request-ID"] = request_id

        ingestion = IngestionRuntimeEvent(store=ingestion_store)
        try:
            ev_result = ingestion.execute(
                DM05Input(
                    webhook_id=body.webhook_id,
                    source="github",
                    event_type="ci_failure",
                    payload={
                        "repo": body.repo,
                        "branch": body.branch,
                        "failed_job": body.failed_job,
                        "raw_log": body.raw_log,
                        "test_report": body.test_report,
                    },
                    commit=CommitEvidence(
                        commit_sha=body.head_sha,
                        branch=body.branch,
                        repo=body.repo,
                    ),
                    ci=CIEvidence(
                        ci_run_id=body.workflow_run_id,
                        ci_status=body.ci_status,
                        ci_url=body.ci_url,
                    ),
                )
            )
        except InvalidWebhookError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_webhook", "reason": str(exc), "request_id": request_id},
            ) from exc

        # 幂等：重复 webhook 不重复创建 case，返回原 case 状态。
        case_id = f"case-{ev_result.evidence_id}"
        if not ev_result.duplicate:
            try:
                case_store.create(
                    case_id=case_id,
                    actor_id="agent:devmate-webhook",
                    payload={
                        "repo": body.repo,
                        "branch": body.branch,
                        "head_sha": body.head_sha,
                        "evidence_id": ev_result.evidence_id,
                    },
                )
                command = CaseCommand(case_store)
                command.execute(
                    DM04Input(
                        case_id=case_id,
                        command_id=f"cmd-{body.webhook_id}-start",
                        event_type="case.start",
                        actor_id="agent:devmate-webhook",
                        target_status=CaseStatus.RUNNING,
                        payload={"evidence_id": ev_result.evidence_id},
                    )
                )
            except IllegalTransitionError as exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "illegal_transition",
                        "request_id": request_id,
                    },
                ) from exc

        record = case_store.get(case_id)
        return WebhookAcceptedResponse(
            webhook_id=ev_result.webhook_id,
            evidence_id=ev_result.evidence_id,
            case_id=case_id,
            duplicate=ev_result.duplicate,
            status=record.status.value if record else "created",
        )

    return router
