"""devmate Case HTTP API：只做协议适配与权限校验。

状态推进全部经 CaseCommand，控制器不含领域状态机逻辑；提供 typed
request/response、稳定错误码、request/correlation ID。
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Response
from pydantic import BaseModel, Field

from app.devmate.cases import (
    CaseCommand,
    CaseNotFoundError,
    CaseStatus,
    CaseStore,
    DM04Input,
    DuplicateCaseError,
    IllegalTransitionError,
)


class CreateCaseRequest(BaseModel):
    case_id: str
    actor_id: str | None = None
    command_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class CreateCaseResponse(BaseModel):
    case_id: str
    status: str
    command_id: str


class CommandRequest(BaseModel):
    command_id: str
    event_type: str
    target_status: CaseStatus
    actor_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class CommandResponse(BaseModel):
    case_id: str
    status: str
    command_id: str
    state_event: str
    audit_info: dict[str, str]


class TimelineEventBody(BaseModel):
    event_id: str
    command_id: str
    event_type: str
    from_status: str
    to_status: str
    actor_id: str
    created_at: str


class TimelineResponse(BaseModel):
    case_id: str
    events: list[TimelineEventBody]


def create_devmate_router(store: CaseStore) -> APIRouter:
    router = APIRouter(prefix="/devmate", tags=["devmate"])

    @router.post("/cases", response_model=CreateCaseResponse, status_code=201)
    def create_case(
        body: CreateCaseRequest,
        response: Response,
        x_request_id: str | None = Header(default=None),
    ) -> CreateCaseResponse:
        request_id = _request_id(x_request_id, response)
        _require_actor(body.actor_id, request_id)
        try:
            record = store.create(
                case_id=body.case_id,
                actor_id=body.actor_id,
                payload=body.payload,
            )
        except DuplicateCaseError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": "case_already_exists", "request_id": request_id},
            ) from exc
        return CreateCaseResponse(
            case_id=record.case_id,
            status=record.status.value,
            command_id=body.command_id,
        )

    @router.post("/cases/{case_id}/commands", response_model=CommandResponse)
    def run_command(
        case_id: str,
        body: CommandRequest,
        response: Response,
        x_request_id: str | None = Header(default=None),
    ) -> CommandResponse:
        request_id = _request_id(x_request_id, response)
        _require_actor(body.actor_id, request_id)
        command = CaseCommand(store)
        try:
            result = command.execute(
                DM04Input(
                    case_id=case_id,
                    command_id=body.command_id,
                    event_type=body.event_type,
                    actor_id=body.actor_id,
                    target_status=body.target_status,
                    payload=body.payload,
                )
            )
        except CaseNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail={"error": "case_not_found", "request_id": request_id},
            ) from exc
        except IllegalTransitionError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": "illegal_transition", "request_id": request_id},
            ) from exc
        return CommandResponse(
            case_id=result.case_id,
            status=result.status.value,
            command_id=result.command_id,
            state_event=result.state_event,
            audit_info=result.audit_info,
        )

    @router.get("/cases/{case_id}/timeline", response_model=TimelineResponse)
    def get_timeline(
        case_id: str,
        response: Response,
        x_request_id: str | None = Header(default=None),
    ) -> TimelineResponse:
        _request_id(x_request_id, response)
        return TimelineResponse(
            case_id=case_id,
            events=[
                TimelineEventBody(
                    event_id=event.event_id,
                    command_id=event.command_id,
                    event_type=event.event_type,
                    from_status=event.from_status.value,
                    to_status=event.to_status.value,
                    actor_id=event.actor_id,
                    created_at=event.created_at,
                )
                for event in store.timeline(case_id)
            ],
        )

    return router


def _require_actor(actor_id: str, request_id: str) -> None:
    if not actor_id or not actor_id.strip():
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "request_id": request_id},
        )


def _request_id(x_request_id: str | None, response: Response) -> str:
    request_id = x_request_id or f"req-{uuid.uuid4().hex[:12]}"
    response.headers["X-Request-ID"] = request_id
    return request_id
