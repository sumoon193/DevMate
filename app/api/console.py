"""DevMate 控制台使用的真实读模型端点。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field

from app.api.dependencies import ServiceContainer, get_container
from app.config import get_settings
from app.core.health import check_infrastructure_health
from app.schemas.memory import EpisodicMemoryRecord

router = APIRouter(tags=["console"])


class MemoryPage(BaseModel):
    """长期记忆控制台分页。"""

    items: list[EpisodicMemoryRecord] = Field(default_factory=list)
    total: int = Field(ge=0)


class InfrastructureService(BaseModel):
    """单项基础设施探测结果。"""

    name: str
    status: str
    latency_ms: float | None = None
    error: str | None = None


class InfrastructureResponse(BaseModel):
    """基础设施状态，不将跳过的探测标记为成功。"""

    mode: str
    acceptance: str
    services: list[InfrastructureService]


@router.get("/memories", response_model=MemoryPage)
async def list_memories(
    tenant_id: str = Header(alias="X-Tenant-ID"),
    container: ServiceContainer = Depends(get_container),
) -> MemoryPage:
    settings = get_settings()
    if settings.app_mode == "full":
        from app.services.security.oidc import current_tenant_id

        tenant_id = current_tenant_id()
    records = await container.memory_store.list_records(tenant_id=tenant_id)
    return MemoryPage(items=records, total=len(records))


@router.delete("/memories/{memory_id}", response_model=EpisodicMemoryRecord)
async def delete_memory(
    memory_id: str,
    tenant_id: str = Header(alias="X-Tenant-ID"),
    container: ServiceContainer = Depends(get_container),
) -> EpisodicMemoryRecord:
    settings = get_settings()
    if settings.app_mode == "full":
        from app.services.security.oidc import current_tenant_id

        tenant_id = current_tenant_id()
    return await container.memory_store.forget(memory_id, tenant_id=tenant_id)


@router.get("/infrastructure", response_model=InfrastructureResponse)
async def infrastructure_status() -> InfrastructureResponse:
    settings = get_settings()
    probes: dict[str, dict[str, Any]] = check_infrastructure_health(settings)
    services = [
        InfrastructureService(
            name=name,
            status=str(result.get("status", "unknown")),
            latency_ms=result.get("latency_ms"),
            error=result.get("error"),
        )
        for name, result in probes.items()
    ]
    return InfrastructureResponse(
        mode=settings.app_mode,
        acceptance=("live-probed" if settings.app_mode == "full" else "offline-only"),
        services=services,
    )
