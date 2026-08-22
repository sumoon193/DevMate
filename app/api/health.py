"""
健康检查端点。
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.config import get_settings
from app.core.health import check_infrastructure_health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        timestamp=datetime.now(UTC),
        mode=settings.app_mode,
        services=check_infrastructure_health(settings),
    )
