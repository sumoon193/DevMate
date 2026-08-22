"""DevMate 控制台读模型 API。"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_container, reset_container
from app.main import create_app


@pytest.fixture()
def client() -> TestClient:
    reset_container()
    with TestClient(create_app()) as test_client:
        yield test_client
    reset_container()


def test_memory_read_model_is_tenant_scoped_and_supports_deletion(client: TestClient) -> None:
    store = get_container().memory_store

    async def seed() -> None:
        await store.remember(
            tenant_id="tenant-a",
            case_id="case-a",
            memory_key="deploy-recovery",
            content="恢复任务前先检查 checkpoint 和副作用账本。",
            provenance_event_ids=["event-a"],
            ttl_seconds=3600,
            importance_score=0.9,
        )
        await store.remember(
            tenant_id="tenant-b",
            case_id="case-b",
            memory_key="private-memory",
            content="另一个租户的私有记忆。",
            provenance_event_ids=["event-b"],
        )

    asyncio.run(seed())

    response = client.get("/memories", headers={"X-Tenant-ID": "tenant-a"})
    assert response.status_code == 200
    assert response.json()["total"] == 1
    memory = response.json()["items"][0]
    assert memory["memory_key"] == "deploy-recovery"
    assert memory["importance_score"] == 0.9
    assert memory["status"] == "active"
    assert "私有记忆" not in response.text

    deleted = client.delete(
        f"/memories/{memory['id']}",
        headers={"X-Tenant-ID": "tenant-a"},
    )
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"

    after_delete = client.get("/memories", headers={"X-Tenant-ID": "tenant-a"})
    assert after_delete.status_code == 200
    assert after_delete.json()["items"][0]["status"] == "deleted"


def test_memory_read_model_requires_tenant_identity(client: TestClient) -> None:
    response = client.get("/memories")
    assert response.status_code == 422


def test_infrastructure_read_model_never_reports_unprobed_services_as_up(
    client: TestClient,
) -> None:
    response = client.get("/infrastructure")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "fallback"
    assert data["acceptance"] == "offline-only"
    assert data["services"]
    assert all(service["status"] != "up" for service in data["services"])
