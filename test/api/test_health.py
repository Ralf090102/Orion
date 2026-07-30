"""Tests for backend/api/health.py -- the health/status/config/metrics endpoints."""

import pytest


@pytest.mark.unit
def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "orion-backend-api"


@pytest.mark.unit
def test_status_reports_degraded_when_ollama_unreachable(client, monkeypatch):
    monkeypatch.setattr("backend.api.health.check_ollama_connection", lambda: False)

    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["ollama_available"] is False
    assert body["knowledge_base"]["total_chunks"] == 0


@pytest.mark.unit
def test_status_reports_healthy_when_ollama_reachable(client, monkeypatch):
    monkeypatch.setattr("backend.api.health.check_ollama_connection", lambda: True)

    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["ollama_available"] is True


@pytest.mark.unit
def test_status_survives_broken_vector_store(client, monkeypatch):
    """If the retriever's vector store throws, /api/status should degrade, not 500."""

    class BrokenVectorStore:
        def get_collection_stats(self):
            raise RuntimeError("ChromaDB unreachable")

    class BrokenRetriever:
        vector_store = BrokenVectorStore()

    from backend.app import app
    from backend.dependencies import get_retriever_dependency

    app.dependency_overrides[get_retriever_dependency] = lambda: BrokenRetriever()

    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["knowledge_base"]["total_chunks"] == 0


@pytest.mark.unit
def test_get_config_summary(client):
    response = client.get("/api/config")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "embedding" in body["config"]
    assert "llm" in body["config"]


@pytest.mark.unit
def test_get_config_detailed(client):
    response = client.get("/api/config?detailed=true")

    assert response.status_code == 200
    body = response.json()
    assert "rag" in body["config"]
    assert "system" in body["config"]


@pytest.mark.unit
def test_get_formats(client):
    response = client.get("/api/formats")

    assert response.status_code == 200
    body = response.json()
    assert body["total_formats"] > 0
    assert ".pdf" in body["all_formats"]


@pytest.mark.unit
def test_get_db_stats(client):
    response = client.get("/api/db/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "total_sessions" in body["stats"]


@pytest.mark.unit
def test_readiness_ready_when_all_checks_pass(client, monkeypatch):
    monkeypatch.setattr("backend.api.health.check_ollama_connection", lambda: True)

    response = client.get("/api/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["checks"]["ollama"] == "ready"


@pytest.mark.unit
def test_readiness_not_ready_when_ollama_required_but_down(client, monkeypatch):
    monkeypatch.setattr("backend.api.health.check_ollama_connection", lambda: False)
    from backend.app import app
    from backend.dependencies import get_config_dependency
    from src.utilities.config import OrionConfig

    strict_config = OrionConfig()
    strict_config.system.require_ollama = True
    app.dependency_overrides[get_config_dependency] = lambda: strict_config

    response = client.get("/api/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert body["status"] == "not_ready"
