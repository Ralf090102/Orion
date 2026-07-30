"""Shared pytest fixtures for the Orion backend test suite.

The FastAPI app's lifespan (`initialize_resources`) loads real ML models
(embeddings, reranker, LLM client) and expects Ollama/ChromaDB to be
reachable. Unit/integration tests here never want that -- they exercise
routing, request/response shape, and error handling, not the ML stack.
So `client` patches the lifespan to a no-op and swaps the heavy
Depends() singletons for lightweight fakes via `app.dependency_overrides`.
"""

import pytest
from fastapi.testclient import TestClient

from backend.dependencies import (
    get_config_dependency,
    get_generator_dependency,
    get_retriever_dependency,
    get_session_manager_dependency,
)
from src.utilities.config import OrionConfig


class FakeVectorStore:
    def get_collection_stats(self):
        return {
            "total_chunks": 0,
            "unique_files": 0,
            "collection_name": "test-collection",
            "persist_directory": "test",
            "file_type_distribution": {},
        }


class FakeRetriever:
    """Stand-in for OrionRetriever with no ChromaDB/embedding-model dependency."""

    def __init__(self):
        self.vector_store = FakeVectorStore()


class FakeSessionManager:
    """Stand-in for SessionManager with no SQLite dependency."""

    def get_database_stats(self):
        return {"total_sessions": 0, "total_messages": 0, "total_tokens": 0, "db_size_bytes": 0}


class FakeGenerator:
    """Stand-in for AnswerGenerator; individual tests can monkeypatch its methods."""

    def generate_chat_response(self, *args, **kwargs):
        raise NotImplementedError("override generate_chat_response in the test that needs it")


@pytest.fixture
def test_config() -> OrionConfig:
    """Default configuration, built from dataclass defaults (no env/file I/O)."""
    return OrionConfig()


@pytest.fixture(autouse=True)
def _reset_metrics():
    """metrics_collector is a process-wide singleton; reset it around every
    test so request counts from one test don't leak into the next."""
    from backend.metrics import metrics_collector

    metrics_collector.reset()
    yield
    metrics_collector.reset()


@pytest.fixture
def client(monkeypatch, test_config):
    """TestClient with the ML lifespan disabled and heavy deps faked out."""
    monkeypatch.setattr("backend.app.initialize_resources", lambda: None)
    monkeypatch.setattr("backend.app.cleanup_resources", lambda: None)

    from backend.app import app

    app.dependency_overrides[get_config_dependency] = lambda: test_config
    app.dependency_overrides[get_retriever_dependency] = lambda: FakeRetriever()
    app.dependency_overrides[get_session_manager_dependency] = lambda: FakeSessionManager()
    app.dependency_overrides[get_generator_dependency] = lambda: FakeGenerator()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
