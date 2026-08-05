"""Regression test for POST /api/ask/stream.

Before 2026-08-05 this endpoint always threw AttributeError -- it called
`context_preparer.prepare_context(...)` (a method that never existed;
the real one is `prepare()`), passed raw SearchResult objects where a
list of dicts was expected, and called `prompt_builder.build_rag_prompt()`
with a `context=` kwarg when the real signature takes `contexts=`. The
generic except-Exception handler swallowed all of that into an SSE
`error` chunk, so it never surfaced as a 500 -- this test exists because
nothing else would have caught it.
"""

import pytest

from backend.dependencies import get_config_dependency, get_generator_dependency
from src.generation.context_preparer import ContextPreparer
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.search import SearchResult


class _FakeStreamRetriever:
    def query(self, query_text, k, formatted):
        return [
            SearchResult(
                document_id="doc-1",
                content="Machine learning is a subset of AI.",
                metadata={"source_file": "ml.md"},
                score=0.8,
                search_type="hybrid",
            )
        ]


class _FakeLLMClient:
    def generate(self, messages, model, temperature, top_p, max_tokens, stream, on_token):
        assert messages[0]["role"] == "system"
        for token in ("Hello", " world."):
            on_token(token)
        return {"message": {"content": "Hello world."}}


class _FakeStreamGenerator:
    """Satisfies the attributes ask_stream() touches directly (retriever,
    context_preparer, prompt_builder, llm_client) -- it calls into those,
    not generator.generate_rag_response()."""

    def __init__(self, config):
        self.retriever = _FakeStreamRetriever()
        self.context_preparer = ContextPreparer()
        self.prompt_builder = PromptBuilder(config)
        self.llm_client = _FakeLLMClient()


@pytest.fixture
def stream_client(client, test_config):
    """Reuses the shared `client` fixture's app/overrides, swapping only
    the generator override for one with the attributes this endpoint needs."""
    from backend.app import app

    app.dependency_overrides[get_generator_dependency] = lambda: _FakeStreamGenerator(test_config)
    app.dependency_overrides[get_config_dependency] = lambda: test_config
    return client


@pytest.mark.unit
def test_ask_stream_does_not_error(stream_client):
    response = stream_client.post("/api/ask/stream", json={"query": "What is ML?", "k": 1})

    assert response.status_code == 200
    body = response.text

    assert '"type":"error"' not in body.replace(" ", "")
    assert "AttributeError" not in body
    assert '"type":"done"' in body.replace(" ", "")


@pytest.mark.unit
def test_ask_stream_sources_use_real_source_file_not_generic_key(stream_client):
    """r.metadata.get("source", ...) used to read the wrong key (should be
    "source_file") -- always fell back to "Unknown"."""
    response = stream_client.post(
        "/api/ask/stream", json={"query": "What is ML?", "k": 1, "include_sources": True}
    )

    assert '"citation":"ml.md"' in response.text.replace(" ", "")


@pytest.mark.unit
def test_ask_stream_yields_tokens_from_the_llm(stream_client):
    response = stream_client.post("/api/ask/stream", json={"query": "What is ML?", "k": 1})

    body = response.text.replace(" ", "")
    assert '"type":"token","content":"Hello"' in body
