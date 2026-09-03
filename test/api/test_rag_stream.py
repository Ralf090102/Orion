"""Regression test for POST /api/ask/stream.

Before 2026-08-05 this endpoint always threw AttributeError -- it called
`context_preparer.prepare_context(...)` (a method that never existed;
the real one is `prepare()`), passed raw SearchResult objects where a
list of dicts was expected, and called `prompt_builder.build_rag_prompt()`
with a `context=` kwarg when the real signature takes `contexts=`. The
generic except-Exception handler swallowed all of that into an SSE
`error` chunk, so it never surfaced as a 500 -- this test exists because
nothing else would have caught it.

Before 2026-09-03 the fix for that bug was itself shallow: ask_stream()
hand-rolled retrieve/prepare/prompt/generate directly against
generator.retriever/context_preparer/prompt_builder/llm_client instead of
calling generator.generate_rag_response() -- and its LLM call was neither
streamed nor offloaded, so it blocked the event loop and delivered every
token in one post-generation burst instead of in real time.
_FakeStreamGenerator below implements only generate_rag_response() (the
real AnswerGenerator interface ask_stream() is supposed to call through),
not those four internals -- if ask_stream() ever regresses to reaching
past that interface again, these tests fail with an AttributeError
instead of silently duplicating the pipeline a second time.
"""

import pytest

from backend.dependencies import get_config_dependency, get_generator_dependency
from src.generation.generate import GenerationResult


class _FakeStreamGenerator:
    """Implements only generate_rag_response() -- deliberately has no
    retriever/context_preparer/prompt_builder/llm_client attributes, so a
    regression to hand-rolling the pipeline in ask_stream() fails loudly."""

    def __init__(self, config):
        self.config = config

    def generate_rag_response(
        self,
        query,
        k=None,
        include_sources=True,
        temperature=None,
        max_tokens=None,
        stream=False,
        on_token=None,
        on_sources=None,
    ):
        assert stream is True, "ask_stream() must request streaming"
        assert on_token is not None, "ask_stream() must pass a token callback"

        if include_sources:
            assert on_sources is not None, (
                "ask_stream() must pass a sources callback when include_sources is True"
            )
            sources = [
                {
                    "index": 1,
                    "text": "Machine learning is a subset of AI...",
                    "score": 0.8,
                    "citation": "ml.md",
                    "source_file": "ml.md",
                }
            ]
            on_sources(sources)
        else:
            sources = []

        for token in ("Hello", " world."):
            on_token(token)

        return GenerationResult(
            answer="Hello world.",
            sources=sources,
            query_type="factual",
            mode="rag",
            metadata={},
            rag_triggered=True,
        )


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
def test_ask_stream_sources_arrive_before_tokens(stream_client):
    """Sources are formatted and handed to on_sources() before generation
    starts (see generate_rag_response), and ask_stream() forwards them to
    the queue in that order -- the SSE body should reflect it."""
    response = stream_client.post(
        "/api/ask/stream", json={"query": "What is ML?", "k": 1, "include_sources": True}
    )

    body = response.text
    sources_idx = body.index('"type":"sources"')
    token_idx = body.index('"type":"token"')
    assert sources_idx < token_idx


@pytest.mark.unit
def test_ask_stream_yields_tokens_from_the_llm(stream_client):
    response = stream_client.post("/api/ask/stream", json={"query": "What is ML?", "k": 1})

    body = response.text.replace(" ", "")
    assert '"type":"token","content":"Hello"' in body


@pytest.mark.unit
def test_ask_stream_forwards_graceful_failure_answer_as_a_token(stream_client, test_config):
    """generate_rag_response() catches retrieval/LLM failures internally
    and returns a GenerationResult with an apologetic .answer instead of
    raising -- so on_token never fires for that case. ask_stream() must
    forward that answer as a token itself, mirroring how
    ChatWebSocketHandler already does for its own generation failures."""
    from backend.app import app

    class _FailingGenerator:
        def __init__(self, config):
            self.config = config

        def generate_rag_response(self, query, **kwargs):
            return GenerationResult(
                answer="I encountered an error while generating the response: boom",
                sources=[],
                query_type="factual",
                mode="rag",
                metadata={"error": "boom", "llm_generation_failed": True},
                rag_triggered=True,
            )

    app.dependency_overrides[get_generator_dependency] = lambda: _FailingGenerator(test_config)
    app.dependency_overrides[get_config_dependency] = lambda: test_config

    response = stream_client.post("/api/ask/stream", json={"query": "What is ML?", "k": 1})

    body = response.text
    assert '"type":"token"' in body.replace(" ", "")
    assert "I encountered an error while generating the response: boom" in body
    assert '"type":"error"' not in body.replace(" ", "")
