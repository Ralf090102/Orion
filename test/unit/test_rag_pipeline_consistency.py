"""Regression tests for the retrieval -> context -> prompt handoff.

Guards against the 2026-08-05 bug where SearchResult.to_dict()'s shape
(metadata nested under "metadata", score keyed "score") silently didn't
match what ContextPreparer._clean_context() and PromptBuilder.build_rag_
prompt()/build_chat_prompt() expected -- every real query lost its
relevance score and all citation metadata (source_file/page/title), and
the LLM was told every retrieved chunk came from "[From: document]"/
"Unknown" regardless of which file it actually came from. None of this
surfaced as an exception anywhere; it silently degraded to empty/generic
values, which is exactly why no test caught it the first time.

These tests exercise the real objects (SearchResult, ContextPreparer,
PromptBuilder) together rather than mocking the boundary between them --
the whole point is verifying the boundary itself.

Updated 2026-09-03: ContextPreparer.prepare() now takes SearchResult
objects directly and returns PreparedContext (a real dataclass, not an
untyped dict) -- confirmed via full-repo grep that no real caller ever
used the old dual-shape (SearchResult.to_dict() vs. an "already-prepared
flat dict") input tolerance, so there's no flat-dict-reentry case left to
test; see Architecture-Review.md.
"""

import pytest

from src.generation.context_preparer import ContextPreparer
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.search import SearchResult
from src.utilities.config import OrionConfig


@pytest.fixture
def markdown_result():
    return SearchResult(
        document_id="doc-1",
        content="Machine learning is a subset of AI.",
        metadata={"source_file": "data/knowledge_base/machine_learning_basics.md", "page": None},
        score=0.87,
        search_type="hybrid_reranked_mmr",
    )


@pytest.fixture
def pdf_result():
    return SearchResult(
        document_id="doc-2",
        content="Dry lab procedures require careful documentation.",
        metadata={"source_file": "data/knowledge_base/drylab.pdf", "page": 3},
        score=0.91,
        search_type="hybrid_reranked_mmr",
    )


@pytest.fixture
def prepared_contexts(markdown_result, pdf_result):
    """The exact call generate.py makes: SearchResults straight into
    ContextPreparer.prepare(), no manual dict conversion."""
    preparer = ContextPreparer()
    return preparer.prepare([markdown_result, pdf_result], sort_by_score=True)


@pytest.mark.unit
class TestSearchResultToContextPreparerHandoff:
    def test_real_relevance_score_survives_as_final_score(self, prepared_contexts):
        scores = {c.source_file: c.final_score for c in prepared_contexts}
        assert scores["data/knowledge_base/machine_learning_basics.md"] == 0.87
        assert scores["data/knowledge_base/drylab.pdf"] == 0.91

    def test_source_file_survives_from_metadata(self, prepared_contexts):
        source_files = {c.source_file for c in prepared_contexts}
        assert "data/knowledge_base/machine_learning_basics.md" in source_files
        assert "data/knowledge_base/drylab.pdf" in source_files

    def test_non_pdf_source_gets_a_real_citation_text_not_none(self, prepared_contexts):
        md_ctx = next(c for c in prepared_contexts if c.source_file.endswith(".md"))
        assert md_ctx.citation_text == "Machine Learning Basics"

    def test_pdf_source_gets_title_and_page_citation(self, prepared_contexts):
        pdf_ctx = next(c for c in prepared_contexts if c.source_file.endswith(".pdf"))
        assert pdf_ctx.citation_text == "Drylab, p. 3"


@pytest.mark.unit
class TestPromptBuilderReadsPreparedContextShape:
    def test_rag_prompt_citation_source_is_not_unknown(self, prepared_contexts):
        builder = PromptBuilder(OrionConfig())
        components = builder.build_rag_prompt(query="What is ML?", contexts=prepared_contexts)

        assert len(components.citations) == 2
        for citation in components.citations:
            assert citation["source"] != "Unknown"
            assert citation["file_name"] != "Unknown"

    def test_chat_prompt_source_label_is_not_generic_document(self, prepared_contexts):
        builder = PromptBuilder(OrionConfig())
        components = builder.build_chat_prompt(query="What is ML?", contexts=prepared_contexts)

        assert "[From: document]" not in components.context
        assert "[From: Machine Learning Basics]" in components.context
        assert "[From: Drylab, p. 3]" in components.context

    def test_chat_prompt_with_no_contexts_still_omits_rag_framing(self):
        builder = PromptBuilder(OrionConfig())
        components = builder.build_chat_prompt(query="Hello", contexts=None)

        assert components.context == ""
