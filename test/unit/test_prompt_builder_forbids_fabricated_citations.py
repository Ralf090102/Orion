"""Regression test: both RAG-context prompts must explicitly forbid the LLM
from inventing citations/sources not present in the provided context.

Found live (2026-09-11) reproducing a real symptom: the exact same query,
with the exact same real retrieved context, sometimes produced a
well-grounded answer citing the real "[From: ...]" label, and sometimes
fabricated a fake academic references list ("Lichtman, S. W., & Walls,
P. W. (1992)...") that traced to nothing in the knowledge base. Root cause:
build_chat_prompt()'s RAG-augmented branch had zero faithfulness
instructions at all (only tone/style guidance), and build_rag_prompt()'s
citation instructions didn't explicitly forbid adding sources beyond the
numbered context either. Both now explicitly instruct the model never to
invent citations/references/sources not in the provided context.

This only locks in the prompt *instructions* -- it can't guarantee the LLM
always complies (that's inherently probabilistic for a local model), but 3
live reruns of the reproduction query after this fix were all correctly
grounded with zero fabrication, down from a real fabricated-references
failure before the fix. See Eru's Sessions/8-RAG-Groundedness-Diagnosis-
2026-09-11-Recap.md for the full narrative.
"""

import pytest

from src.generation.context_preparer import PreparedContext
from src.generation.prompt_builder import PromptBuilder
from src.utilities.config import OrionConfig


def _context(text: str = "Some real knowledge base content.") -> PreparedContext:
    return PreparedContext(
        text=text,
        final_score=0.9,
        length=len(text),
        source_file="doc.pdf",
        normalized_source_file="doc.pdf",
        source_type="pdf",
        title="Doc",
        citation_text="Doc, p. 1",
    )


@pytest.fixture
def builder():
    return PromptBuilder(OrionConfig())


@pytest.mark.unit
class TestPromptsForbidFabricatedCitations:
    def test_build_chat_prompt_with_rag_forbids_invented_sources(self, builder):
        components = builder.build_chat_prompt(query="What is X?", contexts=[_context()])

        assert "never invent" in components.system_prompt.lower()
        assert "[from:" in components.system_prompt.lower() or "[From:" in components.system_prompt

    def test_build_rag_prompt_with_citations_forbids_invented_sources(self, builder):
        components = builder.build_rag_prompt(query="What is X?", contexts=[_context()], include_citations=True)

        assert "never invent" in components.system_prompt.lower()

    def test_build_rag_prompt_without_citations_still_forbids_invented_sources(self, builder):
        components = builder.build_rag_prompt(query="What is X?", contexts=[_context()], include_citations=False)

        assert "never invent" in components.system_prompt.lower()
