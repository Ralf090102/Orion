"""
Tests for Orion's query processing functions.
"""

from app.query import format_context, create_prompt


class TestFormatContext:
    """Tests for format_context()"""

    def test_empty_documents_returns_empty_string(self):
        """Should return empty string when no documents are passed."""
        context, sources = format_context([])
        assert context == ""
        assert sources == []

    def test_formats_multiple_documents(self, fake_docs):
        """Should concatenate document contents."""
        result, sources = format_context(fake_docs)
        assert all(doc.page_content in result for doc in fake_docs)
        assert all(doc.metadata["source"] in str(sources) for doc in fake_docs)


class TestCreatePrompt:
    """Tests for create_prompt()"""

    def test_with_context_includes_both(self):
        """Should include query and context in the prompt."""
        query = "What is the main topic?"
        context = "This is about machine learning."
        prompt = create_prompt(query, context)
        assert query in prompt
        assert context in prompt
        assert "context" in prompt.lower()

    def test_empty_context_includes_no_info_message(self):
        """Should include fallback message when no context exists."""
        query = "What is the main topic?"
        context = ""
        prompt = create_prompt(query, context)

        fallback_indicators = [
            "no relevant context",
            "no relevant information",
            "don't have any relevant information",
            "knowledge base did not contain",
            "no data available",
        ]

        assert any(indicator in prompt for indicator in fallback_indicators)
        assert query in prompt
