"""
Tests for Orion's query processing functions.
"""

from app.query import format_context, create_prompt


class TestFormatContext:
    """Tests for format_context()"""

    def test_empty_documents_returns_empty_string(self):
        """Should return empty string when no documents are passed."""
        assert format_context([])[0] == ""

    def test_formats_multiple_documents(self, fake_docs):
        """Should concatenate document contents."""
        result = format_context(fake_docs)
        assert "Content A" in result
        assert "Content B" in result


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
        assert "don't have any relevant information" in prompt
