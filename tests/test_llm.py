"""
Tests for Orion's LLM interface functions.
"""


class TestCheckModelAvailability:
    """Tests for check_model_availability()"""

    def test_model_found_returns_true(self, mock_ollama_list_success):
        """Should return True if model exists in ollama list."""
        from core.rag.llm import check_model_availability

        assert check_model_availability("mistral") is True
        assert check_model_availability("llama2") is True

    def test_model_not_found_returns_false(self, mock_ollama_list_success):
        """Should return False if model not in ollama list."""
        from core.rag.llm import check_model_availability

        assert check_model_availability("unknown") is False

    def test_ollama_error_returns_false(self, mock_ollama_list_failure):
        """Should return False if ollama.list() raises an exception."""
        from core.rag.llm import check_model_availability

        assert check_model_availability("mistral") is False
