"""
Pytest configuration and shared fixtures for Orion tests.
"""

import pytest
from unittest.mock import patch
from langchain.schema import Document


class Doc:
    def __init__(self, content, meta):
        self.page_content = content
        self.metadata = meta


@pytest.fixture
def mock_ollama_list_success():
    """
    Fixture: Mocks ollama.list() to return a list of models.
    """
    with patch("app.llm.ollama.list") as mock_list:
        mock_list.return_value = {
            "models": [
                {"name": "mistral:latest"},
                {"name": "llama2:latest"},
            ]
        }
        # Clear the LRU cache for model_exists to prevent cached results
        from app.llm import model_exists

        model_exists.cache_clear()
        yield mock_list


@pytest.fixture
def mock_ollama_list_failure():
    """
    Fixture: Mocks ollama.list() and ollama.show() to raise exceptions.
    """
    with (
        patch("app.llm.ollama.list", side_effect=Exception("Ollama down")) as mock_list,
        patch("app.llm.ollama.show", side_effect=Exception("Ollama down")),
    ):
        # Clear the LRU cache for model_exists to prevent cached results
        from app.llm import model_exists

        model_exists.cache_clear()
        yield mock_list


@pytest.fixture
def fake_docs():
    """
    Fixture: Returns a list of fake document-like objects
    with a 'page_content' attribute.
    """
    return [
        Document(
            page_content="Document 1 content",
            metadata={"source": "file1.txt", "page": 1},
        ),
        Document(
            page_content="Document 2 content",
            metadata={"source": "file2.txt", "page": 2},
        ),
    ]
