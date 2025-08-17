"""
Tests for Orion's enhanced query processing functions.
"""
import pytest
from unittest.mock import Mock, patch
from app.query import format_context, create_prompt, query_knowledgebase
from app.query_enhancement import QueryEnhancer


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


class TestQueryEnhancement:
    """Tests for enhanced query processing"""

    @patch('app.query_enhancement.ChatOllama')
    def test_query_expansion(self, mock_llm):
        """Should expand query into multiple variations"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "1. What is machine learning?\n2. How does ML work?\n3. ML fundamentals"
        mock_llm.return_value.invoke.return_value = mock_response
        
        enhancer = QueryEnhancer("test-model")
        variations = enhancer.expand_query("What is machine learning?")
        
        assert len(variations) > 1
        assert "What is machine learning?" in variations  # Original included
        assert any("ML" in var for var in variations)  # Contains variations

    @patch('app.query_enhancement.ChatOllama')
    def test_hyde_generation(self, mock_llm):
        """Should generate hypothetical document"""
        mock_response = Mock()
        mock_response.content = "Machine learning is a subset of AI that enables computers to learn..."
        mock_llm.return_value.invoke.return_value = mock_response
        
        enhancer = QueryEnhancer("test-model")
        hyde_doc = enhancer.generate_hypothetical_document("What is machine learning?")
        
        assert len(hyde_doc) > 50  # Should be substantial
        assert "machine learning" in hyde_doc.lower()

    @patch('app.query_enhancement.ChatOllama')
    def test_query_decomposition_simple_query(self, mock_llm):
        """Should not decompose simple queries"""
        enhancer = QueryEnhancer("test-model")
        sub_questions = enhancer.decompose_complex_query("What is Python?")
        
        # Simple query should return as-is
        assert len(sub_questions) == 1
        assert sub_questions[0] == "What is Python?"

    @patch('app.query_enhancement.ChatOllama')  
    def test_query_decomposition_complex_query(self, mock_llm):
        """Should decompose complex multi-part queries"""
        mock_response = Mock()
        mock_response.content = "1. What is machine learning?\n2. How does it relate to AI?\n3. What are the main types?"
        mock_llm.return_value.invoke.return_value = mock_response
        
        enhancer = QueryEnhancer("test-model")
        complex_query = "What is machine learning and how does it relate to artificial intelligence?"
        sub_questions = enhancer.decompose_complex_query(complex_query)
        
        assert len(sub_questions) > 1
        assert all(len(q) > 10 for q in sub_questions)  # All should be substantial questions


class TestEnhancedQueryKnowledgebase:
    """Tests for the enhanced query_knowledgebase function"""

    @patch('app.query.load_vectorstore')
    @patch('app.query.QueryEnhancer')
    @patch('app.query.search_relevant_documents')
    @patch('app.query.generate_response')
    def test_enhanced_query_with_multiple_variations(self, mock_generate, mock_search, mock_enhancer_class, mock_load_vs):
        """Should use query enhancement by default"""
        # Setup mocks
        mock_vs = Mock()
        mock_load_vs.return_value = mock_vs
        
        mock_enhancer = Mock()
        mock_enhancer.expand_query.return_value = ["original query", "expanded query"]
        mock_enhancer.generate_hypothetical_document.return_value = "hypothetical doc"
        mock_enhancer_class.return_value = mock_enhancer
        
        mock_docs = [Mock(page_content="test content", metadata={"source": "test.pdf"})]
        mock_search.return_value = mock_docs
        mock_generate.return_value = "Test answer"
        
        # Call function
        result = query_knowledgebase("test query", use_query_enhancement=True)
        
        # Assertions
        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        mock_enhancer.expand_query.assert_called_once()
        mock_enhancer.generate_hypothetical_document.assert_called_once()
        # Should search with multiple queries (original + expanded + hyde)
        assert mock_search.call_count >= 2

    @patch('app.query.load_vectorstore')
    @patch('app.query.search_relevant_documents')
    @patch('app.query.generate_response')
    def test_basic_query_without_enhancement(self, mock_generate, mock_search, mock_load_vs):
        """Should work without query enhancement"""
        # Setup mocks
        mock_vs = Mock()
        mock_load_vs.return_value = mock_vs
        
        mock_docs = [Mock(page_content="test content", metadata={"source": "test.pdf"})]
        mock_search.return_value = mock_docs
        mock_generate.return_value = "Test answer"
        
        # Call function
        result = query_knowledgebase("test query", use_query_enhancement=False)
        
        # Assertions
        assert isinstance(result, dict)
        assert "answer" in result
        # Should only search once with original query
        mock_search.assert_called_once()
