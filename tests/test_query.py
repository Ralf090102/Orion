"""
Tests for Orion's enhanced query processing functions.
"""

from unittest.mock import Mock, patch
from app.query import format_context, create_prompt, query_knowledgebase
from app.query_enhancement import QueryEnhancer
from app.query_processor import QueryProcessor, QueryIntent, QueryAnalysis


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

        assert any(indicator in prompt.lower() for indicator in fallback_indicators)


class TestQueryEnhancement:
    """Tests for enhanced query processing"""

    @patch("app.query_enhancement.ChatOllama")
    def test_query_expansion(self, mock_llm):
        """Should expand query into multiple variations"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = (
            "1. What is machine learning?\n2. How does ML work?\n3. ML fundamentals"
        )
        mock_llm.return_value.invoke.return_value = mock_response

        enhancer = QueryEnhancer("test-model")
        variations = enhancer.expand_query("What is machine learning?")

        assert len(variations) > 1
        assert "What is machine learning?" in variations  # Original included
        assert any("ML" in var for var in variations)  # Contains variations

    @patch("app.query_enhancement.ChatOllama")
    def test_hyde_generation(self, mock_llm):
        """Should generate hypothetical document"""
        mock_response = Mock()
        mock_response.content = (
            "Machine learning is a subset of AI that enables computers to learn..."
        )
        mock_llm.return_value.invoke.return_value = mock_response

        enhancer = QueryEnhancer("test-model")
        hyde_doc = enhancer.generate_hypothetical_document("What is machine learning?")

        assert len(hyde_doc) > 50  # Should be substantial
        assert "machine learning" in hyde_doc.lower()

    @patch("app.query_enhancement.ChatOllama")
    def test_query_decomposition_simple_query(self, mock_llm):
        """Should not decompose simple queries"""
        enhancer = QueryEnhancer("test-model")
        sub_questions = enhancer.decompose_complex_query("What is Python?")

        # Simple query should return as-is
        assert len(sub_questions) == 1
        assert sub_questions[0] == "What is Python?"

    @patch("app.query_enhancement.ChatOllama")
    def test_query_decomposition_complex_query(self, mock_llm):
        """Should decompose complex multi-part queries"""
        mock_response = Mock()
        mock_response.content = "1. What is machine learning?\n2. How does it relate to AI?\n3. What are the main types?"
        mock_llm.return_value.invoke.return_value = mock_response

        enhancer = QueryEnhancer("test-model")
        complex_query = "What is machine learning and how does it relate to artificial intelligence?"
        sub_questions = enhancer.decompose_complex_query(complex_query)

        assert len(sub_questions) > 1
        assert all(
            len(q) > 10 for q in sub_questions
        )  # All should be substantial questions


class TestEnhancedQueryKnowledgebase:
    """Tests for the enhanced query_knowledgebase function"""

    @patch("app.query.load_vectorstore")
    @patch("app.query.QueryEnhancer")
    @patch("app.query.search_relevant_documents")
    @patch("app.query.generate_response")
    def test_enhanced_query_with_multiple_variations(
        self, mock_generate, mock_search, mock_enhancer_class, mock_load_vs
    ):
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

    @patch("app.query.load_vectorstore")
    @patch("app.query.search_relevant_documents")
    @patch("app.query.generate_response")
    def test_basic_query_without_enhancement(
        self, mock_generate, mock_search, mock_load_vs
    ):
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


class TestQueryProcessor:
    """Tests for the advanced QueryProcessor system"""

    def setup_method(self):
        """Set up a fresh QueryProcessor for each test"""
        self.processor = QueryProcessor()

    def test_intent_detection_factual(self):
        """Should detect factual queries correctly"""
        test_cases = [
            "What is machine learning?",
            "Define artificial intelligence",
            "Explain how neural networks work",
            "Tell me about Python programming"
        ]
        
        for query in test_cases:
            intent, confidence = self.processor.detect_intent(query)
            assert intent == QueryIntent.FACTUAL
            assert confidence > 0.0

    def test_intent_detection_analytical(self):
        """Should detect analytical/comparison queries"""
        test_cases = [
            "Compare Python and Java",
            "What's the difference between REST and GraphQL?",
            "Analyze the pros and cons of React vs Vue",
            "Evaluate Docker versus Kubernetes"
        ]
        
        for query in test_cases:
            intent, confidence = self.processor.detect_intent(query)
            assert intent == QueryIntent.ANALYTICAL
            assert confidence > 0.0

    def test_intent_detection_procedural(self):
        """Should detect how-to/procedural queries"""
        test_cases = [
            "How to install Python?",
            "How do I create a REST API?",
            "Steps to deploy a web application",
            "Guide to setting up a database"
        ]
        
        for query in test_cases:
            intent, confidence = self.processor.detect_intent(query)
            assert intent == QueryIntent.PROCEDURAL
            assert confidence > 0.0

    def test_intent_detection_troubleshooting(self):
        """Should detect troubleshooting queries"""
        test_cases = [
            "Why doesn't my code work?",
            "Fix this segmentation fault error",
            "Debug this Python problem",
            "Solve this database connection issue"
        ]
        
        for query in test_cases:
            intent, confidence = self.processor.detect_intent(query)
            assert intent == QueryIntent.TROUBLESHOOTING
            assert confidence > 0.0

    def test_intent_detection_creative(self):
        """Should detect creative/generation queries"""
        test_cases = [
            "Write a function to sort arrays",
            "Create a REST API endpoint",
            "Generate a SQL query",
            "Build a simple web page"
        ]
        
        for query in test_cases:
            intent, confidence = self.processor.detect_intent(query)
            assert intent == QueryIntent.CREATIVE
            assert confidence > 0.0

    def test_intent_detection_default_factual(self):
        """Should default to factual for ambiguous queries"""
        ambiguous_queries = [
            "Python",
            "Database stuff",
            "Something about APIs"
        ]
        
        for query in ambiguous_queries:
            intent, confidence = self.processor.detect_intent(query)
            assert intent == QueryIntent.FACTUAL
            assert confidence <= 0.5  # Low confidence for ambiguous queries

    def test_keyword_extraction(self):
        """Should extract meaningful keywords from queries"""
        query = "How to implement machine learning algorithms in Python?"
        keywords = self.processor.extract_keywords(query)
        
        # Should include important technical terms
        expected_keywords = ["implement", "machine", "learning", "algorithms", "python"]
        for keyword in expected_keywords:
            assert keyword in keywords
        
        # Should exclude stop words
        stop_words = ["how", "to", "in"]
        for stop_word in stop_words:
            assert stop_word not in keywords

    def test_keyword_extraction_preserves_order(self):
        """Should preserve order of keywords while removing duplicates"""
        query = "Python Python programming programming language"
        keywords = self.processor.extract_keywords(query)
        
        assert keywords == ["python", "programming", "language"]

    def test_sub_query_generation_analytical(self):
        """Should break down analytical queries into sub-questions"""
        query = "Compare Python and Java for web development"
        intent = QueryIntent.ANALYTICAL
        
        sub_queries = self.processor.break_into_sub_queries(query, intent)
        
        # Should generate multiple sub-questions for comparison
        assert len(sub_queries) > 1
        # Should contain questions about each item
        sub_queries_text = " ".join(sub_queries).lower()
        assert "python" in sub_queries_text
        assert "java" in sub_queries_text

    def test_sub_query_generation_procedural(self):
        """Should break down procedural queries into steps"""
        query = "How to deploy a web application"
        intent = QueryIntent.PROCEDURAL
        
        sub_queries = self.processor.break_into_sub_queries(query, intent)
        
        # Should generate multiple sub-questions
        assert len(sub_queries) > 1
        # Should include prerequisite and step-based questions
        sub_queries_text = " ".join(sub_queries).lower()
        assert "prerequisites" in sub_queries_text or "steps" in sub_queries_text

    def test_sub_query_generation_troubleshooting(self):
        """Should break down troubleshooting queries systematically"""
        query = "Fix this database connection error"
        intent = QueryIntent.TROUBLESHOOTING
        
        sub_queries = self.processor.break_into_sub_queries(query, intent)
        
        # Should generate diagnostic sub-questions
        assert len(sub_queries) > 1
        sub_queries_text = " ".join(sub_queries).lower()
        assert any(word in sub_queries_text for word in ["causes", "solutions", "error"])

    def test_sub_query_generation_simple_query(self):
        """Should not break down simple queries unnecessarily"""
        query = "What is Python?"
        intent = QueryIntent.FACTUAL
        
        sub_queries = self.processor.break_into_sub_queries(query, intent)
        
        # Should return the original query for simple cases
        assert len(sub_queries) == 1
        assert sub_queries[0] == query

    def test_query_validation_time_sensitive(self):
        """Should reject time-sensitive queries we can't answer"""
        time_queries = [
            "What's the weather today?",
            "Latest news about AI",
            "Current stock prices"
        ]
        
        for query in time_queries:
            can_answer, reasoning = self.processor.validate_query(query, [])
            assert can_answer is False
            assert "real-time data" in reasoning.lower() or "time" in reasoning.lower()

    def test_query_validation_personal_info(self):
        """Should reject personal information queries"""
        personal_queries = [
            "Tell me about my personal files",
            "Access my private data",
            "Show me my account information"
        ]
        
        for query in personal_queries:
            can_answer, reasoning = self.processor.validate_query(query, [])
            assert can_answer is False
            assert "personal" in reasoning.lower()
        
        # These should NOT be rejected (they're technical, not personal)
        technical_with_my = [
            "What is my code doing wrong?",
            "Fix my Python script",
            "Debug my application"
        ]
        
        for query in technical_with_my:
            can_answer, reasoning = self.processor.validate_query(query, [])
            assert can_answer is True  # These are technical questions, not personal info

    def test_query_validation_technical_queries(self):
        """Should accept technical queries we can likely answer"""
        technical_queries = [
            "How does a database work?",
            "Explain API design patterns",
            "Programming best practices"
        ]
        
        for query in technical_queries:
            can_answer, reasoning = self.processor.validate_query(query, [])
            assert can_answer is True
            assert "technical" in reasoning.lower() or "documentation" in reasoning.lower()

    def test_query_validation_default_acceptance(self):
        """Should default to accepting queries we might be able to answer"""
        generic_queries = [
            "Tell me about machine learning",
            "Explain data structures",
            "Software architecture patterns"
        ]
        
        for query in generic_queries:
            can_answer, reasoning = self.processor.validate_query(query, [])
            assert can_answer is True

    def test_full_query_analysis(self):
        """Should perform complete query analysis"""
        query = "How to implement a RESTful API in Python?"
        
        analysis = self.processor.analyze_query(query)
        
        # Check all components of the analysis
        assert isinstance(analysis, QueryAnalysis)
        assert analysis.original_query == query
        assert analysis.intent == QueryIntent.PROCEDURAL
        assert analysis.confidence > 0.0
        assert len(analysis.keywords) > 0
        assert len(analysis.sub_queries) >= 1
        assert isinstance(analysis.can_answer, bool)
        assert len(analysis.reasoning) > 0
        
        # Should extract relevant keywords
        keywords_text = " ".join(analysis.keywords)
        assert "implement" in keywords_text
        assert "restful" in keywords_text or "api" in keywords_text
        assert "python" in keywords_text

    def test_complex_comparison_query_analysis(self):
        """Should handle complex comparison queries well"""
        query = "Compare React and Vue.js for building single-page applications"
        
        analysis = self.processor.analyze_query(query)
        
        assert analysis.intent == QueryIntent.ANALYTICAL
        assert analysis.confidence > 0.0
        assert len(analysis.sub_queries) > 1  # Should break down the comparison
        
        # Keywords should include the compared items
        keywords_text = " ".join(analysis.keywords)
        assert "react" in keywords_text
        assert "vue" in keywords_text
        assert "building" in keywords_text or "single" in keywords_text

    def test_troubleshooting_query_analysis(self):
        """Should handle troubleshooting queries appropriately"""
        query = "Why am I getting a 'module not found' error in my Python script?"
        
        analysis = self.processor.analyze_query(query)
        
        assert analysis.intent == QueryIntent.TROUBLESHOOTING
        assert analysis.confidence > 0.0
        assert analysis.can_answer is True  # Technical issue we can help with
        
        # Should have relevant keywords
        keywords_text = " ".join(analysis.keywords)
        assert "module" in keywords_text
        assert "error" in keywords_text
        assert "python" in keywords_text
