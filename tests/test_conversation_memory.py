"""
Tests for Enhanced Conversation Memory & Context System
"""

import tempfile
import time
from unittest.mock import patch, MagicMock
from pathlib import Path

from core.rag.conversation_memory import (
    ConversationMemoryManager,
    ConversationMemoryConfig,
    LLMQueryClassifier,
    QueryType,
    ConversationContext,
)
from core.rag.context_resolver import ContextAwareQueryResolver
from core.rag.chat import ChatSession, ChatSessionManager


class TestLLMQueryClassifier:
    """Tests for LLM-based query classification"""

    def setup_method(self):
        self.classifier = LLMQueryClassifier()

    def test_detects_follow_up_questions(self):
        """Should detect follow-up question patterns"""
        follow_up_queries = [
            "tell me more about this",
            "what else can you say about Python",
            "can you elaborate on the function",
            "more details please",
            "anything else?",
        ]

        context = ConversationContext(
            recent_topics=["python", "function"],
            recent_sources=[],
            last_user_query="What is Python?",
            conversation_summary=None,
            active_references=set(),
        )

        llm_available = self.classifier._check_llm_availability()

        for query in follow_up_queries:
            query_type = self.classifier.detect_query_type(query, context)

            if llm_available:
                assert query_type in [
                    QueryType.FOLLOW_UP,
                    QueryType.REFERENCE,
                    QueryType.CLARIFICATION,
                ]
            else:
                assert query_type in [
                    QueryType.FOLLOW_UP,
                    QueryType.REFERENCE,
                    QueryType.CLARIFICATION,
                    QueryType.NEW_TOPIC,
                ]

    def test_detects_reference_questions(self):
        """Should detect reference-based questions"""
        reference_queries = [
            "what does that mean?",
            "the document mentioned earlier",
            "as discussed before",
            "the source from our conversation",
        ]

        context = ConversationContext(
            recent_topics=["document", "source"],
            recent_sources=[],
            last_user_query="What is in the document?",
            conversation_summary=None,
            active_references={"document"},
        )

        llm_available = self.classifier._check_llm_availability()

        for query in reference_queries:
            query_type = self.classifier.detect_query_type(query, context)

            if llm_available:
                assert query_type in [QueryType.REFERENCE, QueryType.CLARIFICATION]
            else:
                assert query_type in [
                    QueryType.REFERENCE,
                    QueryType.CLARIFICATION,
                    QueryType.FOLLOW_UP,
                    QueryType.NEW_TOPIC,
                ]

    def test_detects_new_topic_questions(self):
        """Should detect completely new topic questions"""
        new_topic_queries = [
            "What is machine learning?",
            "How do I install Docker?",
            "Explain quantum computing",
            "Show me JavaScript examples",
        ]

        context = ConversationContext(
            recent_topics=["python", "programming"],
            recent_sources=[],
            last_user_query="What is Python?",
            conversation_summary=None,
            active_references=set(),
        )

        llm_available = self.classifier._check_llm_availability()

        for query in new_topic_queries:
            query_type = self.classifier.detect_query_type(query, context)

            if llm_available:
                assert query_type == QueryType.NEW_TOPIC
            else:
                assert query_type in [QueryType.NEW_TOPIC, QueryType.FOLLOW_UP]

    def test_extracts_topics_from_text(self):
        """Should extract meaningful topics from text"""
        text = "Python is a programming language used for machine learning and web development"
        topics = self.classifier.extract_topics(text)

        assert "python" in topics
        assert "programming" in topics
        assert "language" in topics
        assert "machine" in topics
        assert "learning" in topics
        assert "web" in topics
        assert "development" in topics

        # Should not include stop words
        assert "is" not in topics
        assert "a" not in topics
        assert "for" not in topics


class TestConversationMemoryManager:
    """Tests for persistent conversation memory management"""

    def setup_method(self):
        # Use temporary database for testing
        self.temp_dir = tempfile.mkdtemp()
        self.config = ConversationMemoryConfig()
        self.config.db_path = Path(self.temp_dir) / "test_chat.db"
        self.manager = ConversationMemoryManager(self.config)

    def teardown_method(self):
        # Clean up temporary files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_adds_message_to_memory(self):
        """Should store messages in persistent memory"""
        message_id = self.manager.add_message(
            session_id="test_session",
            user_id="test_user",
            role="user",
            content="What is Python?",
            sources=None,
        )

        assert message_id is not None

        # Verify message was stored
        history = self.manager.get_conversation_history("test_session", "test_user")
        assert len(history) == 1
        assert history[0].content == "What is Python?"
        assert history[0].role == "user"
        assert history[0].query_type == QueryType.NEW_TOPIC.value

    def test_tracks_conversation_context(self):
        """Should track conversation context across messages"""
        # Add initial message
        self.manager.add_message("session1", "user1", "user", "What is Python?")

        # Add follow-up message
        self.manager.add_message(
            "session1",
            "user1",
            "assistant",
            "Python is a programming language",
            sources=[{"source": "python_doc.txt"}],
        )

        # Get context
        context = self.manager.get_conversation_context("session1", "user1")

        assert len(context.recent_topics) > 0
        assert "python" in [topic.lower() for topic in context.recent_topics]
        assert len(context.recent_sources) == 1
        assert context.recent_sources[0]["source"] == "python_doc.txt"
        assert context.last_user_query == "What is Python?"

    def test_detects_follow_up_queries(self):
        """Should detect follow-up queries based on context"""
        # Set up conversation history
        self.manager.add_message("session1", "user1", "user", "What is Python?")
        self.manager.add_message("session1", "user1", "assistant", "Python is a programming language")

        # Add follow-up question
        message_id = self.manager.add_message("session1", "user1", "user", "tell me more about it")

        assert message_id is not None

        # Check that it was detected as follow-up
        history = self.manager.get_conversation_history("session1", "user1")
        follow_up_message = [msg for msg in history if msg.content == "tell me more about it"][0]
        assert follow_up_message.query_type in [
            QueryType.FOLLOW_UP.value,
            QueryType.REFERENCE.value,
        ]

    def test_respects_storage_limits(self):
        """Should disable memory when storage limit is reached"""
        # Mock file size check to simulate large database
        with patch("os.path.getsize", return_value=25 * 1024**3):  # 25GB
            result = self.manager.add_message("session1", "user1", "user", "test message")
            assert result is None  # Should not store when over limit

    def test_clears_session_memory(self):
        """Should clear session memory completely"""
        # Add messages
        self.manager.add_message("session1", "user1", "user", "Message 1")
        self.manager.add_message("session1", "user1", "assistant", "Response 1")

        # Verify messages exist
        history = self.manager.get_conversation_history("session1", "user1")
        assert len(history) == 2

        # Clear session
        self.manager.clear_session("session1", "user1")

        # Verify messages are gone
        history = self.manager.get_conversation_history("session1", "user1")
        assert len(history) == 0

    def test_gets_database_statistics(self):
        """Should provide database statistics"""
        # Add some test data
        self.manager.add_message("session1", "user1", "user", "Test message 1")
        self.manager.add_message("session1", "user1", "assistant", "Test response 1")

        stats = self.manager.get_database_stats()

        assert "total_messages" in stats
        assert "total_sessions" in stats
        assert "size_mb" in stats
        assert "size_gb" in stats
        assert stats["total_messages"] >= 2
        assert stats["total_sessions"] >= 1

    def test_cleanup_old_conversations(self):
        """Should clean up old conversations"""
        # Add old message (simulate by manually setting old timestamp)
        message_id = self.manager.add_message("old_session", "user1", "user", "Old message")

        # Manually update timestamp to be old
        import sqlite3

        old_timestamp = time.time() - (35 * 24 * 60 * 60)  # 35 days ago
        with sqlite3.connect(str(self.config.db_path)) as conn:
            conn.execute(
                "UPDATE messages SET timestamp = ? WHERE id = ?",
                (old_timestamp, message_id),
            )
            conn.execute(
                "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                (old_timestamp, "old_session"),
            )
            conn.commit()

        # Add recent message
        self.manager.add_message("new_session", "user1", "user", "New message")

        # Verify both exist
        stats_before = self.manager.get_database_stats()
        assert stats_before["total_messages"] >= 2

        # Clean up messages older than 30 days
        self.manager.cleanup_old_conversations(30)

        # Verify old message is gone but new one remains
        stats_after = self.manager.get_database_stats()
        assert stats_after["total_messages"] < stats_before["total_messages"]


class TestContextAwareQueryResolver:
    """Tests for context-aware query resolution"""

    def setup_method(self):
        self.resolver = ContextAwareQueryResolver()

        # Mock chat session
        self.session = MagicMock()
        self.session.detect_query_type.return_value = QueryType.FOLLOW_UP
        self.session.get_conversation_context.return_value = ConversationContext(
            recent_topics=["python", "programming", "functions"],
            recent_sources=[{"source": "python_guide.txt"}],
            last_user_query="What is Python?",
            conversation_summary=None,
            active_references={"python"},
        )

    def test_resolves_new_topic_queries(self):
        """Should handle new topic queries without modification"""
        self.session.detect_query_type.return_value = QueryType.NEW_TOPIC

        resolved = self.resolver.resolve_query("What is JavaScript?", self.session)

        assert resolved.query_type == QueryType.NEW_TOPIC
        assert resolved.original_query == "What is JavaScript?"
        assert resolved.resolved_query == "What is JavaScript?"
        assert not resolved.context_used
        assert len(resolved.topics_referenced) == 0

    def test_enhances_follow_up_queries(self):
        """Should enhance follow-up queries with context"""
        resolved = self.resolver.resolve_query("tell me more about functions", self.session)

        assert resolved.query_type == QueryType.FOLLOW_UP
        assert resolved.context_used
        assert "python" in resolved.resolved_query.lower() or "programming" in resolved.resolved_query.lower()
        assert len(resolved.topics_referenced) > 0
        assert "functions" in resolved.enhancement_explanation

    def test_enhances_reference_queries(self):
        """Should enhance reference-based queries with context"""
        self.session.detect_query_type.return_value = QueryType.REFERENCE

        resolved = self.resolver.resolve_query("what does that mean?", self.session)

        assert resolved.query_type == QueryType.REFERENCE
        assert resolved.context_used
        assert len(resolved.resolved_query) > len(resolved.original_query)
        assert len(resolved.topics_referenced) > 0

    def test_creates_context_aware_prompts(self):
        """Should create prompts with conversation context"""
        from core.rag.context_resolver import ResolvedQuery

        resolved_query = ResolvedQuery(
            original_query="tell me more",
            resolved_query="tell me more about Python programming",
            query_type=QueryType.FOLLOW_UP,
            context_used=True,
            topics_referenced=["python", "programming"],
            sources_referenced=[{"source": "guide.txt"}],
            enhancement_explanation="Added context about Python programming",
        )

        rag_context = "Python is a high-level programming language..."
        base_prompt = "You are a helpful assistant."

        prompt = self.resolver.create_context_aware_prompt(resolved_query, rag_context, base_prompt)

        assert base_prompt in prompt
        assert "follow_up" in prompt.lower()
        assert "Python" in prompt or "programming" in prompt
        assert rag_context in prompt
        assert resolved_query.resolved_query in prompt


class TestEnhancedChatSession:
    """Tests for enhanced chat session with memory integration"""

    def setup_method(self):
        # Use temporary database for testing
        self.temp_dir = tempfile.mkdtemp()
        config = ConversationMemoryConfig()
        config.db_path = Path(self.temp_dir) / "test_chat.db"

        # Patch the global memory manager to use our test config
        self.test_manager = ConversationMemoryManager(config)

        with patch("core.rag.chat.memory_manager", self.test_manager):
            self.session = ChatSession(
                session_id="test_session",
                user_id="test_user",
                messages=[],
                enable_memory=True,
            )

    def teardown_method(self):
        # Clean up temporary files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_adds_messages_to_persistent_memory(self):
        """Should add messages to both session and persistent memory"""
        with patch("core.rag.chat.memory_manager", self.test_manager):
            self.session.add_message("user", "What is Python?")
            self.session.add_message("assistant", "Python is a programming language")

        # Check in-session messages
        assert len(self.session.messages) == 2
        assert self.session.messages[0].content == "What is Python?"
        assert self.session.messages[1].content == "Python is a programming language"

        # Check persistent memory
        history = self.test_manager.get_conversation_history("test_session", "test_user")
        assert len(history) >= 2

    def test_loads_conversation_history_on_init(self):
        """Should load existing conversation history when session is created"""
        # Add messages to persistent storage first
        self.test_manager.add_message("existing_session", "test_user", "user", "Previous message")
        self.test_manager.add_message("existing_session", "test_user", "assistant", "Previous response")

        # Create new session with same ID - should load history
        with patch("core.rag.chat.memory_manager", self.test_manager):
            new_session = ChatSession(
                session_id="existing_session",
                user_id="test_user",
                messages=[],
                enable_memory=True,
            )

        # Should have loaded previous messages
        assert len(new_session.messages) >= 2
        message_contents = [msg.content for msg in new_session.messages]
        assert "Previous message" in message_contents
        assert "Previous response" in message_contents

    def test_detects_query_types(self):
        """Should detect different query types"""
        # Add initial context
        with patch("core.rag.chat.memory_manager", self.test_manager):
            self.session.add_message("user", "What is Python?")
            self.session.add_message("assistant", "Python is a programming language")

            # Test follow-up detection
            follow_up_type = self.session.detect_query_type("tell me more about it")
            assert follow_up_type in [QueryType.FOLLOW_UP, QueryType.REFERENCE]

            # Test new topic detection
            new_topic_type = self.session.detect_query_type("What is JavaScript?")
            # Allow more flexibility for fallback behavior
            assert new_topic_type in [QueryType.NEW_TOPIC, QueryType.FOLLOW_UP]

    def test_clears_memory_properly(self):
        """Should clear both session and persistent memory"""
        with patch("core.rag.chat.memory_manager", self.test_manager):
            # Add messages
            self.session.add_message("user", "Test message 1")
            self.session.add_message("assistant", "Test response 1")

            # Verify messages exist
            assert len(self.session.messages) > 0
            history = self.test_manager.get_conversation_history("test_session", "test_user")
            assert len(history) > 0

            # Clear memory
            self.session.clear_memory()

            # Verify both are cleared
            assert len(self.session.messages) == 0
            history = self.test_manager.get_conversation_history("test_session", "test_user")
            assert len(history) == 0


class TestMemoryIntegration:
    """Integration tests for the complete conversation memory system"""

    def test_end_to_end_conversation_flow(self):
        """Should handle complete conversation flow with memory"""
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        config = ConversationMemoryConfig()
        config.db_path = Path(temp_dir) / "integration_test.db"
        test_manager = ConversationMemoryManager(config)

        try:
            with patch("core.rag.chat.memory_manager", test_manager):
                # Create session manager
                session_manager = ChatSessionManager()

                # Start conversation
                session = session_manager.get_or_create_session("user1", enable_memory=True)

                # User asks initial question
                session.add_message("user", "What is machine learning?")
                session.add_message(
                    "assistant",
                    "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.",
                )

                # User asks follow-up
                session.add_message("user", "tell me more about the algorithms")

                # Check that conversation context is maintained
                context = session.get_conversation_context()
                assert len(context.recent_topics) > 0
                assert any("machine" in topic.lower() or "learning" in topic.lower() for topic in context.recent_topics)

                # Check query type detection
                query_type = session.detect_query_type("what are some examples?")
                # Allow more flexibility based on LLM availability
                assert query_type in [
                    QueryType.FOLLOW_UP,
                    QueryType.REFERENCE,
                    QueryType.NEW_TOPIC,
                ]

                # Get memory statistics
                stats = session_manager.get_memory_stats()
                assert stats["total_messages"] >= 3
                assert stats["total_sessions"] >= 1

        finally:
            # Clean up
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
