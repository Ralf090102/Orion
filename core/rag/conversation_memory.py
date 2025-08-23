"""
Enhanced Conversation Memory & Context Management System

Provides persistent conversation storage, context resolution, and follow-up handling
for the Orion RAG system using SQLite database.
"""

import sqlite3
import json
import time
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum

from core.utils.orion_utils import log_info, log_warning, log_error, log_debug
from core.rag.llm import generate_response, check_ollama_connection


class QueryType(Enum):
    """Types of queries based on conversation context"""

    NEW_TOPIC = "new_topic"
    FOLLOW_UP = "follow_up"
    REFERENCE = "reference"
    CLARIFICATION = "clarification"


@dataclass
class ConversationMessage:
    """Enhanced message with context tracking"""

    id: Optional[int]
    session_id: str
    user_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    sources: Optional[List[Dict]] = None
    topics: Optional[List[str]] = None
    query_type: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON storage"""
        return {
            "sources": self.sources or [],
            "topics": self.topics or [],
            "query_type": self.query_type,
        }


@dataclass
class ConversationContext:
    """Context information for resolving queries"""

    recent_topics: List[str]
    recent_sources: List[Dict]
    last_user_query: Optional[str]
    conversation_summary: Optional[str]
    active_references: Set[str]  # Currently discussed entities


class ConversationMemoryConfig:
    """Configuration for conversation memory system"""

    def __init__(self):
        self.db_path = Path("data/conversations/chat_history.db")
        self.max_db_size_gb = 20
        self.context_window_size = 7
        self.topic_retention_days = 30
        self.enable_memory = True
        self.compression_threshold = 1000  # Messages before compression

        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


class LLMQueryClassifier:
    """LLM-based query classification using Ollama Mistral"""

    def __init__(self, model: str = "mistral", fallback_to_patterns: bool = True):
        self.model = model
        self.fallback_to_patterns = fallback_to_patterns
        self.pattern_fallback = PatternBasedFallback() if fallback_to_patterns else None
        self._llm_available = None

    def _check_llm_availability(self) -> bool:
        """Check if LLM is available (cached for performance)"""
        if self._llm_available is None:
            self._llm_available = check_ollama_connection()
            if not self._llm_available:
                log_warning("Ollama not available, will use pattern-based fallback")
        return self._llm_available

    def detect_query_type(self, query: str, context: ConversationContext) -> QueryType:
        """Detect query type using LLM with fallback to patterns"""

        # Quick pre-filter: if no context at all, likely NEW_TOPIC unless clearly referential
        if (
            not context.recent_topics
            and not context.last_user_query
            and not context.recent_sources
        ):
            # Check for obvious reference words
            if any(
                ref_word in query.lower()
                for ref_word in [
                    "you said",
                    "you mentioned",
                    "mentioned earlier",
                    "discussed before",
                    "that document",
                    "our conversation",
                    "before",
                    "earlier",
                    "previous",
                ]
            ):
                # Even without context, this is clearly referential
                pass  # Continue with LLM classification
            else:
                # No context + no clear reference = likely new topic
                return QueryType.NEW_TOPIC

        # Additional pre-filter: Check if query is asking about a completely different topic
        # Even with context present
        if context.recent_topics:
            query_lower = query.lower()

            # First check if it's clearly a follow-up question regardless of subject
            clear_follow_up_patterns = [
                "what are some examples",
                "give me examples",
                "show me examples",
                "what are the",
                "tell me more",
                "more about",
                "more details",
                "what else",
                "anything else",
                "can you elaborate",
            ]

            if any(pattern in query_lower for pattern in clear_follow_up_patterns):
                # This is clearly a follow-up, don't pre-filter as NEW_TOPIC
                pass  # Continue to LLM classification
            else:
                # Look for patterns that indicate a completely new topic
                new_topic_indicators = [
                    "what is",
                    "how do i",
                    "explain",
                    "show me",
                    "tell me about",
                    "how does",
                    "define",
                    "describe",
                ]

                # If it's a "what is X" or similar pattern and X is not in recent topics
                if any(indicator in query_lower for indicator in new_topic_indicators):
                    # Extract the main subject from the query
                    import re

                    # Look for the main subject after common question starters
                    subject_match = re.search(
                        r"(?:what is|how do i|explain|show me|tell me about|how does|define|describe)"
                        r"\s+([a-zA-Z][a-zA-Z\s]*?)(?:\?|$)",
                        query_lower,
                    )
                    if subject_match:
                        subject = subject_match.group(1).strip()
                        # Check if any word in the subject matches recent topics closely
                        subject_words = subject.split()
                        topic_overlap = any(
                            any(
                                topic_word in recent_topic.lower()
                                or recent_topic.lower() in topic_word
                                for recent_topic in context.recent_topics
                            )
                            for topic_word in subject_words
                        )

                        # If no significant overlap with recent topics, it's likely a new topic
                        if not topic_overlap:
                            return QueryType.NEW_TOPIC

        # Try LLM first if available
        if self._check_llm_availability():
            try:
                return self._classify_with_llm(query, context)
            except Exception as e:
                log_error(f"LLM classification failed: {e}")
                # Fall through to pattern fallback

        # Use pattern-based fallback
        if self.pattern_fallback:
            return self.pattern_fallback.detect_query_type(query, context)

        # Last resort - simple heuristic
        return self._simple_heuristic(query)

    def _classify_with_llm(self, query: str, context: ConversationContext) -> QueryType:
        """Classify query using LLM"""

        # Build context information
        context_info = self._build_context_string(context)

        # Create classification prompt
        prompt = self._create_classification_prompt(query, context_info)

        # Get LLM response
        try:
            response = generate_response(
                prompt, model=self.model, max_tokens=10, temperature=0.1
            )
            classification = self._parse_llm_response(response)

            log_debug(f"LLM classified '{query[:50]}...' as {classification.value}")
            return classification

        except Exception as e:
            log_error(f"Error in LLM classification: {e}")
            raise

    def _build_context_string(self, context: ConversationContext) -> str:
        """Build context information for the prompt"""
        context_parts = []

        if context.last_user_query:
            context_parts.append(f"Previous query: {context.last_user_query}")

        if context.recent_topics:
            topics = ", ".join(context.recent_topics[:5])  # Limit for token efficiency
            context_parts.append(f"Recent topics: {topics}")

        if context.recent_sources:
            sources = [s.get("source", "unknown") for s in context.recent_sources[:3]]
            context_parts.append(f"Recent sources: {', '.join(sources)}")

        return " | ".join(context_parts) if context_parts else "No previous context"

    def _create_classification_prompt(self, query: str, context_info: str) -> str:
        """Create the classification prompt for the LLM"""
        return f"""Classify this user query into exactly one category:

CATEGORIES:
- FOLLOW_UP: User wants more details/examples about the SAME topic we're currently discussing
- REFERENCE: User refers to previous conversation/documents explicitly ("you said", "mentioned earlier")  
- CLARIFICATION: User didn't understand something and needs explanation ("what do you mean", "clarify")
- NEW_TOPIC: User starts a completely different topic OR asks "What is X" where X is unrelated to current discussion

IMPORTANT RULES:
- If query asks "What is [something]" and [something] is NOT closely related to recent topics → NEW_TOPIC
- If query asks about a different technology/concept than what's being discussed → NEW_TOPIC  
- Only use FOLLOW_UP if asking for more info about the SAME topic currently being discussed
- Examples: "What is JavaScript?" when discussing Python = NEW_TOPIC, 
  "Tell me more about functions" when discussing Python = FOLLOW_UP

CONTEXT: {context_info}

USER QUERY: "{query}"

Respond with only one word: FOLLOW_UP, REFERENCE, CLARIFICATION, or NEW_TOPIC"""

    def _parse_llm_response(self, response: str) -> QueryType:
        """Parse LLM response into QueryType enum"""
        response_clean = response.strip().upper()

        # Direct mapping
        mapping = {
            "FOLLOW_UP": QueryType.FOLLOW_UP,
            "REFERENCE": QueryType.REFERENCE,
            "CLARIFICATION": QueryType.CLARIFICATION,
            "NEW_TOPIC": QueryType.NEW_TOPIC,
            "FOLLOW-UP": QueryType.FOLLOW_UP,  # Handle hyphenated version
            "NEW-TOPIC": QueryType.NEW_TOPIC,
        }

        if response_clean in mapping:
            return mapping[response_clean]

        # Try partial matching
        if "FOLLOW" in response_clean:
            return QueryType.FOLLOW_UP
        elif "REFERENCE" in response_clean:
            return QueryType.REFERENCE
        elif "CLARIF" in response_clean:
            return QueryType.CLARIFICATION
        elif "NEW" in response_clean:
            return QueryType.NEW_TOPIC

        # Default fallback
        log_warning(
            f"Could not parse LLM response: '{response}', defaulting to NEW_TOPIC"
        )
        return QueryType.NEW_TOPIC

    def _simple_heuristic(self, query: str) -> QueryType:
        """Simple heuristic when no LLM or patterns available"""
        query_lower = query.lower()

        if any(
            word in query_lower for word in ["more", "tell me", "what else", "continue"]
        ):
            return QueryType.FOLLOW_UP
        elif any(
            word in query_lower
            for word in ["clarify", "explain", "what mean", "understand"]
        ):
            return QueryType.CLARIFICATION
        elif any(
            word in query_lower
            for word in ["mentioned", "said", "discussed", "before", "earlier"]
        ):
            return QueryType.REFERENCE
        else:
            return QueryType.NEW_TOPIC

    def extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text - can also be LLM-enhanced in future"""
        # For now, keep the existing pattern-based approach as it's efficient
        # Could be enhanced with LLM later for better topic extraction

        # Remove common words and extract meaningful terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "can",
            "may",
            "might",
            "must",
            "shall",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
        }

        # Extract words that might be topics (longer than 2 chars, not stop words)
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*[a-zA-Z0-9]\b", text)
        topics = []

        for word in words:
            if len(word) > 2 and word.lower() not in stop_words:
                topics.append(word.lower())

        # Remove duplicates while preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)

        return unique_topics[:10]


class PatternBasedFallback:
    """Lightweight pattern-based fallback for when LLM is unavailable"""

    def detect_query_type(self, query: str, context: ConversationContext) -> QueryType:
        """Simplified pattern-based detection"""
        query_lower = query.lower().strip()

        # High-confidence follow-up patterns
        if any(
            pattern in query_lower
            for pattern in [
                "tell me more",
                "more about",
                "what else",
                "continue",
                "elaborate",
                "more details",
                "give me more",
                "expand on",
                "go deeper",
            ]
        ):
            return QueryType.FOLLOW_UP

        # High-confidence clarification patterns
        if any(
            pattern in query_lower
            for pattern in [
                "what do you mean",
                "clarify",
                "don't understand",
                "explain that",
                "be more specific",
                "break that down",
                "simplify",
            ]
        ):
            return QueryType.CLARIFICATION

        # High-confidence reference patterns
        if any(
            pattern in query_lower
            for pattern in [
                "you mentioned",
                "you said",
                "discussed before",
                "from our conversation",
                "remember when",
                "earlier",
                "previous",
                "that document",
            ]
        ):
            return QueryType.REFERENCE

        # Context-based decisions
        if context.recent_topics:
            # Simple pronoun + context check
            if (
                any(ref in query_lower for ref in ["that", "this", "it"])
                and len(context.recent_topics) > 0
            ):
                return QueryType.REFERENCE

            # Topic overlap check
            matching_topics = [
                topic for topic in context.recent_topics if topic.lower() in query_lower
            ]
            if len(matching_topics) >= 1:
                return QueryType.FOLLOW_UP

        return QueryType.NEW_TOPIC


class ConversationMemoryManager:
    """Manages persistent conversation memory with SQLite storage"""

    def __init__(self, config: Optional[ConversationMemoryConfig] = None):
        self.config = config or ConversationMemoryConfig()
        self.query_classifier = LLMQueryClassifier()
        self._db_path = str(self.config.db_path)
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database with required tables"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                # Create messages table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        sources TEXT,
                        topics TEXT,   
                        query_type TEXT
                    )
                """
                )

                # Create indexes separately
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_id ON messages(session_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_user_id ON messages(user_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp)"
                )

                # Create sessions table for metadata
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        last_activity REAL NOT NULL,
                        message_count INTEGER DEFAULT 0,
                        active_topics TEXT
                    )
                """
                )

                # Create indexes for sessions table
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity)"
                )

                conn.commit()
                log_info(f"Initialized conversation database: {self._db_path}")

        except Exception as e:
            log_error(f"Failed to initialize conversation database: {e}")
            self.config.enable_memory = False

    def _check_storage_limit(self) -> bool:
        """Check if database size is within limits"""
        try:
            if not os.path.exists(self._db_path):
                return True

            size_bytes = os.path.getsize(self._db_path)
            size_gb = size_bytes / (1024**3)

            if size_gb >= self.config.max_db_size_gb:
                log_warning(
                    f"Conversation database size ({size_gb:.2f}GB) exceeds limit "
                    f"({self.config.max_db_size_gb}GB). Memory recording disabled."
                )
                return False

            return True

        except Exception as e:
            log_error(f"Failed to check database size: {e}")
            return False

    def add_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None,
    ) -> Optional[int]:
        """Add a message to conversation memory"""
        if not self.config.enable_memory or not self._check_storage_limit():
            return None

        try:
            # Extract topics and detect query type
            topics = self.query_classifier.extract_topics(content)
            query_type = None

            if role == "user":
                context = self.get_conversation_context(session_id, user_id)
                query_type = self.query_classifier.detect_query_type(
                    content, context
                ).value

            # Create message
            message = ConversationMessage(
                id=None,
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
                timestamp=time.time(),
                sources=sources,
                topics=topics,
                query_type=query_type,
            )

            # Insert into database
            with sqlite3.connect(self._db_path) as conn:
                # Insert message
                cursor = conn.execute(
                    """
                    INSERT INTO messages 
                    (session_id, user_id, role, content, timestamp, sources, topics, query_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        message.session_id,
                        message.user_id,
                        message.role,
                        message.content,
                        message.timestamp,
                        json.dumps(message.sources) if message.sources else None,
                        json.dumps(message.topics) if message.topics else None,
                        message.query_type,
                    ),
                )

                message_id = cursor.lastrowid

                # Update session metadata in same transaction
                self._update_session_metadata_in_transaction(
                    conn, session_id, user_id, topics
                )

                log_info(
                    f"Added message to conversation memory: {session_id} ({query_type})"
                )
                return message_id

        except Exception as e:
            log_error(f"Failed to add message to conversation memory: {e}")
            return None

    def _update_session_metadata_in_transaction(
        self, conn: sqlite3.Connection, session_id: str, user_id: str, topics: List[str]
    ):
        """Update session metadata within an existing transaction"""
        try:
            # Get current session data
            cursor = conn.execute(
                """
                SELECT active_topics, message_count FROM sessions WHERE session_id = ?
            """,
                (session_id,),
            )

            result = cursor.fetchone()

            if result:
                # Update existing session
                current_topics = json.loads(result[0]) if result[0] else []
                message_count = result[1] + 1

                # Merge topics (keep recent ones)
                all_topics = current_topics + topics
                unique_topics = list(dict.fromkeys(all_topics))[
                    -20:
                ]  # Keep last 20 topics

                conn.execute(
                    """
                    UPDATE sessions 
                    SET last_activity = ?, message_count = ?, active_topics = ?
                    WHERE session_id = ?
                """,
                    (time.time(), message_count, json.dumps(unique_topics), session_id),
                )
            else:
                # Create new session
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, user_id, created_at, last_activity, message_count, active_topics)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        session_id,
                        user_id,
                        time.time(),
                        time.time(),
                        1,
                        json.dumps(topics),
                    ),
                )

        except Exception as e:
            log_error(f"Failed to update session metadata in transaction: {e}")

    def _update_session_metadata(
        self, session_id: str, user_id: str, topics: List[str]
    ):
        """Update session metadata with latest activity"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                self._update_session_metadata_in_transaction(
                    conn, session_id, user_id, topics
                )

        except Exception as e:
            log_error(f"Failed to update session metadata: {e}")

    def get_conversation_context(
        self, session_id: str, user_id: str
    ) -> ConversationContext:
        """Get conversation context for query resolution"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                # Get recent messages
                cursor = conn.execute(
                    """
                    SELECT role, content, topics, sources, query_type 
                    FROM messages 
                    WHERE session_id = ? AND user_id = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (session_id, user_id, self.config.context_window_size * 2),
                )

                messages = cursor.fetchall()

                # Extract context information
                recent_topics = []
                recent_sources = []
                last_user_query = None
                active_references = set()

                for role, content, topics_json, sources_json, query_type in messages:
                    if topics_json:
                        topics = json.loads(topics_json)
                        recent_topics.extend(topics)

                    if sources_json:
                        sources = json.loads(sources_json)
                        recent_sources.extend(sources)

                    if role == "user" and last_user_query is None:
                        last_user_query = content

                    # Extract potential references from content
                    refs = re.findall(r"\b[A-Z][a-zA-Z0-9_.-]*\b", content)
                    active_references.update(refs[:5])  # Limit to prevent noise

                # Remove duplicates and limit size
                recent_topics = list(dict.fromkeys(recent_topics))[-15:]
                recent_sources = recent_sources[-10:] if recent_sources else []

                return ConversationContext(
                    recent_topics=recent_topics,
                    recent_sources=recent_sources,
                    last_user_query=last_user_query,
                    conversation_summary=None,  # Could implement summarization later
                    active_references=active_references,
                )

        except Exception as e:
            log_error(f"Failed to get conversation context: {e}")
            return ConversationContext([], [], None, None, set())

    def get_conversation_history(
        self, session_id: str, user_id: str, limit: Optional[int] = None
    ) -> List[ConversationMessage]:
        """Get conversation history for a session"""
        try:
            limit = limit or self.config.context_window_size * 2

            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, session_id, user_id, role, content, timestamp, sources, topics, query_type
                    FROM messages 
                    WHERE session_id = ? AND user_id = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (session_id, user_id, limit),
                )

                messages = []
                for row in cursor.fetchall():
                    sources = json.loads(row[6]) if row[6] else None
                    topics = json.loads(row[7]) if row[7] else None

                    message = ConversationMessage(
                        id=row[0],
                        session_id=row[1],
                        user_id=row[2],
                        role=row[3],
                        content=row[4],
                        timestamp=row[5],
                        sources=sources,
                        topics=topics,
                        query_type=row[8],
                    )
                    messages.append(message)

                return list(reversed(messages))  # Return in chronological order

        except Exception as e:
            log_error(f"Failed to get conversation history: {e}")
            return []

    def clear_session(self, session_id: str, user_id: str):
        """Clear a conversation session"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "DELETE FROM messages WHERE session_id = ? AND user_id = ?",
                    (session_id, user_id),
                )
                conn.execute(
                    "DELETE FROM sessions WHERE session_id = ? AND user_id = ?",
                    (session_id, user_id),
                )
                conn.commit()
                log_info(f"Cleared conversation session: {session_id}")

        except Exception as e:
            log_error(f"Failed to clear conversation session: {e}")

    def cleanup_old_conversations(self, days_old: int = 30):
        """Clean up conversations older than specified days"""
        try:
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)

            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM messages WHERE timestamp < ?", (cutoff_time,)
                )
                deleted_messages = cursor.rowcount

                cursor = conn.execute(
                    "DELETE FROM sessions WHERE last_activity < ?", (cutoff_time,)
                )
                deleted_sessions = cursor.rowcount

                conn.commit()

                log_info(
                    f"Cleaned up {deleted_messages} old messages and {deleted_sessions} old sessions"
                )

        except Exception as e:
            log_error(f"Failed to cleanup old conversations: {e}")

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {}

            # File size
            if os.path.exists(self._db_path):
                size_bytes = os.path.getsize(self._db_path)
                stats["size_mb"] = size_bytes / (1024**2)
                stats["size_gb"] = size_bytes / (1024**3)
            else:
                stats["size_mb"] = 0
                stats["size_gb"] = 0

            # Message and session counts
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM messages")
                stats["total_messages"] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM sessions")
                stats["total_sessions"] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(DISTINCT user_id) FROM sessions")
                stats["unique_users"] = cursor.fetchone()[0]

            stats["enabled"] = self.config.enable_memory
            stats["max_size_gb"] = self.config.max_db_size_gb
            stats["context_window"] = self.config.context_window_size

            return stats

        except Exception as e:
            log_error(f"Failed to get database stats: {e}")
            return {"error": str(e)}


# Global memory manager instance
memory_manager = ConversationMemoryManager()
