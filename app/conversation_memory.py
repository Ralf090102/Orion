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

from app.utils import log_info, log_warning, log_error


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
        self.context_window_size = 7  # Number of message pairs to keep in context
        self.topic_retention_days = 30
        self.enable_memory = True
        self.compression_threshold = 1000  # Messages before compression

        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


class FollowUpDetector:
    """Detects follow-up questions and references using keyword matching"""

    def __init__(self):
        # Follow-up question patterns
        self.follow_up_patterns = [
            # Direct follow-ups
            r"\b(tell me more|more about|what else|anything else|continue|elaborate)\b",
            r"\b(more details|more information|additional|further|deeper)\b",
            r"\b(explain further|go deeper|expand on)\b",
            # Clarification requests
            r"\b(clarify|explain|what do you mean|can you elaborate)\b",
            r"\b(how so|why is that|what does that mean|in what way)\b",
            # Reference patterns
            r"\b(that|this|it|the above|previously|mentioned|discussed)\b",
            r"\b(the document|the file|the source|the information)\b",
            r"\b(from before|earlier|last time|previous)\b",
        ]

        # Question starters that often indicate follow-ups
        self.contextual_starters = [
            r"^(and|also|besides|additionally|furthermore)\b",
            r"^(what about|how about|speaking of)\b",
            r"^(regarding|concerning|about)\b",
        ]

        # Compile patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.follow_up_patterns + self.contextual_starters
        ]

    def detect_query_type(self, query: str, context: ConversationContext) -> QueryType:
        """Detect the type of query based on content and context"""
        query_lower = query.lower().strip()

        # Check for explicit follow-up patterns FIRST (highest priority)
        follow_up_patterns = [
            "tell me more",
            "more about",
            "what else",
            "anything else",
            "continue",
            "elaborate",
            "further",
            "additional",
            "more details",
            "some examples",
            "for example",
            "what are some",
        ]
        if any(pattern in query_lower for pattern in follow_up_patterns):
            return QueryType.FOLLOW_UP

        # Check for clarification patterns (most specific) - but only with context words
        clarification_patterns = [
            "what do you mean",
            "clarify that",
            "explain that",
            "what does that mean",
        ]
        if any(pattern in query_lower for pattern in clarification_patterns):
            return QueryType.CLARIFICATION

        # More specific clarification check - needs context reference
        if ("clarify" in query_lower or "explain" in query_lower) and any(
            ref in query_lower
            for ref in ["that", "this", "it", "what you said", "your answer"]
        ):
            return QueryType.CLARIFICATION

        # Check for reference patterns (pronouns and references)
        reference_patterns = [
            "that document",
            "this source",
            "mentioned earlier",
            "discussed before",
            "the above",
            "from before",
            "from our conversation",
        ]
        if any(pattern in query_lower for pattern in reference_patterns):
            return QueryType.REFERENCE

        # Simple pronouns only count as reference if we have recent context
        if context.recent_topics and any(
            ref in query_lower for ref in ["that", "this", "it"]
        ):
            return QueryType.REFERENCE

        # Check for contextual starters
        if query_lower.startswith(
            ("and ", "also ", "besides ", "furthermore ", "what about ", "how about ")
        ):
            return QueryType.FOLLOW_UP

        # Check if query contains topics from recent conversation (but be more selective)
        if context.recent_topics:
            # Only consider it a follow-up if multiple topic words match
            matching_topics = [
                topic for topic in context.recent_topics if topic.lower() in query_lower
            ]
            if len(matching_topics) >= 2:  # More selective - need 2+ topic matches
                return QueryType.FOLLOW_UP

        # Default to new topic for everything else
        return QueryType.NEW_TOPIC

    def extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text using simple keyword extraction"""
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

        # Return top topics (limit to prevent noise)
        return unique_topics[:10]


class ConversationMemoryManager:
    """Manages persistent conversation memory with SQLite storage"""

    def __init__(self, config: Optional[ConversationMemoryConfig] = None):
        self.config = config or ConversationMemoryConfig()
        self.follow_up_detector = FollowUpDetector()
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
            topics = self.follow_up_detector.extract_topics(content)
            query_type = None

            if role == "user":
                context = self.get_conversation_context(session_id, user_id)
                query_type = self.follow_up_detector.detect_query_type(
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
