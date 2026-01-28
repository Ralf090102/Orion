"""
Session Manager for Orion Chat Mode

Manages conversation sessions with SQLite persistence.
Each session has a unique ID and stores conversation history.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from src.generation.prompt_builder import ConversationMessage

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    """Represents a chat session with metadata."""

    session_id: str
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatSession":
        """Create session from dictionary."""
        return cls(**data)


class SessionManager:
    """
    Manages chat sessions with SQLite persistence.

    Features:
        - Create/retrieve/delete sessions
        - In-memory session cache
        - SQLite database persistence
        - Session metadata tracking
        - Automatic cleanup of old sessions
        - Session expiry management
    """

    def __init__(
        self, persist_to_disk: bool = False, storage_dir: Optional[Path] = None,
        session_expiry_days: int = 7, auto_cleanup: bool = True
    ):
        """
        Initialize session manager.

        Args:
            persist_to_disk: Save sessions to SQLite database
            storage_dir: Directory for database file (default: ./data/chat-data)
            session_expiry_days: Days before sessions expire (default: 7)
            auto_cleanup: Automatically cleanup expired sessions on init (default: True)
        """
        self.persist_to_disk = persist_to_disk
        self.storage_dir = storage_dir or Path("./data/chat-data")
        self.session_expiry_days = session_expiry_days
        self.sessions: dict[str, ChatSession] = {}
        self.db_path: Optional[Path] = None
        
        if self.persist_to_disk:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = self.storage_dir / "sessions.db"
            self._init_database()
            self._load_sessions_from_db()
            
            # Auto cleanup on startup
            if auto_cleanup:
                cleaned = self.cleanup_old_sessions(max_age_days=session_expiry_days)
                if cleaned > 0:
                    logger.info(f"Auto-cleanup: removed {cleaned} expired session(s)")
            
            logger.info(f"Session manager initialized with SQLite: {self.db_path}")
        else:
            logger.info("Session manager initialized (in-memory only)")

    def _init_database(self) -> None:
        """Initialize SQLite database schema with future-proof fields."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table with additional metadata columns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT DEFAULT 'New Chat',
                message_count INTEGER DEFAULT 0,
                last_model TEXT,
                metadata_json TEXT
            )
        """)
        
        # Create messages table with UUID, metadata, and branching support
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                tokens INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                
                -- Metadata fields
                model TEXT,
                rag_triggered BOOLEAN DEFAULT 0,
                processing_time_ms INTEGER,
                metadata_json TEXT,
                
                -- Future branching support (inactive for now)
                parent_id TEXT,
                is_active BOOLEAN DEFAULT 1,
                edited_from TEXT,
                version INTEGER DEFAULT 1,
                
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                FOREIGN KEY (parent_id) REFERENCES messages(id) ON DELETE CASCADE
            )
        """)
        
        # Create sources table for RAG citations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT NOT NULL,
                citation TEXT,
                content TEXT,
                score REAL,
                rank INTEGER,
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON messages(session_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_rag 
            ON messages(rag_triggered) WHERE rag_triggered = 1
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_parent 
            ON messages(parent_id) WHERE parent_id IS NOT NULL
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sources_message 
            ON message_sources(message_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_updated 
            ON sessions(updated_at DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_title 
            ON sessions(title)
        """)
        
        conn.commit()
        conn.close()
        logger.debug("SQLite database schema initialized (future-proof)")

    def create_session(
        self,
        session_id: Optional[str] = None,
        user: Optional[str] = None,
        topic: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Create a new chat session.

        Args:
            session_id: Optional custom session ID (generates UUID if None)
            user: Optional user identifier
            topic: Optional topic/category for the session
            metadata: Optional additional metadata dictionary

        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        now = datetime.now().isoformat()
        
        # Build metadata from parameters
        session_metadata = metadata.copy() if metadata else {}
        if user:
            session_metadata["user"] = user
        if topic:
            session_metadata["topic"] = topic
        
        session = ChatSession(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            messages=[],
            metadata=session_metadata,
        )

        self.sessions[session_id] = session

        if self.persist_to_disk:
            self._save_session_to_db(session)

        logger.info(f"Created session: {session_id} (user={user}, topic={topic})")
        return session_id

    def get_session(self, session_id: str, reload_from_db: bool = False) -> Optional[ChatSession]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier
            reload_from_db: Force reload from database (default: False, use cache)

        Returns:
            ChatSession or None if not found
        """
        # If reload requested and we have disk persistence, load from DB
        if reload_from_db and self.persist_to_disk and self.db_path:
            session = self._load_session_from_db(session_id)
            if session:
                self.sessions[session_id] = session  # Update cache
            return session
        
        # Check cache first (fast path)
        session = self.sessions.get(session_id)
        
        # If not in cache but we have persistence, try loading from DB
        if session is None and self.persist_to_disk and self.db_path:
            session = self._load_session_from_db(session_id)
            if session:
                self.sessions[session_id] = session  # Add to cache
                logger.debug(f"Auto-loaded session {session_id} from DB (cache miss)")
        
        return session

    def invalidate_cache(self, session_id: str) -> None:
        """
        Remove a session from the in-memory cache.
        
        Use this after external modifications to force next read to reload from DB.
        The session will be automatically reloaded on next get_session() call if
        persist_to_disk is enabled.
        
        Args:
            session_id: Session identifier to invalidate
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.debug(f"Invalidated cache for session: {session_id}")

    def reload_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Force reload a session from database, bypassing cache.
        
        This is equivalent to get_session(session_id, reload_from_db=True) but more explicit.
        Use when you need to ensure you have the latest data from disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Reloaded ChatSession or None if not found
        """
        return self.get_session(session_id, reload_from_db=True)

    def delete_last_messages(self, session_id: str, count: int = 2) -> bool:
        """
        Delete the last N messages from a session (for retry functionality).
        
        Args:
            session_id: Session identifier
            count: Number of messages to delete from the end (default 2 for user+assistant pair)
            
        Returns:
            True if deleted, False if session not found or not enough messages
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        if len(session.messages) < count:
            logger.warning(f"Not enough messages to delete (have {len(session.messages)}, need {count})")
            return False
        
        # Get the message IDs to delete
        messages_to_delete = session.messages[-count:]
        message_ids = [msg.get("id") for msg in messages_to_delete if msg.get("id")]
        
        # Delete from database first
        if self.persist_to_disk and message_ids:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Delete messages and their sources
                placeholders = ",".join(["?"] * len(message_ids))
                cursor.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", message_ids)
                cursor.execute(f"DELETE FROM message_sources WHERE message_id IN ({placeholders})", message_ids)
                
                # Update session metadata
                cursor.execute(
                    """
                    UPDATE sessions 
                    SET message_count = message_count - ?, 
                        updated_at = ?
                    WHERE session_id = ?
                    """,
                    (count, datetime.now().isoformat(), session_id)
                )
                
                conn.commit()
                conn.close()
                
                logger.info(f"Deleted {count} messages from session {session_id} in database")
            except Exception as e:
                logger.error(f"Failed to delete messages from database: {e}")
                return False
        
        # Update in-memory session
        session.messages = session.messages[:-count]
        session.message_count = len(session.messages)
        session.updated_at = datetime.now()
        
        # Invalidate cache to force reload on next access
        self.invalidate_cache(session_id)
        
        logger.info(f"Deleted last {count} messages from session {session_id}")
        return True

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        if session_id not in self.sessions:
            return False

        del self.sessions[session_id]

        if self.persist_to_disk:
            self._delete_session_from_db(session_id)

        logger.info(f"Deleted session: {session_id}")
        return True

    def delete_all_sessions(self) -> int:
        """
        Delete all sessions.

        Returns:
            Number of sessions deleted
        """
        count = len(self.sessions)

        if count == 0:
            return 0

        # Clear in-memory sessions
        self.sessions.clear()

        # Clear database if persisting
        if self.persist_to_disk:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM sessions")
                cursor.execute("DELETE FROM messages")
                
                conn.commit()
                conn.close()
                logger.info(f"Deleted all {count} sessions from database")
            except Exception as e:
                logger.error(f"Failed to delete all sessions from database: {e}")

        logger.info(f"Deleted all {count} session(s)")
        return count

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens: int = 0,
        model: Optional[str] = None,
        rag_triggered: bool = False,
        processing_time_ms: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        sources: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Add a message to session history with optional metadata.

        Args:
            session_id: Session identifier
            role: "user", "assistant", or "system"
            content: Message content
            tokens: Token count (optional)
            model: LLM model used (optional)
            rag_triggered: Whether RAG retrieval was triggered (optional)
            processing_time_ms: Processing time in milliseconds (optional)
            metadata: Additional metadata dictionary (optional)
            parent_id: Parent message ID for branching (optional)
            sources: List of source dictionaries for RAG citations (optional)

        Returns:
            Message ID if added, None if session not found
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        # Generate UUID for message
        message_id = str(uuid.uuid4())
        
        message = {
            "id": message_id,
            "role": role,
            "content": content,
            "tokens": tokens,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "rag_triggered": rag_triggered,
            "processing_time_ms": processing_time_ms,
            "metadata": metadata or {},
            "parent_id": parent_id,
            "is_active": True,
            "version": 1,
            "sources": sources or [],
        }

        session.messages.append(message)
        session.updated_at = datetime.now().isoformat()

        if self.persist_to_disk:
            self._add_message_to_db(session_id, message)
            self._update_session_timestamp(session_id)
            
            # Add sources if provided
            if sources:
                self._add_message_sources_to_db(message_id, sources)

        logger.debug(f"Added {role} message {message_id} to session {session_id}")
        return message_id

    def get_messages(self, session_id: str) -> list[ConversationMessage]:
        """
        Get conversation messages for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of ConversationMessage objects
        """
        session = self.get_session(session_id)
        if not session:
            return []

        return [
            ConversationMessage(
                role=msg["role"], content=msg["content"], tokens=msg.get("tokens", 0)
            )
            for msg in session.messages
        ]

    def clear_session_messages(self, session_id: str) -> bool:
        """
        Clear all messages from a session (keeps session metadata).

        Args:
            session_id: Session identifier

        Returns:
            True if cleared, False if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session.messages.clear()
        session.updated_at = datetime.now().isoformat()

        if self.persist_to_disk:
            self._clear_messages_in_db(session_id)
            self._update_session_timestamp(session_id)

        logger.info(f"Cleared messages in session: {session_id}")
        return True

    def get_most_recent_session(self) -> Optional[ChatSession]:
        """
        Get the most recently updated session.

        Returns:
            Most recent ChatSession or None if no sessions exist
        """
        if not self.sessions:
            return None

        # Sort sessions by updated_at timestamp (most recent first)
        sorted_sessions = sorted(
            self.sessions.values(),
            key=lambda s: datetime.fromisoformat(s.updated_at),
            reverse=True
        )

        return sorted_sessions[0] if sorted_sessions else None

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions with metadata.

        Returns:
            List of session summaries
        """
        return [
            {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": len(session.messages),
                "metadata": session.metadata,
            }
            for session in self.sessions.values()
        ]

    def update_session_metadata(
        self, session_id: str, metadata: dict[str, Any]
    ) -> bool:
        """
        Update session metadata.

        Args:
            session_id: Session identifier
            metadata: Metadata dictionary to merge

        Returns:
            True if updated, False if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session.metadata.update(metadata)
        session.updated_at = datetime.now().isoformat()

        if self.persist_to_disk:
            self._update_session_metadata_in_db(session_id, session.metadata)
            self._update_session_timestamp(session_id)

        return True

    # ========== SQLite PERSISTENCE METHODS ==========

    def _save_session_to_db(self, session: ChatSession) -> None:
        """Save session to SQLite database with new schema fields."""
        if not self.persist_to_disk:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract title and last_model from metadata
            title = session.metadata.get("title", "New Chat")
            last_model = session.metadata.get("last_model")
            
            cursor.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, created_at, updated_at, title, message_count, last_model, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.created_at,
                session.updated_at,
                title,
                len(session.messages),
                last_model,
                json.dumps(session.metadata)
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Saved session to database: {session.session_id}")
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")

    def _add_message_to_db(self, session_id: str, message: dict[str, Any]) -> None:
        """Add a message to the database with new schema fields."""
        if not self.persist_to_disk:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO messages (
                    id, session_id, role, content, tokens, timestamp,
                    model, rag_triggered, processing_time_ms, metadata_json,
                    parent_id, is_active, version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message["id"],
                session_id,
                message["role"],
                message["content"],
                message.get("tokens", 0),
                message["timestamp"],
                message.get("model"),
                message.get("rag_triggered", False),
                message.get("processing_time_ms"),
                json.dumps(message.get("metadata", {})),
                message.get("parent_id"),
                message.get("is_active", True),
                message.get("version", 1)
            ))
            
            # Update message count
            cursor.execute("""
                UPDATE sessions 
                SET message_count = message_count + 1
                WHERE session_id = ?
            """, (session_id,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to add message to session {session_id}: {e}")

    def _add_message_sources_to_db(self, message_id: str, sources: list[dict[str, Any]]) -> None:
        """Add message sources (RAG citations) to the database."""
        if not self.persist_to_disk or not sources:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for rank, source in enumerate(sources):
                cursor.execute("""
                    INSERT INTO message_sources (message_id, citation, content, score, rank)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    message_id,
                    source.get("citation", ""),
                    source.get("content", ""),
                    source.get("score", 0.0),
                    rank
                ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Added {len(sources)} sources for message {message_id}")
        except Exception as e:
            logger.error(f"Failed to add sources for message {message_id}: {e}")

    def _clear_messages_in_db(self, session_id: str) -> None:
        """Clear all messages for a session in the database."""
        if not self.persist_to_disk:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("""
                UPDATE sessions 
                SET message_count = 0
                WHERE session_id = ?
            """, (session_id,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to clear messages for session {session_id}: {e}")

    def _update_session_timestamp(self, session_id: str) -> None:
        """Update session's updated_at timestamp."""
        if not self.persist_to_disk:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE sessions 
                SET updated_at = ?
                WHERE session_id = ?
            """, (datetime.now().isoformat(), session_id))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update timestamp for session {session_id}: {e}")

    def _update_session_metadata_in_db(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Update session metadata in database."""
        if not self.persist_to_disk:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE sessions 
                SET metadata = ?
                WHERE session_id = ?
            """, (json.dumps(metadata), session_id))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update metadata for session {session_id}: {e}")

    def _delete_session_from_db(self, session_id: str) -> None:
        """Delete session from database (CASCADE deletes messages too)."""
        if not self.persist_to_disk:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")

    def _load_session_from_db(self, session_id: str) -> Optional[ChatSession]:
        """Load a single session from SQLite database with new schema."""
        if not self.persist_to_disk or not self.db_path.exists():
            return None

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load session metadata
            cursor.execute("""
                SELECT session_id, created_at, updated_at, title, last_model, metadata_json 
                FROM sessions
                WHERE session_id = ?
            """, (session_id,))
            
            session_data = cursor.fetchone()
            if not session_data:
                conn.close()
                return None
            
            session_id, created_at, updated_at, title, last_model, metadata_json = session_data
            
            # Load messages for this session with new fields
            cursor.execute("""
                SELECT id, role, content, tokens, timestamp,
                       model, rag_triggered, processing_time_ms, metadata_json,
                       parent_id, is_active, version
                FROM messages 
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
            
            messages = []
            message_rows = cursor.fetchall()
            
            for row in message_rows:
                msg_id, role, content, tokens, timestamp, model, rag_triggered, processing_time_ms, msg_metadata_json, parent_id, is_active, version = row
                
                # Load sources for this message
                cursor.execute("""
                    SELECT citation, content, score, rank
                    FROM message_sources
                    WHERE message_id = ?
                    ORDER BY rank ASC
                """, (msg_id,))
                
                sources = [
                    {
                        "citation": src[0],
                        "content": src[1],
                        "score": src[2],
                        "rank": src[3]
                    }
                    for src in cursor.fetchall()
                ]
                
                messages.append({
                    "id": msg_id,
                    "role": role,
                    "content": content,
                    "tokens": tokens,
                    "timestamp": timestamp,
                    "model": model,
                    "rag_triggered": bool(rag_triggered),
                    "processing_time_ms": processing_time_ms,
                    "metadata": json.loads(msg_metadata_json) if msg_metadata_json else {},
                    "parent_id": parent_id,
                    "is_active": bool(is_active),
                    "version": version,
                    "sources": sources
                })
            
            conn.close()
            
            # Merge title and last_model into metadata
            metadata = json.loads(metadata_json) if metadata_json else {}
            metadata["title"] = title
            if last_model:
                metadata["last_model"] = last_model
            
            session = ChatSession(
                session_id=session_id,
                created_at=created_at,
                updated_at=updated_at,
                messages=messages,
                metadata=metadata
            )
            
            logger.debug(f"Loaded session {session_id} from database")
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id} from database: {e}")
            return None

    def _load_sessions_from_db(self) -> None:
        """Load all sessions from SQLite database using updated schema."""
        if not self.persist_to_disk or not self.db_path.exists():
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all session IDs
            cursor.execute("SELECT session_id FROM sessions")
            session_ids = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            # Use _load_session_from_db for each session (reuses logic)
            for session_id in session_ids:
                session = self._load_session_from_db(session_id)
                if session:
                    self.sessions[session_id] = session
            
            if session_ids:
                logger.info(f"Loaded {len(session_ids)} session(s) from database")
                
        except Exception as e:
            logger.error(f"Failed to load sessions from database: {e}")

    def cleanup_old_sessions(self, max_age_days: int = 7) -> int:
        """
        Delete sessions older than specified days.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of sessions deleted
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        cutoff_iso = cutoff.isoformat()
        to_delete = []

        for session_id, session in self.sessions.items():
            updated_at = datetime.fromisoformat(session.updated_at)
            if updated_at < cutoff:
                to_delete.append(session_id)

        # Delete from database first (if persisting)
        if self.persist_to_disk and to_delete:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM sessions 
                    WHERE updated_at < ?
                """, (cutoff_iso,))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to cleanup old sessions from database: {e}")

        # Delete from memory
        for session_id in to_delete:
            if session_id in self.sessions:
                del self.sessions[session_id]

        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old session(s)")

        return len(to_delete)

    def get_database_stats(self) -> dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database stats
        """
        if not self.persist_to_disk:
            return {
                "total_sessions": len(self.sessions),
                "total_messages": sum(len(s.messages) for s in self.sessions.values()),
                "persistence": "in-memory",
            }

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(tokens) FROM messages")
            total_tokens = cursor.fetchone()[0] or 0
            
            # Database file size
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
            
            conn.close()
            
            return {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "total_tokens": total_tokens,
                "database_size_mb": round(db_size_mb, 2),
                "database_path": str(self.db_path),
                "persistence": "sqlite",
            }
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}


# Global session manager instance (singleton pattern)
_session_manager: Optional[SessionManager] = None


def get_session_manager(
    persist_to_disk: bool = False, 
    storage_dir: Optional[Path] = None,
    session_expiry_days: int = 7,
    auto_cleanup: bool = True
) -> SessionManager:
    """
    Get or create the global session manager instance.

    Args:
        persist_to_disk: Save sessions to SQLite database
        storage_dir: Directory for database file (default: ./data/chat-data)
        session_expiry_days: Days before sessions expire (default: 7)
        auto_cleanup: Automatically cleanup expired sessions on init (default: True)

    Returns:
        SessionManager instance
    """
    global _session_manager

    if _session_manager is None:
        _session_manager = SessionManager(
            persist_to_disk=persist_to_disk, 
            storage_dir=storage_dir,
            session_expiry_days=session_expiry_days,
            auto_cleanup=auto_cleanup
        )

    return _session_manager


def reset_session_manager() -> None:
    """Reset the global session manager (useful for testing)."""
    global _session_manager
    _session_manager = None
