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
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT,
                message_count INTEGER DEFAULT 0
            )
        """)
        
        # Create messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tokens INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON messages(session_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_updated 
            ON sessions(updated_at)
        """)
        
        conn.commit()
        conn.close()
        logger.debug("SQLite database schema initialized")

    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new chat session.

        Args:
            session_id: Optional custom session ID (generates UUID if None)

        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        now = datetime.now().isoformat()
        session = ChatSession(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            messages=[],
            metadata={},
        )

        self.sessions[session_id] = session

        if self.persist_to_disk:
            self._save_session_to_db(session)

        logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ChatSession or None if not found
        """
        return self.sessions.get(session_id)

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
        self, session_id: str, role: str, content: str, tokens: int = 0
    ) -> bool:
        """
        Add a message to session history.

        Args:
            session_id: Session identifier
            role: "user" or "assistant"
            content: Message content
            tokens: Token count (optional)

        Returns:
            True if added, False if session not found
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False

        message = {
            "role": role,
            "content": content,
            "tokens": tokens,
            "timestamp": datetime.now().isoformat(),
        }

        session.messages.append(message)
        session.updated_at = datetime.now().isoformat()

        if self.persist_to_disk:
            self._add_message_to_db(session_id, message)
            self._update_session_timestamp(session_id)

        logger.debug(f"Added {role} message to session {session_id}")
        return True

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
        """Save session to SQLite database."""
        if not self.persist_to_disk:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, created_at, updated_at, metadata, message_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.created_at,
                session.updated_at,
                json.dumps(session.metadata),
                len(session.messages)
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Saved session to database: {session.session_id}")
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")

    def _add_message_to_db(self, session_id: str, message: dict[str, Any]) -> None:
        """Add a message to the database."""
        if not self.persist_to_disk:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, tokens, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                message["role"],
                message["content"],
                message.get("tokens", 0),
                message["timestamp"]
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

    def _load_sessions_from_db(self) -> None:
        """Load all sessions from SQLite database."""
        if not self.persist_to_disk or not self.db_path.exists():
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load sessions
            cursor.execute("""
                SELECT session_id, created_at, updated_at, metadata 
                FROM sessions
            """)
            
            sessions_data = cursor.fetchall()
            
            for session_id, created_at, updated_at, metadata_json in sessions_data:
                # Load messages for this session
                cursor.execute("""
                    SELECT role, content, tokens, timestamp 
                    FROM messages 
                    WHERE session_id = ?
                    ORDER BY id ASC
                """, (session_id,))
                
                messages = [
                    {
                        "role": row[0],
                        "content": row[1],
                        "tokens": row[2],
                        "timestamp": row[3]
                    }
                    for row in cursor.fetchall()
                ]
                
                session = ChatSession(
                    session_id=session_id,
                    created_at=created_at,
                    updated_at=updated_at,
                    messages=messages,
                    metadata=json.loads(metadata_json) if metadata_json else {}
                )
                
                self.sessions[session_id] = session
            
            conn.close()
            
            if sessions_data:
                logger.info(f"Loaded {len(sessions_data)} session(s) from database")
                
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
