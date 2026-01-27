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
        
        # Check if messages table exists and needs migration
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Check if the table has the new schema
            cursor.execute("PRAGMA table_info(messages)")
            columns = {col[1] for col in cursor.fetchall()}
            needs_migration = "message_id" not in columns or "parent_id" not in columns
            
            if needs_migration:
                logger.info("Existing database detected with old schema - recreating tables")
                # Drop old tables (CASCADE will handle messages)
                cursor.execute("DROP TABLE IF EXISTS messages")
                cursor.execute("DROP TABLE IF EXISTS sessions")
                conn.commit()
        
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
        
        # Create messages table with parent-child relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE NOT NULL,
                session_id TEXT NOT NULL,
                parent_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tokens INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                FOREIGN KEY (parent_id) REFERENCES messages(message_id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON messages(session_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_parent
            ON messages(parent_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_active
            ON messages(session_id, is_active)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_updated 
            ON sessions(updated_at)
        """)
        
        conn.commit()
        conn.close()
        logger.debug("SQLite database schema initialized")

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
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        tokens: int = 0,
        parent_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Add a message to session history with parent-child tracking.

        Args:
            session_id: Session identifier
            role: "user" or "assistant"
            content: Message content
            tokens: Token count (optional)
            parent_id: ID of parent message for branching (optional)
            message_id: Custom message ID (generates UUID if None)

        Returns:
            Message ID if added, None if session not found
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        # Generate message ID if not provided
        if message_id is None:
            message_id = str(uuid.uuid4())

        message = {
            "message_id": message_id,
            "role": role,
            "content": content,
            "tokens": tokens,
            "timestamp": datetime.now().isoformat(),
            "parent_id": parent_id,
            "is_active": True,
        }

        session.messages.append(message)
        session.updated_at = datetime.now().isoformat()

        if self.persist_to_disk:
            self._add_message_to_db(session_id, message)
            self._update_session_timestamp(session_id)

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
        """Add a message to the database with parent tracking."""
        if not self.persist_to_disk:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO messages 
                (message_id, session_id, parent_id, role, content, tokens, timestamp, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message["message_id"],
                session_id,
                message.get("parent_id"),
                message["role"],
                message["content"],
                message.get("tokens", 0),
                message["timestamp"],
                message.get("is_active", True)
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
                    SELECT message_id, role, content, tokens, timestamp, parent_id, is_active
                    FROM messages 
                    WHERE session_id = ?
                    ORDER BY id ASC
                """, (session_id,))
                
                messages = [
                    {
                        "message_id": row[0],
                        "role": row[1],
                        "content": row[2],
                        "tokens": row[3],
                        "timestamp": row[4],
                        "parent_id": row[5],
                        "is_active": bool(row[6])
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

    def delete_message_and_children(self, session_id: str, message_id: str) -> bool:
        """
        Delete a message and all its children (cascading delete).
        Useful for retry/edit functionality.

        Args:
            session_id: Session identifier
            message_id: Message ID to delete

        Returns:
            True if deleted, False if message not found
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False

        # Find all descendant message IDs (recursive)
        def get_all_descendants(msg_id: str) -> set[str]:
            descendants = {msg_id}
            children = [m for m in session.messages if m.get("parent_id") == msg_id]
            for child in children:
                descendants.update(get_all_descendants(child["message_id"]))
            return descendants

        to_delete = get_all_descendants(message_id)

        # Remove from in-memory session
        session.messages = [m for m in session.messages if m["message_id"] not in to_delete]
        session.updated_at = datetime.now().isoformat()

        # Remove from database (CASCADE will handle children)
        if self.persist_to_disk:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Delete the message (CASCADE handles children)
                cursor.execute("""
                    DELETE FROM messages 
                    WHERE message_id = ? AND session_id = ?
                """, (message_id, session_id))
                
                deleted_count = cursor.rowcount
                
                # Update message count
                cursor.execute("""
                    UPDATE sessions 
                    SET message_count = (
                        SELECT COUNT(*) FROM messages WHERE session_id = ?
                    )
                    WHERE session_id = ?
                """, (session_id, session_id))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Deleted message {message_id} and {len(to_delete)} total message(s)")
                return deleted_count > 0
                
            except Exception as e:
                logger.error(f"Failed to delete message {message_id}: {e}")
                return False

        logger.info(f"Deleted {len(to_delete)} message(s) from session {session_id}")
        return True

    def create_branch(
        self, 
        session_id: str, 
        parent_id: str, 
        role: str, 
        content: str, 
        tokens: int = 0,
        deactivate_siblings: bool = True
    ) -> Optional[str]:
        """
        Create a new branch (alternative response) from a parent message.
        Used for retry/edit functionality.

        Args:
            session_id: Session identifier
            parent_id: Parent message ID to branch from
            role: "user" or "assistant"
            content: Message content
            tokens: Token count (optional)
            deactivate_siblings: Mark sibling branches as inactive (default: True)

        Returns:
            New message ID if created, None if failed
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        # Verify parent exists
        parent_exists = any(m["message_id"] == parent_id for m in session.messages)
        if not parent_exists:
            logger.warning(f"Parent message not found: {parent_id}")
            return None

        # Deactivate sibling branches if requested
        if deactivate_siblings:
            self._deactivate_siblings(session_id, parent_id)

        # Create new message as child of parent
        new_message_id = self.add_message(
            session_id=session_id,
            role=role,
            content=content,
            tokens=tokens,
            parent_id=parent_id
        )

        logger.info(f"Created branch: {new_message_id} from parent {parent_id}")
        return new_message_id

    def _deactivate_siblings(self, session_id: str, parent_id: str) -> None:
        """Mark all messages with the same parent as inactive."""
        session = self.get_session(session_id)
        if not session:
            return

        # Deactivate siblings in memory
        for message in session.messages:
            if message.get("parent_id") == parent_id:
                message["is_active"] = False

        # Deactivate in database
        if self.persist_to_disk:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE messages 
                    SET is_active = 0
                    WHERE session_id = ? AND parent_id = ?
                """, (session_id, parent_id))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to deactivate siblings: {e}")

    def get_active_branch_messages(self, session_id: str) -> list[ConversationMessage]:
        """
        Get only messages from the active conversation branch.

        Args:
            session_id: Session identifier

        Returns:
            List of ConversationMessage objects in active branch with metadata
        """
        session = self.get_session(session_id)
        if not session:
            return []

        # Build conversation tree and extract active path
        active_messages = []
        
        # Start with root messages (no parent)
        current_id = None
        
        while True:
            # Find active child of current message
            candidates = [
                m for m in session.messages 
                if m.get("parent_id") == current_id and m.get("is_active", True)
            ]
            
            if not candidates:
                break
            
            # Take first active candidate (should only be one)
            next_msg = candidates[0]
            msg = ConversationMessage(
                role=next_msg["role"],
                content=next_msg["content"],
                tokens=next_msg.get("tokens", 0)
            )
            # Add metadata fields for frontend use
            msg.message_id = next_msg.get("message_id")
            msg.timestamp = next_msg.get("timestamp", "")
            active_messages.append(msg)
            current_id = next_msg["message_id"]

        return active_messages

    def get_message_branches(self, session_id: str, parent_id: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Get all alternative branches from a parent message.

        Args:
            session_id: Session identifier
            parent_id: Parent message ID (None for root messages)

        Returns:
            List of message dictionaries that are children of parent
        """
        session = self.get_session(session_id)
        if not session:
            return []

        return [
            m for m in session.messages 
            if m.get("parent_id") == parent_id
        ]

    def switch_branch(self, session_id: str, message_id: str) -> bool:
        """
        Switch to a different conversation branch by activating a message.

        Args:
            session_id: Session identifier
            message_id: Message ID to activate

        Returns:
            True if switched, False if message not found
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False

        # Find the message
        target_message = next((m for m in session.messages if m["message_id"] == message_id), None)
        if not target_message:
            logger.warning(f"Message not found: {message_id}")
            return False

        parent_id = target_message.get("parent_id")

        # Deactivate all siblings (messages with same parent)
        self._deactivate_siblings(session_id, parent_id)

        # Activate target message
        target_message["is_active"] = True

        # Update in database
        if self.persist_to_disk:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE messages 
                    SET is_active = 1
                    WHERE message_id = ? AND session_id = ?
                """, (message_id, session_id))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to switch branch: {e}")
                return False

        logger.info(f"Switched to branch: {message_id}")
        return True

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
            
            cursor.execute("SELECT COUNT(*) FROM messages WHERE is_active = 1")
            active_messages = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(tokens) FROM messages")
            total_tokens = cursor.fetchone()[0] or 0
            
            # Database file size
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
            
            conn.close()
            
            return {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "active_messages": active_messages,
                "inactive_branches": total_messages - active_messages,
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
