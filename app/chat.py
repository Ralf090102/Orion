"""
Chat session management with context and memory.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from app.utils import log_info


@dataclass
class ChatMessage:
    """Represents a single chat message."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    sources: Optional[List[Dict]] = None  # For RAG citations


@dataclass
class ChatSession:
    """Manages a conversation session with context."""

    session_id: str
    user_id: str
    messages: List[ChatMessage]
    max_context_messages: int = 10  # Limit context to last N exchanges
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def add_message(
        self, role: str, content: str, sources: Optional[List[Dict]] = None
    ):
        """Add a message to the session."""
        message = ChatMessage(
            role=role, content=content, timestamp=time.time(), sources=sources
        )
        self.messages.append(message)

        # Trim context if too long
        if (
            len(self.messages) > self.max_context_messages * 2
        ):  # *2 for user+assistant pairs
            # Keep first message (system prompt if any) + recent messages
            recent_messages = self.messages[-self.max_context_messages :]
            self.messages = self.messages[:1] + recent_messages
            log_info(f"Trimmed chat context to {len(self.messages)} messages")

    def get_context_for_llm(self) -> str:
        """Format recent messages as context for the LLM."""
        if not self.messages:
            return ""

        context_parts = []
        for msg in self.messages[-6:]:  # Last 3 exchanges
            role_label = "Human" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role_label}: {msg.content}")

        return "\n\n".join(context_parts)

    def get_last_user_message(self) -> Optional[str]:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None


class ChatSessionManager:
    """Manages multiple chat sessions."""

    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> current session_id

    def get_or_create_session(
        self, user_id: str, session_id: Optional[str] = None
    ) -> ChatSession:
        """Get existing session or create new one."""
        if session_id is None:
            # Use user's current session or create new one
            session_id = self.user_sessions.get(user_id)
            if session_id is None or session_id not in self.sessions:
                session_id = f"{user_id}_{int(time.time())}"

        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(
                session_id=session_id, user_id=user_id, messages=[]
            )
            log_info(f"Created new chat session: {session_id}")

        # Update user's current session
        self.user_sessions[user_id] = session_id
        return self.sessions[session_id]

    def clear_session(self, session_id: str):
        """Clear a specific session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            # Update user_sessions mapping
            for user_id, curr_session in list(self.user_sessions.items()):
                if curr_session == session_id:
                    del self.user_sessions[user_id]
            log_info(f"Cleared chat session: {session_id}")


# Global session manager instance
chat_manager = ChatSessionManager()
