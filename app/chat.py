"""
Chat session management with enhanced conversation memory and context.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from app.utils import log_info
from app.conversation_memory import memory_manager, ConversationContext, QueryType


@dataclass
class ChatMessage:
    """Represents a single chat message."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    sources: Optional[List[Dict]] = None  # For RAG citations
    query_type: Optional[str] = None  # For conversation context


@dataclass
class ChatSession:
    """Manages a conversation session with enhanced context and memory."""

    session_id: str
    user_id: str
    messages: List[ChatMessage]
    max_context_messages: int = 7  # Configurable context window
    created_at: float = None
    enable_memory: bool = True  # Can disable memory per session

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

        # Load conversation history from persistent storage
        if self.enable_memory:
            self._load_conversation_history()

    def _load_conversation_history(self):
        """Load recent conversation history from memory manager"""
        try:
            history = memory_manager.get_conversation_history(
                self.session_id, self.user_id, self.max_context_messages * 2
            )

            # Convert to ChatMessage format
            for msg in history:
                chat_msg = ChatMessage(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    sources=msg.sources,
                    query_type=msg.query_type,
                )
                self.messages.append(chat_msg)

        except Exception as e:
            log_info(f"Could not load conversation history: {e}")

    def add_message(
        self, role: str, content: str, sources: Optional[List[Dict]] = None
    ):
        """Add a message to the session with enhanced memory tracking."""
        message = ChatMessage(
            role=role, content=content, timestamp=time.time(), sources=sources
        )
        self.messages.append(message)

        # Add to persistent memory
        if self.enable_memory:
            try:
                memory_manager.add_message(
                    self.session_id, self.user_id, role, content, sources
                )
            except Exception as e:
                log_info(f"Could not save message to memory: {e}")

        # Trim context if too long (keep in-memory messages manageable)
        if len(self.messages) > self.max_context_messages * 2:
            # Keep first message (system prompt if any) + recent messages
            recent_messages = self.messages[-self.max_context_messages :]
            self.messages = self.messages[:1] + recent_messages
            log_info(f"Trimmed chat context to {len(self.messages)} messages")

    def get_context_for_llm(self) -> str:
        """Format recent messages as context for the LLM."""
        if not self.messages:
            return ""

        context_parts = []
        # Use configurable context window
        recent_messages = self.messages[-(self.max_context_messages * 2) :]

        for msg in recent_messages:
            role_label = "Human" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role_label}: {msg.content}")

        return "\n\n".join(context_parts)

    def get_conversation_context(self) -> ConversationContext:
        """Get enhanced conversation context for query resolution"""
        if self.enable_memory:
            return memory_manager.get_conversation_context(
                self.session_id, self.user_id
            )
        else:
            # Fallback: create basic context from in-memory messages
            topics = []
            sources = []
            last_query = None

            for msg in self.messages:
                if msg.sources:
                    sources.extend(msg.sources)
                if msg.role == "user":
                    last_query = msg.content

            return ConversationContext(
                recent_topics=topics,
                recent_sources=sources[-10:],
                last_user_query=last_query,
                conversation_summary=None,
                active_references=set(),
            )

    def get_last_user_message(self) -> Optional[str]:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def detect_query_type(self, query: str) -> QueryType:
        """Detect if the query is a follow-up or new topic"""
        if self.enable_memory:
            context = self.get_conversation_context()
            return memory_manager.follow_up_detector.detect_query_type(query, context)
        else:
            # Simple fallback detection
            follow_up_indicators = [
                "more",
                "also",
                "what else",
                "tell me",
                "continue",
                "and",
            ]
            query_lower = query.lower()

            for indicator in follow_up_indicators:
                if indicator in query_lower:
                    return QueryType.FOLLOW_UP

            return QueryType.NEW_TOPIC

    def clear_memory(self):
        """Clear conversation memory for this session"""
        if self.enable_memory:
            memory_manager.clear_session(self.session_id, self.user_id)
        self.messages.clear()
        log_info(f"Cleared memory for session: {self.session_id}")


class ChatSessionManager:
    """Enhanced chat session manager with memory integration."""

    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> current session_id

    def get_or_create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        max_context_messages: int = 7,
        enable_memory: bool = True,
    ) -> ChatSession:
        """Get existing session or create new one with enhanced options."""
        if session_id is None:
            # Use user's current session or create new one
            session_id = self.user_sessions.get(user_id)
            if session_id is None or session_id not in self.sessions:
                session_id = f"{user_id}_{int(time.time())}"

        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(
                session_id=session_id,
                user_id=user_id,
                messages=[],
                max_context_messages=max_context_messages,
                enable_memory=enable_memory,
            )
            log_info(
                f"Created new chat session: {session_id} (memory: {enable_memory})"
            )

        # Update user's current session
        self.user_sessions[user_id] = session_id
        return self.sessions[session_id]

    def clear_session(self, session_id: str):
        """Clear a specific session and its memory."""
        if session_id in self.sessions:
            # Clear persistent memory
            session = self.sessions[session_id]
            session.clear_memory()

            # Remove from active sessions
            del self.sessions[session_id]

            # Update user_sessions mapping
            for user_id, curr_session in list(self.user_sessions.items()):
                if curr_session == session_id:
                    del self.user_sessions[user_id]

            log_info(f"Cleared chat session: {session_id}")

    def get_memory_stats(self) -> Dict:
        """Get conversation memory statistics"""
        return memory_manager.get_database_stats()

    def cleanup_old_conversations(self, days_old: int = 30):
        """Clean up old conversations from persistent storage"""
        memory_manager.cleanup_old_conversations(days_old)


# Global session manager instance
chat_manager = ChatSessionManager()
