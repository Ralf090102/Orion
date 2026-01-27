"""
Prompt Builder for Orion RAG Assistant

Handles both RAG mode (stateless with citations) and Chat mode (conversational with history).
Manages token budgets, context formatting, and intelligent RAG triggering.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from src.utilities.utils import ensure_config, log_debug, log_info, log_warning

if TYPE_CHECKING:
    from src.retrieval.search import SearchResult
    from src.utilities.config import OrionConfig


@dataclass
class ConversationMessage:
    """Represents a single message in conversation history"""

    role: str  # "user" or "assistant"
    content: str
    tokens: int = 0

    def __post_init__(self):
        """Calculate tokens after initialization"""
        if self.tokens == 0:
            self.tokens = self._estimate_tokens(self.content)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4


@dataclass
class PromptComponents:
    """Components of a constructed prompt"""

    system_prompt: str
    context: str = ""
    query: str = ""
    history: List[ConversationMessage] = field(default_factory=list)
    citations: List[dict] = field(default_factory=list)
    total_tokens: int = 0

    def to_messages(self) -> List[dict]:
        """
        Convert to Ollama chat messages format.

        Returns:
            List of message dictionaries for Ollama API
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history (chat mode)
        for msg in self.history:
            messages.append({"role": msg.role, "content": msg.content})

        # Add current query with context (if any)
        user_message = ""
        if self.context:
            user_message = f"{self.context}\n\nQuestion: {self.query}"
        else:
            user_message = self.query

        messages.append({"role": "user", "content": user_message})

        return messages


class PromptBuilder:
    """
    Builds prompts for both RAG and Chat modes.

    Modes:
    - RAG: Stateless, formal answers with citations
    - Chat: Conversational with history, optional RAG augmentation
    """

    def __init__(self, config: Optional["OrionConfig"] = None):
        """
        Initialize prompt builder.

        Args:
            config: Orion configuration object
        """
        self.config = ensure_config(config)
        self.generation_config = self.config.rag.generation
        self.llm_config = self.config.rag.llm

        # Conversation history (chat mode only)
        self.conversation_history: List[ConversationMessage] = []

        # RAG trigger keywords for manual mode
        self.manual_trigger_keywords = ["search:", "find:", "lookup:", "rag:"]

    # ========== RAG MODE ==========

    def build_rag_prompt(
        self, query: str, contexts: list[dict], include_citations: Optional[bool] = None
    ) -> PromptComponents:
        """
        Build a stateless RAG prompt with citations.

        Args:
            query: User's question
            contexts: Retrieved and prepared context dicts
            include_citations: Override citation setting (uses config if None)

        Returns:
            PromptComponents with formatted prompt and metadata
        """
        if include_citations is None:
            include_citations = self.generation_config.enable_citations

        log_debug(f"Building RAG prompt with {len(contexts)} contexts", self.config)

        # Limit contexts to max_context_chunks
        contexts = contexts[: self.generation_config.max_context_chunks]

        # Build context section with citations
        context_parts = []
        citations = []

        for idx, result in enumerate(contexts, start=1):
            # Extract metadata (using dict access)
            metadata = result.get("metadata", {})
            source = metadata.get("file_name", metadata.get("source_file", "Unknown"))
            file_type = metadata.get("file_type", "Unknown")
            content = result.get("content", result.get("text", ""))
            score = result.get("score", result.get("final_score", 0.0))

            # Format citation
            if include_citations:
                citation_marker = self.generation_config.citation_format.format(index=idx, source=source)
                context_parts.append(f"{citation_marker} {content}")

                # Store citation info
                citations.append(
                    {
                        "index": idx,
                        "source": metadata.get("source_file", "Unknown"),
                        "file_name": source,
                        "file_type": file_type,
                        "score": score,
                    }
                )
            else:
                context_parts.append(content)

        # Combine context
        context_text = "\n\n".join(context_parts)

        # Build system prompt for RAG mode
        if include_citations:
            system_prompt = (
                f"{self.llm_config.system_prompt}\n\n"
                "IMPORTANT INSTRUCTIONS:\n"
                "1. Use the provided context to answer the question accurately\n"
                "2. Cite sources using the citation markers (e.g., [1], [2])\n"
                "3. If the context doesn't contain enough information, say so clearly\n"
                "4. Be concise but comprehensive\n"
                "5. Always include citations for factual claims"
            )
        else:
            system_prompt = (
                f"{self.llm_config.system_prompt}\n\n"
                "IMPORTANT: Use the provided context to answer the question accurately. "
                "If the context doesn't contain the answer, acknowledge this clearly."
            )

        # Format final context section
        formatted_context = f"Context:\n{context_text}" if context_text else ""

        # Calculate tokens
        total_tokens = self._estimate_total_tokens(system_prompt, formatted_context, query)

        log_debug(f"RAG prompt built: {total_tokens} tokens, {len(citations)} citations", self.config)

        return PromptComponents(
            system_prompt=system_prompt,
            context=formatted_context,
            query=query,
            citations=citations,
            total_tokens=total_tokens,
        )

    # ========== CHAT MODE ==========

    def build_chat_prompt(
        self, query: str, contexts: Optional[List["SearchResult"]] = None, force_rag: bool = False
    ) -> PromptComponents:
        """
        Build a conversational prompt with optional RAG augmentation.

        Args:
            query: User's message
            contexts: Optional retrieved contexts (if RAG triggered)
            force_rag: Force inclusion of RAG context even if contexts is None

        Returns:
            PromptComponents with conversation history and optional context
        """
        log_debug(f"Building chat prompt (history: {len(self.conversation_history)} messages)", self.config)

        # Determine if we should use RAG context
        use_rag = contexts is not None and len(contexts) > 0

        if use_rag:
            log_debug(f"Chat mode with RAG augmentation ({len(contexts)} contexts)", self.config)
            # Build context similar to RAG mode but without heavy citations
            contexts = contexts[: self.generation_config.max_context_chunks]
            context_parts = []

            for result in contexts:
                # Extract metadata (using dict access)
                metadata = result.get("metadata", {})
                source = metadata.get("file_name", metadata.get("source_file", "document"))
                content = result.get("content", result.get("text", ""))
                context_parts.append(f"[From: {source}]\n{content}")

            context_text = "\n\n".join(context_parts)
            formatted_context = f"Relevant information from knowledge base:\n{context_text}"
        else:
            formatted_context = ""

        # Build system prompt for chat mode
        if use_rag:
            system_prompt = (
                f"{self.llm_config.system_prompt}\n\n"
                "CHAT MODE INSTRUCTIONS:\n"
                "• You're having a casual, friendly conversation with the user\n"
                "• Use the conversation history to remember what you've discussed\n"
                "• When you have information from the knowledge base, weave it naturally into your responses\n"
                "• Be conversational, warm, and approachable - like talking to a friend\n"
                "• Use contractions (it's, you're, that's) and casual language\n"
                "• Keep responses concise unless the user asks for details\n"
                "• It's okay to use phrases like 'Yeah', 'Actually', 'So basically', etc.\n"
                "• Avoid overly formal language, jargon, or academic tone unless specifically requested\n"
                "• Feel free to acknowledge connections to previous messages ('Like we discussed earlier...')"
            )
        else:
            system_prompt = (
                f"{self.llm_config.system_prompt}\n\n"
                "CHAT MODE INSTRUCTIONS:\n"
                "• You're having a casual, friendly conversation with the user\n"
                "• Use the conversation history to maintain context and build on previous messages\n"
                "• Be warm, approachable, and natural - like chatting with a knowledgeable friend\n"
                "• Use contractions and casual language (it's, you're, that's)\n"
                "• Keep it conversational - avoid formal or academic tone\n"
                "• Phrases like 'Yeah', 'Actually', 'So', 'Basically' are totally fine\n"
                "• Be concise unless the user wants more depth\n"
                "• Feel free to acknowledge previous conversation ('As I mentioned before...')"
            )

        # Trim history to fit token budget
        trimmed_history = self._trim_history_to_budget(
            system_prompt=system_prompt, context=formatted_context, query=query
        )

        # Calculate total tokens
        total_tokens = self._estimate_total_tokens_with_history(
            system_prompt, formatted_context, query, trimmed_history
        )

        log_debug(f"Chat prompt built: {total_tokens} tokens, {len(trimmed_history)} history messages", self.config)

        return PromptComponents(
            system_prompt=system_prompt,
            context=formatted_context,
            query=query,
            history=trimmed_history,
            total_tokens=total_tokens,
        )

    # ========== CONVERSATION MANAGEMENT ==========

    def add_to_history(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        if role not in ("user", "assistant"):
            log_warning(f"Invalid role '{role}', expected 'user' or 'assistant'", self.config)
            return

        message = ConversationMessage(role=role, content=content)
        self.conversation_history.append(message)

        log_debug(f"Added {role} message to history ({message.tokens} tokens)", self.config)

        # Enforce max_history_messages limit (keep most recent)
        max_messages = self.generation_config.max_history_messages * 2  # *2 for user+assistant pairs
        if len(self.conversation_history) > max_messages:
            removed = self.conversation_history[: len(self.conversation_history) - max_messages]
            self.conversation_history = self.conversation_history[-max_messages:]
            log_debug(f"Trimmed {len(removed)} old messages from history", self.config)

    def clear_history(self) -> None:
        """Clear conversation history (start new conversation)"""
        count = len(self.conversation_history)
        self.conversation_history.clear()
        log_info(f"Cleared conversation history ({count} messages)", self.config)

    def get_history_summary(self) -> dict:
        """
        Get summary of conversation history.

        Returns:
            Dictionary with history stats
        """
        total_tokens = sum(msg.tokens for msg in self.conversation_history)
        user_messages = sum(1 for msg in self.conversation_history if msg.role == "user")
        assistant_messages = sum(1 for msg in self.conversation_history if msg.role == "assistant")

        return {
            "total_messages": len(self.conversation_history),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "total_tokens": total_tokens,
        }

    # ========== RAG TRIGGERING LOGIC ==========

    def should_retrieve_rag(self, query: str) -> bool:
        """
        Determine if RAG retrieval should be triggered for a chat message.

        Args:
            query: User's message

        Returns:
            True if RAG retrieval should happen, False otherwise
        """
        mode = self.generation_config.rag_trigger_mode

        if mode == "always":
            return True
        elif mode == "never":
            return False
        elif mode == "manual":
            # Check for manual trigger keywords
            query_lower = query.lower().strip()
            return any(query_lower.startswith(keyword) for keyword in self.manual_trigger_keywords)
        else:  # "auto"
            return self._auto_detect_rag_need(query)

    def _auto_detect_rag_need(self, query: str) -> bool:
        """
        Automatically detect if query needs RAG retrieval.

        Uses heuristics to determine if user is asking for factual information.

        Args:
            query: User's message

        Returns:
            True if RAG likely needed
        """
        query_lower = query.lower().strip()

        # Question words typically indicate information seeking
        question_words = [
            "what",
            "who",
            "where",
            "when",
            "why",
            "how",
            "which",
            "can you explain",
            "tell me about",
            "describe",
            "define",
            "find",
            "search",
            "lookup",
            "show me",
            "give me",
            "list",
        ]

        # Check if query starts with question words
        if any(query_lower.startswith(qw) for qw in question_words):
            return True

        # Check for question marks (often indicates factual query)
        if "?" in query:
            return True

        # Check for explicit information requests
        info_patterns = [
            r"\b(information|details|facts|data|documentation)\b",
            r"\b(according to|based on|from|in)\s+\w+",  # References to sources
        ]

        for pattern in info_patterns:
            if re.search(pattern, query_lower):
                return True

        # Default: don't trigger RAG for casual chat
        log_debug(f"Auto-detect: query doesn't need RAG - '{query[:50]}...'", self.config)
        return False

    def strip_manual_trigger(self, query: str) -> str:
        """
        Remove manual RAG trigger keywords from query.

        Args:
            query: User's message

        Returns:
            Query with trigger keyword removed
        """
        query_stripped = query.strip()
        for keyword in self.manual_trigger_keywords:
            if query_stripped.lower().startswith(keyword):
                return query_stripped[len(keyword) :].strip()
        return query_stripped

    # ========== TOKEN MANAGEMENT ==========

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // self.generation_config.chars_per_token

    def _estimate_total_tokens(self, system_prompt: str, context: str, query: str) -> int:
        """Estimate total tokens for RAG prompt"""
        total = 0
        total += self._estimate_tokens(system_prompt)
        total += self._estimate_tokens(context)
        total += self._estimate_tokens(query)
        return total

    def _estimate_total_tokens_with_history(
        self, system_prompt: str, context: str, query: str, history: List[ConversationMessage]
    ) -> int:
        """Estimate total tokens for chat prompt with history"""
        total = self._estimate_total_tokens(system_prompt, context, query)
        total += sum(msg.tokens for msg in history)
        return total

    def _trim_history_to_budget(
        self, system_prompt: str, context: str, query: str
    ) -> List[ConversationMessage]:
        """
        Trim conversation history to fit within token budget.

        Keeps most recent messages that fit within budget.

        Args:
            system_prompt: System prompt text
            context: RAG context text (if any)
            query: Current query

        Returns:
            Trimmed list of conversation messages
        """
        # Calculate base tokens (without history)
        base_tokens = self._estimate_total_tokens(system_prompt, context, query)

        # Calculate available tokens for history
        max_tokens = self.generation_config.max_total_tokens
        reserve_tokens = self.generation_config.reserve_tokens_for_response
        available_for_history = max_tokens - base_tokens - reserve_tokens

        if available_for_history <= 0:
            log_warning("No token budget available for history", self.config)
            return []

        # Trim history from oldest to newest until we fit
        trimmed_history = []
        current_tokens = 0

        # Iterate backwards (most recent first)
        for message in reversed(self.conversation_history):
            if current_tokens + message.tokens <= available_for_history:
                trimmed_history.insert(0, message)  # Insert at beginning to maintain order
                current_tokens += message.tokens
            else:
                break

        if len(trimmed_history) < len(self.conversation_history):
            log_debug(
                f"Trimmed history: {len(self.conversation_history)} -> {len(trimmed_history)} messages",
                self.config,
            )

        return trimmed_history

    # ========== UTILITY METHODS ==========

    def format_sources(self, citations: List[dict]) -> str:
        """
        Format citation sources for display.

        Args:
            citations: List of citation dictionaries

        Returns:
            Formatted sources string
        """
        if not citations:
            return "No sources cited."

        sources_lines = ["Sources:"]
        for citation in citations:
            idx = citation["index"]
            file_name = citation["file_name"]
            file_type = citation.get("file_type", "Unknown")
            score = citation.get("score", 0.0)

            sources_lines.append(f"  [{idx}] {file_name} ({file_type}) - Relevance: {score:.3f}")

        return "\n".join(sources_lines)

    def get_token_budget_info(self) -> dict:
        """
        Get current token budget information.

        Returns:
            Dictionary with token budget details
        """
        history_tokens = sum(msg.tokens for msg in self.conversation_history)

        return {
            "max_total_tokens": self.generation_config.max_total_tokens,
            "reserve_for_response": self.generation_config.reserve_tokens_for_response,
            "history_tokens": history_tokens,
            "history_messages": len(self.conversation_history),
            "available_for_context": self.generation_config.max_total_tokens
            - self.generation_config.reserve_tokens_for_response
            - history_tokens,
        }


# ========== CONVENIENCE FUNCTIONS ==========


def create_prompt_builder(config: Optional["OrionConfig"] = None) -> PromptBuilder:
    """
    Factory function to create a PromptBuilder instance.

    Args:
        config: Orion configuration

    Returns:
        Configured PromptBuilder instance

    Example:
        builder = create_prompt_builder()

        # RAG mode
        prompt = builder.build_rag_prompt(query, contexts)
        messages = prompt.to_messages()

        # Chat mode
        prompt = builder.build_chat_prompt(query, contexts)
        builder.add_to_history("user", query)
        builder.add_to_history("assistant", response)
    """
    return PromptBuilder(config=config)
