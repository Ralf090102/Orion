"""
Context-Aware Query Resolution for Conversation Memory & Context

Integrates conversation memory with the RAG pipeline to provide context-aware
query processing and follow-up handling.
"""

from typing import Dict, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from core.rag.conversation_memory import QueryType, ConversationContext
from core.utils.orion_utils import log_debug

if TYPE_CHECKING:
    from core.rag.chat import ChatSession


@dataclass
class ResolvedQuery:
    """A query enhanced with conversation context"""

    original_query: str
    resolved_query: str
    query_type: QueryType
    context_used: bool
    topics_referenced: List[str]
    sources_referenced: List[Dict]
    enhancement_explanation: str


class ContextAwareQueryResolver:
    """Resolves queries using conversation context and memory"""

    def __init__(self):
        self.context_templates = {
            QueryType.FOLLOW_UP: [
                "Building on our previous discussion about {topics}, {query}",
                "Continuing from our conversation about {topics}, {query}",
                "Following up on {topics}, {query}",
            ],
            QueryType.REFERENCE: [
                "Referring to our previous discussion about {topics}, {query}",
                "In the context of {topics} that we discussed, {query}",
                "Regarding {topics} from our previous conversation, {query}",
            ],
            QueryType.CLARIFICATION: [
                "To clarify our discussion about {topics}, {query}",
                "For better understanding of {topics}, {query}",
                "To explain more about {topics}, {query}",
            ],
        }

    def resolve_query(self, query: str, session: "ChatSession") -> ResolvedQuery:
        """
        Resolve a query using conversation context

        Args:
            query: The user's query
            session: Current chat session with context

        Returns:
            ResolvedQuery with enhanced context
        """
        # Detect query type
        query_type = session.detect_query_type(query)

        if query_type == QueryType.NEW_TOPIC:
            # No context needed for new topics
            return ResolvedQuery(
                original_query=query,
                resolved_query=query,
                query_type=query_type,
                context_used=False,
                topics_referenced=[],
                sources_referenced=[],
                enhancement_explanation="New topic query - no context needed",
            )

        # Get conversation context for follow-ups/references
        context = session.get_conversation_context()

        # Check if we actually have meaningful context
        if not context.recent_topics and not context.recent_sources:
            # No meaningful context available - treat as new topic
            return ResolvedQuery(
                original_query=query,
                resolved_query=query,
                query_type=QueryType.NEW_TOPIC,
                context_used=False,
                topics_referenced=[],
                sources_referenced=[],
                enhancement_explanation="No conversation context available - treating as new topic",
            )

        # Enhance query with context
        enhanced_query, topics_used, sources_used, explanation = (
            self._enhance_query_with_context(query, query_type, context)
        )

        log_debug(f"Query resolution: {query_type.value} -> {explanation}")

        return ResolvedQuery(
            original_query=query,
            resolved_query=enhanced_query,
            query_type=query_type,
            context_used=True,
            topics_referenced=topics_used,
            sources_referenced=sources_used,
            enhancement_explanation=explanation,
        )

    def _enhance_query_with_context(
        self, query: str, query_type: QueryType, context: ConversationContext
    ) -> Tuple[str, List[str], List[Dict], str]:
        """Enhance query using conversation context"""

        # Select relevant topics (limit to most recent/relevant)
        relevant_topics = self._select_relevant_topics(query, context.recent_topics)

        # Select relevant sources
        relevant_sources = context.recent_sources[-3:] if context.recent_sources else []

        if not relevant_topics and not relevant_sources:
            # No useful context found
            return query, [], [], "No relevant context found"

        # Build enhanced query
        enhanced_query = query
        explanation_parts = []

        if relevant_topics:
            # Use templates to enhance query with topics
            templates = self.context_templates.get(
                query_type, self.context_templates[QueryType.FOLLOW_UP]
            )
            template = templates[0]  # Use first template

            topics_str = ", ".join(relevant_topics[:3])  # Limit to top 3 topics
            enhanced_query = template.format(topics=topics_str, query=query)
            explanation_parts.append(f"Added topic context: {topics_str}")

        if relevant_sources:
            # Add source context if available
            source_names = [src.get("source", "Unknown") for src in relevant_sources]
            source_context = f" (referencing sources: {', '.join(source_names[:2])})"
            enhanced_query += source_context
            explanation_parts.append(
                f"Added source references: {len(relevant_sources)} sources"
            )

        explanation = (
            "; ".join(explanation_parts)
            if explanation_parts
            else "Context enhancement applied"
        )

        return enhanced_query, relevant_topics, relevant_sources, explanation

    def _select_relevant_topics(
        self, query: str, available_topics: List[str]
    ) -> List[str]:
        """Select topics from context that are relevant to the query"""
        if not available_topics:
            return []

        query_lower = query.lower()
        relevant_topics = []

        # Simple relevance scoring based on word overlap
        for topic in available_topics:
            topic_lower = topic.lower()

            # Direct mention
            if topic_lower in query_lower:
                relevant_topics.append(topic)
            # Similar words (simple stem matching)
            elif any(
                word in topic_lower for word in query_lower.split() if len(word) > 3
            ):
                relevant_topics.append(topic)

        # If no direct matches, include most recent topics for follow-up queries
        if not relevant_topics and available_topics:
            relevant_topics = available_topics[:2]  # Take 2 most recent topics

        return relevant_topics[:5]  # Limit to prevent prompt bloat

    def create_context_aware_prompt(
        self, resolved_query: ResolvedQuery, rag_context: str, base_prompt: str
    ) -> str:
        """
        Create a prompt that includes conversation context

        Args:
            resolved_query: The resolved query with context
            rag_context: Context from RAG retrieval
            base_prompt: Base system prompt

        Returns:
            Enhanced prompt with conversation context
        """
        # Start with base prompt
        prompt_parts = [base_prompt]

        # Add conversation context explanation if query was enhanced
        if resolved_query.context_used:
            context_note = f"\nConversation Context: This is a {resolved_query.query_type.value} query. "
            context_note += resolved_query.enhancement_explanation

            if resolved_query.topics_referenced:
                context_note += f" Key topics from our conversation: {', '.join(resolved_query.topics_referenced)}."

            prompt_parts.append(context_note)

        # Add RAG context
        if rag_context:
            prompt_parts.append(f"\nRelevant Information:\n{rag_context}")

        # Add the (potentially enhanced) query
        prompt_parts.append(f"\nQuery: {resolved_query.resolved_query}")

        # Add guidance based on query type
        if resolved_query.query_type == QueryType.FOLLOW_UP:
            prompt_parts.append(
                "\nPlease provide additional information building on our previous discussion."
            )
        elif resolved_query.query_type == QueryType.REFERENCE:
            prompt_parts.append(
                "\nPlease reference our previous conversation when relevant."
            )
        elif resolved_query.query_type == QueryType.CLARIFICATION:
            prompt_parts.append(
                "\nPlease provide clarification and more detailed explanation."
            )

        return "\n".join(prompt_parts)


# Global resolver instance
context_resolver = ContextAwareQueryResolver()
