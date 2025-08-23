"""
Advanced Query Processing for Orion RAG System

This module handles sophisticated query analysis including:
- Intent detection (what type of answer is needed)
- Multi-step reasoning (breaking complex queries into parts)
- Query validation (can we answer this with our documents)
"""

from enum import Enum
from typing import List
from dataclasses import dataclass
import re
from core.utils.orion_utils import log_info, log_debug


class QueryIntent(Enum):
    """Different types of query intents we can handle"""

    FACTUAL = "factual"  # "What is X?" - needs definition/facts
    ANALYTICAL = "analytical"  # "Compare X and Y" - needs analysis
    PROCEDURAL = "procedural"  # "How do I do X?" - needs steps
    TROUBLESHOOTING = "troubleshooting"  # "Why doesn't X work?" - needs solution
    CREATIVE = "creative"  # "Write/Create/Generate X" - needs generation
    EXPLORATORY = "exploratory"  # "Tell me about X" - needs broad overview


@dataclass
class QueryAnalysis:
    """Results of analyzing a user query"""

    original_query: str
    intent: QueryIntent
    confidence: float  # How confident we are in the intent detection
    sub_queries: List[str]  # For multi-step reasoning
    keywords: List[str]  # Important terms extracted
    can_answer: bool  # Whether we think we can answer this
    reasoning: str  # Why we think we can/can't answer


class QueryProcessor:
    """
    Advanced query processing system that analyzes user queries
    to provide better, more targeted responses.
    """

    def __init__(self):
        # Keywords that help identify different intents
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r"\bwhat is\b",
                r"\bdefine\b",
                r"\bexplain\b",
                r"\btell me about\b",
            ],
            QueryIntent.ANALYTICAL: [
                r"\bcompare\b",
                r"\bdifference\b",
                r"\bversus\b",
                r"\bvs\b",
                r"\banalyze\b",
                r"\bevaluate\b",
            ],
            QueryIntent.PROCEDURAL: [
                r"\bhow to\b",
                r"\bhow do i\b",
                r"\bsteps to\b",
                r"\bguide\b",
                r"\btutorial\b",
                r"\binstruct\b",
            ],
            QueryIntent.TROUBLESHOOTING: [
                r"\bwhy.*not work\b",
                r"\bwhy.*doesn\'t.*work\b",
                r"\berror\b",
                r"\bproblem\b",
                r"\bissue\b",
                r"\bfix\b",
                r"\bdebug\b",
                r"\btrouble\b",
                r"\bsolve\b",
                r"\bwhy.*getting\b",
            ],
            QueryIntent.CREATIVE: [
                r"\bwrite\b",
                r"\bcreate\b",
                r"\bgenerate\b",
                r"\bbuild\b",
                r"\bmake\b",
                r"\bdesign\b",
            ],
            QueryIntent.EXPLORATORY: [
                r"\btell me about\b",
                r"\blearn about\b",
                r"\bexplore\b",
                r"\boverview\b",
                r"\bintroduce\b",
            ],
        }

    def detect_intent(self, query: str) -> tuple[QueryIntent, float]:
        """
        Analyze the query to determine what type of answer the user wants.

        Args:
            query: The user's question

        Returns:
            Tuple of (detected_intent, confidence_score)
        """
        query_lower = query.lower()
        intent_scores = {}

        # Score each intent based on pattern matching
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches

            if score > 0:
                intent_scores[intent] = score / len(patterns)  # Normalize

        if not intent_scores:
            # Default to factual if no patterns match
            return QueryIntent.FACTUAL, 0.3

        # Return the highest scoring intent
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(intent_scores[best_intent], 1.0)  # Cap at 1.0

        log_debug(
            f"Intent detection: {best_intent.value} (confidence: {confidence:.2f})"
        )
        return best_intent, confidence

    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from the query for better retrieval.

        This is a simple implementation - in production you might use
        more sophisticated NLP techniques.
        """
        # Remove common stop words
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
            "how",
            "what",
            "when",
            "where",
            "why",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "can",
            "may",
            "might",
            "must",
        }

        # Simple keyword extraction
        words = re.findall(r"\b\w+\b", query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Remove duplicates while preserving order
        unique_keywords = list(dict.fromkeys(keywords))

        log_debug(f"Extracted keywords: {unique_keywords}")
        return unique_keywords

    def break_into_sub_queries(self, query: str, intent: QueryIntent) -> List[str]:
        """
        For complex queries, break them down into simpler sub-questions.

        This helps us provide more comprehensive answers by addressing
        each part of a complex question.
        """
        sub_queries = []
        query_lower = query.lower()

        # Different strategies based on intent
        if intent == QueryIntent.ANALYTICAL:
            # Look for comparison patterns
            if (
                "compare" in query_lower
                or "vs" in query_lower
                or "versus" in query_lower
            ):
                # Try to extract the items being compared
                # This is a simple heuristic - could be much more sophisticated
                if " and " in query_lower:
                    parts = query_lower.split(" and ")
                    if len(parts) == 2:
                        item1 = parts[0].strip()
                        item2 = parts[1].strip()
                        sub_queries = [
                            f"What is {item1}?",
                            f"What is {item2}?",
                            f"How do {item1} and {item2} differ?",
                        ]

        elif intent == QueryIntent.PROCEDURAL:
            # For "how to" questions, break down into steps
            if "how to" in query_lower:
                main_topic = query_lower.replace("how to", "").strip()
                sub_queries = [
                    f"What is {main_topic}?",
                    f"What are the prerequisites for {main_topic}?",
                    f"What are the steps to {main_topic}?",
                ]

        elif intent == QueryIntent.TROUBLESHOOTING:
            # For problems, understand the issue first
            if any(word in query_lower for word in ["error", "problem", "issue"]):
                sub_queries = [
                    "What is the specific error or problem?",
                    "What are common causes of this issue?",
                    "What are the solutions to fix this?",
                ]

        # If no sub-queries generated, return the original
        if not sub_queries:
            sub_queries = [query]

        log_debug(f"Sub-queries generated: {sub_queries}")
        return sub_queries

    def validate_query(self, query: str, keywords: List[str]) -> tuple[bool, str]:
        """
        Check if we can likely answer this query based on our knowledge domain.

        This is a heuristic check - in a real system you might:
        - Check if keywords match document content
        - Use embedding similarity to find relevant docs
        - Check against a knowledge base index
        """
        # Simple validation rules

        # Weather, news, real-time data - we can't answer
        # Be more specific about what constitutes "real-time" vs conversation context
        query_lower = query.lower()

        # Real-time data patterns (specific and contextual)
        real_time_patterns = [
            "weather",
            "news",
            "stock price",
            "current events",
            "today's news",
            "latest news",
            "breaking news",
            "current weather",
            "temperature today",
            "forecast",
            "stock market today",
            "crypto price",
            "exchange rate",
        ]

        # Conversation context patterns that should NOT be rejected
        conversation_patterns = [
            "right now",
            "what we're",
            "we are talking",
            "our conversation",
            "this conversation",
            "what we discussed",
            "our discussion",
            "current topic",
            "currently discussing",
        ]

        # Check for conversation context first (higher priority)
        if any(pattern in query_lower for pattern in conversation_patterns):
            # This is asking about conversation context, not real-time data
            pass  # Continue with validation
        elif any(pattern in query_lower for pattern in real_time_patterns):
            return False, "Query requires real-time data that we don't have access to"

        # Personal information - we can't answer (but be more specific)
        personal = [
            r"\bmy personal\b",
            r"\bi am\b",
            r"\bmy private\b",
            r"\bmy account\b",
        ]
        if any(re.search(pattern, query.lower()) for pattern in personal):
            return False, "Query requires personal information we don't have access to"

        # If we have technical keywords, we're probably good
        technical_domains = [
            "programming",
            "code",
            "software",
            "computer",
            "algorithm",
            "function",
            "class",
            "method",
            "api",
            "database",
        ]
        if any(word in query.lower() for word in technical_domains):
            return (
                True,
                "Query appears to be about technical topics we likely have documentation for",
            )

        # Default assumption - we can probably help
        return True, "Query appears answerable with available documentation"

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Perform complete analysis of a user query.

        This is the main method that ties everything together.
        """
        log_info(f"Analyzing query: '{query}'")

        # Step 1: Detect intent
        intent, confidence = self.detect_intent(query)

        # Step 2: Extract keywords
        keywords = self.extract_keywords(query)

        # Step 3: Break into sub-queries if needed
        sub_queries = self.break_into_sub_queries(query, intent)

        # Step 4: Validate if we can answer
        can_answer, reasoning = self.validate_query(query, keywords)

        analysis = QueryAnalysis(
            original_query=query,
            intent=intent,
            confidence=confidence,
            sub_queries=sub_queries,
            keywords=keywords,
            can_answer=can_answer,
            reasoning=reasoning,
        )

        log_info(
            f"Query analysis complete: Intent={intent.value}, Can answer={can_answer}"
        )
        return analysis
