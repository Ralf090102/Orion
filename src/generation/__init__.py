"""
Generation module for Orion RAG Assistant.

This module handles answer generation using both RAG and Chat modes.
"""

from src.generation.context_preparer import ContextPreparer, prepare_contexts
from src.generation.generate import AnswerGenerator, GenerationResult, generate_answer
from src.generation.prompt_builder import (
    ConversationMessage,
    PromptBuilder,
    PromptComponents,
)
from src.generation.query_classifier import QueryClassification, QueryClassifier
from src.generation.session_manager import (
    ChatSession,
    SessionManager,
    get_session_manager,
    reset_session_manager,
)

__all__ = [
    # Main generator
    "AnswerGenerator",
    "GenerationResult",
    "generate_answer",
    # Prompt building
    "PromptBuilder",
    "PromptComponents",
    "ConversationMessage",
    # Context preparation
    "ContextPreparer",
    "prepare_contexts",
    # Query classification
    "QueryClassifier",
    "QueryClassification",
    # Session management
    "SessionManager",
    "ChatSession",
    "get_session_manager",
    "reset_session_manager",
]
