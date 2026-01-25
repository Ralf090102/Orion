import logging
from dataclasses import dataclass
from typing import Any

from src.core.llm import OllamaClient
from src.generation.context_preparer import ContextPreparer
from src.generation.prompt_builder import PromptBuilder
from src.generation.query_classifier import QueryClassifier
from src.retrieval.retriever import OrionRetriever
from src.utilities.config import OrionConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from answer generation."""

    answer: str
    sources: list[dict[str, Any]]
    query_type: str
    mode: str
    metadata: dict[str, Any]


class AnswerGenerator:
    """
    Orchestrates the complete RAG/Chat pipeline.

    Responsibilities:
        - Query classification
        - Document retrieval (if needed)
        - Context preparation
        - Prompt building
        - LLM generation
        - Answer post-processing
    """

    def __init__(self, config: OrionConfig):
        """
        Initialize the answer generator.

        Args:
            config: Orion configuration
        """
        self.config = config
        self.generation_config = config.rag.generation

        # Initialize components
        self.retriever = OrionRetriever(config)
        self.context_preparer = ContextPreparer(
            similarity_threshold=config.rag.preprocessing.similarity_threshold
        )
        self.prompt_builder = PromptBuilder(config)
        self.query_classifier = QueryClassifier()
        self.llm_client = OllamaClient(
            base_url=config.rag.llm.base_url, timeout=config.rag.llm.timeout
        )

        logger.info(
            f"Initialized AnswerGenerator in {self.generation_config.mode} mode"
        )

    def generate_rag_response(
        self, query: str, k: int | None = None, include_sources: bool = True
    ) -> GenerationResult:
        """
        Generate a RAG response with citations.

        This is the stateless mode where every query:
        1. Retrieves relevant documents
        2. Builds a prompt with citations
        3. Generates an answer grounded in the knowledge base

        Args:
            query: User query
            k: Number of contexts to retrieve (uses config default if None)
            include_sources: Include source information in response

        Returns:
            GenerationResult with answer and sources
        """
        logger.info(f"Generating RAG response for query: {query[:100]}...")

        # Classify query type for better prompt adaptation
        classification = self.query_classifier.classify(query)
        logger.debug(
            f"Query classified as: {classification.query_type} "
            f"(confidence: {classification.confidence:.2f})"
        )

        # Retrieve relevant documents
        k = k or self.config.rag.retrieval.default_k
        logger.debug(f"Retrieving top {k} documents")

        try:
            search_results = self.retriever.query(
                query=query, k=k, formatted=False, include_metadata=True
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return GenerationResult(
                answer=f"I apologize, but I encountered an error while searching the knowledge base: {str(e)}",
                sources=[],
                query_type=classification.query_type,
                mode="rag",
                metadata={"error": str(e), "retrieval_failed": True},
            )

        if not search_results:
            logger.warning("No documents retrieved")
            return GenerationResult(
                answer="I couldn't find any relevant information in the knowledge base to answer your question.",
                sources=[],
                query_type=classification.query_type,
                mode="rag",
                metadata={"no_results": True},
            )

        logger.info(f"Retrieved {len(search_results)} documents")

        # Prepare contexts (clean, deduplicate, format citations)
        prepared_contexts = self.context_preparer.prepare(
            contexts=search_results,
            return_full=True,
            include_citations=False,  # We'll handle citations in prompt builder
            sort_by_score=True,
        )

        # Limit to max_context_chunks
        max_chunks = self.generation_config.max_context_chunks
        prepared_contexts = prepared_contexts[:max_chunks]
        logger.debug(f"Using {len(prepared_contexts)} prepared contexts")

        # Build RAG prompt with citations
        try:
            prompt_components = self.prompt_builder.build_rag_prompt(
                query=query, contexts=prepared_contexts, query_type=classification.query_type
            )
        except Exception as e:
            logger.error(f"Prompt building failed: {e}")
            return GenerationResult(
                answer=f"I encountered an error while preparing the response: {str(e)}",
                sources=[],
                query_type=classification.query_type,
                mode="rag",
                metadata={"error": str(e), "prompt_building_failed": True},
            )

        # Convert to Ollama message format
        messages = self.prompt_builder.to_messages(prompt_components)

        # Generate answer using LLM
        logger.debug("Calling LLM for generation")
        try:
            response = self.llm_client.generate(
                messages=messages,
                model=self.config.rag.llm.model,
                temperature=self.config.rag.llm.temperature,
                top_p=self.config.rag.llm.top_p,
                max_tokens=self.config.rag.llm.max_tokens,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return GenerationResult(
                answer=f"I encountered an error while generating the response: {str(e)}",
                sources=prepared_contexts if include_sources else [],
                query_type=classification.query_type,
                mode="rag",
                metadata={"error": str(e), "llm_generation_failed": True},
            )

        answer = response.get("message", {}).get("content", "").strip()

        # Post-process answer
        answer = self._post_process_answer(answer, prepared_contexts)

        # Format sources
        sources = []
        if include_sources:
            sources = self._format_sources(prepared_contexts)

        # Build metadata
        metadata = {
            "query_type": classification.query_type,
            "num_contexts_used": len(prepared_contexts),
            "num_contexts_retrieved": len(search_results),
            "total_tokens": prompt_components.total_tokens,
            "llm_model": self.config.rag.llm.model,
        }

        logger.info("RAG response generated successfully")
        return GenerationResult(
            answer=answer,
            sources=sources,
            query_type=classification.query_type,
            mode="rag",
            metadata=metadata,
        )

    def generate_chat_response(
        self, message: str, include_sources: bool = False
    ) -> GenerationResult:
        """
        Generate a conversational chat response.

        This is the stateful mode with conversation history.
        RAG retrieval is triggered based on rag_trigger_mode setting.

        Args:
            message: User message
            include_sources: Include source information if RAG was used

        Returns:
            GenerationResult with answer and optional sources
        """
        logger.info(f"Generating chat response for message: {message[:100]}...")

        # Classify query type
        classification = self.query_classifier.classify(message)

        # Determine if RAG retrieval is needed
        should_retrieve = self.prompt_builder.should_retrieve_rag(message)
        logger.debug(f"RAG retrieval needed: {should_retrieve}")

        prepared_contexts = []
        search_results = []

        # Retrieve documents if needed
        if should_retrieve:
            logger.debug("Triggering RAG retrieval in chat mode")
            try:
                k = self.config.rag.retrieval.default_k
                search_results = self.retriever.query(
                    query=message, k=k, formatted=False, include_metadata=True
                )

                if search_results:
                    prepared_contexts = self.context_preparer.prepare(
                        contexts=search_results,
                        return_full=True,
                        include_citations=False,
                        sort_by_score=True,
                    )
                    max_chunks = self.generation_config.max_context_chunks
                    prepared_contexts = prepared_contexts[:max_chunks]
                    logger.info(f"Retrieved and prepared {len(prepared_contexts)} contexts for chat")
            except Exception as e:
                logger.warning(f"RAG retrieval in chat mode failed: {e}")
                # Continue without RAG context

        # Build chat prompt (with or without RAG context)
        try:
            prompt_components = self.prompt_builder.build_chat_prompt(
                query=message,
                contexts=prepared_contexts if prepared_contexts else None,
            )
        except Exception as e:
            logger.error(f"Chat prompt building failed: {e}")
            return GenerationResult(
                answer=f"I encountered an error while preparing the response: {str(e)}",
                sources=[],
                query_type=classification.query_type,
                mode="chat",
                metadata={"error": str(e), "prompt_building_failed": True},
            )

        # Convert to Ollama message format
        messages = self.prompt_builder.to_messages(prompt_components)

        # Generate answer using LLM
        logger.debug("Calling LLM for chat generation")
        try:
            response = self.llm_client.generate(
                messages=messages,
                model=self.config.rag.llm.model,
                temperature=self.config.rag.llm.temperature,
                top_p=self.config.rag.llm.top_p,
                max_tokens=self.config.rag.llm.max_tokens,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return GenerationResult(
                answer=f"I encountered an error while generating the response: {str(e)}",
                sources=[],
                query_type=classification.query_type,
                mode="chat",
                metadata={"error": str(e), "llm_generation_failed": True},
            )

        answer = response.get("message", {}).get("content", "").strip()

        # Add to conversation history
        self.prompt_builder.add_to_history(role="user", content=message)
        self.prompt_builder.add_to_history(role="assistant", content=answer)

        # Format sources if RAG was used
        sources = []
        if include_sources and prepared_contexts:
            sources = self._format_sources(prepared_contexts)

        # Build metadata
        metadata = {
            "query_type": classification.query_type,
            "rag_retrieval_triggered": should_retrieve,
            "num_contexts_used": len(prepared_contexts),
            "conversation_turns": len(self.prompt_builder.conversation_history) // 2,
            "total_tokens": prompt_components.total_tokens,
            "llm_model": self.config.rag.llm.model,
        }

        logger.info("Chat response generated successfully")
        return GenerationResult(
            answer=answer,
            sources=sources,
            query_type=classification.query_type,
            mode="chat",
            metadata=metadata,
        )

    def generate(
        self, query: str, mode: str | None = None, **kwargs
    ) -> GenerationResult:
        """
        Generate a response using the configured or specified mode.

        Args:
            query: User query or message
            mode: Override generation mode ("rag" or "chat")
            **kwargs: Additional arguments passed to mode-specific methods

        Returns:
            GenerationResult
        """
        mode = mode or self.generation_config.mode

        if mode == "rag":
            return self.generate_rag_response(query, **kwargs)
        elif mode == "chat":
            return self.generate_chat_response(query, **kwargs)
        else:
            raise ValueError(f"Invalid generation mode: {mode}. Must be 'rag' or 'chat'")

    def clear_conversation(self) -> None:
        """Clear conversation history (for chat mode)."""
        self.prompt_builder.clear_history()
        logger.info("Conversation history cleared")

    def get_conversation_summary(self) -> dict[str, Any]:
        """
        Get summary of current conversation.

        Returns:
            Dictionary with conversation statistics
        """
        return self.prompt_builder.get_history_summary()

    def _post_process_answer(
        self, answer: str, contexts: list[dict[str, Any]]
    ) -> str:
        """
        Post-process the generated answer.

        Tasks:
            - Validate citations exist in contexts
            - Remove hallucinated citations
            - Clean up formatting

        Args:
            answer: Raw LLM answer
            contexts: List of context dicts used

        Returns:
            Cleaned answer
        """
        # For now, just basic cleanup
        # TODO: Add citation validation once we implement citation extraction
        answer = answer.strip()

        # Remove any trailing incomplete sentences
        if answer and not answer[-1] in ".!?\"'":
            # Find last complete sentence
            for delimiter in [". ", "! ", "? "]:
                last_idx = answer.rfind(delimiter)
                if last_idx > len(answer) * 0.7:  # Only if we keep >70% of answer
                    answer = answer[: last_idx + 1]
                    break

        return answer

    def _format_sources(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format sources for the response.

        Args:
            contexts: List of prepared context dicts

        Returns:
            List of formatted source dicts
        """
        sources = []

        for idx, ctx in enumerate(contexts, 1):
            source = {
                "index": idx,
                "text": ctx.get("text", "")[:200] + "...",  # Preview
                "score": ctx.get("final_score", 0.0),
            }

            # Add citation if available
            if ctx.get("citation_text"):
                source["citation"] = ctx["citation_text"]

            # Add source metadata
            if ctx.get("source_file"):
                source["source_file"] = ctx.get("normalized_source_file") or ctx["source_file"]

            if ctx.get("page") is not None:
                source["page"] = ctx["page"]

            if ctx.get("title"):
                source["title"] = ctx["title"]

            if ctx.get("url"):
                source["url"] = ctx["url"]

            sources.append(source)

        return sources


def generate_answer(
    query: str,
    config: OrionConfig | None = None,
    mode: str | None = None,
    **kwargs,
) -> GenerationResult:
    """
    Functional API: Generate an answer for a query.

    Args:
        query: User query
        config: Orion configuration (creates default if None)
        mode: Generation mode override
        **kwargs: Additional arguments

    Returns:
        GenerationResult
    """
    from src.utilities.config import get_config

    if config is None:
        config = get_config()

    generator = AnswerGenerator(config)
    return generator.generate(query, mode=mode, **kwargs)
