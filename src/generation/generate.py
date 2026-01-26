import logging
import time
from dataclasses import dataclass
from typing import Any

from src.core.llm import OllamaClient
from src.generation.context_preparer import ContextPreparer
from src.generation.prompt_builder import PromptBuilder
from src.generation.query_classifier import QueryClassifier
from src.retrieval.retriever import OrionRetriever
from src.utilities.config import OrionConfig, TimingBreakdown

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from answer generation."""

    answer: str
    sources: list[dict[str, Any]]
    query_type: str
    mode: str
    metadata: dict[str, Any]
    timing: TimingBreakdown | None = None


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
        import time
        
        logger.info(f"Generating RAG response for query: {query[:100]}...")
        
        # Initialize timing
        timing = TimingBreakdown()
        overall_start = time.time()

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
            search_results, retrieval_timing = self.retriever.query(
                query_text=query, k=k, formatted=False, return_timing=True
            )
            # Merge retrieval timing
            timing.embedding_time = retrieval_timing.embedding_time
            timing.search_time = retrieval_timing.search_time
            timing.reranking_time = retrieval_timing.reranking_time
            timing.mmr_time = retrieval_timing.mmr_time
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            timing.total_time = time.time() - overall_start
            return GenerationResult(
                answer=f"I apologize, but I encountered an error while searching the knowledge base: {str(e)}",
                sources=[],
                query_type=classification.query_type,
                mode="rag",
                metadata={"error": str(e), "retrieval_failed": True},
                timing=timing,
            )

        if not search_results:
            logger.warning("No documents retrieved")
            timing.total_time = time.time() - overall_start
            return GenerationResult(
                answer="I couldn't find any relevant information in the knowledge base to answer your question.",
                sources=[],
                query_type=classification.query_type,
                mode="rag",
                metadata={"no_results": True},
                timing=timing,
            )

        logger.info(f"Retrieved {len(search_results)} documents")

        # Convert SearchResult objects to dicts for context_preparer
        search_results_dicts = [r.to_dict() for r in search_results]

        # Prepare contexts (clean, deduplicate, format citations)
        prep_start = time.time()
        prepared_contexts = self.context_preparer.prepare(
            contexts=search_results_dicts,
            return_full=True,
            include_citations=False,  # We'll handle citations in prompt builder
            sort_by_score=True,
        )
        timing.context_preparation_time = time.time() - prep_start

        # Limit to max_context_chunks
        max_chunks = self.generation_config.max_context_chunks
        prepared_contexts = prepared_contexts[:max_chunks]
        logger.debug(f"Using {len(prepared_contexts)} prepared contexts")

        # Build RAG prompt with citations
        try:
            prompt_start = time.time()
            prompt_components = self.prompt_builder.build_rag_prompt(
                query=query, contexts=prepared_contexts
            )
            timing.prompt_building_time = time.time() - prompt_start
        except Exception as e:
            logger.error(f"Prompt building failed: {e}")
            timing.total_time = time.time() - overall_start
            return GenerationResult(
                answer=f"I encountered an error while preparing the response: {str(e)}",
                sources=[],
                query_type=classification.query_type,
                mode="rag",
                metadata={"error": str(e), "prompt_building_failed": True},
                timing=timing,
            )

        # Convert to Ollama message format
        messages = prompt_components.to_messages()

        # Generate answer using LLM
        logger.debug("Calling LLM for generation")
        try:
            llm_start = time.time()
            response = self.llm_client.generate(
                messages=messages,
                model=self.config.rag.llm.model,
                temperature=self.config.rag.llm.temperature,
                top_p=self.config.rag.llm.top_p,
                max_tokens=self.config.rag.llm.max_tokens,
            )
            timing.llm_generation_time = time.time() - llm_start
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            timing.llm_generation_time = time.time() - llm_start
            timing.total_time = time.time() - overall_start
            return GenerationResult(
                answer=f"I encountered an error while generating the response: {str(e)}",
                sources=prepared_contexts if include_sources else [],
                query_type=classification.query_type,
                mode="rag",
                metadata={"error": str(e), "llm_generation_failed": True},
                timing=timing,
            )

        answer = response.get("message", {}).get("content", "").strip()

        # Post-process answer
        answer = self._post_process_answer(answer, prepared_contexts)

        # Format sources
        sources = []
        if include_sources:
            sources = self._format_sources(prepared_contexts)

        # Calculate total timing
        timing.total_time = time.time() - overall_start
        
        # Extract citation statistics
        citations_used = self._extract_citations(answer)
        
        # Build metadata
        metadata = {
            "query_type": classification.query_type,
            "num_contexts_used": len(prepared_contexts),
            "num_contexts_retrieved": len(search_results),
            "citations_in_answer": len(citations_used),
            "citation_numbers": citations_used,
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
            timing=timing,
        )

    def generate_chat_response(
        self,
        message: str,
        session_id: str | None = None,
        session_manager: Any | None = None,
        rag_mode: str | None = None,
        include_sources: bool = False,
        on_token: Any | None = None,
        stream: bool = False,
        temperature: float | None = None,
    ) -> GenerationResult:
        """
        Generate a conversational chat response.

        This is the stateful mode with conversation history.
        RAG retrieval is triggered based on rag_trigger_mode setting or rag_mode parameter.

        Args:
            message: User message
            session_id: Optional session ID for session-based history
            session_manager: Optional SessionManager instance for persistence
            rag_mode: RAG trigger mode override (always/auto/manual/never)
            include_sources: Include source information if RAG was used
            on_token: Optional callback for streaming tokens
            stream: Enable streaming mode
            temperature: LLM temperature override

        Returns:
            GenerationResult with answer and optional sources
        """
        logger.info(f"Generating chat response for message: {message[:100]}...")
        overall_start = time.time()
        timing = TimingBreakdown()

        # Load conversation history from session if provided
        if session_manager and session_id:
            conversation_messages = session_manager.get_messages(session_id)
            # Update prompt builder with session history
            self.prompt_builder.conversation_history = conversation_messages
            logger.debug(f"Loaded {len(conversation_messages)} messages from session {session_id}")

        # Classify query type
        classification = self.query_classifier.classify(message)

        # Determine if RAG retrieval is needed
        # Use rag_mode parameter if provided, otherwise use config setting
        if rag_mode:
            # Temporarily override config for this request
            original_rag_mode = self.generation_config.rag_trigger_mode
            self.generation_config.rag_trigger_mode = rag_mode
            should_retrieve = self.prompt_builder.should_retrieve_rag(message)
            self.generation_config.rag_trigger_mode = original_rag_mode
        else:
            should_retrieve = self.prompt_builder.should_retrieve_rag(message)
        
        logger.debug(f"RAG retrieval needed: {should_retrieve} (mode={rag_mode or self.generation_config.rag_trigger_mode})")

        prepared_contexts = []
        search_results = []

        # Retrieve documents if needed
        if should_retrieve:
            logger.debug("Triggering RAG retrieval in chat mode")
            try:
                k = self.config.rag.retrieval.default_k
                search_results, retrieval_timing = self.retriever.query(
                    query_text=message, k=k, formatted=False, return_timing=True
                )
                
                # Copy retrieval timing
                timing.embedding_time = retrieval_timing.embedding_time
                timing.search_time = retrieval_timing.search_time
                timing.reranking_time = retrieval_timing.reranking_time
                timing.mmr_time = retrieval_timing.mmr_time

                if search_results:
                    context_start = time.time()
                    # Convert SearchResult objects to dicts
                    search_results_dicts = [r.to_dict() for r in search_results]
                    prepared_contexts = self.context_preparer.prepare(
                        contexts=search_results_dicts,
                        return_full=True,
                        include_citations=False,
                        sort_by_score=True,
                    )
                    timing.context_preparation_time = time.time() - context_start
                    max_chunks = self.generation_config.max_context_chunks
                    prepared_contexts = prepared_contexts[:max_chunks]
                    logger.info(f"Retrieved and prepared {len(prepared_contexts)} contexts for chat")
            except Exception as e:
                logger.warning(f"RAG retrieval in chat mode failed: {e}")
                # Continue without RAG context

        # Build chat prompt (with or without RAG context)
        try:
            prompt_start = time.time()
            prompt_components = self.prompt_builder.build_chat_prompt(
                query=message,
                contexts=prepared_contexts if prepared_contexts else None,
            )
            timing.prompt_building_time = time.time() - prompt_start
        except Exception as e:
            logger.error(f"Chat prompt building failed: {e}")
            timing.total_time = time.time() - overall_start
            return GenerationResult(
                answer=f"I encountered an error while preparing the response: {str(e)}",
                sources=[],
                query_type=classification.query_type,
                mode="chat",
                metadata={"error": str(e), "prompt_building_failed": True},
                timing=timing,
            )

        # Convert to Ollama message format
        messages = prompt_components.to_messages()

        # Generate answer using LLM
        logger.debug("Calling LLM for chat generation")
        try:
            llm_start = time.time()
            
            # Use temperature override if provided
            llm_temperature = temperature if temperature is not None else self.config.rag.llm.temperature
            
            response = self.llm_client.generate(
                messages=messages,
                model=self.config.rag.llm.model,
                temperature=llm_temperature,
                top_p=self.config.rag.llm.top_p,
                max_tokens=self.config.rag.llm.max_tokens,
            )
            timing.llm_generation_time = time.time() - llm_start
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            timing.llm_generation_time = time.time() - llm_start
            timing.total_time = time.time() - overall_start
            return GenerationResult(
                answer=f"I encountered an error while generating the response: {str(e)}",
                sources=[],
                query_type=classification.query_type,
                mode="chat",
                metadata={"error": str(e), "llm_generation_failed": True},
                timing=timing,
            )

        answer = response.get("message", {}).get("content", "").strip()

        # Estimate token counts (simple approximation: ~4 chars per token)
        user_tokens = len(message) // 4
        assistant_tokens = len(answer) // 4

        # Store messages in session if session_manager provided
        if session_manager and session_id:
            session_manager.add_message(
                session_id=session_id,
                role="user",
                content=message,
                tokens=user_tokens,
            )
            session_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=answer,
                tokens=assistant_tokens,
            )
            logger.debug(f"Stored messages in session {session_id}")
        else:
            # Fallback to prompt builder history (old behavior)
            self.prompt_builder.add_to_history(role="user", content=message)
            self.prompt_builder.add_to_history(role="assistant", content=answer)

        # Format sources if RAG was used
        sources = []
        if include_sources and prepared_contexts:
            sources = self._format_sources(prepared_contexts)

        # Calculate total timing
        timing.total_time = time.time() - overall_start

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
        
        # Add rag_triggered attribute for API compatibility
        result = GenerationResult(
            answer=answer,
            sources=sources,
            query_type=classification.query_type,
            mode="chat",
            metadata=metadata,
            timing=timing,
        )
        result.rag_triggered = should_retrieve  # Add as dynamic attribute
        
        return result

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

    def _extract_citations(self, text: str) -> list[int]:
        """
        Extract citation numbers from text.
        
        Finds all citations in format [1], [2], etc.
        
        Args:
            text: Text containing citations
            
        Returns:
            List of citation indices (as integers)
        """
        import re
        
        # Match [1], [2], [3], etc.
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        
        # Convert to integers and deduplicate while preserving order
        seen = set()
        citations = []
        for match in matches:
            num = int(match)
            if num not in seen:
                citations.append(num)
                seen.add(num)
        
        return citations
    
    def _validate_citations(
        self, answer: str, num_sources: int
    ) -> tuple[str, list[int]]:
        """
        Validate and clean citations in the answer.
        
        Removes citations that reference non-existent sources.
        
        Args:
            answer: Answer text with citations
            num_sources: Number of available sources
            
        Returns:
            Tuple of (cleaned_answer, list_of_invalid_citations)
        """
        import re
        
        # Extract all citations
        citations = self._extract_citations(answer)
        
        # Find invalid citations (beyond available sources)
        invalid = [c for c in citations if c > num_sources or c < 1]
        
        if not invalid:
            return answer, []
        
        # Remove invalid citations
        cleaned = answer
        for citation_num in invalid:
            # Remove [N] where N is invalid
            pattern = rf'\[{citation_num}\]'
            cleaned = re.sub(pattern, '', cleaned)
        
        # Clean up any double spaces created by removal
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        logger.warning(
            f"Removed {len(invalid)} invalid citation(s): {invalid}. "
            f"Only {num_sources} source(s) available."
        )
        
        return cleaned, invalid
    
    def _expand_citations(
        self, answer: str, contexts: list[dict[str, Any]]
    ) -> str:
        """
        Replace numeric citations with full citation text.
        
        Converts [1] to (Source Title, p. 42) format.
        
        Args:
            answer: Answer with numeric citations [1], [2]
            contexts: List of context dicts with citation_text
            
        Returns:
            Answer with expanded citations
        """
        import re
        
        expanded = answer
        
        # Process each context
        for idx, ctx in enumerate(contexts, 1):
            citation_text = ctx.get('citation_text')
            if not citation_text:
                # Fallback to source file if no citation
                citation_text = ctx.get('normalized_source_file') or ctx.get('source_file', 'Unknown')
            
            # Replace [N] with (citation text)
            pattern = rf'\[{idx}\]'
            replacement = f'({citation_text})'
            expanded = re.sub(pattern, replacement, expanded)
        
        return expanded
    
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
        answer = answer.strip()
        
        # Validate citations if enabled
        if self.generation_config.validate_citations:
            answer, invalid_citations = self._validate_citations(answer, len(contexts))
            
            # Log invalid citations in metadata if any were found
            if invalid_citations:
                logger.debug(f"Removed invalid citations: {invalid_citations}")
        
        # Expand citations if enabled
        if self.generation_config.expand_citations:
            answer = self._expand_citations(answer, contexts)

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
