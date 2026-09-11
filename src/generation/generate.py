import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from src.core.llm import OllamaClient
from src.generation.context_preparer import ContextPreparer, PreparedContext
from src.generation.prompt_builder import PromptBuilder
from src.generation.query_classifier import QueryClassifier
from src.generation.trace import QueryTrace
from src.retrieval.retriever import OrionRetriever
from src.utilities.config import OrionConfig, TimingBreakdown
from src.utilities.request_context import new_request_id, request_id_var

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from answer generation."""

    answer: str
    sources: list[dict[str, Any]]
    query_type: str
    mode: str
    metadata: dict[str, Any]
    rag_triggered: bool = False
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
        self,
        query: str,
        k: int | None = None,
        include_sources: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        on_token: Any | None = None,
        on_sources: Any | None = None,
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
            temperature: LLM temperature override (uses config default if None)
            max_tokens: LLM max_tokens override (uses config default if None)
            stream: Enable streaming mode (mirrors generate_chat_response)
            on_token: Optional callback for streaming tokens
            on_sources: Optional callback fired once Sources are ready --
                after context prep (dedup, citation formatting), before
                generation starts. See CONTEXT.md's "Sources" entry: this
                is deliberately the prepared, post-dedup contexts, not the
                raw Search results -- callers get exactly what the LLM was
                grounded on, not an earlier, possibly-different set.

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
            search_results, retrieval_timing = self.retriever.query(query_text=query, k=k)
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
                rag_triggered=True,
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
                rag_triggered=True,
                timing=timing,
            )

        logger.info(f"Retrieved {len(search_results)} documents")

        # Prepare contexts (clean, deduplicate, format citations) -- takes
        # SearchResults directly now, not a hand-converted dict list; see
        # ContextPreparer.prepare().
        prep_start = time.time()
        prepared_contexts = self.context_preparer.prepare(
            search_results,
            sort_by_score=True,
        )
        timing.context_preparation_time = time.time() - prep_start

        # Limit to max_context_chunks
        max_chunks = self.generation_config.max_context_chunks
        prepared_contexts = prepared_contexts[:max_chunks]
        logger.debug(f"Using {len(prepared_contexts)} prepared contexts")

        # Format and surface sources now, ahead of generation -- callers
        # that stream (on_sources set) get them before the LLM call starts,
        # already in the shape /api/ask returns, not the raw pre-dedup
        # search_results a caller might otherwise reach for.
        sources = self._format_sources(prepared_contexts) if include_sources else []
        if on_sources is not None:
            on_sources(sources)

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
                sources=sources if include_sources else [],
                query_type=classification.query_type,
                mode="rag",
                metadata={"error": str(e), "prompt_building_failed": True},
                rag_triggered=True,
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
                temperature=temperature if temperature is not None else self.config.rag.llm.temperature,
                top_p=self.config.rag.llm.top_p,
                max_tokens=max_tokens if max_tokens is not None else self.config.rag.llm.max_tokens,
                stream=stream,
                on_token=on_token,
            )
            timing.llm_generation_time = time.time() - llm_start
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            timing.llm_generation_time = time.time() - llm_start
            timing.total_time = time.time() - overall_start
            return GenerationResult(
                answer=f"I encountered an error while generating the response: {str(e)}",
                sources=sources if include_sources else [],
                query_type=classification.query_type,
                mode="rag",
                metadata={"error": str(e), "llm_generation_failed": True},
                rag_triggered=True,
                timing=timing,
            )

        answer = response.get("message", {}).get("content", "").strip()

        # Post-process answer
        answer = self._post_process_answer(answer, prepared_contexts)

        # sources was already formatted and handed to on_sources, above,
        # before generation started -- reused here, not recomputed.

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
            rag_triggered=True,
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
        voice_mode: bool = False,
        request_id: str | None = None,
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
            voice_mode: If True, use brief response mode for voice conversation
            request_id: Correlation ID for this request, normally minted by
                the caller (backend/api/chat.py, backend/websockets/chat.py)
                via new_request_id() before this is called. Auto-minted here
                if omitted (e.g. direct/test callers), so this is never None
                downstream. Reused as the persisted assistant message's id
                (see SessionManager.add_message's message_id param) and as
                the key of the QueryTrace this call assembles and persists.

        Returns:
            GenerationResult with answer and optional sources
        """
        logger.info(f"Generating chat response for message: {message[:100]}..." + (" [voice_mode]" if voice_mode else ""))
        request_id = request_id or new_request_id()
        request_id_var.set(request_id)  # idempotent safety net if caller didn't already set it
        trace = QueryTrace(request_id=request_id, session_id=session_id, query_text=message)
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
        # Distinct from should_retrieve (whether retrieval was attempted) --
        # this tracks whether an attempted retrieval actually failed, so the
        # two are never conflated in metadata below: "RAG wasn't needed" and
        # "RAG was needed but silently failed" used to look identical to any
        # caller, which is exactly what made an empty/misconfigured
        # knowledge base indistinguishable from working-as-intended chat.
        rag_retrieval_failed = False
        rag_retrieval_error: str | None = None

        # Retrieve documents if needed
        if should_retrieve:
            logger.debug("Triggering RAG retrieval in chat mode")
            try:
                k = self.config.rag.retrieval.default_k
                search_results, retrieval_timing = self.retriever.query(
                    query_text=message, k=k, request_id=request_id, trace=trace
                )

                # Copy retrieval timing
                timing.embedding_time = retrieval_timing.embedding_time
                timing.search_time = retrieval_timing.search_time
                timing.reranking_time = retrieval_timing.reranking_time
                timing.mmr_time = retrieval_timing.mmr_time

                if search_results:
                    context_start = time.time()
                    prepared_contexts = self.context_preparer.prepare(
                        search_results,
                        sort_by_score=True,
                    )
                    timing.context_preparation_time = time.time() - context_start
                    max_chunks = self.generation_config.max_context_chunks
                    prepared_contexts = prepared_contexts[:max_chunks]
                    trace.context_chunks.extend(
                        {
                            "source_file": c.source_file,
                            "citation_text": c.citation_text,
                            "final_score": c.final_score,
                            "length": c.length,
                        }
                        for c in prepared_contexts
                    )
                    logger.info(f"Retrieved and prepared {len(prepared_contexts)} contexts for chat")
            except Exception as e:
                # Chat mode keeps answering conversationally without context
                # on a retrieval failure -- that's the correct degrade-
                # gracefully design here (unlike explicit RAG mode, which
                # returns a visible error/no-results answer instead, see
                # generate_rag_response() above). But silently continuing
                # used to leave zero trace anywhere that retrieval was even
                # attempted, let alone that it failed -- e.g. an empty
                # knowledge base (_check_knowledge_base()'s ValueError) was
                # completely indistinguishable from RAG legitimately not
                # being needed. Log loud and record it in metadata instead,
                # so the failure is at least diagnosable.
                logger.error(f"RAG retrieval in chat mode failed: {e}", exc_info=True)
                rag_retrieval_failed = True
                rag_retrieval_error = str(e)
                trace.rag_retrieval_failed = True
                trace.rag_retrieval_error = rag_retrieval_error
                # Continue without RAG context

        # Build chat prompt (with or without RAG context)
        try:
            prompt_start = time.time()
            prompt_components = self.prompt_builder.build_chat_prompt(
                query=message,
                contexts=prepared_contexts if prepared_contexts else None,
                voice_mode=voice_mode,
            )
            timing.prompt_building_time = time.time() - prompt_start
        except Exception as e:
            logger.error(f"Chat prompt building failed: {e}")
            timing.total_time = time.time() - overall_start
            error_answer = f"I encountered an error while preparing the response: {str(e)}"

            # Mirrors the LLM-generation-failure branch below: an early
            # return here used to skip message/trace persistence entirely,
            # so a prompt-building failure (e.g. a template or context-
            # length error) left no record anywhere -- not the user's
            # message, not even that anything was attempted. See that
            # branch's comment for the full history of this bug class.
            if session_manager and session_id:
                session_manager.add_message(
                    session_id=session_id,
                    role="user",
                    content=message,
                    tokens=len(message) // 4,
                )
                session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=error_answer,
                    tokens=len(error_answer) // 4,
                    model=self.config.rag.llm.model,
                    rag_triggered=False,
                    processing_time_ms=int(timing.total_time * 1000),
                    metadata={"error": str(e), "prompt_building_failed": True, "request_id": request_id},
                    message_id=request_id,
                )
                logger.debug(f"Stored failed-generation messages in session {session_id}")

                # Retrieval (if it ran) already succeeded by this point -- the
                # trace still has real retrieval/context data worth
                # persisting for diagnosis, same reasoning as the
                # LLM-failure branch below.
                trace.model = self.config.rag.llm.model
                trace.timing = asdict(timing)
                session_manager.save_query_trace(trace)

            return GenerationResult(
                answer=error_answer,
                sources=[],
                query_type=classification.query_type,
                mode="chat",
                metadata={"error": str(e), "prompt_building_failed": True, "request_id": request_id},
                rag_triggered=False,
                timing=timing,
            )

        # Convert to Ollama message format
        messages = prompt_components.to_messages()

        # Generate answer using LLM
        logger.debug("Calling LLM for chat generation" + (" [voice_mode: brief response]" if voice_mode else ""))
        try:
            llm_start = time.time()
            
            # Use temperature override if provided
            llm_temperature = temperature if temperature is not None else self.config.rag.llm.temperature
            
            # In voice mode, limit max_tokens for faster, more concise responses
            if voice_mode:
                llm_max_tokens = self.config.conversation_mode.brief_response_max_tokens
                logger.debug(f"Voice mode: limiting max_tokens to {llm_max_tokens}")
            else:
                llm_max_tokens = self.config.rag.llm.max_tokens
            
            response = self.llm_client.generate(
                messages=messages,
                model=self.config.rag.llm.model,
                temperature=llm_temperature,
                top_p=self.config.rag.llm.top_p,
                max_tokens=llm_max_tokens,
                stream=stream,
                on_token=on_token,
            )
            timing.llm_generation_time = time.time() - llm_start
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            timing.llm_generation_time = time.time() - llm_start
            timing.total_time = time.time() - overall_start
            error_answer = f"I encountered an error while generating the response: {str(e)}"

            # This early return used to skip the "store messages in session"
            # block entirely (it only runs after this try/except, on the
            # success path below) -- so a failed generation (e.g. a real
            # Ollama error like "model requires more system memory") left
            # *no trace at all* in the session: not the user's message, not
            # even an error note. Reopening the conversation later showed it
            # as if nothing had ever been sent. Persist both sides here too,
            # mirroring the success path, so the history is honest about
            # what was actually asked and that it failed.
            if session_manager and session_id:
                session_manager.add_message(
                    session_id=session_id,
                    role="user",
                    content=message,
                    tokens=len(message) // 4,
                )
                session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=error_answer,
                    tokens=len(error_answer) // 4,
                    model=self.config.rag.llm.model,
                    rag_triggered=False,
                    processing_time_ms=int(timing.total_time * 1000),
                    metadata={"error": str(e), "llm_generation_failed": True, "request_id": request_id},
                    message_id=request_id,
                )
                logger.debug(f"Stored failed-generation messages in session {session_id}")

                # Retrieval (if it ran) already succeeded by this point -- the
                # LLM call is what failed -- so the trace still has real
                # retrieval/context data worth persisting for diagnosis.
                trace.model = self.config.rag.llm.model
                trace.timing = asdict(timing)
                session_manager.save_query_trace(trace)

            return GenerationResult(
                answer=error_answer,
                sources=[],
                query_type=classification.query_type,
                mode="chat",
                metadata={"error": str(e), "llm_generation_failed": True, "request_id": request_id},
                rag_triggered=False,
                timing=timing,
            )

        answer = response.get("message", {}).get("content", "").strip()

        # Estimate token counts (simple approximation: ~4 chars per token)
        user_tokens = len(message) // 4
        assistant_tokens = len(answer) // 4

        # Format sources if RAG was used
        sources = []
        if include_sources and prepared_contexts:
            sources = self._format_sources(prepared_contexts)

        # Calculate total timing
        timing.total_time = time.time() - overall_start

        # Build metadata (before storing to session)
        metadata = {
            "request_id": request_id,
            "query_type": classification.query_type,
            "rag_retrieval_triggered": should_retrieve,
            "rag_retrieval_failed": rag_retrieval_failed,
            "num_contexts_used": len(prepared_contexts),
            "conversation_turns": len(self.prompt_builder.conversation_history) // 2,
            "total_tokens": prompt_components.total_tokens,
            "llm_model": self.config.rag.llm.model,
        }
        if rag_retrieval_failed:
            metadata["rag_retrieval_error"] = rag_retrieval_error

        # Finalize the trace alongside metadata -- same information, two
        # destinations: metadata is the small per-message summary already
        # exposed to callers/clients; the trace is the full stage-by-stage
        # record kept only in query_traces (see src/generation/trace.py).
        trace.rag_retrieval_triggered = should_retrieve
        trace.model = self.config.rag.llm.model
        trace.timing = asdict(timing)

        # Store messages in session if session_manager provided
        if session_manager and session_id:
            # Store user message
            session_manager.add_message(
                session_id=session_id,
                role="user",
                content=message,
                tokens=user_tokens,
            )

            # Store assistant message with metadata. Reuses request_id as
            # this message's id (see SessionManager.add_message) -- one
            # request = one chat turn = exactly one assistant message row,
            # so messages and query_traces can be joined on the same id
            # with no extra bookkeeping.
            processing_time_ms = int(timing.total_time * 1000) if timing else None

            session_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=answer,
                tokens=assistant_tokens,
                model=self.config.rag.llm.model,
                rag_triggered=should_retrieve,
                processing_time_ms=processing_time_ms,
                metadata=metadata,
                sources=sources if should_retrieve and sources else None,
                message_id=request_id,
            )
            logger.debug(f"Stored messages in session {session_id}")
            session_manager.save_query_trace(trace)
        else:
            # Fallback to prompt builder history (old behavior)
            self.prompt_builder.add_to_history(role="user", content=message)
            self.prompt_builder.add_to_history(role="assistant", content=answer)

        logger.info("Chat response generated successfully")
        
        return GenerationResult(
            answer=answer,
            sources=sources,
            query_type=classification.query_type,
            mode="chat",
            metadata=metadata,
            rag_triggered=should_retrieve,
            timing=timing,
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
        self, answer: str, contexts: list[PreparedContext]
    ) -> str:
        """
        Replace numeric citations with full citation text.

        Converts [1] to (Source Title, p. 42) format.

        Args:
            answer: Answer with numeric citations [1], [2]
            contexts: List of PreparedContext with citation_text

        Returns:
            Answer with expanded citations
        """
        import re

        expanded = answer

        # Process each context
        for idx, ctx in enumerate(contexts, 1):
            citation_text = ctx.citation_text or ctx.normalized_source_file or ctx.source_file or "Unknown"

            # Replace [N] with (citation text)
            pattern = rf'\[{idx}\]'
            replacement = f'({citation_text})'
            expanded = re.sub(pattern, replacement, expanded)

        return expanded

    def _post_process_answer(
        self, answer: str, contexts: list[PreparedContext]
    ) -> str:
        """
        Post-process the generated answer.

        Tasks:
            - Validate citations exist in contexts
            - Remove hallucinated citations
            - Clean up formatting

        Args:
            answer: Raw LLM answer
            contexts: List of PreparedContext used

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

    def _format_sources(self, contexts: list[PreparedContext]) -> list[dict[str, Any]]:
        """
        Format Sources for the response wire format.

        Args:
            contexts: List of PreparedContext (see CONTEXT.md's "Sources" entry)

        Returns:
            List of formatted source dicts (API/wire shape, not part of the
            typed retrieval->generation contract)
        """
        sources = []

        for idx, ctx in enumerate(contexts, 1):
            source: dict[str, Any] = {
                "index": idx,
                "text": ctx.text[:200] + "...",  # Preview
                "score": ctx.final_score,
            }

            if ctx.citation_text:
                source["citation"] = ctx.citation_text

            if ctx.source_file:
                source["source_file"] = ctx.normalized_source_file or ctx.source_file

            if ctx.page is not None:
                source["page"] = ctx.page

            if ctx.title:
                source["title"] = ctx.title

            if ctx.url:
                source["url"] = ctx.url

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
