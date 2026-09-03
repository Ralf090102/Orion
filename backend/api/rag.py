"""
RAG API Endpoints

Endpoints for Retrieval-Augmented Generation:
- Semantic search (query only)
- RAG ask (query + LLM generation)
- Streaming responses
"""

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.dependencies import (
    get_config_dependency,
    get_generator_dependency,
    get_retriever_dependency,
)
from backend.models.rag import (
    AskRequest,
    AskResponse,
    QueryRequest,
    QueryResponse,
    SearchResult,
    Source,
    StreamChunk,
    TimingBreakdown,
)
from src.generation.generate import AnswerGenerator
from src.retrieval.retriever import OrionRetriever
from src.utilities.config import OrionConfig

logger = logging.getLogger(__name__)

router = APIRouter()


# ========== SEMANTIC QUERY (RETRIEVAL ONLY) ==========
@router.post(
    "/api/query",
    response_model=QueryResponse,
    summary="Semantic search",
    description="Search knowledge base using semantic similarity (no LLM)",
    tags=["RAG"],
)
async def query_knowledge_base(
    request: QueryRequest,
    retriever: OrionRetriever = Depends(get_retriever_dependency),
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Perform semantic search on the knowledge base.
    
    Separates required argument (query) from optional settings.
    Matches run.py query command pattern:
    - Required: query string
    - Optional: k, enable_reranking, similarity_threshold, verbose
    
    Optional settings override config defaults when provided.
    
    Args:
        request: Query request with search parameters
        retriever: Retriever instance (injected)
        config: Configuration instance (injected)
        
    Returns:
        QueryResponse with search results and metadata
        
    Raises:
        HTTPException: If knowledge base is empty or search fails
    """
    try:
        start_time = time.time()
        
        # ===== EXTRACT REQUIRED ARGUMENT =====
        query = request.query
        
        # ===== APPLY OPTIONAL SETTINGS (override config defaults) =====
        k = request.k if request.k is not None else config.rag.retrieval.default_k
        enable_reranking = (
            request.enable_reranking
            if request.enable_reranking is not None
            else config.rag.retrieval.enable_reranking
        )
        similarity_threshold = (
            request.similarity_threshold
            if request.similarity_threshold is not None
            else config.rag.retrieval.similarity_threshold
        )
        verbose = request.verbose
        
        logger.info(
            f"Query request: '{query}' (k={k}, reranking={enable_reranking}, "
            f"threshold={similarity_threshold}, verbose={verbose})"
        )
        
        # Perform retrieval (matches run.py pattern)
        results = retriever.query(
            query_text=query,
            k=k,
            enable_reranking=enable_reranking,
            formatted=False,  # Get raw SearchResult objects
        )
        
        # Filter by similarity threshold
        original_count = len(results)
        results = [r for r in results if r.score >= similarity_threshold]
        filtered_count = original_count - len(results)
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} results below threshold {similarity_threshold}")
        
        processing_time = time.time() - start_time
        
        # Convert to response model
        search_results = [
            SearchResult(
                content=r.content,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]
        
        logger.info(
            f"Query completed: {len(search_results)} results in {processing_time:.3f}s"
        )
        
        # Build response metadata
        response_metadata = {
            "k_requested": k,
            "k_returned": len(search_results),
            "filtered_count": filtered_count,
            "reranking_enabled": enable_reranking,
            "hybrid_search": config.rag.retrieval.enable_hybrid_search,
            "mmr_enabled": config.rag.retrieval.enable_mmr,
            "similarity_threshold": similarity_threshold,
        }
        
        # Add verbose timing if requested
        if verbose:
            response_metadata["timing"] = {
                "total_time": processing_time,
                "results_per_second": len(search_results) / processing_time if processing_time > 0 else 0,
            }
        
        return QueryResponse(
            results=search_results,
            total_results=len(search_results),
            query=query,
            processing_time=processing_time,
            metadata=response_metadata,
        )
        
    except ValueError as e:
        # Knowledge base empty or validation error
        logger.warning(f"Query validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


# ========== RAG ASK (RETRIEVAL + LLM) ==========
@router.post(
    "/api/ask",
    response_model=AskResponse,
    summary="RAG question answering",
    description="Answer questions using RAG (retrieval + LLM generation)",
    tags=["RAG"],
)
async def ask_question(
    request: AskRequest,
    generator: AnswerGenerator = Depends(get_generator_dependency),
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Answer a question using RAG pipeline.
    
    Separates required argument (query) from optional settings.
    Matches run.py ask command pattern:
    - Required: query string
    - Optional: k, include_sources, temperature, max_tokens, verbose
    
    When verbose=True, includes detailed timing breakdown (matches run.py --verbose flag).
    
    Args:
        request: Ask request with question and parameters
        generator: Generator instance (injected)
        config: Configuration instance (injected)
        
    Returns:
        AskResponse with answer, sources, and optional timing breakdown
        
    Raises:
        HTTPException: If generation fails or knowledge base is empty
    """
    try:
        start_time = time.time()
        
        # ===== EXTRACT REQUIRED ARGUMENT =====
        query = request.query
        
        # ===== APPLY OPTIONAL SETTINGS (override config defaults) =====
        k = request.k if request.k is not None else config.rag.retrieval.default_k
        include_sources = request.include_sources
        verbose = request.verbose
        
        logger.info(
            f"Ask request: '{query}' (k={k}, sources={include_sources}, verbose={verbose})"
        )
        
        # Build generation kwargs (matches run.py pattern)
        generation_kwargs = {}
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_kwargs["max_tokens"] = request.max_tokens
        
        # Generate RAG response (same as run.py ask command)
        result = generator.generate_rag_response(
            query=query,
            k=k,
            include_sources=include_sources,
            **generation_kwargs,
        )
        
        processing_time = time.time() - start_time
        
        # Convert sources to response model (matches run.py source handling)
        sources = []
        if include_sources and result.sources:
            sources = [
                Source(
                    index=i + 1,
                    content=src.get("text", ""),
                    citation=src.get("citation", ""),
                    score=src.get("score", 0.0),
                    metadata={
                        k: v
                        for k, v in {
                            "title": src.get("title"),
                            "source_file": src.get("source_file"),
                            "page": src.get("page"),
                        }.items()
                        if v is not None
                    },
                )
                for i, src in enumerate(result.sources)
            ]
        
        # Extract timing breakdown if verbose mode (matches run.py --verbose)
        timing_breakdown = None
        if verbose and result.timing:
            timing_breakdown = TimingBreakdown(
                embedding_time=result.timing.embedding_time,
                search_time=result.timing.search_time,
                reranking_time=result.timing.reranking_time,
                llm_generation_time=result.timing.llm_generation_time,
                total_time=result.timing.total_time,
            )
        
        logger.info(
            f"Ask completed: {len(result.answer)} chars, "
            f"{len(sources)} sources in {processing_time:.3f}s"
        )
        
        return AskResponse(
            answer=result.answer,
            sources=sources,
            query=query,
            query_type=result.query_type,
            processing_time=processing_time,
            metadata={
                "mode": result.mode,
                "model": config.rag.llm.model,
                "temperature": request.temperature if request.temperature is not None else config.rag.llm.temperature,
                "k": k,
            },
            timing=timing_breakdown,  # Only included when verbose=True
        )
        
    except ValueError as e:
        logger.warning(f"Ask validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Ask failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ask failed: {str(e)}",
        )


# ========== STREAMING RAG ASK ==========
@router.post(
    "/api/ask/stream",
    summary="Streaming RAG response",
    description="Stream answer tokens in real-time (Server-Sent Events)",
    tags=["RAG"],
)
async def ask_stream(
    request: AskRequest,
    generator: AnswerGenerator = Depends(get_generator_dependency),
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Stream RAG answer tokens in real-time.
    
    Returns Server-Sent Events (SSE) stream with:
    - Token chunks as they're generated
    - Source citations
    - Metadata
    - Done signal
    
    Delegates to AnswerGenerator.generate_rag_response(stream=True, ...) --
    the same pipeline /api/ask uses, non-streaming -- offloaded to a worker
    thread so it doesn't block the event loop.
    Separate endpoint from /api/ask due to fundamentally different response type (SSE vs JSON).
    
    Args:
        request: Ask request with question and parameters
        generator: Generator instance (injected)
        config: Configuration instance (injected)
        
    Returns:
        StreamingResponse with SSE events
        
    Raises:
        HTTPException: If streaming fails
    """
    async def event_generator():
        """Generate SSE events for streaming response.

        Streams through AnswerGenerator.generate_rag_response() -- the same
        interface /api/ask uses -- instead of hand-rolling retrieve/prepare/
        prompt/generate here. The LLM call is a blocking sync call, so it's
        offloaded via asyncio.to_thread; on_token/on_sources run on that
        worker thread and hop back onto the event loop via
        call_soon_threadsafe, the same shape ChatWebSocketHandler's
        queue_token() uses (see backend/websockets/chat.py) -- kept as a
        separate, local implementation rather than a shared helper, since
        that handler's queue is tied to a connection's lifecycle and this
        one is a one-shot local queue for a single request.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()
        tokens_sent = 0

        def _put_nowait(item: tuple[str, Any] | None):
            """Actually enqueue; must run on the event-loop thread."""
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                logger.warning("Stream queue full, dropping event")

        def on_sources(sources: list[dict]):
            loop.call_soon_threadsafe(_put_nowait, ("sources", sources))

        def on_token(token: str):
            nonlocal tokens_sent
            tokens_sent += 1
            loop.call_soon_threadsafe(_put_nowait, ("token", token))

        try:
            query = request.query
            k = request.k if request.k is not None else config.rag.retrieval.default_k
            include_sources = request.include_sources
            temperature = request.temperature if request.temperature is not None else config.rag.llm.temperature
            max_tokens = request.max_tokens

            logger.info(f"Stream request: '{query}' (k={k}, sources={include_sources})")
            start_time = time.time()

            async def run_generation():
                try:
                    return await asyncio.to_thread(
                        generator.generate_rag_response,
                        query=query,
                        k=k,
                        include_sources=include_sources,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        on_token=on_token,
                        on_sources=on_sources if include_sources else None,
                    )
                finally:
                    # Sentinel: tells the queue-draining loop below there's
                    # nothing more coming, success or failure either way.
                    loop.call_soon_threadsafe(_put_nowait, None)

            gen_task = asyncio.create_task(run_generation())

            while True:
                item = await queue.get()
                if item is None:
                    break
                kind, payload = item
                if kind == "sources":
                    chunk = StreamChunk(type="sources", content="", data={"sources": payload})
                else:
                    chunk = StreamChunk(type="token", content=payload, data={})
                yield f"data: {chunk.model_dump_json()}\n\n"

            result = await gen_task

            # generate_rag_response() catches retrieval/prompt/LLM failures
            # internally and returns a normal GenerationResult with a
            # friendly result.answer describing what went wrong, rather
            # than raising -- so no tokens streamed via on_token above in
            # that case. Forward it as a token now, the same way
            # ChatWebSocketHandler does for its own generation failures,
            # so the client sees an apologetic message instead of nothing.
            if tokens_sent == 0 and result.answer:
                token_chunk = StreamChunk(type="token", content=result.answer, data={})
                yield f"data: {token_chunk.model_dump_json()}\n\n"

            # Send metadata
            processing_time = time.time() - start_time
            metadata_chunk = StreamChunk(
                type="metadata",
                content="",
                data={
                    "processing_time": processing_time,
                    "model": config.rag.llm.model,
                    "temperature": temperature,
                    "k": request.k,
                },
            )
            yield f"data: {metadata_chunk.model_dump_json()}\n\n"

            # Send done signal
            done_chunk = StreamChunk(
                type="done",
                content="",
                data={},
            )
            yield f"data: {done_chunk.model_dump_json()}\n\n"

            logger.info(f"Stream completed in {processing_time:.3f}s")

        except Exception as e:
            logger.error(f"Stream failed: {e}", exc_info=True)
            error_chunk = StreamChunk(
                type="error",
                content=str(e),
                data={},
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
