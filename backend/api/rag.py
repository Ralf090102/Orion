"""
RAG API Endpoints

Endpoints for Retrieval-Augmented Generation:
- Semantic search (query only)
- RAG ask (query + LLM generation)
- Streaming responses
"""

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
                    content=src.get("content", ""),
                    citation=src.get("citation", ""),
                    score=src.get("score", 0.0),
                    metadata=src.get("metadata", {}),
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
                "temperature": request.temperature or config.rag.llm.temperature,
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
    
    Uses the core RAG pipeline from run.py with streaming enabled.
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
        """Generate SSE events for streaming response."""
        try:
            # ===== EXTRACT REQUIRED ARGUMENT =====
            query = request.query
            
            # ===== APPLY OPTIONAL SETTINGS =====
            k = request.k if request.k is not None else config.rag.retrieval.default_k
            include_sources = request.include_sources
            
            logger.info(f"Stream request: '{query}' (k={k}, sources={include_sources})")
            start_time = time.time()
            
            # Retrieve context using retriever (matches run.py pattern)
            results = generator.retriever.query(
                query_text=query,
                k=k,
                formatted=False,
            )
            
            # Prepare and send sources first
            if include_sources and results:
                sources = [
                    {
                        "index": i + 1,
                        "content": r.content[:500],  # Truncate for streaming
                        "citation": r.metadata.get("source", "Unknown"),
                        "score": r.score,
                        "metadata": r.metadata,
                    }
                    for i, r in enumerate(results)
                ]
                
                chunk = StreamChunk(
                    type="sources",
                    content="",
                    data={"sources": sources},
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Prepare context (same as run.py pipeline)
            prepared_context = generator.context_preparer.prepare_context(
                results, request.query
            )
            
            # Build prompt (same as run.py pipeline)
            prompt = generator.prompt_builder.build_rag_prompt(
                query=request.query,
                context=prepared_context,
            )
            
            # Stream LLM response with token callback
            from src.core.llm import generate_response
            
            token_buffer = []
            
            def stream_token(token: str):
                """Collect tokens for streaming."""
                token_buffer.append(token)
            
            # Override generation settings if provided
            temperature = request.temperature or config.rag.llm.temperature
            max_tokens = request.max_tokens
            
            # Generate with streaming (follows run.py pattern)
            generate_response(
                prompt=prompt,
                config=config,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                on_token=stream_token,
            )
            
            # Stream buffered tokens
            for token in token_buffer:
                chunk = StreamChunk(
                    type="token",
                    content=token,
                    data={},
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            
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
