"""
Chat API Endpoints

Endpoints for conversational chat with session management:
- Session CRUD operations
- Message sending (streaming and non-streaming)
- Message history retrieval
"""

import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.dependencies import (
    get_config_dependency,
    get_generator_dependency,
    get_session_manager_dependency,
)
from backend.models.chat import (
    BranchesResponse,
    BranchInfo,
    BranchResponse,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    CreateBranchRequest,
    CreateSessionRequest,
    DeleteMessageRequest,
    DeleteMessageResponse,
    DeleteSessionResponse,
    Message,
    MessageHistoryResponse,
    SessionInfo,
    SessionListResponse,
    SessionResponse,
    SwitchBranchRequest,
    SwitchBranchResponse,
    UpdateSessionRequest,
)
from src.generation.generate import AnswerGenerator
from src.generation.session_manager import SessionManager
from src.utilities.config import OrionConfig

logger = logging.getLogger(__name__)

router = APIRouter()


# ========== SESSION CRUD OPERATIONS ==========
@router.post(
    "/api/chat/sessions",
    response_model=SessionResponse,
    summary="Create new chat session",
    description="Create a new chat session with optional custom ID and metadata",
    tags=["Chat"],
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    request: CreateSessionRequest,
    session_manager: SessionManager = Depends(get_session_manager_dependency),
):
    """
    Create a new chat session.
    
    Args:
        request: Session creation request with optional ID and metadata
        session_manager: Session manager instance (injected)
        
    Returns:
        SessionResponse with created session info
        
    Raises:
        HTTPException: If session creation fails
    """
    try:
        logger.info(f"Creating new chat session (custom_id={request.session_id})")
        
        # Extract user and topic from metadata if present
        user = request.metadata.get("user") if request.metadata else None
        topic = request.metadata.get("topic") if request.metadata else None
        
        # Create session with metadata
        session_id = session_manager.create_session(
            session_id=request.session_id,
            user=user,
            topic=topic,
            metadata=request.metadata,
        )
        
        # Get the created session
        session = session_manager.get_session(session_id)
        
        # Convert to response model
        session_info = SessionInfo(
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=len(session.messages),
            metadata=session.metadata,
        )
        
        logger.info(f"Created session: {session_id}")
        
        return SessionResponse(
            status="success",
            message="Session created successfully",
            session=session_info,
        )
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}",
        )


@router.get(
    "/api/chat/sessions",
    response_model=SessionListResponse,
    summary="List all chat sessions",
    description="Retrieve all chat sessions with basic info",
    tags=["Chat"],
)
async def list_sessions(
    session_manager: SessionManager = Depends(get_session_manager_dependency),
):
    """
    List all chat sessions.
    
    Args:
        session_manager: Session manager instance (injected)
        
    Returns:
        SessionListResponse with list of sessions
    """
    try:
        logger.info("Listing all chat sessions")
        
        # Get all sessions (returns list of dicts)
        sessions_data = session_manager.list_sessions()
        
        # Convert to response models
        session_infos = [
            SessionInfo(
                session_id=s["session_id"],
                created_at=s["created_at"],
                updated_at=s["updated_at"],
                message_count=s["message_count"],
                metadata=s["metadata"],
            )
            for s in sessions_data
        ]
        
        logger.info(f"Found {len(session_infos)} sessions")
        
        return SessionListResponse(
            sessions=session_infos,
            total=len(session_infos),
        )
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}",
        )


@router.get(
    "/api/chat/sessions/{session_id}",
    response_model=MessageHistoryResponse,
    summary="Get session details",
    description="Retrieve session details including full message history",
    tags=["Chat"],
)
async def get_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager_dependency),
):
    """
    Get session details with message history.
    
    Args:
        session_id: Session identifier
        session_manager: Session manager instance (injected)
        
    Returns:
        MessageHistoryResponse with session messages
        
    Raises:
        HTTPException: If session not found
    """
    try:
        logger.info(f"Retrieving session: {session_id}")
        
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}",
            )
        
        # Convert messages to response model
        messages = [
            Message(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                tokens=msg.get("tokens", 0),
                timestamp=msg.get("timestamp", ""),
            )
            for msg in session.messages
        ]
        
        total_tokens = sum(m.tokens for m in messages)
        
        logger.info(f"Session {session_id}: {len(messages)} messages, {total_tokens} tokens")
        
        return MessageHistoryResponse(
            session_id=session_id,
            messages=messages,
            total_messages=len(messages),
            total_tokens=total_tokens,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session: {str(e)}",
        )


@router.delete(
    "/api/chat/sessions/{session_id}",
    response_model=DeleteSessionResponse,
    summary="Delete chat session",
    description="Delete a specific chat session and its history",
    tags=["Chat"],
)
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager_dependency),
):
    """
    Delete a chat session.
    
    Args:
        session_id: Session identifier
        session_manager: Session manager instance (injected)
        
    Returns:
        DeleteSessionResponse with deletion status
        
    Raises:
        HTTPException: If session not found or deletion fails
    """
    try:
        logger.info(f"Deleting session: {session_id}")
        
        # Check if session exists
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}",
            )
        
        # Delete session
        session_manager.delete_session(session_id)
        
        logger.info(f"Deleted session: {session_id}")
        
        return DeleteSessionResponse(
            status="success",
            message="Session deleted successfully",
            session_id=session_id,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}",
        )


@router.patch(
    "/api/chat/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Update session metadata",
    description="Update session title and/or metadata",
    tags=["Chat"],
)
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    session_manager: SessionManager = Depends(get_session_manager_dependency),
):
    """
    Update session metadata (title, tags, etc.).
    
    Args:
        session_id: Session identifier
        request: Update request with title and/or metadata
        session_manager: Session manager instance (injected)
        
    Returns:
        SessionResponse with updated session info
        
    Raises:
        HTTPException: If session not found or update fails
    """
    try:
        logger.info(f"Updating session {session_id}: title={request.title}")
        
        # Check if session exists
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}",
            )
        
        # Build metadata update
        metadata_update = request.metadata.copy() if request.metadata else {}
        if request.title is not None:
            metadata_update["title"] = request.title
        
        # Update session metadata
        if metadata_update:
            session_manager.update_session_metadata(session_id, metadata_update)
            logger.info(f"Updated session {session_id} metadata: {list(metadata_update.keys())}")
        
        # Get updated session
        updated_session = session_manager.get_session(session_id)
        
        return SessionResponse(
            status="success",
            message="Session updated successfully",
            session=SessionInfo(
                session_id=updated_session.session_id,
                created_at=updated_session.created_at,
                updated_at=updated_session.updated_at,
                message_count=len(updated_session.messages),
                metadata=updated_session.metadata,
            ),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update session: {str(e)}",
        )


@router.delete(
    "/api/chat/sessions",
    response_model=dict[str, Any],
    summary="Delete all chat sessions",
    description="Delete all chat sessions (requires confirmation)",
    tags=["Chat"],
)
async def delete_all_sessions(
    session_manager: SessionManager = Depends(get_session_manager_dependency),
):
    """
    Delete all chat sessions.
    
    Args:
        session_manager: Session manager instance (injected)
        
    Returns:
        Dictionary with deletion status and count
    """
    try:
        logger.warning("Deleting ALL chat sessions")
        
        # Get count before deletion
        sessions = session_manager.list_sessions()
        count = len(sessions)
        
        # Delete all sessions
        session_manager.clear_all_sessions()
        
        logger.info(f"Deleted {count} sessions")
        
        return {
            "status": "success",
            "message": f"Deleted {count} sessions",
            "deleted_count": count,
        }
        
    except Exception as e:
        logger.error(f"Failed to delete all sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete sessions: {str(e)}",
        )


# ========== MESSAGE BRANCHING OPERATIONS ==========
@router.delete(
    "/api/chat/sessions/{session_id}/messages/{message_id}",
    response_model=DeleteMessageResponse,
    summary="Delete message and children",
    description="Delete a message and all its descendant messages (for retry/edit)",
    tags=["Chat", "Branching"],
)
async def delete_message(
    session_id: str,
    message_id: str,
    session_manager: SessionManager = Depends(get_session_manager_dependency),
):
    """
    Delete a message and all its children.
    
    Useful for retry/edit functionality where you want to remove
    a message and its response before creating a new attempt.
    
    Args:
        session_id: Session identifier
        message_id: Message ID to delete
        session_manager: Session manager instance (injected)
        
    Returns:
        DeleteMessageResponse with deletion status
        
    Raises:
        HTTPException: If session or message not found
    """
    try:
        logger.info(f"Deleting message {message_id} from session {session_id}")
        
        # Check if session exists
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}",
            )
        
        # Count messages before deletion
        before_count = len(session.messages)
        
        # Delete message and children
        success = session_manager.delete_message_and_children(session_id, message_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Message not found: {message_id}",
            )
        
        # Count deleted messages
        after_count = len(session.messages)
        deleted_count = before_count - after_count
        
        logger.info(f"Deleted {deleted_count} message(s) from session {session_id}")
        
        return DeleteMessageResponse(
            status="success",
            message=f"Deleted message and {deleted_count - 1} children",
            deleted_count=deleted_count,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete message {message_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete message: {str(e)}",
        )


@router.post(
    "/api/chat/sessions/{session_id}/branches",
    response_model=BranchResponse,
    summary="Create conversation branch",
    description="Create an alternative message branch from a parent (for retry/edit)",
    tags=["Chat", "Branching"],
    status_code=status.HTTP_201_CREATED,
)
async def create_branch(
    session_id: str,
    request: CreateBranchRequest,
    session_manager: SessionManager = Depends(get_session_manager_dependency),
):
    """
    Create a new conversation branch.
    
    Used for retry/edit functionality to create alternative responses
    while preserving the conversation tree structure.
    
    Args:
        session_id: Session identifier
        request: Branch creation request
        session_manager: Session manager instance (injected)
        
    Returns:
        BranchResponse with new message ID
        
    Raises:
        HTTPException: If session or parent message not found
    """
    try:
        logger.info(f"Creating branch in session {session_id} from parent {request.parent_id}")
        
        # Check if session exists
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}",
            )
        
        # Create branch
        new_message_id = session_manager.create_branch(
            session_id=session_id,
            parent_id=request.parent_id,
            role=request.role,
            content=request.content,
            tokens=request.tokens,
            deactivate_siblings=request.deactivate_siblings,
        )
        
        if not new_message_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Parent message not found: {request.parent_id}",
            )
        
        logger.info(f"Created branch {new_message_id} from {request.parent_id}")
        
        return BranchResponse(
            status="success",
            message="Branch created successfully",
            message_id=new_message_id,
            parent_id=request.parent_id,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create branch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create branch: {str(e)}",
        )


@router.get(
    "/api/chat/sessions/{session_id}/branches",
    response_model=BranchesResponse,
    summary="Get message branches",
    description="Get all alternative branches from a parent message",
    tags=["Chat", "Branching"],
)
async def get_branches(
    session_id: str,
    parent_id: Optional[str] = None,
    session_manager: SessionManager = Depends(get_session_manager_dependency),
):
    """
    Get all alternative branches from a parent message.
    
    Used to display alternative responses in the UI (e.g., "2 other responses").
    
    Args:
        session_id: Session identifier
        parent_id: Parent message ID (None for root messages)
        session_manager: Session manager instance (injected)
        
    Returns:
        BranchesResponse with list of branches
        
    Raises:
        HTTPException: If session not found
    """
    try:
        logger.info(f"Getting branches for session {session_id}, parent {parent_id}")
        
        # Check if session exists
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}",
            )
        
        # Get branches
        branches_data = session_manager.get_message_branches(session_id, parent_id)
        
        # Convert to response model
        branches = [
            BranchInfo(
                message_id=b["message_id"],
                role=b["role"],
                content=b["content"],
                tokens=b.get("tokens", 0),
                timestamp=b["timestamp"],
                is_active=b.get("is_active", True),
            )
            for b in branches_data
        ]
        
        logger.info(f"Found {len(branches)} branch(es)")
        
        return BranchesResponse(
            status="success",
            parent_id=parent_id,
            branches=branches,
            total=len(branches),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get branches: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get branches: {str(e)}",
        )


@router.post(
    "/api/chat/sessions/{session_id}/branches/switch",
    response_model=SwitchBranchResponse,
    summary="Switch active branch",
    description="Switch to a different conversation branch",
    tags=["Chat", "Branching"],
)
async def switch_branch(
    session_id: str,
    request: SwitchBranchRequest,
    session_manager: SessionManager = Depends(get_session_manager_dependency),
):
    """
    Switch to a different conversation branch.
    
    Deactivates the current active branch and activates the specified message,
    allowing users to explore different conversation paths.
    
    Args:
        session_id: Session identifier
        request: Switch branch request with message ID
        session_manager: Session manager instance (injected)
        
    Returns:
        SwitchBranchResponse with switch status
        
    Raises:
        HTTPException: If session or message not found
    """
    try:
        logger.info(f"Switching branch in session {session_id} to {request.message_id}")
        
        # Check if session exists
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}",
            )
        
        # Switch branch
        success = session_manager.switch_branch(session_id, request.message_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Message not found: {request.message_id}",
            )
        
        logger.info(f"Switched to branch {request.message_id}")
        
        return SwitchBranchResponse(
            status="success",
            message="Branch switched successfully",
            message_id=request.message_id,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to switch branch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch branch: {str(e)}",
        )


@router.get(
    "/api/chat/sessions/{session_id}/messages/active",
    response_model=MessageHistoryResponse,
    summary="Get active branch messages",
    description="Get only messages from the active conversation branch",
    tags=["Chat", "Branching"],
)
async def get_active_messages(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager_dependency),
):
    """
    Get only messages from the active conversation branch.
    
    Returns a linear conversation history following the active path
    through the conversation tree, ignoring inactive branches.
    
    Args:
        session_id: Session identifier
        session_manager: Session manager instance (injected)
        
    Returns:
        MessageHistoryResponse with active branch messages
        
    Raises:
        HTTPException: If session not found
    """
    try:
        logger.info(f"Getting active branch messages for session {session_id}")
        
        # Check if session exists
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}",
            )
        
        # Get active branch messages
        conversation_messages = session_manager.get_active_branch_messages(session_id)
        
        # Convert to response model
        messages = [
            Message(
                role=msg.role,
                content=msg.content,
                tokens=msg.tokens,
                timestamp="",  # ConversationMessage doesn't have timestamp
            )
            for msg in conversation_messages
        ]
        
        total_tokens = sum(m.tokens for m in messages)
        
        logger.info(f"Session {session_id}: {len(messages)} active messages, {total_tokens} tokens")
        
        return MessageHistoryResponse(
            session_id=session_id,
            messages=messages,
            total_messages=len(messages),
            total_tokens=total_tokens,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get active messages: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active messages: {str(e)}",
        )


# ========== CHAT MESSAGING ==========
@router.post(
    "/api/chat/sessions/{session_id}/message",
    response_model=ChatResponse,
    summary="Send chat message",
    description="Send a message to the chat session and receive response (non-streaming)",
    tags=["Chat"],
)
async def send_message(
    session_id: str,
    request: ChatRequest,
    generator: AnswerGenerator = Depends(get_generator_dependency),
    session_manager: SessionManager = Depends(get_session_manager_dependency),
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Send a message to the chat session.
    
    Matches run.py chat command behavior:
    - RAG trigger mode: always, auto, manual, never
    - Session-based conversation history
    - Optional source citations
    
    Args:
        session_id: Session identifier
        request: Chat message request
        generator: Generator instance (injected)
        session_manager: Session manager instance (injected)
        config: Configuration instance (injected)
        
    Returns:
        ChatResponse with assistant's reply
        
    Raises:
        HTTPException: If session not found or generation fails
    """
    try:
        start_time = time.time()
        
        # ===== EXTRACT REQUIRED ARGUMENT =====
        message = request.message
        
        # ===== APPLY OPTIONAL SETTINGS =====
        rag_mode = request.rag_mode or config.rag.generation.rag_trigger_mode
        include_sources = request.include_sources
        
        logger.info(
            f"Chat message in session {session_id}: '{message}' "
            f"(rag_mode={rag_mode}, sources={include_sources})"
        )
        
        # Check if session exists
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}",
            )
        
        # Build generation kwargs
        generation_kwargs = {}
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature
        
        # Generate chat response (matches run.py chat command)
        result = generator.generate_chat_response(
            message=message,
            session_id=session_id,
            session_manager=session_manager,
            rag_mode=rag_mode,
            include_sources=include_sources,
            **generation_kwargs,
        )
        
        processing_time = time.time() - start_time
        
        # Extract sources if available
        sources = []
        if include_sources and hasattr(result, "sources") and result.sources:
            sources = [
                {
                    "index": i + 1,
                    "citation": src.get("citation", ""),
                    "content": src.get("content", "")[:200],  # Truncate
                    "score": src.get("score", 0.0),
                }
                for i, src in enumerate(result.sources)
            ]
        
        # Build metadata
        metadata = {
            "rag_retrieval_triggered": result.rag_triggered,
            "query_type": getattr(result, "query_type", "conversational"),
            "model": config.rag.llm.model,
            "rag_mode": rag_mode,
        }
        
        logger.info(
            f"Chat response generated: {len(result.answer)} chars, "
            f"RAG={result.rag_triggered}, {processing_time:.3f}s"
        )
        
        return ChatResponse(
            session_id=session_id,
            message=result.answer,
            sources=sources,
            metadata=metadata,
            processing_time=processing_time,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat message failed in session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat message failed: {str(e)}",
        )


# ========== STREAMING CHAT ==========
@router.post(
    "/api/chat/sessions/{session_id}/stream",
    summary="Send streaming chat message",
    description="Send a message and stream the response tokens in real-time (SSE)",
    tags=["Chat"],
)
async def send_message_stream(
    session_id: str,
    request: ChatRequest,
    generator: AnswerGenerator = Depends(get_generator_dependency),
    session_manager: SessionManager = Depends(get_session_manager_dependency),
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Stream chat response in real-time.
    
    Returns Server-Sent Events (SSE) stream with:
    - Token chunks as they're generated
    - Source citations (if RAG triggered)
    - Metadata
    - Done signal
    
    Matches run.py chat streaming behavior.
    
    Args:
        session_id: Session identifier
        request: Chat message request
        generator: Generator instance (injected)
        session_manager: Session manager instance (injected)
        config: Configuration instance (injected)
        
    Returns:
        StreamingResponse with SSE events
        
    Raises:
        HTTPException: If session not found or streaming fails
    """
    async def event_generator():
        """Generate SSE events for streaming chat response."""
        try:
            # ===== EXTRACT REQUIRED ARGUMENT =====
            message = request.message
            
            # ===== APPLY OPTIONAL SETTINGS =====
            rag_mode = request.rag_mode or config.rag.generation.rag_trigger_mode
            include_sources = request.include_sources
            
            logger.info(
                f"Streaming chat in session {session_id}: '{message}' "
                f"(rag_mode={rag_mode})"
            )
            
            # Check if session exists
            session = session_manager.get_session(session_id)
            if not session:
                error_chunk = ChatStreamChunk(
                    type="error",
                    content=f"Session not found: {session_id}",
                    data={"code": 404},
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
                return
            
            start_time = time.time()
            
            # Build generation kwargs
            generation_kwargs = {"stream": True}
            if request.temperature is not None:
                generation_kwargs["temperature"] = request.temperature
            
            # Token buffer for streaming
            token_buffer = []
            
            def stream_token(token: str):
                """Collect tokens for streaming."""
                token_buffer.append(token)
            
            # Generate chat response with streaming
            result = generator.generate_chat_response(
                message=message,
                session_id=session_id,
                session_manager=session_manager,
                rag_mode=rag_mode,
                include_sources=include_sources,
                on_token=stream_token,
                **generation_kwargs,
            )
            
            # Stream buffered tokens
            for token in token_buffer:
                chunk = ChatStreamChunk(
                    type="token",
                    content=token,
                    data={},
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Send sources if available
            if include_sources and result.rag_triggered and hasattr(result, "sources") and result.sources:
                sources = [
                    {
                        "index": i + 1,
                        "citation": src.get("citation", ""),
                        "score": src.get("score", 0.0),
                    }
                    for i, src in enumerate(result.sources)
                ]
                
                sources_chunk = ChatStreamChunk(
                    type="sources",
                    content="",
                    data={"sources": sources},
                )
                yield f"data: {sources_chunk.model_dump_json()}\n\n"
            
            # Send metadata
            processing_time = time.time() - start_time
            metadata_chunk = ChatStreamChunk(
                type="metadata",
                content="",
                data={
                    "rag_triggered": result.rag_triggered,
                    "query_type": getattr(result, "query_type", "conversational"),
                    "model": config.rag.llm.model,
                    "rag_mode": rag_mode,
                },
            )
            yield f"data: {metadata_chunk.model_dump_json()}\n\n"
            
            # Send done signal
            done_chunk = ChatStreamChunk(
                type="done",
                content="",
                data={
                    "processing_time": processing_time,
                    "session_id": session_id,
                },
            )
            yield f"data: {done_chunk.model_dump_json()}\n\n"
            
            logger.info(f"Stream completed in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Stream failed in session {session_id}: {e}", exc_info=True)
            error_chunk = ChatStreamChunk(
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
            "X-Accel-Buffering": "no",
        },
    )
