"""
WebSocket Chat Handler

Real-time bidirectional chat via WebSocket with robust token streaming.
Compatible with HuggingFace chat-ui and other WebSocket clients.
"""

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect, status

from backend.models.chat import WebSocketMessage
from src.generation.generate import AnswerGenerator
from src.generation.session_manager import SessionManager
from src.utilities.config import OrionConfig

logger = logging.getLogger(__name__)


class ChatWebSocketHandler:
    """Handler for WebSocket chat connections with robust token streaming."""
    
    def __init__(
        self,
        websocket: WebSocket,
        session_id: str,
        session_manager: SessionManager,
        generator: AnswerGenerator,
        config: OrionConfig,
    ):
        """
        Initialize WebSocket handler.
        
        Args:
            websocket: WebSocket connection
            session_id: Chat session identifier
            session_manager: Session manager instance
            generator: Answer generator instance
            config: Configuration instance
        """
        self.websocket = websocket
        self.session_id = session_id
        self.session_manager = session_manager
        self.generator = generator
        self.config = config
        self.connected = False
        
        # Token streaming queue (fixes fire-and-forget issues)
        self.token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._queue_task: asyncio.Task | None = None
        self._is_processing = False
    
    async def _process_token_queue(self):
        """
        Process tokens from the queue and send them to the client.
        
        This runs in a dedicated task to ensure tokens are sent in order
        without race conditions or dropped messages.
        """
        logger.debug(f"Token queue processor started for session {self.session_id}")
        
        try:
            while self.connected or not self.token_queue.empty():
                try:
                    # Get token with timeout to allow checking connected status
                    token = await asyncio.wait_for(
                        self.token_queue.get(),
                        timeout=0.1
                    )
                    
                    # None is sentinel value for "done streaming"
                    if token is None:
                        logger.debug("Received end-of-stream signal")
                        break
                    
                    # Send token to client
                    await self.send_message(message_type="token", content=token)
                    self.token_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No token available, loop again to check connection status
                    continue
                except Exception as e:
                    logger.error(f"Error processing token: {e}", exc_info=True)
                    break
        
        finally:
            logger.debug(f"Token queue processor stopped for session {self.session_id}")
            self._is_processing = False
    
    def queue_token(self, token: str):
        """
        Queue a token for sending (thread-safe).
        
        This is called from the sync streaming callback and safely
        adds tokens to the async queue.
        
        Args:
            token: Token content to send
        """
        try:
            # Put token in queue (non-blocking)
            self.token_queue.put_nowait(token)
        except asyncio.QueueFull:
            logger.warning(f"Token queue full, dropping token: {token[:20]}...")
        except Exception as e:
            logger.error(f"Failed to queue token: {e}")
    
    async def start_token_streaming(self):
        """Start the token queue processor task."""
        if not self._is_processing:
            self._is_processing = True
            self._queue_task = asyncio.create_task(self._process_token_queue())
            logger.debug("Started token streaming task")
    
    async def stop_token_streaming(self):
        """Stop the token queue processor and signal end of stream."""
        # Signal end of stream
        await self.token_queue.put(None)
        
        # Wait for queue to finish processing
        if self._queue_task and not self._queue_task.done():
            try:
                await asyncio.wait_for(self._queue_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Token queue processor did not finish in time")
                if not self._queue_task.done():
                    self._queue_task.cancel()
        
        logger.debug("Stopped token streaming task")
    
    async def connect(self) -> bool:
        """
        Accept WebSocket connection and verify session.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            await self.websocket.accept()
            self.connected = True
            
            # Verify session exists
            session = self.session_manager.get_session(self.session_id)
            if not session:
                logger.warning(f"Session not found: {self.session_id}")
                await self.send_error(
                    f"Session not found: {self.session_id}",
                    code=404,
                )
                return False
            
            logger.info(f"WebSocket connected for session: {self.session_id}")
            
            # Send connection success
            await self.send_message(
                message_type="connected",
                data={
                    "session_id": self.session_id,
                    "message": "WebSocket connection established",
                },
            )
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}", exc_info=True)
            self.connected = False
            return False
    
    async def disconnect(self):
        """Close WebSocket connection and cleanup resources."""
        if self.connected:
            try:
                # Stop token streaming first
                if self._is_processing:
                    await self.stop_token_streaming()
                
                # Close WebSocket
                await self.websocket.close()
                logger.info(f"WebSocket disconnected for session: {self.session_id}")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.connected = False
    
    async def send_message(
        self,
        message_type: str,
        content: str | None = None,
        data: dict[str, Any] | None = None,
    ):
        """
        Send a message to the client.
        
        Args:
            message_type: Message type (token, sources, metadata, done, error)
            content: Message content (optional)
            data: Additional data payload (optional)
        """
        try:
            message = WebSocketMessage(
                type=message_type,
                content=content,
                data=data or {},
            )
            await self.websocket.send_text(message.model_dump_json())
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
    
    async def send_error(self, error_message: str, code: int = 500):
        """
        Send an error message to the client.
        
        Args:
            error_message: Error description
            code: Error code
        """
        await self.send_message(
            message_type="error",
            content=error_message,
            data={"code": code},
        )
    
    async def _maybe_generate_title(self, first_message: str):
        """
        Generate a session title if this is the first user message.
        
        Args:
            first_message: The user's first message
        """
        try:
            # Reload session from database to get latest messages
            session = self.session_manager.get_session(self.session_id, reload_from_db=True)
            if not session:
                return
            
            # Only generate if no title or title is default "New Chat"
            existing_title = session.metadata.get("title", "")
            if existing_title and existing_title != "New Chat":
                logger.debug(f"Session {self.session_id} already has title: '{existing_title}'")
                return
            
            # Only generate for the first exchange (2 messages: 1 user + 1 assistant)
            # Note: We just added both messages, so should have exactly 2
            message_count = len(session.messages)
            logger.debug(f"Session {self.session_id} has {message_count} messages")
            
            if message_count != 2:
                logger.debug(f"Skipping title generation - need exactly 2 messages, have {message_count}")
                return
            
            logger.info(f"Auto-generating title for session {self.session_id}")
            
            # Use LLM to generate a concise title
            title_prompt = f"""Generate a very short, concise title (maximum 6 words) for a conversation that starts with this message:

"{first_message}"

Reply with ONLY the title, nothing else. No quotes, no explanations."""

            messages = [
                {"role": "user", "content": title_prompt}
            ]
            
            # Generate title using LLM (non-streaming)
            from src.core.llm import OllamaClient
            llm = OllamaClient(timeout=10)
            
            response = llm.generate(
                messages=messages,
                model=self.config.rag.llm.model,
                temperature=0.7,
                top_p=0.9,
                max_tokens=20,  # Keep it short
                stream=False,
            )
            
            generated_title = response.get("message", {}).get("content", "").strip()
            
            # Clean up the title (remove quotes if present)
            generated_title = generated_title.strip('"\'').strip()
            
            # Limit length
            if len(generated_title) > 60:
                generated_title = generated_title[:57] + "..."
            
            if generated_title and generated_title != first_message:
                # Update session metadata with generated title
                self.session_manager.update_session_metadata(
                    self.session_id,
                    {"title": generated_title}
                )
                
                logger.info(f"Generated title for session {self.session_id}: '{generated_title}'")
                
                # Optionally send title update to client
                await self.send_message(
                    message_type="title",
                    content=generated_title,
                    data={"session_id": self.session_id}
                )
            
        except Exception as e:
            logger.warning(f"Failed to generate title for session {self.session_id}: {e}")
            # Don't fail the entire request if title generation fails
    
    async def handle_user_message(self, message: str, options: dict[str, Any] | None = None):
        """
        Handle incoming user message and generate response with queued streaming.
        
        Args:
            message: User message content
            options: Optional settings (rag_mode, include_sources, temperature, files, etc.)
        """
        try:
            options = options or {}
            start_time = time.time()
            
            # Extract optional settings
            rag_mode = options.get("rag_mode") or self.config.rag.generation.rag_trigger_mode
            include_sources = options.get("include_sources", False)
            temperature = options.get("temperature")
            files = options.get("files", [])
            
            # Parse uploaded files if present
            file_context = ""
            file_metadata = []
            if files and len(files) > 0:
                try:
                    from src.utilities.file_parser import parse_multiple_files
                    
                    logger.info(f"Processing {len(files)} uploaded file(s)")
                    file_context, file_metadata = parse_multiple_files(
                        files,
                        config=self.config,
                        max_per_file=5000  # Limit chars per file
                    )
                    logger.info(f"Extracted {len(file_context)} chars from {len(files)} file(s)")
                except Exception as e:
                    logger.error(f"Failed to parse uploaded files: {e}")
                    await self.send_error(f"Failed to parse uploaded files: {str(e)}")
                    return
            
            # Prepend file context to user message if present
            enhanced_message = message
            if file_context:
                enhanced_message = f"{file_context}\n\n**User Question:**\n{message}"
            
            logger.info(
                f"Processing message in session {self.session_id}: '{message[:50]}...' "
                f"(rag_mode={rag_mode}, sources={include_sources}, files={len(files)})"
            )
            
            # Build generation kwargs
            generation_kwargs = {"stream": True}
            if temperature is not None:
                generation_kwargs["temperature"] = temperature
            
            # Start token streaming task
            await self.start_token_streaming()
            
            # Define sync callback for token streaming (called by Ollama)
            def stream_token(token: str):
                """
                Queue tokens for async sending (thread-safe).
                
                This is called from the sync Ollama streaming callback.
                Tokens are queued and sent by the async queue processor.
                """
                self.queue_token(token)
            
            # Generate chat response with streaming
            result = self.generator.generate_chat_response(
                message=enhanced_message,  # Use enhanced message with file context
                session_id=self.session_id,
                session_manager=self.session_manager,
                rag_mode=rag_mode,
                include_sources=include_sources,
                on_token=stream_token,
                **generation_kwargs,
            )
            
            # Stop token streaming (sends end-of-stream signal)
            await self.stop_token_streaming()
            
            # Send sources if available
            if include_sources and result.rag_triggered and hasattr(result, "sources") and result.sources:
                sources = [
                    {
                        "index": i + 1,
                        "citation": src.get("citation", ""),
                        "content": src.get("content", "")[:200],  # Truncate
                        "score": src.get("score", 0.0),
                    }
                    for i, src in enumerate(result.sources)
                ]
                
                await self.send_message(
                    message_type="sources",
                    data={"sources": sources},
                )
            
            # Send metadata
            processing_time = time.time() - start_time
            await self.send_message(
                message_type="metadata",
                data={
                    "rag_triggered": result.rag_triggered,
                    "query_type": getattr(result, "query_type", "conversational"),
                    "model": self.config.rag.llm.model,
                    "rag_mode": rag_mode,
                    "processing_time": processing_time,
                },
            )
            
            # Send done signal
            await self.send_message(
                message_type="done",
                data={
                    "session_id": self.session_id,
                    "processing_time": processing_time,
                },
            )
            
            logger.info(
                f"WebSocket message processed: {len(result.answer)} chars, "
                f"RAG={result.rag_triggered}, {processing_time:.3f}s"
            )
            
            # Auto-generate title for first message
            await self._maybe_generate_title(message)
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}", exc_info=True)
            # Stop streaming on error
            if self._is_processing:
                await self.stop_token_streaming()
            await self.send_error(f"Failed to process message: {str(e)}")
    
    async def handle_ping(self):
        """Handle ping message (keepalive)."""
        await self.send_message(message_type="pong")
    
    async def listen(self):
        """
        Main message loop - listen for incoming messages and handle them.
        
        Runs until the connection is closed or an error occurs.
        """
        try:
            while self.connected:
                # Receive message from client
                raw_message = await self.websocket.receive_text()
                
                # Parse message
                try:
                    data = json.loads(raw_message)
                    message_type = data.get("type", "message")
                    content = data.get("content")
                    options = data.get("data", {})
                    
                    # Handle different message types
                    if message_type == "message":
                        if not content:
                            await self.send_error("Message content is required", code=400)
                            continue
                        
                        await self.handle_user_message(content, options)
                    
                    elif message_type == "ping":
                        await self.handle_ping()
                    
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                        await self.send_error(f"Unknown message type: {message_type}", code=400)
                
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {raw_message}")
                    await self.send_error("Invalid JSON format", code=400)
                
        except WebSocketDisconnect:
            logger.info(f"Client disconnected from session: {self.session_id}")
            self.connected = False
        
        except Exception as e:
            logger.error(f"WebSocket error in session {self.session_id}: {e}", exc_info=True)
            await self.send_error(f"WebSocket error: {str(e)}")
            self.connected = False
        
        finally:
            await self.disconnect()


async def chat_websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    session_manager: SessionManager,
    generator: AnswerGenerator,
    config: OrionConfig,
):
    """
    WebSocket endpoint for real-time chat.
    
    Compatible with HuggingFace chat-ui and standard WebSocket clients.
    
    Args:
        websocket: WebSocket connection
        session_id: Chat session identifier
        session_manager: Session manager instance
        generator: Answer generator instance
        config: Configuration instance
    
    Message format (client → server):
        {
            "type": "message",
            "content": "What is machine learning?",
            "data": {
                "rag_mode": "auto",
                "include_sources": true,
                "temperature": 0.7
            }
        }
    
    Message format (server → client):
        {
            "type": "token",         # or "sources", "metadata", "done", "error"
            "content": "Machine",     # token content (for type="token")
            "data": {...}             # additional data
        }
    """
    handler = ChatWebSocketHandler(
        websocket=websocket,
        session_id=session_id,
        session_manager=session_manager,
        generator=generator,
        config=config,
    )
    
    # Connect and verify session
    if not await handler.connect():
        return
    
    # Start message loop
    await handler.listen()
