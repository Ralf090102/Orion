"""
WebSocket Chat Handler

Real-time bidirectional chat via WebSocket.
Compatible with HuggingFace chat-ui and other WebSocket clients.
"""

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
    """Handler for WebSocket chat connections."""
    
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
        """Close WebSocket connection."""
        if self.connected:
            try:
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
    
    async def handle_user_message(self, message: str, options: dict[str, Any] | None = None):
        """
        Handle incoming user message and generate response.
        
        Args:
            message: User message content
            options: Optional settings (rag_mode, include_sources, temperature, etc.)
        """
        try:
            options = options or {}
            start_time = time.time()
            
            # Extract optional settings
            rag_mode = options.get("rag_mode") or self.config.rag.generation.rag_trigger_mode
            include_sources = options.get("include_sources", False)
            temperature = options.get("temperature")
            
            logger.info(
                f"Processing message in session {self.session_id}: '{message}' "
                f"(rag_mode={rag_mode}, sources={include_sources})"
            )
            
            # Build generation kwargs
            generation_kwargs = {"stream": True}
            if temperature is not None:
                generation_kwargs["temperature"] = temperature
            
            # Token buffer for streaming
            token_buffer = []
            
            def stream_token(token: str):
                """Collect tokens for streaming."""
                token_buffer.append(token)
            
            # Generate chat response with streaming
            result = self.generator.generate_chat_response(
                message=message,
                session_id=self.session_id,
                session_manager=self.session_manager,
                rag_mode=rag_mode,
                include_sources=include_sources,
                on_token=stream_token,
                **generation_kwargs,
            )
            
            # Stream tokens to client
            for token in token_buffer:
                await self.send_message(
                    message_type="token",
                    content=token,
                )
            
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
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}", exc_info=True)
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
