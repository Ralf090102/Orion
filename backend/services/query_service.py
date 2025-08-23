"""
Business logic for document querying and search
"""

from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from core.rag.query import query_knowledgebase
from backend.models.query import (
    QueryResponse,
    SearchResult,
    ConversationHistory,
    ChatMessage,
)
from core.utils.orion_utils import log_info, log_error


class QueryService:
    def __init__(self):
        self.conversations: Dict[str, List[ChatMessage]] = {}

    async def search(
        self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search documents in vectorstore
        """
        try:
            log_info(f"Searching for: {query}")

            # Use your existing query function
            results = await query_knowledgebase(query=query, top_k=k)

            search_results = []
            for i, result in enumerate(results.get("contexts", [])):
                search_results.append(
                    SearchResult(
                        content=result.get("content", ""),
                        metadata=result.get("metadata", {}),
                        score=result.get("score", 0.0),
                        document_id=result.get("metadata", {}).get(
                            "source", f"doc_{i}"
                        ),
                        chunk_id=f"chunk_{i}",
                    )
                )

            return search_results

        except Exception as e:
            log_error(f"Search failed: {e}")
            raise

    async def chat(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        model: str = "llama3.2:3b",
    ) -> QueryResponse:
        """
        Chat with documents using RAG
        """
        try:
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())

            # Get search results
            search_results = await self.search(query, k=5)

            # Use your existing query function for RAG
            rag_result = await query_knowledgebase(query=query, model=model, top_k=5)

            answer = rag_result.get("answer", "No answer generated")

            # Store conversation
            user_message = ChatMessage(
                role="user", content=query, timestamp=datetime.now()
            )

            assistant_message = ChatMessage(
                role="assistant",
                content=answer,
                timestamp=datetime.now(),
                sources=search_results,
            )

            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []

            self.conversations[conversation_id].extend(
                [user_message, assistant_message]
            )

            return QueryResponse(
                query=query,
                answer=answer,
                results=search_results,
                total_results=len(search_results),
                conversation_id=conversation_id,
                model_used=model,
            )

        except Exception as e:
            log_error(f"Chat failed: {e}")
            raise

    async def find_similar(
        self, document_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to given document
        """
        # TODO: Implement similarity search
        return []

    async def get_conversations(self) -> List[ConversationHistory]:
        """
        Get list of conversation histories
        """
        conversations = []
        for conv_id, messages in self.conversations.items():
            conversations.append(
                ConversationHistory(
                    conversation_id=conv_id,
                    created_at=messages[0].timestamp if messages else datetime.now(),
                    updated_at=messages[-1].timestamp if messages else datetime.now(),
                    message_count=len(messages),
                    title=(
                        messages[0].content[:50] + "..."
                        if messages
                        else "Empty conversation"
                    ),
                )
            )
        return conversations

    async def delete_conversation(self, conversation_id: str):
        """
        Delete a conversation
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            log_info(f"Deleted conversation: {conversation_id}")
        else:
            raise KeyError(f"Conversation {conversation_id} not found")
