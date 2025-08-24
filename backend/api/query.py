"""
Query and Search API Endpoints
"""

from fastapi import APIRouter, HTTPException

from backend.models.query import QueryRequest, QueryResponse
from backend.services.query_service import QueryService

router = APIRouter()
query_service = QueryService()


@router.post("/search", response_model=QueryResponse)
async def search_documents(request: QueryRequest):
    """
    Search documents in the vectorstore
    """
    try:
        results = await query_service.search(query=request.query, k=request.k, filters=request.filters)

        return QueryResponse(query=request.query, results=results, total_results=len(results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=QueryResponse)
async def chat_with_documents(request: QueryRequest):
    """
    Chat with documents using RAG
    """
    try:
        response = await query_service.chat(
            query=request.query,
            conversation_id=request.conversation_id,
            model=request.model,
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar/{document_id}")
async def find_similar_documents(document_id: str, limit: int = 5):
    """
    Find documents similar to the given document
    """
    try:
        results = await query_service.find_similar(document_id, limit)
        return {"similar_documents": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations")
async def list_conversations():
    """
    List all conversation histories
    """
    try:
        conversations = await query_service.get_conversations()
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation history
    """
    try:
        await query_service.delete_conversation(conversation_id)
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
