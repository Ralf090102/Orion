"""
Document Ingestion API Endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks

from backend.models.ingest import IngestRequest, IngestResponse, IngestStatus
from backend.services.ingest_service import IngestService

router = APIRouter()
ingest_service = IngestService()


@router.post("/documents", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest documents from a folder path
    """
    try:
        if request.async_processing:
            # Start background processing
            task_id = await ingest_service.start_async_ingest(
                request.folder_path, request.chunk_size, request.chunk_overlap
            )

            return IngestResponse(
                task_id=task_id,
                status=IngestStatus.PROCESSING,
                message="Ingestion started in background",
            )
        else:
            # Synchronous processing
            result = await ingest_service.ingest_documents(
                request.folder_path, request.chunk_size, request.chunk_overlap
            )

            return IngestResponse(
                task_id=None,
                status=IngestStatus.COMPLETED,
                message=f"Ingested {result['document_count']} documents",
                document_count=result["document_count"],
                chunk_count=result["chunk_count"],
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=IngestResponse)
async def get_ingest_status(task_id: str):
    """
    Get the status of an ingestion task
    """
    try:
        status = await ingest_service.get_task_status(task_id)
        return status
    except KeyError:
        raise HTTPException(status_code=404, detail="Task not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/incremental", response_model=IngestResponse)
async def incremental_ingest(request: IngestRequest):
    """
    Perform incremental document ingestion (only changed files)
    """
    try:
        result = await ingest_service.incremental_ingest(
            request.folder_path, request.chunk_size, request.chunk_overlap
        )

        return IngestResponse(
            status=IngestStatus.COMPLETED,
            message=f"Incremental ingestion complete: {result['added_count']} added, {result['updated_count']} updated",
            document_count=result.get("total_documents", 0),
            chunk_count=result.get("total_chunks", 0),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vectorstore")
async def clear_vectorstore():
    """
    Clear the vectorstore (delete all indexed documents)
    """
    try:
        await ingest_service.clear_vectorstore()
        return {"message": "Vectorstore cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
