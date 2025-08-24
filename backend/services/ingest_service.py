"""
Business logic for document ingestion
"""

import asyncio
import uuid
from typing import Dict, Any
from pathlib import Path

from core.rag.ingest import rebuild_vectorstore_async, incremental_vectorstore_async
from core.utils.orion_utils import log_info, log_error
from backend.models.ingest import IngestStatus, IngestResponse


class IngestService:
    def __init__(self):
        self.active_tasks: Dict[str, Dict[str, Any]] = {}

    async def ingest_documents(self, folder_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        """
        Synchronous document ingestion
        """
        log_info(f"Starting document ingestion for: {folder_path}")

        try:
            vectorstore = await rebuild_vectorstore_async(
                folder_path=folder_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            if vectorstore is None:
                raise Exception("Failed to build vectorstore")

            # Get document count (approximate)
            doc_count = len(list(Path(folder_path).rglob("*.pdf")))  # Simplified
            chunk_count = vectorstore.index.ntotal if hasattr(vectorstore.index, "ntotal") else 0

            return {
                "document_count": doc_count,
                "chunk_count": chunk_count,
                "status": "completed",
            }

        except Exception as e:
            log_error(f"Ingestion failed: {e}")
            raise

    async def start_async_ingest(self, folder_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        """
        Start asynchronous document ingestion
        """
        task_id = str(uuid.uuid4())

        # Initialize task status
        self.active_tasks[task_id] = {
            "status": IngestStatus.PENDING,
            "progress": 0.0,
            "message": "Task queued",
        }

        # Start background task
        asyncio.create_task(self._run_async_ingest(task_id, folder_path, chunk_size, chunk_overlap))

        return task_id

    async def _run_async_ingest(self, task_id: str, folder_path: str, chunk_size: int, chunk_overlap: int):
        """
        Run the actual ingestion task
        """
        try:
            self.active_tasks[task_id]["status"] = IngestStatus.PROCESSING
            self.active_tasks[task_id]["message"] = "Processing documents..."

            result = await self.ingest_documents(folder_path, chunk_size, chunk_overlap)

            self.active_tasks[task_id].update(
                {
                    "status": IngestStatus.COMPLETED,
                    "progress": 100.0,
                    "message": "Ingestion completed successfully",
                    "document_count": result["document_count"],
                    "chunk_count": result["chunk_count"],
                }
            )

        except Exception as e:
            self.active_tasks[task_id].update({"status": IngestStatus.FAILED, "message": str(e), "error": str(e)})

    async def get_task_status(self, task_id: str) -> IngestResponse:
        """
        Get the status of an ingestion task
        """
        if task_id not in self.active_tasks:
            raise KeyError(f"Task {task_id} not found")

        task_data = self.active_tasks[task_id]

        return IngestResponse(
            task_id=task_id,
            status=task_data["status"],
            message=task_data["message"],
            progress=task_data.get("progress"),
            document_count=task_data.get("document_count"),
            chunk_count=task_data.get("chunk_count"),
            error=task_data.get("error"),
        )

    async def incremental_ingest(self, folder_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        """
        Perform incremental document ingestion
        """
        log_info(f"Starting incremental ingestion for: {folder_path}")

        try:
            vectorstore = await incremental_vectorstore_async(
                folder_path=folder_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            if vectorstore is None:
                raise Exception("Incremental ingestion failed")

            return {
                "added_count": 0,  # TODO: Track actual counts
                "updated_count": 0,
                "total_documents": 0,
                "total_chunks": 0,
                "status": "completed",
            }

        except Exception as e:
            log_error(f"Incremental ingestion failed: {e}")
            raise

    async def clear_vectorstore(self):
        """
        Clear the vectorstore
        """
        # TODO: Implement vectorstore clearing
        log_info("Clearing vectorstore...")
        pass
