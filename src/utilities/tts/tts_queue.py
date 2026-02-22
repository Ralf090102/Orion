"""Background TTS queue for asynchronous synthesis.

This module provides a queue system for handling slow TTS synthesis operations
(like Qwen3-TTS) in the background without blocking API responses.

Features:
- Async task processing with worker threads
- Progress tracking (0-100%)
- Task cancellation support
- Priority queue (future enhancement)
- Status polling via task ID

Why needed for Qwen3:
- Qwen3 synthesis can take 5-60 seconds
- Can't block HTTP response for that long
- Users need progress updates
- Support cancellation of long-running tasks
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """TTS task status enum."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TTSTask:
    """Background TTS synthesis task.
    
    Attributes:
        task_id: Unique task identifier (UUID)
        text: Text to synthesize
        voice_id: Voice to use for synthesis
        engine: TTS engine ("piper" or "qwen3")
        speed: Speech speed multiplier
        format: Audio format ("wav", "mp3")
        status: Current task status
        progress: Progress percentage (0.0 to 1.0)
        audio_bytes: Generated audio (when completed)
        error: Error message (when failed)
        created_at: Task creation timestamp
        started_at: Processing start timestamp
        completed_at: Completion timestamp
    """
    task_id: str
    text: str
    voice_id: str
    engine: str = "qwen3"
    speed: float = 1.0
    format: str = "wav"
    language: str = "en"
    
    # Task state
    status: TaskStatus = TaskStatus.QUEUED
    progress: float = 0.0  # 0.0 to 1.0
    audio_bytes: Optional[bytes] = None
    error: Optional[str] = None
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API responses."""
        return {
            "task_id": self.task_id,
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "voice_id": self.voice_id,
            "engine": self.engine,
            "status": self.status.value,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": (self.completed_at - self.started_at) if self.completed_at and self.started_at else None,
        }


class TTSQueue:
    """Asynchronous TTS task queue.
    
    Manages background processing of TTS synthesis tasks with progress tracking
    and cancellation support.
    
    Example:
        >>> queue = TTSQueue(max_concurrent=1)
        >>> await queue.start()
        >>> 
        >>> # Enqueue task
        >>> task_id = await queue.enqueue("Hello world", "my_voice", "qwen3")
        >>> 
        >>> # Poll status
        >>> task = await queue.get_status(task_id)
        >>> print(f"Progress: {task.progress * 100}%")
        >>> 
        >>> # Get audio when complete
        >>> if task.status == TaskStatus.COMPLETED:
        >>>     audio = task.audio_bytes
    """
    
    def __init__(
        self,
        max_concurrent: int = 1,
        synthesize_callback: Optional[Callable] = None,
    ):
        """Initialize TTSQueue.
        
        Args:
            max_concurrent: Maximum concurrent synthesis tasks (1 recommended for Qwen3)
            synthesize_callback: Async function to call for synthesis
                                Signature: async def synthesize(task: TTSTask) -> bytes
        """
        self.max_concurrent = max_concurrent
        self.synthesize_callback = synthesize_callback
        
        # Queue and task storage
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: Dict[str, TTSTask] = {}
        self.active_tasks: set = set()
        
        # Worker control
        self.workers: list = []
        self.running = False
        
        # Stats
        self.total_processed = 0
        self.total_failed = 0
        self.total_cancelled = 0
        
        logger.info(f"TTSQueue initialized (max_concurrent: {max_concurrent})")
    
    async def start(self) -> None:
        """Start background worker threads."""
        if self.running:
            logger.warning("Queue already running")
            return
        
        self.running = True
        
        # Start worker coroutines
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        
        logger.info(f"✓ Started {self.max_concurrent} TTS workers")
    
    async def stop(self) -> None:
        """Stop all workers and cancel pending tasks."""
        if not self.running:
            return
        
        logger.info("Stopping TTS queue...")
        self.running = False
        
        # Cancel all pending tasks
        while not self.queue.empty():
            try:
                task_id = self.queue.get_nowait()
                if task_id in self.tasks:
                    self.tasks[task_id].status = TaskStatus.CANCELLED
                    self.total_cancelled += 1
            except asyncio.QueueEmpty:
                break
        
        # Wait for active workers to finish
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("✓ All TTS workers stopped")
    
    async def enqueue(
        self,
        text: str,
        voice_id: str,
        engine: str = "qwen3",
        speed: float = 1.0,
        format: str = "wav",
        language: str = "en",
    ) -> str:
        """Enqueue a TTS synthesis task.
        
        Args:
            text: Text to synthesize
            voice_id: Voice identifier
            engine: TTS engine ("piper" or "qwen3")
            speed: Speech speed multiplier
            format: Audio format
            language: Language code
        
        Returns:
            task_id: Unique task identifier for polling
        """
        # Create task
        task_id = str(uuid.uuid4())
        task = TTSTask(
            task_id=task_id,
            text=text,
            voice_id=voice_id,
            engine=engine,
            speed=speed,
            format=format,
            language=language,
        )
        
        # Store and queue
        self.tasks[task_id] = task
        await self.queue.put(task_id)
        
        logger.info(f"Enqueued task {task_id[:8]}: {len(text)} chars, engine={engine}")
        return task_id
    
    async def get_status(self, task_id: str) -> Optional[TTSTask]:
        """Get task status.
        
        Args:
            task_id: Task identifier
        
        Returns:
            TTSTask object or None if not found
        """
        return self.tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            True if cancelled, False if not found or already completed
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # Can only cancel queued or processing tasks
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = time.time()
        self.total_cancelled += 1
        
        logger.info(f"Cancelled task: {task_id[:8]}")
        return True
    
    async def _worker(self, worker_id: int) -> None:
        """Background worker that processes TTS tasks.
        
        Args:
            worker_id: Worker identifier for logging
        """
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get next task (with timeout to allow graceful shutdown)
                task_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                task = self.tasks.get(task_id)
                if not task or task.status == TaskStatus.CANCELLED:
                    continue
                
                # Mark as processing
                task.status = TaskStatus.PROCESSING
                task.started_at = time.time()
                task.progress = 0.1
                self.active_tasks.add(task_id)
                
                logger.info(f"Worker {worker_id} processing: {task_id[:8]}")
                
                try:
                    # Call synthesis callback
                    if self.synthesize_callback:
                        # Update progress periodically
                        task.progress = 0.3
                        
                        # Perform synthesis
                        audio_bytes = await self.synthesize_callback(task)
                        
                        # Check if cancelled during synthesis
                        if task.status == TaskStatus.CANCELLED:
                            continue
                        
                        task.progress = 0.9
                        task.audio_bytes = audio_bytes
                        task.status = TaskStatus.COMPLETED
                        self.total_processed += 1
                        
                        logger.info(f"✓ Task completed: {task_id[:8]} ({len(audio_bytes)} bytes)")
                    else:
                        raise RuntimeError("No synthesize callback configured")
                
                except Exception as e:
                    logger.error(f"Task failed: {task_id[:8]} - {e}")
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    self.total_failed += 1
                
                finally:
                    task.progress = 1.0
                    task.completed_at = time.time()
                    self.active_tasks.discard(task_id)
                    self.queue.task_done()
            
            except asyncio.TimeoutError:
                # No tasks available, continue loop
                continue
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    def get_queue_size(self) -> int:
        """Get number of pending tasks.
        
        Returns:
            Queue size
        """
        return self.queue.qsize()
    
    def get_active_count(self) -> int:
        """Get number of actively processing tasks.
        
        Returns:
            Active task count
        """
        return len(self.active_tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics.
        
        Returns:
            Dictionary with queue stats
        """
        return {
            "running": self.running,
            "max_concurrent": self.max_concurrent,
            "queue_size": self.get_queue_size(),
            "active_tasks": self.get_active_count(),
            "total_tasks": len(self.tasks),
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "total_cancelled": self.total_cancelled,
        }
    
    async def clear_old_tasks(self, max_age_seconds: int = 3600) -> int:
        """Clear completed tasks older than max_age.
        
        Args:
            max_age_seconds: Maximum age to keep completed tasks (default 1 hour)
        
        Returns:
            Number of tasks cleared
        """
        now = time.time()
        cleared = 0
        
        # Find old completed/failed tasks
        to_remove = [
            task_id for task_id, task in self.tasks.items()
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            and task.completed_at
            and (now - task.completed_at) > max_age_seconds
        ]
        
        # Remove them
        for task_id in to_remove:
            del self.tasks[task_id]
            cleared += 1
        
        if cleared > 0:
            logger.info(f"Cleared {cleared} old tasks")
        
        return cleared
