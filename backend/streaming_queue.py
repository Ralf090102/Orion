"""
Thread-safe event relay for the worker-thread -> asyncio-event-loop handoff.

Both backend/websockets/chat.py (ChatWebSocketHandler's token queue) and
backend/api/rag.py (ask_stream's SSE queue) independently implement the same
plumbing: a blocking LLM call offloaded to a worker thread via
asyncio.to_thread, with tokens (and in rag.py's case, sources too) marshalled
back to the event loop that owns the request/connection via
loop.call_soon_threadsafe + asyncio.Queue + a None sentinel meaning "done".
ThreadSafeEventQueue is the shared seam -- two real adapters already
justified it (architecture-review candidate, 2026-09-11).

The two callers' consumer shapes genuinely differ (chat.py's is a long-lived,
connection-scoped task that polls with a timeout so it can also observe
connection-liveness between waits; rag.py's is a one-shot, request-scoped
plain blocking wait) -- get()'s optional `timeout` covers both without
forcing either caller's drain-loop shape to change.
"""

import asyncio
import logging
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ThreadSafeEventQueue(Generic[T]):
    """Relays events pushed from a worker thread to the asyncio event loop
    that owns this queue, via loop.call_soon_threadsafe. A None item is the
    sentinel meaning "no more events"."""

    def __init__(self) -> None:
        self._queue: "asyncio.Queue[T | None]" = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Call once, from the event-loop thread, before any worker thread
        pushes to this queue (typically right after
        asyncio.get_running_loop())."""
        self._loop = loop

    def put_threadsafe(self, item: T) -> None:
        """Push an item from a worker thread. Safe to call from any thread
        once bind_loop() has registered the owning loop."""
        if self._loop is None:
            logger.error("ThreadSafeEventQueue.put_threadsafe: no event loop registered, dropping item")
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def close(self) -> None:
        """Push the completion sentinel directly. Call from the event-loop
        thread -- no thread hop needed when the caller is already there."""
        self._queue.put_nowait(None)

    def close_threadsafe(self) -> None:
        """Push the completion sentinel from a worker thread."""
        self.put_threadsafe(None)  # type: ignore[arg-type]

    def empty(self) -> bool:
        return self._queue.empty()

    async def get(self, timeout: float | None = None) -> T | None:
        """Await the next item (None = the completion sentinel).

        Pass `timeout` to poll a liveness condition between waits (raises
        asyncio.TimeoutError on expiry, same as asyncio.wait_for) --
        matches chat.py's connection-liveness-check loop. Omit it for a
        plain blocking wait -- matches rag.py's one-shot drain.
        """
        if timeout is None:
            return await self._queue.get()
        return await asyncio.wait_for(self._queue.get(), timeout=timeout)
