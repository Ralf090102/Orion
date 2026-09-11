"""Unit tests for backend.streaming_queue.ThreadSafeEventQueue
(architecture-review "unify the thread-safe streaming-queue pattern"
candidate, implemented 2026-09-11).

Covers the two real differences chat.py and rag.py had before unification:
generic item shape (bare value vs. tagged tuple) and both consumer drain
styles (timeout-based polling vs. plain blocking wait).
"""

import asyncio
import threading

import pytest

from backend.streaming_queue import ThreadSafeEventQueue


@pytest.mark.unit
@pytest.mark.asyncio
class TestThreadSafeEventQueue:
    async def test_put_threadsafe_crosses_a_real_thread_boundary(self):
        """The actual point of this class: a genuine OS thread (not just a
        same-thread call) pushing an item must reach the event-loop side."""
        queue: ThreadSafeEventQueue[str] = ThreadSafeEventQueue()
        queue.bind_loop(asyncio.get_running_loop())

        def worker():
            queue.put_threadsafe("hello from a worker thread")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        item = await queue.get()
        assert item == "hello from a worker thread"

    async def test_tagged_tuple_items_round_trip(self):
        """rag.py's item shape: tuple[str, Any], not a bare value."""
        queue: ThreadSafeEventQueue[tuple[str, object]] = ThreadSafeEventQueue()
        queue.bind_loop(asyncio.get_running_loop())

        queue.put_threadsafe(("token", "hi"))
        queue.put_threadsafe(("sources", [{"id": 1}]))

        first = await queue.get()
        second = await queue.get()
        assert first == ("token", "hi")
        assert second == ("sources", [{"id": 1}])

    async def test_close_delivers_none_sentinel_directly(self):
        """chat.py's stop_token_streaming() pattern: called from the
        event-loop thread, no hop needed."""
        queue: ThreadSafeEventQueue[str] = ThreadSafeEventQueue()

        queue.close()

        assert await queue.get() is None

    async def test_close_threadsafe_delivers_none_sentinel_via_thread_hop(self):
        """rag.py's run_generation() finally-block pattern."""
        queue: ThreadSafeEventQueue[str] = ThreadSafeEventQueue()
        queue.bind_loop(asyncio.get_running_loop())

        thread = threading.Thread(target=queue.close_threadsafe)
        thread.start()
        thread.join()

        assert await queue.get() is None

    async def test_get_with_timeout_raises_timeout_error_when_empty(self):
        """chat.py's liveness-polling drain loop pattern."""
        queue: ThreadSafeEventQueue[str] = ThreadSafeEventQueue()

        with pytest.raises(asyncio.TimeoutError):
            await queue.get(timeout=0.05)

    async def test_get_without_timeout_blocks_until_an_item_arrives(self):
        """rag.py's plain one-shot drain pattern -- no TimeoutError, just
        waits."""
        queue: ThreadSafeEventQueue[str] = ThreadSafeEventQueue()
        queue.bind_loop(asyncio.get_running_loop())

        async def push_after_delay():
            await asyncio.sleep(0.01)
            queue.put_threadsafe("delayed item")

        push_task = asyncio.create_task(push_after_delay())
        item = await queue.get()
        await push_task

        assert item == "delayed item"


@pytest.mark.unit
def test_put_threadsafe_without_bind_loop_drops_item_and_logs():
    """No event loop registered yet -- must not raise, just drop and log
    (matches queue_token()'s pre-existing "no event loop registered"
    behavior)."""
    queue: ThreadSafeEventQueue[str] = ThreadSafeEventQueue()

    queue.put_threadsafe("lost")  # should not raise

    assert queue.empty()
