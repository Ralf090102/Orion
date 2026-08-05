"""Regression test: transcription must not block the event loop.

Found live: "STT works 100%, though 'python backend isn't running' popup
shows up." Root cause -- WhisperManager.transcribe() is a plain synchronous
method (lazy model load + CPU/GPU-bound faster-whisper inference), called
directly (no asyncio.to_thread/run_in_executor) from an `async def` FastAPI
route. Uvicorn runs on a single event loop by default, so that blocking call
froze it for the whole transcription -- during which the frontend's
periodic GET /health check (every 10s, 3s client-side timeout) couldn't be
serviced, timed out, and flipped the UI's "backend not running" banner even
though the backend was alive and the transcription itself succeeded a
moment later. Fixed by moving the call onto a worker thread via
asyncio.to_thread(), the same pattern already used for the LLM call in
_maybe_generate_title() (backend/websockets/chat.py).

This test proves the actual guarantee -- the event loop keeps servicing
other coroutines *during* a slow transcription -- rather than just checking
transcribe_audio() still returns the right text (it always did; that's why
the user saw STT "work 100%" despite the bug).
"""

import asyncio
import io
import time
from unittest.mock import MagicMock

import pytest
from starlette.datastructures import UploadFile

from backend.api.speech import transcribe_audio


class _FakeConfig:
    class whisper:
        model_size = "base"
        device = "cpu"
        compute_type = "int8"


@pytest.mark.unit
@pytest.mark.asyncio
class TestTranscribeDoesNotBlockEventLoop:
    async def test_event_loop_stays_responsive_during_transcription(self, monkeypatch):
        SLEEP_SECONDS = 0.3

        def blocking_transcribe(audio_path, language=None):
            # Real time.sleep(), not asyncio.sleep() -- this is what a
            # genuinely CPU-bound synchronous call looks like, and the only
            # thing that actually exercises whether the caller offloaded it
            # to a thread. An unpatched (bugged) route awaiting this
            # directly would freeze the event loop for the full duration.
            time.sleep(SLEEP_SECONDS)
            return {"text": "hello world", "language": "en", "duration": 1.2}

        fake_whisper_manager = MagicMock()
        fake_whisper_manager.transcribe.side_effect = blocking_transcribe
        monkeypatch.setattr(
            "backend.api.speech.get_whisper_manager", lambda: fake_whisper_manager
        )

        fake_audio = UploadFile(file=io.BytesIO(b"fake audio bytes"), filename="test.webm")

        heartbeat_ticks = 0

        async def heartbeat():
            nonlocal heartbeat_ticks
            # If the event loop were blocked by the transcription, this
            # loop would never get a chance to increment during it.
            while True:
                await asyncio.sleep(0.02)
                heartbeat_ticks += 1

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            start = time.monotonic()
            result = await transcribe_audio(audio=fake_audio, language=None, config=_FakeConfig())
            elapsed = time.monotonic() - start
        finally:
            heartbeat_task.cancel()

        assert result.text == "hello world"
        # Small tolerance for timer-precision jitter -- time.sleep(0.3) can
        # occasionally measure a hair under 0.3s via time.monotonic() under
        # scheduling load; the real signal that the call actually did its
        # blocking work (rather than being skipped/mocked away) is this
        # loose lower bound plus the heartbeat-tick check below.
        assert elapsed >= SLEEP_SECONDS * 0.9
        # With the event loop free, ~0.3s / 0.02s ticks should land well
        # into double digits; require a conservative minimum to avoid
        # flakiness while still failing hard against a truly blocked loop
        # (which would produce 0 or 1).
        assert heartbeat_ticks >= 5, (
            f"event loop only ticked {heartbeat_ticks} times during transcription -- "
            "looks blocked, not offloaded to a thread"
        )
