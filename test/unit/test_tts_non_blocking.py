"""Regression test: TTS synthesis must not block the event loop.

Same bug class as test_transcribe_non_blocking.py, found and fixed in the
same file (backend/api/speech.py) right after the STT fix: every TTS
synthesis/voice-cloning/voice-design call was a plain synchronous call to
tts_manager/qwen3_manager, invoked directly (no asyncio.to_thread) from an
`async def` route. Any one of them running for real seconds would freeze
Uvicorn's single event loop for that whole window -- same false "backend
not running" popup risk as the STT bug, just not yet reported for TTS.

Covers two shapes:
- synthesize_speech(): a single blocking call, wrapped directly in
  asyncio.to_thread() (same pattern as the STT fix).
- synthesize_speech_stream(): a blocking *generator* (each next() call does
  real synthesis work per sentence-chunk) -- needed its own per-chunk
  asyncio.to_thread(next, gen) loop instead of a single wrap.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from backend.api.speech import synthesize_speech, synthesize_speech_stream
from backend.models.speech import TTSRequest


async def _run_with_heartbeat(coro_factory):
    """Runs coro_factory() while a heartbeat coroutine ticks concurrently.
    Returns (result, tick_count). A blocked event loop => tick_count stays
    at 0 (or very low); a free event loop => many ticks."""
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(heartbeat())
    try:
        result = await coro_factory()
    finally:
        task.cancel()
    return result, ticks


@pytest.mark.unit
@pytest.mark.asyncio
class TestSynthesizeSpeechDoesNotBlockEventLoop:
    async def test_event_loop_stays_responsive_during_synthesis(self, monkeypatch):
        SLEEP_SECONDS = 0.3

        def blocking_synthesize(text, voice_id=None, speed=1.0, output_format="mp3"):
            time.sleep(SLEEP_SECONDS)  # genuine blocking call, not asyncio.sleep
            return b"fake-audio-bytes"

        fake_tts_manager = MagicMock()
        fake_tts_manager.synthesize.side_effect = blocking_synthesize
        monkeypatch.setattr("backend.api.speech.get_tts_manager", lambda: fake_tts_manager)

        request = TTSRequest(text="hello world", format="wav")

        response, ticks = await _run_with_heartbeat(
            lambda: synthesize_speech(request=request, config=MagicMock())
        )

        assert response.body == b"fake-audio-bytes"
        assert ticks >= 5, f"event loop only ticked {ticks} times -- looks blocked"


@pytest.mark.unit
@pytest.mark.asyncio
class TestSynthesizeSpeechStreamDoesNotBlockEventLoop:
    async def test_event_loop_stays_responsive_between_and_during_chunks(self, monkeypatch):
        CHUNK_SLEEP_SECONDS = 0.15
        NUM_CHUNKS = 3

        def blocking_stream_generator(text, voice_id=None, speed=1.0, output_format="wav", language="en"):
            for i in range(NUM_CHUNKS):
                time.sleep(CHUNK_SLEEP_SECONDS)  # genuine blocking work per chunk
                yield f"chunk-{i}".encode()

        fake_tts_manager = MagicMock()
        fake_tts_manager.synthesize_stream.side_effect = blocking_stream_generator
        monkeypatch.setattr("backend.api.speech.get_tts_manager", lambda: fake_tts_manager)

        request = TTSRequest(text="hello world", format="wav")

        async def collect():
            response = await synthesize_speech_stream(request=request, config=MagicMock())
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            return chunks

        chunks, ticks = await _run_with_heartbeat(collect)

        # 3 chunk lines + 1 final {"done": true} line.
        assert len(chunks) == NUM_CHUNKS + 1
        assert ticks >= 5, f"event loop only ticked {ticks} times across {NUM_CHUNKS} chunks -- looks blocked"
