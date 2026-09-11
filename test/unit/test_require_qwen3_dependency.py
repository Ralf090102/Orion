"""Unit tests for backend.dependencies.require_qwen3 (architecture-review
candidate #3, 2026-09-11).

Before this fix, the same three-step precondition (engine selected, Qwen3
enabled, Qwen3 manager available) was re-derived by hand in all 8 Qwen3
routes in backend/api/speech.py, and had already drifted: 3 of the 8 skipped
the "enabled" check entirely, and 2 of the 8 (list_cloned_voices,
get_qwen3_stats) returned a soft 200 instead of raising on unavailability.
require_qwen3() collapses all 8 into one FastAPI dependency and standardizes
behavior: every route now raises consistently.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from backend.dependencies import require_qwen3
from src.utilities.config import OrionConfig


def _config(*, engine: str = "qwen3", qwen3_enabled: bool = True) -> OrionConfig:
    config = OrionConfig()
    config.tts.default_engine = engine
    config.qwen3.enabled = qwen3_enabled
    return config


def _tts_manager(*, qwen3_manager=MagicMock()) -> SimpleNamespace:
    """A minimal stand-in with just the attribute require_qwen3 inspects."""
    return SimpleNamespace(qwen3_manager=qwen3_manager)


@pytest.mark.unit
class TestRequireQwen3:
    def test_wrong_engine_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            require_qwen3(config=_config(engine="piper"), tts_manager=_tts_manager())
        assert exc_info.value.status_code == 400
        assert "Qwen3-TTS" in exc_info.value.detail

    def test_disabled_raises_503(self):
        with pytest.raises(HTTPException) as exc_info:
            require_qwen3(
                config=_config(engine="qwen3", qwen3_enabled=False), tts_manager=_tts_manager()
            )
        assert exc_info.value.status_code == 503
        assert "not enabled" in exc_info.value.detail

    def test_manager_missing_attribute_raises_503(self):
        tts_manager = SimpleNamespace()  # no qwen3_manager attribute at all
        with pytest.raises(HTTPException) as exc_info:
            require_qwen3(config=_config(), tts_manager=tts_manager)
        assert exc_info.value.status_code == 503
        assert "not available" in exc_info.value.detail

    def test_manager_none_raises_503(self):
        with pytest.raises(HTTPException) as exc_info:
            require_qwen3(config=_config(), tts_manager=_tts_manager(qwen3_manager=None))
        assert exc_info.value.status_code == 503
        assert "not available" in exc_info.value.detail

    def test_all_conditions_met_returns_tts_manager(self):
        tts_manager = _tts_manager()
        result = require_qwen3(config=_config(), tts_manager=tts_manager)
        assert result is tts_manager
