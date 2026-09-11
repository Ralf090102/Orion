"""Regression test for the Qwen3 guard-collapse behavior change
(architecture-review candidate #3, 2026-09-11).

Before this fix, list_cloned_voices (GET /api/speech/cloned-voices) and
get_qwen3_stats (GET /api/speech/qwen3/stats) were the only 2 of 8
Qwen3-only routes that returned a soft 200 (empty list / "unavailable"
status dict) instead of raising 503 when Qwen3 was unavailable -- an
inconsistency with the other 6 routes. require_qwen3() standardizes all 8
to raise 503 consistently; this test pins the behavior change directly at
the route level for these two.
"""

import pytest
from types import SimpleNamespace

from backend.dependencies import get_config_dependency, get_tts_manager
from src.utilities.config import OrionConfig


@pytest.fixture
def qwen3_unavailable_client(client, test_config: OrionConfig):
    """The `client` fixture's default config/tts_manager overrides don't
    touch Qwen3 at all; layer on top of it so config.tts.default_engine ==
    "qwen3" and config.qwen3.enabled == True (clearing the first two
    require_qwen3 checks) while the tts_manager has no working qwen3_manager
    (failing the third) -- isolates the test to the availability check these
    two routes used to skip raising on."""
    from backend.app import app

    test_config.tts.default_engine = "qwen3"
    test_config.qwen3.enabled = True
    app.dependency_overrides[get_config_dependency] = lambda: test_config
    app.dependency_overrides[get_tts_manager] = lambda: SimpleNamespace(qwen3_manager=None)
    yield client


@pytest.mark.unit
class TestQwen3SoftFailEndpointsNowRaise503:
    def test_list_cloned_voices_503s_when_unavailable(self, qwen3_unavailable_client):
        response = qwen3_unavailable_client.get("/api/speech/cloned-voices")
        assert response.status_code == 503

    def test_get_qwen3_stats_503s_when_unavailable(self, qwen3_unavailable_client):
        response = qwen3_unavailable_client.get("/api/speech/qwen3/stats")
        assert response.status_code == 503
