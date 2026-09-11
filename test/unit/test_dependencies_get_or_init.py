"""Unit tests for backend.dependencies._get_or_init (architecture-review
candidate #5, 2026-09-11) and the two getters it newly gives error handling
to.

Before this fix, get_config_dependency/get_session_manager_dependency had no
try/except at all -- a construction failure propagated as an unhandled 500,
despite their own docstrings claiming "Raises: HTTPException". Wrapping them
in the same _get_or_init helper get_retriever_dependency/get_generator_dependency
already used gives all four the same 503 contract.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import backend.dependencies as deps
from backend.dependencies import _get_or_init


@pytest.mark.unit
class TestGetOrInit:
    def test_returns_existing_instance_without_calling_factory(self):
        existing = object()
        factory = MagicMock()

        result = _get_or_init(existing, factory, "irrelevant")

        assert result is existing
        factory.assert_not_called()

    def test_builds_via_factory_when_none(self):
        built = object()
        factory = MagicMock(return_value=built)

        result = _get_or_init(None, factory, "irrelevant")

        assert result is built
        factory.assert_called_once()

    def test_factory_failure_raises_503_with_message(self):
        def factory():
            raise RuntimeError("boom")

        with pytest.raises(HTTPException) as exc_info:
            _get_or_init(None, factory, "Widget service unavailable")

        assert exc_info.value.status_code == 503
        assert "Widget service unavailable" in exc_info.value.detail
        assert "boom" in exc_info.value.detail


@pytest.mark.unit
class TestConfigAndSessionManagerGettersNowRaise503:
    """Regression coverage for the behavior fix: these two getters used to
    let a construction failure propagate unhandled."""

    def test_get_config_dependency_raises_503_on_build_failure(self, monkeypatch):
        monkeypatch.setattr(deps, "_config", None)
        monkeypatch.setattr(
            deps, "_build_config", MagicMock(side_effect=RuntimeError("bad config"))
        )

        with pytest.raises(HTTPException) as exc_info:
            deps.get_config_dependency()

        assert exc_info.value.status_code == 503
        assert "Config service unavailable" in exc_info.value.detail

    def test_get_session_manager_dependency_raises_503_on_build_failure(self, monkeypatch):
        monkeypatch.setattr(deps, "_session_manager", None)
        monkeypatch.setattr(
            deps, "get_session_manager", MagicMock(side_effect=RuntimeError("bad session store"))
        )

        with pytest.raises(HTTPException) as exc_info:
            deps.get_session_manager_dependency()

        assert exc_info.value.status_code == 503
        assert "Session manager service unavailable" in exc_info.value.detail
