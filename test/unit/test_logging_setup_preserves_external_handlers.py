"""Regression test found by a bug sweep on the logging-persistence diff
itself: configure_logging() originally called root.handlers.clear()
unconditionally, discarding any handler another library had already
attached to the root logger -- a real behavior change from the
logging.basicConfig() this replaced, which is a documented no-op once any
handler exists. Nothing in this repo's test suite used `caplog` at the
time this was found, so it wasn't an active failure, but it was a landmine
for the first test that did (pytest's logging plugin attaches its own
handler to support that fixture), and for anyone passing
`--log-cli-level`. Fixed to remove only the handlers this module itself
previously installed.
"""

import logging

import pytest

from backend.logging_setup import configure_logging


@pytest.mark.unit
class TestConfigureLoggingPreservesExternalHandlers:
    def test_a_pre_existing_external_handler_survives_configure_logging(self):
        root = logging.getLogger()
        external_handler = logging.NullHandler()
        root.addHandler(external_handler)
        try:
            configure_logging()

            assert external_handler in root.handlers
        finally:
            root.removeHandler(external_handler)

    def test_re_calling_configure_logging_does_not_duplicate_its_own_handlers(self):
        root = logging.getLogger()

        configure_logging()
        first_count = len(root.handlers)
        configure_logging()
        second_count = len(root.handlers)

        assert second_count == first_count
