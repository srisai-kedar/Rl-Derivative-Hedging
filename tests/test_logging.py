"""Tests for run-id structured logging context."""

from __future__ import annotations

import logging

from src.utils.logging import RunIdFilter, configure_run_logging, current_run_id


def test_run_id_is_attached_without_changing_formatter():
    handler = logging.StreamHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        configure_run_logging("test-run")
        record = logging.LogRecord(
            "test", logging.INFO, __file__, 1, "message", (), None
        )
        assert RunIdFilter().filter(record)
        assert record.run_id == "test-run"
        assert current_run_id() == "test-run"
        assert handler.formatter is None
    finally:
        root.removeHandler(handler)
