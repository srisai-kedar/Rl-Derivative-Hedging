"""Structured logging context shared by training and evaluation entry points."""

from __future__ import annotations

import contextvars
import logging

_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")


class RunIdFilter(logging.Filter):
    """Attach the current run identifier as structured LogRecord data."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get()
        return True


def configure_run_logging(run_id: str) -> None:
    """Set a run id and attach it without changing existing formatters/output."""
    _run_id.set(run_id)
    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(item, RunIdFilter) for item in handler.filters):
            handler.addFilter(RunIdFilter())


def current_run_id() -> str:
    """Return the current structured logging run id."""
    return _run_id.get()
