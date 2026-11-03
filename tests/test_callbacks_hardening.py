"""Callback tests. Require stable-baselines3 (skipped when it is absent)."""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

import pytest

pytest.importorskip("stable_baselines3")

from src.training.callbacks import HedgingMetricsCallback


class _FakeLogger:
    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def record(self, key: str, value: float) -> None:
        self.records[key] = value


def _cb(window: int) -> HedgingMetricsCallback:
    cb = HedgingMetricsCallback(window=window)
    mock_model = MagicMock()
    mock_model.logger = _FakeLogger()
    cb.init_callback(mock_model)
    return cb


def test_window_is_bounded_deque_and_evicts_oldest():
    cb = _cb(window=3)
    assert isinstance(cb._terminal_pnls, deque)
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        cb.locals = {"infos": [{"terminal_pnl": v, "total_cost": v / 10}]}
        assert cb._on_step() is True
    assert list(cb._terminal_pnls) == [3.0, 4.0, 5.0]
    assert cb.logger.records["hedging/terminal_pnl_mean"] == pytest.approx(4.0)  # type: ignore[attr-defined]


def test_non_terminal_infos_are_ignored():
    cb = _cb(window=3)
    cb.locals = {"infos": [{"step_pnl": 1.0}]}
    cb._on_step()
    assert len(cb._terminal_pnls) == 0
