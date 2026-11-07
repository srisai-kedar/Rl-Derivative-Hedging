"""Callback tests. Require stable-baselines3 (skipped when it is absent)."""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from src.training.callbacks import HedgingEvalCallback, HedgingMetricsCallback
from src.training.env_factory import build_eval_env, build_training_envs
from src.training.hyperparams import EnvironmentConfig, VecNormalizeConfig


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


def test_hedging_eval_callback_replays_identical_paths_with_eval_seed_base(tmp_path):
    train_env = build_training_envs(
        EnvironmentConfig(), VecNormalizeConfig(), n_envs=1, seed=42
    )
    eval_env = build_eval_env(EnvironmentConfig(), train_env, seed=42)

    mock_model = MagicMock()
    mock_model.predict.return_value = (np.array([[0.5]]), None)
    fake_logger = _FakeLogger()
    mock_model.logger = fake_logger

    cb = HedgingEvalCallback(
        train_env=train_env,
        eval_env=eval_env,
        eval_freq=10,
        n_eval_episodes=2,
        save_path=str(tmp_path / "checkpoints"),
        eval_seed_base=5000,
        verbose=0,
    )
    cb.init_callback(mock_model)

    cb.num_timesteps = 10
    cb._on_step()
    mean_round_1 = fake_logger.records["eval/mean_reward"]
    std_round_1 = fake_logger.records["eval/std_reward"]

    cb.num_timesteps = 20
    cb._on_step()
    mean_round_2 = fake_logger.records["eval/mean_reward"]
    std_round_2 = fake_logger.records["eval/std_reward"]

    assert mean_round_1 == mean_round_2
    assert std_round_1 == std_round_2


def test_hedging_eval_callback_uses_raw_step_pnl_when_risk_aversion_set(tmp_path):
    train_env = build_training_envs(
        EnvironmentConfig(), VecNormalizeConfig(), n_envs=1, seed=42
    )
    eval_env = build_eval_env(EnvironmentConfig(risk_aversion=2.0), train_env, seed=42)

    mock_model = MagicMock()
    mock_model.predict.return_value = (np.array([[0.5]]), None)
    fake_logger = _FakeLogger()
    mock_model.logger = fake_logger

    cb = HedgingEvalCallback(
        train_env=train_env,
        eval_env=eval_env,
        eval_freq=10,
        n_eval_episodes=1,
        save_path=str(tmp_path / "checkpoints"),
        eval_seed_base=7000,
        verbose=0,
    )
    cb.init_callback(mock_model)
    cb.num_timesteps = 10
    cb._on_step()

    eval_mean = fake_logger.records["eval/mean_reward"]

    # Replay identical trajectory directly to obtain raw vs shaped sums
    eval_env.seed(7000)
    eval_env.reset()
    done = np.array([False])
    raw_pnl_sum = 0.0
    shaped_reward_sum = 0.0
    while not done[0]:
        _, reward, done, infos = eval_env.step(np.array([[0.5]]))
        raw_pnl_sum += float(infos[0]["step_pnl"])
        shaped_reward_sum += float(reward[0])

    assert eval_mean == pytest.approx(raw_pnl_sum, rel=1e-5)
    assert abs(eval_mean - shaped_reward_sum) > 0.01
