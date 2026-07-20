"""Factories for Stable-Baselines3 hedging environments."""

from __future__ import annotations

import copy
import os
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.hedging_env import HedgingEnv
from src.training.hyperparams import EnvironmentConfig, VecNormalizeConfig


class FreshResetSeedWrapper(gym.Wrapper):
    """
    Provides deterministic fresh reset seeds when SB3 resets without one.

    ``HedgingEnv`` preserves explicit reset seeds for reproducible tests and
    evaluation. During training, SB3 resets environments without passing a seed
    after the first reset; this wrapper advances a local seed stream so each
    episode receives a fresh, reproducible GBM path.
    """

    def __init__(self, env: HedgingEnv, seed: int) -> None:
        super().__init__(env)
        self._reset_rng = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is None:
            seed = int(self._reset_rng.integers(0, np.iinfo(np.int32).max))
        return self.env.reset(seed=seed, options=options)


def _make_env_fn(
    env_config: EnvironmentConfig,
    rank: int,
    base_seed: int,
    log_dir: str | None = None,
) -> Callable[[], Monitor]:
    """
    Return a factory that creates one monitored ``HedgingEnv``.

    Each worker receives a unique base seed to decorrelate replay-buffer
    experience across vectorized environments.
    """

    def _init() -> Monitor:
        env_seed = base_seed + rank
        env = HedgingEnv(**env_config.as_dict(), seed=env_seed)
        env = FreshResetSeedWrapper(env, seed=env_seed)
        monitor_path = os.path.join(log_dir, f"env_{rank}") if log_dir else None
        return Monitor(env, filename=monitor_path)

    return _init


def build_training_envs(
    env_config: EnvironmentConfig,
    vn_config: VecNormalizeConfig,
    n_envs: int,
    seed: int,
    log_dir: str | None = None,
) -> VecNormalize:
    """
    Build the training environment stack.

    Wrapping order is ``HedgingEnv -> Monitor -> DummyVecEnv -> VecNormalize``.
    Observations and rewards are normalized, and statistics update in training.
    """
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    env_fns = [
        _make_env_fn(env_config, rank=i, base_seed=seed, log_dir=log_dir)
        for i in range(n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)
    return VecNormalize(
        vec_env,
        norm_obs=vn_config.norm_obs,
        norm_reward=vn_config.norm_reward,
        clip_obs=vn_config.clip_obs,
        clip_reward=vn_config.clip_reward,
    )


def build_eval_env(
    env_config: EnvironmentConfig,
    train_env: VecNormalize,
    seed: int,
) -> VecNormalize:
    """
    Build a single evaluation environment with frozen normalization stats.

    Evaluation returns raw rewards, so ``norm_reward`` is disabled while
    observation normalization is synced from the current training environment.
    """
    eval_env = DummyVecEnv(
        [_make_env_fn(env_config, rank=0, base_seed=seed + 10_000)]
    )
    vec_env = VecNormalize(
        eval_env,
        norm_obs=train_env.norm_obs,
        norm_reward=False,
        clip_obs=train_env.clip_obs,
        clip_reward=train_env.clip_reward,
        training=False,
    )
    sync_normalization_stats(source=train_env, target=vec_env)
    return vec_env


def sync_normalization_stats(source: VecNormalize, target: VecNormalize) -> None:
    """Copy VecNormalize running statistics from ``source`` to ``target``."""
    target.obs_rms = copy.deepcopy(source.obs_rms)
    target.ret_rms = copy.deepcopy(source.ret_rms)


def load_env_for_inference(
    env_config: EnvironmentConfig,
    vecnorm_path: str,
    seed: int = 0,
) -> VecNormalize:
    """
    Load saved VecNormalize statistics around a fresh evaluation environment.

    The loaded environment is ready for deterministic inference/evaluation and
    returns raw rewards.
    """
    eval_env = DummyVecEnv([_make_env_fn(env_config, rank=0, base_seed=seed)])
    vec_env = VecNormalize.load(vecnorm_path, eval_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env
