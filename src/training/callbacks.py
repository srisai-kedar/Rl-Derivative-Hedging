"""Stable-Baselines3 callbacks for SAC hedging training."""

from __future__ import annotations

import logging
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from src.training.env_factory import sync_normalization_stats

logger = logging.getLogger(__name__)


class HedgingCheckpointCallback(BaseCallback):
    """Save SAC model checkpoints and VecNormalize stats as inseparable pairs."""

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        train_env: VecNormalize,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.train_env = train_env
        self._last_save_step = 0
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save_step < self.save_freq:
            return True

        step = self.num_timesteps
        model_path = os.path.join(self.save_path, f"model_{step}_steps")
        vecnorm_path = os.path.join(self.save_path, f"vecnorm_{step}_steps.pkl")

        self.model.save(model_path)
        self.train_env.save(vecnorm_path)
        self._last_save_step = step

        if self.verbose >= 1:
            logger.info("Checkpoint saved at step %s", step)
        return True


class HedgingEvalCallback(BaseCallback):
    """
    Evaluate with synced observation normalization and save best model pairs.

    Evaluation rewards remain raw P&L values because the eval env is configured
    with ``norm_reward=False``.
    """

    def __init__(
        self,
        train_env: VecNormalize,
        eval_env: VecNormalize,
        eval_freq: int,
        n_eval_episodes: int,
        save_path: str,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self._last_eval_step = 0
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps
        sync_normalization_stats(source=self.train_env, target=self.eval_env)

        episode_rewards: list[float] = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = np.array([False])
            episode_reward = 0.0

            while not bool(done[0]):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                episode_reward += float(reward[0])

            episode_rewards.append(episode_reward)

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/n_episodes", self.n_eval_episodes)

        if self.verbose >= 1:
            logger.info(
                "Step %s | Eval mean reward: %.4f +/- %.4f",
                self.num_timesteps,
                mean_reward,
                std_reward,
            )

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            model_path = os.path.join(self.save_path, "best_model")
            vecnorm_path = os.path.join(self.save_path, "best_vecnorm.pkl")
            self.model.save(model_path)
            self.train_env.save(vecnorm_path)

            if self.verbose >= 1:
                logger.info("New best model saved (reward: %.4f)", mean_reward)

        return True


class HedgingMetricsCallback(BaseCallback):
    """Log rolling terminal P&L and transaction-cost metrics."""

    def __init__(self, window: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window = window
        self._terminal_pnls: list[float] = []
        self._total_costs: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "terminal_pnl" not in info:
                continue

            self._terminal_pnls.append(float(info["terminal_pnl"]))
            self._total_costs.append(float(info["total_cost"]))

            if len(self._terminal_pnls) > self.window:
                self._terminal_pnls.pop(0)
                self._total_costs.pop(0)

            self.logger.record(
                "hedging/terminal_pnl_mean",
                float(np.mean(self._terminal_pnls)),
            )
            self.logger.record(
                "hedging/terminal_pnl_std",
                float(np.std(self._terminal_pnls)),
            )
            self.logger.record(
                "hedging/total_cost_mean",
                float(np.mean(self._total_costs)),
            )

        return True
