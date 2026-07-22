"""Shared-seed backtesting utilities for hedging-policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from tqdm import tqdm

from src.envs.hedging_env import HedgingEnv
from src.training.hyperparams import EnvironmentConfig

logger = logging.getLogger(__name__)
EPISODE_COLUMNS = ["episode_id", "agent_type", "seed", "terminal_pnl", "total_cost", "n_steps"]
STEP_COLUMNS = ["episode_id", "agent_type", "step", "price", "hedge_ratio", "bs_delta", "step_pnl", "step_cost"]


@dataclass
class EpisodeResult:
    """All data collected from one episode of one policy."""

    episode_id: int
    agent_type: str
    seed: int
    terminal_pnl: float
    total_cost: float
    n_steps: int
    step_pnls: np.ndarray
    step_costs: np.ndarray
    hedge_ratios: np.ndarray
    bs_deltas: np.ndarray
    prices: np.ndarray


@dataclass
class BacktestResults:
    """Aggregate episode-level and step-level backtest data."""

    episode_df: pd.DataFrame
    step_df: pd.DataFrame

    @classmethod
    def from_episode_results(cls, results: list[EpisodeResult]) -> "BacktestResults":
        episode_rows: list[dict] = []
        step_rows: list[dict] = []
        for result in results:
            episode_rows.append({
                "episode_id": result.episode_id, "agent_type": result.agent_type,
                "seed": result.seed, "terminal_pnl": result.terminal_pnl,
                "total_cost": result.total_cost, "n_steps": result.n_steps,
            })
            for step_idx in range(result.n_steps):
                step_rows.append({
                    "episode_id": result.episode_id, "agent_type": result.agent_type,
                    "step": step_idx, "price": result.prices[step_idx],
                    "hedge_ratio": result.hedge_ratios[step_idx],
                    "bs_delta": result.bs_deltas[step_idx], "step_pnl": result.step_pnls[step_idx],
                    "step_cost": result.step_costs[step_idx],
                })
        return cls(
            episode_df=pd.DataFrame(episode_rows, columns=EPISODE_COLUMNS),
            step_df=pd.DataFrame(step_rows, columns=STEP_COLUMNS),
        )

    def filter_agent(self, agent_type: str) -> "BacktestResults":
        """Return results for one policy only."""
        return BacktestResults(
            self.episode_df[self.episode_df["agent_type"] == agent_type].reset_index(drop=True),
            self.step_df[self.step_df["agent_type"] == agent_type].reset_index(drop=True),
        )


@runtime_checkable
class HedgingPolicy(Protocol):
    """Structural interface implemented by every evaluated hedging policy."""

    name: str

    def predict(self, obs: np.ndarray, info: dict) -> float:
        """Return a target hedge ratio in [0, 1]."""
        ...


class RLAgentPolicy:
    """SAC policy evaluated on raw observations with saved normalisation stats."""

    name = "rl_agent"

    def __init__(self, model: SAC, vec_normalize: VecNormalize, deterministic: bool = True) -> None:
        self.model = model
        self.vn = vec_normalize
        self.deterministic = deterministic

    def predict(self, obs: np.ndarray, info: dict) -> float:
        action, _ = self.model.predict(self._normalise(obs)[np.newaxis, :], deterministic=self.deterministic)
        return float(np.clip(action[0, 0], 0.0, 1.0))

    def _normalise(self, obs: np.ndarray) -> np.ndarray:
        epsilon = getattr(self.vn, "epsilon", 1e-8)
        normalised = (obs - self.vn.obs_rms.mean) / np.sqrt(self.vn.obs_rms.var + epsilon)
        return np.clip(normalised, -self.vn.clip_obs, self.vn.clip_obs).astype(np.float32)


class BSDeltaPolicy:
    """Textbook Black-Scholes delta hedge baseline."""

    name = "bs_delta"

    def predict(self, obs: np.ndarray, info: dict) -> float:
        return float(np.clip(info["bs_delta"], 0.0, 1.0))


class ZeroHedgePolicy:
    """Unhedged short-call lower-bound reference policy."""

    name = "zero_hedge"

    def predict(self, obs: np.ndarray, info: dict) -> float:
        return 0.0


def _initial_info(env: HedgingEnv, obs: np.ndarray, info: dict) -> dict:
    """Supply step-zero policy information without changing the Phase 2 environment."""
    if "bs_delta" in info:
        return info
    assert env.state is not None
    return {
        "S": float(env.state.S), "bs_delta": float(obs[6]),
        "hedge_pos": float(env.state.hedge_pos), "option_value": float(env.state.option_value),
    }


def _run_single_episode(policy: HedgingPolicy, env: HedgingEnv, episode_id: int, seed: int) -> EpisodeResult:
    """Run one raw-environment episode and collect its complete trajectory."""
    obs, reset_info = env.reset(seed=seed)
    info = _initial_info(env, obs, reset_info)
    step_pnls: list[float] = []
    step_costs: list[float] = []
    hedge_ratios: list[float] = []
    bs_deltas: list[float] = []
    prices: list[float] = []
    terminated = truncated = False
    while not (terminated or truncated):
        prices.append(float(info["S"]))
        bs_deltas.append(float(info["bs_delta"]))
        action = policy.predict(obs, info)
        hedge_ratios.append(action)
        obs, _, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        step_pnls.append(float(info["step_pnl"]))
        step_costs.append(float(info["transaction_cost"]))
    return EpisodeResult(
        episode_id=episode_id, agent_type=policy.name, seed=seed,
        terminal_pnl=float(info["terminal_pnl"]), total_cost=float(info["total_cost"]),
        n_steps=len(step_pnls), step_pnls=np.asarray(step_pnls, dtype=np.float32),
        step_costs=np.asarray(step_costs, dtype=np.float32),
        hedge_ratios=np.asarray(hedge_ratios, dtype=np.float32),
        bs_deltas=np.asarray(bs_deltas, dtype=np.float32), prices=np.asarray(prices, dtype=np.float32),
    )


def run_backtest(policies: list[HedgingPolicy], env_config: EnvironmentConfig, n_episodes: int, show_progress: bool = True) -> BacktestResults:
    """Run all policies on identical seed-indexed price paths."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    all_results: list[EpisodeResult] = []
    seeds = list(range(n_episodes))
    for policy in policies:
        env = HedgingEnv(**env_config.as_dict())
        logger.info("Running %s episodes for policy: %s", n_episodes, policy.name)
        try:
            for episode_id, seed in tqdm(enumerate(seeds), total=n_episodes, desc=policy.name, disable=not show_progress):
                all_results.append(_run_single_episode(policy, env, episode_id, seed))
        finally:
            env.close()
    return BacktestResults.from_episode_results(all_results)


def run_robustness_sweep(policies: list[HedgingPolicy], base_config: EnvironmentConfig, kappa_values: list[float], sigma_values: list[float], n_episodes: int = 200) -> pd.DataFrame:
    """Backtest every policy over the requested volatility/cost grid."""
    from src.evaluation.metrics import compute_all_metrics

    rows: list[dict] = []
    combinations = [(sigma, kappa) for sigma in sigma_values for kappa in kappa_values]
    for sigma, kappa in tqdm(combinations, desc="Robustness sweep"):
        sweep_config = EnvironmentConfig(**{**base_config.as_dict(), "sigma": sigma, "kappa": kappa, "randomise_params": False})
        results = run_backtest(policies, sweep_config, n_episodes, show_progress=False)
        for policy in policies:
            rows.append({"kappa": kappa, "sigma": sigma, "agent_type": policy.name, **compute_all_metrics(results, policy.name)})
    return pd.DataFrame(rows)


def save_results(results: BacktestResults, metrics: dict[str, dict[str, float]], output_dir: str) -> None:
    """Persist episode data, step data, and metrics for later dashboard use."""
    os.makedirs(output_dir, exist_ok=True)
    results.episode_df.to_csv(os.path.join(output_dir, "episode_data.csv"), index=False)
    results.step_df.to_csv(os.path.join(output_dir, "step_data.csv"), index=False)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=float)
    logger.info("Results saved to %s", output_dir)


def load_results(output_dir: str) -> tuple[BacktestResults, dict]:
    """Load prior persisted evaluation results."""
    episode_df = pd.read_csv(os.path.join(output_dir, "episode_data.csv"))
    step_df = pd.read_csv(os.path.join(output_dir, "step_data.csv"))
    with open(os.path.join(output_dir, "metrics.json"), encoding="utf-8") as file:
        metrics = json.load(file)
    return BacktestResults(episode_df, step_df), metrics
