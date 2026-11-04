"""Shared-seed backtesting utilities for hedging-policy evaluation."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

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
    def from_episode_results(cls, results: list[EpisodeResult]) -> BacktestResults:
        """Construct BacktestResults column-wise from episode results."""
        if not results:
            return cls(
                episode_df=pd.DataFrame(columns=EPISODE_COLUMNS),
                step_df=pd.DataFrame(columns=STEP_COLUMNS),
            )
        episode_df = pd.DataFrame(
            {
                "episode_id": np.array([r.episode_id for r in results], dtype=np.int64),
                "agent_type": np.array([r.agent_type for r in results], dtype=object),
                "seed": np.array([r.seed for r in results], dtype=np.int64),
                "terminal_pnl": np.array([r.terminal_pnl for r in results], dtype=np.float64),
                "total_cost": np.array([r.total_cost for r in results], dtype=np.float64),
                "n_steps": np.array([r.n_steps for r in results], dtype=np.int64),
            },
            columns=EPISODE_COLUMNS,
        )

        step_counts = [r.n_steps for r in results]
        step_df = pd.DataFrame(
            {
                "episode_id": np.repeat(np.array([r.episode_id for r in results], dtype=np.int64), step_counts),
                "agent_type": np.repeat(np.array([r.agent_type for r in results], dtype=object), step_counts),
                "step": np.concatenate([np.arange(r.n_steps, dtype=np.int64) for r in results]),
                "price": np.concatenate([r.prices for r in results]).astype(np.float32),
                "hedge_ratio": np.concatenate([r.hedge_ratios for r in results]).astype(np.float32),
                "bs_delta": np.concatenate([r.bs_deltas for r in results]).astype(np.float32),
                "step_pnl": np.concatenate([r.step_pnls for r in results]).astype(np.float32),
                "step_cost": np.concatenate([r.step_costs for r in results]).astype(np.float32),
            },
            columns=STEP_COLUMNS,
        )
        return cls(episode_df=episode_df, step_df=step_df)

    def filter_agent(self, agent_type: str) -> BacktestResults:
        """Return results for one policy only."""
        return BacktestResults(
            self.episode_df[self.episode_df["agent_type"] == agent_type].reset_index(drop=True),
            self.step_df[self.step_df["agent_type"] == agent_type].reset_index(drop=True),
        )


@runtime_checkable
class HedgingPolicy(Protocol):
    """Structural interface implemented by every evaluated hedging policy."""

    name: ClassVar[str]

    def predict(self, obs: np.ndarray, info: dict) -> float:
        """Return a target hedge ratio in [0, 1]."""
        ...

    def predict_batch(self, obs_batch: np.ndarray, infos: list[dict]) -> np.ndarray:
        """Return target hedge ratios for a batch of observations in [0, 1]."""
        ...


class RLAgentPolicy:
    """SAC policy evaluated on raw observations with saved normalisation stats."""

    name: ClassVar[str] = "rl_agent"

    def __init__(self, model: SAC, vec_normalize: VecNormalize, deterministic: bool = True) -> None:
        self.model = model
        self.vn = vec_normalize
        self.deterministic = deterministic

    def predict(self, obs: np.ndarray, info: dict) -> float:
        action, _ = self.model.predict(self._normalise(obs)[np.newaxis, :], deterministic=self.deterministic)
        return float(np.clip(action[0, 0], 0.0, 1.0))

    def predict_batch(self, obs_batch: np.ndarray, infos: list[dict]) -> np.ndarray:
        norm_obs = self._normalise_batch(obs_batch)
        actions, _ = self.model.predict(norm_obs, deterministic=self.deterministic)
        actions_flat = np.asarray(actions, dtype=np.float32).reshape(-1)
        return np.clip(actions_flat, 0.0, 1.0)

    def _normalise(self, obs: np.ndarray) -> np.ndarray:
        return self._normalise_batch(obs[np.newaxis, :])[0]

    def _normalise_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        epsilon = getattr(self.vn, "epsilon", 1e-8)
        obs_mean = self.vn.obs_rms.mean  # type: ignore[attr-defined]
        obs_var = self.vn.obs_rms.var  # type: ignore[attr-defined]
        normalised = (obs_batch - obs_mean) / np.sqrt(obs_var + epsilon)
        return np.clip(normalised, -self.vn.clip_obs, self.vn.clip_obs).astype(np.float32)


class BSDeltaPolicy:
    """Textbook Black-Scholes delta hedge baseline."""

    name: ClassVar[str] = "bs_delta"

    def predict(self, obs: np.ndarray, info: dict) -> float:
        return float(np.clip(info["bs_delta"], 0.0, 1.0))

    def predict_batch(self, obs_batch: np.ndarray, infos: list[dict]) -> np.ndarray:
        deltas = np.array([info["bs_delta"] for info in infos], dtype=np.float32)
        return np.clip(deltas, 0.0, 1.0)


class ZeroHedgePolicy:
    """Unhedged short-call lower-bound reference policy."""

    name: ClassVar[str] = "zero_hedge"

    def predict(self, obs: np.ndarray, info: dict) -> float:
        return 0.0

    def predict_batch(self, obs_batch: np.ndarray, infos: list[dict]) -> np.ndarray:
        return np.zeros(len(obs_batch), dtype=np.float32)


def _initial_info(info: dict) -> dict:
    """Use the state information supplied by ``HedgingEnv.reset()``."""
    return info


def _run_single_episode(policy: HedgingPolicy, env: HedgingEnv, episode_id: int, seed: int) -> EpisodeResult:
    """Run one raw-environment episode and collect its complete trajectory."""
    obs, reset_info = env.reset(seed=seed)
    info = _initial_info(reset_info)
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
        episode_id=episode_id,
        agent_type=policy.name,
        seed=seed,
        terminal_pnl=float(info["terminal_pnl"]),
        total_cost=float(info["total_cost"]),
        n_steps=len(step_pnls),
        step_pnls=np.asarray(step_pnls, dtype=np.float32),
        step_costs=np.asarray(step_costs, dtype=np.float32),
        hedge_ratios=np.asarray(hedge_ratios, dtype=np.float32),
        bs_deltas=np.asarray(bs_deltas, dtype=np.float32),
        prices=np.asarray(prices, dtype=np.float32),
    )


def _predict_policy_batch(policy: HedgingPolicy, obs_batch: np.ndarray, infos: list[dict]) -> np.ndarray:
    """Helper to dispatch batch prediction if implemented, else element-wise."""
    if hasattr(policy, "predict_batch"):
        return policy.predict_batch(obs_batch, infos)
    return np.array([policy.predict(obs, info) for obs, info in zip(obs_batch, infos, strict=True)], dtype=np.float32)


def _run_batched_policy_episodes(
    policy: HedgingPolicy,
    env_config: EnvironmentConfig,
    seeds: list[int],
    batch_size: int = 128,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run episodes in lock-step batches across environments and return (episode_df, step_df)."""
    n_episodes = len(seeds)
    if n_episodes == 0:
        return pd.DataFrame(columns=EPISODE_COLUMNS), pd.DataFrame(columns=STEP_COLUMNS)

    episode_dfs: list[pd.DataFrame] = []
    step_dfs: list[pd.DataFrame] = []

    pbar = tqdm(total=n_episodes, desc=policy.name, disable=not show_progress)
    effective_batch_size = max(1, batch_size)

    for start_idx in range(0, n_episodes, effective_batch_size):
        chunk_seeds = seeds[start_idx : start_idx + effective_batch_size]
        B = len(chunk_seeds)
        chunk_ep_ids = np.arange(start_idx, start_idx + B, dtype=np.int64)

        envs = [HedgingEnv(**env_config.as_dict()) for _ in range(B)]
        try:
            obs_list = []
            info_list = []
            for j in range(B):
                obs_j, reset_info_j = envs[j].reset(seed=chunk_seeds[j])
                info_j = _initial_info(reset_info_j)
                obs_list.append(obs_j)
                info_list.append(info_j)

            obs_batch = np.stack(obs_list, axis=0)
            n_steps = envs[0].T

            prices = np.empty((B, n_steps), dtype=np.float32)
            hedge_ratios = np.empty((B, n_steps), dtype=np.float32)
            bs_deltas = np.empty((B, n_steps), dtype=np.float32)
            step_pnls = np.empty((B, n_steps), dtype=np.float32)
            step_costs = np.empty((B, n_steps), dtype=np.float32)

            for t in range(n_steps):
                prices[:, t] = [inf["S"] for inf in info_list]
                bs_deltas[:, t] = [inf["bs_delta"] for inf in info_list]

                actions = _predict_policy_batch(policy, obs_batch, info_list)
                hedge_ratios[:, t] = actions

                for j in range(B):
                    obs_j, _, _, _, inf_j = envs[j].step(np.array([actions[j]], dtype=np.float32))
                    obs_batch[j] = obs_j
                    info_list[j] = inf_j
                    step_pnls[j, t] = inf_j["step_pnl"]
                    step_costs[j, t] = inf_j["transaction_cost"]

            terminal_pnls = np.array([inf["terminal_pnl"] for inf in info_list], dtype=np.float64)
            total_costs = np.array([inf["total_cost"] for inf in info_list], dtype=np.float64)

            chunk_ep_df = pd.DataFrame(
                {
                    "episode_id": chunk_ep_ids,
                    "agent_type": np.array([policy.name] * B, dtype=object),
                    "seed": np.array(chunk_seeds, dtype=np.int64),
                    "terminal_pnl": terminal_pnls,
                    "total_cost": total_costs,
                    "n_steps": np.full(B, n_steps, dtype=np.int64),
                },
                columns=EPISODE_COLUMNS,
            )

            chunk_step_df = pd.DataFrame(
                {
                    "episode_id": np.repeat(chunk_ep_ids, n_steps),
                    "agent_type": np.repeat(np.array([policy.name] * B, dtype=object), n_steps),
                    "step": np.tile(np.arange(n_steps, dtype=np.int64), B),
                    "price": prices.ravel(),
                    "hedge_ratio": hedge_ratios.ravel(),
                    "bs_delta": bs_deltas.ravel(),
                    "step_pnl": step_pnls.ravel(),
                    "step_cost": step_costs.ravel(),
                },
                columns=STEP_COLUMNS,
            )

            episode_dfs.append(chunk_ep_df)
            step_dfs.append(chunk_step_df)
            pbar.update(B)
        finally:
            for env in envs:
                env.close()

    pbar.close()
    return pd.concat(episode_dfs, ignore_index=True), pd.concat(step_dfs, ignore_index=True)


def _run_backtest_legacy(
    policies: list[HedgingPolicy],
    env_config: EnvironmentConfig,
    n_episodes: int,
    show_progress: bool = True,
) -> BacktestResults:
    """Legacy sequential backtester (reference implementation)."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    all_results: list[EpisodeResult] = []
    seeds = list(range(n_episodes))
    for policy in policies:
        env = HedgingEnv(**env_config.as_dict())
        logger.info("Running %s episodes for policy: %s (legacy)", n_episodes, policy.name)
        try:
            for episode_id, seed in tqdm(
                enumerate(seeds),
                total=n_episodes,
                desc=f"{policy.name} (legacy)",
                disable=not show_progress,
            ):
                all_results.append(_run_single_episode(policy, env, episode_id, seed))
        finally:
            env.close()
    return BacktestResults.from_episode_results(all_results)


def run_backtest(
    policies: list[HedgingPolicy],
    env_config: EnvironmentConfig,
    n_episodes: int,
    show_progress: bool = True,
    batch_size: int = 128,
    legacy: bool = False,
) -> BacktestResults:
    """Run all policies on identical seed-indexed price paths.

    Parameters
    ----------
    policies:
        List of policies to evaluate.
    env_config:
        Environment parameters.
    n_episodes:
        Number of episodes per policy.
    show_progress:
        Whether to show progress bar.
    batch_size:
        Batch size for lock-step vectorised evaluation (default 128).
    legacy:
        If True, run the legacy sequential backtester.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if legacy:
        return _run_backtest_legacy(policies, env_config, n_episodes, show_progress=show_progress)

    seeds = list(range(n_episodes))
    ep_dfs: list[pd.DataFrame] = []
    step_dfs: list[pd.DataFrame] = []

    for policy in policies:
        logger.info("Running %s batched episodes for policy: %s", n_episodes, policy.name)
        policy_ep_df, policy_step_df = _run_batched_policy_episodes(
            policy=policy,
            env_config=env_config,
            seeds=seeds,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        ep_dfs.append(policy_ep_df)
        step_dfs.append(policy_step_df)

    combined_ep_df = pd.concat(ep_dfs, ignore_index=True) if ep_dfs else pd.DataFrame(columns=EPISODE_COLUMNS)
    combined_step_df = pd.concat(step_dfs, ignore_index=True) if step_dfs else pd.DataFrame(columns=STEP_COLUMNS)
    return BacktestResults(episode_df=combined_ep_df, step_df=combined_step_df)


def run_robustness_sweep(
    policies: list[HedgingPolicy],
    base_config: EnvironmentConfig,
    kappa_values: list[float],
    sigma_values: list[float],
    n_episodes: int = 200,
    batch_size: int = 128,
    legacy: bool = False,
) -> pd.DataFrame:
    """Backtest every policy over the requested volatility/cost grid."""
    from src.evaluation.metrics import compute_all_metrics

    rows: list[dict] = []
    combinations = [(sigma, kappa) for sigma in sigma_values for kappa in kappa_values]
    for sigma, kappa in tqdm(combinations, desc="Robustness sweep"):
        sweep_config = EnvironmentConfig(
            **{**base_config.as_dict(), "sigma": sigma, "kappa": kappa, "randomise_params": False}
        )
        results = run_backtest(
            policies,
            sweep_config,
            n_episodes,
            show_progress=False,
            batch_size=batch_size,
            legacy=legacy,
        )
        for policy in policies:
            rows.append(
                {
                    "kappa": kappa,
                    "sigma": sigma,
                    "agent_type": policy.name,
                    **compute_all_metrics(results, policy.name),
                }
            )
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
