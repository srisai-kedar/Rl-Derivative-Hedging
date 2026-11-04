"""
Hyperparameter optimisation for the RL hedging agent.

Uses Optuna with TPE sampler and Median pruner. Run with:
    python -m src.training.hpo --trials 20 --config configs/training.yaml

Each trial trains SAC for HPO_TIMESTEPS steps and evaluates on 50 episodes.
Total wall-clock time: approximately 10-30 minutes for 20 trials on CPU,
depending on n_envs and hardware.

Results are printed to stdout and optionally saved to an SQLite database
for resumption. To view results interactively:
    optuna-dashboard sqlite:///hpo.db
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)

HPO_TIMESTEPS: int = 300_000
HPO_N_EVAL_EPISODES: int = 50
HPO_N_ENVS: int = 2
HPO_EVAL_SEED_START: int = 10_000
HPO_N_SEEDS_PER_TRIAL: int = 1
HPO_OBJECTIVE: str = "mean_reward"
HPO_RISK_LAMBDA: float = 0.0
HPO_OBSERVATION_MODES = (
    "full",
    "no_greeks",
    "price_only",
    "no_bs_delta",
    "no_bs_delta_no_sigma",
)


def suggest_sac_params(
    trial: optuna.Trial,
    search_risk_aversion: bool = False,
    search_observation_mode: bool = False,
) -> dict[str, Any]:
    """Define the SAC and optional environment search space."""
    params: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "tau": trial.suggest_float("tau", 0.001, 0.02),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.01, 0.05, 0.1]),
        "net_arch_str": trial.suggest_categorical(
            "net_arch_str",
            ["64-64", "128-128", "256-256", "256-256-128"],
        ),
    }
    if search_risk_aversion:
        params["risk_aversion"] = trial.suggest_float("risk_aversion", 0.0, 1.0)
    if search_observation_mode:
        params["observation_mode"] = trial.suggest_categorical(
            "observation_mode", list(HPO_OBSERVATION_MODES)
        )
    return params


def _net_arch_from_str(arch_str: str) -> list[int]:
    """Convert ``256-256`` to ``[256, 256]``."""
    return [int(x) for x in arch_str.split("-")]


def score_episode_rewards(
    rewards: list[float], objective: str = HPO_OBJECTIVE, risk_lambda: float = HPO_RISK_LAMBDA
) -> float:
    """Score episode rewards using a mean, mean-minus-risk, or CVaR objective."""
    if not rewards:
        raise ValueError("At least one episode reward is required")
    values = np.asarray(rewards, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Episode rewards must be finite")
    if risk_lambda < 0:
        raise ValueError("risk_lambda must be non-negative")
    if objective == "mean_reward":
        return float(np.mean(values))
    if objective == "mean_minus_lambda_std":
        return float(np.mean(values) - risk_lambda * np.std(values))
    if objective == "cvar":
        return float(np.mean(values[values <= np.quantile(values, 0.05)]))
    raise ValueError(
        "objective must be one of 'mean_reward', 'mean_minus_lambda_std', or 'cvar'"
    )


def _eval_seed_block(
    n_episodes: int,
    seed_start: int,
    legacy_seed_behavior: bool,
) -> list[int]:
    """Return the evaluation seeds, with legacy mode reproducing 0..N-1."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    return list(range(n_episodes)) if legacy_seed_behavior else list(
        range(seed_start, seed_start + n_episodes)
    )


def _trial_seeds(
    base_seed: int,
    trial_number: int,
    n_seeds: int,
    legacy_seed_behavior: bool,
) -> list[int]:
    """Return deterministic training seeds for one trial."""
    if n_seeds <= 0:
        raise ValueError("n_seeds_per_trial must be positive")
    if legacy_seed_behavior:
        return [base_seed]
    return [base_seed + trial_number * n_seeds + offset for offset in range(n_seeds)]


def _sqlite_metadata_path(storage: str | None) -> Path | None:
    """Return the sidecar metadata path for a SQLite storage URL."""
    if not storage or not storage.startswith("sqlite:///"):
        return None
    database = storage.removeprefix("sqlite:///")
    if database.startswith("/"):
        database = database[1:]
    return Path(database).with_name(f"{Path(database).name}.metadata.json")


def _write_study_metadata(
    storage: str | None,
    *,
    study_name: str,
    config_path: str,
    eval_seed_start: int,
    n_eval_episodes: int,
    n_seeds_per_trial: int,
    objective_name: str,
    risk_lambda: float,
    search_risk_aversion: bool,
    search_observation_mode: bool,
    legacy_seed_behavior: bool,
    study: optuna.Study | None = None,
) -> Path | None:
    """Write reproducibility metadata beside a persistent SQLite study."""
    metadata_path = _sqlite_metadata_path(storage)
    if metadata_path is None:
        return None
    from src.training.hyperparams import load_config
    from src.utils.manifest import create_manifest

    config = load_config(config_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "study_name": study_name,
        "config_path": config_path,
        "evaluation_seeds": {
            "start": eval_seed_start,
            "n_episodes": n_eval_episodes,
            "legacy_seed_behavior": legacy_seed_behavior,
        },
        "training_seeds": {
            "base": config.run.seed,
            "n_seeds_per_trial": n_seeds_per_trial,
            "legacy_seed_behavior": legacy_seed_behavior,
        },
        "objective": {"name": objective_name, "risk_lambda": risk_lambda},
        "search_space": {
            "sac": [
                "learning_rate", "batch_size", "tau", "gamma", "ent_coef", "net_arch_str"
            ],
            "risk_aversion": search_risk_aversion,
            "observation_mode": list(HPO_OBSERVATION_MODES) if search_observation_mode else None,
        },
        "study": {
            "trials_completed": len(study.trials) if study is not None else 0,
        },
        "manifest": create_manifest(
            run_type="hpo",
            config=config,
            seed=config.run.seed,
            extra_metadata={
                "study_name": study_name,
                "storage": storage,
                "evaluation_seed_start": eval_seed_start,
                "n_eval_episodes": n_eval_episodes,
                "n_seeds_per_trial": n_seeds_per_trial,
                "objective": objective_name,
                "risk_lambda": risk_lambda,
                "legacy_seed_behavior": legacy_seed_behavior,
            },
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def _reset_eval_env(eval_env: Any, seed: int) -> np.ndarray:
    """Reset the wrapped single environment with an explicit episode seed."""
    raw_obs, _ = eval_env.venv.envs[0].reset(seed=seed)
    return eval_env.normalize_obs(np.asarray([raw_obs]))


def objective(
    trial: optuna.Trial,
    config_path: str,
    *,
    eval_seed_start: int | None = None,
    n_eval_episodes: int | None = None,
    n_seeds_per_trial: int | None = None,
    objective_name: str | None = None,
    risk_lambda: float | None = None,
    search_risk_aversion: bool = False,
    search_observation_mode: bool = False,
    legacy_seed_behavior: bool = False,
) -> float:
    """
    Optuna objective function for SAC hyperparameter search.

    Trains fresh SAC models with trial-suggested parameters and evaluates each
    on a held-out seed block. Multiple training seeds are averaged when
    ``n_seeds_per_trial`` is greater than one.

    Returns:
        Mean episode reward on the eval env (maximised by Optuna).

    Raises:
        optuna.TrialPruned: if the trial is pruned by MedianPruner.
    """
    from stable_baselines3 import SAC

    from src.training.env_factory import (
        build_eval_env,
        build_training_envs,
        sync_normalization_stats,
    )
    from src.training.hyperparams import load_config

    eval_seed_start = HPO_EVAL_SEED_START if eval_seed_start is None else eval_seed_start
    n_eval_episodes = HPO_N_EVAL_EPISODES if n_eval_episodes is None else n_eval_episodes
    n_seeds_per_trial = HPO_N_SEEDS_PER_TRIAL if n_seeds_per_trial is None else n_seeds_per_trial
    objective_name = HPO_OBJECTIVE if objective_name is None else objective_name
    risk_lambda = HPO_RISK_LAMBDA if risk_lambda is None else risk_lambda
    config = load_config(config_path)
    params = suggest_sac_params(
        trial,
        search_risk_aversion=search_risk_aversion,
        search_observation_mode=search_observation_mode,
    )
    net_arch = _net_arch_from_str(params.pop("net_arch_str"))

    environment = config.environment
    eval_environment = config.eval_environment
    if "risk_aversion" in params:
        environment = replace(environment, risk_aversion=params["risk_aversion"])
        eval_environment = replace(eval_environment, risk_aversion=params["risk_aversion"])
    if "observation_mode" in params:
        environment = replace(environment, observation_mode=params["observation_mode"])
        eval_environment = replace(eval_environment, observation_mode=params["observation_mode"])

    trial_sac = replace(
        config.sac,
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        tau=params["tau"],
        gamma=params["gamma"],
        ent_coef=params["ent_coef"],
        policy_kwargs={"net_arch": net_arch},
        buffer_size=min(config.sac.buffer_size, 100_000),
    )
    all_seed_scores: list[float] = []
    eval_seeds = _eval_seed_block(n_eval_episodes, eval_seed_start, legacy_seed_behavior)
    for training_seed in _trial_seeds(
        config.run.seed, trial.number, n_seeds_per_trial, legacy_seed_behavior
    ):
        train_env = build_training_envs(
            env_config=environment,
            vn_config=config.vec_normalize,
            n_envs=HPO_N_ENVS,
            seed=training_seed,
            log_dir=None,
        )
        eval_env = build_eval_env(
            env_config=eval_environment,
            train_env=train_env,
            seed=training_seed,
        )
        try:
            model = SAC(
                policy=trial_sac.policy,
                env=train_env,
                learning_rate=trial_sac.learning_rate,
                buffer_size=trial_sac.buffer_size,
                learning_starts=trial_sac.learning_starts,
                batch_size=trial_sac.batch_size,
                tau=trial_sac.tau,
                gamma=trial_sac.gamma,
                train_freq=trial_sac.train_freq,
                gradient_steps=trial_sac.gradient_steps,
                ent_coef=trial_sac.ent_coef,
                policy_kwargs=trial_sac.policy_kwargs,
                seed=training_seed,
                verbose=0,
            )
            model.learn(total_timesteps=HPO_TIMESTEPS, reset_num_timesteps=True)
            sync_normalization_stats(source=train_env, target=eval_env)
            episode_rewards: list[float] = []
            for ep_index, ep_seed in enumerate(eval_seeds):
                obs = _reset_eval_env(eval_env, ep_seed)
                done = np.array([False])
                ep_reward = 0.0
                while not bool(done[0]):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _ = eval_env.step(action)
                    ep_reward += float(reward[0])
                episode_rewards.append(ep_reward)
                if ep_index == min(9, len(eval_seeds) - 1):
                    trial.report(
                        score_episode_rewards(episode_rewards, objective_name, risk_lambda),
                        step=0,
                    )
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            all_seed_scores.append(score_episode_rewards(episode_rewards, objective_name, risk_lambda))
        finally:
            train_env.close()
            eval_env.close()
    return float(np.mean(all_seed_scores))


def run_hpo(
    config_path: str = "configs/training.yaml",
    n_trials: int = 20,
    study_name: str = "sac_hedging_hpo",
    storage: str | None = None,
    eval_seed_start: int = HPO_EVAL_SEED_START,
    n_eval_episodes: int = HPO_N_EVAL_EPISODES,
    n_seeds_per_trial: int = HPO_N_SEEDS_PER_TRIAL,
    objective_name: str = HPO_OBJECTIVE,
    risk_lambda: float = HPO_RISK_LAMBDA,
    search_risk_aversion: bool = False,
    search_observation_mode: bool = False,
    legacy_seed_behavior: bool = False,
) -> optuna.Study:
    """
    Run an Optuna hyperparameter search.

    Args:
        config_path: Path to the base training config YAML.
        n_trials: Total number of trials to run.
        study_name: Name for the Optuna study.
        storage: Optuna storage URL for persistence, e.g. ``sqlite:///hpo.db``.

    Returns:
        Completed Optuna study with all trial results.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=5),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    _write_study_metadata(
        storage,
        study_name=study_name,
        config_path=config_path,
        eval_seed_start=eval_seed_start,
        n_eval_episodes=n_eval_episodes,
        n_seeds_per_trial=n_seeds_per_trial,
        objective_name=objective_name,
        risk_lambda=risk_lambda,
        search_risk_aversion=search_risk_aversion,
        search_observation_mode=search_observation_mode,
        legacy_seed_behavior=legacy_seed_behavior,
        study=study,
    )

    logger.info("Starting HPO: %s trials, study=%s", n_trials, study_name)

    try:
        study.optimize(
            func=lambda trial: objective(
                trial,
                config_path,
                eval_seed_start=eval_seed_start,
                n_eval_episodes=n_eval_episodes,
                n_seeds_per_trial=n_seeds_per_trial,
                objective_name=objective_name,
                risk_lambda=risk_lambda,
                search_risk_aversion=search_risk_aversion,
                search_observation_mode=search_observation_mode,
                legacy_seed_behavior=legacy_seed_behavior,
            ),
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )
    finally:
        _write_study_metadata(
            storage,
            study_name=study_name,
            config_path=config_path,
            eval_seed_start=eval_seed_start,
            n_eval_episodes=n_eval_episodes,
            n_seeds_per_trial=n_seeds_per_trial,
            objective_name=objective_name,
            risk_lambda=risk_lambda,
            search_risk_aversion=search_risk_aversion,
            search_observation_mode=search_observation_mode,
            legacy_seed_behavior=legacy_seed_behavior,
            study=study,
        )

    print(f"\n{'=' * 50}")
    print(f"HPO COMPLETE — {len(study.trials)} trials")
    print(f"{'=' * 50}")
    print(f"Best trial:  #{study.best_trial.number}")
    print(f"Best reward: {study.best_value:.4f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key:<20} {value}")
    print(f"{'=' * 50}\n")

    return study


def main() -> None:
    """Parse CLI arguments and run hyperparameter search."""
    parser = argparse.ArgumentParser(description="Optuna HPO for RL hedging agent")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--study", default="sac_hedging_hpo")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL, e.g. sqlite:///hpo.db",
    )
    parser.add_argument("--eval-seed-start", type=int, default=HPO_EVAL_SEED_START)
    parser.add_argument("--eval-episodes", type=int, default=HPO_N_EVAL_EPISODES)
    parser.add_argument("--seeds-per-trial", type=int, default=HPO_N_SEEDS_PER_TRIAL)
    parser.add_argument(
        "--objective",
        choices=["mean_reward", "mean_minus_lambda_std", "cvar"],
        default=HPO_OBJECTIVE,
    )
    parser.add_argument("--risk-lambda", type=float, default=HPO_RISK_LAMBDA)
    parser.add_argument("--search-risk-aversion", action="store_true")
    parser.add_argument("--search-observation-mode", action="store_true")
    parser.add_argument(
        "--legacy-seed-behavior",
        action="store_true",
        help="Reproduce the old seed behavior: training seed=config.run.seed and eval seeds 0..N-1",
    )
    args = parser.parse_args()

    run_hpo(
        config_path=args.config,
        n_trials=args.trials,
        study_name=args.study,
        storage=args.storage,
        eval_seed_start=args.eval_seed_start,
        n_eval_episodes=args.eval_episodes,
        n_seeds_per_trial=args.seeds_per_trial,
        objective_name=args.objective,
        risk_lambda=args.risk_lambda,
        search_risk_aversion=args.search_risk_aversion,
        search_observation_mode=args.search_observation_mode,
        legacy_seed_behavior=args.legacy_seed_behavior,
    )


if __name__ == "__main__":
    main()
