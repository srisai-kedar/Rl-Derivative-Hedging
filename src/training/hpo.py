"""
Hyperparameter optimisation for the RL hedging agent.

Uses Optuna with TPE sampler and Median pruner. Run with:
    python -m src.training.hpo --trials 20 --config configs/training.yaml

Each trial trains SAC for HPO_TIMESTEPS steps and evaluates on 50 episodes.
Total wall-clock time: approximately 10–30 minutes for 20 trials on CPU,
depending on n_envs and hardware.

Results are printed to stdout and optionally saved to an SQLite database
for resumption. To view results interactively:
    optuna-dashboard sqlite:///hpo.db
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)

HPO_TIMESTEPS: int = 300_000
HPO_N_EVAL_EPISODES: int = 50
HPO_N_ENVS: int = 2


def suggest_sac_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    Define the Optuna search space for SAC hyperparameters.

    Returns a dict of parameters suitable for overriding SACConfig fields.
    The ``net_arch_str`` key uses a string encoding because Optuna's categorical
    sampler requires hashable values.
    """
    return {
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


def _net_arch_from_str(arch_str: str) -> list[int]:
    """Convert ``256-256`` to ``[256, 256]``."""
    return [int(x) for x in arch_str.split("-")]


def objective(trial: optuna.Trial, config_path: str) -> float:
    """
    Optuna objective function for SAC hyperparameter search.

    Trains a fresh SAC model with trial-suggested parameters for
    HPO_TIMESTEPS steps, then evaluates on HPO_N_EVAL_EPISODES episodes.

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

    config = load_config(config_path)
    params = suggest_sac_params(trial)
    net_arch = _net_arch_from_str(params.pop("net_arch_str"))

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

    train_env = build_training_envs(
        env_config=config.environment,
        vn_config=config.vec_normalize,
        n_envs=HPO_N_ENVS,
        seed=config.run.seed,
        log_dir=None,
    )
    eval_env = build_eval_env(
        env_config=config.eval_environment,
        train_env=train_env,
        seed=config.run.seed,
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
            seed=config.run.seed,
            verbose=0,
        )

        model.learn(
            total_timesteps=HPO_TIMESTEPS,
            reset_num_timesteps=True,
        )

        sync_normalization_stats(source=train_env, target=eval_env)
        episode_rewards: list[float] = []

        for ep_seed in range(HPO_N_EVAL_EPISODES):
            obs = eval_env.reset()
            done = np.array([False])
            ep_reward = 0.0
            while not bool(done[0]):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                ep_reward += float(reward[0])
            episode_rewards.append(ep_reward)

            if ep_seed == 9:
                trial.report(float(np.mean(episode_rewards)), step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return float(np.mean(episode_rewards))
    finally:
        train_env.close()
        eval_env.close()


def run_hpo(
    config_path: str = "configs/training.yaml",
    n_trials: int = 20,
    study_name: str = "sac_hedging_hpo",
    storage: str | None = None,
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

    logger.info("Starting HPO: %s trials, study=%s", n_trials, study_name)

    study.optimize(
        func=lambda trial: objective(trial, config_path),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
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
    args = parser.parse_args()

    run_hpo(
        config_path=args.config,
        n_trials=args.trials,
        study_name=args.study,
        storage=args.storage,
    )


if __name__ == "__main__":
    main()
