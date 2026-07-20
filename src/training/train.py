"""Main SAC training entry point for the hedging agent."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from src.training.callbacks import (
    HedgingCheckpointCallback,
    HedgingEvalCallback,
    HedgingMetricsCallback,
)
from src.training.env_factory import build_eval_env, build_training_envs
from src.training.hyperparams import TrainingConfig, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def setup_output_dirs(config: TrainingConfig) -> tuple[str, str]:
    """Create timestamped checkpoint and result directories for a training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.run.name}_{timestamp}"

    checkpoint_dir = os.path.join(config.checkpoints_dir, run_name)
    results_dir = os.path.join(config.results_dir, run_name)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Run: %s", run_name)
    logger.info("Checkpoints: %s", checkpoint_dir)
    logger.info("Results: %s", results_dir)

    return checkpoint_dir, results_dir


def build_model(
    config: TrainingConfig,
    train_env: VecNormalize,
    results_dir: str,
) -> SAC:
    """Initialize SAC from the YAML-backed training config."""
    sac_cfg = config.sac

    model = SAC(
        policy=sac_cfg.policy,
        env=train_env,
        learning_rate=sac_cfg.learning_rate,
        buffer_size=sac_cfg.buffer_size,
        learning_starts=sac_cfg.learning_starts,
        batch_size=sac_cfg.batch_size,
        tau=sac_cfg.tau,
        gamma=sac_cfg.gamma,
        train_freq=sac_cfg.train_freq,
        gradient_steps=sac_cfg.gradient_steps,
        ent_coef=sac_cfg.ent_coef,
        policy_kwargs=sac_cfg.policy_kwargs,
        tensorboard_log=os.path.join(results_dir, "tb_logs"),
        seed=config.run.seed,
        verbose=0,
    )

    logger.info(
        "SAC model initialized | policy=%s | net_arch=%s | lr=%s",
        sac_cfg.policy,
        sac_cfg.policy_kwargs.get("net_arch"),
        sac_cfg.learning_rate,
    )
    return model


def build_callbacks(
    config: TrainingConfig,
    train_env: VecNormalize,
    eval_env: VecNormalize,
    checkpoint_dir: str,
) -> CallbackList:
    """Assemble Phase 3 callbacks in the required order."""
    cb_cfg = config.callbacks

    metrics_cb = HedgingMetricsCallback(window=100, verbose=0)
    checkpoint_cb = HedgingCheckpointCallback(
        save_freq=cb_cfg.save_freq,
        save_path=checkpoint_dir,
        train_env=train_env,
        verbose=cb_cfg.verbose,
    )
    eval_cb = HedgingEvalCallback(
        train_env=train_env,
        eval_env=eval_env,
        eval_freq=cb_cfg.eval_freq,
        n_eval_episodes=cb_cfg.n_eval_episodes,
        save_path=checkpoint_dir,
        verbose=cb_cfg.verbose,
    )

    return CallbackList([metrics_cb, checkpoint_cb, eval_cb])


def train(config_path: str = "configs/training.yaml") -> None:
    """
    Run SAC training and always save final model/VecNormalize stats as a pair.

    This function is the importable entry point used by tests and the CLI.
    """
    config = load_config(config_path)
    np.random.seed(config.run.seed)

    train_env: VecNormalize | None = None
    eval_env: VecNormalize | None = None
    model: SAC | None = None

    try:
        checkpoint_dir, results_dir = setup_output_dirs(config)

        train_env = build_training_envs(
            env_config=config.environment,
            vn_config=config.vec_normalize,
            n_envs=config.run.n_envs,
            seed=config.run.seed,
            log_dir=os.path.join(results_dir, "monitor"),
        )
        eval_env = build_eval_env(
            env_config=config.eval_environment,
            train_env=train_env,
            seed=config.run.seed,
        )

        model = build_model(config, train_env, results_dir)
        callbacks = build_callbacks(config, train_env, eval_env, checkpoint_dir)

        logger.info(
            "Training started | total_timesteps=%s | n_envs=%s",
            f"{config.run.total_timesteps:,}",
            config.run.n_envs,
        )

        try:
            model.learn(
                total_timesteps=config.run.total_timesteps,
                callback=callbacks,
                tb_log_name=config.tb_log_name,
                reset_num_timesteps=True,
                progress_bar=config.run.progress_bar,
            )
        finally:
            final_model_path = os.path.join(checkpoint_dir, "final_model")
            final_vecnorm_path = os.path.join(checkpoint_dir, "final_vecnorm.pkl")
            model.save(final_model_path)
            train_env.save(final_vecnorm_path)
            logger.info("Final model and VecNormalize saved to %s", checkpoint_dir)
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()


def main() -> None:
    """Parse CLI arguments and start training."""
    parser = argparse.ArgumentParser(description="Train RL hedging agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()
    train(config_path=args.config)


if __name__ == "__main__":
    main()
