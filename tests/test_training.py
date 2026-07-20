"""Smoke tests for the Phase 3 SAC training pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml


def _write_fast_training_config(tmp_path: Path) -> str:
    cfg = {
        "run": {
            "name": "test_run",
            "seed": 0,
            "total_timesteps": 2_000,
            "n_envs": 2,
            "progress_bar": False,
        },
        "environment": {
            "S0": 100.0,
            "K": 100.0,
            "T": 30,
            "r": 0.05,
            "sigma": 0.20,
            "dt": 0.003968,
            "kappa": 0.001,
            "randomise_params": False,
            "sigma_range": [0.10, 0.40],
        },
        "eval_environment": {
            "randomise_params": False,
            "sigma": 0.20,
        },
        "sac": {
            "policy": "MlpPolicy",
            "learning_rate": 3.0e-4,
            "buffer_size": 5_000,
            "learning_starts": 100,
            "batch_size": 64,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "policy_kwargs": {"net_arch": [64, 64]},
        },
        "vec_normalize": {
            "norm_obs": True,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
        },
        "callbacks": {
            "eval_freq": 1_000,
            "n_eval_episodes": 10,
            "save_freq": 1_000,
            "verbose": 0,
        },
        "paths": {
            "checkpoints_dir": str(tmp_path / "checkpoints"),
            "results_dir": str(tmp_path / "results"),
            "tb_log_name": "SAC",
        },
    }
    config_path = tmp_path / "training.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return str(config_path)


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """Run one short training job and reuse its artifacts across tests."""
    from src.training.train import train

    tmp_path = tmp_path_factory.mktemp("training_run")
    config_path = _write_fast_training_config(tmp_path)
    train(config_path=config_path)
    return tmp_path, config_path


def test_config_loads_from_yaml(fast_training_config):
    """Config file loads without error and fields are accessible."""
    from src.training.hyperparams import load_config

    config = load_config(fast_training_config)

    assert config.run.seed == 0
    assert config.run.n_envs == 2
    assert config.sac.policy == "MlpPolicy"
    assert config.environment.kappa == 0.001
    assert isinstance(config.environment.sigma_range, tuple)
    assert len(config.environment.sigma_range) == 2


def test_training_env_builds(training_envs):
    """VecNormalize wraps correctly and returns expected obs shape."""
    from stable_baselines3.common.vec_env import VecNormalize

    assert isinstance(training_envs, VecNormalize)
    obs = training_envs.reset()
    assert obs.shape == (2, 7), f"Expected (n_envs=2, obs_dim=7), got {obs.shape}"
    assert obs.dtype == np.float32


def test_eval_env_builds(loaded_config, training_envs):
    """Eval env builds with training=False and norm_reward=False."""
    from stable_baselines3.common.vec_env import VecNormalize

    from src.training.env_factory import build_eval_env

    eval_env = build_eval_env(
        env_config=loaded_config.eval_environment,
        train_env=training_envs,
        seed=loaded_config.run.seed,
    )
    assert isinstance(eval_env, VecNormalize)
    assert eval_env.training is False
    assert eval_env.norm_reward is False

    obs = eval_env.reset()
    assert obs.shape == (1, 7)
    eval_env.close()


def test_vecnorm_stats_sync(loaded_config, training_envs):
    """Syncing stats copies obs_rms from train to eval env correctly."""
    from src.training.env_factory import build_eval_env, sync_normalization_stats

    eval_env = build_eval_env(
        env_config=loaded_config.eval_environment,
        train_env=training_envs,
        seed=0,
    )

    training_envs.obs_rms.mean[:] = 999.0
    sync_normalization_stats(source=training_envs, target=eval_env)

    np.testing.assert_array_equal(
        eval_env.obs_rms.mean,
        training_envs.obs_rms.mean,
        err_msg="Stats sync did not copy obs_rms correctly",
    )
    eval_env.close()


def test_model_initialises(loaded_config, training_envs, tmp_path):
    """SAC model initialises without error with given config."""
    from src.training.train import build_model

    model = build_model(loaded_config, training_envs, str(tmp_path))
    assert model is not None
    assert model.policy is not None


def test_short_training_run(trained_run):
    """2000-step training run completes and saves a final model."""
    tmp_path, _ = trained_run
    found = list(tmp_path.rglob("final_model.zip"))
    assert len(found) == 1, "final_model.zip not saved after training"


def test_vecnorm_saved_with_model(trained_run):
    """Every model checkpoint must have its paired VecNormalize file."""
    tmp_path, _ = trained_run
    zip_files = list(tmp_path.rglob("*.zip"))

    assert zip_files, "No model checkpoints found"

    expected_pairs = {
        "final_model": "final_vecnorm.pkl",
        "best_model": "best_vecnorm.pkl",
    }

    for zip_path in zip_files:
        if zip_path.stem in expected_pairs:
            vecnorm_path = zip_path.with_name(expected_pairs[zip_path.stem])
        elif zip_path.stem.startswith("model_") and zip_path.stem.endswith("_steps"):
            step = zip_path.stem.removeprefix("model_")
            vecnorm_path = zip_path.with_name(f"vecnorm_{step}.pkl")
        else:
            raise AssertionError(f"Unexpected model checkpoint name: {zip_path.name}")

        assert vecnorm_path.exists(), (
            f"{zip_path.name} missing paired VecNormalize file {vecnorm_path.name}"
        )


def test_checkpoint_save_and_load(trained_run):
    """A saved model loaded with VecNormalize produces deterministic actions."""
    from stable_baselines3 import SAC

    from src.training.env_factory import load_env_for_inference
    from src.training.hyperparams import load_config

    tmp_path, config_path = trained_run
    config = load_config(config_path)

    checkpoint_dir = list(tmp_path.rglob("final_model.zip"))[0].parent
    model_path = str(checkpoint_dir / "final_model")
    vecnorm_path = str(checkpoint_dir / "final_vecnorm.pkl")

    assert os.path.exists(vecnorm_path), "final_vecnorm.pkl not found"

    inf_env = load_env_for_inference(
        env_config=config.eval_environment,
        vecnorm_path=vecnorm_path,
        seed=42,
    )
    model = SAC.load(model_path, env=inf_env)

    obs = inf_env.reset()
    action1, _ = model.predict(obs, deterministic=True)
    action2, _ = model.predict(obs, deterministic=True)

    np.testing.assert_array_equal(
        action1,
        action2,
        err_msg="Deterministic model produced different actions on same input",
    )
    inf_env.close()


def test_model_actions_in_valid_range(trained_run):
    """Model predictions after training must stay in action-space bounds."""
    from stable_baselines3 import SAC

    from src.training.env_factory import load_env_for_inference
    from src.training.hyperparams import load_config

    tmp_path, config_path = trained_run
    config = load_config(config_path)

    checkpoint_dir = list(tmp_path.rglob("final_model.zip"))[0].parent
    model_path = str(checkpoint_dir / "final_model")
    vecnorm_path = str(checkpoint_dir / "final_vecnorm.pkl")

    inf_env = load_env_for_inference(
        env_config=config.eval_environment,
        vecnorm_path=vecnorm_path,
        seed=0,
    )
    model = SAC.load(model_path, env=inf_env)
    obs = inf_env.reset()

    for _ in range(30):
        action, _ = model.predict(obs, deterministic=False)
        action_value = float(np.asarray(action).reshape(-1)[0])
        assert 0.0 <= action_value <= 1.0, (
            f"Action {action_value} is outside [0, 1]"
        )
        obs, _, done, _ = inf_env.step(action)
        if bool(done[0]):
            break

    inf_env.close()


def test_tensorboard_log_dir_created(trained_run):
    """TensorBoard log directory must exist after training."""
    tmp_path, _ = trained_run
    tb_dirs = list(tmp_path.rglob("tb_logs"))
    assert len(tb_dirs) >= 1, "TensorBoard log directory not created"


def test_no_nan_during_training(trained_run):
    """Verify no NaN values appear in model parameters after short training."""
    from stable_baselines3 import SAC
    import torch

    tmp_path, _ = trained_run
    checkpoint_dir = list(tmp_path.rglob("final_model.zip"))[0].parent
    model = SAC.load(str(checkpoint_dir / "final_model"))

    for name, param in model.policy.named_parameters():
        assert not torch.any(torch.isnan(param)), f"NaN found in parameter: {name}"
