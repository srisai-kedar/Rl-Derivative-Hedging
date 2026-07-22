import numpy as np
import pytest


@pytest.fixture
def bs_params():
    """Standard ATM option parameters for testing."""
    return {"S": 100.0, "K": 100.0, "r": 0.05, "sigma": 0.20, "T": 1.0}


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def gbm_simulator():
    from src.simulation.gbm import GBMSimulator

    return GBMSimulator(S0=100.0, mu=0.05, sigma=0.20, dt=1 / 252, seed=42)


@pytest.fixture
def env_config():
    """Standard environment config for testing."""
    return {
        "S0": 100.0,
        "K": 100.0,
        "T": 30,
        "r": 0.05,
        "sigma": 0.20,
        "dt": 1 / 252,
        "kappa": 0.001,
        "randomise_params": False,
        "seed": 0,
    }


@pytest.fixture
def env(env_config):
    from src.envs.hedging_env import HedgingEnv

    return HedgingEnv(**env_config)


@pytest.fixture
def zero_cost_env(env_config):
    from src.envs.hedging_env import HedgingEnv

    cfg = {**env_config, "kappa": 0.0}
    return HedgingEnv(**cfg)


@pytest.fixture
def fast_training_config(tmp_path):
    """
    Minimal training config for fast smoke tests.

    Writes a temporary YAML file and returns its path.
    """
    import yaml

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


@pytest.fixture
def loaded_config(fast_training_config):
    from src.training.hyperparams import load_config

    return load_config(fast_training_config)


@pytest.fixture
def training_envs(loaded_config):
    from src.training.env_factory import build_training_envs

    env = build_training_envs(
        env_config=loaded_config.environment,
        vn_config=loaded_config.vec_normalize,
        n_envs=loaded_config.run.n_envs,
        seed=loaded_config.run.seed,
    )
    yield env
    env.close()


N_EVAL_EPISODES = 50


@pytest.fixture
def n_eval_episodes():
    return N_EVAL_EPISODES


@pytest.fixture
def fast_eval_config():
    from src.training.hyperparams import EnvironmentConfig

    return EnvironmentConfig(
        S0=100.0,
        K=100.0,
        T=30,
        r=0.05,
        sigma=0.20,
        dt=1 / 252,
        kappa=0.001,
        randomise_params=False,
    )


@pytest.fixture
def backtest_results_fixture(fast_eval_config):
    from src.evaluation.backtest import BSDeltaPolicy, ZeroHedgePolicy, run_backtest

    return run_backtest(
        policies=[BSDeltaPolicy(), ZeroHedgePolicy()],
        env_config=fast_eval_config,
        n_episodes=N_EVAL_EPISODES,
        show_progress=False,
    )


@pytest.fixture
def results_fixture(backtest_results_fixture):
    return backtest_results_fixture
