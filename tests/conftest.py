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
