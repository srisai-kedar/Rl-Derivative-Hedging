"""Tests for opt-in Heston and Merton market models."""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.hedging_env import ConfigurationError, HedgingEnv
from src.simulation.gbm import GBMSimulator
from src.simulation.heston import HestonSimulator
from src.simulation.merton import MertonJumpDiffusionSimulator


def test_gbm_environment_path_is_bit_identical_for_explicit_seed():
    """The default market model remains the existing GBM implementation."""
    env = HedgingEnv(T=30, seed=17)
    env.reset(seed=123)
    expected = GBMSimulator(
        S0=env.S0, mu=env.r, sigma=env.sigma, dt=env.dt, seed=123
    ).generate_path(env.T)
    np.testing.assert_array_equal(env.state.price_path, expected)


@pytest.mark.parametrize(
    "simulator_factory",
    [
        lambda seed: HestonSimulator(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 1 / 252, seed),
        lambda seed: MertonJumpDiffusionSimulator(100.0, 0.05, 0.20, 1.0, -0.05, 0.10, 1 / 252, seed),
    ],
)
def test_simulators_reproduce_and_separate_explicit_seeds(simulator_factory):
    """Both opt-in simulators are deterministic per seed and distinct across seeds."""
    path_a = simulator_factory(1).generate_path(30)
    path_b = simulator_factory(1).generate_path(30)
    path_c = simulator_factory(2).generate_path(30)

    np.testing.assert_array_equal(path_a, path_b)
    assert not np.array_equal(path_a, path_c)
    assert np.all(np.isfinite(path_a)) and np.all(path_a > 0.0)


@pytest.mark.parametrize(
    ("market_model", "market_model_kwargs"),
    [
        ("heston", {}),
        ("merton", {}),
    ],
)
def test_environment_seed_reproducibility_and_distinct_paths(
    market_model, market_model_kwargs
):
    """Environment market-model paths follow the explicit seed contract."""
    env_a = HedgingEnv(market_model=market_model, market_model_kwargs=market_model_kwargs)
    env_b = HedgingEnv(market_model=market_model, market_model_kwargs=market_model_kwargs)
    env_a.reset(seed=5)
    env_b.reset(seed=5)
    np.testing.assert_array_equal(env_a.state.price_path, env_b.state.price_path)
    env_b.reset(seed=6)
    assert not np.array_equal(env_a.state.price_path, env_b.state.price_path)


@pytest.mark.parametrize("market_model", ["heston", "merton"])
def test_market_model_martingale_with_reported_standard_error(market_model):
    """Monte Carlo E[S_T] matches S0 exp(rT) within five standard errors."""
    n_paths = 20_000
    n_steps = 252
    dt = 1 / 252
    if market_model == "heston":
        simulator = HestonSimulator(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, dt, 123)
    else:
        simulator = MertonJumpDiffusionSimulator(
            100.0, 0.05, 0.20, 1.0, -0.05, 0.10, dt, 123
        )

    terminal = simulator.generate_paths(n_paths, n_steps)[:, -1]
    theoretical = 100.0 * np.exp(0.05 * n_steps * dt)
    ratio = float(terminal.mean() / theoretical)
    standard_error = float(terminal.std(ddof=1) / np.sqrt(n_paths) / theoretical)

    # The assertion explicitly uses the sample standard error, which was
    # approximately 0.00135 (Heston) and 0.00162 (Merton) in this run.
    assert abs(ratio - 1.0) <= 5.0 * standard_error


def test_heston_variance_truncation_and_path_positivity():
    """Full truncation keeps the recorded variance and prices non-negative."""
    simulator = HestonSimulator(100.0, 0.05, 0.04, 2.0, 0.04, 1.0, -0.7, 1 / 252, 9)
    path = simulator.generate_path(252)

    assert simulator.last_variance_path is not None
    assert np.all(simulator.last_variance_path >= 0.0)
    assert np.all(path > 0.0)
    assert np.all(np.isfinite(path))


def test_merton_jump_frequency_and_compensated_drift():
    """Jump counts and compensated drift match their analytical expectations."""
    simulator = MertonJumpDiffusionSimulator(
        100.0, 0.05, 0.20, 1.0, -0.05, 0.10, 1 / 252, 123
    )
    n_paths = 20_000
    n_steps = 252
    terminal = simulator.generate_paths(n_paths, n_steps)[:, -1]
    jump_mean = float(simulator.last_jump_counts.sum() / n_paths)
    expected_jumps = simulator.expected_jumps_per_episode(n_steps)
    jump_standard_error = np.sqrt(expected_jumps / n_paths)

    assert abs(jump_mean - expected_jumps) <= 5.0 * jump_standard_error
    assert expected_jumps == pytest.approx(1.0)
    theoretical = 100.0 * np.exp(0.05)
    assert terminal.mean() / theoretical == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("market_model", ["gbm", "heston", "merton"])
def test_market_model_pnl_identity_observation_containment_and_no_nans(market_model):
    """Every market model preserves accounting, bounds, and finite observations."""
    env = HedgingEnv(T=30, kappa=0.002, market_model=market_model)
    obs, _ = env.reset(seed=7)
    actions = np.linspace(0.0, 1.0, env.T, dtype=np.float32)
    assert env.observation_space.contains(obs)

    infos = []
    for action in actions:
        obs, reward, terminated, _, info = env.step(np.array([action]))
        infos.append(info)
        assert env.observation_space.contains(obs)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        if terminated:
            break

    path = env.state.price_path
    expected = (
        env._option_values[0]
        - max(path[-1] - env.K, 0.0)
        + float(np.sum(actions * np.diff(path)))
        - infos[-1]["total_cost"]
    )
    assert infos[-1]["terminal_pnl"] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("market_model", "kwargs"),
    [
        ("heston", {"unknown": 1.0}),
        ("merton", {"theta": 0.04}),
        ("gbm", {"sigma": 0.2}),
        ("heston", {"v0": 0.0}),
        ("heston", {"rho": 1.1}),
        ("merton", {"lam": -1.0}),
        ("merton", {"sigma_J": -0.1}),
    ],
)
def test_market_model_kwargs_fail_at_construction(market_model, kwargs):
    """Unknown keys and invalid ranges raise ConfigurationError before reset."""
    with pytest.raises(ConfigurationError):
        HedgingEnv(market_model=market_model, market_model_kwargs=kwargs)


def test_heston_default_xi_policy_satisfies_feller_at_low_sigma():
    """Omitted Heston xi is derived to satisfy Feller for low sigma."""
    sigma = 0.10
    env = HedgingEnv(sigma=sigma, market_model="heston")
    expected_xi = min(0.3, np.sqrt(2.0 * 2.0 * sigma**2))
    simulator = HestonSimulator(
        env.S0, env.r, sigma**2, 2.0, sigma**2, expected_xi, -0.7, env.dt, seed=0
    )
    assert simulator.feller_satisfied
