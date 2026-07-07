"""Tests for GBM price path simulation."""

import numpy as np
from scipy import stats

from src.simulation.gbm import GBMSimulator


def test_path_shape(gbm_simulator):
    path = gbm_simulator.generate_path(30)
    paths = gbm_simulator.generate_paths(100, 30)
    assert path.shape == (31,)
    assert paths.shape == (100, 31)


def test_path_starts_at_S0(gbm_simulator):
    path = gbm_simulator.generate_path(30)
    paths = gbm_simulator.generate_paths(10, 30)
    assert path[0] == gbm_simulator.S0
    assert np.all(paths[:, 0] == gbm_simulator.S0)


def test_paths_positive(gbm_simulator):
    paths = gbm_simulator.generate_paths(50, 30)
    assert np.all(paths > 0.0)


def test_terminal_price_log_normality():
    n_steps = 252
    dt = 1 / 252
    mu = 0.05
    sigma = 0.20
    S0 = 100.0
    n_paths = 10_000

    sim = GBMSimulator(S0=S0, mu=mu, sigma=sigma, dt=dt, seed=123)
    paths = sim.generate_paths(n_paths, n_steps)
    log_terminal = np.log(paths[:, -1])

    T = n_steps * dt
    mean = np.log(S0) + (mu - 0.5 * sigma**2) * T
    std = sigma * np.sqrt(T)
    _, p_value = stats.kstest(log_terminal, "norm", args=(mean, std))
    assert p_value > 0.05


def test_reproducibility():
    sim_a = GBMSimulator(S0=100.0, mu=0.05, sigma=0.20, dt=1 / 252, seed=42)
    sim_b = GBMSimulator(S0=100.0, mu=0.05, sigma=0.20, dt=1 / 252, seed=42)
    path_a = sim_a.generate_path(30)
    path_b = sim_b.generate_path(30)
    np.testing.assert_array_equal(path_a, path_b)


def test_different_seeds_differ():
    sim_a = GBMSimulator(S0=100.0, mu=0.05, sigma=0.20, dt=1 / 252, seed=1)
    sim_b = GBMSimulator(S0=100.0, mu=0.05, sigma=0.20, dt=1 / 252, seed=2)
    path_a = sim_a.generate_path(30)
    path_b = sim_b.generate_path(30)
    assert not np.allclose(path_a, path_b)


def test_vectorised_vs_loop():
    n_paths = 50
    n_steps = 30
    sim_vectorised = GBMSimulator(
        S0=100.0, mu=0.05, sigma=0.20, dt=1 / 252, seed=0
    )
    batch_paths = sim_vectorised.generate_paths(n_paths, n_steps)

    sim_loop = GBMSimulator(S0=100.0, mu=0.05, sigma=0.20, dt=1 / 252, seed=0)
    loop_paths = np.array([sim_loop.generate_path(n_steps) for _ in range(n_paths)])

    np.testing.assert_allclose(batch_paths, loop_paths, rtol=1e-12)
