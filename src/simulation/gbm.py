"""Geometric Brownian Motion price path simulator."""

from __future__ import annotations

import numpy as np


class GBMSimulator:
    """
    Geometric Brownian Motion price path simulator.

    S_{t+dt} = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    where Z ~ N(0, 1)
    """

    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        dt: float,
        seed: int | None = None,
    ) -> None:
        """Initialise simulator with market parameters and optional seed."""
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.rng = np.random.default_rng(seed)

    def _log_increments(self, Z: np.ndarray) -> np.ndarray:
        """Compute log-price increments from standard normal shocks."""
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z
        return drift + diffusion

    def generate_path(self, n_steps: int) -> np.ndarray:
        """Generate a single price path of shape (n_steps + 1,)."""
        Z = self.rng.standard_normal(n_steps)
        log_increments = self._log_increments(Z)
        log_prices = np.log(self.S0) + np.cumsum(log_increments)
        prices = np.exp(log_prices)
        return np.concatenate(([self.S0], prices))

    def generate_paths(self, n_paths: int, n_steps: int) -> np.ndarray:
        """Generate multiple price paths of shape (n_paths, n_steps + 1)."""
        Z = self.rng.standard_normal((n_paths, n_steps))
        log_increments = self._log_increments(Z)
        log_prices = np.log(self.S0) + np.cumsum(log_increments, axis=1)
        prices = np.exp(log_prices)
        s0_column = np.full((n_paths, 1), self.S0)
        return np.concatenate((s0_column, prices), axis=1)

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset the random number generator for reproducible evaluation."""
        self.rng = np.random.default_rng(seed)
