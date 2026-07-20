"""Episode state dataclass for hedging environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeState:
    """
    Holds all mutable state for a single hedging episode.

    Created fresh at each env.reset() call.
    Updated in-place at each env.step() call.
    Never shared between environments.
    """

    # --- Fixed for the episode (set at reset) ---
    price_path: np.ndarray  # Shape: (n_steps + 1,). Full path pre-generated.
    S0: float  # Initial stock price
    K: float  # Strike price
    r: float  # Risk-free rate (annualised)
    sigma: float  # Volatility (annualised)
    T_days: int  # Total steps in this episode
    dt: float  # Time per step (fraction of a year)

    # --- Mutable state (updated each step) ---
    step: int = 0
    hedge_pos: float = 0.0  # Fraction of delta currently held [0, 1]
    option_value: float = 0.0  # Current BS call value
    cumulative_pnl: float = 0.0
    cumulative_cost: float = 0.0

    def __post_init__(self) -> None:
        assert self.price_path.shape == (self.T_days + 1,), (
            f"price_path must have shape ({self.T_days + 1},), "
            f"got {self.price_path.shape}"
        )
        assert self.dt > 0, "dt must be positive"
        assert self.sigma > 0, "sigma must be positive"

    @property
    def S(self) -> float:
        """Current stock price."""
        return float(self.price_path[self.step])

    @property
    def time_remaining(self) -> float:
        """Time remaining as fraction of T, in [0, 1]."""
        return float(self.T_days - self.step) / float(self.T_days)

    @property
    def T_remaining_years(self) -> float:
        """Absolute time remaining in years."""
        return (self.T_days - self.step) * self.dt

    @property
    def is_terminal(self) -> bool:
        return self.step >= self.T_days
