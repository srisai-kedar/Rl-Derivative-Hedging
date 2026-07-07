"""Transaction cost models for hedging rebalances."""

from __future__ import annotations

import numpy as np


def proportional_cost(
    delta_change: np.ndarray,
    S: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Cost = kappa * |delta_change| * S."""
    return kappa * np.abs(delta_change) * S


def fixed_plus_proportional_cost(
    delta_change: np.ndarray,
    S: np.ndarray,
    kappa_fixed: float,
    kappa_prop: float,
) -> np.ndarray:
    """Cost = kappa_fixed + kappa_prop * |delta_change| * S when trade occurs."""
    trade = np.abs(delta_change) > 0
    cost = kappa_fixed + kappa_prop * np.abs(delta_change) * S
    return np.where(trade, cost, 0.0)
