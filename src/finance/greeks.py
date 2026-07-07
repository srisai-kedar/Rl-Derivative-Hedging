"""Numerical Greeks via central finite differences."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numerical_delta(
    price_fn: Callable[..., np.ndarray],
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    T: float | np.ndarray,
    dS: float = 0.01,
) -> np.ndarray:
    """Central finite difference delta: (f(S+dS) - f(S-dS)) / (2*dS)."""
    price_up = price_fn(S + dS, K, r, sigma, T)
    price_down = price_fn(S - dS, K, r, sigma, T)
    return (price_up - price_down) / (2.0 * dS)


def numerical_gamma(
    price_fn: Callable[..., np.ndarray],
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    T: float | np.ndarray,
    dS: float = 0.01,
) -> np.ndarray:
    """Central finite difference gamma."""
    price_up = price_fn(S + dS, K, r, sigma, T)
    price_center = price_fn(S, K, r, sigma, T)
    price_down = price_fn(S - dS, K, r, sigma, T)
    return (price_up - 2.0 * price_center + price_down) / (dS**2)


def numerical_vega(
    price_fn: Callable[..., np.ndarray],
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    T: float | np.ndarray,
    dsigma: float = 0.001,
) -> np.ndarray:
    """Central finite difference vega."""
    price_up = price_fn(S, K, r, sigma + dsigma, T)
    price_down = price_fn(S, K, r, sigma - dsigma, T)
    return (price_up - price_down) / (2.0 * dsigma)
