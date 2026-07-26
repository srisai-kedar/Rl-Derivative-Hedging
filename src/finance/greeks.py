"""Numerical Greeks via central finite differences."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np

PriceFn = Callable[..., np.ndarray]
Numeric: TypeAlias = float | np.ndarray


def numerical_delta(
    price_fn: PriceFn,
    S: Numeric,
    K: Numeric,
    r: Numeric,
    sigma: Numeric,
    T: Numeric,
    dS: float = 0.01,
) -> np.ndarray:
    """Central finite difference delta: (f(S+dS) - f(S-dS)) / (2*dS)."""
    price_up = price_fn(S + dS, K, r, sigma, T)
    price_down = price_fn(S - dS, K, r, sigma, T)
    return (price_up - price_down) / (2.0 * dS)


def numerical_gamma(
    price_fn: PriceFn,
    S: Numeric,
    K: Numeric,
    r: Numeric,
    sigma: Numeric,
    T: Numeric,
    dS: float = 0.01,
) -> np.ndarray:
    """Central finite difference gamma."""
    price_up = price_fn(S + dS, K, r, sigma, T)
    price_center = price_fn(S, K, r, sigma, T)
    price_down = price_fn(S - dS, K, r, sigma, T)
    return (price_up - 2.0 * price_center + price_down) / (dS**2)


def numerical_vega(
    price_fn: PriceFn,
    S: Numeric,
    K: Numeric,
    r: Numeric,
    sigma: Numeric,
    T: Numeric,
    dsigma: float = 0.001,
) -> np.ndarray:
    """Central finite difference vega."""
    price_up = price_fn(S, K, r, sigma + dsigma, T)
    price_down = price_fn(S, K, r, sigma - dsigma, T)
    return (price_up - price_down) / (2.0 * dsigma)
