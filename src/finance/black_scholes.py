"""Black-Scholes European option pricing and analytical Greeks."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _broadcast_inputs(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    T: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert inputs to float arrays with common broadcasting shape."""
    arrays = [np.asarray(x, dtype=float) for x in (S, K, r, sigma, T)]
    return tuple(np.broadcast_arrays(*arrays))  # type: ignore[return-value]


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal probability density function."""
    return np.exp(-0.5 * x**2) / _SQRT_2PI


def _d1(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    T: float | np.ndarray,
) -> np.ndarray:
    """Compute d1 term of Black-Scholes formula."""
    S, K, r, sigma, T = _broadcast_inputs(S, K, r, sigma, T)

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    at_expiry = T == 0
    expiry_d1 = np.where(S > K, np.inf, np.where(S < K, -np.inf, 0.0))
    return np.where(at_expiry, expiry_d1, d1)


def _d2(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    T: float | np.ndarray,
) -> np.ndarray:
    """Compute d2 = d1 - sigma * sqrt(T)."""
    S, K, r, sigma, T = _broadcast_inputs(S, K, r, sigma, T)
    d1 = _d1(S, K, r, sigma, T)

    with np.errstate(invalid="ignore"):
        d2 = d1 - sigma * np.sqrt(T)

    at_expiry = T == 0
    expiry_d2 = np.where(S > K, np.inf, np.where(S < K, -np.inf, 0.0))
    return np.where(at_expiry, expiry_d2, d2)


def call_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    T: float | np.ndarray,
) -> np.ndarray:
    """Black-Scholes European call option price."""
    S, K, r, sigma, T = _broadcast_inputs(S, K, r, sigma, T)
    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(S, K, r, sigma, T)

    price = S * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    intrinsic = np.maximum(S - K, 0.0)
    return np.where(T == 0, intrinsic, price)


def put_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    T: float | np.ndarray,
) -> np.ndarray:
    """Black-Scholes European put option price."""
    S, K, r, sigma, T = _broadcast_inputs(S, K, r, sigma, T)
    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(S, K, r, sigma, T)

    price = K * np.exp(-r * T) * ndtr(-d2) - S * ndtr(-d1)
    intrinsic = np.maximum(K - S, 0.0)
    return np.where(T == 0, intrinsic, price)


def call_delta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    T: float | np.ndarray,
) -> np.ndarray:
    """Call delta = N(d1). Range: [0, 1]."""
    S, K, r, sigma, T = _broadcast_inputs(S, K, r, sigma, T)
    d1 = _d1(S, K, r, sigma, T)
    delta = ndtr(d1)
    intrinsic_delta = np.where(S > K, 1.0, np.where(S < K, 0.0, 0.0))
    return np.where(T == 0, intrinsic_delta, delta)


def put_delta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    T: float | np.ndarray,
) -> np.ndarray:
    """Put delta = N(d1) - 1. Range: [-1, 0]."""
    S, K, r, sigma, T = _broadcast_inputs(S, K, r, sigma, T)
    d1 = _d1(S, K, r, sigma, T)
    delta = ndtr(d1) - 1.0
    intrinsic_delta = np.where(S > K, 0.0, np.where(S < K, -1.0, 0.0))
    return np.where(T == 0, intrinsic_delta, delta)


def gamma(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    T: float | np.ndarray,
) -> np.ndarray:
    """Gamma = N'(d1) / (S * sigma * sqrt(T)). Same for call and put."""
    S, K, r, sigma, T = _broadcast_inputs(S, K, r, sigma, T)
    d1 = _d1(S, K, r, sigma, T)

    with np.errstate(divide="ignore", invalid="ignore"):
        g = _norm_pdf(d1) / (S * sigma * np.sqrt(T))

    return np.where(T == 0, 0.0, g)


def vega(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    T: float | np.ndarray,
) -> np.ndarray:
    """Vega = S * N'(d1) * sqrt(T), per unit of vol."""
    S, K, r, sigma, T = _broadcast_inputs(S, K, r, sigma, T)
    d1 = _d1(S, K, r, sigma, T)

    with np.errstate(invalid="ignore"):
        v = S * _norm_pdf(d1) * np.sqrt(T)

    return np.where(T == 0, 0.0, v)


def theta_call(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    T: float | np.ndarray,
) -> np.ndarray:
    """Call theta (time decay per unit time, annualised)."""
    S, K, r, sigma, T = _broadcast_inputs(S, K, r, sigma, T)
    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(S, K, r, sigma, T)

    with np.errstate(divide="ignore", invalid="ignore"):
        theta = (
            -(S * _norm_pdf(d1) * sigma) / (2.0 * np.sqrt(T))
            - r * K * np.exp(-r * T) * ndtr(d2)
        )

    return np.where(T == 0, 0.0, theta)
