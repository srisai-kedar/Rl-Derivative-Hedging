"""Tests for Black-Scholes pricing and analytical Greeks."""

import numpy as np
import pytest

from src.finance.black_scholes import (
    call_delta,
    call_price,
    gamma,
    put_delta,
    put_price,
    vega,
)
from src.finance.greeks import numerical_delta, numerical_gamma


PUT_CALL_PARITY_CASES = [
    (100, 100, 1.0, 0.05, 0.20),
    (110, 100, 1.0, 0.05, 0.20),
    (90, 100, 1.0, 0.05, 0.20),
    (100, 100, 0.1, 0.05, 0.30),
    (100, 100, 2.0, 0.02, 0.15),
]


@pytest.mark.parametrize("S,K,T,r,sigma", PUT_CALL_PARITY_CASES)
def test_put_call_parity(S, K, T, r, sigma):
    call = call_price(S, K, r, sigma, T)
    put = put_price(S, K, r, sigma, T)
    parity = S - K * np.exp(-r * T)
    np.testing.assert_allclose(call - put, parity, rtol=1e-5)


def test_call_delta_bounds(bs_params):
    S_values = np.arange(60, 141, 10)
    deltas = call_delta(
        S_values,
        bs_params["K"],
        bs_params["r"],
        bs_params["sigma"],
        bs_params["T"],
    )
    assert np.all(deltas >= 0.0)
    assert np.all(deltas <= 1.0)


def test_put_delta_bounds(bs_params):
    S_values = np.arange(60, 141, 10)
    deltas = put_delta(
        S_values,
        bs_params["K"],
        bs_params["r"],
        bs_params["sigma"],
        bs_params["T"],
    )
    assert np.all(deltas >= -1.0)
    assert np.all(deltas <= 0.0)


@pytest.mark.parametrize("S", [90, 100, 110])
def test_call_intrinsic_at_expiry(S, bs_params):
    price = call_price(S, bs_params["K"], bs_params["r"], bs_params["sigma"], 0.0)
    expected = max(S - bs_params["K"], 0.0)
    np.testing.assert_allclose(price, expected, rtol=1e-5)


@pytest.mark.parametrize("S", [90, 100, 110])
def test_put_intrinsic_at_expiry(S, bs_params):
    price = put_price(S, bs_params["K"], bs_params["r"], bs_params["sigma"], 0.0)
    expected = max(bs_params["K"] - S, 0.0)
    np.testing.assert_allclose(price, expected, rtol=1e-5)


def test_gamma_positive(bs_params):
    S_values = np.linspace(80, 120, 20)
    gammas = gamma(
        S_values,
        bs_params["K"],
        bs_params["r"],
        bs_params["sigma"],
        bs_params["T"],
    )
    assert np.all(gammas >= 0.0)


def test_vega_positive(bs_params):
    S_values = np.linspace(80, 120, 20)
    vegas = vega(
        S_values,
        bs_params["K"],
        bs_params["r"],
        bs_params["sigma"],
        bs_params["T"],
    )
    assert np.all(vegas >= 0.0)


def test_call_delta_vs_numerical(bs_params):
    analytical = call_delta(
        bs_params["S"],
        bs_params["K"],
        bs_params["r"],
        bs_params["sigma"],
        bs_params["T"],
    )
    numerical = numerical_delta(
        call_price,
        bs_params["S"],
        bs_params["K"],
        bs_params["r"],
        bs_params["sigma"],
        bs_params["T"],
    )
    np.testing.assert_allclose(analytical, numerical, rtol=1e-4)


def test_gamma_vs_numerical(bs_params):
    analytical = gamma(
        bs_params["S"],
        bs_params["K"],
        bs_params["r"],
        bs_params["sigma"],
        bs_params["T"],
    )
    numerical = numerical_gamma(
        call_price,
        bs_params["S"],
        bs_params["K"],
        bs_params["r"],
        bs_params["sigma"],
        bs_params["T"],
    )
    np.testing.assert_allclose(analytical, numerical, rtol=1e-4)


def test_numpy_array_inputs(bs_params):
    S_values = np.linspace(80, 120, 50)
    prices = call_price(
        S_values,
        bs_params["K"],
        bs_params["r"],
        bs_params["sigma"],
        bs_params["T"],
    )
    assert prices.shape == (50,)
    assert np.all(prices >= 0.0)


def test_call_price_increases_with_sigma(bs_params):
    sigmas = np.linspace(0.05, 0.50, 20)
    prices = call_price(
        bs_params["S"],
        bs_params["K"],
        bs_params["r"],
        sigmas,
        bs_params["T"],
    )
    assert np.all(np.diff(prices) > 0.0)


def test_deep_itm_delta_approaches_one(bs_params):
    delta = call_delta(200, bs_params["K"], bs_params["r"], bs_params["sigma"], 1.0)
    assert float(delta) > 0.99


def test_deep_otm_delta_approaches_zero(bs_params):
    delta = call_delta(50, bs_params["K"], bs_params["r"], bs_params["sigma"], 1.0)
    assert float(delta) < 0.01
