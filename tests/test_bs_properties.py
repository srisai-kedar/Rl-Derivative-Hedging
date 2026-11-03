"""Property-based no-arbitrage tests for the Black-Scholes layer."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.finance.black_scholes import call_delta, call_price, put_delta, put_price

S = st.floats(20.0, 500.0)
K = st.floats(20.0, 500.0)
R = st.floats(0.0, 0.2)
SIG = st.floats(0.05, 1.0)
T = st.floats(1e-3, 3.0)
SETTINGS = settings(max_examples=150, deadline=None)


@SETTINGS
@given(S, K, R, SIG, T)
def test_put_call_parity(s, k, r, sig, t):
    lhs = call_price(s, k, r, sig, t) - put_price(s, k, r, sig, t)
    assert lhs == pytest.approx(s - k * np.exp(-r * t), abs=1e-7 * max(s, k))


@SETTINGS
@given(S, K, R, SIG, T)
def test_call_price_no_arbitrage_bounds(s, k, r, sig, t):
    c = float(call_price(s, k, r, sig, t))
    lower = max(s - k * np.exp(-r * t), 0.0)
    assert lower - 1e-8 * s <= c <= s + 1e-8 * s


@SETTINGS
@given(S, K, R, SIG, T)
def test_call_delta_bounds_and_put_call_delta_relation(s, k, r, sig, t):
    d = float(call_delta(s, k, r, sig, t))
    assert 0.0 <= d <= 1.0
    assert float(put_delta(s, k, r, sig, t)) == pytest.approx(d - 1.0, abs=1e-12)


@SETTINGS
@given(K, R, SIG, T, st.floats(20.0, 400.0), st.floats(0.01, 50.0))
def test_call_price_and_delta_monotone_in_spot(k, r, sig, t, s, bump):
    c1, c2 = float(call_price(s, k, r, sig, t)), float(call_price(s + bump, k, r, sig, t))
    d1, d2 = float(call_delta(s, k, r, sig, t)), float(call_delta(s + bump, k, r, sig, t))
    assert c2 >= c1 - 1e-9
    assert d2 >= d1 - 1e-9


@SETTINGS
@given(S, K, R, SIG, T)
def test_analytic_delta_matches_central_difference(s, k, r, sig, t):
    h = 1e-5 * s
    fd = (call_price(s + h, k, r, sig, t) - call_price(s - h, k, r, sig, t)) / (2 * h)
    assert float(call_delta(s, k, r, sig, t)) == pytest.approx(float(fd), abs=1e-5)
