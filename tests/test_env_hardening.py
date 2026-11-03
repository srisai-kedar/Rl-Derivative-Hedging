"""Property-based and contract tests for the hardened HedgingEnv."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.envs.hedging_env import ConfigurationError, HedgingEnv

PROFILE = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
actions = st.lists(st.floats(min_value=-0.5, max_value=1.5), min_size=30, max_size=30)
seeds = st.integers(min_value=0, max_value=10_000)


def _rollout(env: HedgingEnv, acts: list[float], seed: int):
    env.reset(seed=seed)
    infos, rewards = [], []
    for a in acts:
        _, r, term, _, info = env.step(np.array([a], dtype=np.float32))
        infos.append(info)
        rewards.append(r)
    assert term
    return infos, rewards


# ----------------------------- accounting identity ------------------------- #
@PROFILE
@given(acts=actions, seed=seeds)
def test_terminal_pnl_telescoping_identity(acts, seed):
    """P&L == C0 - payoff + sum(a_t dS_t) - sum(cost): the report's Eq. (7)."""
    env = HedgingEnv(kappa=0.002)
    infos, _ = _rollout(env, acts, seed)
    path = env.state.price_path
    a = np.clip(np.array(acts, dtype=np.float32).astype(float), 0, 1)
    c0 = env._option_values[0]
    payoff = max(path[-1] - env.K, 0.0)
    gains = float(np.sum(a * np.diff(path)))
    total_cost = infos[-1]["total_cost"]
    expected = c0 - payoff + gains - total_cost
    assert infos[-1]["terminal_pnl"] == pytest.approx(expected, abs=1e-9)


@PROFILE
@given(acts=actions, seed=seeds)
def test_liquidation_adds_exact_closing_cost(acts, seed):
    base, _ = _rollout(HedgingEnv(kappa=0.003), acts, seed)
    liq, _ = _rollout(HedgingEnv(kappa=0.003, liquidate_at_expiry=True), acts, seed)
    final_pos = base[-1]["hedge_pos"]
    s_t = base[-1]["S"]
    extra = 0.003 * final_pos * s_t
    assert liq[-1]["total_cost"] == pytest.approx(base[-1]["total_cost"] + extra)
    assert liq[-1]["terminal_pnl"] == pytest.approx(base[-1]["terminal_pnl"] - extra)


@PROFILE
@given(acts=actions, seed=seeds)
def test_financing_identity(acts, seed):
    """Financing = sum_t r*dt*a_t*S_t (funded at the risk-free rate)."""
    base, _ = _rollout(HedgingEnv(), acts, seed)
    env = HedgingEnv(include_financing=True)
    fin, _ = _rollout(env, acts, seed)
    target = np.array([i["hedge_pos"] for i in fin])
    s_before = env.state.price_path[:-1]  # price at which each position is held
    expected = float(np.sum(env.r * env.dt * target * s_before))
    assert sum(i["financing_cost"] for i in fin) == pytest.approx(expected)
    assert fin[-1]["terminal_pnl"] == pytest.approx(
        base[-1]["terminal_pnl"] - expected, abs=1e-9
    )


@PROFILE
@given(acts=actions, seed=seeds, lam=st.floats(min_value=0.0, max_value=5.0))
def test_risk_aversion_shapes_reward_but_not_pnl(acts, seed, lam):
    raw_infos, raw_r = _rollout(HedgingEnv(), acts, seed)
    sh_infos, sh_r = _rollout(HedgingEnv(risk_aversion=lam), acts, seed)
    for ri, si, rr, sr in zip(raw_infos, sh_infos, raw_r, sh_r, strict=True):
        assert si["step_pnl"] == pytest.approx(ri["step_pnl"])  # metrics untouched
        assert sr == pytest.approx(rr - lam * rr * rr)
    assert sh_infos[-1]["terminal_pnl"] == pytest.approx(raw_infos[-1]["terminal_pnl"])


@PROFILE
@given(acts=actions, seed=seeds)
def test_observations_always_inside_space(acts, seed):
    env = HedgingEnv(randomise_params=True)
    obs, _ = env.reset(seed=seed)
    assert env.observation_space.contains(obs)
    for a in acts:
        obs, r, *_ = env.step(np.array([a], dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)


# ----------------------------- seeding contract ---------------------------- #
def test_explicit_seed_reproduces_exactly():
    a, b = HedgingEnv(), HedgingEnv()
    a.reset(seed=7)
    b.reset(seed=7)
    np.testing.assert_array_equal(a.state.price_path, b.state.price_path)


def test_unseeded_resets_produce_fresh_paths():
    """Regression: constructor seed must not freeze every episode to one path."""
    env = HedgingEnv(seed=42)
    env.reset()
    p1 = env.state.price_path.copy()
    env.reset()
    p2 = env.state.price_path.copy()
    assert not np.allclose(p1, p2)


def test_constructor_seed_gives_reproducible_stream():
    e1, e2 = HedgingEnv(seed=42), HedgingEnv(seed=42)
    for _ in range(3):
        e1.reset()
        e2.reset()
        np.testing.assert_array_equal(e1.state.price_path, e2.state.price_path)


# ----------------------------- error handling ------------------------------ #
def test_step_before_reset_raises():
    with pytest.raises(RuntimeError, match="reset"):
        HedgingEnv().step(np.array([0.5], dtype=np.float32))


def test_step_after_termination_raises():
    env = HedgingEnv(T=2)
    env.reset(seed=0)
    env.step(np.array([0.5]))
    env.step(np.array([0.5]))
    with pytest.raises(RuntimeError, match="finished"):
        env.step(np.array([0.5]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_action_rejected(bad):
    env = HedgingEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="finite"):
        env.step(np.array([bad], dtype=np.float32))


def test_wrong_action_shape_rejected():
    env = HedgingEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="scalar"):
        env.step(np.array([0.1, 0.2]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"S0": 0.0},
        {"K": -1.0},
        {"T": 0},
        {"T": 2.5},
        {"dt": 0.0},
        {"kappa": -0.1},
        {"sigma": 0.0},
        {"sigma": float("nan")},
        {"r": 0.5},
        {"K": 300.0},
        {"risk_aversion": -1.0},
        {"sigma_range": (0.4, 0.1)},
        {"sigma_range": (0.0, 0.4)},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ConfigurationError):
        HedgingEnv(**kwargs)
