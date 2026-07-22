import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_cost_efficiency,
    compute_cvar,
    compute_hedging_error,
    compute_improvement_over_baseline,
    compute_sharpe,
)


def test_cvar_known_value():
    pnl = pd.Series(np.random.default_rng(42).normal(size=50_000))
    assert compute_cvar(pnl, alpha=0.05) == pytest.approx(-2.063, abs=0.05)


def test_cvar_is_worse_than_mean():
    rng = np.random.default_rng(42)
    for _ in range(10):
        pnl = pd.Series(rng.normal(size=1_000))
        assert compute_cvar(pnl) <= pnl.mean() + 1e-10


def test_cvar_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_cvar(pd.Series([], dtype=float))


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_cvar_invalid_alpha_raises(alpha):
    with pytest.raises(ValueError):
        compute_cvar(pd.Series([1.0, 2.0]), alpha=alpha)


def test_hedging_error_constant_series():
    assert compute_hedging_error(pd.Series([1.5] * 100)) == pytest.approx(0.0)


def test_hedging_error_positive():
    assert compute_hedging_error(pd.Series(np.random.default_rng(42).normal(size=100))) >= 0.0


def test_sharpe_zero_std_returns_zero():
    assert compute_sharpe(pd.Series([2.0] * 100)) == 0.0


def test_sharpe_positive_for_positive_mean():
    assert compute_sharpe(pd.Series([1.0, 2.0, 3.0])) > 0.0


def test_cost_efficiency_zero_cost_returns_inf():
    assert np.isinf(compute_cost_efficiency(pd.Series([1.0, 2.0]), pd.Series([0.0, 0.0])))


def test_improvement_rl_better_lower_is_better():
    assert compute_improvement_over_baseline(0.4, 0.6) == pytest.approx(33.33, abs=0.01)


def test_improvement_rl_worse():
    assert compute_improvement_over_baseline(0.7, 0.5) < 0.0


def test_improvement_zero_baseline_returns_zero():
    assert compute_improvement_over_baseline(0.4, 0.0) == 0.0


def test_compute_all_metrics_keys(results_fixture):
    required = {"mean_pnl", "std_pnl", "cvar_95", "sharpe", "pct_positive", "mean_cost", "cost_efficiency", "n_episodes"}
    for agent_type in results_fixture.episode_df["agent_type"].unique():
        assert required <= compute_all_metrics(results_fixture, agent_type).keys()


def test_compute_all_metrics_unknown_agent_raises(results_fixture):
    with pytest.raises(ValueError, match="No episodes"):
        compute_all_metrics(results_fixture, "unknown")
