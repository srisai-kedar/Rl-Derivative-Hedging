"""Tests for bootstrap / multi-seed uncertainty quantification."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics, uncertainty


# --------------------- consistency with the existing metrics --------------- #
@pytest.mark.parametrize("seed", range(5))
def test_row_statistics_match_metrics_module(seed):
    x = np.random.default_rng(seed).standard_t(3, size=257)
    s = pd.Series(x)
    assert uncertainty.stat_mean(x[None, :])[0] == pytest.approx(s.mean())
    assert uncertainty.stat_std(x[None, :])[0] == pytest.approx(metrics.compute_hedging_error(s))
    assert uncertainty.make_stat_cvar(0.05)(x[None, :])[0] == pytest.approx(metrics.compute_cvar(s))
    assert uncertainty.make_stat_cvar(0.10)(x[None, :])[0] == pytest.approx(
        metrics.compute_cvar(s, alpha=0.10)
    )


# ------------------------------ bootstrap CI ------------------------------- #
def test_bootstrap_ci_brackets_point_and_is_reproducible():
    x = np.random.default_rng(1).normal(2.0, 1.0, 300)
    a = uncertainty.bootstrap_ci(x, uncertainty.stat_mean, n_boot=2000, seed=3)
    b = uncertainty.bootstrap_ci(x, uncertainty.stat_mean, n_boot=2000, seed=3)
    assert a == b
    assert a.lo < a.point < a.hi


def test_bootstrap_mean_ci_coverage_is_near_nominal():
    rng = np.random.default_rng(0)
    hits, reps = 0, 200
    for i in range(reps):
        x = rng.normal(0.0, 1.0, 80)
        ci = uncertainty.bootstrap_ci(x, uncertainty.stat_mean, n_boot=400, seed=i)
        hits += ci.lo <= 0.0 <= ci.hi
    assert 0.88 <= hits / reps <= 0.995  # nominal 0.95, allow MC noise


def test_cvar_ci_is_wider_than_mean_ci_for_heavy_tails():
    x = np.random.default_rng(2).standard_t(2.5, 500)
    m = uncertainty.bootstrap_ci(x, uncertainty.stat_mean, n_boot=1000)
    c = uncertainty.bootstrap_ci(x, uncertainty.make_stat_cvar(), n_boot=1000)
    assert (c.hi - c.lo) > (m.hi - m.lo)


# ------------------------------ paired bootstrap --------------------------- #
def test_paired_diff_detects_real_shift_and_ignores_noise():
    rng = np.random.default_rng(4)
    base = rng.normal(0, 1, 400)
    shifted = base + 0.5 + rng.normal(0, 0.1, 400)
    d, p = uncertainty.paired_bootstrap_diff(shifted, base, uncertainty.stat_mean, n_boot=2000)
    assert d.excludes_zero()
    assert p < 0.01
    same = base + rng.normal(0, 0.1, 400)
    d0, p0 = uncertainty.paired_bootstrap_diff(same, base, uncertainty.stat_mean, n_boot=2000)
    assert not d0.excludes_zero()
    assert p0 > 0.05


def test_pairing_reduces_variance_versus_unpaired():
    rng = np.random.default_rng(5)
    common = rng.normal(0, 3, 400)  # shared path noise
    a = common + rng.normal(0, 0.2, 400) + 0.1
    b = common + rng.normal(0, 0.2, 400)
    paired, _ = uncertainty.paired_bootstrap_diff(a, b, uncertainty.stat_mean, n_boot=2000)
    ia = uncertainty.bootstrap_ci(a, uncertainty.stat_mean, n_boot=2000)
    ib = uncertainty.bootstrap_ci(b, uncertainty.stat_mean, n_boot=2000)
    unpaired_width = np.hypot(ia.hi - ia.lo, ib.hi - ib.lo)
    assert (paired.hi - paired.lo) < 0.25 * unpaired_width


@pytest.mark.parametrize(
    "kwargs",
    [{"n_boot": 10}, {"alpha": 0.0}, {"alpha": 1.0}],
)
def test_bad_arguments_rejected(kwargs):
    with pytest.raises(ValueError, match=r"n_boot|alpha"):
        uncertainty.bootstrap_ci(np.arange(10.0), **kwargs)


@pytest.mark.parametrize("bad", [[1.0], [1.0, np.nan, 2.0], [[1.0, 2.0], [3.0, 4.0]]])
def test_bad_samples_rejected(bad):
    with pytest.raises(ValueError, match="values"):
        uncertainty.bootstrap_ci(np.asarray(bad))


# --------------------------- policy-level report --------------------------- #
def _episode_df(n=300, seed=0, drop_seed=None):
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 1, n)
    rows = []
    for agent, scale in (("rl_agent", 2.0), ("bs_delta", 0.5)):
        for i in range(n):
            rows.append(
                {"agent_type": agent, "seed": i, "terminal_pnl": scale * common[i]}
            )
    df = pd.DataFrame(rows)
    if drop_seed is not None:
        df = df[~((df.agent_type == "rl_agent") & (df.seed == drop_seed))]
    return df


def test_compare_with_ci_structure_and_direction():
    rep = uncertainty.compare_with_ci(_episode_df(), n_boot=500)
    assert set(rep) == {"mean_pnl", "std_pnl", "cvar_95"}
    std = rep["std_pnl"]
    assert std["diff"]["point"] > 0
    assert std["diff"]["lo"] > 0  # RL strictly noisier
    assert 0 < std["diff"]["p_value"] <= 1


def test_compare_with_ci_rejects_mismatched_seeds():
    with pytest.raises(ValueError, match="identical seed"):
        uncertainty.compare_with_ci(_episode_df(drop_seed=3), n_boot=200)


def test_compare_with_ci_rejects_any_policy_with_different_seed_set():
    frame = _episode_df()
    frame = frame[~((frame.agent_type == "bs_delta") & (frame.seed == 4))]
    with pytest.raises(ValueError, match="identical seed"):
        uncertainty.compare_with_ci(frame, n_boot=200)


def test_compare_with_ci_rejects_missing_policy_and_columns():
    with pytest.raises(ValueError, match="Need both"):
        uncertainty.compare_with_ci(_episode_df().query("agent_type == 'rl_agent'"), n_boot=200)
    with pytest.raises(ValueError, match="missing columns"):
        uncertainty.compare_with_ci(pd.DataFrame({"x": [1]}), n_boot=200)


# ------------------------- across training seeds --------------------------- #
def test_aggregate_run_metrics_t_interval(tmp_path):
    vals = [2.0, 2.4, 1.8, 2.2, 2.6]
    paths = []
    for i, v in enumerate(vals):
        p = tmp_path / f"m{i}.json"
        p.write_text(json.dumps({"rl_agent": {"std_pnl": v, "cost_efficiency": float("inf")}}))
        paths.append(p)
    out = uncertainty.aggregate_run_metrics(paths)
    row = out.loc["std_pnl"]
    arr = np.array(vals)
    assert row["mean"] == pytest.approx(arr.mean())
    assert row["std"] == pytest.approx(arr.std(ddof=1))
    half = 2.7764451 * arr.std(ddof=1) / np.sqrt(5)  # t_{0.975, 4}
    assert row["ci_lo"] == pytest.approx(arr.mean() - half, abs=1e-5)
    assert np.isnan(out.loc["cost_efficiency", "mean"])  # inf -> NaN, not crash


def test_aggregate_requires_two_runs_and_known_agent(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"rl_agent": {"std_pnl": 1.0}}))
    with pytest.raises(ValueError, match=">= 2"):
        uncertainty.aggregate_run_metrics([p])
    with pytest.raises(KeyError):
        uncertainty.aggregate_run_metrics([p, p], agent="nope")
