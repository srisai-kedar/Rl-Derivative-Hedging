"""Uncertainty quantification for hedging metrics.

Point estimates over N episodes say nothing about how much of a gap between two
policies is signal. This module adds:

* ``bootstrap_ci``            - percentile bootstrap CI for one policy/metric.
* ``paired_bootstrap_diff``   - CI for ``stat(A) - stat(B)`` where A and B were
                                evaluated on the *same seeds* (paired resampling
                                preserves the common-random-number variance
                                reduction that the backtester already pays for).
* ``compare_with_ci``         - ready-made RL-vs-baseline report from an
                                ``episode_df``.
* ``aggregate_run_metrics``   - mean / Student-t CI across independent
                                *training* seeds (the dominant source of
                                variance in RL, invisible to episode bootstraps).

All CVaR/std definitions match ``src.evaluation.metrics`` (pandas linear-
interpolated quantile, ``ddof=1``), which the tests assert.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

StatFn = Callable[[np.ndarray], np.ndarray]
"""Row-wise statistic: maps an (m, n) array to an (m,) array."""

_CHUNK_ROWS = 1000  # bound peak memory at ~chunk * n * 8 bytes


@dataclass(frozen=True)
class Interval:
    """A point estimate with a (1 - alpha) confidence interval."""

    point: float
    lo: float
    hi: float
    alpha: float = 0.05

    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "alpha": self.alpha}


# --------------------------------------------------------------------------- #
# Row-wise statistics (vectorised over bootstrap replicates)
# --------------------------------------------------------------------------- #
def stat_mean(x: np.ndarray) -> np.ndarray:
    return np.asarray(x.mean(axis=1))


def stat_std(x: np.ndarray) -> np.ndarray:
    return np.asarray(x.std(axis=1, ddof=1))


def make_stat_cvar(alpha: float = 0.05) -> StatFn:
    """CVaR_alpha with the same semantics as ``metrics.compute_cvar``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    def _cvar(x: np.ndarray) -> np.ndarray:
        thr = np.quantile(x, alpha, axis=1)  # linear interpolation == pandas
        mask = x <= thr[:, None]
        # mask always contains >=1 element per row (the quantile lies within
        # the sample range), so the denominator is never zero.
        return np.asarray((x * mask).sum(axis=1) / mask.sum(axis=1))

    return _cvar


STAT_LIBRARY: dict[str, StatFn] = {
    "mean_pnl": stat_mean,
    "std_pnl": stat_std,
    "cvar_95": make_stat_cvar(0.05),
}


# --------------------------------------------------------------------------- #
# Core bootstrap
# --------------------------------------------------------------------------- #
def _validate_1d(x: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size < 2:
        raise ValueError(f"{name} needs >= 2 observations, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _check_args(n_boot: int, alpha: float) -> None:
    if n_boot < 100:
        raise ValueError(f"n_boot must be >= 100 for a meaningful CI, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _boot_distribution(
    arrays: Sequence[np.ndarray], stat_fn: StatFn, n_boot: int, seed: int
) -> np.ndarray:
    """Bootstrap distribution of the statistic for each array, jointly resampled.

    All arrays are indexed with the *same* random indices, i.e. a paired
    bootstrap. Returns an array of shape (len(arrays), n_boot).
    """
    n = arrays[0].shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty((len(arrays), n_boot))
    done = 0
    while done < n_boot:
        m = min(_CHUNK_ROWS, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        for j, arr in enumerate(arrays):
            out[j, done : done + m] = stat_fn(arr[idx])
        done += m
    return out


def bootstrap_ci(
    values: np.ndarray | Sequence[float],
    stat_fn: StatFn = stat_mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for ``stat_fn`` applied to one sample."""
    _check_args(n_boot, alpha)
    x = _validate_1d(values, "values")
    point = float(stat_fn(x[None, :])[0])
    dist = _boot_distribution([x], stat_fn, n_boot, seed)[0]
    lo, hi = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    return Interval(point, float(lo), float(hi), alpha)


def paired_bootstrap_diff(
    a: np.ndarray | Sequence[float],
    b: np.ndarray | Sequence[float],
    stat_fn: StatFn = stat_mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[Interval, float]:
    """CI for ``stat(a) - stat(b)`` with episode-paired resampling.

    ``a[i]`` and ``b[i]`` must come from the same seed/path. Returns the
    interval and a two-sided bootstrap p-value for "difference == 0"
    (an approximation: it is ``2 * min(P(d<=0), P(d>=0))`` over replicates,
    floored at ``1 / n_boot``).
    """
    _check_args(n_boot, alpha)
    xa, xb = _validate_1d(a, "a"), _validate_1d(b, "b")
    if xa.shape != xb.shape:
        raise ValueError(f"Paired samples must match: {xa.shape} vs {xb.shape}")
    point = float(stat_fn(xa[None, :])[0] - stat_fn(xb[None, :])[0])
    dist = _boot_distribution([xa, xb], stat_fn, n_boot, seed)
    diff = dist[0] - dist[1]
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    p = 2.0 * min(float((diff <= 0).mean()), float((diff >= 0).mean()))
    return Interval(point, float(lo), float(hi), alpha), max(min(p, 1.0), 1.0 / n_boot)


# --------------------------------------------------------------------------- #
# Policy-level reporting
# --------------------------------------------------------------------------- #
def _paired_arrays(
    episode_df: pd.DataFrame, primary: str, baseline: str
) -> tuple[np.ndarray, np.ndarray]:
    required = {"agent_type", "seed", "terminal_pnl"}
    missing = required - set(episode_df.columns)
    if missing:
        raise ValueError(f"episode_df is missing columns: {sorted(missing)}")
    cols = ["seed", "terminal_pnl"]
    pa = episode_df.loc[episode_df["agent_type"] == primary, cols]
    pb = episode_df.loc[episode_df["agent_type"] == baseline, cols]
    if pa.empty or pb.empty:
        have = sorted(episode_df["agent_type"].unique().tolist())
        raise ValueError(f"Need both '{primary}' and '{baseline}'; have {have}")
    if pa["seed"].duplicated().any() or pb["seed"].duplicated().any():
        raise ValueError("Duplicate seeds within a policy; pairing is ambiguous")
    merged = pa.merge(pb, on="seed", suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(pa) or len(merged) != len(pb):
        raise ValueError("Policies were not evaluated on identical seed sets")
    merged = merged.sort_values("seed")
    return merged["terminal_pnl_a"].to_numpy(), merged["terminal_pnl_b"].to_numpy()


def _validate_seed_alignment(episode_df: pd.DataFrame) -> None:
    """Require every policy in a backtest to use the same seed set."""
    required = {"agent_type", "seed"}
    missing = required - set(episode_df.columns)
    if missing:
        return
    seed_sets = {
        agent: set(group["seed"].tolist())
        for agent, group in episode_df.groupby("agent_type")
    }
    if len(seed_sets) < 2:
        return
    first_agent, first_seeds = next(iter(seed_sets.items()))
    mismatched = [agent for agent, seeds in seed_sets.items() if seeds != first_seeds]
    if mismatched:
        raise ValueError(
            "Policies were not evaluated on identical seed sets: "
            f"{first_agent} differs from {mismatched}."
        )


def compare_with_ci(
    episode_df: pd.DataFrame,
    primary: str = "rl_agent",
    baseline: str = "bs_delta",
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-metric CIs for each policy and for the paired difference.

    Returns ``{metric: {"primary": ci, "baseline": ci, "diff": ci+p_value}}``.
    A difference CI that excludes 0 supports a real gap *across episodes*; it
    says nothing about variability across training runs (see
    ``aggregate_run_metrics``).
    """
    _validate_seed_alignment(episode_df)
    a, b = _paired_arrays(episode_df, primary, baseline)
    report: dict[str, dict[str, dict[str, float]]] = {}
    for name, fn in STAT_LIBRARY.items():
        d, p = paired_bootstrap_diff(a, b, fn, n_boot, alpha, seed)
        report[name] = {
            "primary": bootstrap_ci(a, fn, n_boot, alpha, seed).as_dict(),
            "baseline": bootstrap_ci(b, fn, n_boot, alpha, seed).as_dict(),
            "diff": {**d.as_dict(), "p_value": p},
        }
    return report


# --------------------------------------------------------------------------- #
# Across independent training runs
# --------------------------------------------------------------------------- #
def aggregate_run_metrics(
    metrics_paths: Sequence[str | Path],
    agent: str = "rl_agent",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Mean, std and Student-t CI of each metric across training runs.

    Each path is a ``metrics.json`` written by the evaluation CLI for one
    independently trained seed. With k runs the CI uses k-1 degrees of
    freedom; k < 2 is rejected because a single run has no variance estimate.
    """
    if len(metrics_paths) < 2:
        raise ValueError("Need metrics from >= 2 independent training runs")
    rows = []
    for p in metrics_paths:
        with open(p, encoding="utf-8") as fh:
            payload = json.load(fh)
        if agent not in payload:
            raise KeyError(f"'{agent}' not found in {p}; keys={list(payload)}")
        rows.append(payload[agent])
    df = pd.DataFrame(rows).select_dtypes("number")
    df = df.replace([np.inf, -np.inf], np.nan)
    k = len(df)
    mean = df.mean()
    sd = df.std(ddof=1)
    half = sps.t.ppf(1 - alpha / 2, k - 1) * sd / np.sqrt(k)
    return pd.DataFrame(
        {"mean": mean, "std": sd, "ci_lo": mean - half, "ci_hi": mean + half, "k": k}
    )
