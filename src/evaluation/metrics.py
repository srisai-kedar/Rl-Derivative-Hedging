"""Pure hedging performance metric calculations."""

from __future__ import annotations

import pandas as pd


def compute_hedging_error(pnl: pd.Series) -> float:
    """Return the sample standard deviation of terminal P&L (lower is better)."""
    return float(pnl.std(ddof=1))


def compute_cvar(pnl: pd.Series, alpha: float = 0.05) -> float:
    """Return expected terminal P&L within the worst ``alpha`` outcomes."""
    if len(pnl) == 0:
        raise ValueError("Cannot compute CVaR on empty series")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    threshold = pnl.quantile(alpha)
    tail = pnl[pnl <= threshold]
    return float(threshold if len(tail) == 0 else tail.mean())


def compute_sharpe(pnl: pd.Series) -> float:
    """Return mean terminal P&L divided by sample standard deviation."""
    std = compute_hedging_error(pnl)
    return 0.0 if std == 0.0 else float(pnl.mean()) / std


def compute_cost_efficiency(pnl: pd.Series, costs: pd.Series) -> float:
    """Return hedging error per unit of mean transaction cost."""
    mean_cost = float(costs.mean())
    return float("inf") if mean_cost <= 0.0 else compute_hedging_error(pnl) / mean_cost


def compute_improvement_over_baseline(rl_metric: float, baseline_metric: float, lower_is_better: bool = True) -> float:
    """Return RL's percentage improvement over a baseline metric."""
    if baseline_metric == 0.0:
        return 0.0
    numerator = baseline_metric - rl_metric if lower_is_better else rl_metric - baseline_metric
    return float(numerator / abs(baseline_metric) * 100)


def compute_all_metrics(results, agent_type: str) -> dict[str, float]:
    """Compute the complete metric set for one policy in a backtest."""
    subset = results.episode_df[results.episode_df["agent_type"] == agent_type]
    if len(subset) == 0:
        available = results.episode_df["agent_type"].unique().tolist()
        raise ValueError(f"No episodes found for agent_type='{agent_type}'. Available: {available}")
    pnl = subset["terminal_pnl"]
    costs = subset["total_cost"]
    return {
        "mean_pnl": float(pnl.mean()), "std_pnl": compute_hedging_error(pnl),
        "cvar_95": compute_cvar(pnl), "sharpe": compute_sharpe(pnl),
        "pct_positive": float((pnl > 0).mean() * 100), "mean_cost": float(costs.mean()),
        "cost_efficiency": compute_cost_efficiency(pnl, costs), "n_episodes": len(pnl),
    }


def compare_metrics(results, primary: str = "rl_agent", baseline: str = "bs_delta") -> dict[str, dict[str, float]]:
    """Compute per-policy metrics and optional primary-vs-baseline deltas."""
    agent_types = results.episode_df["agent_type"].unique().tolist()
    metrics = {agent_type: compute_all_metrics(results, agent_type) for agent_type in agent_types}
    if primary in metrics and baseline in metrics:
        pm, bm = metrics[primary], metrics[baseline]
        metrics["improvement"] = {
            "std_pnl_pct": compute_improvement_over_baseline(pm["std_pnl"], bm["std_pnl"]),
            "cvar_95_pct": compute_improvement_over_baseline(pm["cvar_95"], bm["cvar_95"], lower_is_better=False),
            "cost_pct": compute_improvement_over_baseline(pm["mean_cost"], bm["mean_cost"]),
        }
    return metrics
