"""Stateless Plotly figures for hedging-policy evaluation."""

from __future__ import annotations

import logging
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.evaluation.metrics import compute_cvar

logger = logging.getLogger(__name__)
COLOURS = {"rl_agent": "#3b82f6", "bs_delta": "#f59e0b", "zero_hedge": "#6b7280", "cost": "#ef4444", "positive": "#22c55e", "neutral": "#94a3b8"}
BASE_LAYOUT = {
    "paper_bgcolor": "#1e2130", "plot_bgcolor": "#0f1117",
    "font": {"color": "#e2e8f0", "family": "Inter, system-ui, sans-serif", "size": 13},
    "legend": {"bgcolor": "#1e2130", "bordercolor": "#2d3748", "borderwidth": 1},
    "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
    "xaxis": {"gridcolor": "#2d3748", "linecolor": "#2d3748", "zerolinecolor": "#2d3748"},
    "yaxis": {"gridcolor": "#2d3748", "linecolor": "#2d3748", "zerolinecolor": "#2d3748"},
}
AGENT_DISPLAY_NAMES = {"rl_agent": "RL Agent", "bs_delta": "BS Delta", "zero_hedge": "Zero Hedge"}


def _name(agent_type: str) -> str:
    return AGENT_DISPLAY_NAMES.get(agent_type, agent_type.replace("_", " ").title())


def _transparent(colour: str, opacity: float = 0.12) -> str:
    rgb = tuple(int(colour[index : index + 2], 16) for index in (1, 3, 5))
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"


def _axis_style(fig: go.Figure) -> None:
    fig.update_xaxes(gridcolor="#2d3748", linecolor="#2d3748", zerolinecolor="#2d3748")
    fig.update_yaxes(gridcolor="#2d3748", linecolor="#2d3748", zerolinecolor="#2d3748")


def plot_pnl_distribution(results, n_bins: int = 60, title: str = "Terminal P&L Distribution") -> go.Figure:
    fig = go.Figure()
    for agent_type in results.episode_df["agent_type"].unique():
        pnl = results.episode_df.loc[results.episode_df["agent_type"] == agent_type, "terminal_pnl"]
        colour = COLOURS.get(agent_type, COLOURS["neutral"])
        fig.add_trace(go.Histogram(x=pnl, name=_name(agent_type), opacity=0.65, nbinsx=n_bins, marker_color=colour))
        fig.add_vline(x=float(pnl.mean()), line_dash="dash", line_color=colour)
        if agent_type == "rl_agent":
            fig.add_vline(x=compute_cvar(pnl), line_dash="dot", line_color=colour, annotation_text="RL CVaR@95%")
    fig.update_layout(**BASE_LAYOUT, title=title, barmode="overlay", xaxis_title="Terminal P&L ($)", yaxis_title="Episode Count")
    return fig


def plot_hedge_ratio_over_time(results, agent_types: list[str] | None = None, title: str = "Hedge Ratio Over Time") -> go.Figure:
    fig = go.Figure()
    available = results.step_df["agent_type"].unique().tolist()
    for agent_type in agent_types or available:
        data = results.step_df[results.step_df["agent_type"] == agent_type]
        if data.empty:
            continue
        grouped = data.groupby("step")["hedge_ratio"].agg(["mean", "std"]).fillna(0.0)
        x, mean, std = grouped.index.to_numpy(), grouped["mean"].to_numpy(), grouped["std"].to_numpy()
        colour = COLOURS.get(agent_type, COLOURS["neutral"])
        fig.add_trace(go.Scatter(x=np.concatenate((x, x[::-1])), y=np.concatenate((mean + std, (mean - std)[::-1])), fill="toself", fillcolor=_transparent(colour), line={"color": "rgba(0,0,0,0)"}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", name=_name(agent_type), line={"color": colour}))
    fig.update_layout(**BASE_LAYOUT, title=title, xaxis_title="Episode Step", yaxis_title="Hedge Ratio", yaxis_range=[-0.05, 1.05])
    return fig


def plot_metric_comparison(metrics: dict[str, dict[str, float]], metric_keys: list[str] | None = None, title: str = "Metric Comparison") -> go.Figure:
    fig = go.Figure()
    keys = metric_keys or ["std_pnl", "cvar_95", "mean_cost"]
    agents = [agent for agent, values in metrics.items() if agent != "improvement" and isinstance(values, dict)]
    for agent_type in agents:
        fig.add_trace(go.Bar(name=_name(agent_type), x=keys, y=[metrics[agent_type].get(key, np.nan) for key in keys], marker_color=COLOURS.get(agent_type, COLOURS["neutral"])))
    fig.update_layout(**BASE_LAYOUT, title=title, barmode="group", xaxis_title="Metric", yaxis_title="Value")
    return fig


def plot_cost_sensitivity(robustness_df, metric: str = "std_pnl", title: str = "Transaction Cost Sensitivity") -> go.Figure:
    fig = go.Figure()
    for agent_type in robustness_df["agent_type"].unique():
        data = robustness_df[robustness_df["agent_type"] == agent_type].groupby("kappa", as_index=False)[metric].mean().sort_values("kappa")
        fig.add_trace(go.Scatter(x=data["kappa"], y=data[metric], mode="lines+markers", name=_name(agent_type), line={"color": COLOURS.get(agent_type, COLOURS["neutral"])}))
    fig.update_layout(**BASE_LAYOUT, title=title, xaxis_title="Transaction Cost Coefficient (kappa)", yaxis_title=metric)
    return fig


def plot_episode_replay(results, episode_id: int, title: str | None = None) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Stock Price", "Hedge Ratio", "Step P&L", "Cumulative P&L and Cost"))
    data = results.step_df[results.step_df["episode_id"] == episode_id]
    if data.empty:
        raise ValueError(f"No step data found for episode_id={episode_id}")
    first = data["agent_type"].iloc[0]
    price_data = data[data["agent_type"] == first]
    fig.add_trace(go.Scatter(x=price_data["step"], y=price_data["price"], mode="lines", name="Stock Price", line={"color": COLOURS["neutral"]}), row=1, col=1)
    bs_added = False
    for agent_type in data["agent_type"].unique():
        policy_data = data[data["agent_type"] == agent_type]
        dash = "dash" if agent_type == "bs_delta" else "solid"
        fig.add_trace(go.Scatter(x=policy_data["step"], y=policy_data["hedge_ratio"], mode="lines", name=f"{_name(agent_type)} hedge", line={"color": COLOURS.get(agent_type, COLOURS["neutral"]), "dash": dash}), row=1, col=2)
        if not bs_added:
            fig.add_trace(go.Scatter(x=policy_data["step"], y=policy_data["bs_delta"], mode="lines", name="BS Delta", line={"color": COLOURS["neutral"], "dash": "dot"}), row=1, col=2)
            bs_added = True
        colours = [COLOURS["positive"] if value >= 0 else COLOURS["cost"] for value in policy_data["step_pnl"]]
        fig.add_trace(go.Bar(x=policy_data["step"], y=policy_data["step_pnl"], name=f"{_name(agent_type)} step P&L", marker_color=colours), row=2, col=1)
        fig.add_trace(go.Scatter(x=policy_data["step"], y=policy_data["step_pnl"].cumsum(), mode="lines", name=f"{_name(agent_type)} cumulative P&L", line={"color": COLOURS.get(agent_type, COLOURS["neutral"])}), row=2, col=2)
        fig.add_trace(go.Scatter(x=policy_data["step"], y=policy_data["step_cost"].cumsum(), mode="lines", name=f"{_name(agent_type)} cumulative cost", line={"color": COLOURS["cost"], "dash": "dot"}), row=2, col=2)
    fig.update_layout(**BASE_LAYOUT, title=title or f"Episode {episode_id} Replay", height=600)
    _axis_style(fig)
    return fig


def plot_robustness_heatmap(robustness_df, metric: str = "std_pnl", agent_type: str = "rl_agent", title: str | None = None) -> go.Figure:
    data = robustness_df[robustness_df["agent_type"] == agent_type]
    pivot = data.pivot(index="sigma", columns="kappa", values=metric).sort_index().sort_index(axis=1)
    fig = go.Figure(go.Heatmap(z=pivot.to_numpy(), x=[f"k={value:.4f}" for value in pivot.columns], y=[f"s={value:.2f}" for value in pivot.index], colorscale="RdBu_r"))
    fig.update_layout(**BASE_LAYOUT, title=title or f"{_name(agent_type)} {metric} Robustness", xaxis_title="Transaction Cost Coefficient (kappa)", yaxis_title="Volatility (sigma)")
    return fig


def save_figure(fig: go.Figure, output_dir: str, name: str) -> None:
    """Save HTML always and PNG when Kaleido is available."""
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, f"{name}.html"))
    try:
        fig.write_image(os.path.join(output_dir, f"{name}.png"))
    except (ImportError, OSError, ValueError) as error:
        logger.warning("Could not save PNG for %s: %s", name, error)
