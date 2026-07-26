"""P&L analysis chart section for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

from src.evaluation.backtest import BacktestResults
from src.evaluation.plots import (
    plot_hedge_ratio_over_time,
    plot_metric_comparison,
    plot_pnl_distribution,
)

PLOTLY_CONFIG = {"displayModeBar": True}


def render_pnl_section(
    results: BacktestResults,
    agents_to_show: list[str],
    metrics: dict,
) -> None:
    """Render the three main overview analysis charts."""
    filtered_results = _filter_results(results, agents_to_show)

    st.markdown(
        '<div class="section-header">Terminal P&L Distribution</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Overlaid distribution of terminal P&L across all evaluation episodes. "
        "A narrower RL distribution indicates better hedging performance."
    )
    fig_dist = plot_pnl_distribution(filtered_results)
    st.plotly_chart(fig_dist, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown(
        '<div class="section-header">Mean Hedge Ratio Over Episode</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Average hedge ratio with one standard deviation shading held by each "
        "policy at each trading day within an episode."
    )
    fig_hedge = plot_hedge_ratio_over_time(filtered_results, agent_types=agents_to_show)
    st.plotly_chart(fig_hedge, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown(
        '<div class="section-header">Key Metrics Comparison</div>',
        unsafe_allow_html=True,
    )
    filtered_metrics = {
        key: value
        for key, value in metrics.items()
        if key in agents_to_show or key == "improvement"
    }
    fig_compare = plot_metric_comparison(filtered_metrics)
    st.plotly_chart(fig_compare, use_container_width=True, config=PLOTLY_CONFIG)


def _filter_results(results: BacktestResults, agents: list[str]) -> BacktestResults:
    """Return a BacktestResults containing only the specified agents."""
    episode_df = results.episode_df[results.episode_df["agent_type"].isin(agents)]
    step_df = results.step_df[results.step_df["agent_type"].isin(agents)]
    return BacktestResults(
        episode_df=episode_df.reset_index(drop=True),
        step_df=step_df.reset_index(drop=True),
    )
