"""Metric card row component for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

from src.evaluation.metrics import compute_improvement_over_baseline

CARD_SPECS: list[tuple[str, str, bool]] = [
    ("P&L Std Dev", "std_pnl", True),
    ("CVaR @ 95%", "cvar_95", False),
    ("Mean P&L", "mean_pnl", False),
    ("Mean Tx Cost", "mean_cost", True),
]


def render_metric_cards(
    metrics: dict,
    primary: str = "rl_agent",
    baseline: str = "bs_delta",
) -> None:
    """
    Render a horizontal row of metric cards.

    The primary agent value is prominent, the baseline value is secondary,
    and the delta is colored by whether it improves over the baseline.
    """
    primary_m = metrics.get(primary, {})
    baseline_m = metrics.get(baseline, {})
    baseline_display = _agent_display_name(baseline)

    cols = st.columns(len(CARD_SPECS), gap="small")

    for col, (label, key, lower_is_better) in zip(cols, CARD_SPECS, strict=True):
        with col:
            primary_val = primary_m.get(key)
            baseline_val = baseline_m.get(key)

            primary_str = f"{primary_val:.4f}" if primary_val is not None else "--"
            baseline_str = f"{baseline_val:.4f}" if baseline_val is not None else "--"

            if primary_val is not None and baseline_val is not None:
                improvement = compute_improvement_over_baseline(
                    primary_val,
                    baseline_val,
                    lower_is_better,
                )
                if improvement > 0.5:
                    delta_class = "metric-delta-positive"
                    delta_text = f"+{improvement:.1f}% better"
                elif improvement < -0.5:
                    delta_class = "metric-delta-negative"
                    delta_text = f"{improvement:.1f}% worse"
                else:
                    delta_class = "metric-delta-neutral"
                    delta_text = "near parity"
            else:
                delta_class = "metric-delta-neutral"
                delta_text = "no comparison"

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-rl-value">{primary_str}</div>
                    <div class="metric-baseline-row">
                        <span class="metric-bs-label">{baseline_display}</span>
                        <span class="metric-bs-value">{baseline_str}</span>
                    </div>
                    <div class="{delta_class}">{delta_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _agent_display_name(agent_type: str) -> str:
    return {
        "rl_agent": "RL Agent",
        "bs_delta": "BS Delta",
        "zero_hedge": "Zero Hedge",
    }.get(agent_type, agent_type)
