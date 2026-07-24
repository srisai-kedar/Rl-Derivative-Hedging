"""Experiment comparison table component."""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st


METRIC_COLUMNS = {
    "rl_std_pnl": ("RL P&L Std", True),
    "bs_std_pnl": ("BS P&L Std", True),
    "rl_cvar_95": ("RL CVaR@95", False),
    "rl_mean_cost": ("RL Mean Cost", True),
    "improvement": ("Improvement %", False),
    "n_episodes": ("N Episodes", False),
}


def render_experiment_table(base_dir: str = "results") -> None:
    """Render a comparison table of all evaluation runs in base_dir."""
    if not os.path.isdir(base_dir):
        _show_no_results()
        return

    rows = _collect_rows(base_dir)
    if not rows:
        _show_no_results()
        return

    df = pd.DataFrame(rows)

    st.markdown(
        f'<div class="section-header">All Evaluation Runs ({len(df)} found)</div>',
        unsafe_allow_html=True,
    )

    if len(df) == 1:
        st.info(
            "Only one evaluation run found. Run more experiments "
            "(different kappa, sigma, or training duration) to compare here."
        )

    styled = _style_dataframe(df)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button(
        label="Export as CSV",
        data=csv,
        file_name="experiment_comparison.csv",
        mime="text/csv",
    )


def _collect_rows(base_dir: str) -> list[dict]:
    """Scan base_dir and build one row per valid evaluation run."""
    rows = []
    for entry in sorted(os.scandir(base_dir), key=lambda e: e.name, reverse=True):
        if not entry.is_dir():
            continue

        metrics_path = os.path.join(entry.path, "metrics.json")
        if not os.path.exists(metrics_path):
            continue

        try:
            with open(metrics_path, encoding="utf-8") as file:
                metrics = json.load(file)
        except Exception:
            continue

        rl_m = metrics.get("rl_agent", {})
        bs_m = metrics.get("bs_delta", {})
        imp = metrics.get("improvement", {})

        rows.append(
            {
                "Run": entry.name,
                "RL P&L Std": rl_m.get("std_pnl"),
                "BS P&L Std": bs_m.get("std_pnl"),
                "RL CVaR@95%": rl_m.get("cvar_95"),
                "RL Mean Cost": rl_m.get("mean_cost"),
                "Improvement %": imp.get("std_pnl_pct"),
                "N Episodes": int(rl_m.get("n_episodes", 0))
                or int(bs_m.get("n_episodes", 0)),
            }
        )
    return rows


def _style_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Apply conditional formatting to the comparison DataFrame.

    Minimum values are highlighted for lower-is-better columns; maximum values
    are highlighted for higher-is-better columns.
    """
    styler = df.style

    if len(df) < 2:
        return styler.format(precision=4, na_rep="-")

    lower_better_cols = [
        col for col in ["RL P&L Std", "BS P&L Std", "RL Mean Cost"] if col in df.columns
    ]
    higher_better_cols = [col for col in ["Improvement %"] if col in df.columns]

    for col in lower_better_cols:
        styler = styler.highlight_min(subset=[col], color="#22c55e33", axis=0)
    for col in higher_better_cols:
        styler = styler.highlight_max(subset=[col], color="#22c55e33", axis=0)

    return styler.format(
        {col: "{:.4f}" for col in df.select_dtypes("float").columns},
        na_rep="-",
    )


def _show_no_results() -> None:
    """Render the no-results empty state."""
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-title">No evaluation runs found</div>
            <div class="empty-state-body">
                Run the evaluation pipeline to generate results:<br><br>
                <code>python -m src.evaluation.evaluate --checkpoint checkpoints/.../best_model.zip --vecnorm checkpoints/.../best_vecnorm.pkl</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
