"""
RL Derivative Hedging Streamlit Dashboard.

Entry point. Run with: streamlit run src/dashboard/app.py

Architecture:
    - Page config must be the first Streamlit call in any module.
    - CSS is injected immediately after.
    - All page-level logic is delegated to page-render functions.
    - Components are stateless functions; state lives in st.session_state.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from src.evaluation.backtest import BacktestResults

# st.set_page_config MUST be the first Streamlit call.
st.set_page_config(
    page_title="RL Hedging Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)


def _load_css() -> None:
    """Inject custom CSS from style.css into the Streamlit app."""
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as file:
            st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
    else:
        logger.warning("CSS file not found at %s", css_path)


_load_css()


@st.cache_data(ttl=30)
def discover_result_dirs(base_dir: str = "results") -> list[str]:
    """
    Find all subdirectories of base_dir that contain a valid evaluation run.

    A valid run must contain both episode_data.csv and metrics.json. Results
    are sorted newest-first by directory name.
    """
    if not os.path.isdir(base_dir):
        return []

    valid_dirs: list[str] = []
    for entry in sorted(os.scandir(base_dir), key=lambda e: e.name, reverse=True):
        if not entry.is_dir():
            continue
        has_episodes = os.path.exists(os.path.join(entry.path, "episode_data.csv"))
        has_metrics = os.path.exists(os.path.join(entry.path, "metrics.json"))
        if has_episodes and has_metrics:
            valid_dirs.append(entry.path)
    return valid_dirs


@st.cache_data
def load_results_cached(output_dir: str) -> tuple[BacktestResults | None, dict | None]:
    """
    Load BacktestResults and metrics dict from a result directory.

    Returns (None, None) if the directory cannot be loaded.
    """
    from src.evaluation.backtest import load_results

    try:
        results, metrics = load_results(output_dir)
        return results, metrics
    except Exception as error:
        logger.error("Failed to load results from %s: %s", output_dir, error)
        return None, None


def _render_sidebar() -> tuple[str, str, bool]:
    """
    Render the full sidebar and return user selections.

    Returns:
        (selected_page, selected_results_dir, show_zero_hedge)
    """
    with st.sidebar:
        st.markdown("## RL Hedging")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            options=["Overview", "Episode Replay", "Experiments", "Training Logs"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Evaluation Results**")
        result_dirs = discover_result_dirs("results")

        if result_dirs:
            dir_labels = {os.path.basename(path): path for path in result_dirs}
            selected_label = st.selectbox(
                "Select run",
                options=list(dir_labels.keys()),
                label_visibility="collapsed",
            )
            selected_dir = dir_labels[selected_label]
            st.markdown(
                f'<span class="run-label">{selected_label}</span>',
                unsafe_allow_html=True,
            )
        else:
            selected_dir = ""
            st.info("No evaluation runs found.\nRun the evaluation pipeline first.")

        st.markdown("---")
        st.markdown("**Display Settings**")
        show_zero_hedge = st.toggle("Show zero-hedge baseline", value=False)

        st.markdown("---")
        st.markdown("**Quick Demo**")
        if st.button("Run baseline-only eval (100 eps)", use_container_width=True):
            _run_quick_eval_into_session_state()

    return page, selected_dir, show_zero_hedge


def _run_quick_eval_into_session_state() -> None:
    """
    Run a fast baseline-only backtest and store it in session_state.

    This demo result is not saved to disk.
    """
    from src.evaluation.backtest import BSDeltaPolicy, ZeroHedgePolicy, run_backtest
    from src.evaluation.metrics import compare_metrics
    from src.training.hyperparams import EnvironmentConfig

    env_config = EnvironmentConfig()

    with st.spinner("Running 100-episode baseline evaluation..."):
        results = run_backtest(
            policies=[BSDeltaPolicy(), ZeroHedgePolicy()],
            env_config=env_config,
            n_episodes=100,
            show_progress=False,
        )
        metrics = compare_metrics(results, primary="bs_delta", baseline="zero_hedge")

    st.session_state["quick_eval_results"] = results
    st.session_state["quick_eval_metrics"] = metrics
    st.success("Quick eval complete. Viewing results below.")
    st.rerun()


def main() -> None:
    """Render the selected dashboard page."""
    page, selected_dir, show_zero_hedge = _render_sidebar()

    if selected_dir:
        results, metrics = load_results_cached(selected_dir)
    elif "quick_eval_results" in st.session_state:
        results = st.session_state["quick_eval_results"]
        metrics = st.session_state["quick_eval_metrics"]
    else:
        results, metrics = None, None

    if page == "Overview":
        render_overview_page(results, metrics, show_zero_hedge)
    elif page == "Episode Replay":
        render_replay_page(results)
    elif page == "Experiments":
        render_experiments_page()
    elif page == "Training Logs":
        render_training_logs_page(selected_dir)


def render_overview_page(
    results: BacktestResults | None,
    metrics: dict | None,
    show_zero_hedge: bool,
) -> None:
    """
    Render the overview page with metric cards and analysis charts.
    """
    from src.dashboard.components.metric_cards import render_metric_cards
    from src.dashboard.components.pnl_chart import render_pnl_section

    st.markdown("## Overview")

    if results is None or metrics is None:
        _render_empty_state(
            title="No results loaded",
            body=(
                "Select an evaluation run from the sidebar,<br>"
                "or click 'Run baseline-only eval' to generate a quick demo."
            ),
        )
        return

    available_agents = results.episode_df["agent_type"].unique().tolist()
    agents_to_show = available_agents if show_zero_hedge else [
        agent for agent in available_agents if agent != "zero_hedge"
    ]

    st.markdown(
        '<div class="section-header">Performance vs Black-Scholes Delta</div>',
        unsafe_allow_html=True,
    )
    render_metric_cards(metrics)

    st.markdown("---")
    render_pnl_section(results, agents_to_show, metrics)


def render_replay_page(results: BacktestResults | None) -> None:
    """Render the episode replay page."""
    from src.dashboard.components.episode_replay import render_episode_replay_page

    st.markdown("## Episode Replay")

    if results is None:
        _render_empty_state(
            title="No results loaded",
            body="Load an evaluation run from the sidebar to replay episodes.",
        )
        return

    render_episode_replay_page(results)


def render_experiments_page() -> None:
    """Render the experiment comparison page."""
    from src.dashboard.components.experiment_table import render_experiment_table

    st.markdown("## Experiment Comparison")
    render_experiment_table(base_dir="results")


def render_training_logs_page(selected_dir: str) -> None:
    """
    Render Monitor CSV training logs associated with the selected run.
    """
    st.markdown("## Training Logs")

    if not selected_dir:
        _render_empty_state(
            title="No run selected",
            body=(
                "Select an evaluation run from the sidebar.<br>"
                "Training logs are read from the run's monitor/ directory."
            ),
        )
        return

    monitor_dir = os.path.normpath(os.path.join(selected_dir, "..", "monitor"))
    csv_files = glob.glob(os.path.join(monitor_dir, "*.monitor.csv"))

    if not csv_files:
        alt_monitor = os.path.join(selected_dir, "monitor")
        csv_files = glob.glob(os.path.join(alt_monitor, "*.monitor.csv"))

    if not csv_files:
        _render_empty_state(
            title="No training logs found",
            body=(
                "Monitor CSV files were not found for this run.<br>"
                "Ensure training was run with Monitor wrapping enabled<br>"
                "and that the log_dir was set in env_factory.py."
            ),
        )
        return

    _render_monitor_chart(csv_files)


def _render_empty_state(title: str, body: str) -> None:
    """Render a centered empty-state card with a title and body message."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_monitor_chart(csv_files: list[str]) -> None:
    """Read Monitor CSVs and render a rolling mean reward chart."""
    import pandas as pd
    import plotly.graph_objects as go

    from src.evaluation.plots import BASE_LAYOUT

    dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, comment="#")
            df.columns = df.columns.str.strip()
            dfs.append(df)
        except Exception as error:
            logger.warning("Could not parse monitor CSV %s: %s", file_path, error)

    if not dfs:
        st.warning("Monitor CSV files found but could not be parsed.")
        return

    monitor_df = pd.concat(dfs, ignore_index=True).sort_values("t")
    monitor_df["rolling_reward"] = (
        monitor_df["r"].rolling(window=50, min_periods=1).mean()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monitor_df["t"],
            y=monitor_df["r"],
            name="Episode Reward",
            line={"color": "#2d3748", "width": 1},
            opacity=0.4,
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monitor_df["t"],
            y=monitor_df["rolling_reward"],
            name="Rolling Mean (50 eps)",
            line={"color": "#3b82f6", "width": 2},
            mode="lines",
        )
    )
    fig.update_layout(
        **BASE_LAYOUT,
        title={"text": "Training Episode Reward Over Time", "x": 0.5, "xanchor": "center"},
        xaxis_title="Training Time (seconds)",
        yaxis_title="Episode Reward",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


if __name__ == "__main__":
    main()
