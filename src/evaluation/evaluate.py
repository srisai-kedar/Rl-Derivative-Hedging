"""Command-line orchestration for the Phase 4 evaluation pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os

from stable_baselines3 import SAC

from src.evaluation.backtest import BSDeltaPolicy, RLAgentPolicy, ZeroHedgePolicy, run_backtest, run_robustness_sweep, save_results
from src.evaluation.metrics import compare_metrics
from src.evaluation.plots import plot_cost_sensitivity, plot_episode_replay, plot_hedge_ratio_over_time, plot_metric_comparison, plot_pnl_distribution, plot_robustness_heatmap, save_figure
from src.training.env_factory import load_env_for_inference
from src.training.hyperparams import load_config

logger = logging.getLogger(__name__)


def _print_metrics(metrics: dict[str, dict[str, float]]) -> None:
    """Print a compact metrics summary."""
    print("Agent        | Mean P&L   | Std P&L   | CVaR@95%  | Mean Cost")
    print("-" * 68)
    for agent, values in metrics.items():
        if agent != "improvement":
            print(f"{agent:12} | {values['mean_pnl']:10.4f} | {values['std_pnl']:9.4f} | {values['cvar_95']:10.4f} | {values['mean_cost']:9.4f}")
    if "improvement" in metrics:
        print("RL improvement vs BS Delta: " + ", ".join(f"{key}={value:.2f}%" for key, value in metrics["improvement"].items()))


def evaluate(checkpoint_path: str, vecnorm_path: str, config_path: str = "configs/training.yaml", n_episodes: int = 1000, output_dir: str | None = None, run_robustness: bool = True) -> None:
    """Evaluate SAC alongside BS-delta and zero-hedge baselines."""
    config = load_config(config_path)
    output_dir = output_dir or os.path.join(config.results_dir, f"evaluation_{datetime.now():%Y%m%d_%H%M%S}")
    model = SAC.load(checkpoint_path)
    vec_normalize = load_env_for_inference(config.eval_environment, vecnorm_path, seed=config.run.seed)
    try:
        policies = [RLAgentPolicy(model, vec_normalize), BSDeltaPolicy(), ZeroHedgePolicy()]
        results = run_backtest(policies, config.eval_environment, n_episodes)
        metrics = compare_metrics(results)
        _print_metrics(metrics)
        save_results(results, metrics, output_dir)
        plots_dir = os.path.join(output_dir, "plots")
        save_figure(plot_pnl_distribution(results), plots_dir, "pnl_distribution")
        save_figure(plot_hedge_ratio_over_time(results), plots_dir, "hedge_ratio_over_time")
        save_figure(plot_metric_comparison(metrics), plots_dir, "metric_comparison")
        save_figure(plot_episode_replay(results, episode_id=0), plots_dir, "episode_replay")
        if run_robustness:
            robustness_df = run_robustness_sweep(policies, config.eval_environment, [0.0, 0.0005, 0.001, 0.002, 0.005, 0.010], [0.10, 0.15, 0.20, 0.25, 0.30, 0.40], n_episodes=200)
            robustness_df.to_csv(os.path.join(output_dir, "robustness.csv"), index=False)
            save_figure(plot_cost_sensitivity(robustness_df), plots_dir, "cost_sensitivity")
            for agent in robustness_df["agent_type"].unique():
                save_figure(plot_robustness_heatmap(robustness_df, agent_type=agent), plots_dir, f"robustness_heatmap_{agent}")
        logger.info("Evaluation artefacts saved to %s", output_dir)
    finally:
        vec_normalize.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RL hedging checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to the SAC model checkpoint (.zip)")
    parser.add_argument("--vecnorm", required=True, help="Path to the matching VecNormalize .pkl file")
    parser.add_argument("--config", default="configs/training.yaml", help="Training configuration path")
    parser.add_argument("--n-episodes", type=int, default=1000, help="Episodes per policy")
    parser.add_argument("--output-dir", default=None, help="Directory for evaluation artefacts")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip the robustness sweep")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.vecnorm, args.config, args.n_episodes, args.output_dir, not args.skip_robustness)


if __name__ == "__main__":
    main()
