"""Command-line orchestration for the Phase 4 evaluation pipeline."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime

from stable_baselines3 import SAC

from src.evaluation.backtest import (
    BSDeltaPolicy,
    RLAgentPolicy,
    ZeroHedgePolicy,
    run_backtest,
    run_robustness_sweep,
    save_results,
)
from src.evaluation.metrics import compare_metrics
from src.evaluation.plots import (
    plot_cost_sensitivity,
    plot_episode_replay,
    plot_hedge_ratio_over_time,
    plot_metric_comparison,
    plot_pnl_distribution,
    plot_robustness_heatmap,
    save_figure,
)
from src.training.env_factory import load_env_for_inference
from src.training.hyperparams import load_config
from src.utils.manifest import verify_checkpoint_integrity, write_manifest

logger = logging.getLogger(__name__)


def _print_metrics_summary(metrics: dict[str, dict[str, float]]) -> None:
    """Print evaluation results summary to stdout."""
    print("EVALUATION RESULTS")
    print("Agent        | Mean P&L   | Std P&L   | CVaR@95%  | Mean Cost")
    print("-" * 68)
    for agent, values in metrics.items():
        if agent != "improvement":
            print(
                f"{agent:12} | {values['mean_pnl']:10.4f} | "
                f"{values['std_pnl']:9.4f} | {values['cvar_95']:10.4f} | "
                f"{values['mean_cost']:9.4f}"
            )
    if "improvement" in metrics:
        print(
            "RL improvement vs BS Delta: "
            + ", ".join(
                f"{key}={value:.2f}%" for key, value in metrics["improvement"].items()
            )
        )


def evaluate(
    checkpoint_path: str,
    vecnorm_path: str,
    config_path: str = "configs/training.yaml",
    n_episodes: int = 1000,
    output_dir: str | None = None,
    run_robustness: bool = True,
    batch_size: int = 128,
    legacy: bool = False,
    allow_unverified: bool = False,
) -> None:
    """Evaluate SAC alongside BS-delta and zero-hedge baselines."""
    from src.evaluation.backtest import HedgingPolicy

    config = load_config(config_path)
    output_dir = output_dir or os.path.join(config.results_dir, f"evaluation_{datetime.now():%Y%m%d_%H%M%S}")

    # Check cryptographic integrity if manifest is present
    verify_checkpoint_integrity(
        checkpoint_path=checkpoint_path,
        vecnorm_path=vecnorm_path,
        allow_unverified=allow_unverified,
    )

    model = SAC.load(checkpoint_path)
    vec_normalize = load_env_for_inference(
        config.eval_environment,
        vecnorm_path,
        seed=config.run.seed,
        allow_unverified=allow_unverified,
    )
    try:
        policies: list[HedgingPolicy] = [
            RLAgentPolicy(model, vec_normalize),
            BSDeltaPolicy(),
            ZeroHedgePolicy(),
        ]
        results = run_backtest(
            policies,
            config.eval_environment,
            n_episodes,
            batch_size=batch_size,
            legacy=legacy,
        )
        metrics = compare_metrics(results)
        _print_metrics_summary(metrics)
        save_results(results, metrics, output_dir)
        plots_dir = os.path.join(output_dir, "plots")
        save_figure(plot_pnl_distribution(results), plots_dir, "pnl_distribution")
        save_figure(plot_hedge_ratio_over_time(results), plots_dir, "hedge_ratio_over_time")
        save_figure(plot_metric_comparison(metrics), plots_dir, "metric_comparison")
        save_figure(plot_episode_replay(results, episode_id=0), plots_dir, "episode_replay")
        if run_robustness:
            robustness_df = run_robustness_sweep(
                policies,
                config.eval_environment,
                [0.0, 0.0005, 0.001, 0.002, 0.005, 0.010],
                [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
                n_episodes=200,
                batch_size=batch_size,
                legacy=legacy,
            )
            robustness_df.to_csv(os.path.join(output_dir, "robustness.csv"), index=False)
            save_figure(plot_cost_sensitivity(robustness_df), plots_dir, "cost_sensitivity")
            for agent in robustness_df["agent_type"].unique():
                save_figure(plot_robustness_heatmap(robustness_df, agent_type=agent), plots_dir, f"robustness_heatmap_{agent}")

        # Save evaluation run manifest
        write_manifest(
            output_dir=output_dir,
            run_type="evaluation",
            config=config,
            seed=config.run.seed,
            checkpoint_files=[checkpoint_path, vecnorm_path],
            extra_metadata={"n_episodes": n_episodes, "output_dir": output_dir},
        )
        logger.info("Evaluation artefacts and run manifest saved to %s", output_dir)
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
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for vectorised backtest")
    parser.add_argument("--legacy-backtest", action="store_true", help="Use legacy sequential backtester")
    parser.add_argument("--allow-unverified", action="store_true", help="Allow loading checkpoint even if manifest hash check fails")
    args = parser.parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        vecnorm_path=args.vecnorm,
        config_path=args.config,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
        run_robustness=not args.skip_robustness,
        batch_size=args.batch_size,
        legacy=args.legacy_backtest,
        allow_unverified=args.allow_unverified,
    )


if __name__ == "__main__":
    main()
