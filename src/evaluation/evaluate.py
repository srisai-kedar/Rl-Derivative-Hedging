"""Command-line orchestration for the Phase 4 evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import replace
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
from src.training.observation import (
    model_observation_dimension,
    validate_observation_dimensions,
    vecnormalize_observation_dimension,
    vecnormalize_stats_dimension,
)
from src.utils.logging import configure_run_logging
from src.utils.manifest import write_manifest

logger = logging.getLogger(__name__)

DEFAULT_KAPPA_GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.010]
DEFAULT_SIGMA_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
DEFAULT_ROBUSTNESS_EPISODES = 200
DEFAULT_SEED_OFFSET = 0
DEFAULT_N_BOOT = 10_000


def _print_metrics_summary(metrics: dict) -> None:
    """Print evaluation results summary to stdout."""
    print("EVALUATION RESULTS")
    print("Agent        | Mean P&L   | Std P&L   | CVaR@95%  | Mean Cost")
    print("-" * 68)
    for agent, values in metrics.items():
        if agent not in {"improvement", "confidence_intervals"}:
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
    confidence_intervals = metrics.get("confidence_intervals")
    if confidence_intervals:
        print("95% bootstrap confidence intervals (RL / BS / paired difference):")
        for metric, report in confidence_intervals.items():
            primary = report["primary"]
            baseline = report["baseline"]
            difference = report["diff"]
            print(
                f"{metric:12} | RL [{primary['lo']:.4f}, {primary['hi']:.4f}] | "
                f"BS [{baseline['lo']:.4f}, {baseline['hi']:.4f}] | "
                f"diff [{difference['lo']:.4f}, {difference['hi']:.4f}]"
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
    allow_manifest_mismatch: bool = False,
    require_manifest: bool = False,
    expected_sha256: Mapping[str, str] | str | None = None,
    seed_offset: int = DEFAULT_SEED_OFFSET,
    robustness_episodes: int = DEFAULT_ROBUSTNESS_EPISODES,
    kappa_grid: list[float] | None = None,
    sigma_grid: list[float] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    market_model: str | None = None,
    market_model_kwargs: dict[str, float] | None = None,
) -> None:
    """Evaluate SAC alongside BS-delta and zero-hedge baselines."""
    from src.evaluation.backtest import HedgingPolicy

    config = load_config(config_path)
    configure_run_logging(
        f"{getattr(config.run, 'name', 'run')}-seed{getattr(config.run, 'seed', 0)}"
    )
    if market_model is not None or market_model_kwargs is not None:
        environment = config.eval_environment
        config = replace(
            config,
            eval_environment=replace(
                environment,
                market_model=market_model or environment.market_model,
                market_model_kwargs=(
                    market_model_kwargs
                    if market_model_kwargs is not None
                    else environment.market_model_kwargs
                ),
            ),
        )
    kappa_grid = DEFAULT_KAPPA_GRID if kappa_grid is None else kappa_grid
    sigma_grid = DEFAULT_SIGMA_GRID if sigma_grid is None else sigma_grid
    output_dir = output_dir or os.path.join(
        config.results_dir, f"evaluation_{datetime.now():%Y%m%d_%H%M%S}"
    )

    vec_normalize = load_env_for_inference(
        config.eval_environment,
        vecnorm_path,
        seed=config.run.seed,
        allow_unverified=allow_unverified,
        allow_manifest_mismatch=allow_manifest_mismatch,
        checkpoint_path=checkpoint_path,
        require_manifest=require_manifest,
        expected_sha256=expected_sha256,
    )
    try:
        model = SAC.load(checkpoint_path)
        validate_observation_dimensions(
            env_dimension=vecnormalize_observation_dimension(vec_normalize),
            model_dimension=model_observation_dimension(model),
            vecnorm_dimension=vecnormalize_stats_dimension(vec_normalize),
            observation_mode=config.eval_environment.observation_mode,
        )
        policies: list[HedgingPolicy] = [
            RLAgentPolicy(
                model,
                vec_normalize,
                observation_mode=config.eval_environment.observation_mode,
            ),
            BSDeltaPolicy(),
            ZeroHedgePolicy(),
        ]
        results = run_backtest(
            policies,
            config.eval_environment,
            n_episodes,
            batch_size=batch_size,
            legacy=legacy,
            seed_offset=seed_offset,
        )
        metrics: dict = compare_metrics(results)
        from src.evaluation.uncertainty import compare_with_ci

        metrics["confidence_intervals"] = compare_with_ci(
            results.episode_df,
            n_boot=n_boot,
            seed=seed_offset,
        )
        _print_metrics_summary(metrics)
        save_results(results, metrics, output_dir)
        plots_dir = os.path.join(output_dir, "plots")
        save_figure(plot_pnl_distribution(results), plots_dir, "pnl_distribution")
        save_figure(
            plot_hedge_ratio_over_time(results), plots_dir, "hedge_ratio_over_time"
        )
        save_figure(plot_metric_comparison(metrics), plots_dir, "metric_comparison")
        save_figure(
            plot_episode_replay(results, episode_id=0), plots_dir, "episode_replay"
        )
        if run_robustness:
            robustness_df = run_robustness_sweep(
                policies,
                config.eval_environment,
                kappa_grid,
                sigma_grid,
                n_episodes=robustness_episodes,
                batch_size=batch_size,
                legacy=legacy,
                seed_offset=seed_offset,
            )
            robustness_df.to_csv(
                os.path.join(output_dir, "robustness.csv"), index=False
            )
            save_figure(
                plot_cost_sensitivity(robustness_df), plots_dir, "cost_sensitivity"
            )
            for agent in robustness_df["agent_type"].unique():
                save_figure(
                    plot_robustness_heatmap(robustness_df, agent_type=agent),
                    plots_dir,
                    f"robustness_heatmap_{agent}",
                )

        # Save evaluation run manifest
        write_manifest(
            output_dir=output_dir,
            run_type="evaluation",
            config=config,
            seed=config.run.seed,
            checkpoint_files=[checkpoint_path, vecnorm_path],
            extra_metadata={
                "n_episodes": n_episodes,
                "output_dir": output_dir,
                "seed_offset": seed_offset,
                "robustness_episodes": robustness_episodes,
                "kappa_grid": kappa_grid,
                "sigma_grid": sigma_grid,
                "n_boot": n_boot,
                "market_model": config.eval_environment.market_model,
                "market_model_kwargs": config.eval_environment.market_model_kwargs,
            },
        )
        logger.info("Evaluation artefacts and run manifest saved to %s", output_dir)
    finally:
        vec_normalize.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RL hedging checkpoint")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the SAC model checkpoint (.zip)"
    )
    parser.add_argument(
        "--vecnorm", required=True, help="Path to the matching VecNormalize .pkl file"
    )
    parser.add_argument(
        "--config", default="configs/training.yaml", help="Training configuration path"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=1000, help="Episodes per policy"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Directory for evaluation artefacts"
    )
    parser.add_argument(
        "--skip-robustness", action="store_true", help="Skip the robustness sweep"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for vectorised backtest"
    )
    parser.add_argument(
        "--legacy-backtest",
        action="store_true",
        help="Use legacy sequential backtester",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow loading checkpoint even if manifest hash check fails",
    )
    parser.add_argument(
        "--allow-manifest-mismatch",
        action="store_true",
        help="Warn instead of raising when manifest environment settings differ",
    )
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="Fail if run_manifest.json or required checkpoint hashes are missing",
    )
    parser.add_argument(
        "--expected-sha256",
        action="append",
        default=[],
        metavar="FILE=HASH",
        help="Out-of-band SHA-256 anchor; repeat for model and VecNormalize files",
    )
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET)
    parser.add_argument(
        "--robustness-episodes", type=int, default=DEFAULT_ROBUSTNESS_EPISODES
    )
    parser.add_argument(
        "--kappa-grid", type=float, nargs="+", default=DEFAULT_KAPPA_GRID
    )
    parser.add_argument(
        "--sigma-grid", type=float, nargs="+", default=DEFAULT_SIGMA_GRID
    )
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument(
        "--market-model", choices=["gbm", "heston", "merton"], default=None
    )
    parser.add_argument(
        "--market-model-kwargs",
        default=None,
        help="JSON object of model-specific market parameters",
    )
    args = parser.parse_args()
    expected_sha256: dict[str, str] = {}
    for entry in args.expected_sha256:
        if "=" not in entry:
            parser.error("--expected-sha256 must use FILE=HASH format")
        file_name, file_hash = entry.split("=", 1)
        expected_sha256[file_name] = file_hash
    market_model_kwargs = None
    if args.market_model_kwargs is not None:
        try:
            market_model_kwargs = json.loads(args.market_model_kwargs)
        except json.JSONDecodeError as error:
            parser.error(f"--market-model-kwargs must be valid JSON: {error}")
        if not isinstance(market_model_kwargs, dict):
            parser.error("--market-model-kwargs must decode to a JSON object")
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
        allow_manifest_mismatch=args.allow_manifest_mismatch,
        require_manifest=args.require_manifest,
        expected_sha256=expected_sha256 or None,
        seed_offset=args.seed_offset,
        robustness_episodes=args.robustness_episodes,
        kappa_grid=args.kappa_grid,
        sigma_grid=args.sigma_grid,
        n_boot=args.n_boot,
        market_model=args.market_model,
        market_model_kwargs=market_model_kwargs,
    )


if __name__ == "__main__":
    main()
