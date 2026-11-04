"""Train and evaluate independent seeds without leaking per-run configuration.

Every run gets ``<results-root>/<run-name>_s<seed>/config.yaml`` and all
training/evaluation commands use that file. Training writes a run-info JSON
file containing the exact checkpoint paths; this script never discovers
checkpoints by globbing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.uncertainty import aggregate_run_metrics


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _commands(
    config_path: Path,
    run_info_path: Path,
    checkpoint_path: Path,
    vecnorm_path: Path,
    evaluation_config: Path,
    evaluation_dir: Path,
    n_episodes: int,
    skip_robustness: bool,
) -> tuple[list[str], list[str]]:
    train = [
        sys.executable,
        "-m",
        "src.training.train",
        "--config",
        str(config_path),
        "--run-info",
        str(run_info_path),
    ]
    evaluate = [
        sys.executable,
        "-m",
        "src.evaluation.evaluate",
        "--checkpoint",
        str(checkpoint_path),
        "--vecnorm",
        str(vecnorm_path),
        "--config",
        str(evaluation_config),
        "--n-episodes",
        str(n_episodes),
        "--output-dir",
        str(evaluation_dir),
    ]
    if skip_robustness:
        evaluate.append("--skip-robustness")
    return train, evaluate


def _is_complete(run_dir: Path) -> bool:
    return (run_dir / "evaluation" / "metrics.json").is_file() and (
        run_dir / "evaluation" / "run_manifest.json"
    ).is_file()


def run_multiseed(
    config_path: str,
    seeds: list[int],
    n_episodes: int,
    results_root: str,
    *,
    resume: bool = False,
    dry_run: bool = False,
    name: str | None = None,
    skip_robustness: bool = False,
) -> int:
    """Run the per-seed workflow and aggregate completed metrics."""
    base = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    base_name = name or base["run"]["name"]
    root = Path(results_root)
    metrics_files: list[Path] = []

    for seed in seeds:
        run_dir = root / f"{base_name}_s{seed}"
        evaluation_dir = run_dir / "evaluation"
        config_file = run_dir / "config.yaml"
        run_info = run_dir / "training_run.json"
        if resume and _is_complete(run_dir):
            print(f"RESUME: skipping completed seed {seed}: {evaluation_dir}")
            metrics_files.append(evaluation_dir / "metrics.json")
            continue

        run_config = _deep_update(
            base,
            {
                "run": {"seed": seed, "name": f"{base_name}_s{seed}"},
                "paths": {
                    "checkpoints_dir": str(run_dir / "checkpoints"),
                    "results_dir": str(run_dir / "training_results"),
                },
            },
        )
        checkpoint_path = run_dir / "checkpoints" / "final_model.zip"
        vecnorm_path = run_dir / "checkpoints" / "final_vecnorm.pkl"
        train_cmd, evaluate_cmd = _commands(
            config_file,
            run_info,
            checkpoint_path,
            vecnorm_path,
            config_file,
            evaluation_dir,
            n_episodes,
            skip_robustness,
        )
        if dry_run:
            print(f"DRY-RUN seed {seed}: would write {config_file}")
            print("  " + " ".join(train_cmd))
            print("  " + " ".join(evaluate_cmd))
            print(
                "  MEASURED: nothing; no training, evaluation, or aggregation performed"
            )
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        config_file.write_text(
            yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8"
        )
        subprocess.run(train_cmd, check=True)
        info = json.loads(run_info.read_text(encoding="utf-8"))
        if info.get("status") != "completed":
            print(f"NOT RUN: seed {seed} training status is {info.get('status')!r}")
            continue
        evaluate_cmd[4] = str(info["checkpoint_path"])
        evaluate_cmd[6] = str(info["vecnorm_path"])
        subprocess.run(evaluate_cmd, check=True)
        if _is_complete(run_dir):
            metrics_files.append(evaluation_dir / "metrics.json")
        else:
            print(f"NOT RUN: seed {seed} has incomplete evaluation outputs")

    if dry_run:
        print("DRY-RUN SUMMARY: MEASURED NOTHING")
        return 0
    if not metrics_files:
        print("NOT RUN: no completed experiment metrics available")
        return 0
    print(aggregate_run_metrics(metrics_files).round(4).to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-episodes", type=int, default=1000)
    ap.add_argument("--results-root", default="results/multiseed")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--skip-robustness", action="store_true")
    args = ap.parse_args()
    if len(args.seeds) < 2:
        ap.error("need >= 2 seeds for across-seed statistics")
    return run_multiseed(
        args.config,
        args.seeds,
        args.n_episodes,
        args.results_root,
        resume=args.resume,
        dry_run=args.dry_run,
        name=args.name,
        skip_robustness=args.skip_robustness,
    )


if __name__ == "__main__":
    raise SystemExit(main())
