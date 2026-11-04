"""Run a declarative experiment specification through the multiseed workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from scripts.multiseed import _deep_update, run_multiseed


def run_experiment(
    spec_path: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
    smoke: bool = False,
    output_root: str | None = None,
) -> int:
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    root = Path(output_root or spec["results_root"])
    base_path = Path(spec["base_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config = _deep_update(base, spec.get("config_overrides", {}))
    seeds = list(spec["seeds"])
    n_episodes = int(spec["n_episodes"])
    if smoke:
        seeds = seeds[:2] or [0, 1]
        n_episodes = 2
        config = _deep_update(
            config,
            {
                "run": {"total_timesteps": 100, "n_envs": 1, "progress_bar": False},
                "callbacks": {"eval_freq": 10_000, "save_freq": 10_000},
            },
        )
        print("SMOKE: not a result")

    config_file = root / "experiment_config.yaml"
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (root / "SMOKE_NOT_A_RESULT.txt").write_text(
            "SMOKE: not a result\n" if smoke else "", encoding="utf-8"
        )
    else:
        config_file = base_path

    return run_multiseed(
        str(config_file),
        seeds,
        n_episodes,
        str(root),
        resume=resume,
        dry_run=dry_run,
        name=spec["name"],
        skip_robustness=smoke,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Path to an experiments/*.yaml spec")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    return run_experiment(
        args.experiment,
        dry_run=args.dry_run,
        resume=args.resume,
        smoke=args.smoke,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
