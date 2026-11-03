"""Train + evaluate k independent seeds, then report across-seed statistics.

NOT executed in the authoring environment (it needs torch / SB3 and hours of
compute). It only composes the project's documented CLIs:
    python -m src.training.train --config <cfg>
    python -m src.evaluation.evaluate --checkpoint ... --vecnorm ... --n-episodes N
and then calls ``aggregate_run_metrics`` (unit-tested separately).

Usage:  python scripts/multiseed.py --seeds 0 1 2 3 4 --n-episodes 1000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from src.evaluation.uncertainty import aggregate_run_metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-episodes", type=int, default=1000)
    ap.add_argument("--results-root", default="results")
    args = ap.parse_args()
    if len(args.seeds) < 2:
        ap.error("need >= 2 seeds for across-run statistics")

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    metrics_files: list[Path] = []
    for seed in args.seeds:
        cfg = {**base, "run": {**base["run"], "seed": seed, "name": f"{base['run']['name']}_s{seed}"}}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            yaml.safe_dump(cfg, fh)
        subprocess.run([sys.executable, "-m", "src.training.train", "--config", fh.name], check=True)
        ckpt_dir = max(Path(cfg.get("checkpoints_dir", "checkpoints")).glob(f"{cfg['run']['name']}_*"))
        out = Path(args.results_root) / f"{cfg['run']['name']}_eval"
        subprocess.run(
            [sys.executable, "-m", "src.evaluation.evaluate",
             "--checkpoint", str(ckpt_dir / "best_model.zip"),
             "--vecnorm", str(ckpt_dir / "best_vecnorm.pkl"),
             "--n-episodes", str(args.n_episodes), "--output-dir", str(out)],
            check=True,
        )
        metrics_files.append(out / "metrics.json")
    print(aggregate_run_metrics(metrics_files).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
