"""Render experiment reports from experiment output files only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def collect_report_rows(root: str | Path) -> list[dict[str, Any]]:
    """Read metrics and manifests below root; never infer results from configs."""
    rows: list[dict[str, Any]] = []
    root_path = Path(root)
    metric_paths = sorted(root_path.rglob("metrics.json")) if root_path.exists() else []
    for metrics_path in metric_paths:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ci = metrics.get("confidence_intervals", {}).get("mean_pnl", {})
        rl = metrics.get("rl_agent", {})
        rows.append(
            {
                "run": str(metrics_path.parent.relative_to(root_path)),
                "mean_pnl": rl.get("mean_pnl", "NOT RUN"),
                "std_pnl": rl.get("std_pnl", "NOT RUN"),
                "mean_pnl_ci": _interval(ci.get("primary")),
                "rl_bs_diff_ci": _interval(ci.get("diff")),
                "manifest": (metrics_path.parent / "run_manifest.json").is_file(),
            }
        )
    if not rows:
        return [
            {
                "run": "NOT RUN",
                "mean_pnl": "NOT RUN",
                "std_pnl": "NOT RUN",
                "mean_pnl_ci": "NOT RUN",
                "rl_bs_diff_ci": "NOT RUN",
                "manifest": False,
            }
        ]
    return rows


def _interval(interval: dict[str, Any] | None) -> str:
    if not interval or "lo" not in interval or "hi" not in interval:
        return "NOT RUN"
    return f"[{interval['lo']:.4f}, {interval['hi']:.4f}]"


def render_markdown(rows: list[dict[str, Any]], label: str | None = None) -> str:
    lines = ["# Experiment report", ""]
    if label:
        lines.extend([label, ""])
    lines.extend(
        [
            "| Run | Mean P&L | Std P&L | Mean P&L 95% CI | RL-BS diff 95% CI | Manifest |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['mean_pnl']} | {row['std_pnl']} | "
            f"{row['mean_pnl_ci']} | {row['rl_bs_diff_ci']} | "
            f"{'present' if row['manifest'] else 'NOT RUN'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    rows = collect_report_rows(args.root)
    label = (
        "SMOKE: not a result"
        if (Path(args.root) / "SMOKE_NOT_A_RESULT.txt").is_file()
        else None
    )
    markdown = render_markdown(rows, label=label)
    print(markdown, end="")
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
