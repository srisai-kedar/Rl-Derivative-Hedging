"""Tests for the declarative experiments and multiseed orchestration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load_module("experiments/report.py", "experiment_report")
multiseed = _load_module("scripts/multiseed.py", "multiseed_script")


def test_multiseed_config_updates_are_nested_without_mutating_base():
    base = {"run": {"seed": 1, "name": "base"}, "paths": {"results_dir": "results"}}
    updated = multiseed._deep_update(base, {"run": {"seed": 2}})
    assert updated == {
        "run": {"seed": 2, "name": "base"},
        "paths": {"results_dir": "results"},
    }
    assert base["run"]["seed"] == 1


def test_resume_requires_metrics_and_manifest(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "evaluation").mkdir(parents=True)
    assert not multiseed._is_complete(run_dir)
    (run_dir / "evaluation" / "metrics.json").write_text("{}", encoding="utf-8")
    assert not multiseed._is_complete(run_dir)
    (run_dir / "evaluation" / "run_manifest.json").write_text("{}", encoding="utf-8")
    assert multiseed._is_complete(run_dir)


def test_report_reads_only_outputs_and_marks_missing_ci(tmp_path):
    run_dir = tmp_path / "seed_0" / "evaluation"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"rl_agent": {"mean_pnl": 1.25, "std_pnl": 0.5}}), encoding="utf-8"
    )
    rows = report.collect_report_rows(tmp_path)
    assert rows[0]["mean_pnl"] == 1.25
    assert rows[0]["mean_pnl_ci"] == "NOT RUN"
    assert "NOT RUN" in report.render_markdown(rows)


def test_report_marks_empty_input_not_run(tmp_path):
    rows = report.collect_report_rows(tmp_path / "missing")
    assert rows == [
        {
            "run": "NOT RUN",
            "mean_pnl": "NOT RUN",
            "std_pnl": "NOT RUN",
            "mean_pnl_ci": "NOT RUN",
            "rl_bs_diff_ci": "NOT RUN",
            "manifest": False,
        }
    ]


def test_multiseed_passes_per_run_config_and_exact_checkpoint(monkeypatch, tmp_path):
    config = tmp_path / "base.yaml"
    config.write_text(
        yaml.safe_dump(
            {"run": {"name": "demo", "seed": 42}, "paths": {"checkpoints_dir": "old"}}
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def fake_run(command, check):
        commands.append(command)
        if "src.training.train" in command:
            info = Path(command[command.index("--run-info") + 1])
            exact = info.parent / "exact" / "model.zip"
            vecnorm = info.parent / "exact" / "stats.pkl"
            info.parent.mkdir(parents=True, exist_ok=True)
            info.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "checkpoint_path": str(exact),
                        "vecnorm_path": str(vecnorm),
                    }
                ),
                encoding="utf-8",
            )
        else:
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "metrics.json").write_text(
                json.dumps({"rl_agent": {"mean_pnl": 1, "std_pnl": 1}}),
                encoding="utf-8",
            )
            (output / "run_manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(multiseed.subprocess, "run", fake_run)
    assert multiseed.run_multiseed(str(config), [0, 1], 2, str(tmp_path / "out")) == 0
    assert len(commands) == 4
    train_config = commands[0][commands[0].index("--config") + 1]
    evaluate_config = commands[1][commands[1].index("--config") + 1]
    assert train_config == evaluate_config
    checkpoint = Path(commands[1][commands[1].index("--checkpoint") + 1])
    assert checkpoint.parent.name == "exact"
    assert checkpoint.name == "model.zip"
