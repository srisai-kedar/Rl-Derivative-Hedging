"""Tiny end-to-end train/evaluate/manifest verification smoke test."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.slow
def test_tiny_train_evaluate_manifest_smoke(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "experiment_runner", Path("experiments/runner.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "smoke"
    result = module.run_experiment(
        "experiments/baseline_multiseed.yaml",
        smoke=True,
        output_root=str(output),
    )
    assert result == 0
    assert (output / "SMOKE_NOT_A_RESULT.txt").read_text(
        encoding="utf-8"
    ) == "SMOKE: not a result\n"
    for seed in (0, 1):
        evaluation = output / f"baseline_multiseed_s{seed}" / "evaluation"
        metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
        assert "confidence_intervals" in metrics
        assert (evaluation / "run_manifest.json").is_file()
