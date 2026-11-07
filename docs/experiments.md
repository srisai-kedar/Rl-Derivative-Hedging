# Experiments

The `experiments/` directory contains declarative Part B plans:

- `baseline_multiseed.yaml`
- `long_training.yaml`
- `risk_aversion_sweep.yaml`
- `hpo_best_retrain.yaml`
- `obs_ablation.yaml`
- `market_robustness.yaml`

Run a plan without measuring anything:

```bash
python -m experiments.runner --experiment experiments/baseline_multiseed.yaml --dry-run
```

Run a tiny labelled smoke check:

```bash
python -m experiments.runner --experiment experiments/baseline_multiseed.yaml --smoke
```

Resume completed seeds using both evaluation `metrics.json` and
`run_manifest.json`:

```bash
python -m experiments.runner --experiment experiments/baseline_multiseed.yaml --resume
```

Generate reports only from saved experiment outputs:

```bash
python -m experiments.report \
  --root results/experiments/baseline_multiseed \
  --output report.md --json-output report.json
```

Missing metrics, confidence intervals, or manifests are reported as `NOT RUN`.
Smoke output is explicitly marked `SMOKE: not a result`.
