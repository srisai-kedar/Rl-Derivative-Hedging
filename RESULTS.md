# Results

All main comparison results were produced with:

```bash
python -m src.evaluation.evaluate \
    --checkpoint checkpoints/sac_hedging_20260721_002714/best_model.zip \
    --vecnorm checkpoints/sac_hedging_20260721_002714/best_vecnorm.pkl \
    --n-episodes 1000 \
    --output-dir results/phase6_eval_1000
```

Training config: `configs/training.yaml`  
Checkpoint: `checkpoints/sac_hedging_20260721_002714/best_model.zip`  
Training duration represented by checkpoint: 300,000 environment steps  
Evaluation episodes: 1,000 shared seeds per policy  
Robustness sweep: 200 episodes per `(kappa, sigma)` pair

## Main Comparison

| Metric | RL Agent | BS Delta | Zero Hedge | RL vs BS |
|--------|----------|----------|------------|----------|
| P&L Std Dev | 2.563 | 0.449 | 4.552 | -470.7% |
| Mean P&L | 0.307 | 0.102 | -0.122 | +199.6% |
| CVaR @ 95% | -6.815 | -0.939 | -13.156 | -625.6% |
| Mean Tx Cost | 0.192 | 0.224 | 0.000 | +14.4% |
| Cost Efficiency | 13.379 | 2.007 | Inf | -566.5% |

## Robustness: Volatility Regime

Testing at fixed `kappa=0.001`, varying `sigma`. The model was trained with
volatility randomisation in `[0.10, 0.40]`.

| Sigma | RL Std | BS Std | Improvement |
|-------|--------|--------|-------------|
| 0.10 | 1.659 | 0.277 | -499.0% |
| 0.15 | 2.378 | 0.375 | -533.3% |
| 0.20 | 2.704 | 0.485 | -457.2% |
| 0.25 | 3.288 | 0.600 | -448.2% |
| 0.30 | 3.409 | 0.713 | -378.0% |
| 0.40 | 6.547 | 0.939 | -597.5% |

## Robustness: Transaction Cost Regime

Testing at fixed `sigma=0.20`, varying `kappa`.

| Kappa | RL Std | BS Std | Improvement |
|-------|--------|--------|-------------|
| 0.000 | 2.689 | 0.472 | -469.9% |
| 0.001 | 2.704 | 0.485 | -457.2% |
| 0.002 | 2.722 | 0.506 | -438.4% |
| 0.005 | 2.792 | 0.599 | -366.0% |
| 0.010 | 2.959 | 0.822 | -260.1% |

## Interpretation

The SAC checkpoint trades less than the Black-Scholes delta baseline and reduces
mean transaction cost from 0.224 to 0.192. That is a real cost improvement, but
the policy currently accepts much larger hedge error. Its P&L standard deviation
is 2.563 versus 0.449 for BS delta, and its 95% CVaR is materially worse.

The robustness sweep shows the same pattern across volatility and transaction
cost regimes: BS delta remains a much stronger risk hedge, while the RL policy
only has a cost advantage. The gap narrows at high transaction costs, but not
enough for the current checkpoint to outperform BS on P&L variance.

The honest conclusion is that the pipeline is working, but this checkpoint is
not yet a better hedging policy. Phase 6 therefore documents a reproducible
baseline for future HPO or longer training rather than claiming a false win.

## Reproduction

```bash
# Train
python -m src.training.train --config configs/training.yaml

# Evaluate
python -m src.evaluation.evaluate \
    --checkpoint checkpoints/<your-run>/best_model.zip \
    --vecnorm checkpoints/<your-run>/best_vecnorm.pkl \
    --n-episodes 1000
```

Evaluation outputs are written to `results/evaluation_<timestamp>/` unless
`--output-dir` is provided.

## Experiment checkpoint (2026-11-06)

This dated section is appended to preserve the generated E0, E1, and E6
experiment artifacts. Existing RESULTS.md numbers above are unchanged. The
paste-ready values and source paths are in
`docs/experiments/E7_results_tables.md`; the Word-report figure/table list is
in `docs/experiments/E7_word_figures_tables.md`.

### Completed experiments

- E0 checkpoint reevaluation completed with both batched and legacy
  backtesters. Sources: `results/E0_batched_1000/` and
  `results/E0_legacy_1000/`.
- E1 five-seed, 300,000-step baseline completed for seeds 0--4. Source:
  `results/experiments/baseline_multiseed_e1/`.
- E6 Heston/Merton evaluation completed for the five E1 checkpoints. No
  retraining was performed. Source:
  `results/experiments/e6_market_robustness/`.

### NOT RUN experiments

- E2: five-seed 2,000,000-step training — NOT RUN. No long-budget result or
  plateau conclusion is supported.
- E3: multi-seed risk-aversion sweep — NOT RUN. No risk-aversion effect is
  supported.
- E4: fixed HPO and best-configuration retraining — NOT RUN. No HPO selection-
  bias estimate is supported.
- E5: observation-mode ablation — NOT RUN. No conclusion about BS-delta or
  sigma leakage is supported.

These NOT RUN statuses remain limitations for the report.
