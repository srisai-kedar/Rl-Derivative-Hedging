# E6: Market-model robustness

## Objective

E6 evaluates the five completed E1 SAC checkpoints under Heston and Merton
price dynamics. The purpose is to measure how a policy trained under GBM
degrades when the evaluation dynamics include stochastic volatility or jumps.
The Black-Scholes delta and zero-hedge policies are evaluated in the same
episodes as the RL policy for every run.

E2 was not run, so there are no E2 checkpoints in this experiment. No agent was
retrained under Heston or Merton; this is an evaluation-only robustness
experiment.

## Methodology

- Five E1 training seeds: 0, 1, 2, 3, and 4.
- 1,000 evaluation episodes per policy, with shared evaluation seeds 0--999.
- 200 episodes per robustness cell.
- The default robustness grid contains 6 transaction-cost values and 6
  volatility values, for 36 cells:
  - `kappa`: `0.0`, `0.0005`, `0.001`, `0.002`, `0.005`, `0.01`.
  - `sigma`: `0.10`, `0.15`, `0.20`, `0.25`, `0.30`, `0.40`.
- Bootstrap confidence intervals in each run use 10,000 resamples.
- Across-seed intervals below are Student-t 95% CIs with 5 independent seeds
  and 4 degrees of freedom.
- The GBM comparison uses the corresponding E1 GBM evaluation for the same
  training seed and evaluation seed block.

The cross-dynamics evaluation intentionally used
`--allow-manifest-mismatch`: the checkpoints are GBM-trained, while the
evaluation environment is Heston or Merton. Checkpoint file integrity was
still verified.

## E1 checkpoints

| Seed | Model checkpoint | VecNormalize statistics |
|---:|---|---|
| 0 | `results/experiments/baseline_multiseed_e1/baseline_multiseed_s0/checkpoints/baseline_multiseed_s0_20261104_220208/final_model.zip` | matching `final_vecnorm.pkl` |
| 1 | `results/experiments/baseline_multiseed_e1/baseline_multiseed_s1/checkpoints/baseline_multiseed_s1_20261104_222742/final_model.zip` | matching `final_vecnorm.pkl` |
| 2 | `results/experiments/baseline_multiseed_e1/baseline_multiseed_s2/checkpoints/baseline_multiseed_s2_20261104_225002/final_model.zip` | matching `final_vecnorm.pkl` |
| 3 | `results/experiments/baseline_multiseed_e1/baseline_multiseed_s3/checkpoints/baseline_multiseed_s3_20261105_101206/final_model.zip` | matching `final_vecnorm.pkl` |
| 4 | `results/experiments/baseline_multiseed_e1/baseline_multiseed_s4/checkpoints/baseline_multiseed_s4_20261105_103600/final_model.zip` | matching `final_vecnorm.pkl` |

Each checkpoint was evaluated under both dynamics. The ten output directories
are under `results/experiments/e6_market_robustness/e1_s{0..4}/{heston,merton}`;
each contains `metrics.json`, `robustness.csv`, and `run_manifest.json`.

## Market-model parameters

The parameters below are the environment defaults used by the generated
manifests. `market_model_kwargs` was `{}` in the command, so these defaults
were applied by `HedgingEnv`.

### Heston

- `v0 = sigma^2`.
- Variance mean-reversion `kappa = 2.0`.
- Long-run variance `theta = sigma^2`.
- Correlation `rho = -0.7`.
- Derived vol-of-vol `xi = min(0.3, sqrt(2*kappa*theta))`.
- Full-truncation Euler was used for the variance process.

### Merton

- Jump intensity `lambda = 1.0` jump/year.
- Mean log-jump size `mu_J = -0.05`.
- Log-jump standard deviation `sigma_J = 0.10`.
- The compensated drift preserves the simulator's risk-neutral martingale
  convention.

In both models, options were marked with Black-Scholes using the environment's
configured sigma, not Heston's instantaneous variance or a jump-implied
volatility.

## Main evaluation metrics

The following are actual absolute metrics averaged across the five E1 seeds;
the intervals are across-seed Student-t 95% CIs. The BS intervals are
degenerate because the same deterministic baseline was evaluated on the same
shared seed block in every run.

| Dynamics | Policy | Mean P&L | Std P&L | CVaR 95% | Mean cost |
|---|---|---:|---:|---:|---:|
| GBM | RL | 0.2300 [0.1386, 0.3214] | 2.6102 [2.3142, 2.9061] | -6.3165 [-7.1527, -5.4803] | 0.2095 [0.1752, 0.2437] |
| GBM | BS delta | 0.1025 [0.1025, 0.1025] | 0.4491 [0.4491, 0.4491] | -0.9392 [-0.9392, -0.9392] | 0.2237 [0.2237, 0.2237] |
| Heston | RL | 0.3487 [0.2640, 0.4333] | 2.6433 [2.3778, 2.9089] | -6.5625 [-7.3456, -5.7794] | 0.2050 [0.1707, 0.2392] |
| Heston | BS delta | 0.1013 [0.1013, 0.1013] | 0.6197 [0.6197, 0.6197] | -1.4420 [-1.4420, -1.4420] | 0.2241 [0.2241, 0.2241] |
| Merton | RL | -0.1291 [-0.2114, -0.0467] | 3.3080 [2.9929, 3.6230] | -9.5957 [-10.1121, -9.0792] | 0.2136 [0.1820, 0.2452] |
| Merton | BS delta | -0.2388 [-0.2388, -0.2388] | 1.6464 [1.6464, 1.6464] | -6.3193 [-6.3193, -6.3193] | 0.2220 [0.2220, 0.2220] |

## Degradation relative to GBM

Values are stress minus GBM, averaged across corresponding training seeds.
For std P&L, a positive value is degradation; for CVaR and mean P&L, a more
negative value is degradation. Intervals are across-seed Student-t 95% CIs.

| Dynamics | Policy | Std P&L delta | CVaR delta | Mean P&L delta | Mean-cost delta |
|---|---|---:|---:|---:|---:|
| Heston | RL | +0.0332 [-0.0514, 0.1178] | -0.2460 [-0.6043, 0.1123] | +0.1186 [+0.0943, +0.1430] | -0.0045 [-0.0088, -0.0002] |
| Heston | BS delta | +0.1706 | -0.5028 | -0.0011 | +0.0004 |
| Merton | RL | +0.6978 [+0.5913, +0.8043] | -3.2792 [-3.8647, -2.6936] | -0.3591 [-0.3992, -0.3189] | +0.0041 [-0.0017, +0.0100] |
| Merton | BS delta | +1.1973 | -5.3801 | -0.3412 | -0.0017 |

At the default robustness cell (`kappa=0.001`, `sigma=0.20`), the RL deltas
were:

| Dynamics | Std P&L delta | CVaR delta | Mean P&L delta | Mean-cost delta |
|---|---:|---:|---:|---:|
| Heston | -0.0943 [-0.2075, +0.0188] | -0.0300 [-0.5795, +0.5195] | +0.1557 [+0.0158, +0.2956] | -0.0084 [-0.0156, -0.0012] |
| Merton | +0.5670 [+0.5039, +0.6302] | -2.5869 [-3.2296, -1.9442] | -0.3529 [-0.4066, -0.2993] | +0.0036 [-0.0014, +0.0086] |

## 36-cell robustness sweep

Every Heston and Merton run produced all 36 cells in its
`robustness.csv`. The following tables report the actual five-seed average
std P&L for RL and BS delta in the two standard slices of that grid.

### Volatility slice: fixed `kappa=0.001`

| Sigma | Heston RL | Heston BS | Merton RL | Merton BS |
|---:|---:|---:|---:|---:|
| 0.10 | 1.3757 | 0.3582 | 2.5446 | 1.9980 |
| 0.15 | 1.9399 | 0.4868 | 2.7266 | 1.7733 |
| 0.20 | 2.5665 | 0.5680 | 3.2279 | 1.6185 |
| 0.25 | 2.9207 | 0.6556 | 3.5843 | 1.5181 |
| 0.30 | 3.6928 | 0.7485 | 4.2176 | 1.4512 |
| 0.40 | 5.3801 | 0.9445 | 5.8124 | 1.4482 |

### Transaction-cost slice: fixed `sigma=0.20`

| Kappa | Heston RL | Heston BS | Merton RL | Merton BS |
|---:|---:|---:|---:|---:|
| 0.0000 | 2.5453 | 0.5449 | 3.2049 | 1.6124 |
| 0.0005 | 2.5558 | 0.5557 | 3.2163 | 1.6151 |
| 0.0010 | 2.5665 | 0.5680 | 3.2279 | 1.6185 |
| 0.0020 | 2.5886 | 0.5963 | 3.2520 | 1.6268 |
| 0.0050 | 2.6602 | 0.7062 | 3.3310 | 1.6644 |
| 0.0100 | 2.7959 | 0.9401 | 3.4827 | 1.7663 |

Across all 36 cells, the five-seed average RL ranges were:

| Dynamics | Mean P&L range | Std P&L range | CVaR range | Mean-cost range |
|---|---:|---:|---:|---:|
| Heston | [-3.6747, 0.6361] | [1.3676, 5.5535] | [-17.0890, -2.8000] | [0.0000, 4.3016] |
| Merton | [-4.4004, 0.1434] | [2.5346, 6.0177] | [-19.2094, -8.2256] | [0.0000, 4.3426] |

The complete cell-level records, including all three policies, are preserved
in the ten `robustness.csv` files under the E6 result directory.

## Findings and limitations

- Heston evaluation did not show a statistically clear RL increase in main
  std P&L or deterioration in CVaR relative to GBM. RL mean P&L was higher in
  this evaluation, with its across-seed CI excluding zero.
- Merton evaluation produced clear RL degradation: higher std P&L, more
  negative CVaR, and lower mean P&L relative to GBM.
- These are zero-shot evaluations of GBM-trained agents. They do not measure
  the performance of agents retrained for Heston or Merton.
- E2 was not run, so no 2,000,000-step checkpoint was evaluated.
- Heston and Merton results use the environment's Black-Scholes marking and
  information convention; they are not calibrated market estimates.
- The BS CIs are degenerate across training seeds because the baseline is not
  trained. The intervals do not represent uncertainty in the model parameters
  or in future market conditions.
- The robustness sweep changes both volatility and transaction cost across
  cells; it should be read as a sensitivity analysis, not as an independent
  estimate of a single market regime.

## Conclusion

The preserved E6 artifacts support a material zero-shot robustness concern for
Merton jumps: RL risk and tail losses worsen relative to GBM, while mean P&L
falls. Heston changes were smaller and less conclusive for the main RL risk
metrics under the tested parameters. No claim about retraining benefits or
real-market performance is warranted from this evaluation-only experiment.
