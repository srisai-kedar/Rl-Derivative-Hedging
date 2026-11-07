# E7 paste-ready experiment tables

Date: 2026-11-06

All values below were extracted from generated experiment artifacts. Paths are
repository-relative source paths.

## E0 checkpoint reevaluation

Source: `results/E0_batched_1000/metrics.json` and
`results/E0_legacy_1000/metrics.json`.

| Backtester | Policy | Mean P&L | Std P&L | CVaR 95% | Mean cost |
|---|---|---:|---:|---:|---:|
| Batched | RL | 0.3070 | 2.5626 | -6.8147 | 0.1915 |
| Batched | BS delta | 0.1025 | 0.4491 | -0.9392 | 0.2237 |
| Legacy | RL | 0.3070 | 2.5626 | -6.8148 | 0.1915 |
| Legacy | BS delta | 0.1025 | 0.4491 | -0.9392 | 0.2237 |

## E1 across-seed baseline

Source: `results/experiments/baseline_multiseed_e1/baseline_multiseed_s{0..4}/evaluation/metrics.json`.

Student-t 95% CIs use five training seeds.

| Policy | Metric | Mean | Across-seed SD | 95% CI |
|---|---|---:|---:|---:|
| RL | Mean P&L | 0.2300 | 0.0736 | [0.1386, 0.3214] |
| RL | Std P&L | 2.6102 | 0.2384 | [2.3142, 2.9061] |
| RL | CVaR 95% | -6.3165 | 0.6734 | [-7.1527, -5.4803] |
| RL | Mean cost | 0.2095 | 0.0276 | [0.1752, 0.2437] |
| BS delta | Mean P&L | 0.1025 | 0.0000 | [0.1025, 0.1025] |
| BS delta | Std P&L | 0.4491 | 0.0000 | [0.4491, 0.4491] |
| BS delta | CVaR 95% | -0.9392 | 0.0000 | [-0.9392, -0.9392] |
| BS delta | Mean cost | 0.2237 | 0.0000 | [0.2237, 0.2237] |

## E6 cross-dynamics main metrics

Source: `results/experiments/e6_market_robustness/e1_s{0..4}/{heston,merton}/metrics.json`.

Values are five-seed means with Student-t 95% CIs.

| Dynamics | Policy | Mean P&L | Std P&L | CVaR 95% | Mean cost |
|---|---|---:|---:|---:|---:|
| GBM | RL | 0.2300 [0.1386, 0.3214] | 2.6102 [2.3142, 2.9061] | -6.3165 [-7.1527, -5.4803] | 0.2095 [0.1752, 0.2437] |
| GBM | BS delta | 0.1025 [0.1025, 0.1025] | 0.4491 [0.4491, 0.4491] | -0.9392 [-0.9392, -0.9392] | 0.2237 [0.2237, 0.2237] |
| Heston | RL | 0.3487 [0.2640, 0.4333] | 2.6433 [2.3778, 2.9089] | -6.5625 [-7.3456, -5.7794] | 0.2050 [0.1707, 0.2392] |
| Heston | BS delta | 0.1013 [0.1013, 0.1013] | 0.6197 [0.6197, 0.6197] | -1.4420 [-1.4420, -1.4420] | 0.2241 [0.2241, 0.2241] |
| Merton | RL | -0.1291 [-0.2114, -0.0467] | 3.3080 [2.9929, 3.6230] | -9.5957 [-10.1121, -9.0792] | 0.2136 [0.1820, 0.2452] |
| Merton | BS delta | -0.2388 [-0.2388, -0.2388] | 1.6464 [1.6464, 1.6464] | -6.3193 [-6.3193, -6.3193] | 0.2220 [0.2220, 0.2220] |

## E6 degradation relative to GBM

Source: the same E6 `metrics.json` files above and the E1 GBM metrics files.
Each value is stress minus GBM; intervals are across-seed Student-t 95% CIs.

| Dynamics | Policy | Std P&L delta | CVaR delta | Mean P&L delta | Mean-cost delta |
|---|---|---:|---:|---:|---:|
| Heston | RL | +0.0332 [-0.0514, 0.1178] | -0.2460 [-0.6043, 0.1123] | +0.1186 [0.0943, 0.1430] | -0.0045 [-0.0088, -0.0002] |
| Merton | RL | +0.6978 [0.5913, 0.8043] | -3.2792 [-3.8647, -2.6936] | -0.3591 [-0.3992, -0.3189] | +0.0041 [-0.0017, 0.0100] |

## E6 robustness slices

Source: `results/experiments/e6_market_robustness/e1_s{0..4}/{heston,merton}/robustness.csv`.
Values are five-seed averages of std P&L.

### Fixed `kappa=0.001`, varying `sigma`

| Sigma | Heston RL | Heston BS | Merton RL | Merton BS |
|---:|---:|---:|---:|---:|
| 0.10 | 1.3757 | 0.3582 | 2.5446 | 1.9980 |
| 0.15 | 1.9399 | 0.4868 | 2.7266 | 1.7733 |
| 0.20 | 2.5665 | 0.5680 | 3.2279 | 1.6185 |
| 0.25 | 2.9207 | 0.6556 | 3.5843 | 1.5181 |
| 0.30 | 3.6928 | 0.7485 | 4.2176 | 1.4512 |
| 0.40 | 5.3801 | 0.9445 | 5.8124 | 1.4482 |

### Fixed `sigma=0.20`, varying `kappa`

| Kappa | Heston RL | Heston BS | Merton RL | Merton BS |
|---:|---:|---:|---:|---:|
| 0.0000 | 2.5453 | 0.5449 | 3.2049 | 1.6124 |
| 0.0005 | 2.5558 | 0.5557 | 3.2163 | 1.6151 |
| 0.0010 | 2.5665 | 0.5680 | 3.2279 | 1.6185 |
| 0.0020 | 2.5886 | 0.5963 | 3.2520 | 1.6268 |
| 0.0050 | 2.6602 | 0.7062 | 3.3310 | 1.6644 |
| 0.0100 | 2.7959 | 0.9401 | 3.4827 | 1.7663 |

## Experiment status

| Experiment | Status | Generated source |
|---|---|---|
| E0 checkpoint reevaluation | COMPLETED | `results/E0_batched_1000/`, `results/E0_legacy_1000/` |
| E1 five-seed 300k baseline | COMPLETED | `results/experiments/baseline_multiseed_e1/` |
| E2 five-seed 2M training | NOT RUN | No generated E2 artifacts |
| E3 risk-aversion sweep | NOT RUN | No generated E3 artifacts |
| E4 HPO and retraining | NOT RUN | No generated E4 artifacts |
| E5 observation ablation | NOT RUN | No generated E5 artifacts |
| E6 Heston/Merton evaluation | COMPLETED | `results/experiments/e6_market_robustness/` |
