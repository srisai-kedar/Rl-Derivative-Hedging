# E7 Word-report figures and tables

Date: 2026-11-06

This is a paste-ready list for Chapter 10 and the Limitations chapter. Every
listed item points to a generated artifact. E2--E5 have no generated figures
or tables and are explicitly marked NOT RUN.

## Chapter 10: Results

| ID | Suggested caption | Type | Source artifact |
|---|---|---|---|
| Table 10.1 | E0 batched versus legacy checkpoint metrics | Table | `results/E0_batched_1000/metrics.json`; `results/E0_legacy_1000/metrics.json` |
| Table 10.2 | E1 across-seed baseline metrics with Student-t CIs | Table | `results/experiments/baseline_multiseed_e1/baseline_multiseed_s{0..4}/evaluation/metrics.json` |
| Table 10.3 | E6 GBM, Heston, and Merton main metrics | Table | `results/experiments/e6_market_robustness/e1_s{0..4}/{heston,merton}/metrics.json`; E1 GBM metrics above |
| Table 10.4 | E6 degradation relative to GBM | Table | E1 metrics plus E6 metrics listed above |
| Table 10.5 | E6 volatility-regime robustness slice | Table | `results/experiments/e6_market_robustness/e1_s{0..4}/{heston,merton}/robustness.csv` |
| Table 10.6 | E6 transaction-cost robustness slice | Table | `results/experiments/e6_market_robustness/e1_s{0..4}/{heston,merton}/robustness.csv` |
| Figure 10.1 | E0 P&L distribution for RL, BS delta, and zero hedge | Figure | `results/E0_batched_1000/plots/pnl_distribution.png` |
| Figure 10.2 | E0 metric comparison | Figure | `results/E0_batched_1000/plots/metric_comparison.png` |
| Figure 10.3 | E0 RL robustness heatmap | Figure | `results/E0_batched_1000/plots/robustness_heatmap_rl_agent.png` |
| Figure 10.4 | E6 Heston RL robustness heatmap | Figure | `results/experiments/e6_market_robustness/e1_s0/heston/plots/robustness_heatmap_rl_agent.png` |
| Figure 10.5 | E6 Merton RL robustness heatmap | Figure | `results/experiments/e6_market_robustness/e1_s0/merton/plots/robustness_heatmap_rl_agent.png` |

The E6 heatmaps are representative generated plots; the corresponding
`robustness.csv` files contain the complete five-seed evaluation records.

## Limitations chapter

| ID | Limitation or evidence to discuss | Source artifact |
|---|---|---|
| L10.1 | E2 2M-step training was NOT RUN; no long-budget conclusion is supported | No E2 artifact exists; E2 status recorded in `docs/experiments/E7_results_tables.md` |
| L10.2 | E3 risk-aversion sweep was NOT RUN; no risk-aversion effect or plateau is supported | No E3 artifact exists |
| L10.3 | E4 HPO and best-configuration retraining were NOT RUN; no selection-bias estimate is supported | No E4 artifact exists |
| L10.4 | E5 observation-mode ablation was NOT RUN; no leakage-ablation conclusion is supported | No E5 artifact exists |
| L10.5 | E6 is zero-shot evaluation of GBM-trained agents, not Heston/Merton retraining | `results/experiments/e6_market_robustness/e1_s{0..4}/{heston,merton}/run_manifest.json` |
| L10.6 | E6 uses the environment's Black-Scholes marking convention under alternate dynamics | E6 `run_manifest.json` files and `src/simulation/heston.py`, `src/simulation/merton.py` |
| L10.7 | BS baseline across-seed CIs are degenerate because the baseline is deterministic on shared seeds | E1 and E6 `metrics.json` files |
| L10.8 | Robustness cells vary both volatility and transaction cost and are sensitivity tests, not calibrated market regimes | E6 `robustness.csv` files |
