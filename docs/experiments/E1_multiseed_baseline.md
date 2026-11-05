\# E1 – Multi-Seed Baseline Training



\## Training Setup

\- Independent training seeds: 0, 1, 2, 3, 4

\- Training steps: 300,000 per seed

\- Evaluation episodes: 1,000 per seed

\- Robustness evaluation: Enabled

\- Baselines: Black-Scholes Delta and Zero Hedging



\## Across-Seed Results



| Metric | SAC Mean ± SD | SAC 95% CI | Black-Scholes |

|---|---:|---:|---:|

| P\&L Std | 2.610 ± 0.238 | \[2.314, 2.906] | 0.449 ± 0.000 |

| CVaR 95% | -6.317 ± 0.673 | \[-7.153, -5.480] | -0.939 ± 0.000 |

| Mean P\&L | 0.230 ± 0.074 | \[0.139, 0.321] | 0.102 ± 0.000 |

| Mean Cost | 0.209 ± 0.028 | \[0.175, 0.244] | 0.224 ± 0.000 |



\## Main Findings



\- SAC achieved higher average P\&L.

\- SAC showed substantially higher P\&L variability.

\- SAC had worse CVaR and downside risk.

\- Average transaction-cost improvement was not statistically supported across training seeds.

\- Risk differences were consistent across all five runs.



\## Within-Run Paired Confidence Intervals



10,000-resample paired bootstrap confidence intervals over shared evaluation episodes.



| Seed | Mean P\&L Difference CI | Std P\&L Difference CI | CVaR Difference CI |

|---|---|---|---|

| 0 | \[-0.102, 0.269] | \[2.432, 2.715] | \[-6.120, -5.296] |

| 1 | \[-0.121, 0.192] | \[1.668, 2.357] | \[-7.396, -5.264] |

| 2 | \[0.028, 0.333] | \[1.977, 2.244] | \[-5.815, -4.750] |

| 3 | \[-0.031, 0.261] | \[1.838, 2.122] | \[-5.133, -4.031] |

| 4 | \[0.070, 0.367] | \[1.910, 2.154] | \[-5.369, -4.552] |



\## Conclusion



The five-seed experiment shows that SAC can achieve higher average returns, but its financial risk remains substantially higher than Black-Scholes.



Further risk-aware training and optimisation are required.

