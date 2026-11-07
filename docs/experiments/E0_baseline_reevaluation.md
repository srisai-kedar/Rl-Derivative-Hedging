\# E0 – Baseline Checkpoint Reevaluation



\## Objective



Re-evaluate the existing 300,000-step SAC hedging checkpoint using the new batched backtester and compare it against the legacy backtester and the published results.



\## Evaluation Setup



\- Existing SAC checkpoint: `checkpoints/sac\_hedging\_20260721\_002714/`

\- Evaluation episodes: 1,000

\- Evaluation seeds: 0–999

\- Robustness evaluation: 36 cells

\- Strategies: SAC, Black-Scholes Delta Hedging and Zero Hedging.

\- No new model training was performed.



\## Main Results



| Metric | SAC | Black-Scholes | Zero Hedging |

|---|---:|---:|---:|

| Mean P\&L | 0.3070 | 0.1025 | -0.1222 |

| P\&L standard deviation | 2.5626 | 0.4491 | 4.5519 |

| CVaR 95% | -6.8147 | -0.9392 | -13.1556 |



\### Transaction Costs



\- SAC mean transaction cost: 0.1915

\- Black-Scholes mean transaction cost: 0.2237



The SAC agent had lower transaction costs, but substantially higher P\&L variability and worse downside risk than Black-Scholes.



\## Batched vs Legacy Backtester



Both implementations produced practically identical results.



\- SAC mean P\&L difference: 4.44e-06

\- SAC P\&L standard deviation difference: -1.69e-05

\- SAC CVaR difference: 8.92e-05

\- Black-Scholes and Zero Hedging metrics were identical.



The published rounded values in `RESULTS.md` were reproduced.



\## Robustness Evaluation



\- All six volatility values matched the published results at `kappa=0.001`.

\- All five published transaction-cost values matched at `sigma=0.20`.

\- An additional evaluation at `kappa=0.0005` produced SAC standard deviation of 2.697 and Black-Scholes standard deviation of 0.478.



\## Runtime



A 5,000-step CPU smoke run took 26.95 seconds, with measured throughput of approximately 185.53 steps per second.



Estimated runtime based on this single run:



\- 300,000 steps: approximately 26.95 minutes.

\- 2,000,000 steps: approximately 179.65 minutes.



These are linear estimates, not measured full-training runtimes.



\## Checkpoint Integrity



The original checkpoint directory had no manifest, so the initial evaluation produced a legacy-unverified warning.



The evaluation outputs generated manifests covering `best\_model.zip` and `best\_vecnorm.pkl`.



No existing checkpoint or `RESULTS.md` was modified.



\## Conclusion



The batched backtester successfully reproduced the legacy evaluation results and published rounded metrics.



The SAC agent reduced transaction costs compared with Black-Scholes, but Black-Scholes remained considerably better in risk management.



Further multi-seed training and risk-aware optimisation are required before drawing conclusions about improved RL hedging performance.

