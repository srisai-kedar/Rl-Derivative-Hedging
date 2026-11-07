# RL Derivative Hedging

Reinforcement learning agent for hedging a short European call option under
discrete rebalancing and proportional transaction costs. The SAC policy is
benchmarked against Black-Scholes delta hedging and a zero-hedge baseline.

## The Problem

When a desk sells a call option, it takes on exposure to the underlying asset.
Black-Scholes delta hedging gives the number of shares to hold under idealised
assumptions: continuous rebalancing, zero transaction costs, and constant
volatility.

The environment action is the target number of underlying shares held per
option, in `[0, 1]`. It is an absolute position, not a fraction of the
Black-Scholes delta. The BS-delta policy uses the current delta as its target;
the RL action has the same shares-per-option interpretation.

This project relaxes those assumptions. The agent hedges at discrete time steps,
pays transaction costs, observes a simulated market state, and is rewarded by
the P&L of the hedged short-option portfolio after costs.

## Key Results

Full evaluation run: `results/phase6_eval_1000/`, 1,000 shared-seed episodes per
policy using checkpoint `checkpoints/sac_hedging_20260721_002714/best_model.zip`.

| Metric | RL Agent | BS Delta | Zero Hedge |
|--------|----------|----------|------------|
| P&L Std Dev (lower is better) | 2.563 | 0.449 | 4.552 |
| CVaR @ 95% (higher is better) | -6.815 | -0.939 | -13.156 |
| Mean Tx Cost (lower is better) | 0.192 | 0.224 | 0.000 |
| Improvement vs BS | -470.7% std | baseline | n/a |

![P&L distribution](docs/figures/pnl_distribution_phase6.png)

The current SAC checkpoint reduces average transaction cost by 14.4% versus
Black-Scholes delta, but it does not beat the analytical baseline on P&L
variance or tail risk. The agent has learned to trade less aggressively, but the
reduced trading cost is not enough to offset the larger hedge error.

See [RESULTS.md](RESULTS.md) for the full comparison and robustness tables.

## Architecture

```text
Simulation (GBM) -> Gymnasium Env -> SAC Agent -> Evaluation -> Dashboard
      |                 |                            |
Black-Scholes      Reward: P&L             RL vs BS Comparison
Greeks             minus tx cost           Robustness Tests
                                           Plotly Charts
```

Two separations are intentional:

- Finance utilities are pure NumPy/SciPy code and do not depend on RL modules.
- The Gymnasium environment is consumed by training and evaluation, but does not
  know about Stable-Baselines3 configuration.

## Setup

Requirements: Python 3.11+ and pip.

```bash
git clone https://github.com/yourusername/rl-derivative-hedging
cd rl-derivative-hedging
pip install -e .
pip install -r requirements-dev.txt
```

For reproducible runtime installs, use the hashed lockfile matching the Python
minor version: `pip install -r requirements-py311.lock.txt` or
`pip install -r requirements-py312.lock.txt`.

## Configuration

The YAML file has `environment` and `eval_environment` blocks. The evaluation
block inherits unspecified environment values. Supported environment keys are:

| Key | Meaning | Default |
|---|---|---:|
| `risk_aversion` | Reward penalty `raw_pnl - lambda * raw_pnl**2` | `0.0` |
| `liquidate_at_expiry` | Charge the final position-closing trade | `false` |
| `include_financing` | Deduct risk-free financing on held stock | `false` |
| `market_model` | `gbm`, `heston`, or `merton` | `gbm` |
| `market_model_kwargs` | Validated model-specific parameters | `{}` |
| `observation_mode` | Observation ablation mode | `full` |

`kappa` at the top level remains the proportional transaction-cost coefficient.
Heston's variance mean-reversion speed uses the backward-compatible
`market_model_kwargs.kappa` key. The callback block supports `eval_seed_base`
for fixed-seed callback evaluation.

### Observation modes

| Mode | Features | Dimension |
|---|---|---:|
| `full` | price ratio, time remaining, sigma, rate, strike ratio, hedge position, BS delta | 7 |
| `no_greeks` | price ratio, time remaining, rate, strike ratio, hedge position | 5 |
| `price_only` | price ratio, time remaining, hedge position | 3 |
| `no_bs_delta` | full mode without BS delta | 6 |
| `no_bs_delta_no_sigma` | full mode without BS delta or sigma | 5 |

Observation dimensions are part of the trained SAC model and VecNormalize
state. Changing `observation_mode` requires retraining with that mode; an
incompatible checkpoint is rejected during evaluation.

## Usage

### 1. Train

```bash
python -m src.training.train --config configs/training.yaml
tensorboard --logdir results/
```

Checkpoints are saved under `checkpoints/<run>/`; TensorBoard logs are saved
under `results/<run>/`.

### 2. Evaluate

```bash
python -m src.evaluation.evaluate \
    --checkpoint checkpoints/<run>/best_model.zip \
    --vecnorm checkpoints/<run>/best_vecnorm.pkl \
    --config configs/training.yaml \
    --n-episodes 1000 \
    --seed-offset 0 \
    --n-boot 10000
```

Other evaluation flags include `--robustness-episodes`, `--kappa-grid`,
`--sigma-grid`, `--market-model gbm|heston|merton`,
`--market-model-kwargs '{"kappa": 1.5}'`, `--legacy-backtest`, and
`--skip-robustness`. Integrity controls are `--require-manifest`,
`--allow-unverified`, `--allow-manifest-mismatch`, and repeatable
`--expected-sha256 FILE=HASH` anchors.

Evaluation outputs include `episode_data.csv`, `step_data.csv`, `metrics.json`,
`robustness.csv`, and Plotly HTML/PNG charts under `results/<evaluation_run>/`.

### 3. Dashboard

```bash
streamlit run src/dashboard/app.py
```

Open `http://localhost:8501`, select a saved evaluation run, and inspect the
overview, P&L analysis, episode replay, experiment table, and training logs.

### 4. Hyperparameter Search

```bash
python -m src.training.hpo --trials 20 --storage sqlite:///hpo.db
```

The HPO module uses Optuna with a TPE sampler and median pruner. Each trial
trains SAC for a reduced budget and evaluates the resulting policy.

HPO defaults use a held-out evaluation block beginning at seed `10000`. Use
`--seeds-per-trial`, `--objective mean_reward|mean_minus_lambda_std|cvar`,
`--risk-lambda`, `--search-risk-aversion`, and
`--search-observation-mode` to expand the search. The
`--legacy-seed-behavior` flag restores the historical seed policy. Persistent
SQLite studies receive a `.metadata.json` sidecar with seeds, search space,
objective, and a manifest.

### 5. Experiment framework

```bash
python -m experiments.runner --experiment experiments/baseline_multiseed.yaml --dry-run
python -m experiments.runner --experiment experiments/baseline_multiseed.yaml --smoke
python -m experiments.runner --experiment experiments/baseline_multiseed.yaml --resume
python -m experiments.report --root results/experiments/baseline_multiseed \
    --output report.md --json-output report.json
```

Smoke outputs are labelled `SMOKE: not a result`; reports print `NOT RUN` for
missing inputs.

## Project Structure

```text
rl-derivative-hedging/
|-- configs/              YAML training configuration
|-- src/
|   |-- finance/          Black-Scholes pricing, Greeks, transaction costs
|   |-- simulation/       GBM, Heston, and Merton price-path generators
|   |-- envs/             Gymnasium hedging environment
|   |-- training/         SAC training, env factory, callbacks, HPO
|   |-- evaluation/       Backtest, metrics, robustness, Plotly charts
|   `-- dashboard/        Streamlit dashboard
|-- tests/                Pytest suite with coverage gate
|-- notebooks/            Mathematical exploration notebook
|-- checkpoints/          Saved models and VecNormalize stats, git-ignored
`-- results/              Evaluation outputs and plots, git-ignored
```

## Design Decisions

**SAC over PPO:** SAC handles continuous actions directly and uses replay-buffer
sample reuse, which is useful for a low-dimensional continuous-control problem.

**VecNormalize saved with every model:** Observation and reward normalisation are
part of the trained policy state. Evaluation loads matching VecNormalize stats
alongside the SAC checkpoint.

**Shared seeds for baseline comparison:** RL, Black-Scholes delta, and zero hedge
are evaluated on identical price paths for each episode seed.

**Seeding contract:** `reset(seed=s)` is deterministic. A seedless `reset()`
draws a fresh path seed from the Gymnasium RNG instead of silently repeating
the constructor seed. See [ADR-0001](docs/adr/0001-seeding-contract.md).

**Market-model convention:** `market_model="gbm"` remains the default and
preserves the existing GBM behaviour. `heston` uses full-truncation Euler and
`merton` uses an exact-conditional jump-diffusion step. In every model, option
marks and `bs_delta` use Black-Scholes with the environment's configured
`sigma`; the hedger does not observe Heston's instantaneous variance or Merton
jump state. The environment's top-level `kappa` is the transaction-cost rate.
Heston's variance mean-reversion speed is configured separately, using the
backward-compatible `market_model_kwargs["kappa"]` key. When Heston `xi` is
omitted, the environment derives `min(0.3, sqrt(2*kappa*theta))` to satisfy
the Feller condition; explicit `xi` values are honoured with full truncation.

**Checkpoint integrity:** Training and evaluation write `run_manifest.json`
with configuration and checkpoint hashes. Evaluation verifies it before loading
model and VecNormalize files. Use `--require-manifest` for strict mode or
`--expected-sha256 FILE=HASH` for an out-of-band trust anchor. SAC archives and
VecNormalize pickles can execute arbitrary code during deserialization; never
load untrusted files. See [Manifest and integrity usage](docs/integrity.md).

**Streamlit over React:** The dashboard is a local analysis tool. Streamlit and
Plotly keep the project Python-native while still supporting interactive charts.

## Known Limitations

- The option tested is a European call on a single underlying.
- No real market data is used.
- The current SAC checkpoint is undertrained relative to the analytical baseline
  on risk metrics; more training or HPO is needed before claiming improvement.
- Transaction costs are proportional only and do not include nonlinear market
  impact.

## Project Status

- Phase 0: project skeleton and tooling
- Phase 1: Black-Scholes pricing, Greeks, and GBM simulation
- Phase 2: Gymnasium hedging environment
- Phase 3: SAC training pipeline
- Phase 4: evaluation and robustness pipeline
- Phase 5: Streamlit dashboard
- Phase 6: HPO, quality gates, documentation, and notebook polish

## References

- Whalley and Wilmott (1997), transaction-cost option hedging
- Halperin (2020), QLBS: Q-Learner in the Black-Scholes Worlds
- Kolm and Ritter (2019), dynamic replication and hedging with RL
- Cao et al. (2020), Deep Hedging of Derivatives Using Reinforcement Learning
- Haarnoja et al. (2018), Soft Actor-Critic
