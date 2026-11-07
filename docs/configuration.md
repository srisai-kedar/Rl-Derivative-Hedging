# Configuration reference

`configs/training.yaml` is parsed into typed training and environment
configuration. `eval_environment` inherits the main `environment` block and
then applies its own keys.

## Environment switches

- `risk_aversion: 0.0` applies the step-reward penalty
  `raw_pnl - risk_aversion * raw_pnl**2`; metrics continue to use raw P&L.
- `liquidate_at_expiry: false` charges the final position-closing trade when
  enabled.
- `include_financing: false` deducts risk-free financing on stock held during a
  step when enabled.
- `market_model: gbm` selects `gbm`, `heston`, or `merton`.
- `market_model_kwargs: {}` contains validated model-specific parameters.
- `observation_mode: full` selects the feature subset below.

## Pricing and information convention

GBM is the default and preserves the published behavior. Heston uses
full-truncation Euler variance dynamics; Merton uses a compensated jump-
diffusion. In every market model, option values and `bs_delta` are
Black-Scholes marks using the environment's configured (or randomized)
`sigma`. The hedger does not receive Heston instantaneous variance or Merton
jump state as hidden extra information. The top-level `kappa` is transaction
cost; Heston mean reversion uses `market_model_kwargs.kappa`.

The action is the target number of underlying shares held per option in
`[0, 1]`; it is not a fraction of delta. `hedge_pos` uses the same
shares-per-option unit.

## Observation modes

| Mode | Index list | Keys | Dimension |
|---|---|---|---:|
| `full` | `[0,1,2,3,4,5,6]` | price, time, sigma, rate, strike, hedge position, BS delta | 7 |
| `no_greeks` | `[0,1,3,4,5]` | price, time, rate, strike, hedge position | 5 |
| `price_only` | `[0,1,5]` | price, time, hedge position | 3 |
| `no_bs_delta` | `[0,1,2,3,4,5]` | full without BS delta | 6 |
| `no_bs_delta_no_sigma` | `[0,1,3,4,5]` | full without BS delta or sigma | 5 |

Observation dimension must match the SAC model and VecNormalize statistics.
Changing this setting requires a checkpoint trained with the same mode; it is
not a post-processing switch for a legacy 7-dimensional model.

## Evaluation seeds

`callbacks.eval_seed_base` controls fixed callback evaluation seeds. For final
evaluation, `--seed-offset` selects the first shared seed, with default seeds
`0..n-1`. HPO uses a separate held-out block beginning at `10000` by default.
