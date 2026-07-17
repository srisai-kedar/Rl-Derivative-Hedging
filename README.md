# RL Derivative Hedging

Reinforcement learning system for hedging short options positions under
realistic transaction costs and discrete rebalancing.

## Setup

```bash
pip install -e .
pip install -r requirements-dev.txt
```

## Run Tests

```bash
pytest tests/test_black_scholes.py -v
pytest tests/test_simulation.py -v
pytest --cov=src/finance --cov=src/simulation
```

## Project Status

- [x] Phase 0: Project skeleton and tooling
- [x] Phase 1: Black-Scholes pricing, Greeks, GBM simulation
- [ ] Phase 2: Gymnasium hedging environment
- [ ] Phase 3: SAC training pipeline
- [ ] Phase 4: Evaluation and backtesting
- [ ] Phase 5: Streamlit dashboard
- [ ] Phase 6: Hyperparameter search (Optuna)
