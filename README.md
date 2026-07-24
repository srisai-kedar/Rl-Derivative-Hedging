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

## Run Dashboard

The Streamlit dashboard reads pre-computed evaluation results from `results/`.
Run it from the project root so the relative results path resolves correctly.

```bash
streamlit run src/dashboard/app.py
```

## Project Status

- [x] Phase 0: Project skeleton and tooling
- [x] Phase 1: Black-Scholes pricing, Greeks, GBM simulation
- [x] Phase 2: Gymnasium hedging environment
- [x] Phase 3: SAC training pipeline
- [X] Phase 4: Evaluation and backtesting
- [X] Phase 5: Streamlit dashboard
- [ ] Phase 6: Hyperparameter search (Optuna)
