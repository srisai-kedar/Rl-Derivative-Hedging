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

- Phase 0: Project skeleton and tooling
- Phase 1: Black-Scholes pricing, Greeks, GBM simulation
