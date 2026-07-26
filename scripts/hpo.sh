#!/usr/bin/env bash
# Optuna hyperparameter search wrapper
set -euo pipefail
python -m src.training.hpo "$@"
