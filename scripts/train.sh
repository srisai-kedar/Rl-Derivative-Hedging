#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train --config "${1:-configs/training.yaml}"
