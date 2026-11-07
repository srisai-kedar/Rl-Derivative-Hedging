"""Price path simulation utilities.

Available simulators
--------------------
GBMSimulator                 – Geometric Brownian Motion (default)
HestonSimulator              – Heston stochastic-volatility (full-truncation Euler)
MertonJumpDiffusionSimulator – Merton (1976) jump-diffusion (exact-conditional)

Select via the ``market_model`` config key: "gbm" | "heston" | "merton".
"""

from src.simulation.gbm import GBMSimulator
from src.simulation.heston import HestonSimulator
from src.simulation.merton import MertonJumpDiffusionSimulator

__all__ = ["GBMSimulator", "HestonSimulator", "MertonJumpDiffusionSimulator"]
