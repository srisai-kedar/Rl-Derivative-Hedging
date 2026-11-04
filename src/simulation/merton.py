"""Merton jump-diffusion price path simulator.

Model
-----
Merton (1976) adds a compound Poisson jump process to GBM:

    dS / S = (mu - lambda * k_bar) dt + sigma dW + (J - 1) dN

where
  * dN  ~ Poisson(lambda dt) — jump arrival process
  * J   ~ lognormal(mu_J, sigma_J^2) — random jump size (multiplicative)
  * k_bar = exp(mu_J + 0.5 * sigma_J^2) - 1  — mean excess jump return

The drift is adjusted so E[S_t] = S_0 * exp(mu * t) (risk-neutral).

Discretisation (exact conditional on jump count)
-------------------------------------------------
Over each interval [t, t+dt] we draw N_t ~ Poisson(lambda * dt) jumps.
Conditional on N_t jumps, the log-return is:

    log(S_{t+dt}/S_t) = (mu - lambda*k_bar - 0.5*sigma^2)*dt
                        + sigma*sqrt(dt)*Z
                        + sum_{i=1}^{N_t} Y_i

where Y_i ~ N(mu_J, sigma_J^2) are iid log jump sizes.

Information / pricing convention
---------------------------------
Options are marked with Black-Scholes at the GBM sigma passed to the
environment (identical convention to HestonSimulator — see that module's
docstring for the rationale).  The telescoping P&L identity remains valid
under the environment's own BS marks.
"""

from __future__ import annotations

import numpy as np


class MertonJumpDiffusionSimulator:
    """Exact-conditional discretisation of the Merton (1976) jump-diffusion.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    mu : float
        Risk-neutral drift.
    sigma : float
        Diffusion volatility (GBM component).
    lam : float
        Poisson jump arrival rate (jumps per year).
    mu_J : float
        Mean of the log jump size Y ~ N(mu_J, sigma_J^2).
    sigma_J : float
        Standard deviation of the log jump size.
    dt : float
        Time step in years.
    seed : int | None
        Optional RNG seed.
    """

    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        lam: float,
        mu_J: float,
        sigma_J: float,
        dt: float,
        seed: int | None = None,
    ) -> None:
        if S0 <= 0:
            raise ValueError(f"S0 must be > 0, got {S0}.")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}.")
        if lam < 0:
            raise ValueError(f"lam must be >= 0, got {lam}.")
        if sigma_J < 0:
            raise ValueError(f"sigma_J must be >= 0, got {sigma_J}.")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}.")

        self.S0 = float(S0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.mu_J = float(mu_J)
        self.sigma_J = float(sigma_J)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)

        # Pre-compute drift adjustment so E[S_t] = S_0 * exp(mu*t)
        self._k_bar = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0
        self._adj_drift = (mu - lam * self._k_bar - 0.5 * sigma**2) * dt

    # ------------------------------------------------------------------ #
    # Core simulation
    # ------------------------------------------------------------------ #

    def _step_log_returns(self, n: int) -> np.ndarray:
        """Generate n log-returns for one time step."""
        diffusion = self._adj_drift + self.sigma * np.sqrt(self.dt) * self.rng.standard_normal(n)
        # Jump component
        n_jumps = self.rng.poisson(self.lam * self.dt, size=n)
        jump_contrib = np.zeros(n)
        for idx in np.where(n_jumps > 0)[0]:
            k = int(n_jumps[idx])
            jump_contrib[idx] = self.rng.normal(self.mu_J, self.sigma_J, k).sum()
        return diffusion + jump_contrib

    def generate_path(self, n_steps: int) -> np.ndarray:
        """Generate a single price path of shape (n_steps + 1,)."""
        log_rets = self._step_log_returns(n_steps)
        prices = np.empty(n_steps + 1)
        prices[0] = self.S0
        log_prices = np.log(self.S0) + np.cumsum(log_rets)
        prices[1:] = np.exp(log_prices)
        return prices

    def generate_paths(self, n_paths: int, n_steps: int) -> np.ndarray:
        """Generate multiple price paths of shape (n_paths, n_steps + 1)."""
        out = np.empty((n_paths, n_steps + 1))
        out[:, 0] = self.S0

        for step in range(n_steps):
            log_rets = self._step_log_returns(n_paths)
            out[:, step + 1] = out[:, step] * np.exp(log_rets)

        return out

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset the RNG for reproducible evaluation."""
        self.rng = np.random.default_rng(seed)

    @property
    def expected_jumps_per_episode(self) -> float:
        """Expected number of jumps over one full episode of dt*n_steps years."""
        return self.lam * self.dt
