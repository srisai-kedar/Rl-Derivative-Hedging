"""Heston stochastic-volatility price path simulator.

Discretisation
--------------
Full-truncation Euler (Lord et al., 2010): the variance process is integrated
with an Euler-Maruyama step and then *truncated* to a small positive floor
before taking the square root for the diffusion term and before updating the
variance.  This prevents the "absorbed" (zero-diffusion) and "reflecting"
(negative variance folded back) biases that affect plain Euler and the
Milstein scheme at low Feller ratios.

    v_{t+dt}  =  max(v_t, 0) + kappa*(theta - max(v_t, 0))*dt
                 + xi * sqrt(max(v_t, 0) * dt) * Z_v

    S_{t+dt}  =  S_t * exp((mu - 0.5 * max(v_t, 0)) * dt
                            + sqrt(max(v_t, 0) * dt) * Z_S)

where (Z_S, Z_v) are correlated standard normals with correlation rho.

Information / pricing convention
---------------------------------
The hedger observes the *Black-Scholes implied sigma* that the environment
was initialised with (or randomised with), **not** the true instantaneous
variance sqrt(v_t).  Options are marked to market using Black-Scholes with
that fixed sigma.  This is a deliberate modelling choice: in practice the
hedger cannot perfectly observe the true vol process, and a consistent BS
marking convention lets the telescoping P&L identity remain valid under the
environment's own marks.

Feller condition:  2 * kappa * theta >= xi^2  guarantees v > 0 a.s. in
continuous time.  The HedgingEnv derives an omitted default ``xi`` as
``min(0.3, sqrt(2*kappa*theta))``; an explicitly supplied ``xi`` is honoured
even when it violates Feller, while full-truncation keeps the simulation
well-defined.  The environment's top-level ``kappa`` is transaction cost;
this simulator's ``kappa`` is variance mean reversion.
"""

from __future__ import annotations

import numpy as np


class HestonSimulator:
    """Full-truncation Euler discretisation of the Heston (1993) model.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    mu : float
        Risk-neutral drift (typically the risk-free rate r).
    v0 : float
        Initial variance (spot vol squared).  Default = sigma^2 where
        sigma is the BS vol used in the companion HedgingEnv.
    kappa : float
        Mean-reversion speed of the variance process.
    theta : float
        Long-run (steady-state) variance.
    xi : float
        Vol-of-vol (volatility of the variance process).
    rho : float
        Correlation between the spot and variance Brownian motions.
    dt : float
        Time step in years.
    seed : int | None
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        S0: float,
        mu: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        dt: float,
        seed: int | None = None,
    ) -> None:
        if S0 <= 0:
            raise ValueError(f"S0 must be > 0, got {S0}.")
        if v0 <= 0:
            raise ValueError(f"v0 must be > 0, got {v0}.")
        if kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {kappa}.")
        if theta <= 0:
            raise ValueError(f"theta must be > 0, got {theta}.")
        if xi <= 0:
            raise ValueError(f"xi must be > 0, got {xi}.")
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {rho}.")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}.")

        self.S0 = float(S0)
        self.mu = float(mu)
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        self.last_variance_path: np.ndarray | None = None

        # Cholesky factor for the 2-D correlated Brownians:
        # [ Z_S ]   [ 1    0            ] [ W_1 ]
        # [ Z_v ] = [ rho  sqrt(1-rho^2) ] [ W_2 ]
        self._chol = np.array(
            [[1.0, 0.0], [rho, np.sqrt(max(1.0 - rho**2, 0.0))]]
        )

    # ------------------------------------------------------------------ #
    # Core simulation
    # ------------------------------------------------------------------ #

    def _correlated_normals(self, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Return (Z_S, Z_v) correlated standard normals with shape `shape`."""
        W = self.rng.standard_normal((*shape, 2))  # shape + (2,)
        Z = W @ self._chol.T  # apply Cholesky: Z = W * L^T
        return Z[..., 0], Z[..., 1]

    def generate_path(self, n_steps: int) -> np.ndarray:
        """Generate a single spot price path of shape (n_steps + 1,).

        Variance is simulated internally but not returned (the hedger
        does not observe it directly).
        """
        prices = np.empty(n_steps + 1)
        prices[0] = self.S0
        v = self.v0
        variance_path = np.empty(n_steps + 1)
        variance_path[0] = v

        Z_S_arr, Z_v_arr = self._correlated_normals((n_steps,))
        sqrt_dt = np.sqrt(self.dt)

        for i in range(n_steps):
            v_plus = max(v, 0.0)
            sv = np.sqrt(v_plus * self.dt)
            # Variance step (full-truncation Euler)
            v = v_plus + self.kappa * (self.theta - v_plus) * self.dt + self.xi * sv * Z_v_arr[i]
            v = max(v, 0.0)
            variance_path[i + 1] = v
            # Spot step (log-Euler for exact lognormal conditional on v_plus)
            prices[i + 1] = prices[i] * np.exp(
                (self.mu - 0.5 * v_plus) * self.dt + np.sqrt(v_plus) * sqrt_dt * Z_S_arr[i]
            )

        self.last_variance_path = variance_path
        return prices

    def generate_paths(self, n_paths: int, n_steps: int) -> np.ndarray:
        """Generate multiple spot price paths of shape (n_paths, n_steps + 1)."""
        out = np.empty((n_paths, n_steps + 1))
        out[:, 0] = self.S0

        Z_S, Z_v = self._correlated_normals((n_paths, n_steps))
        sqrt_dt = np.sqrt(self.dt)
        v = np.full(n_paths, self.v0)

        for i in range(n_steps):
            v_plus = np.maximum(v, 0.0)
            sv = np.sqrt(v_plus * self.dt)
            v = v_plus + self.kappa * (self.theta - v_plus) * self.dt + self.xi * sv * Z_v[:, i]
            v = np.maximum(v, 0.0)
            out[:, i + 1] = out[:, i] * np.exp(
                (self.mu - 0.5 * v_plus) * self.dt + np.sqrt(v_plus) * sqrt_dt * Z_S[:, i]
            )

        return out

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset the RNG for reproducible evaluation."""
        self.rng = np.random.default_rng(seed)

    @property
    def feller_satisfied(self) -> bool:
        """Return True if the Feller condition 2κθ ≥ ξ² holds."""
        return 2.0 * self.kappa * self.theta >= self.xi**2
