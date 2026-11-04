"""Gymnasium environment for learning to hedge a short European call option.

Design notes (changes relative to the original implementation)
---------------------------------------------------------------
* **Validation, not assertions.** Constructor arguments and actions are
  validated with explicit exceptions. ``assert`` statements are stripped under
  ``python -O`` and must never guard user-facing input.
* **Gymnasium seeding contract.** ``reset(seed=None)`` now draws a *fresh*
  path seed from ``self.np_random`` instead of silently re-using the
  constructor seed. Previously every episode of an env built with ``seed=42``
  was the *same* path unless a wrapper injected seeds. ``reset(seed=s)`` still
  generates exactly ``GBMSimulator(seed=s)``'s path, so published evaluation
  results (shared seeds 0..N-1) are reproduced bit-for-bit.
* **Vectorised pricing.** The whole GBM path is known at ``reset()``, so the
  option values and Black-Scholes deltas for all ``T + 1`` dates are computed
  in one NumPy call (O(T)) instead of two scalar SciPy calls per step.
* **Optional realism switches** (all default to the legacy behaviour):
  ``liquidate_at_expiry``, ``include_financing`` and ``risk_aversion``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.episode_state import EpisodeState
from src.envs.reward import compute_step_reward
from src.finance.black_scholes import call_delta, call_price
from src.finance.transaction_costs import proportional_cost
from src.simulation.gbm import GBMSimulator
from src.simulation.heston import HestonSimulator
from src.simulation.merton import MertonJumpDiffusionSimulator

_MARKET_MODELS = {"gbm", "heston", "merton"}

_OBS_KEYS = [
    "price_ratio",
    "time_remaining",
    "sigma",
    "r",
    "strike_ratio",
    "hedge_pos",
    "bs_delta",
]
_OBS_MODE_INDICES = {
    "full": [0, 1, 2, 3, 4, 5, 6],
    "no_greeks": [0, 1, 3, 4, 5],
    "price_only": [0, 1, 5],
    "no_bs_delta": [0, 1, 2, 3, 4, 5],
    "no_bs_delta_no_sigma": [0, 1, 3, 4, 5],
}
_OBS_KEYS_BY_MODE = {
    mode: [_OBS_KEYS[index] for index in indices]
    for mode, indices in _OBS_MODE_INDICES.items()
}

_MAX_SEED = np.iinfo(np.int32).max
_MIN_TIME_FOR_DELTA = 1e-8  # avoids 0/0 in d1 at expiry; delta is a step there


class ConfigurationError(ValueError):
    """Raised when environment parameters are invalid or inconsistent."""


class HedgingEnv(gym.Env):
    """
    Environment for hedging a **short** European call with the underlying.

    Each day the agent chooses the *target number of shares held per option*
    ``a`` in ``[0, 1]`` (it is an absolute stock position, not a fraction of
    delta). The environment charges proportional transaction costs on the
    position change, advances the GBM path one day, re-prices the option and
    returns the one-period hedged-portfolio P&L as reward.

    Observation (7-d, float32): see ``OBS_KEYS``; clipped to the Box bounds.
    Action: Box(1,) in [0, 1]; non-finite actions raise ``ValueError``.
    Episode: exactly ``T`` steps, then ``terminated=True``.

    Pricing/information convention: option values and ``bs_delta`` are always
    Black-Scholes marks using the environment's configured (or randomised)
    ``sigma``. Heston variance and Merton jumps are not inserted into those
    marks. The top-level ``kappa`` remains the transaction-cost coefficient;
    Heston's variance mean-reversion speed uses the backward-compatible
    ``market_model_kwargs["kappa"]`` key.

    Parameters
    ----------
    risk_aversion:
        ``lambda >= 0``. The *returned* reward is ``pnl - lambda * pnl**2``
        (a mean-variance style penalty on step P&L). ``info["step_pnl"]`` and
        all P&L accumulators always hold the *raw* P&L, so evaluation metrics
        are unaffected. ``0.0`` (default) reproduces the original reward.
    liquidate_at_expiry:
        If ``True``, the closing trade of the final stock position is charged
        at the expiry price (cost counted in ``transaction_cost``).
    include_financing:
        If ``True``, the stock position is assumed funded at the risk-free
        rate: ``r * dt * a * S_prev`` is deducted each step (reported in
        ``info["financing_cost"]``, *not* in transaction cost).
    """

    OBS_KEYS: ClassVar[list[str]] = _OBS_KEYS
    OBS_MODE_INDICES: ClassVar[dict[str, list[int]]] = _OBS_MODE_INDICES
    OBS_KEYS_BY_MODE: ClassVar[dict[str, list[str]]] = _OBS_KEYS_BY_MODE
    _OBS_MODES = OBS_MODE_INDICES
    _OBS_LOW: ClassVar[np.ndarray] = np.array(
        [0.0, 0.0, 0.01, 0.00, 0.5, 0.0, 0.0], dtype=np.float32
    )
    _OBS_HIGH: ClassVar[np.ndarray] = np.array(
        [5.0, 1.0, 1.00, 0.20, 2.0, 1.0, 1.0], dtype=np.float32
    )

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        S0: float = 100.0,
        K: float = 100.0,
        T: int = 30,
        r: float = 0.05,
        sigma: float = 0.20,
        dt: float = 1 / 252,
        kappa: float = 0.001,
        randomise_params: bool = False,
        sigma_range: tuple[float, float] = (0.10, 0.40),
        seed: int | None = None,
        risk_aversion: float = 0.0,
        liquidate_at_expiry: bool = False,
        include_financing: bool = False,
        market_model: str = "gbm",
        market_model_kwargs: dict[str, Any] | None = None,
        observation_mode: str = "full",
    ) -> None:
        super().__init__()
        self._validate_config(S0, K, T, r, sigma, dt, kappa, sigma_range, risk_aversion)
        if market_model not in _MARKET_MODELS:
            raise ConfigurationError(
                f"market_model must be one of {sorted(_MARKET_MODELS)}, got {market_model!r}."
            )

        self.S0 = float(S0)
        self.K = float(K)
        self.T = int(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self._sigma_base = float(sigma)
        self.dt = float(dt)
        self.kappa = float(kappa)
        self.randomise_params = randomise_params
        self.sigma_range = (float(sigma_range[0]), float(sigma_range[1]))
        self.risk_aversion = float(risk_aversion)
        self.liquidate_at_expiry = liquidate_at_expiry
        self.include_financing = include_financing
        self.market_model = market_model
        self.market_model_kwargs = self._validate_market_model_kwargs(
            market_model, market_model_kwargs
        )

        if observation_mode not in self._OBS_MODES:
            raise ConfigurationError(
                f"observation_mode must be one of {sorted(self._OBS_MODES)}, "
                f"got {observation_mode!r}."
            )
        self.observation_mode = observation_mode
        self._obs_indices = self._OBS_MODES[observation_mode]

        self._seed = seed
        self._first_reset = True
        self.state: EpisodeState | None = None
        # Per-episode caches, filled in reset(): index = step number 0..T.
        self._option_values: np.ndarray = np.empty(0)
        self._bs_deltas: np.ndarray = np.empty(0)

        obs_idx = self._OBS_MODES[observation_mode]
        self.observation_space = spaces.Box(
            low=self._OBS_LOW[obs_idx].copy(),
            high=self._OBS_HIGH[obs_idx].copy(),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_market_model_kwargs(
        market_model: str, market_model_kwargs: dict[str, Any] | None
    ) -> dict[str, float]:
        """Validate model-specific kwargs before any episode reset."""
        if market_model_kwargs is None:
            return {}
        if not isinstance(market_model_kwargs, dict):
            raise ConfigurationError("market_model_kwargs must be a dictionary.")

        allowed = {
            "gbm": set(),
            "heston": {"v0", "kappa", "theta", "xi", "rho"},
            "merton": {"lam", "mu_J", "sigma_J"},
        }[market_model]
        unknown = set(market_model_kwargs) - allowed
        if unknown:
            raise ConfigurationError(
                f"Unknown market_model_kwargs for {market_model!r}: "
                f"{sorted(unknown, key=str)}; allowed keys are {sorted(allowed)}."
            )

        values: dict[str, float] = {}
        for key, value in market_model_kwargs.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"market_model_kwargs[{key!r}] must be numeric, got {value!r}."
                ) from exc
            if not math.isfinite(numeric_value):
                raise ConfigurationError(
                    f"market_model_kwargs[{key!r}] must be finite, got {value!r}."
                )
            values[key] = numeric_value

        if market_model == "heston":
            if "v0" in values and values["v0"] <= 0:
                raise ConfigurationError("Heston v0 must be > 0.")
            if "kappa" in values and values["kappa"] <= 0:
                raise ConfigurationError(
                    "Heston market_model_kwargs['kappa'] must be > 0."
                )
            if "theta" in values and values["theta"] <= 0:
                raise ConfigurationError("Heston theta must be > 0.")
            if "xi" in values and values["xi"] <= 0:
                raise ConfigurationError("Heston xi must be > 0.")
            if "rho" in values and not -1.0 <= values["rho"] <= 1.0:
                raise ConfigurationError("Heston rho must be in [-1, 1].")
        elif market_model == "merton":
            if "lam" in values and values["lam"] < 0:
                raise ConfigurationError("Merton lam must be >= 0.")
            if "sigma_J" in values and values["sigma_J"] < 0:
                raise ConfigurationError("Merton sigma_J must be >= 0.")
        return values

    @classmethod
    def _validate_config(
        cls,
        S0: float,
        K: float,
        T: int,
        r: float,
        sigma: float,
        dt: float,
        kappa: float,
        sigma_range: tuple[float, float],
        risk_aversion: float,
    ) -> None:
        """Fail fast on parameters that would corrupt P&L or the Box bounds."""
        finite = all(
            math.isfinite(x) for x in (S0, K, r, sigma, dt, kappa, risk_aversion)
        )
        if not finite:
            raise ConfigurationError("All numeric parameters must be finite.")
        if S0 <= 0 or K <= 0:
            raise ConfigurationError(f"S0 and K must be > 0 (got {S0}, {K}).")
        if int(T) != T or T < 1:
            raise ConfigurationError(f"T must be a positive integer (got {T}).")
        if dt <= 0:
            raise ConfigurationError(f"dt must be > 0 (got {dt}).")
        if kappa < 0:
            raise ConfigurationError(f"kappa must be >= 0 (got {kappa}).")
        if risk_aversion < 0:
            raise ConfigurationError(
                f"risk_aversion must be >= 0 (got {risk_aversion})."
            )
        lo_s, hi_s = float(cls._OBS_LOW[2]), float(cls._OBS_HIGH[2])
        lo_r, hi_r = float(cls._OBS_LOW[3]), float(cls._OBS_HIGH[3])
        lo_k, hi_k = float(cls._OBS_LOW[4]), float(cls._OBS_HIGH[4])
        if not lo_r <= r <= hi_r:
            raise ConfigurationError(
                f"r={r} is outside the observation bounds [{lo_r}, {hi_r}]; "
                "widen _OBS_LOW/_OBS_HIGH deliberately instead of clipping silently."
            )
        if not lo_k <= K / S0 <= hi_k:
            raise ConfigurationError(
                f"K/S0={K / S0:.3f} is outside the observation bounds [{lo_k}, {hi_k}]."
            )
        if not lo_s <= sigma <= hi_s:
            raise ConfigurationError(f"sigma={sigma} outside [{lo_s}, {hi_s}].")
        if len(sigma_range) != 2 or not lo_s <= sigma_range[0] < sigma_range[1] <= hi_s:
            raise ConfigurationError(
                f"sigma_range must satisfy {lo_s} <= low < high <= {hi_s} "
                f"(got {sigma_range})."
            )

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        ``seed`` makes the episode exactly reproducible (path = GBM(seed)).
        With ``seed=None`` a fresh path seed is drawn from ``self.np_random``;
        the constructor seed, if any, seeds that stream on the first reset.
        ``options`` is accepted for API compatibility and currently unused.
        """
        if seed is None and self._first_reset and self._seed is not None:
            super().reset(seed=self._seed)
        else:
            super().reset(seed=seed)
        self._first_reset = False

        if self.randomise_params:
            self.sigma = float(
                self.np_random.uniform(self.sigma_range[0], self.sigma_range[1])
            )
        else:
            self.sigma = self._sigma_base

        path_seed = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(0, _MAX_SEED))
        )
        price_path = self._generate_price_path(path_seed)

        # One vectorised pass over all T+1 dates (replaces 2 scalar calls/step).
        t_rem = (self.T - np.arange(self.T + 1)) * self.dt
        self._option_values = np.asarray(
            call_price(price_path, self.K, self.r, self.sigma, t_rem), dtype=float
        )
        self._bs_deltas = np.asarray(
            call_delta(
                price_path,
                self.K,
                self.r,
                self.sigma,
                np.maximum(t_rem, _MIN_TIME_FOR_DELTA),
            ),
            dtype=float,
        )

        self.state = EpisodeState(
            price_path=price_path,
            S0=self.S0,
            K=self.K,
            r=self.r,
            sigma=self.sigma,
            T_days=self.T,
            dt=self.dt,
            step=0,
            hedge_pos=0.0,
            option_value=float(self._option_values[0]),
            cumulative_pnl=0.0,
            cumulative_cost=0.0,
        )
        reset_info = {
            "S": float(self.state.S),
            "bs_delta": float(self._bs_deltas[0]),
            "hedge_pos": float(self.state.hedge_pos),
            "option_value": float(self.state.option_value),
        }
        return self._build_obs(), reset_info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.state.is_terminal:
            raise RuntimeError("Episode has finished; call reset().")

        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size != 1:
            raise ValueError(f"Expected a scalar action, got shape {a.shape}.")
        if not np.isfinite(a[0]):
            raise ValueError(f"Action must be finite, got {a[0]!r}.")
        target_hedge = float(np.clip(a[0], 0.0, 1.0))

        S_prev = self.state.S
        option_value_prev = self.state.option_value

        trade = target_hedge - self.state.hedge_pos
        cost = float(proportional_cost(trade, S_prev, self.kappa))
        financing = (
            self.r * self.dt * target_hedge * S_prev if self.include_financing else 0.0
        )

        self.state.hedge_pos = target_hedge
        self.state.step += 1
        S_next = self.state.S
        terminated = self.state.is_terminal

        # Index T holds intrinsic value (BS at tau=0), so no special case.
        option_value_next = float(self._option_values[self.state.step])
        self.state.option_value = option_value_next

        if terminated and self.liquidate_at_expiry:
            cost += float(proportional_cost(target_hedge, S_next, self.kappa))

        raw_pnl = (
            compute_step_reward(
                option_value_prev=option_value_prev,
                option_value_next=option_value_next,
                hedge_pos=target_hedge,
                S_prev=S_prev,
                S_next=S_next,
                transaction_cost=cost,
            )
            - financing
        )
        reward = raw_pnl - self.risk_aversion * raw_pnl * raw_pnl

        self.state.cumulative_pnl += raw_pnl
        self.state.cumulative_cost += cost

        info = self._build_info(cost=cost, step_pnl=raw_pnl)
        info["financing_cost"] = financing
        info["shaped_reward"] = reward
        return self._build_obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _generate_price_path(self, seed: int) -> np.ndarray:
        """Generate a price path using the configured market model."""
        if self.market_model == "gbm":
            sim = GBMSimulator(
                S0=self.S0, mu=self.r, sigma=self.sigma, dt=self.dt, seed=seed
            )
        elif self.market_model == "heston":
            # Defaults: v0=sigma^2, Heston kappa=2, theta=sigma^2,
            # rho=-0.7. Derive xi when omitted to satisfy Feller while
            # retaining xi=0.3 at the legacy sigma=0.20 default.
            kwargs = {
                "v0": self.sigma ** 2,
                "kappa": 2.0,
                "theta": self.sigma ** 2,
                "rho": -0.7,
            }
            kwargs.update(self.market_model_kwargs)
            if "xi" not in self.market_model_kwargs:
                kwargs["xi"] = min(
                    0.3, np.sqrt(2.0 * kwargs["kappa"] * kwargs["theta"])
                )
            sim = HestonSimulator(  # type: ignore[assignment]
                S0=self.S0,
                mu=self.r,
                dt=self.dt,
                seed=seed,
                **kwargs,
            )
        elif self.market_model == "merton":
            # Defaults: lam=1 jump/yr, mu_J=-0.05, sigma_J=0.1
            kwargs = {
                "lam": 1.0,
                "mu_J": -0.05,
                "sigma_J": 0.10,
            }
            kwargs.update(self.market_model_kwargs)
            sim = MertonJumpDiffusionSimulator(  # type: ignore[assignment]
                S0=self.S0,
                mu=self.r,
                sigma=self.sigma,
                dt=self.dt,
                seed=seed,
                **kwargs,
            )
        else:
            raise ConfigurationError(f"Unknown market_model: {self.market_model!r}")
        return sim.generate_path(self.T)

    def _build_obs(self) -> np.ndarray:
        """Build the observation vector from the current episode state.

        The full 7-d feature vector is constructed first, then sliced
        according to ``self.observation_mode``.
        """
        assert self.state is not None  # internal invariant, not user input
        full_obs = np.array(
            [
                self.state.S / self.S0,
                self.state.time_remaining,
                self.sigma,
                self.r,
                self.K / self.S0,
                self.state.hedge_pos,
                self._bs_deltas[self.state.step],
            ],
            dtype=np.float32,
        )
        obs = np.clip(full_obs, self._OBS_LOW, self._OBS_HIGH)[self._obs_indices]
        return obs

    def _build_info(self, cost: float, step_pnl: float) -> dict[str, Any]:
        """Per-step diagnostics (raw, un-shaped P&L)."""
        assert self.state is not None
        info: dict[str, Any] = {
            "step": self.state.step,
            "S": self.state.S,
            "bs_delta": float(self._bs_deltas[self.state.step]),
            "hedge_pos": self.state.hedge_pos,
            "option_value": self.state.option_value,
            "transaction_cost": cost,
            "step_pnl": step_pnl,
            "cumulative_pnl": self.state.cumulative_pnl,
            "cumulative_cost": self.state.cumulative_cost,
        }
        if self.state.is_terminal:
            info["terminal_pnl"] = self.state.cumulative_pnl
            info["total_cost"] = self.state.cumulative_cost
        return info
