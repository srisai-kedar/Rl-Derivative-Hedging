"""Gymnasium environment for learning to hedge a short European call option."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.episode_state import EpisodeState
from src.envs.reward import compute_step_reward
from src.finance.black_scholes import call_delta, call_price
from src.finance.transaction_costs import proportional_cost
from src.simulation.gbm import GBMSimulator


class HedgingEnv(gym.Env):
    """
    Gymnasium environment for learning to hedge a short European call option.

    The agent observes market state each day and decides what fraction of
    one option delta to hold in the underlying stock. The environment
    advances time by one trading day, moves the stock price by GBM, and
    returns the step P&L of the hedged portfolio as reward.

    Observation space: 7-dimensional continuous vector (see _build_obs)
    Action space: 1-dimensional continuous scalar in [0, 1]
    Episode length: n_steps (fixed, no early termination)
    """

    OBS_KEYS = [
        "price_ratio",
        "time_remaining",
        "sigma",
        "r",
        "strike_ratio",
        "hedge_pos",
        "bs_delta",
    ]

    metadata = {"render_modes": []}

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
    ) -> None:
        super().__init__()

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self._sigma_base = sigma
        self.dt = dt
        self.kappa = kappa
        self.randomise_params = randomise_params
        self.sigma_range = sigma_range
        self._seed = seed
        self._first_reset = True
        self.state: EpisodeState | None = None

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.01, 0.00, 0.5, 0.0, 0.0], dtype=np.float32),
            high=np.array([5.0, 1.0, 1.00, 0.20, 2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        if self.randomise_params:
            self.sigma = float(
                self.np_random.uniform(self.sigma_range[0], self.sigma_range[1])
            )
        else:
            self.sigma = self._sigma_base

        self._first_reset = False

        gbm_seed = seed if seed is not None else self._seed
        sim = GBMSimulator(
            S0=self.S0,
            mu=self.r,
            sigma=self.sigma,
            dt=self.dt,
            seed=gbm_seed,
        )
        price_path = sim.generate_path(self.T)

        initial_option_value = float(
            call_price(self.S0, self.K, self.r, self.sigma, self.T * self.dt)
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
            option_value=initial_option_value,
            cumulative_pnl=0.0,
            cumulative_cost=0.0,
        )

        return self._build_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.state is not None

        target_hedge = float(np.clip(action[0], 0.0, 1.0))

        delta_trade = target_hedge - self.state.hedge_pos
        cost = float(proportional_cost(delta_trade, self.state.S, self.kappa))

        S_prev = self.state.S
        option_value_prev = self.state.option_value

        self.state.hedge_pos = target_hedge

        self.state.step += 1
        S_next = self.state.S

        terminated = self.state.is_terminal
        truncated = False

        if terminated:
            option_value_next = float(np.maximum(S_next - self.K, 0.0))
        else:
            T_rem = self.state.T_remaining_years
            option_value_next = float(
                call_price(S_next, self.K, self.r, self.sigma, T_rem)
            )
        self.state.option_value = option_value_next

        reward = compute_step_reward(
            option_value_prev=option_value_prev,
            option_value_next=option_value_next,
            hedge_pos=target_hedge,
            S_prev=S_prev,
            S_next=S_next,
            transaction_cost=cost,
        )

        self.state.cumulative_pnl += reward
        self.state.cumulative_cost += cost

        obs = self._build_obs()
        info = self._build_info(cost=cost, step_pnl=reward)

        return obs, reward, terminated, truncated, info

    def _build_obs(self) -> np.ndarray:
        """Build the 7-dimensional observation vector from current episode state."""
        assert self.state is not None

        T_for_delta = max(self.state.T_remaining_years, 1e-8)
        bs_d = float(
            call_delta(
                self.state.S,
                self.K,
                self.r,
                self.sigma,
                T_for_delta,
            )
        )
        obs = np.array(
            [
                self.state.S / self.S0,
                self.state.time_remaining,
                self.sigma,
                self.r,
                self.K / self.S0,
                self.state.hedge_pos,
                bs_d,
            ],
            dtype=np.float32,
        )

        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def _build_info(self, cost: float, step_pnl: float) -> dict:
        """Build the info dict for the current step."""
        assert self.state is not None

        T_for_delta = max(self.state.T_remaining_years, 1e-8)
        info = {
            "step": self.state.step,
            "S": self.state.S,
            "bs_delta": float(
                call_delta(
                    self.state.S,
                    self.K,
                    self.r,
                    self.sigma,
                    T_for_delta,
                )
            ),
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
