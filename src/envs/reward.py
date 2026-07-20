"""Pure reward functions for hedging environment step P&L."""

from __future__ import annotations

import numpy as np


def compute_step_reward(
    option_value_prev: float,
    option_value_next: float,
    hedge_pos: float,
    S_prev: float,
    S_next: float,
    transaction_cost: float,
) -> float:
    """
    P&L for one step of a short-call / long-stock hedged portfolio.

    The desk is SHORT the call and LONG (hedge_pos) shares.
    Over one period [t, t+1]:
      - Option liability changes by (C_next - C_prev) — this is a COST if option rises
      - Stock position gains hedge_pos * (S_next - S_prev)
      - Transaction cost is paid for rebalancing

    P&L = -(C_next - C_prev) + hedge_pos * (S_next - S_prev) - transaction_cost
        = (C_prev - C_next) + hedge_pos * (S_next - S_prev) - transaction_cost

    A positive reward means the hedged portfolio gained value this step.
    A perfect delta hedge in a frictionless BS world gives reward ≈ 0 each step.
    In reality, residual P&L from discrete rebalancing and costs accumulates.
    """
    option_change = option_value_prev - option_value_next
    stock_pnl = hedge_pos * (S_next - S_prev)
    return float(option_change + stock_pnl - transaction_cost)


def compute_terminal_reward(
    option_value_prev: float,
    S_prev: float,
    S_terminal: float,
    K: float,
    hedge_pos: float,
    transaction_cost: float,
) -> float:
    """P&L for the final step of the episode using intrinsic settlement value."""
    intrinsic_value = float(np.maximum(S_terminal - K, 0.0))
    return compute_step_reward(
        option_value_prev=option_value_prev,
        option_value_next=intrinsic_value,
        hedge_pos=hedge_pos,
        S_prev=S_prev,
        S_next=S_terminal,
        transaction_cost=transaction_cost,
    )
