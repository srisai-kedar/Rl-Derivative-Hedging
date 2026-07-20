"""Tests for the Gymnasium hedging environment."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.envs.reward import compute_step_reward


def test_env_checker(env):
    """Official Gymnasium environment checker must pass with zero warnings."""
    check_env(env, warn=True, skip_render_check=True)


def test_obs_within_bounds(env):
    """Observation must always stay inside declared observation_space bounds."""
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs), f"Initial obs out of bounds: {obs}"

    for _ in range(env.T):
        action = env.action_space.sample()
        obs, _, terminated, _, _ = env.step(action)
        assert env.observation_space.contains(obs), f"Obs out of bounds: {obs}"
        if terminated:
            break


def test_episode_terminates_at_T(env):
    """Episode must terminate exactly at step T. Never before. Never after."""
    env.reset(seed=0)
    step_count = 0
    terminated = False
    while not terminated:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        step_count += 1
        assert not truncated, "Episodes should never be truncated"
    assert step_count == env.T, f"Expected {env.T} steps, got {step_count}"


def test_no_nan_in_rewards(env):
    """No step should produce NaN or Inf reward over 50 random episodes."""
    for episode in range(50):
        env.reset(seed=episode)
        for _ in range(env.T):
            _, reward, terminated, _, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward), f"Non-finite reward at episode {episode}"
            if terminated:
                break


def test_no_nan_in_observations(env):
    """No step should produce NaN or Inf observation values."""
    for episode in range(50):
        obs, _ = env.reset(seed=episode)
        assert np.all(np.isfinite(obs)), f"Non-finite initial obs at episode {episode}"
        for _ in range(env.T):
            obs, _, terminated, _, _ = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs)), f"Non-finite obs at episode {episode}"
            if terminated:
                break


def test_time_remaining_decreases_monotonically(env):
    """Observation index 1 (time remaining) must decrease from ~1.0 to ~0.0."""
    env.reset(seed=0)
    prev_time = 1.0
    for i in range(env.T):
        obs, _, terminated, _, _ = env.step(np.array([0.5], dtype=np.float32))
        time_remaining = float(obs[1])
        assert time_remaining < prev_time, f"Time remaining did not decrease at step {i}"
        prev_time = time_remaining
        if terminated:
            break
    assert prev_time < 1e-6, "Time remaining should reach ~0 at final step"


def test_bs_delta_in_obs_matches_manual(env):
    """Observation index 6 must match manually computed BS delta."""
    from src.finance.black_scholes import call_delta

    obs, _ = env.reset(seed=0)
    S = float(obs[0]) * env.S0
    T_rem = float(obs[1]) * env.T * env.dt
    expected_delta = float(call_delta(S, env.K, env.r, env.sigma, T_rem))
    actual_delta = float(obs[6])

    np.testing.assert_allclose(
        actual_delta,
        expected_delta,
        rtol=1e-4,
        err_msg="BS delta in obs does not match manual calculation",
    )


def test_bs_delta_baseline_beats_random(env):
    """
    A Black-Scholes delta hedge should produce lower P&L std than random actions.
    Run 200 episodes each. Test that std(pnl_bs) < std(pnl_random).
    """
    def run_episodes(policy_fn, n=200):
        pnls = []
        for seed in range(n):
            obs, _ = env.reset(seed=seed)
            total_pnl = 0.0
            for _ in range(env.T):
                action = policy_fn(obs, env)
                obs, reward, terminated, _, _ = env.step(action)
                total_pnl += reward
                if terminated:
                    break
            pnls.append(total_pnl)
        return np.array(pnls)

    def bs_policy(obs, env):
        bs_d = obs[6]
        return np.array([bs_d], dtype=np.float32)

    def random_policy(obs, env):
        return env.action_space.sample()

    pnl_bs = run_episodes(bs_policy)
    pnl_random = run_episodes(random_policy)

    std_bs = np.std(pnl_bs)
    std_random = np.std(pnl_random)

    assert std_bs < std_random, (
        f"BS delta std {std_bs:.4f} is not less than random std {std_random:.4f}. "
        f"Reward function or timing logic is likely wrong."
    )


def test_zero_cost_bs_pnl_variance(zero_cost_env):
    """
    With zero transaction costs, BS delta hedging should produce near-zero P&L
    each step (residual from discrete rebalancing only).
    The std of terminal P&L should be much smaller than option value.
    """
    from src.finance.black_scholes import call_price

    env = zero_cost_env
    pnls = []
    for seed in range(500):
        obs, _ = env.reset(seed=seed)
        total_pnl = 0.0
        for _ in range(env.T):
            bs_d = float(obs[6])
            action = np.array([bs_d], dtype=np.float32)
            obs, reward, terminated, _, _ = env.step(action)
            total_pnl += reward
            if terminated:
                break
        pnls.append(total_pnl)

    pnl_std = np.std(pnls)
    initial_option_value = call_price(100.0, 100.0, 0.05, 0.20, 30 / 252)

    assert pnl_std < 0.20 * initial_option_value, (
        f"Zero-cost BS P&L std {pnl_std:.4f} is too high vs "
        f"option value {initial_option_value:.4f}. "
        f"Reward accounting is likely wrong."
    )


def test_out_of_bounds_action_handled(env):
    """Actions outside [0, 1] must not crash the environment."""
    env.reset(seed=0)
    env.step(np.array([1.5], dtype=np.float32))
    env.step(np.array([-0.5], dtype=np.float32))
    env.step(np.array([0.5], dtype=np.float32))


def test_same_seed_produces_same_episode(env):
    """Same seed must produce identical price paths and rewards."""
    def run_one(seed):
        env.reset(seed=seed)
        rewards = []
        for _ in range(env.T):
            action = np.array([0.5], dtype=np.float32)
            _, r, terminated, _, _ = env.step(action)
            rewards.append(r)
            if terminated:
                break
        return rewards

    r1 = run_one(seed=42)
    r2 = run_one(seed=42)
    np.testing.assert_array_equal(
        r1,
        r2,
        err_msg="Same seed produced different reward sequences",
    )


def test_different_seeds_produce_different_episodes(env):
    """Different seeds must produce different price paths."""
    def first_step_reward(seed):
        env.reset(seed=seed)
        _, reward, _, _, _ = env.step(np.array([0.5], dtype=np.float32))
        return reward

    r0 = first_step_reward(0)
    r1 = first_step_reward(1)
    assert not np.isclose(r0, r1), (
        "Different seeds produced identical first-step rewards"
    )


def test_info_dict_has_required_keys(env):
    """Info dict must contain all required keys at every step."""
    required_keys = {
        "step",
        "S",
        "bs_delta",
        "hedge_pos",
        "option_value",
        "transaction_cost",
        "step_pnl",
        "cumulative_pnl",
        "cumulative_cost",
    }
    terminal_keys = {"terminal_pnl", "total_cost"}

    env.reset(seed=0)
    for i in range(env.T):
        _, _, terminated, _, info = env.step(env.action_space.sample())
        assert required_keys.issubset(info.keys()), (
            f"Missing keys at step {i}: {required_keys - info.keys()}"
        )
        if terminated:
            assert terminal_keys.issubset(info.keys()), (
                f"Missing terminal keys: {terminal_keys - info.keys()}"
            )
            break


def test_terminated_only_at_final_step(env):
    """terminated must be False for steps 1..T-1 and True only at step T."""
    env.reset(seed=0)
    for i in range(env.T - 1):
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert not terminated, f"Episode terminated early at step {i + 1}"
    _, _, terminated, _, _ = env.step(env.action_space.sample())
    assert terminated, "Episode did not terminate at final step"


def test_transaction_cost_non_negative(env):
    """Transaction costs must always be non-negative."""
    env.reset(seed=0)
    for _ in range(env.T):
        _, _, terminated, _, info = env.step(env.action_space.sample())
        assert info["transaction_cost"] >= 0.0, (
            f"Negative transaction cost: {info['transaction_cost']}"
        )
        if terminated:
            break


def test_reward_function_direction():
    """Unit test the reward function independently from the environment."""
    r = compute_step_reward(
        option_value_prev=5.0,
        option_value_next=4.5,
        hedge_pos=0.5,
        S_prev=100.0,
        S_next=101.0,
        transaction_cost=0.0,
    )
    assert r > 0, f"Expected positive reward, got {r}"

    r = compute_step_reward(
        option_value_prev=5.0,
        option_value_next=5.5,
        hedge_pos=0.5,
        S_prev=100.0,
        S_next=99.0,
        transaction_cost=0.0,
    )
    assert r < 0, f"Expected negative reward, got {r}"
