"""New environment switches survive YAML -> dataclass -> HedgingEnv."""

from __future__ import annotations

import textwrap

from src.envs.hedging_env import HedgingEnv
from src.training.hyperparams import EnvironmentConfig, load_config


def test_defaults_preserve_legacy_behaviour():
    cfg = EnvironmentConfig()
    assert (cfg.risk_aversion, cfg.liquidate_at_expiry, cfg.include_financing) == (
        0.0,
        False,
        False,
    )
    env = HedgingEnv(**cfg.as_dict())
    assert env.risk_aversion == 0.0


def test_yaml_values_reach_the_environment(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        textwrap.dedent("""
        run: {name: t, seed: 1, total_timesteps: 10, n_envs: 1}
        environment:
          risk_aversion: 0.5
          liquidate_at_expiry: true
          include_financing: true
        callbacks: {eval_seed_base: 10000}
    """)
    )
    cfg = load_config(str(p))
    env = HedgingEnv(**cfg.environment.as_dict())
    assert (env.risk_aversion, env.liquidate_at_expiry, env.include_financing) == (
        0.5,
        True,
        True,
    )
    assert cfg.callbacks.eval_seed_base == 10_000
