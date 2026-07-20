"""Typed configuration loading for SAC training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class EnvironmentConfig:
    """Parameters passed to ``HedgingEnv``."""

    S0: float = 100.0
    K: float = 100.0
    T: int = 30
    r: float = 0.05
    sigma: float = 0.20
    dt: float = 1 / 252
    kappa: float = 0.001
    randomise_params: bool = False
    sigma_range: tuple[float, float] = (0.10, 0.40)

    def as_dict(self) -> dict[str, Any]:
        """Return environment parameters suitable for ``HedgingEnv(**kwargs)``."""
        return {
            "S0": self.S0,
            "K": self.K,
            "T": self.T,
            "r": self.r,
            "sigma": self.sigma,
            "dt": self.dt,
            "kappa": self.kappa,
            "randomise_params": self.randomise_params,
            "sigma_range": tuple(self.sigma_range),
        }


@dataclass
class SACConfig:
    """Stable-Baselines3 SAC hyperparameters."""

    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    buffer_size: int = 500_000
    learning_starts: int = 10_000
    batch_size: int = 512
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str | float = "auto"
    policy_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"net_arch": [256, 256]}
    )


@dataclass
class VecNormalizeConfig:
    """VecNormalize options for training environments."""

    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0


@dataclass
class CallbackConfig:
    """Training callback cadence and verbosity."""

    eval_freq: int = 50_000
    n_eval_episodes: int = 100
    save_freq: int = 100_000
    verbose: int = 1


@dataclass
class RunConfig:
    """Run-level training parameters."""

    name: str = "sac_hedging"
    seed: int = 42
    total_timesteps: int = 2_000_000
    n_envs: int = 4
    progress_bar: bool = True


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    run: RunConfig = field(default_factory=RunConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    eval_environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    vec_normalize: VecNormalizeConfig = field(default_factory=VecNormalizeConfig)
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"
    tb_log_name: str = "SAC"


def _coerce_sigma_range(raw: dict[str, Any]) -> dict[str, Any]:
    if "sigma_range" in raw:
        raw = {**raw, "sigma_range": tuple(raw["sigma_range"])}
    return raw


def load_config(path: str) -> TrainingConfig:
    """
    Load a YAML training config into typed dataclasses.

    ``eval_environment`` inherits the main environment block and overrides only
    the keys it declares, so evaluation stays fixed while training can randomise.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise KeyError("Training config is empty")

    env_raw = _coerce_sigma_range(raw.get("environment", {}))
    eval_env_raw = _coerce_sigma_range({**env_raw, **raw.get("eval_environment", {})})

    sac_raw = raw.get("sac", {})
    if "policy_kwargs" not in sac_raw:
        sac_raw = {**sac_raw, "policy_kwargs": {"net_arch": [256, 256]}}

    paths = raw.get("paths", {})

    return TrainingConfig(
        run=RunConfig(**raw.get("run", {})),
        environment=EnvironmentConfig(**env_raw),
        eval_environment=EnvironmentConfig(**eval_env_raw),
        sac=SACConfig(**sac_raw),
        vec_normalize=VecNormalizeConfig(**raw.get("vec_normalize", {})),
        callbacks=CallbackConfig(**raw.get("callbacks", {})),
        checkpoints_dir=paths.get("checkpoints_dir", "checkpoints"),
        results_dir=paths.get("results_dir", "results"),
        tb_log_name=paths.get("tb_log_name", "SAC"),
    )
