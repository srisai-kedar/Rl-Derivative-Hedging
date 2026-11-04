"""Observation-space compatibility checks for checkpoint loading."""

from __future__ import annotations

from typing import Any

import numpy as np


def _space_dimension(space: Any, label: str) -> int:
    shape = getattr(space, "shape", None)
    if shape is None or len(shape) != 1:
        raise ValueError(f"{label} observation space must be one-dimensional, got {shape!r}")
    return int(shape[0])


def model_observation_dimension(model: Any) -> int:
    """Return the flat observation dimension declared by an SB3 model."""
    return _space_dimension(model.observation_space, "model")


def vecnormalize_observation_dimension(vec_normalize: Any) -> int:
    """Return the flat observation dimension exposed by VecNormalize."""
    return _space_dimension(vec_normalize.observation_space, "VecNormalize")


def vecnormalize_stats_dimension(vec_normalize: Any) -> int:
    """Return the observation dimension stored in VecNormalize running stats."""
    mean = np.asarray(vec_normalize.obs_rms.mean)
    if mean.ndim != 1:
        raise ValueError(
            "VecNormalize observation statistics must be one-dimensional, "
            f"got shape {mean.shape!r}"
        )
    return int(mean.shape[0])


def validate_observation_dimensions(
    *,
    env_dimension: int,
    model_dimension: int | None,
    vecnorm_dimension: int,
    observation_mode: str,
) -> None:
    """Raise a clear error when checkpoint and environment dimensions differ."""
    dimensions = {
        "env": env_dimension,
        "model": model_dimension if model_dimension is not None else "not provided",
        "VecNormalize stats": vecnorm_dimension,
    }
    mismatches = [
        f"{label}={dimension}"
        for label, dimension in dimensions.items()
        if dimension != "not provided" and dimension != env_dimension
    ]
    if mismatches:
        detail = ", ".join(
            f"{label}={dimension}" for label, dimension in dimensions.items()
        )
        raise ValueError(
            "Observation dimension mismatch for "
            f"observation_mode={observation_mode!r}: {detail}."
        )


def infer_observation_mode(vec_normalize: Any) -> str:
    """Best-effort mode lookup for direct ``RLAgentPolicy`` construction."""
    try:
        values = vec_normalize.venv.get_attr("observation_mode")
        if values:
            return str(values[0])
    except (AttributeError, IndexError, TypeError):
        pass
    return "unknown"
