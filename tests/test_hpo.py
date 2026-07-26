"""Smoke tests for the Optuna HPO module."""

from __future__ import annotations

import pytest
import optuna


def test_suggest_sac_params_returns_required_keys():
    """suggest_sac_params must return all expected keys."""
    from src.training.hpo import suggest_sac_params

    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    params = suggest_sac_params(trial)
    required_keys = {
        "learning_rate",
        "batch_size",
        "tau",
        "gamma",
        "ent_coef",
        "net_arch_str",
    }
    assert required_keys.issubset(params.keys()), (
        f"Missing keys: {required_keys - params.keys()}"
    )


def test_net_arch_from_str():
    """_net_arch_from_str converts encoded strings to layer size lists."""
    from src.training.hpo import _net_arch_from_str

    assert _net_arch_from_str("64-64") == [64, 64]
    assert _net_arch_from_str("256-256-128") == [256, 256, 128]
    assert _net_arch_from_str("128-128") == [128, 128]


def test_learning_rate_in_valid_range():
    """Trial-suggested learning rate must be in [1e-5, 1e-3]."""
    from src.training.hpo import suggest_sac_params

    study = optuna.create_study(direction="maximize")
    for _ in range(10):
        trial = study.ask()
        params = suggest_sac_params(trial)
        assert 1e-5 <= params["learning_rate"] <= 1e-3


@pytest.mark.slow
def test_hpo_objective_single_trial(fast_training_config):
    """One HPO trial must complete without error and return a finite float."""
    import src.training.hpo as hpo_module
    from src.training.hpo import objective

    original_steps = hpo_module.HPO_TIMESTEPS
    original_eps = hpo_module.HPO_N_EVAL_EPISODES
    hpo_module.HPO_TIMESTEPS = 500
    hpo_module.HPO_N_EVAL_EPISODES = 5

    try:
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        result = objective(trial, fast_training_config)
        assert isinstance(result, float)
        assert result == result
    finally:
        hpo_module.HPO_TIMESTEPS = original_steps
        hpo_module.HPO_N_EVAL_EPISODES = original_eps


@pytest.mark.slow
def test_run_hpo_single_trial(fast_training_config):
    """run_hpo with n_trials=1 must return a valid optuna.Study."""
    import src.training.hpo as hpo_module
    from src.training.hpo import run_hpo

    original_steps = hpo_module.HPO_TIMESTEPS
    original_eps = hpo_module.HPO_N_EVAL_EPISODES
    hpo_module.HPO_TIMESTEPS = 500
    hpo_module.HPO_N_EVAL_EPISODES = 5

    try:
        study = run_hpo(
            config_path=fast_training_config,
            n_trials=1,
            study_name="test_hpo",
            storage=None,
        )
        assert isinstance(study, optuna.Study)
        assert len(study.trials) == 1
        assert study.best_value is not None
    finally:
        hpo_module.HPO_TIMESTEPS = original_steps
        hpo_module.HPO_N_EVAL_EPISODES = original_eps
