"""Smoke tests for the Optuna HPO module."""

from __future__ import annotations

import numpy as np
import optuna
import pytest


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


def test_hpo_seed_blocks_are_disjoint_and_multi_seeded():
    from src.training.hpo import _eval_seed_block, _trial_seeds

    eval_seeds = _eval_seed_block(5, 10_000, False)
    assert eval_seeds == [10_000, 10_001, 10_002, 10_003, 10_004]
    assert not set(eval_seeds) & set(range(10_000))
    assert _trial_seeds(42, 3, 2, False) == [48, 49]
    assert _eval_seed_block(3, 10_000, True) == [0, 1, 2]
    assert _trial_seeds(42, 3, 2, True) == [42]


def test_hpo_objective_definitions():
    from src.training.hpo import score_episode_rewards

    rewards = [1.0, 2.0, 3.0, 4.0]
    assert score_episode_rewards(rewards) == pytest.approx(2.5)
    assert score_episode_rewards(
        rewards, "mean_minus_lambda_std", 1.0
    ) == pytest.approx(2.5 - np.std(rewards))
    assert score_episode_rewards(rewards, "cvar") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="objective"):
        score_episode_rewards(rewards, "unknown")


def test_optional_environment_search_dimensions():
    from src.training.hpo import suggest_sac_params

    study = optuna.create_study(direction="maximize")
    params = suggest_sac_params(
        study.ask(), search_risk_aversion=True, search_observation_mode=True
    )
    assert 0.0 <= params["risk_aversion"] <= 1.0
    assert params["observation_mode"] in {
        "full",
        "no_greeks",
        "price_only",
        "no_bs_delta",
        "no_bs_delta_no_sigma",
    }


def test_run_hpo_resumes_and_writes_metadata(
    monkeypatch, fast_training_config, tmp_path
):
    import src.training.hpo as hpo_module
    from src.training.hpo import run_hpo

    monkeypatch.setattr(
        hpo_module, "objective", lambda trial, *args, **kwargs: float(trial.number)
    )
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    first = run_hpo(
        config_path=fast_training_config,
        n_trials=1,
        study_name="resume_test",
        storage=storage,
        eval_seed_start=10_000,
        n_eval_episodes=2,
        n_seeds_per_trial=2,
        objective_name="mean_minus_lambda_std",
        risk_lambda=0.5,
    )
    first_count = len(first.trials)
    second = run_hpo(
        config_path=fast_training_config,
        n_trials=1,
        study_name="resume_test",
        storage=storage,
        eval_seed_start=10_000,
        n_eval_episodes=2,
        n_seeds_per_trial=2,
        objective_name="mean_minus_lambda_std",
        risk_lambda=0.5,
    )
    assert first_count == 1
    assert len(second.trials) == 2
    metadata = (tmp_path / "study.db.metadata.json").read_text(encoding="utf-8")
    assert '"n_seeds_per_trial": 2' in metadata
    assert '"mean_minus_lambda_std"' in metadata


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
