"""Tests for run manifest creation, integrity verification, and tamper detection."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.utils.manifest import (
    IntegrityError,
    ManifestConfigMismatchError,
    ManifestRequiredError,
    canonicalize_config,
    compute_file_sha256,
    get_git_info,
    validate_manifest_environment,
    verify_checkpoint_integrity,
    write_manifest,
)


def test_manifest_round_trip(tmp_path):
    """Manifest is written to disk and round-trips all required metadata."""
    dummy_model = tmp_path / "model.zip"
    dummy_vecnorm = tmp_path / "vecnorm.pkl"
    dummy_model.write_bytes(b"dummy_model_bytes_12345")
    dummy_vecnorm.write_bytes(b"dummy_vecnorm_bytes_67890")

    cfg = {"run": {"seed": 42}, "env": {"T": 30}}
    manifest_path = write_manifest(
        output_dir=tmp_path,
        run_type="training",
        config=cfg,
        seed=42,
        checkpoint_files=[dummy_model, dummy_vecnorm],
    )
    assert manifest_path.exists()

    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["run_type"] == "training"
    assert data["seed"] == 42
    assert "git" in data
    assert "system" in data
    assert "dependencies" in data
    assert "numpy" in data["dependencies"]
    assert "config" in data
    assert "sha256" in data["config"]
    assert data["checkpoint_hashes"]["model.zip"] == compute_file_sha256(dummy_model)
    assert data["checkpoint_hashes"]["vecnorm.pkl"] == compute_file_sha256(dummy_vecnorm)


def test_tamper_detection(tmp_path):
    """Integrity verification detects altered checkpoint files and raises IntegrityError."""
    model_file = tmp_path / "best_model.zip"
    vecnorm_file = tmp_path / "best_vecnorm.pkl"
    model_file.write_bytes(b"original_model_content")
    vecnorm_file.write_bytes(b"original_vecnorm_content")

    write_manifest(
        output_dir=tmp_path,
        run_type="training",
        config={"seed": 42},
        checkpoint_files=[model_file, vecnorm_file],
    )

    # Clean verification passes
    verify_checkpoint_integrity(model_file, vecnorm_file)

    # Tamper with model file
    model_file.write_bytes(b"tampered_model_content_evil")

    with pytest.raises(IntegrityError, match="Checkpoint integrity verification failed"):
        verify_checkpoint_integrity(model_file, vecnorm_file)

    # With allow_unverified=True, raises no exception
    verify_checkpoint_integrity(model_file, vecnorm_file, allow_unverified=True)


def test_missing_hash_is_not_silent(tmp_path):
    """A manifest must cover every checkpoint file supplied for verification."""
    model_file = tmp_path / "model.zip"
    vecnorm_file = tmp_path / "vecnorm.pkl"
    model_file.write_bytes(b"model")
    vecnorm_file.write_bytes(b"vecnorm")
    write_manifest(tmp_path, "training", {}, checkpoint_files=[model_file])

    with pytest.raises(ManifestRequiredError, match="No SHA-256 hash recorded"):
        verify_checkpoint_integrity(model_file, vecnorm_file)


def test_require_manifest_fails_for_legacy_checkpoint(tmp_path):
    """Strict mode rejects a legacy checkpoint directory without a manifest."""
    model_file = tmp_path / "model.zip"
    vecnorm_file = tmp_path / "vecnorm.pkl"
    model_file.write_bytes(b"model")
    vecnorm_file.write_bytes(b"vecnorm")

    with pytest.raises(ManifestRequiredError, match="Required run_manifest.json"):
        verify_checkpoint_integrity(
            model_file,
            vecnorm_file,
            require_manifest=True,
        )


def test_out_of_band_hash_detects_manifest_replacement(tmp_path):
    """An attacker cannot replace both a checkpoint and its adjacent manifest."""
    model_file = tmp_path / "model.zip"
    vecnorm_file = tmp_path / "vecnorm.pkl"
    model_file.write_bytes(b"trusted model")
    vecnorm_file.write_bytes(b"trusted vecnorm")
    expected = {
        model_file.name: compute_file_sha256(model_file),
        vecnorm_file.name: compute_file_sha256(vecnorm_file),
    }
    write_manifest(tmp_path, "training", {}, checkpoint_files=[model_file, vecnorm_file])

    model_file.write_bytes(b"replacement model")
    write_manifest(tmp_path, "training", {}, checkpoint_files=[model_file, vecnorm_file])

    with pytest.raises(IntegrityError, match="Out-of-band SHA-256"):
        verify_checkpoint_integrity(model_file, vecnorm_file, expected_sha256=expected)


def test_config_hash_is_order_stable_and_rejects_unserialisable_values():
    """Canonical JSON is order-independent and never falls back to str(obj)."""
    _, hash_a = canonicalize_config({"b": {"y": 2, "x": 1}, "a": [1, 2]})
    _, hash_b = canonicalize_config({"a": [1, 2], "b": {"x": 1, "y": 2}})
    assert hash_a == hash_b

    class Unsupported:
        __slots__ = ()

    with pytest.raises(TypeError, match="Unserialisable config value"):
        canonicalize_config({"unsupported": Unsupported()})


def test_legacy_checkpoint_compatibility(tmp_path):
    """Directories without run_manifest.json load safely with a warning."""
    model_file = tmp_path / "best_model.zip"
    vecnorm_file = tmp_path / "best_vecnorm.pkl"
    model_file.write_bytes(b"legacy_model")
    vecnorm_file.write_bytes(b"legacy_vecnorm")

    # Should not raise any error despite missing manifest
    verify_checkpoint_integrity(model_file, vecnorm_file)


def test_behaviour_outside_git_repo():
    """get_git_info gracefully reports non-git environments without crashing."""
    with patch("subprocess.check_output", side_effect=Exception("Not a git repository")):
        info = get_git_info()
        assert info["is_git_repo"] is False
        assert info["commit"] is None
        assert info["dirty"] is None


def test_load_env_for_inference_tamper_detection(tmp_path):
    """load_env_for_inference respects integrity verification."""
    from src.envs.hedging_env import HedgingEnv
    from src.training.env_factory import (
        DummyVecEnv,
        VecNormalize,
        load_env_for_inference,
    )
    from src.training.hyperparams import EnvironmentConfig

    vecnorm_file = tmp_path / "vecnorm.pkl"
    model_file = tmp_path / "model.zip"
    model_file.write_bytes(b"model_bytes")
    eval_env = DummyVecEnv([lambda: HedgingEnv()])
    vn = VecNormalize(eval_env)
    vn.save(str(vecnorm_file))

    write_manifest(
        output_dir=tmp_path,
        run_type="training",
        config={},
        checkpoint_files=[model_file, vecnorm_file],
    )

    # Valid load passes
    loaded_vn = load_env_for_inference(
        EnvironmentConfig(),
        str(vecnorm_file),
        checkpoint_path=str(model_file),
    )
    loaded_vn.close()

    # Tamper with the model file; inference loading must verify both files.
    model_file.write_bytes(b"corrupted_model_bytes")

    with pytest.raises(IntegrityError):
        load_env_for_inference(
            EnvironmentConfig(),
            str(vecnorm_file),
            checkpoint_path=str(model_file),
        )

    # allow_unverified=True bypasses the integrity error (though pickle might fail if bytes are garbage)
    eval_env.close()


def test_legacy_vecnormalize_still_loads_without_manifest(tmp_path, caplog):
    """Default compatibility mode still loads a legacy VecNormalize file."""
    from src.envs.hedging_env import HedgingEnv
    from src.training.env_factory import (
        DummyVecEnv,
        VecNormalize,
        load_env_for_inference,
    )
    from src.training.hyperparams import EnvironmentConfig

    vecnorm_file = tmp_path / "legacy_vecnorm.pkl"
    eval_env = DummyVecEnv([lambda: HedgingEnv()])
    vecnorm = VecNormalize(eval_env)
    vecnorm.save(str(vecnorm_file))

    loaded = load_env_for_inference(EnvironmentConfig(), str(vecnorm_file))
    loaded.close()
    eval_env.close()
    assert "legacy unverified checkpoint" in caplog.text


def test_manifest_environment_mismatch_can_raise_or_warn(tmp_path, caplog):
    """Evaluation cannot silently use a different market model or mode."""
    from src.training.hyperparams import EnvironmentConfig

    write_manifest(
        output_dir=tmp_path,
        run_type="training",
        config={
            "eval_environment": {
                "observation_mode": "full",
                "market_model": "gbm",
            }
        },
    )
    config = EnvironmentConfig(observation_mode="no_bs_delta", market_model="heston")

    with pytest.raises(ManifestConfigMismatchError, match="observation_mode"):
        validate_manifest_environment(tmp_path / "vecnorm.pkl", config)

    validate_manifest_environment(tmp_path / "vecnorm.pkl", config, allow_mismatch=True)
    assert "Continuing because allow_mismatch=True" in caplog.text
