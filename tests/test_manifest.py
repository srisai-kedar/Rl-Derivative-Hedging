"""Tests for run manifest creation, integrity verification, and tamper detection."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.utils.manifest import (
    IntegrityError,
    compute_file_sha256,
    get_git_info,
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
    eval_env = DummyVecEnv([lambda: HedgingEnv()])
    vn = VecNormalize(eval_env)
    vn.save(str(vecnorm_file))

    write_manifest(
        output_dir=tmp_path,
        run_type="training",
        config={},
        checkpoint_files=[vecnorm_file],
    )

    # Valid load passes
    loaded_vn = load_env_for_inference(EnvironmentConfig(), str(vecnorm_file))
    loaded_vn.close()

    # Tamper with vecnorm file
    vecnorm_file.write_bytes(b"corrupted_bytes_here")

    with pytest.raises(IntegrityError):
        load_env_for_inference(EnvironmentConfig(), str(vecnorm_file))

    # allow_unverified=True bypasses the integrity error (though pickle might fail if bytes are garbage)
    eval_env.close()
