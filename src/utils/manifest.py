"""Run manifest tracking and checkpoint integrity verification.

Security Notice (Pickle / Deserialization Risk):
------------------------------------------------
Both PyTorch models (via torch.load / zipfile) and VecNormalize statistics (via pickle)
execute arbitrary Python code during deserialization. Never load checkpoints or manifests
from untrusted, unverified, or public sources. Always verify cryptographic hashes using
run_manifest.json before unpickling model weights or normalization state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class IntegrityError(ValueError):
    """Raised when checkpoint cryptographic verification fails against run_manifest.json."""


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file in 64 KiB chunks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot compute hash: {p} does not exist or is not a file")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> dict[str, Any]:
    """Retrieve git commit SHA, dirty status, and branch, handling non-git environments gracefully."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {
            "is_git_repo": True,
            "commit": commit,
            "dirty": len(status) > 0,
            "branch": branch,
        }
    except Exception:
        return {
            "is_git_repo": False,
            "commit": None,
            "dirty": None,
            "branch": None,
        }


def get_system_info() -> dict[str, Any]:
    """Capture runtime platform, Python environment, and execution arguments."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "command_line": sys.argv,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def get_dependency_versions() -> dict[str, str | None]:
    """Record versions of key numerical, RL, and optimization packages."""
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "gymnasium",
        "stable_baselines3",
        "torch",
        "optuna",
    ]
    versions: dict[str, str | None] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = None
    return versions


def canonicalize_config(config: Any) -> tuple[dict[str, Any], str]:
    """Convert config to dictionary and compute canonical JSON SHA-256."""
    if hasattr(config, "as_dict"):
        cfg_dict = config.as_dict()
    elif hasattr(config, "__dict__"):
        cfg_dict = {
            k: getattr(config, k).as_dict() if hasattr(getattr(config, k), "as_dict") else getattr(config, k)
            for k in dir(config)
            if not k.startswith("_") and not callable(getattr(config, k))
        }
    elif isinstance(config, dict):
        cfg_dict = config
    else:
        cfg_dict = {"raw": str(config)}

    canonical_json = json.dumps(cfg_dict, sort_keys=True, default=str)
    cfg_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return cfg_dict, cfg_hash


def create_manifest(
    run_type: str,
    config: Any,
    seed: int | None = None,
    checkpoint_files: Sequence[str | Path] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a comprehensive run manifest dictionary."""
    cfg_dict, cfg_hash = canonicalize_config(config)

    file_hashes: dict[str, str] = {}
    if checkpoint_files:
        for f in checkpoint_files:
            p = Path(f)
            if p.is_file():
                file_hashes[p.name] = compute_file_sha256(p)

    manifest: dict[str, Any] = {
        "run_type": run_type,
        "seed": seed,
        "git": get_git_info(),
        "system": get_system_info(),
        "dependencies": get_dependency_versions(),
        "config": {
            "sha256": cfg_hash,
            "data": cfg_dict,
        },
        "checkpoint_hashes": file_hashes,
        "extra": extra_metadata or {},
    }
    return manifest


def write_manifest(
    output_dir: str | Path,
    run_type: str,
    config: Any,
    seed: int | None = None,
    checkpoint_files: Sequence[str | Path] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write run_manifest.json into output_dir and return file path."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest = create_manifest(
        run_type=run_type,
        config=config,
        seed=seed,
        checkpoint_files=checkpoint_files,
        extra_metadata=extra_metadata,
    )
    manifest_file = out_path / "run_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Run manifest written to %s", manifest_file)
    return manifest_file


def verify_checkpoint_integrity(
    checkpoint_path: str | Path,
    vecnorm_path: str | Path,
    allow_unverified: bool = False,
) -> None:
    """Verify cryptographic integrity of checkpoint and vecnorm files against run_manifest.json.

    Parameters
    ----------
    checkpoint_path:
        Path to the SAC model (.zip).
    vecnorm_path:
        Path to the matching VecNormalize stats (.pkl).
    allow_unverified:
        If True, log a warning on hash mismatch instead of raising IntegrityError.
    """
    ckpt_p = Path(checkpoint_path)
    vecnorm_p = Path(vecnorm_path)

    # Search for run_manifest.json in the checkpoint directory or its parent
    candidate_dirs = [ckpt_p.parent, vecnorm_p.parent]
    manifest_file: Path | None = None
    for d in candidate_dirs:
        candidate = d / "run_manifest.json"
        if candidate.is_file():
            manifest_file = candidate
            break

    if manifest_file is None:
        logger.warning(
            "No run_manifest.json found for checkpoint at %s; loading legacy unverified checkpoint. "
            "Security note: deserializing untrusted checkpoints entails arbitrary code execution risks.",
            ckpt_p,
        )
        return

    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)

    hashes = manifest.get("checkpoint_hashes", {})
    if not hashes:
        logger.warning("run_manifest.json at %s does not contain checkpoint_hashes; skipping verification.", manifest_file)
        return

    for path_obj in [ckpt_p, vecnorm_p]:
        if not path_obj.is_file():
            continue
        fname = path_obj.name
        if fname in hashes:
            expected_hash = hashes[fname]
            actual_hash = compute_file_sha256(path_obj)
            if expected_hash != actual_hash:
                msg = (
                    f"Checkpoint integrity verification failed for {path_obj}!\n"
                    f"Expected SHA-256: {expected_hash}\n"
                    f"Actual SHA-256:   {actual_hash}\n"
                    f"File may be corrupted or tampered with."
                )
                if allow_unverified:
                    logger.warning("%s Continuing because allow_unverified=True.", msg)
                else:
                    raise IntegrityError(msg)
