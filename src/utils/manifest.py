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
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class IntegrityError(ValueError):
    """Raised when checkpoint cryptographic verification fails against run_manifest.json."""


class ManifestConfigMismatchError(ValueError):
    """Raised when checkpoint manifest environment settings differ from evaluation."""


class ManifestRequiredError(IntegrityError):
    """Raised when strict verification requires a missing manifest or hash."""


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
        dirty_diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--binary"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return {
            "is_git_repo": True,
            "commit": commit,
            "dirty": len(status) > 0,
            "branch": branch,
            "dirty_diff_sha256": (
                hashlib.sha256(dirty_diff.encode("utf-8")).hexdigest()
                if dirty_diff
                else None
            ),
        }
    except Exception:
        return {
            "is_git_repo": False,
            "commit": None,
            "dirty": None,
            "branch": None,
            "dirty_diff_sha256": None,
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
        "plotly",
        "streamlit",
        "kaleido",
    ]
    versions: dict[str, str | None] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = None
    return versions


def _to_json_safe(obj: Any, path: str = "$config") -> Any:
    """Recursively convert supported config values to JSON-safe primitives."""
    import dataclasses

    if hasattr(obj, "as_dict") and callable(obj.as_dict):
        return _to_json_safe(obj.as_dict(), path)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_json_safe(dataclasses.asdict(obj), path)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        result: dict[str, Any] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Unserialisable config key at {path}: {key!r}; "
                    "keys must be strings."
                )
            result[key] = _to_json_safe(value, f"{path}.{key}")
        return result
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x, f"{path}[{index}]") for index, x in enumerate(obj)]
    if hasattr(obj, "__dict__"):
        return _to_json_safe(vars(obj), path)
    if isinstance(obj, float) and not (obj == obj and abs(obj) != float("inf")):
        raise TypeError(f"Unserialisable non-finite float at {path}: {obj!r}")
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    raise TypeError(
        f"Unserialisable config value at {path}: {obj!r} "
        f"({type(obj).__name__})."
    )


def canonicalize_config(config: Any) -> tuple[dict[str, Any], str]:
    """Convert config to dictionary and compute canonical JSON SHA-256."""
    cfg_dict = _to_json_safe(config)
    if not isinstance(cfg_dict, dict):
        cfg_dict = {"config": cfg_dict}

    canonical_json = json.dumps(
        cfg_dict,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
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
    covered_files: list[str] = []
    if checkpoint_files:
        for f in checkpoint_files:
            p = Path(f)
            if p.is_file():
                covered_files.append(p.name)
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
        "checkpoint_files": covered_files,
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
    checkpoint_path: str | Path | None,
    vecnorm_path: str | Path,
    allow_unverified: bool = False,
    require_manifest: bool = False,
    expected_sha256: Mapping[str, str] | str | None = None,
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
    require_manifest:
        If True, a manifest and hash entry for every supplied file are required.
    expected_sha256:
        Optional out-of-band hash anchor. A string anchors the model; a mapping
        may be keyed by absolute path or basename for either checkpoint file.
    """
    ckpt_p = Path(checkpoint_path) if checkpoint_path is not None else None
    vecnorm_p = Path(vecnorm_path)
    paths = [path for path in (ckpt_p, vecnorm_p) if path is not None]

    def expected_for(path_obj: Path) -> str | None:
        if expected_sha256 is None:
            return None
        if isinstance(expected_sha256, str):
            return expected_sha256 if path_obj == ckpt_p else None
        return expected_sha256.get(str(path_obj), expected_sha256.get(path_obj.name))

    for path_obj in paths:
        expected_hash = expected_for(path_obj)
        if expected_hash is not None:
            actual_hash = compute_file_sha256(path_obj)
            if actual_hash != expected_hash:
                raise IntegrityError(
                    f"Out-of-band SHA-256 verification failed for {path_obj}: "
                    f"expected {expected_hash}, actual {actual_hash}."
                )

    # Search for run_manifest.json in the checkpoint directory or its parent
    candidate_dirs = [path.parent for path in paths]
    manifest_file: Path | None = None
    for d in candidate_dirs:
        candidate = d / "run_manifest.json"
        if candidate.is_file():
            manifest_file = candidate
            break

    if manifest_file is None:
        if require_manifest:
            raise ManifestRequiredError(
                "Required run_manifest.json is missing for "
                f"checkpoint at {ckpt_p or vecnorm_p}."
            )
        logger.warning(
            "No run_manifest.json found for checkpoint at %s; loading legacy unverified checkpoint. "
            "Security note: deserializing untrusted checkpoints entails arbitrary code execution risks.",
            ckpt_p or vecnorm_p,
        )
        return

    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)

    hashes = manifest.get("checkpoint_hashes")
    if not isinstance(hashes, dict):
        message = f"run_manifest.json at {manifest_file} does not contain checkpoint_hashes."
        if require_manifest or not allow_unverified:
            raise ManifestRequiredError(message)
        logger.warning("%s Continuing because allow_unverified=True.", message)
        return

    for path_obj in paths:
        if not path_obj.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {path_obj}")
        fname = path_obj.name
        if fname not in hashes:
            msg = f"No SHA-256 hash recorded for required checkpoint file {path_obj}."
            if require_manifest or not allow_unverified:
                raise ManifestRequiredError(msg)
            logger.warning("%s Continuing because allow_unverified=True.", msg)
            continue
        expected_hash = hashes[fname]
        actual_hash = compute_file_sha256(path_obj)
        if expected_hash != actual_hash:
            msg = (
                f"Checkpoint integrity verification failed for {path_obj}!\n"
                f"Expected SHA-256: {expected_hash}\n"
                f"Actual SHA-256:   {actual_hash}\n"
                "File may be corrupted or tampered with."
            )
            if allow_unverified:
                logger.warning("%s Continuing because allow_unverified=True.", msg)
            else:
                raise IntegrityError(msg)


def validate_manifest_environment(
    vecnorm_path: str | Path,
    env_config: Any,
    allow_mismatch: bool = False,
    checkpoint_path: str | Path | None = None,
) -> None:
    """Compare manifest market settings with the evaluation environment config."""
    vecnorm_p = Path(vecnorm_path)
    candidate_dirs = [vecnorm_p.parent]
    if checkpoint_path is not None:
        candidate_dirs.insert(0, Path(checkpoint_path).parent)
    manifest_file = next(
        (directory / "run_manifest.json" for directory in candidate_dirs if (directory / "run_manifest.json").is_file()),
        None,
    )
    if manifest_file is None:
        return

    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)
    config_data = manifest.get("config", {}).get("data", {})
    stored_environment = config_data.get("eval_environment") or config_data.get("environment")
    if not isinstance(stored_environment, dict):
        stored_environment = config_data
    expected = env_config.as_dict()
    mismatches = []
    for key in ("observation_mode", "market_model"):
        if key in stored_environment and stored_environment[key] != expected[key]:
            mismatches.append(
                f"{key}: manifest={stored_environment[key]!r}, evaluation={expected[key]!r}"
            )
    if not mismatches:
        return

    message = (
        f"Manifest environment mismatch at {manifest_file}: "
        + "; ".join(mismatches)
    )
    if allow_mismatch:
        logger.warning("%s Continuing because allow_mismatch=True.", message)
    else:
        raise ManifestConfigMismatchError(message)
