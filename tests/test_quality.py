"""Code quality tests for linting, typing, and coverage configuration."""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys

import pytest


def test_ruff_no_violations():
    """ruff check src/ must return zero violations."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ruff found violations:\n{result.stdout}\n{result.stderr}"
    )


def test_mypy_no_errors():
    """mypy src/ must return zero errors."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "src/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"mypy found type errors:\n{result.stdout}\n{result.stderr}"
    )


def test_no_print_statements_in_library_code():
    """No bare print() calls in src/ except evaluate.py and hpo.py."""
    allowed_files = {"evaluate.py", "hpo.py"}
    violations: list[str] = []

    for root, dirs, files in os.walk("src"):
        dirs[:] = [directory for directory in dirs if directory != "dashboard"]
        for fname in files:
            if not fname.endswith(".py") or fname in allowed_files:
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, encoding="utf-8") as handle:
                try:
                    tree = ast.parse(handle.read(), filename=fpath)
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    violations.append(f"{fpath}:{node.lineno}")

    assert not violations, (
        "Found bare print() calls in library code:\n" + "\n".join(violations)
    )


def test_all_public_modules_importable():
    """Every module in src/ must be importable without error."""
    failed: list[str] = []
    for root, dirs, files in os.walk("src"):
        dirs[:] = [directory for directory in dirs if not directory.startswith("__")]
        for fname in files:
            if fname.endswith(".py") and not fname.startswith("__"):
                fpath = os.path.join(root, fname)
                module_name = fpath.replace(os.sep, ".").replace(".py", "")
                if "dashboard.app" in module_name:
                    continue
                try:
                    importlib.import_module(module_name)
                except Exception as error:
                    failed.append(f"{module_name}: {error}")

    assert not failed, (
        "The following modules could not be imported:\n" + "\n".join(failed)
    )


def test_coverage_above_threshold():
    """Coverage threshold must be configured at 85% in pyproject.toml."""
    with open("pyproject.toml", encoding="utf-8") as handle:
        contents = handle.read()
    assert "fail_under = 85" in contents
