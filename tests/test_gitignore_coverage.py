"""Adversarial coverage test for .gitignore: every artifact class enumerated in
issue #18 must actually be ignored, and tracked source files must NOT be
silenced by an over-broad pattern. Enforcement is `git check-ignore`,
not prose.

uv.lock is pinned here although not enumerated in issue #18: it is
pre-existing in .gitignore (generated per-machine) and this test pins the
state, it does not expand the issue scope.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Representative paths for every artifact class enumerated in issue #18.
MUST_BE_IGNORED = [
    # Python bytecode / compilation
    "__pycache__/engine.cpython-312.pyc",
    "src/agenteval_bench/engine.cpython-312.pyc",
    "legacy.pyo",
    "native.so",
    "Parser$py.class",
    # Packaging / build
    "build/lib/agenteval_bench/engine.py",
    "dist/agenteval-bench-0.1.0.tar.gz",
    "pkg.egg",
    "pkg.egg-info/PKG-INFO",
    "pip-wheel-metadata/top_level.txt",
    ".eggs/setuptools-68",
    # Virtual environments
    ".venv/bin/activate",
    "venv/bin/activate",
    "env/bin/activate",
    "ENV/bin/activate",
    # Generated dependency lock
    "uv.lock",
    # Test / coverage
    ".pytest_cache/CACHEDIR.TAG",
    ".coverage",
    ".coverage.localhost.1234.5678",
    "coverage.xml",
    "htmlcov/index.html",
    ".tox/py312/bin/pytest",
    ".nox/lint/bin/ruff",
    ".cache/pip/x",
    ".hypothesis/examples/x",
    # Lint / type-check caches
    ".ruff_cache/0.5.0/CACHEDIR.TAG",
    ".mypy_cache/3.12/engine.data.json",
    ".dmypy.json",
    ".pyre/x",
    ".pytype/py3.12/imports/x",
    # Notebooks
    ".ipynb_checkpoints/eda-checkpoint.ipynb",
    # Docs build output
    "docs/_build/html/index.html",
    "site/index.html",
    # Generated eval artifacts
    "reports/run1/junit.xml",
    "runs/2026-10-09/eval.json",
    "exp.report.json",
    # Secrets / local config / logs
    ".env",
    ".env.local",
    "model.pkl",
    "debug.log",
    # jev routing decision telemetry: append-only local audit log, never committed
    "jev/decisions.jsonl",
    # IDE / editor
    ".idea/workspace.xml",
    ".vscode/settings.json",
    "engine.py.swp",
    "engine.py.swo",
    "engine.py~",
    ".project",
    ".classpath",
    # OS artifacts
    ".DS_Store",
    ".AppleDouble",
    ".LSOverride",
    "Thumbs.db",
    "ehthumbs.db",
    "Desktop.ini",
    "$RECYCLE.BIN/x",
]

# Real source files that must never be swallowed by an over-broad pattern.
MUST_NOT_BE_IGNORED = [
    "src/agenteval_bench/engine.py",
    "src/agenteval_bench/__init__.py",
    "src/novelty_oracle/oracle.py",
    "jev/router.py",
    "jev/checks.py",
    "pyproject.toml",
    "README.md",
    "AGENTS.md",
    "tests/test_engine.py",
    ".github/workflows/ci.yml",
    "docs/spec.md",
    ".gitignore",
]


def _check_ignore(paths: list[str]) -> set[str]:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "--", *paths],
        capture_output=True,
        text=True,
        check=False,
    )
    return {line for line in proc.stdout.splitlines() if line}


def _in_git_work_tree() -> bool:
    """True only when REPO_ROOT is inside a git work tree.

    These checks enforce ``git check-ignore`` / ``git ls-files`` against the
    live repo. Throwaway copies (bugsmith's mutation baseline, the tiered
    pipeline's nightly phase) deliberately carry no ``.git`` directory, so
    the git commands fail and the assertions would fail vacuously — or pass
    vacuously for the tree-observation test. Skipping there is honest:
    there is no repo to check. In any real checkout the tests run in full.
    """
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def test_all_issue18_artifact_classes_are_ignored() -> None:
    if not _in_git_work_tree():
        pytest.skip("not inside a git work tree: .gitignore checks need git")
    ignored = _check_ignore(MUST_BE_IGNORED)
    missing = [p for p in MUST_BE_IGNORED if p not in ignored]
    assert not missing, f"paths NOT ignored by .gitignore: {missing}"


def test_no_overbroad_pattern_swallows_sources() -> None:
    if not _in_git_work_tree():
        pytest.skip("not inside a git work tree: .gitignore checks need git")
    ignored = _check_ignore(MUST_NOT_BE_IGNORED)
    assert not ignored, f"source files wrongly ignored: {sorted(ignored)}"


# Basename/dirname glob patterns for artifact classes: anything a working tree
# should never carry as an *unignored* entry. This is the tree-observation half
# of issue #18's acceptance (`git status --ignored` shows expected categories,
# no stray unignored artifacts) — unlike the pattern assertions above, it looks
# at the actual tree, so a stray artifact dropped outside the ignored paths
# fails the suite instead of passing silently.
ARTIFACT_NAME_PATTERNS = [
    "__pycache__",
    "*.py[cod]",
    "*$py.class",
    "*.so",
    "build",
    "dist",
    "*.egg-info",
    "*.egg",
    ".eggs",
    "pip-wheel-metadata",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    ".coverage*",
    "coverage.xml",
    "htmlcov",
    ".tox",
    ".nox",
    ".cache",
    ".hypothesis",
    ".ruff_cache",
    ".mypy_cache",
    ".dmypy.json",
    ".pyre",
    ".pytype",
    ".ipynb_checkpoints",
    "site",
    "reports",
    "runs",
    "*.report.json",
    ".env*",
    "*.pkl",
    "*.log",
    ".idea",
    ".vscode",
    "*.swp",
    "*.swo",
    "*~",
    ".project",
    ".classpath",
    ".DS_Store",
    ".AppleDouble",
    ".LSOverride",
    "Thumbs.db",
    "ehthumbs.db",
    "Desktop.ini",
]

# Known limitation (documented, not a defect): the tree-observation test only
# catches artifact names on ARTIFACT_NAME_PATTERNS. A novel artifact name
# (e.g. benchmark_results.json, output/) passes silently — extending the list
# is the cost of keeping the check executable rather than hand-wavy.


def _untracked_unignored() -> list[str]:
    """Untracked, unignored paths in the working tree (the stray-artifact set)."""
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in proc.stdout.splitlines() if line]


def test_no_stray_unignored_artifacts_in_tree() -> None:
    if not _in_git_work_tree():
        pytest.skip("not inside a git work tree: tree observation needs git")
    import fnmatch

    def _looks_like_artifact(path: str) -> bool:
        parts = Path(path).parts
        return any(
            fnmatch.fnmatchcase(part, pattern)
            for part in parts
            for pattern in ARTIFACT_NAME_PATTERNS
        )

    strays = [p for p in _untracked_unignored() if _looks_like_artifact(p)]
    assert not strays, (
        "stray unignored artifacts in working tree "
        "(issue #18: these should be covered by .gitignore): "
        f"{sorted(strays)}"
    )
