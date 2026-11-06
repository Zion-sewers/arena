"""Validation harness for bug candidates (agenteval-bench#37).

``harness.valid`` in the issue's pipeline: run the target repo's test
suite on the clean tree (baseline), apply one candidate patch to a
throwaway copy, re-run the suite, and keep only candidates that break at
least one test. Reports carry FAIL_TO_PASS / PASS_TO_PASS lists computed
from test node IDs, never from prose.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bugsmith.images import TargetImage, repo_digest
from bugsmith.patch import PatchError, apply_patch
from bugsmith.types import BugCandidate, BugsmithError, ValidationReport

PYTEST_ARGS = ["-q", "--tb=no", "-p", "no:cacheprovider"]
# The harness owns its pytest invocation: a target's own addopts (this
# repo ships ``-v``) would otherwise perturb it — ``-v`` plus the
# harness's ``-q`` cancel to default verbosity, and ``--collect-only``
# then prints the tree format instead of one node ID per line, so
# _parse_collected finds zero IDs and the baseline refuses with "zero
# tests". -o addopts= neutralizes the target's addopts on every call.
# Contract change (arena#39): this is harness-wide, not nightly-only —
# every LocalRunner/DockerRunner invocation for every target runs with
# the target's addopts neutralized and --rootdir pinned to the work
# tree, so a target that relies on its own addopts (plugins, markers,
# traceback style) silently loses them. That is the price of a stable
# node-ID contract across copies.
PYTEST_OVERRIDES = ["-o", "addopts="]
RUN_TIMEOUT_S = 300

_FAILED_RE = re.compile(r"^(FAILED|ERROR) (.*)$")


def resolve_node_id(tail: str, known_ids: frozenset[str]) -> str:
    """Resolve a short-summary tail (``<nodeid> - <reason>``) to a node ID.

    Both the node ID and the reason may contain ``" - "`` (spaced
    parametrized IDs like ``test_p[a - b]``; reasons echoing them), so a
    blind split is ambiguous. Resolution order:
    1. exact match against the collected (known) IDs;
    2. longest known ID that prefixes the tail followed by ``" - "``;
    3. fallback: split at the last ``" - "`` (best effort, no known set).
    """
    if tail in known_ids:
        return tail
    cands = [k for k in known_ids if tail.startswith(k + " - ")]
    if cands:
        return max(cands, key=len)
    return tail.rsplit(" - ", 1)[0] if " - " in tail else tail


@dataclass(frozen=True)
class TestOutcome:
    failed: frozenset[str]
    collection_error: bool
    returncode: int


def parse_pytest_output(
    output: str, returncode: int, known_ids: frozenset[str] = frozenset()
) -> TestOutcome:
    """Extract test node IDs from ``pytest -q --tb=no`` output.

    A collection error poisons the whole run: every test counts as failed
    (the bug broke the suite structurally, not just behaviorally).

    ``known_ids`` (the --collect-only set) disambiguates summary tails when
    node IDs or reasons contain ``" - "``; without it, resolution is
    best-effort.
    """
    failed: set[str] = set()
    in_summary = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("====") and "short test summary info" in stripped:
            in_summary = True
            continue
        if in_summary:
            m = _FAILED_RE.match(stripped)
            if m:
                failed.add(resolve_node_id(m.group(2), known_ids))
                continue
            if stripped.startswith("===="):
                in_summary = False
    collection_error = (
        returncode == 2
        and ("error during collection" in output or "errors during collection" in output)
    )
    return TestOutcome(failed=frozenset(failed), collection_error=collection_error,
                       returncode=returncode)


# pytest exit codes with a defined meaning for the harness. Anything else
# (segfault, internal error, timeout kill) is infrastructure failure, never
# a test result.
KNOWN_EXIT_CODES = frozenset({0, 1, 2, 5})


class Runner(Protocol):
    """Executes the target repo's test suite in a prepared work tree."""

    def run(
        self, work_tree: Path, known_ids: frozenset[str] = frozenset()
    ) -> TestOutcome:
        """Run the suite; ``known_ids`` disambiguates summary-line parsing."""
        ...

    def collect(self, work_tree: Path) -> frozenset[str]:
        """All test node IDs, via --collect-only (independent of pass/fail)."""
        ...


def _parse_collected(output: str) -> frozenset[str]:
    ids = set()
    for line in output.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith(("=", "<", "ERROR")):
            # Node IDs may contain spaces (parametrized IDs like
            # test_p[a - b]); the whole line is the ID, never split it.
            ids.add(line)
    return frozenset(ids)


def _local_env(work_tree: Path) -> dict[str, str]:
    env = dict(os.environ)
    new_path = _pythonpath_for(work_tree)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{new_path}{os.pathsep}{existing}" if existing else new_path
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _pythonpath_for(tree: Path) -> str:
    src = tree / "src"
    return str(src if src.is_dir() else tree)


def _docker_base(image: TargetImage, work_tree: Path) -> list[str]:
    """Shared docker invocation: host-owned files, no __pycache__ litter."""
    return [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", f"{work_tree.resolve()}:/work",
        image.tag,
    ]


def _container_pytest_script(pytest_args: list[str]) -> str:
    """Pytest command for inside the container, with the target importable.

    Single construction site for the PYTHONPATH logic: every docker-side
    pytest invocation (run + collect) goes through here so the two cannot
    drift apart again.
    """
    inner = (
        "python3 -c \"import pathlib; "
        "p=pathlib.Path('/work/src'); "
        "print(p if p.is_dir() else pathlib.Path('/work'))\""
    )
    return (
        f"cd /work && PYTHONPATH=$({inner}) "
        f"python3 -m pytest {' '.join(pytest_args)}"
    )


class LocalRunner:
    """Subprocess runner. Used by unit tests and as an explicit opt-in.

    Not a silent fallback: callers choose it deliberately (e.g. --local).

    ``--rootdir`` is pinned to the work tree on every invocation: without
    it pytest resolves rootdir upward from the work tree, so any config
    file in an ancestor (e.g. a stray pyproject.toml under /tmp, where
    pytest's tmp_path lives) makes node IDs embed the work-tree path —
    and the clean-tree and patched-tree IDs for the SAME test then differ,
    so fail_to_pass matching silently reports every mutant as surviving.
    ``-o addopts=`` is pinned for the same reason: a target's own addopts
    (this repo ships ``-v``) cancel the harness's ``-q`` to default
    verbosity, and ``--collect-only`` then prints the tree format instead
    of one node ID per line, so collection finds zero IDs.
    """

    def run(
        self, work_tree: Path, known_ids: frozenset[str] = frozenset()
    ) -> TestOutcome:
        proc = subprocess.run(
            ["python3", "-m", "pytest", *PYTEST_ARGS, *PYTEST_OVERRIDES,
             f"--rootdir={work_tree}"],
            cwd=str(work_tree),
            env=_local_env(work_tree),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=RUN_TIMEOUT_S,
            check=False,
            text=True,
        )
        return parse_pytest_output(proc.stdout, proc.returncode, known_ids)

    def collect(self, work_tree: Path) -> frozenset[str]:
        proc = subprocess.run(
            ["python3", "-m", "pytest", "--collect-only", "-q",
             "-p", "no:cacheprovider", *PYTEST_OVERRIDES,
             f"--rootdir={work_tree}"],
            cwd=str(work_tree),
            env=_local_env(work_tree),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=RUN_TIMEOUT_S,
            check=False,
            text=True,
        )
        return _parse_collected(proc.stdout)


class DockerRunner:
    """Runs the suite inside the built target image (the required path).

    The candidate's patched tree is bind-mounted read-write at /work; the
    image's own /target copy stays pristine.
    """

    def __init__(self, image: TargetImage) -> None:
        self._image = image

    def run(
        self, work_tree: Path, known_ids: frozenset[str] = frozenset()
    ) -> TestOutcome:
        proc = subprocess.run(
            _docker_base(self._image, work_tree)
            + ["sh", "-c", _container_pytest_script(
                [*PYTEST_ARGS, *PYTEST_OVERRIDES, "--rootdir=/work"])],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=RUN_TIMEOUT_S,
            check=False,
            text=True,
        )
        return parse_pytest_output(proc.stdout, proc.returncode, known_ids)

    def collect(self, work_tree: Path) -> frozenset[str]:
        proc = subprocess.run(
            _docker_base(self._image, work_tree)
            + ["sh", "-c", _container_pytest_script(
                ["--collect-only", "-q", "-p", "no:cacheprovider",
                 *PYTEST_OVERRIDES, "--rootdir=/work"])],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=RUN_TIMEOUT_S,
            check=False,
            text=True,
        )
        return _parse_collected(proc.stdout)


def _copy_tree(repo_root: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        repo_root, dest,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )


def validate(
    candidate: BugCandidate,
    repo_root: Path,
    runner: Runner,
    work_dir: Path,
    baseline_passed: frozenset[str],
) -> ValidationReport:
    """Validate one candidate: baseline vs patched test outcomes.

    ``baseline_passed`` is the clean-tree passing set (computed once per
    repo digest by :func:`baseline`). The candidate's patched tree is a
    throwaway copy; the caller's tree is never mutated.
    """
    repo_root = repo_root.resolve()
    digest = repo_digest(repo_root)
    run_dir = work_dir / "runs" / digest / candidate.patch_sha
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    patched = run_dir / "patched"
    _copy_tree(repo_root, patched)
    try:
        apply_patch(patched, candidate.patch)
    except PatchError as e:
        raise BugsmithError(f"candidate patch does not apply: {e}") from e

    # Collect BEFORE running: the --collect-only node-ID set disambiguates
    # FAILED/ERROR summary tails (node IDs and reasons may both contain
    # " - "), so parsed failures resolve to real, addressable test IDs.
    collected = runner.collect(patched)
    outcome = runner.run(patched, known_ids=collected)
    if outcome.returncode not in KNOWN_EXIT_CODES:
        raise BugsmithError(
            f"pytest exited with code {outcome.returncode} on the patched tree: "
            "infrastructure failure, refusing to score it as a test result"
        )
    if outcome.collection_error:
        failed_after = frozenset(baseline_passed)  # structural break: everything fails
        passed_after = frozenset()
    else:
        # passed_after = collected - failed. The patched-tree collect above
        # (not a re-collect) is the same set, so renamed/deleted tests cannot
        # silently vanish from the math.
        failed_after = outcome.failed
        passed_after = collected - failed_after

    return ValidationReport(
        candidate=candidate,
        baseline_passed=tuple(sorted(baseline_passed)),
        failed_after=tuple(sorted(failed_after)),
        passed_after=tuple(sorted(passed_after)),
        collection_error=outcome.collection_error,
    )


def baseline(repo_root: Path, runner: Runner, work_dir: Path) -> frozenset[str]:
    """Test node IDs passing on the clean tree (cached per repo digest)."""
    repo_root = repo_root.resolve()
    digest = repo_digest(repo_root)
    cache = work_dir / "runs" / digest / "baseline_passed.txt"
    if cache.is_file():
        return frozenset(l.strip() for l in cache.read_text().splitlines() if l.strip())
    clean = work_dir / "runs" / digest / "clean"
    _copy_tree(repo_root, clean)
    outcome = runner.run(clean)
    if outcome.returncode not in KNOWN_EXIT_CODES:
        raise BugsmithError(
            f"pytest exited with code {outcome.returncode} on the clean tree: "
            "infrastructure failure, refusing to certify a baseline"
        )
    if outcome.collection_error or outcome.failed:
        raise BugsmithError(
            "clean-tree baseline is not green: the target repo's own suite "
            f"must pass before bug injection (failed={sorted(outcome.failed)}, "
            f"collection_error={outcome.collection_error})"
        )
    passed = runner.collect(clean)
    if not passed:
        raise BugsmithError("baseline collected zero tests: refusing an empty benchmark")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(sorted(passed)) + "\n", encoding="utf-8")
    return passed
