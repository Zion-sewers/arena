"""Adversarial tests for the fix-commit flywheel (arena#40).

No mocks anywhere in this repo's tests (test_no_mocks_on_boundaries.py):
doubles are hand-written stubs, and the dry-run tests use the real
LocalRunner against the real vendored fixture. Each test pins a gate
the issue demands: PR-only, dry-run-first, gate-catch as the leading
metric, and no human step in the loop.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from bugsmith.harness import LocalRunner
from bugsmith.images import repo_digest
from flywheel import (
    DraftRule,
    DryRunReport,
    FixCommit,
    FlywheelError,
    RuleKind,
    ValidatedRule,
    append_ledger,
    dry_run,
    is_fix_commit,
    leading_gate_catch_rate,
    promote,
    read_ledger,
    reflect,
    render_pr_payload,
    run_loop,
    sample_commits,
)
from flywheel.detect import parse_commits_payload
from flywheel.rules import already_enforced, inject, scan, wiring

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TARGET = REPO_ROOT / "src" / "bugsmith" / "fixtures" / "target"


def _commit(sha: str, message: str) -> FixCommit:
    return FixCommit(sha=sha, message=message, repo="Zion-sewers/arena")


def _draft(kind: RuleKind = "no-bare-except") -> DraftRule:
    return DraftRule(
        kind=kind,
        target_repo="Zion-sewers/arena",
        evidence_shas=("a1b2c3d", "b2c3d4e"),
        support=2,
        rationale="test draft",
    )


def _report(
    *,
    kind: RuleKind = "no-bare-except",
    target_repo: str = "Zion-sewers/arena",
    injected: int = 2,
    caught: int = 2,
    false_positives: int = 0,
) -> DryRunReport:
    return DryRunReport(
        kind=kind,
        target_repo=target_repo,
        injected=injected,
        caught=caught,
        false_positives=false_positives,
        clean_flagged=(),
        cases=(),
        requires_behavioral_break=True,
    )


class RecordedTransport:
    """Hand-written stub transport carrying recorded commits."""

    def __init__(self, commits: list[FixCommit]) -> None:
        self._commits = commits

    def list_fix_commits(self, repo: str, since: str | None = None) -> list[FixCommit]:
        del since
        return [c for c in self._commits if c.repo == repo and is_fix_commit(c.message)]


# --- Detect ---------------------------------------------------------------


def test_is_fix_commit_classifies_prefix_and_reference() -> None:
    assert is_fix_commit("fix: replace bare except in engine")
    assert is_fix_commit("fix(cli): blind except swallowed input")
    assert is_fix_commit("Repair loop fixes #40")
    assert not is_fix_commit("feat: add scoreboard")
    assert not is_fix_commit("prefix fix for the demo")
    assert not is_fix_commit("")
    assert not is_fix_commit("fixing the docs")  # 'fixing' is not a fix-commit


def test_parse_commits_payload_skips_malformed_entries() -> None:
    payload = [
        {"sha": "abc123", "commit": {"message": "fix: bare except"}},
        {"commit": {"message": "fix: no sha here"}},
        {"sha": "def456", "commit": {"message": ""}},
        {"sha": "ghi789", "commit": "not a mapping"},
        {"sha": "jkl012", "commit": {"message": "fix: mock leak"}},
    ]
    parsed = parse_commits_payload(payload, "Zion-sewers/arena")
    assert [c.sha for c in parsed] == ["abc123", "jkl012"]


def test_sample_commits_is_seeded_and_sorted() -> None:
    commits = [_commit(f"sha{i:02d}", f"fix: thing {i}") for i in range(5)]
    first = sample_commits(commits, 3, seed=7)
    second = sample_commits(commits, 3, seed=7)
    assert first == second
    assert len(first) == 3
    # The sample is drawn from sha-sorted order, so shuffling the input
    # listing cannot change it.
    shuffled = list(reversed(commits))
    assert sample_commits(shuffled, 3, seed=7) == first
    assert first == tuple(random.Random(7).sample(sorted(commits, key=lambda c: c.sha), 3))
    with pytest.raises(FlywheelError):
        sample_commits(commits, 0, seed=7)


# --- Predicates -------------------------------------------------------------


def test_scan_bare_except_flags_only_bare_handlers(tmp_path: Path) -> None:
    tree = tmp_path / "t"
    (tree / "src").mkdir(parents=True)
    (tree / "src" / "a.py").write_text(
        "try:\n    f()\nexcept:\n    pass\n", encoding="utf-8"
    )
    (tree / "src" / "b.py").write_text(
        "try:\n    f()\nexcept ValueError:\n    pass\n", encoding="utf-8"
    )
    (tree / "src" / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    findings = scan("no-bare-except", tree)
    assert [(f.file, f.line) for f in findings] == [("src/a.py", 3)]


def test_scan_mock_flags_tests_only(tmp_path: Path) -> None:
    tree = tmp_path / "t"
    (tree / "tests").mkdir(parents=True)
    (tree / "src").mkdir(parents=True)
    (tree / "tests" / "test_x.py").write_text(
        "from unittest.mock import patch\n", encoding="utf-8"
    )
    (tree / "src" / "y.py").write_text(
        "from unittest.mock import patch\n", encoding="utf-8"
    )
    findings = scan("no-mock-on-boundary", tree)
    assert [f.file for f in findings] == ["tests/test_x.py"]


def test_inject_refuses_absent_site(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "src" / "mcalc").mkdir(parents=True)
    (corpus / "src" / "mcalc" / "stats.py").write_text(
        "def other() -> float:\n    return 1.0\n", encoding="utf-8"
    )
    with pytest.raises(FlywheelError):
        inject("no-bare-except", corpus, "bare-except-median")


# --- Reflect ----------------------------------------------------------------


def _target_with(root: Path, *, enforcer: bool, ruff_select: bool) -> Path:
    if enforcer:
        (root / "tests").mkdir(exist_ok=True)
        (root / "tests" / "test_no_mocks_on_boundaries.py").write_text(
            "# enforcer\n", encoding="utf-8"
        )
    pyproject = '[project]\nname = "t"\n'
    if ruff_select:
        pyproject += '\n[tool.ruff.lint]\nselect = ["E", "F", "E722"]\n'
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    return root


def test_reflect_needs_support_floor(tmp_path: Path) -> None:
    sample = (_commit("a1", "fix: replace bare except in engine"),)
    reflection = reflect(sample, _target_with(tmp_path, enforcer=False, ruff_select=False))
    assert reflection.drafts == ()
    assert reflection.skipped == ()


def test_reflect_proposes_at_floor_and_skips_enforced(tmp_path: Path) -> None:
    sample = (
        _commit("a1", "fix: replace bare except in engine"),
        _commit("b2", "fix: blind except swallowed input"),
        _commit("c3", "fix: mock leaked across tests"),
        _commit("d4", "fix: remove mock of the boundary"),
    )
    reflection = reflect(sample, _target_with(tmp_path, enforcer=True, ruff_select=False))
    assert [d.kind for d in reflection.drafts] == ["no-bare-except"]
    assert [s.kind for s in reflection.skipped] == ["no-mock-on-boundary"]
    assert "test_no_mocks_on_boundaries.py" in reflection.skipped[0].reason


def test_reflect_skips_ruff_enforced_bare_except(tmp_path: Path) -> None:
    sample = (
        _commit("a1", "fix: replace bare except in engine"),
        _commit("b2", "fix: blind except swallowed input"),
    )
    reflection = reflect(sample, _target_with(tmp_path, enforcer=False, ruff_select=True))
    assert reflection.drafts == ()
    assert reflection.skipped[0].kind == "no-bare-except"


def test_already_enforced_is_honest_about_missing_config(tmp_path: Path) -> None:
    assert already_enforced("no-bare-except", tmp_path) is None
    assert already_enforced("no-mock-on-boundary", tmp_path) is None


# --- Validate / Promote -------------------------------------------------------


def test_dry_run_promotes_bare_except_on_fixture(tmp_path: Path) -> None:
    """The real gate: injected bare-except faults are caught 2/2, the
    clean trees are unflagged, and each injection breaks a test."""
    target = tmp_path / "target"
    target.mkdir()
    (target / "pyproject.toml").write_text('[project]\nname = "t"\n', encoding="utf-8")
    report = dry_run(
        _draft(),
        corpus_root=FIXTURE_TARGET,
        target_root=target,
        work_dir=tmp_path / "work",
        runner=LocalRunner(),
    )
    assert report.injected == 2
    assert report.caught == 2
    assert report.catch_rate == 1.0
    assert report.false_positives == 0
    assert report.promotable
    assert all(c.suite_broke for c in report.cases)
    assert all(c.fail_to_pass for c in report.cases)
    assert {c.name for c in report.cases} == {"bare-except-median", "bare-except-mean"}


def test_dry_run_refuses_clean_tree_false_positives(tmp_path: Path) -> None:
    """A catch-all rule (flags the clean tree) must not promote, however
    well it catches injected faults."""
    target = tmp_path / "target"
    (target / "src").mkdir(parents=True)
    (target / "src" / "a.py").write_text(
        "try:\n    f()\nexcept:\n    pass\n", encoding="utf-8"
    )
    (target / "pyproject.toml").write_text('[project]\nname = "t"\n', encoding="utf-8")
    report = dry_run(
        _draft(),
        corpus_root=FIXTURE_TARGET,
        target_root=target,
        work_dir=tmp_path / "work",
        runner=LocalRunner(),
    )
    assert report.false_positives == 1
    assert not report.promotable
    with pytest.raises(FlywheelError):
        promote(_draft(), report)


def test_promote_refuses_vacuous_partial_and_mismatched_reports() -> None:
    draft = _draft()
    with pytest.raises(FlywheelError):
        promote(draft, _report(injected=0, caught=0))
    with pytest.raises(FlywheelError):
        promote(draft, _report(caught=1))
    with pytest.raises(FlywheelError):
        promote(draft, _report(false_positives=2))
    with pytest.raises(FlywheelError):
        promote(draft, _report(kind="no-mock-on-boundary"))
    with pytest.raises(FlywheelError):
        ValidatedRule(draft=draft, report=_report(caught=0))


def test_render_pr_payload_refuses_unvalidated_draft(tmp_path: Path) -> None:
    """A bare DraftRule cannot render a payload: the signature demands a
    ValidatedRule, and a draft has no report to render from."""
    with pytest.raises(AttributeError):
        render_pr_payload(_draft(), tmp_path)  # type: ignore[arg-type]


def test_wiring_refuses_existing_ruff_lint_section(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "t"\n\n[tool.ruff.lint]\nselect = ["E"]\n', encoding="utf-8"
    )
    with pytest.raises(FlywheelError):
        wiring("no-bare-except", tmp_path)


# --- Ledger -------------------------------------------------------------------


def test_ledger_round_trip_and_leading_metric(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    assert leading_gate_catch_rate(ledger) is None
    append_ledger(ledger, _report())
    append_ledger(ledger, _report(caught=1))
    entries = read_ledger(ledger)
    assert len(entries) == 2
    assert entries[0]["schema_version"] == 1
    assert leading_gate_catch_rate(ledger) == 0.75
    ledger.write_text("not json\n", encoding="utf-8")
    with pytest.raises(FlywheelError):
        read_ledger(ledger)


# --- Loop (PR-only, no human step) --------------------------------------------


def _recorded_loop_commits() -> list[FixCommit]:
    return [
        _commit("a1b2c3d4", "fix: replace bare except in engine"),
        _commit("b2c3d4e5", "fix: blind except swallowed input"),
        _commit("c3d4e5f6", "fix: mock leaked across tests"),
        _commit("d4e5f6a7", "fix: remove mock of the boundary"),
        _commit("e5f6a7b8", "feat: add scoreboard"),
    ]


def test_loop_is_pr_only_and_needs_no_human(tmp_path: Path) -> None:
    target = tmp_path / "target"
    (target / "src").mkdir(parents=True)
    (target / "src" / "a.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (target / "pyproject.toml").write_text('[project]\nname = "t"\n', encoding="utf-8")
    digest_before = repo_digest(target)
    result = run_loop(
        transport=RecordedTransport(_recorded_loop_commits()),
        target_repo="Zion-sewers/arena",
        corpus_root=FIXTURE_TARGET,
        target_root=target,
        work_dir=tmp_path / "work",
        runner=LocalRunner(),
        ledger_path=tmp_path / "ledger.jsonl",
    )
    assert repo_digest(target) == digest_before
    assert result.detected == 4
    # The temp target has no mock enforcer, so BOTH kinds propose and
    # validate through the loop (the already-enforced skip is proven by
    # the demo against the real repo target).
    assert len(result.payloads) == 2
    assert {p.rule_kind for p in result.payloads} == {
        "no-bare-except",
        "no-mock-on-boundary",
    }
    bare = next(p for p in result.payloads if p.rule_kind == "no-bare-except")
    assert "E722" in bare.files["pyproject.toml"]
    assert result.skipped == ()
    assert len(read_ledger(tmp_path / "ledger.jsonl")) == 2


def test_demo_runs_end_to_end_with_stdin_closed() -> None:
    """The demo is the no-human-step proof: it runs the whole loop as a
    subprocess with stdin closed — any prompt for input would fail."""
    proc = subprocess.run(
        [sys.executable, "-m", "flywheel.demo"],
        cwd=str(REPO_ROOT),
        env={
            "PYTHONPATH": str(REPO_ROOT / "src"),
            # The demo's LocalRunner shells out to `python3`: keep it on
            # the venv python (which has pytest) plus the system paths.
            "PATH": os.pathsep.join(
                [str(Path(sys.executable).parent), "/usr/bin", "/bin"]
            ),
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "gate-catch rate (leading metric): 100%" in proc.stdout
