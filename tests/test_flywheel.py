"""Adversarial tests for the fix-commit flywheel (arena#40).

No mocks anywhere in this repo's tests (test_no_mocks_on_boundaries.py):
doubles are hand-written stubs, and the dry-run tests use the real
LocalRunner against the real vendored fixture. Each test pins a gate
the issue demands: PR-only, dry-run-first, gate-catch as the leading
metric, and no human step in the loop.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from bugsmith.harness import LocalRunner
from bugsmith.images import repo_digest
from flywheel import (
    DraftRule,
    DryRunReport,
    Finding,
    FixCommit,
    FlywheelError,
    InjectionCase,
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
from flywheel.rules import (
    _MOCK_ENFORCER_TEST,
    _mock_violations,
    already_enforced,
    inject,
    scan,
    wiring,
)

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


def _case(
    name: str,
    *,
    flagged: bool = True,
    suite_broke: bool = True,
    enforcer_failed: bool = False,
) -> InjectionCase:
    return InjectionCase(
        name=name,
        file="src/mcalc/stats.py",
        flagged=flagged,
        fail_to_pass=("test_x",) if suite_broke else (),
        suite_broke=suite_broke,
        enforcer_failed=enforcer_failed,
    )


def _report(
    *,
    kind: RuleKind = "no-bare-except",
    target_repo: str = "Zion-sewers/arena",
    injected: int = 2,
    caught: int = 2,
    false_positives: int = 0,
    requires_behavioral_break: bool = True,
    requires_enforcer_check: bool = False,
) -> DryRunReport:
    """A CONSISTENT report: cases, counts, and clean findings agree by
    construction. Forged variants are built directly in the tests that
    pin the refusal."""
    cases = tuple(
        _case(
            f"c{i + 1}",
            flagged=(i < caught),
            suite_broke=requires_behavioral_break and (i < caught),
            enforcer_failed=requires_enforcer_check and (i < caught),
        )
        for i in range(injected)
    )
    clean_flagged = tuple(
        Finding(file="src/a.py", line=1, message="bare except")
        for _ in range(false_positives)
    )
    return DryRunReport(
        kind=kind,
        target_repo=target_repo,
        injected=injected,
        caught=caught,
        false_positives=false_positives,
        clean_flagged=clean_flagged,
        cases=cases,
        requires_behavioral_break=requires_behavioral_break,
        requires_enforcer_check=requires_enforcer_check,
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
        {"commit": {"message": "no sha here"}},
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


def _live_find_mock_usage() -> object:
    """The live enforcer's detector, loaded from the repo's own test
    file: the flywheel scan is pinned against it, not against a copy."""
    path = REPO_ROOT / "tests" / "test_no_mocks_on_boundaries.py"
    spec = importlib.util.spec_from_file_location("live_mock_enforcer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.find_mock_usage


def test_scan_mock_matches_live_enforcer_on_every_shape(tmp_path: Path) -> None:
    """Scan ≡ live enforcer, both directions, on the shapes the first
    review found diverging: `from unittest import mock`, attribute use,
    the `mocker` fixture, and a plain import (control)."""
    live = _live_find_mock_usage()
    probes = [
        "from unittest.mock import patch\n",
        "from unittest import mock\n",
        "import unittest.mock.patch\n",
        "from unittest.mock.patch import patch\n",
        "x = unittest.mock.MagicMock()\n",
        "x = mock.patch('os.path.join')\n",
        "def test_something(mocker):\n    pass\n",
        "def replay_fn(input_text: str) -> str:\n    return ''\n",
        "patch = open('fix.patch').read()\n",
    ]
    for source in probes:
        tree = tmp_path / "t"
        (tree / "tests").mkdir(parents=True, exist_ok=True)
        (tree / "tests" / "test_probe.py").write_text(source, encoding="utf-8")
        flywheel_hits = scan("no-mock-on-boundary", tree)
        live_hits = live(source)
        assert bool(flywheel_hits) == bool(live_hits), source
        # The detector functions agree directly, not just via the scan.
        assert bool(_mock_violations(source)) == bool(live_hits), source


def test_scan_mock_ignores_nested_test_dirs_like_the_live_enforcer(
    tmp_path: Path,
) -> None:
    """The live enforcer globs top-level tests/*.py only; the scan must
    cover the same file set, not more."""
    tree = tmp_path / "t"
    (tree / "tests" / "unit").mkdir(parents=True)
    (tree / "tests" / "unit" / "test_deep.py").write_text(
        "from unittest.mock import patch\n", encoding="utf-8"
    )
    assert scan("no-mock-on-boundary", tree) == []


def test_shipped_enforcer_test_is_the_live_detector(tmp_path: Path) -> None:
    """The file the wiring installs carries the live detector (same
    function, same controls): the thing validated is the thing the PR
    installs."""
    assert "def find_mock_usage" in _MOCK_ENFORCER_TEST
    assert "def test_no_mocks_in_test_suite" in _MOCK_ENFORCER_TEST
    assert "def test_detector_flags_mock_imports" in _MOCK_ENFORCER_TEST
    # It runs green on a clean tree and red on a mock import.
    tree = tmp_path / "t"
    (tree / "tests").mkdir(parents=True)
    (tree / "tests" / "test_no_mocks_on_boundaries.py").write_text(
        _MOCK_ENFORCER_TEST, encoding="utf-8"
    )
    (tree / "tests" / "test_clean.py").write_text(
        "def test_ok() -> None:\n    assert True\n", encoding="utf-8"
    )
    clean = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_no_mocks_on_boundaries.py",
         "-q", "-p", "no:cacheprovider", f"--rootdir={tree}"],
        cwd=str(tree), capture_output=True, text=True, timeout=120, check=False,
    )
    assert clean.returncode == 0, clean.stdout[-1000:]
    (tree / "tests" / "test_clean.py").write_text(
        "from unittest.mock import patch\n", encoding="utf-8"
    )
    dirty = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_no_mocks_on_boundaries.py",
         "-q", "-p", "no:cacheprovider", f"--rootdir={tree}"],
        cwd=str(tree), capture_output=True, text=True, timeout=120, check=False,
    )
    assert dirty.returncode == 1


def test_inject_refuses_absent_site(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "src" / "mcalc").mkdir(parents=True)
    (corpus / "src" / "mcalc" / "stats.py").write_text(
        "def other() -> float:\n    return 1.0\n", encoding="utf-8"
    )
    with pytest.raises(FlywheelError):
        inject("no-bare-except", corpus, "bare-except-median")
    with pytest.raises(FlywheelError):
        inject("no-bare-except", corpus, "no-such-case")


# --- Reflect ----------------------------------------------------------------


def _target_with(root: Path, *, enforcer: bool, ruff_select: bool) -> Path:
    if enforcer:
        (root / "tests").mkdir(exist_ok=True)
        (root / "tests" / "test_no_mocks_on_boundaries.py").write_text(
            "def find_mock_usage(source):\n    return []\n", encoding="utf-8"
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


def _ruff_target(root: Path, content: str) -> Path:
    (root / "pyproject.toml").write_text(content, encoding="utf-8")
    return root


def test_ruff_probe_reads_prefix_extend_select_and_ruff_toml(
    tmp_path: Path,
) -> None:
    """A select prefix that covers E722 enforces the kind, wherever it
    lives: `select = ["E"]`, `select = ["ALL"]`, `extend-select`, and
    `ruff.toml` all count; a bare-except rule must not be re-proposed
    onto a repo that already enforces it."""
    sample = (
        _commit("a1", "fix: replace bare except in engine"),
        _commit("b2", "fix: blind except swallowed input"),
    )
    for index, content in enumerate(
        (
            '[project]\nname = "t"\n\n[tool.ruff.lint]\nselect = ["E"]\n',
            '[project]\nname = "t"\n\n[tool.ruff.lint]\nselect = ["ALL"]\n',
            '[project]\nname = "t"\n\n[tool.ruff.lint]\nextend-select = ["E722"]\n',
        )
    ):
        root = tmp_path / f"t{index}"
        root.mkdir()
        assert already_enforced("no-bare-except", _ruff_target(root, content)) is not None
        assert reflect(sample, root).drafts == ()
    ruff_toml = tmp_path / "rt"
    ruff_toml.mkdir()
    (ruff_toml / "ruff.toml").write_text('[lint]\nselect = ["E7"]\n', encoding="utf-8")
    assert already_enforced("no-bare-except", ruff_toml) is not None


def test_ruff_probe_does_not_count_ble_and_survives_bad_config(
    tmp_path: Path,
) -> None:
    """BLE001 flags `except Exception`, never a bare `except:` — it must
    not stand in for E722. A config the probe cannot parse is 'not
    enforced', never an exception out of Reflect."""
    for index, content in enumerate(
        (
            '[project]\nname = "t"\n\n[tool.ruff.lint]\nselect = ["BLE"]\n',
            '[project]\nname = "t"\n\n[tool.ruff.lint]\nselect = ["BLE001"]\n',
            '[project]\nname = "t"\n\n[tool.ruff.lint]\nselect = "E722"\n',
            '[project]\nname = "t"\n\n[tool]\nruff = "x"\n',
            '[project]\nname = "t"\n\n[tool.ruff]\nlint = "x"\n',
        )
    ):
        root = tmp_path / f"t{index}"
        root.mkdir()
        assert already_enforced("no-bare-except", _ruff_target(root, content)) is None


def test_already_enforced_is_honest_about_missing_config(tmp_path: Path) -> None:
    assert already_enforced("no-bare-except", tmp_path) is None
    assert already_enforced("no-mock-on-boundary", tmp_path) is None


def test_already_enforced_mock_needs_the_detector_not_just_the_name(
    tmp_path: Path,
) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_no_mocks_on_boundaries.py").write_text(
        "# some other test\n", encoding="utf-8"
    )
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


def test_dry_run_mock_promotes_only_with_enforcer_evidence(
    tmp_path: Path,
) -> None:
    """The mock kind's faults break no test by design; its promotion
    rests on the installed enforcer test failing on the injected tree —
    the thing validated is the thing the PR installs."""
    target = tmp_path / "target"
    target.mkdir()
    (target / "pyproject.toml").write_text('[project]\nname = "t"\n', encoding="utf-8")
    report = dry_run(
        _draft("no-mock-on-boundary"),
        corpus_root=FIXTURE_TARGET,
        target_root=target,
        work_dir=tmp_path / "work",
        runner=LocalRunner(),
    )
    assert report.injected == 1
    assert report.caught == 1
    assert report.false_positives == 0
    assert report.cases[0].fail_to_pass == ()
    assert not report.cases[0].suite_broke
    assert report.cases[0].enforcer_failed
    assert report.promotable
    assert promote(_draft("no-mock-on-boundary"), report).draft.kind == "no-mock-on-boundary"


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


def test_dry_run_report_refuses_inconsistent_numbers() -> None:
    """A hand-built report cannot authorize a payload with evidence it
    does not carry: the numbers must be derived from the listed cases."""
    with pytest.raises(FlywheelError):
        DryRunReport(
            kind="no-bare-except",
            target_repo="Zion-sewers/arena",
            injected=2,
            caught=2,
            false_positives=0,
            clean_flagged=(),
            cases=(),
            requires_behavioral_break=True,
        )
    with pytest.raises(FlywheelError):
        DryRunReport(
            kind="no-bare-except",
            target_repo="Zion-sewers/arena",
            injected=2,
            caught=2,
            false_positives=0,
            clean_flagged=(),
            cases=(_case("c1"), _case("c2", flagged=False)),
            requires_behavioral_break=True,
        )
    with pytest.raises(FlywheelError):
        DryRunReport(
            kind="no-bare-except",
            target_repo="Zion-sewers/arena",
            injected=1,
            caught=1,
            false_positives=0,
            clean_flagged=(
                Finding(file="src/a.py", line=1, message="bare except"),
            ),
            cases=(_case("c1"),),
            requires_behavioral_break=True,
        )


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


def test_wiring_extends_existing_select_with_e722_only(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "t"\n\n[tool.ruff.lint]\nselect = ["F"]\n', encoding="utf-8"
    )
    files = wiring("no-bare-except", tmp_path)
    select = tomllib.loads(files["pyproject.toml"])["tool"]["ruff"]["lint"]["select"]
    assert select == ["F", "E722"]


def test_wiring_selects_e722_only_on_a_fresh_target(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "t"\n', encoding="utf-8")
    files = wiring("no-bare-except", tmp_path)
    assert 'select = ["E722"]' in files["pyproject.toml"]


def test_wiring_is_a_noop_when_e722_already_covered(tmp_path: Path) -> None:
    text = '[project]\nname = "t"\n\n[tool.ruff.lint]\nselect = ["E"]\n'
    (tmp_path / "pyproject.toml").write_text(text, encoding="utf-8")
    assert wiring("no-bare-except", tmp_path)["pyproject.toml"] == text


def test_wiring_refuses_a_select_it_cannot_extend(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "t"\n\n[tool.ruff.lint]\nselect = "E"\n', encoding="utf-8"
    )
    with pytest.raises(FlywheelError):
        wiring("no-bare-except", tmp_path)


def test_wiring_refuses_to_clobber_a_foreign_enforcer_test(
    tmp_path: Path,
) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_no_mocks_on_boundaries.py").write_text(
        "# someone else's test\n", encoding="utf-8"
    )
    with pytest.raises(FlywheelError):
        wiring("no-mock-on-boundary", tmp_path)


# --- Ledger -------------------------------------------------------------------


def test_ledger_round_trip_and_leading_metric(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    assert leading_gate_catch_rate(ledger) is None
    append_ledger(ledger, _report())
    append_ledger(ledger, _report(caught=1))
    entries = read_ledger(ledger)
    assert len(entries) == 2
    assert entries[0]["schema_version"] == 1
    # Pooled rate: (2 + 1) / (2 + 2) = 0.75.
    assert leading_gate_catch_rate(ledger) == 0.75
    ledger.write_text("not json\n", encoding="utf-8")
    with pytest.raises(FlywheelError):
        read_ledger(ledger)


def test_ledger_refuses_corrupt_numbers(tmp_path: Path) -> None:
    """A ledger line claiming catches it does not carry, or a rate
    outside [0, 1], is refused on read — the metric never bends."""
    ledger = tmp_path / "ledger.jsonl"
    base = {
        "schema_version": 1,
        "ts": "2026-11-06T00:00:00+00:00",
        "rule_kind": "no-bare-except",
        "target_repo": "Zion-sewers/arena",
        "injected": 2,
        "caught": 2,
        "catch_rate": 1.0,
        "false_positives": 0,
        "promotable": True,
    }
    for bad in (
        {**base, "catch_rate": 2.5},
        {**base, "caught": 3},
        {**base, "caught": 1, "catch_rate": 1.0},
        {**base, "catch_rate": True},
        {**base, "injected": -1},
    ):
        ledger.write_text(json.dumps(bad) + "\n", encoding="utf-8")
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
    boundary_payload = next(
        p for p in result.payloads if p.rule_kind == "no-mock-on-boundary"
    )
    assert "find_mock_usage" in boundary_payload.files["tests/test_no_mocks_on_boundaries.py"]
    assert result.skipped == ()
    assert len(read_ledger(tmp_path / "ledger.jsonl")) == 2


def test_loop_survives_a_wiring_refusal_on_one_draft(tmp_path: Path) -> None:
    """A wiring failure on one draft is a skip, not a dead loop: the
    other draft still renders, and the ledger still holds both runs."""
    target = tmp_path / "target"
    target.mkdir()
    # A string select: the probe cannot see enforcement, so bare-except
    # proposes — but the wiring cannot extend a non-list select.
    (target / "pyproject.toml").write_text(
        '[project]\nname = "t"\n\n[tool.ruff.lint]\nselect = "E"\n', encoding="utf-8"
    )
    result = run_loop(
        transport=RecordedTransport(_recorded_loop_commits()),
        target_repo="Zion-sewers/arena",
        corpus_root=FIXTURE_TARGET,
        target_root=target,
        work_dir=tmp_path / "work",
        runner=LocalRunner(),
        ledger_path=tmp_path / "ledger.jsonl",
    )
    assert [p.rule_kind for p in result.payloads] == ["no-mock-on-boundary"]
    assert len(result.skipped) == 1
    assert "no-bare-except" in result.skipped[0]
    assert "wiring refused" in result.skipped[0]
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
            # HOME must pass through: when the outer pytest runs under a
            # python whose pytest lives in user site (~/.local), the
            # demo's `python3 -m pytest` cannot see it without HOME, the
            # dry-run baseline collects zero tests, and the demo fails
            # closed. (arena#40 fix-forward: system-python run failed
            # deterministically until HOME was passed through.)
            **({"HOME": os.environ["HOME"]} if "HOME" in os.environ else {}),
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


# --- CLI entry point ----------------------------------------------------------


def test_cli_flywheel_renders_payloads_and_persists_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agenteval-bench flywheel` is the loop's live entry point: it runs
    the real loop (stub transport, real runner and fixture corpus) and
    writes payload files plus a durable ledger under the target."""
    from agenteval_bench.cli import cmd_flywheel

    target = tmp_path / "target"
    (target / "src").mkdir(parents=True)
    (target / "src" / "a.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (target / "pyproject.toml").write_text('[project]\nname = "t"\n', encoding="utf-8")
    monkeypatch.setattr(
        "flywheel.detect.GhCliTransport",
        lambda: RecordedTransport(_recorded_loop_commits()),
    )
    rc = cmd_flywheel(
        [
            "--repo",
            "Zion-sewers/arena",
            "--target-root",
            str(target),
            "--corpus-root",
            str(FIXTURE_TARGET),
        ]
    )
    assert rc == 0
    payloads = target / ".flywheel" / "payloads"
    assert (payloads / "no-bare-except" / "PR.md").is_file()
    assert (payloads / "no-mock-on-boundary" / "PR.md").is_file()
    assert (
        "E722"
        in (payloads / "no-bare-except" / "pyproject.toml").read_text(encoding="utf-8")
    )
    ledger = target / ".flywheel" / "gate-catch-ledger.jsonl"
    assert len(read_ledger(ledger)) == 2
    assert leading_gate_catch_rate(ledger) == 1.0
