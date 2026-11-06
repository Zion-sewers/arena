"""Tiered pipeline tests (arena#39): phase table, orchestration, report.

Orchestration tests use a fake executor that records argv and returns
canned exit codes, plus a fake mutation function — no subprocess, no
bugsmith. The real commands are proven by the CLI tests at the bottom:
gate runs for real, and expanded runs for real end to end
(``test_cli_tiered_expanded_runs_for_real``, which executes the full
suite through the CLI). Nightly runs for real in the nightly workflow
and in the measured proof runs; the suite pins the mechanism with a
real (tiny) mutation target below.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from agenteval_bench import cli
from agenteval_bench.tiered import (
    PHASE_ORDER,
    PHASE_SPECS,
    CommandResult,
    MutationScore,
    TieredError,
    TieredReport,
    phases_up_to,
    run_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeExecutor:
    """Records every argv; returns 0 unless a command matches a failure rule."""

    def __init__(self, fail_substrings: tuple[str, ...] = ()) -> None:
        self.calls: list[list[str]] = []
        self.fail_substrings = fail_substrings
        self.coverage_percent = 85.0

    def __call__(
        self, argv: list[str], cwd: Path, env: Mapping[str, str] | None = None
    ) -> CommandResult:
        self.calls.append(list(argv))
        # Simulate pytest's coverage side effect so expanded can be judged.
        for arg in argv:
            if arg.startswith("--cov-report=json:"):
                path = Path(arg.split(":", 1)[1])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({"totals": {"percent_covered": self.coverage_percent}}),
                    encoding="utf-8",
                )
        joined = " ".join(argv)
        failed = any(s in joined for s in self.fail_substrings)
        return CommandResult(argv=tuple(argv), returncode=2 if failed else 0, stdout="")

    def saw(self, substring: str) -> bool:
        return any(substring in " ".join(call) for call in self.calls)


def _fake_mutation(root: Path, *, mutants: int, seed: int, work_dir: Path) -> MutationScore:
    return MutationScore(killed=6, validated=8, skipped=0, score=0.75)


def _low_mutation(root: Path, *, mutants: int, seed: int, work_dir: Path) -> MutationScore:
    return MutationScore(killed=2, validated=8, skipped=0, score=0.25)


# --- phase table ---


def test_phase_order_is_gate_smoke_expanded_nightly() -> None:
    assert PHASE_ORDER == ("gate", "smoke", "expanded", "nightly")


def test_every_phase_has_explicit_entry_and_exit_criteria() -> None:
    assert set(PHASE_SPECS) == set(PHASE_ORDER)
    for name in PHASE_ORDER:
        spec = PHASE_SPECS[name]
        assert spec.name == name
        assert spec.description.strip()
        assert spec.entry_criterion.strip()
        assert spec.exit_criterion.strip()


def test_phases_up_to_returns_prefix_ending_at_phase() -> None:
    assert phases_up_to("gate") == ("gate",)
    assert phases_up_to("smoke") == ("gate", "smoke")
    assert phases_up_to("expanded") == ("gate", "smoke", "expanded")
    assert phases_up_to("nightly") == PHASE_ORDER
    assert phases_up_to("all") == PHASE_ORDER


def test_phases_up_to_rejects_unknown_phase() -> None:
    with pytest.raises(TieredError, match="unknown phase"):
        phases_up_to("bogus")


# --- mutation score math ---


def test_mutation_score_rejects_zero_validated() -> None:
    with pytest.raises(TieredError, match="zero validated"):
        MutationScore(killed=0, validated=0, skipped=2, score=0.0)


def test_mutation_score_rejects_killed_above_validated() -> None:
    with pytest.raises(TieredError, match="outside"):
        MutationScore(killed=9, validated=8, skipped=0, score=1.125)


def test_mutation_score_rejects_inconsistent_ratio() -> None:
    with pytest.raises(TieredError, match="!="):
        MutationScore(killed=6, validated=8, skipped=0, score=0.5)


# --- orchestration with the fake executor ---


def test_gate_only_run_passes_with_fake_executor() -> None:
    report = run_pipeline(REPO_ROOT, upto="gate", executor=FakeExecutor())
    assert report.verdict == "PASS"
    assert [p.phase for p in report.phases] == ["gate"]
    assert report.phases[0].status == "passed"
    assert report.phases[0].entry_met and report.phases[0].exit_met


def test_expanded_run_reports_coverage_from_fake_pytest() -> None:
    executor = FakeExecutor()
    executor.coverage_percent = 85.0
    report = run_pipeline(REPO_ROOT, upto="expanded", executor=executor)
    assert report.verdict == "PASS"
    assert report.coverage_percent == 85.0
    assert executor.saw("pytest")
    assert executor.saw("--cov-fail-under=80")


def test_expanded_fails_when_coverage_below_floor() -> None:
    executor = FakeExecutor()
    executor.coverage_percent = 60.0
    report = run_pipeline(REPO_ROOT, upto="expanded", executor=executor)
    assert report.verdict == "FAIL"
    expanded = report.phases[-1]
    assert expanded.phase == "expanded"
    assert expanded.status == "failed"
    assert not expanded.exit_met
    assert report.coverage_percent == 60.0


def test_failing_gate_blocks_later_phases_and_never_runs_them() -> None:
    # Adversarial: the gate's eval command fails, so smoke/expanded must
    # be SKIPPED (entry criterion unmet) and pytest must never be invoked.
    executor = FakeExecutor(fail_substrings=("run --suite",))
    report = run_pipeline(REPO_ROOT, upto="expanded", executor=executor)
    assert report.verdict == "FAIL"
    assert [p.phase for p in report.phases] == ["gate", "smoke", "expanded"]
    assert report.phases[0].status == "failed"
    assert report.phases[1].status == "skipped"
    assert report.phases[2].status == "skipped"
    assert not report.phases[1].entry_met
    assert not report.phases[2].entry_met
    assert not executor.saw("pytest"), "smoke/expanded ran pytest after a failed gate"


def test_failing_smoke_blocks_expanded_but_gate_stays_passed() -> None:
    executor = FakeExecutor(fail_substrings=("test_engine.py",))
    report = run_pipeline(REPO_ROOT, upto="expanded", executor=executor)
    assert report.verdict == "FAIL"
    assert report.phases[0].status == "passed"
    assert report.phases[1].status == "failed"
    assert report.phases[2].status == "skipped"


def test_nightly_passes_when_mutation_score_meets_floor() -> None:
    report = run_pipeline(
        REPO_ROOT,
        upto="nightly",
        executor=FakeExecutor(),
        mutation_fn=_fake_mutation,
    )
    assert report.verdict == "PASS"
    nightly = report.phases[-1]
    assert nightly.status == "passed"
    assert report.mutation_score == 0.75


def test_nightly_fails_when_mutation_score_below_floor() -> None:
    report = run_pipeline(
        REPO_ROOT,
        upto="nightly",
        executor=FakeExecutor(),
        mutation_fn=_low_mutation,
    )
    assert report.verdict == "FAIL"
    nightly = report.phases[-1]
    assert nightly.status == "failed"
    assert report.mutation_score == 0.25


def test_nightly_errors_when_mutation_score_unmeasurable() -> None:
    def _boom(root: Path, *, mutants: int, seed: int, work_dir: Path) -> MutationScore:
        raise TieredError("zero validated mutants")

    report = run_pipeline(REPO_ROOT, upto="nightly", executor=FakeExecutor(), mutation_fn=_boom)
    assert report.verdict == "FAIL"
    nightly = report.phases[-1]
    assert nightly.status == "error"
    assert not nightly.exit_met
    assert report.mutation_score is None


def test_invalid_thresholds_rejected_before_any_phase_runs() -> None:
    executor = FakeExecutor()
    with pytest.raises(TieredError, match="min_mutation_score"):
        run_pipeline(REPO_ROOT, upto="gate", executor=executor, min_mutation_score=1.5)
    with pytest.raises(TieredError, match="min_coverage"):
        run_pipeline(REPO_ROOT, upto="gate", executor=executor, min_coverage=101.0)
    assert executor.calls == []


def test_missing_golden_suite_rejected() -> None:
    import tempfile

    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.raises(TieredError, match="golden suite not found"),
    ):
        run_pipeline(Path(tmp), upto="gate", executor=FakeExecutor())


def test_mutation_score_runs_against_tiny_target_with_dev_dep(tmp_path: Path) -> None:
    """End-to-end mutation scoring against a real (tiny) target.

    The target's test imports hypothesis, a dev-only dependency: it only
    collects when the mutant runs execute under an interpreter that has
    the dev deps. If mutation_score ever goes back to bare system
    ``python3``, the baseline refuses (collection error) and this test
    fails with TieredError instead of silently measuring nothing.
    """
    from agenteval_bench.tiered import mutation_score

    target = tmp_path / "target"
    (target / "src" / "tiny").mkdir(parents=True)
    (target / "src" / "tiny" / "__init__.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8"
    )
    (target / "tests").mkdir(parents=True)
    (target / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (target / "tests" / "test_tiny.py").write_text(
        "from hypothesis import given\n"
        "from hypothesis import strategies as st\n"
        "from tiny import add\n\n\n"
        "def test_add() -> None:\n"
        "    assert add(1, 2) == 3\n\n\n"
        "@given(a=st.integers(), b=st.integers())\n"
        "def test_add_is_commutative(a: int, b: int) -> None:\n"
        "    assert add(a, b) == add(b, a)\n",
        encoding="utf-8",
    )
    result = mutation_score(
        target, mutants=1, seed=42, work_dir=tmp_path / "work"
    )
    assert result.validated == 1
    assert result.killed == 1
    assert result.score == 1.0
    assert result.details[0].target_file == "src/tiny/__init__.py"


def test_report_json_round_trip() -> None:
    report = run_pipeline(
        REPO_ROOT,
        upto="nightly",
        executor=FakeExecutor(),
        mutation_fn=_fake_mutation,
    )
    restored = TieredReport.from_json(report.to_json())
    assert restored.to_dict() == report.to_dict()


# --- CLI surface (real commands) ---


def test_cli_tiered_unknown_phase_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.cmd_tiered(["--phase", "bogus", "--root", str(REPO_ROOT)]) == 1
    assert "unknown phase" in capsys.readouterr().err


def test_cli_tiered_unknown_flag_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.cmd_tiered(["--phase", "gate", "--root", str(REPO_ROOT), "--bogus", "x"]) == 1
    assert "unknown argument" in capsys.readouterr().err


def test_cli_tiered_gate_runs_for_real(tmp_path: Path) -> None:
    out = tmp_path / "tiered-report.json"
    rc = cli.cmd_tiered(["--phase", "gate", "--root", str(REPO_ROOT), "--out", str(out)])
    assert rc == 0
    assert out.is_file()
    report = TieredReport.from_json(out.read_text(encoding="utf-8"))
    assert report.verdict == "PASS"
    assert [p.phase for p in report.phases] == ["gate"]
    assert report.phases_requested == ("gate",)


def test_cli_tiered_expanded_runs_for_real(tmp_path: Path) -> None:
    """End-to-end expanded phase for real, through the CLI subprocess.

    Runs gate -> smoke -> expanded on this repo itself and asserts the
    JSON report: all three phases passed, coverage at or above the floor.
    This is the test that would catch a bug in the real expanded command
    construction (the fake-executor orchestration tests cannot).

    Recursion guard: the inner run executes the full suite, which
    includes this test. The inner run sets TIERED_E2E_ACTIVE, and this
    test skips when it is already set, so the nesting stops at depth 1.
    The same guard is set by tiered.mutation_score for baseline/mutant
    runs, so nightly does not pay for a nested expanded run per mutant.
    """
    if os.environ.get("TIERED_E2E_ACTIVE"):
        pytest.skip("nested expanded run: recursion guard")
    out = tmp_path / "tiered-expanded-report.json"
    env = dict(os.environ)
    env["TIERED_E2E_ACTIVE"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenteval_bench.cli",
            "tiered",
            "--phase",
            "expanded",
            "--root",
            str(REPO_ROOT),
            "--out",
            str(out),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert out.is_file()
    report = TieredReport.from_json(out.read_text(encoding="utf-8"))
    assert report.verdict == "PASS"
    assert [p.phase for p in report.phases] == ["gate", "smoke", "expanded"]
    assert report.phases_requested == ("gate", "smoke", "expanded")
    assert report.coverage_percent is not None
    assert report.coverage_percent >= 80
    # A partial run is honest about what it measured: no mutation score.
    assert report.mutation_score is None


def test_report_rejects_corrupt_phase_and_verdict() -> None:
    report = run_pipeline(
        REPO_ROOT,
        upto="gate",
        executor=FakeExecutor(),
    )
    doc = report.to_dict()
    doc["phases"][0]["phase"] = "bogus"
    with pytest.raises(TieredError, match="unknown phase"):
        TieredReport.from_json(json.dumps(doc))
    doc = report.to_dict()
    doc["verdict"] = "MAYBE"
    with pytest.raises(TieredError, match="unknown verdict"):
        TieredReport.from_json(json.dumps(doc))


def test_run_pipeline_rejects_zero_mutants_up_front() -> None:
    with pytest.raises(TieredError, match="mutants must be >= 1"):
        run_pipeline(REPO_ROOT, upto="nightly", mutants=0, executor=FakeExecutor())


def test_cli_tiered_unwritable_report_is_usage_error(tmp_path: Path) -> None:
    """An unwritable --out path exits 1, not 0: the report is the evidence.

    Kills the cli.py mutant that flips the OSError return from 1 to -1:
    a failed report write must fail the command, never pass silently.
    """
    missing_dir = tmp_path / "no-such-dir"
    rc = cli.cmd_tiered(
        ["--phase", "gate", "--root", str(REPO_ROOT), "--out", str(missing_dir / "r.json")]
    )
    assert rc == 1


def test_cli_tiered_bad_numbers_are_usage_error() -> None:
    assert cli.cmd_tiered(["--mutants", "not-a-number"]) == 1
