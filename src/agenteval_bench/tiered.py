"""Tiered eval pipeline (arena#39): gate -> smoke -> expanded -> nightly.

The standard eval runner for the agent company. Four phases, each with
explicit entry/exit criteria, executed in order; a phase runs only when
every earlier phase passed (entry criterion), and the pipeline stops at
the first failing phase.

- gate: the deterministic eval gate. Golden replay suite at threshold 0.9
  plus a generate-then-verify replay round-trip (bit-exactness).
- smoke: the core deterministic surface (engine + CLI + replay) as pytest.
- expanded: the full pytest suite with a coverage floor, the mock-rejection
  scan and the property tests. Coverage is always reported *alongside* the
  mutation score (nightly); a coverage percentage alone is never quality
  evidence.
- nightly: mutation scoring. Seeded AST mutants (bugsmith's procedural
  generator) are injected into throwaway copies of the repo and run against
  the repo's own suite via LocalRunner; the score is killed / validated.
  Slow by construction (one full suite run per mutant) so it lives only
  in nightly, never in a PR gate.

Every phase delegates to an existing primitive (``run --ci``, ``replay
--check``, pytest, bugsmith); this module is the phase table, the
orchestration, and the report — no new test framework, no new runner.

The runner never writes into the repo tree: replay logs, coverage data
and bugsmith work dirs all live under a temporary directory (``--out`` is
the only caller-chosen write, for the JSON report).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict

if TYPE_CHECKING:
    from bugsmith.harness import LocalRunner
    from bugsmith.types import BugCandidate

PhaseName = Literal["gate", "smoke", "expanded", "nightly"]
PhaseStatus = Literal["passed", "failed", "skipped", "error"]

PHASE_ORDER: tuple[PhaseName, ...] = ("gate", "smoke", "expanded", "nightly")

GOLDEN_SUITE = "examples/support-agent.yaml"
GATE_THRESHOLD = 0.9
DEFAULT_MUTANTS = 8
DEFAULT_SEED = 42
DEFAULT_MIN_MUTATION_SCORE = 0.5
# Pinned at the measured coverage of src/agenteval_bench on main at the
# time this module landed (80% total, 2026-11-06). Raise deliberately when
# coverage genuinely rises; never raise to an unmeasured number.
DEFAULT_MIN_COVERAGE = 80
COVERAGE_PACKAGE = "src/agenteval_bench"

# The deterministic core surface: scoring engine, CLI entry points, replay.
SMOKE_TESTS: tuple[str, ...] = (
    "tests/test_engine.py",
    "tests/test_cli.py",
    "tests/test_replay.py",
)


class TieredError(Exception):
    """Pipeline failure: a phase could not run or its evidence is missing."""


@dataclass(frozen=True)
class PhaseSpec:
    """One phase's contract: what must hold to enter, what must hold to exit."""

    name: PhaseName
    description: str
    entry_criterion: str
    exit_criterion: str


PHASE_SPECS: dict[PhaseName, PhaseSpec] = {
    "gate": PhaseSpec(
        name="gate",
        description="Deterministic eval gate on the golden replay suite.",
        entry_criterion="No earlier phase (gate is first).",
        exit_criterion=(
            f"Golden suite pass_rate >= {GATE_THRESHOLD:.0%} via `run --ci` "
            "AND a fresh replay log verifies bit-exact via `replay --check`."
        ),
    ),
    "smoke": PhaseSpec(
        name="smoke",
        description="Core deterministic surface as pytest (engine, CLI, replay).",
        entry_criterion="Gate phase passed.",
        exit_criterion="Core subset (tests/test_engine.py, tests/test_cli.py, "
        "tests/test_replay.py) green.",
    ),
    "expanded": PhaseSpec(
        name="expanded",
        description=("Full suite with coverage floor, mock-rejection scan and property tests."),
        entry_criterion="Gate and smoke phases passed.",
        exit_criterion=(
            "Full pytest suite green (includes the mock-rejection scan in "
            "tests/test_no_mocks_on_boundaries.py and the property tests in "
            "tests/test_tiered_property.py) AND measured coverage of "
            f"{COVERAGE_PACKAGE} >= floor (default {DEFAULT_MIN_COVERAGE}%)."
        ),
    ),
    "nightly": PhaseSpec(
        name="nightly",
        description=(
            "Mutation scoring: seeded AST mutants against the repo's own "
            "suite (bugsmith procedural generator + LocalRunner)."
        ),
        entry_criterion="Gate, smoke and expanded phases passed.",
        exit_criterion=(
            "Mutation score computed and reported alongside coverage AND "
            "score >= floor (default "
            f"{DEFAULT_MIN_MUTATION_SCORE:.0%}). A score that cannot be "
            "computed (red baseline, zero validated mutants) fails the phase."
        ),
    ),
}


@dataclass(frozen=True)
class CommandResult:
    """One executed command: argv, exit code, captured output."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str


class Executor(Protocol):
    """Runs a command in a directory; prod runs it for real, tests fake it."""

    def __call__(
        self, argv: list[str], cwd: Path, env: Mapping[str, str] | None = None
    ) -> CommandResult: ...


def subprocess_executor(
    argv: list[str], cwd: Path, env: Mapping[str, str] | None = None
) -> CommandResult:
    """Run argv in cwd, streaming its output to stdout as it goes.

    The subprocess inherits a PYTHONPATH rooted at the target tree so the
    target's own sources are importable even without an installed package.
    """
    merged = dict(os.environ)
    src = str(cwd / "src")
    existing = merged.get("PYTHONPATH")
    merged["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    if env:
        merged.update(env)
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    return CommandResult(argv=tuple(argv), returncode=proc.returncode, stdout=proc.stdout)


def _python_cli(argv: list[str]) -> list[str]:
    return [sys.executable, "-m", "agenteval_bench.cli", *argv]


class MutantDetailDict(TypedDict):
    target_file: str
    site_description: str
    killed: bool
    skipped: bool
    fail_to_pass: list[str]


class CommandRecordDict(TypedDict):
    argv: list[str]
    returncode: int


class PhaseOutcomeDict(TypedDict):
    phase: str
    status: str
    entry_met: bool
    exit_met: bool
    duration_s: float
    commands: list[CommandRecordDict]
    coverage_percent: float | None
    mutation_score: float | None
    mutation_details: list[MutantDetailDict]
    notes: list[str]


class TieredReportDict(TypedDict):
    tool: str
    root: str
    verdict: str
    phases_requested: list[str]
    coverage_percent: float | None
    mutation_score: float | None
    mutation_details: list[MutantDetailDict]
    phases: list[PhaseOutcomeDict]


@dataclass(frozen=True)
class PhaseOutcome:
    phase: PhaseName
    status: PhaseStatus
    entry_met: bool
    exit_met: bool
    duration_s: float
    commands: tuple[CommandResult, ...] = ()
    coverage_percent: float | None = None
    mutation_score: float | None = None
    mutation_details: tuple[MutantDetail, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> PhaseOutcomeDict:
        return {
            "phase": self.phase,
            "status": self.status,
            "entry_met": self.entry_met,
            "exit_met": self.exit_met,
            "duration_s": round(self.duration_s, 3),
            "commands": [{"argv": list(c.argv), "returncode": c.returncode} for c in self.commands],
            "coverage_percent": self.coverage_percent,
            "mutation_score": self.mutation_score,
            "mutation_details": [
                {
                    "target_file": d.target_file,
                    "site_description": d.site_description,
                    "killed": d.killed,
                    "skipped": d.skipped,
                    "fail_to_pass": list(d.fail_to_pass),
                }
                for d in self.mutation_details
            ],
            "notes": list(self.notes),
        }

    @staticmethod
    def from_dict(doc: Mapping[str, object]) -> PhaseOutcome:
        phase = str(doc["phase"])
        if phase not in PHASE_ORDER:
            raise TieredError(f"corrupt report: unknown phase {phase!r}")
        status = str(doc["status"])
        if status not in ("passed", "failed", "skipped", "error"):
            raise TieredError(f"corrupt report: unknown status {status!r}")
        commands = tuple(
            CommandResult(
                argv=tuple(str(a) for a in c["argv"]),  # type: ignore[index]
                returncode=int(c["returncode"]),  # type: ignore[index]
                stdout="",
            )
            for c in doc["commands"]  # type: ignore[union-attr]
        )
        details = tuple(
            MutantDetail(
                target_file=str(d["target_file"]),  # type: ignore[index]
                site_description=str(d["site_description"]),  # type: ignore[index]
                killed=bool(d["killed"]),  # type: ignore[index]
                skipped=bool(d.get("skipped", False)),  # type: ignore[attr-defined]
                fail_to_pass=tuple(str(t) for t in d["fail_to_pass"]),  # type: ignore[index]
            )
            for d in doc.get("mutation_details", [])  # type: ignore[union-attr]
        )
        return PhaseOutcome(
            phase=phase,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            entry_met=bool(doc["entry_met"]),
            exit_met=bool(doc["exit_met"]),
            duration_s=float(doc["duration_s"]),
            commands=commands,
            coverage_percent=(
                None if doc["coverage_percent"] is None else float(doc["coverage_percent"])  # type: ignore[arg-type]
            ),
            mutation_score=(
                None if doc["mutation_score"] is None else float(doc["mutation_score"])  # type: ignore[arg-type]
            ),
            mutation_details=details,
            notes=tuple(str(n) for n in doc["notes"]),  # type: ignore[union-attr]
        )


@dataclass(frozen=True)
class TieredReport:
    root: str
    verdict: Literal["PASS", "FAIL"]
    phases: tuple[PhaseOutcome, ...]
    phases_requested: tuple[PhaseName, ...] = ()
    coverage_percent: float | None = None
    mutation_score: float | None = None
    mutation_details: tuple[MutantDetail, ...] = ()

    def to_dict(self) -> TieredReportDict:
        return {
            "tool": "agenteval-bench",
            "root": self.root,
            "verdict": self.verdict,
            "phases_requested": [p for p in self.phases_requested] or [p.phase for p in self.phases],
            "coverage_percent": self.coverage_percent,
            "mutation_score": self.mutation_score,
            "mutation_details": [
                {
                    "target_file": d.target_file,
                    "site_description": d.site_description,
                    "killed": d.killed,
                    "skipped": d.skipped,
                    "fail_to_pass": list(d.fail_to_pass),
                }
                for d in self.mutation_details
            ],
            "phases": [p.to_dict() for p in self.phases],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def from_dict(doc: Mapping[str, object]) -> TieredReport:
        phases = tuple(
            PhaseOutcome.from_dict(p)  # type: ignore[arg-type]
            for p in doc["phases"]  # type: ignore[union-attr]
        )
        details = tuple(
            MutantDetail(
                target_file=str(d["target_file"]),  # type: ignore[index]
                site_description=str(d["site_description"]),  # type: ignore[index]
                killed=bool(d["killed"]),  # type: ignore[index]
                skipped=bool(d.get("skipped", False)),  # type: ignore[attr-defined]
                fail_to_pass=tuple(str(t) for t in d["fail_to_pass"]),  # type: ignore[index]
            )
            for d in doc.get("mutation_details", [])  # type: ignore[union-attr]
        )
        verdict = str(doc["verdict"])
        if verdict not in ("PASS", "FAIL"):
            raise TieredError(f"corrupt report: unknown verdict {verdict!r}")
        requested = tuple(
            str(p) for p in doc.get("phases_requested", [p.phase for p in phases])  # type: ignore[union-attr]
        )
        for p in requested:
            if p not in PHASE_ORDER:
                raise TieredError(f"corrupt report: unknown requested phase {p!r}")
        return TieredReport(
            root=str(doc["root"]),
            verdict=verdict,  # type: ignore[arg-type]
            phases=phases,
            phases_requested=requested,  # type: ignore[arg-type]
            coverage_percent=(
                None if doc["coverage_percent"] is None else float(doc["coverage_percent"])  # type: ignore[arg-type]
            ),
            mutation_score=(
                None if doc["mutation_score"] is None else float(doc["mutation_score"])  # type: ignore[arg-type]
            ),
            mutation_details=details,
        )

    @staticmethod
    def from_json(text: str) -> TieredReport:
        doc: Mapping[str, object] = json.loads(text)
        return TieredReport.from_dict(doc)


def phases_up_to(phase: str) -> tuple[PhaseName, ...]:
    """Phases to run for ``--phase <name>``: the prefix ending at that phase.

    ``all`` runs every phase. An unknown name raises TieredError (usage
    error, never a silent subset).
    """
    if phase == "all":
        return PHASE_ORDER
    if phase not in PHASE_ORDER:
        raise TieredError(
            f"unknown phase {phase!r}: expected one of {', '.join([*PHASE_ORDER, 'all'])}"
        )
    idx = PHASE_ORDER.index(phase)  # type: ignore[arg-type]
    return PHASE_ORDER[: idx + 1]


@dataclass(frozen=True)
class MutantDetail:
    target_file: str
    site_description: str
    killed: bool
    skipped: bool = False
    fail_to_pass: tuple[str, ...] = ()


@dataclass(frozen=True)
class MutationScore:
    killed: int
    validated: int
    skipped: int
    score: float
    details: tuple[MutantDetail, ...] = ()

    def __post_init__(self) -> None:
        if self.validated < 1:
            raise TieredError(
                "mutation score undefined: zero validated mutants "
                "(nothing was measured, refusing to report 0.0 or 1.0)"
            )
        if not 0 <= self.killed <= self.validated:
            raise TieredError(f"killed={self.killed} outside [0, validated={self.validated}]")
        expected = self.killed / self.validated
        if abs(self.score - expected) > 1e-9:
            raise TieredError(f"mutation score {self.score} != killed/validated {expected}")


def mutation_score(root: Path, *, mutants: int, seed: int, work_dir: Path) -> MutationScore:
    """Kill-rate of seeded AST mutants against the repo's own suite.

    Reuses bugsmith's procedural generator + LocalRunner: mutants are
    injected into throwaway copies, never the live tree, and the clean-tree
    baseline must be green or the measurement refuses to run (a red
    baseline would make every mutant look killed).

    The scorer must not score itself: mutants are drawn from the evaluated
    packages only, never from ``src/bugsmith`` — the generator and the
    harness that compute the kill verdicts are excluded as targets, so a
    mutant in the scoring path cannot fabricate the score (fail-open
    toward 1.0) or void it. The exclusion is enforced by the generator
    (``exclude``) and pinned by a test in tests/test_bugsmith.py.

    The mutant runs execute under this process's own interpreter: the
    harness invokes bare ``python3``, which on a developer machine may
    lack the repo's dev dependencies (hypothesis lives in ``.[dev]``),
    and a collection error there would refuse the measurement for an
    environment reason, not a suite reason. Prepending the current
    interpreter's directory to PATH for the duration of the measurement
    keeps the mutation target honest in every environment, including CI
    (where the two interpreters are already the same).
    """
    from bugsmith.buggen import ProceduralBugGenerator
    from bugsmith.harness import LocalRunner

    if mutants < 1:
        raise TieredError(f"mutants must be >= 1, got {mutants}")
    work_dir.mkdir(parents=True, exist_ok=True)
    generator = ProceduralBugGenerator(seed)
    candidates = generator.generate(root, mutants, exclude=("src/bugsmith",))
    if not candidates:
        raise TieredError("bug generator produced zero candidates: nothing to score")
    runner = LocalRunner()
    path_before = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{path_before}"
    # The suite contains a real end-to-end expanded test
    # (tests/test_tiered.py::test_cli_tiered_expanded_runs_for_real) that
    # spawns its own `tiered --phase expanded` subprocess: inside a mutant
    # copy that would recurse and multiply nightly runtime, so mutant and
    # baseline runs skip it (the env guard in that test).
    e2e_before = os.environ.get("TIERED_E2E_ACTIVE")
    os.environ["TIERED_E2E_ACTIVE"] = "1"
    try:
        passed = _mutation_measure(candidates, root, runner, work_dir)
    finally:
        os.environ["PATH"] = path_before
        if e2e_before is None:
            os.environ.pop("TIERED_E2E_ACTIVE", None)
        else:
            os.environ["TIERED_E2E_ACTIVE"] = e2e_before
    return passed


def _mutation_measure(
    candidates: list[BugCandidate],
    root: Path,
    runner: LocalRunner,
    work_dir: Path,
) -> MutationScore:
    """Baseline + validate loop. Caller owns the interpreter PATH setup."""
    from bugsmith.harness import baseline, validate
    from bugsmith.types import BugsmithError

    try:
        passed = baseline(root, runner, work_dir)
    except BugsmithError as e:
        raise TieredError(f"mutation baseline refused: {e}") from e
    killed = 0
    validated = 0
    skipped = 0
    details: list[MutantDetail] = []
    for candidate in candidates:
        try:
            report = validate(candidate, root, runner, work_dir, passed)
        except BugsmithError as e:
            skipped += 1
            details.append(
                MutantDetail(
                    target_file=candidate.record.target_file,
                    site_description=(f"{candidate.record.site_description} (unvalidatable: {e})"),
                    killed=False,
                    skipped=True,
                )
            )
            continue
        validated += 1
        is_killed = report.is_valid_instance
        killed += 1 if is_killed else 0
        details.append(
            MutantDetail(
                target_file=candidate.record.target_file,
                site_description=candidate.record.site_description,
                killed=is_killed,
                fail_to_pass=tuple(report.fail_to_pass),
            )
        )
    if validated == 0:
        raise TieredError(
            f"mutation score undefined: {len(candidates)} generated, "
            f"{skipped} unvalidatable, 0 validated"
        )
    return MutationScore(
        killed=killed,
        validated=validated,
        skipped=skipped,
        score=killed / validated,
        details=tuple(details),
    )


def _timed(fn: Callable[[], PhaseOutcome]) -> PhaseOutcome:
    start = time.monotonic()
    outcome = fn()
    return PhaseOutcome(
        phase=outcome.phase,
        status=outcome.status,
        entry_met=outcome.entry_met,
        exit_met=outcome.exit_met,
        duration_s=time.monotonic() - start,
        commands=outcome.commands,
        coverage_percent=outcome.coverage_percent,
        mutation_score=outcome.mutation_score,
        mutation_details=outcome.mutation_details,
        notes=outcome.notes,
    )


def _run_gate(root: Path, executor: Executor, tmp: Path) -> PhaseOutcome:
    results: list[CommandResult] = []
    notes: list[str] = []
    replay_log = tmp / "gate-replay.json"
    results.append(
        executor(
            _python_cli(
                ["run", "--suite", GOLDEN_SUITE, "--ci", "--threshold", str(GATE_THRESHOLD)]
            ),
            root,
        )
    )
    results.append(
        executor(
            _python_cli(["run", "--suite", GOLDEN_SUITE, "--replay-log", str(replay_log)]),
            root,
        )
    )
    results.append(
        executor(
            _python_cli(["replay", "--log", str(replay_log), "--suite", GOLDEN_SUITE, "--check"]),
            root,
        )
    )
    ok = all(r.returncode == 0 for r in results)
    if not ok:
        failed = [r for r in results if r.returncode != 0]
        notes.append(
            f"gate exit criterion NOT met: {len(failed)} command(s) failed "
            f"(first: {' '.join(failed[0].argv)} -> {failed[0].returncode})"
        )
    return PhaseOutcome(
        phase="gate",
        status="passed" if ok else "failed",
        entry_met=True,
        exit_met=ok,
        duration_s=0.0,
        commands=tuple(results),
        notes=tuple(notes),
    )


def _run_smoke(root: Path, executor: Executor, tmp: Path) -> PhaseOutcome:
    results = [
        executor(
            [sys.executable, "-m", "pytest", *SMOKE_TESTS, "-q", "-p", "no:cacheprovider"],
            root,
        )
    ]
    ok = results[0].returncode == 0
    notes: list[str] = []
    if not ok:
        notes.append("smoke exit criterion NOT met: core pytest subset failed")
    return PhaseOutcome(
        phase="smoke",
        status="passed" if ok else "failed",
        entry_met=True,
        exit_met=ok,
        duration_s=0.0,
        commands=tuple(results),
        notes=tuple(notes),
    )


def _coverage_from_json(path: Path) -> float | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        totals = doc["totals"]
        return float(totals["percent_covered"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _run_expanded(root: Path, executor: Executor, tmp: Path, min_coverage: float) -> PhaseOutcome:
    coverage_json = tmp / "coverage.json"
    coverage_data = tmp / ".coverage"
    # Invocation note (pinned deviation from the old CI Test step): the old
    # step ran `pytest tests/ -v --tb=short`. This runs `-q` instead of `-v`
    # (verbosity only; --tb=short is pytest's default) and adds
    # `-p no:cacheprovider` so the run never writes into the repo tree.
    # Same test selection, same failure semantics.
    results = [
        executor(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--cov={COVERAGE_PACKAGE}",
                f"--cov-fail-under={min_coverage:g}",
                f"--cov-report=json:{coverage_json}",
            ],
            root,
            env={"COVERAGE_FILE": str(coverage_data)},
        )
    ]
    coverage = _coverage_from_json(coverage_json)
    notes: list[str] = [
        "mock scan enforced via tests/test_no_mocks_on_boundaries.py (pytest)",
        "property tests enforced via tests/test_tiered_property.py (pytest)",
    ]
    if coverage is None:
        notes.append("coverage report missing: pytest produced no coverage.json")
    pytest_ok = results[0].returncode == 0
    coverage_ok = coverage is not None and coverage >= min_coverage
    ok = pytest_ok and coverage_ok
    if not ok:
        reasons = []
        if not pytest_ok:
            reasons.append(f"pytest exit {results[0].returncode}")
        if not coverage_ok:
            reasons.append(
                f"coverage {coverage} < floor {min_coverage:g}%"
                if coverage is not None
                else "coverage unmeasured"
            )
        notes.append(f"expanded exit criterion NOT met: {'; '.join(reasons)}")
    return PhaseOutcome(
        phase="expanded",
        status="passed" if ok else "failed",
        entry_met=True,
        exit_met=ok,
        duration_s=0.0,
        commands=tuple(results),
        coverage_percent=coverage,
        notes=tuple(notes),
    )


def _run_nightly(
    root: Path,
    tmp: Path,
    mutants: int,
    seed: int,
    min_mutation_score: float,
    mutation_fn: Callable[..., MutationScore],
) -> PhaseOutcome:
    notes: list[str] = []
    try:
        result = mutation_fn(root, mutants=mutants, seed=seed, work_dir=tmp / "mutation")
    except TieredError as e:
        return PhaseOutcome(
            phase="nightly",
            status="error",
            entry_met=True,
            exit_met=False,
            duration_s=0.0,
            notes=(f"nightly exit criterion NOT met: mutation score unmeasurable: {e}",),
        )
    except Exception as e:  # noqa: BLE001 — any scorer failure must land in the report, not a traceback
        return PhaseOutcome(
            phase="nightly",
            status="error",
            entry_met=True,
            exit_met=False,
            duration_s=0.0,
            notes=(f"nightly exit criterion NOT met: mutation scorer crashed: {type(e).__name__}: {e}",),
        )
    ok = result.score >= min_mutation_score
    notes.append(
        f"mutation score {result.killed}/{result.validated} = "
        f"{result.score:.0%} (skipped {result.skipped})"
    )
    for d in result.details:
        if not d.killed:
            notes.append(f"survived: {d.target_file} ({d.site_description})")
    if not ok:
        notes.append(
            f"nightly exit criterion NOT met: score {result.score:.0%} "
            f"< floor {min_mutation_score:.0%}"
        )
    return PhaseOutcome(
        phase="nightly",
        status="passed" if ok else "failed",
        entry_met=True,
        exit_met=ok,
        duration_s=0.0,
        mutation_score=result.score,
        mutation_details=result.details,
        notes=tuple(notes),
    )


def run_pipeline(
    root: Path,
    *,
    upto: str = "all",
    mutants: int = DEFAULT_MUTANTS,
    seed: int = DEFAULT_SEED,
    min_mutation_score: float = DEFAULT_MIN_MUTATION_SCORE,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    executor: Executor | None = None,
    mutation_fn: Callable[..., MutationScore] | None = None,
) -> TieredReport:
    """Run the tiered pipeline up to ``upto`` (a phase name or ``all``).

    Phases run in order; the first failing phase stops the pipeline and
    later phases are reported ``skipped`` (their entry criterion — all
    earlier phases passed — is not met). Returns the report; callers map
    verdict FAIL to a non-zero exit.
    """
    phases = phases_up_to(upto)
    if not 0.0 <= min_mutation_score <= 1.0:
        raise TieredError(f"min_mutation_score must be in [0, 1], got {min_mutation_score}")
    if not 0.0 <= min_coverage <= 100.0:
        raise TieredError(f"min_coverage must be in [0, 100], got {min_coverage}")
    if mutants < 1:
        raise TieredError(f"mutants must be >= 1, got {mutants}")
    run_executor = executor or subprocess_executor
    run_mutation = mutation_fn or mutation_score
    root = root.resolve()
    if not (root / GOLDEN_SUITE).is_file():
        raise TieredError(f"golden suite not found under root: {root / GOLDEN_SUITE}")
    outcomes: list[PhaseOutcome] = []
    coverage: float | None = None
    mutation: float | None = None
    mutation_details: tuple[MutantDetail, ...] = ()
    failed = False
    with tempfile.TemporaryDirectory(prefix="tiered-") as tmp_s:
        tmp = Path(tmp_s)
        for phase in phases:
            if failed:
                outcomes.append(
                    PhaseOutcome(
                        phase=phase,
                        status="skipped",
                        entry_met=False,
                        exit_met=False,
                        duration_s=0.0,
                        notes=("entry criterion NOT met: an earlier phase failed",),
                    )
                )
                continue
            if phase == "gate":
                outcome = _timed(lambda: _run_gate(root, run_executor, tmp))
            elif phase == "smoke":
                outcome = _timed(lambda: _run_smoke(root, run_executor, tmp))
            elif phase == "expanded":
                outcome = _timed(lambda: _run_expanded(root, run_executor, tmp, min_coverage))
            else:
                outcome = _timed(
                    lambda: _run_nightly(root, tmp, mutants, seed, min_mutation_score, run_mutation)
                )
            outcomes.append(outcome)
            if outcome.coverage_percent is not None:
                coverage = outcome.coverage_percent
            if outcome.mutation_score is not None:
                mutation = outcome.mutation_score
            if outcome.mutation_details:
                mutation_details = outcome.mutation_details
            if outcome.status != "passed":
                failed = True
    verdict: Literal["PASS", "FAIL"] = "FAIL" if failed else "PASS"
    return TieredReport(
        root=str(root),
        verdict=verdict,
        phases=tuple(outcomes),
        phases_requested=phases,
        coverage_percent=coverage,
        mutation_score=mutation,
        mutation_details=mutation_details,
    )
