"""Validate: dry-run a proposed rule before it may become blocking (arena#40).

The dry-run is the gate between Propose and PR, and it runs against real
execution, not prose:
1. Clean-tree check: the rule predicate runs over the real target repo
   AND the validation corpus. Any flag on clean code is a false positive,
   and a blocking rule with false positives is refused.
2. Class injection: a deterministic fault of the rule's class is injected
   into a throwaway copy of the corpus (the vendored mcalc fixture). The
   bugsmith harness validates the injection as a real bug (fail_to_pass
   evidence from a real suite run via the injected runner), and the rule
   predicate must flag the injected tree.
3. Promotion requires: at least one injected fault (no vacuous dry-run),
   every injected fault caught, zero false positives, and — for kinds
   whose faults must be behavioral — every injected fault broke a test.
4. Enforcer check: for kinds whose faults are stylistic (no behavioral
   break to measure), the test the PR would install is placed into the
   injected copy and run: it must FAIL there. The thing validated is the
   thing the PR installs, not a second implementation of the same scan.

The target tree is never written: injection happens on throwaway copies
under the caller-owned work_dir, and bugsmith's own validate() copies
again before applying the patch. Any harness failure that is not a test
result is wrapped as FlywheelError (fail-closed): an unmeasurable
dry-run never reports a number, and no raw runner exception escapes.
"""

from __future__ import annotations

import difflib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from bugsmith.harness import Runner, baseline
from bugsmith.harness import validate as bugsmith_validate
from bugsmith.types import BugCandidate, BugsmithError, BugStrategy, GenerationRecord
from flywheel.rules import RULES, inject, scan, wiring
from flywheel.types import (
    DraftRule,
    DryRunReport,
    FlywheelError,
    InjectionCase,
    RuleKind,
    ValidatedRule,
)


def _unified_patch(rel: str, original: str, injected: str) -> str:
    """A unified diff (a/ b/ form, as bugsmith's patch parser expects)
    turning the clean file into the injected file."""
    original_lines = original.splitlines(keepends=True)
    injected_lines = injected.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines,
        injected_lines,
        fromfile=f"a/{rel}",
        tofile=f"b/{rel}",
    )
    return "".join(diff)


def _fresh_copy(source: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        source, dest, ignore=shutil.ignore_patterns("__pycache__", ".git", ".hypothesis")
    )
    return dest


def _run_enforcer_check(injected_copy: Path, kind: RuleKind) -> bool:
    """Run the test the PR would install against the injected tree.

    The wiring's enforcer file is placed into the throwaway copy and run
    with pytest: exit code 1 means the enforcer failed there, i.e. it
    caught the injected fault. Exit 0 means the enforcer passed on a
    fault of its own kind (no evidence). Any other outcome is
    infrastructure failure, not a measurement — fail-closed.
    """
    files = wiring(kind, injected_copy)
    enforcer_rel = next(iter(files))
    enforcer_path = injected_copy / enforcer_rel
    enforcer_path.parent.mkdir(parents=True, exist_ok=True)
    enforcer_path.write_text(files[enforcer_rel], encoding="utf-8")
    env = dict(os.environ)
    src = injected_copy / "src"
    env["PYTHONPATH"] = str(src if src.is_dir() else injected_copy)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                enforcer_rel,
                "-q",
                "-p",
                "no:cacheprovider",
                f"--rootdir={injected_copy}",
            ],
            cwd=str(injected_copy),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
            text=True,
        )
    except subprocess.TimeoutExpired as e:
        raise FlywheelError(f"enforcer check timed out for {kind}: {e}") from e
    except OSError as e:
        raise FlywheelError(f"enforcer check could not run for {kind}: {e}") from e
    if proc.returncode == 1:
        return True
    if proc.returncode == 0:
        return False
    raise FlywheelError(
        f"enforcer check for {kind} ended with pytest exit "
        f"{proc.returncode} (infrastructure, not a measurement): "
        f"{proc.stdout[-500:]}"
    )


def dry_run(
    draft: DraftRule,
    *,
    corpus_root: Path,
    target_root: Path,
    work_dir: Path,
    runner: Runner,
    seed: int = 42,
) -> DryRunReport:
    """Dry-run one DraftRule. Raises FlywheelError (fail-closed) when the
    harness cannot establish a baseline or a suite run fails as
    infrastructure — an unmeasurable dry-run never reports a number."""
    kind = draft.kind
    spec = RULES[kind]
    clean_flagged = tuple(scan(kind, target_root)) + tuple(scan(kind, corpus_root))
    try:
        baseline_passed = baseline(corpus_root, runner, work_dir)
    except BugsmithError as e:
        raise FlywheelError(f"dry-run baseline failed for {kind}: {e}") from e
    except Exception as e:
        raise FlywheelError(f"dry-run baseline failed for {kind}: {e}") from e

    cases: list[InjectionCase] = []
    for index, case_name in enumerate(spec.injection_sites):
        injected_copy = _fresh_copy(
            corpus_root, work_dir / f"flywheel-inject-{kind}-{index}"
        )
        rel, injected_text = inject(kind, injected_copy, case_name)
        original_text = (corpus_root / rel).read_text(encoding="utf-8")
        patch = _unified_patch(rel, original_text, injected_text)
        if not patch.strip():
            raise FlywheelError(
                f"injector for {kind}:{case_name} produced an empty patch: "
                "refusing a vacuous injection"
            )
        (injected_copy / rel).write_text(injected_text, encoding="utf-8")
        candidate = BugCandidate(
            record=GenerationRecord(
                strategy=BugStrategy.PROCEDURAL_AST,
                seed=seed,
                generator_version="flywheel-1",
                target_file=rel,
                site_description=f"flywheel:{kind}:{case_name}",
                extra={"flywheel_kind": kind, "injection": "kind-targeted"},
            ),
            patch=patch,
        )
        try:
            report = bugsmith_validate(candidate, corpus_root, runner, work_dir, baseline_passed)
        except BugsmithError as e:
            raise FlywheelError(
                f"dry-run validation failed for {kind}:{case_name}: {e}"
            ) from e
        except Exception as e:
            raise FlywheelError(
                f"dry-run validation failed for {kind}:{case_name}: {e}"
            ) from e
        flagged = bool(scan(kind, injected_copy))
        enforcer_failed = (
            _run_enforcer_check(injected_copy, kind)
            if spec.requires_enforcer_check
            else False
        )
        cases.append(
            InjectionCase(
                name=case_name,
                file=rel,
                flagged=flagged,
                fail_to_pass=tuple(report.fail_to_pass),
                suite_broke=bool(report.fail_to_pass),
                enforcer_failed=enforcer_failed,
            )
        )

    caught = sum(1 for c in cases if c.flagged)
    return DryRunReport(
        kind=kind,
        target_repo=draft.target_repo,
        injected=len(cases),
        caught=caught,
        false_positives=len(clean_flagged),
        clean_flagged=clean_flagged,
        cases=tuple(cases),
        requires_behavioral_break=spec.requires_behavioral_break,
        requires_enforcer_check=spec.requires_enforcer_check,
    )


def promote(draft: DraftRule, report: DryRunReport) -> ValidatedRule:
    """Authorize a DraftRule for a PR payload. Refuses (raises) unless the
    dry-run is promotable; the ValidatedRule constructor re-checks."""
    if report.kind != draft.kind or report.target_repo != draft.target_repo:
        raise FlywheelError(
            f"dry-run report does not belong to this draft "
            f"({report.kind}/{report.target_repo} vs {draft.kind}/{draft.target_repo})"
        )
    if not report.promotable:
        raise FlywheelError(
            f"cannot promote {draft.kind}: dry-run not promotable "
            f"(injected={report.injected}, caught={report.caught}, "
            f"catch_rate={report.catch_rate:.2f}, "
            f"false_positives={report.false_positives}, "
            f"behavioral_breaks={sum(1 for c in report.cases if c.suite_broke)}, "
            f"enforcer_failures={sum(1 for c in report.cases if c.enforcer_failed)})"
        )
    return ValidatedRule(draft=draft, report=report)
