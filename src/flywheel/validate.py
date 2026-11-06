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

The target tree is never written: injection happens on throwaway copies
under the caller-owned work_dir, and bugsmith's own validate() copies
again before applying the patch.
"""

from __future__ import annotations

import difflib
import shutil
from pathlib import Path

from bugsmith.harness import Runner, baseline
from bugsmith.harness import validate as bugsmith_validate
from bugsmith.types import BugCandidate, BugsmithError, BugStrategy, GenerationRecord
from flywheel.rules import RULES, inject, scan
from flywheel.types import (
    DraftRule,
    DryRunReport,
    FlywheelError,
    InjectionCase,
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
        flagged = bool(scan(kind, injected_copy))
        cases.append(
            InjectionCase(
                name=case_name,
                file=rel,
                flagged=flagged,
                fail_to_pass=tuple(report.fail_to_pass),
                suite_broke=bool(report.fail_to_pass),
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
            f"behavioral_breaks={sum(1 for c in report.cases if c.suite_broke)})"
        )
    return ValidatedRule(draft=draft, report=report)
