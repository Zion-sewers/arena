"""Loop: tie Detect -> Sample -> Reflect -> Propose -> Validate -> PR (arena#40).

The orchestrator takes every external effect as an injected seam
(transport, runner, work_dir, ledger path). There is no human step in
the loop: no callback, no prompt, no approval call — the function
signature has nowhere for a human to sit. Manual spot-checking is not
the loop; the loop completes on its own or fails loudly.
"""

from __future__ import annotations

from pathlib import Path

from bugsmith.harness import Runner
from flywheel.detect import GitHubTransport, sample_commits
from flywheel.propose import append_ledger, render_pr_payload
from flywheel.reflect import reflect
from flywheel.types import FlywheelError, LoopResult
from flywheel.validate import dry_run, promote


def run_loop(
    *,
    transport: GitHubTransport,
    target_repo: str,
    corpus_root: Path,
    target_root: Path,
    work_dir: Path,
    runner: Runner,
    ledger_path: Path,
    sample_size: int = 10,
    seed: int = 42,
) -> LoopResult:
    """Run one full loop. Returns the PR payloads (possibly zero), the
    skipped kinds/drafts with reasons, and every dry-run report. Raises
    FlywheelError when a stage cannot measure what it claims."""
    commits = transport.list_fix_commits(target_repo)
    sampled = sample_commits(commits, sample_size, seed=seed)
    reflection = reflect(sampled, target_root)

    payloads = []
    skipped: list[str] = [
        f"{s.kind}: already enforced ({s.reason})" for s in reflection.skipped
    ]
    reports = []
    for draft in reflection.drafts:
        report = dry_run(
            draft,
            corpus_root=corpus_root,
            target_root=target_root,
            work_dir=work_dir,
            runner=runner,
            seed=seed,
        )
        reports.append(report)
        if report.promotable:
            try:
                payloads.append(render_pr_payload(promote(draft, report), target_root))
            except FlywheelError as e:
                # A wiring failure on one draft must not kill the loop or
                # discard the other drafts' payloads: record the skip and
                # keep going. The ledger still records the dry-run (the
                # measurement happened); the skip records that no PR came
                # out of it.
                skipped.append(f"{draft.kind}: wiring refused ({e})")
        else:
            skipped.append(
                f"{draft.kind}: dry-run not promotable "
                f"(injected={report.injected}, caught={report.caught}, "
                f"catch_rate={report.catch_rate:.2f}, "
                f"false_positives={report.false_positives})"
            )
        # Ledger after the draft's disposition is known, exactly once per
        # draft: a crashed loop must never record a promotable run that
        # produced no payload.
        append_ledger(ledger_path, report)
    return LoopResult(
        payloads=tuple(payloads),
        skipped=tuple(skipped),
        reports=tuple(reports),
        detected=len(commits),
        sampled=len(sampled),
        drafts=reflection.drafts,
    )
