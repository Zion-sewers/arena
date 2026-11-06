"""Seeded end-to-end demo of the fix-commit flywheel (arena#40).

Runs the full loop offline on recorded fix-commits against the vendored
mcalc fixture corpus, with this repo as the dogfood target. Exit 0 only
if every acceptance expectation holds:
- the loop completes with no human step (stdin is never read);
- a bare-except draft is proposed from the recorded fix-commits, passes
  its dry-run, and becomes a PR payload (PR only — the target tree is
  never written, pinned by digest before/after);
- the mock kind is skipped as already enforced in this repo;
- the gate-catch rate is ledgered (the leading metric exists).

Any failure raises, so CI can gate on this demo directly.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from bugsmith.harness import LocalRunner
from bugsmith.images import repo_digest
from flywheel.detect import is_fix_commit
from flywheel.loop import run_loop
from flywheel.propose import leading_gate_catch_rate, read_ledger
from flywheel.types import FixCommit

HERE = Path(__file__).resolve().parent
# HERE is src/flywheel, so parents[0]=src, parents[1]=repo root.
REPO_ROOT = HERE.parents[1]
FIXTURE_TARGET = HERE.parent / "bugsmith" / "fixtures" / "target"

# Recorded fix-commits (demo fixtures, not live history): three signal
# bare-except, two signal mock-on-boundary, one is not a fix-commit.
RECORDED_COMMITS = [
    FixCommit(
        sha="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
        message="fix: replace bare except in engine retry loop",
        repo="Zion-sewers/arena",
    ),
    FixCommit(
        sha="b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1",
        message="fix(cli): blind except swallowed KeyboardInterrupt in replay",
        repo="Zion-sewers/arena",
    ),
    FixCommit(
        sha="c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2",
        message="fix: swallowed exception hid a crash in snapshots",
        repo="Zion-sewers/arena",
    ),
    FixCommit(
        sha="d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3",
        message="fix: mock leaked across tests in the engine suite",
        repo="Zion-sewers/arena",
    ),
    FixCommit(
        sha="e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4",
        message="fix: remove mock of the engine boundary in replay tests",
        repo="Zion-sewers/arena",
    ),
    FixCommit(
        sha="f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5",
        message="feat: add animated suite scoreboard",
        repo="Zion-sewers/arena",
    ),
]


class RecordedTransport:
    """Hand-written stub transport carrying the recorded commits."""

    def __init__(self, commits: list[FixCommit]) -> None:
        self._commits = commits

    def list_fix_commits(self, repo: str, since: str | None = None) -> list[FixCommit]:
        del since
        return [c for c in self._commits if c.repo == repo and is_fix_commit(c.message)]


def main() -> int:
    target_root = REPO_ROOT
    digest_before = repo_digest(target_root)
    with tempfile.TemporaryDirectory(prefix="flywheel-demo-") as tmp:
        work_dir = Path(tmp)
        ledger_path = work_dir / "gate-catch-ledger.jsonl"
        result = run_loop(
            transport=RecordedTransport(RECORDED_COMMITS),
            target_repo="Zion-sewers/arena",
            corpus_root=FIXTURE_TARGET,
            target_root=target_root,
            work_dir=work_dir,
            runner=LocalRunner(),
            ledger_path=ledger_path,
        )
        digest_after = repo_digest(target_root)
        assert digest_before == digest_after, "loop wrote into the target tree"
        assert result.detected == 5, f"expected 5 fix-commits, got {result.detected}"
        assert result.sampled == 5, f"expected 5 sampled, got {result.sampled}"
        assert len(result.payloads) == 1, f"expected 1 PR payload, got {len(result.payloads)}"
        payload = result.payloads[0]
        assert payload.rule_kind == "no-bare-except", payload.rule_kind
        assert payload.target_repo == "Zion-sewers/arena"
        assert "pyproject.toml" in payload.files, "payload must wire the ruff select"
        assert 'E722' in payload.files["pyproject.toml"], "payload must select E722"
        assert any("no-mock-on-boundary" in s for s in result.skipped), (
            f"mock kind must be skipped as already enforced, skipped={result.skipped}"
        )
        assert len(result.reports) == 1
        report = result.reports[0]
        assert report.promotable, "bare-except dry-run must promote"
        assert report.catch_rate == 1.0, f"catch rate {report.catch_rate}"
        assert report.false_positives == 0
        entries = read_ledger(ledger_path)
        assert len(entries) == 1, f"ledger must hold 1 dry-run, got {len(entries)}"
        rate = leading_gate_catch_rate(ledger_path)
        assert rate == 1.0, f"leading gate-catch rate {rate}"
        print(f"flywheel demo: detected={result.detected} sampled={result.sampled}")
        print(f"payload: {payload.title}")
        print(f"skipped: {'; '.join(result.skipped)}")
        print(f"gate-catch rate (leading metric): {rate:.0%}")
        print("target tree untouched (digest pinned before/after)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
