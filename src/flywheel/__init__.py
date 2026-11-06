"""Fix-commit flywheel (arena#40): Detect -> Sample -> Reflect -> Propose -> Validate -> PR.

A meta-agent loop that turns fix-commits in agent repos into proposed
blocking rules, opened as PRs — never by editing the target code directly.
The loop is a closed chain of typed stages: only a rule that survived a
dry-run (injected faults of its class caught, zero false positives on the
clean tree) can become a PR payload, and the gate-catch rate of every
dry-run is appended to a JSONL ledger as the leading metric.

Evidence levels in this module: the stage mechanics are established (they
run in tests and in the demo); the cross-repo rollout beyond arena is
adaptation (arena is the dogfood target; other repos are a follow-up).
"""

from __future__ import annotations

from flywheel.detect import GhCliTransport, GitHubTransport, is_fix_commit, sample_commits
from flywheel.loop import run_loop
from flywheel.propose import append_ledger, leading_gate_catch_rate, read_ledger, render_pr_payload
from flywheel.reflect import Reflection, reflect
from flywheel.types import (
    DraftRule,
    DryRunReport,
    Finding,
    FixCommit,
    FlywheelError,
    InjectionCase,
    LoopResult,
    PrPayload,
    RuleKind,
    ValidatedRule,
)
from flywheel.validate import dry_run, promote

__all__ = [
    "DraftRule",
    "DryRunReport",
    "Finding",
    "FixCommit",
    "FlywheelError",
    "GhCliTransport",
    "GitHubTransport",
    "InjectionCase",
    "LoopResult",
    "PrPayload",
    "Reflection",
    "RuleKind",
    "ValidatedRule",
    "append_ledger",
    "dry_run",
    "is_fix_commit",
    "leading_gate_catch_rate",
    "promote",
    "read_ledger",
    "reflect",
    "render_pr_payload",
    "run_loop",
    "sample_commits",
]
