"""Reflect: aggregate sampled fix-commits into rule proposals (arena#40).

Reflect is deterministic aggregation over the closed rule registry, not
a language model: it counts fix-commits whose message carries a kind's
signal keywords, and proposes the kind when the count meets the support
floor. A kind already enforced in the target repo is skipped with the
reason kept — proposing a rule that already blocks is noise, not signal.

Free-form rule synthesis (a model reading fix-commits and inventing new
rule kinds) is a named follow-up: it needs a model transport this repo
does not have, and this module does not simulate one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flywheel.rules import RULES, already_enforced
from flywheel.types import DraftRule, FixCommit, RuleKind, SkippedKind

# A single fix-commit is anecdote; two independent fix-commits with the
# same signal are a pattern worth proposing a rule for.
SUPPORT_FLOOR = 2


@dataclass(frozen=True)
class Reflection:
    """What Reflect produced from one sample."""

    drafts: tuple[DraftRule, ...]
    skipped: tuple[SkippedKind, ...]


def _signals(kind: RuleKind, commit: FixCommit) -> bool:
    message = commit.message.lower()
    return any(keyword in message for keyword in RULES[kind].signal_keywords)


def reflect(sample: tuple[FixCommit, ...], target_root: Path) -> Reflection:
    """Aggregate one sample into DraftRules. ``target_root`` is the repo
    the rules would land in; already-enforced kinds are skipped with
    their enforcer named."""
    drafts: list[DraftRule] = []
    skipped: list[SkippedKind] = []
    for kind in RULES:
        evidence = [c for c in sample if _signals(kind, c)]
        if len(evidence) < SUPPORT_FLOOR:
            continue
        enforced = already_enforced(kind, target_root)
        if enforced is not None:
            skipped.append(SkippedKind(kind=kind, reason=enforced))
            continue
        drafts.append(
            DraftRule(
                kind=kind,
                target_repo=evidence[0].repo,
                evidence_shas=tuple(c.sha for c in evidence),
                support=len(evidence),
                rationale=(
                    f"{len(evidence)} sampled fix-commits signal {kind} "
                    f"({', '.join(sorted({c.sha[:7] for c in evidence}))})"
                ),
            )
        )
    return Reflection(drafts=tuple(drafts), skipped=tuple(skipped))
