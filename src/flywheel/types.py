"""Typed stage data for the fix-commit flywheel (arena#40).

Stage separation is enforced by distinct frozen types, not by string
flags: a DraftRule is unvalidated by construction, and a ValidatedRule
cannot exist without a promotable DryRunReport (its __post_init__ refuses
anything else). Only a ValidatedRule can be rendered into a PR payload,
so the dry-run gate is structural, not a comment asking nicely.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

RuleKind = Literal["no-bare-except", "no-mock-on-boundary"]

# A blocking rule must catch every injected instance of its class.
# Anything less means the predicate itself is wrong, not unlucky.
MIN_CATCH_RATE = 1.0


class FlywheelError(Exception):
    """The loop refuses a step: bad input, inconclusive dry-run, or a rule
    that must not become blocking. Never silently degraded."""


@dataclass(frozen=True)
class Finding:
    """One place a rule predicate fires."""

    file: str  # repo-relative path
    line: int
    message: str


@dataclass(frozen=True)
class FixCommit:
    """One fix-commit as Detect sees it. ``files`` is empty from the list
    endpoint (per-commit file fetch is a follow-up); Reflect signals use
    the message only, and says so."""

    sha: str
    message: str
    files: tuple[str, ...] = ()
    repo: str = ""


@dataclass(frozen=True)
class DraftRule:
    """A proposed blocking rule that has NOT been validated yet. It must
    never reach a PR payload in this form."""

    kind: RuleKind
    target_repo: str
    evidence_shas: tuple[str, ...]
    support: int
    rationale: str


@dataclass(frozen=True)
class InjectionCase:
    """One injected fault of the rule's class, and what happened to it."""

    name: str
    file: str  # repo-relative path of the injected file
    flagged: bool  # the rule predicate fired on the injected tree
    fail_to_pass: tuple[str, ...]  # suite tests the injection broke
    suite_broke: bool  # fail_to_pass non-empty: the injection is a real bug


@dataclass(frozen=True)
class DryRunReport:
    """The dry-run verdict for one DraftRule. ``promotable`` is the only
    path to a PR payload."""

    kind: RuleKind
    target_repo: str
    injected: int
    caught: int
    false_positives: int
    clean_flagged: tuple[Finding, ...]
    cases: tuple[InjectionCase, ...]
    requires_behavioral_break: bool

    @property
    def catch_rate(self) -> float:
        """Gate-catch rate: the leading metric. 0.0 when nothing was
        injected (a vacuous dry-run scores zero, never 100%)."""
        if self.injected == 0:
            return 0.0
        return self.caught / self.injected

    @property
    def promotable(self) -> bool:
        """A rule becomes blocking only when: at least one fault of its
        class was injected (no vacuous promotion), every injected fault
        was caught, the clean tree is unflagged, and — for kinds whose
        faults must be behavioral — every injected fault broke a test."""
        if self.injected == 0:
            return False
        if self.catch_rate < MIN_CATCH_RATE:
            return False
        if self.false_positives != 0:
            return False
        if self.requires_behavioral_break:
            return all(c.suite_broke for c in self.cases)
        return True


@dataclass(frozen=True)
class ValidatedRule:
    """A DraftRule plus the passing dry-run that authorizes it. Cannot be
    constructed from a failing report."""

    draft: DraftRule
    report: DryRunReport

    def __post_init__(self) -> None:
        if self.report.kind != self.draft.kind:
            raise FlywheelError(
                f"report kind {self.report.kind} does not match draft kind "
                f"{self.draft.kind}: refusing to authorize"
            )
        if self.report.target_repo != self.draft.target_repo:
            raise FlywheelError(
                f"report target {self.report.target_repo} does not match draft "
                f"target {self.draft.target_repo}: refusing to authorize"
            )
        if not self.report.promotable:
            raise FlywheelError(
                f"dry-run for {self.draft.kind} is not promotable "
                f"(injected={self.report.injected}, caught={self.report.caught}, "
                f"false_positives={self.report.false_positives}): refusing to "
                "authorize a PR payload"
            )


@dataclass(frozen=True)
class PrPayload:
    """A postable PR for a validated rule: title, body, and full file
    contents. The flywheel never writes these into the target tree; the
    fleet runner posts the payload via the GitHub API."""

    title: str
    body: str
    files: Mapping[str, str]
    target_repo: str
    rule_kind: RuleKind


@dataclass(frozen=True)
class SkippedKind:
    """A rule kind Reflect considered and refused, with the reason kept."""

    kind: RuleKind
    reason: str


@dataclass(frozen=True)
class LoopResult:
    """What one full Detect -> PR loop run produced."""

    payloads: tuple[PrPayload, ...] = ()
    skipped: tuple[str, ...] = ()
    reports: tuple[DryRunReport, ...] = ()
    detected: int = 0
    sampled: int = 0
    drafts: tuple[DraftRule, ...] = field(default_factory=tuple)
