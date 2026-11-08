"""Metric-assumption ledger + hard-backup rule (arena#56).

Berman 'Financial Intelligence' (catalog G.11), INSPIRED: metrics reflect
assumptions, and every soft metric needs a hard verifying metric — the
"cash test". This module is the ledger for the metrics this repo reports.

The hard/soft line
------------------
HARD: the number is a count or a sum of recorded observations, with no
inference step between the phenomenon and the number (pass_rate is a
count of scored cases; cost is a sum of measured per-case costs).
SOFT: the number is produced by an inference — a judge's grade, an
interval, a label, an assessment. A soft metric is only reportable
with a named HARD backup: the hard count it must be checkable against.

Acceptance (arena#56), enforced here, not in prose:
- every reported metric lists its assumptions — ``MetricSpec`` refuses
  an empty ``assumptions`` tuple at construction;
- a soft metric without hard backup is flagged — ``audit_ledger``
  flags it, and the fail-closed parsers refuse to build it;
- assumption-free metrics are NOT accepted — construction, parsing,
  and audit all refuse them.

Honesty boundary: the ledger lists assumptions; it cannot verify they
are TRUE. Each assumption carries a ``falsifier`` — the observation
that would break it — so every assumption is checkable, the same rule
as the prespecification ``Assumption`` (src/experiment/prespec.py).

Scope: the ledger covers the metrics in the ``run`` and ``decide``
reports (pass_rate, the four decision dimensions, and the soft
metrics those reports print). NOT covered yet: tiered
``MutationScore`` (kill rate), regression aggregate scores vs pinned
baselines, alt-test alignment scores and p-values, novelty-oracle
``novelty_score``. Those reports carry no ledger entry; the "every
metric" claim is scoped to the covered set until they are wired in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class MetricKind(str, Enum):
    """Whether a metric is directly measured or inferred."""

    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True)
class MetricAssumption:
    """One stated assumption behind a metric, with its falsifier.

    An assumption without a falsifier is a slogan, not a checkable
    claim: construction refuses empty text or an empty falsifier.
    """

    text: str
    falsifier: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("a metric assumption needs non-empty text")
        if not self.falsifier:
            raise ValueError(
                f"assumption {self.text!r} needs a falsifier: state what "
                "observation would break it"
            )

    def to_dict(self) -> dict[str, str]:
        """JSON-serialisable form for reports."""
        return {"text": self.text, "falsifier": self.falsifier}


@dataclass(frozen=True)
class MetricSpec:
    """The ledger entry for one reported metric.

    ``assumptions`` must be non-empty: an assumption-free metric is not
    accepted. A SOFT metric must name ``hard_backup`` — a HARD metric in
    the same ledger the soft number is checkable against. A HARD metric
    carries no backup (it IS the checkable count).
    """

    name: str
    kind: MetricKind
    phenomenon: str
    assumptions: tuple[MetricAssumption, ...]
    hard_backup: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a metric spec needs a non-empty name")
        if not self.phenomenon:
            raise ValueError(f"metric {self.name!r} needs a named phenomenon")
        if not self.assumptions:
            raise ValueError(
                f"metric {self.name!r} is assumption-free: assumption-free "
                "metrics are not accepted"
            )
        if self.kind is MetricKind.SOFT and not self.hard_backup:
            raise ValueError(
                f"soft metric {self.name!r} needs a hard backup: name the "
                "HARD metric it is checkable against"
            )
        if self.kind is MetricKind.HARD and self.hard_backup is not None:
            raise ValueError(
                f"hard metric {self.name!r} carries no backup: it IS the "
                "checkable count"
            )

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form for reports."""
        return {
            "kind": self.kind.value,
            "phenomenon": self.phenomenon,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "hard_backup": self.hard_backup,
        }


def _a(text: str, falsifier: str) -> MetricAssumption:
    """Shorthand for the ledger entries below."""
    return MetricAssumption(text=text, falsifier=falsifier)


#: The repo's reported metrics, with their assumptions and hard backups.
#: HARD entries are grounded in the modules that compute them; SOFT
#: entries name the HARD count each is checkable against.
METRIC_LEDGER: Mapping[str, MetricSpec] = MappingProxyType(
    {
        "pass_rate": MetricSpec(
            name="pass_rate",
            kind=MetricKind.HARD,
            phenomenon="fraction of scored cases the agent passed",
            assumptions=(
                _a(
                    "The scorer's matchers decide pass/fail exactly as "
                    "each case specifies",
                    "a case passes whose output does not satisfy its "
                    "expected output on manual inspection",
                ),
                _a(
                    "Skipped cases are excluded from the denominator, "
                    "never counted as passes or failures",
                    "a skipped case changes pass_rate",
                ),
                _a(
                    "One seed, one pass: the rate carries no cross-run "
                    "variation",
                    "re-running with a different seed changes pass_rate",
                ),
            ),
        ),
        "quality": MetricSpec(
            name="quality",
            kind=MetricKind.HARD,
            phenomenon="fraction of quality-dimension cases passed",
            assumptions=(
                _a(
                    "Case dimension tags are correct: quality cases "
                    "measure quality",
                    "a case tagged quality is found to measure safety "
                    "instead",
                ),
                _a(
                    "The proportion is over scored cases only; skipped "
                    "cases are excluded",
                    "a skipped quality case changes the quality proportion",
                ),
                _a(
                    "The Wilson 95% interval is the uncertainty carried "
                    "with the proportion, not the point estimate",
                    "a report quotes the quality proportion with no "
                    "interval",
                ),
            ),
        ),
        "safety": MetricSpec(
            name="safety",
            kind=MetricKind.HARD,
            phenomenon="fraction of safety-dimension cases passed",
            assumptions=(
                _a(
                    "Case dimension tags are correct: safety cases "
                    "measure safety",
                    "a case tagged safety is found to measure quality "
                    "instead",
                ),
                _a(
                    "The proportion is over scored cases only; skipped "
                    "cases are excluded",
                    "a skipped safety case changes the safety proportion",
                ),
                _a(
                    "The Wilson 95% interval is the uncertainty carried "
                    "with the proportion, not the point estimate",
                    "a report quotes the safety proportion with no "
                    "interval",
                ),
            ),
        ),
        "reliability": MetricSpec(
            name="reliability",
            kind=MetricKind.HARD,
            phenomenon="agreement of the agent's verdicts across repeated "
            "seeded runs",
            assumptions=(
                _a(
                    "The seeded agent function genuinely perturbs with "
                    "the run seed",
                    "two runs with different seeds produce byte-identical "
                    "outputs on every case",
                ),
                _a(
                    "Runs are independent samples, not replays of one "
                    "execution",
                    "the agent returns cached outputs regardless of seed",
                ),
                _a(
                    "Replay-agreement is not live-agent stability",
                    "a replay-agreement number is quoted as evidence of "
                    "live-agent flakiness",
                ),
            ),
        ),
        "cost": MetricSpec(
            name="cost",
            kind=MetricKind.HARD,
            phenomenon="total measured cost of the run in the policy's "
            "cost units",
            assumptions=(
                _a(
                    "The cost function measures in the policy's cost units",
                    "a case cost recorded in different units changes the "
                    "total",
                ),
                _a(
                    "One unmeasured case makes the whole cost dimension "
                    "unmeasured, never silently gated on a partial sample",
                    "a run with an unmeasured case reports a cost total",
                ),
            ),
        ),
        "judge_score": MetricSpec(
            name="judge_score",
            kind=MetricKind.SOFT,
            phenomenon="a judge's graded score for the agent's output",
            assumptions=(
                _a(
                    "The judge's prompt measures the criterion it claims "
                    "to grade",
                    "the judge scores highly an output that fails the "
                    "criterion on manual inspection",
                ),
                _a(
                    "The judge is calibrated: its scores may gate only "
                    "when calibrated",
                    "an uncalibrated judge's scores gate a decision",
                ),
                _a(
                    "The judge is decorrelated from the agent it grades "
                    "(HOST != FAMILY)",
                    "judge and agent share a host/family and the score is "
                    "quoted as independent",
                ),
            ),
            hard_backup="quality",
        ),
        "calibration_gap": MetricSpec(
            name="calibration_gap",
            kind=MetricKind.SOFT,
            phenomenon="mean judge confidence minus measured agreement rate",
            assumptions=(
                _a(
                    "Confidence is self-reported by the judge, not "
                    "measured",
                    "a confidence value is quoted as a measured accuracy",
                ),
                _a(
                    "The agreement rate is a measured proportion of "
                    "correct judgments",
                    "agreement_rate is computed from unrecorded judgments",
                ),
            ),
            hard_backup="pass_rate",
        ),
        "scorer_validity": MetricSpec(
            name="scorer_validity",
            kind=MetricKind.SOFT,
            phenomenon="whether the scorer's matchers measure what they "
            "claim to measure",
            assumptions=(
                _a(
                    "A proxy grade is earned by a validation against "
                    "ground truth, never asserted",
                    "a matcher carries VALIDATED_PROXY with no validation "
                    "run",
                ),
                _a(
                    "The bar applies on the Wilson lower bound, not the "
                    "point estimate",
                    "a 5/5 proxy at small n clears the bar",
                ),
            ),
            hard_backup="pass_rate",
        ),
        "perturbation_stability": MetricSpec(
            name="perturbation_stability",
            kind=MetricKind.SOFT,
            phenomenon="whether the agent's verdict survives input "
            "perturbation",
            assumptions=(
                _a(
                    "Stability is the drop versus the clean-regime pass "
                    "rate, not an absolute level",
                    "a stability label is quoted without the clean-regime "
                    "rate",
                ),
                _a(
                    "Recorded perturbed inputs equal perturb_input() for "
                    "the seed, or the regime is unmeasured",
                    "a regime is scored from a recording whose input does "
                    "not match the perturbation",
                ),
            ),
            hard_backup="pass_rate",
        ),
    }
)


class LedgerStatus(str, Enum):
    """The audit verdict for one metric in a ledger."""

    OK = "ok"
    FLAGGED_NO_ASSUMPTIONS = "flagged_no_assumptions"
    FLAGGED_NO_BACKUP = "flagged_no_backup"
    FLAGGED_BAD_BACKUP = "flagged_bad_backup"


@dataclass(frozen=True)
class LedgerFinding:
    """One metric's audit verdict, with the reason it was flagged."""

    metric: str
    status: LedgerStatus
    detail: str

    def to_dict(self) -> dict[str, str]:
        """JSON-serialisable form for reports."""
        return {
            "metric": self.metric,
            "status": self.status.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LedgerAudit:
    """The audit of a whole ledger: every metric's finding.

    ``findings`` is copied and frozen at construction, like Decision's
    verdicts: a frozen dataclass alone does not stop tuple mutation.
    """

    findings: tuple[LedgerFinding, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))

    @property
    def flagged(self) -> list[str]:
        """Metrics the audit flagged, in metric order."""
        return sorted(
            f.metric for f in self.findings if f.status is not LedgerStatus.OK
        )

    @property
    def all_ok(self) -> bool:
        """True only when no metric in the ledger is flagged."""
        return not self.flagged

    def summary(self) -> str:
        """Human-readable table: every metric with its kind, backup, and
        assumption count. A flagged metric is named, never hidden."""
        lines = [
            (
                f"Metric ledger: {len(self.findings) - len(self.flagged)}"
                f"/{len(self.findings)} metrics clean"
            )
        ]
        for finding in sorted(self.findings, key=lambda f: f.metric):
            spec = METRIC_LEDGER.get(finding.metric)
            if spec is None:
                lines.append(f"  {finding.metric:22s} UNKNOWN — {finding.detail}")
                continue
            backup = spec.hard_backup or "—"
            lines.append(
                f"  {finding.metric:22s} {spec.kind.value:4s} "
                f"backup={backup:10s} "
                f"assumptions={len(spec.assumptions)} "
                f"{'OK' if finding.status is LedgerStatus.OK else 'FLAGGED: ' + finding.detail}"
            )
        if self.flagged:
            lines.append("  FLAGGED: " + ", ".join(self.flagged))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form for reports."""
        return {
            "findings": [f.to_dict() for f in self.findings],
            "all_ok": self.all_ok,
            "flagged": self.flagged,
        }


def audit_ledger(
    ledger: Mapping[str, MetricSpec] = METRIC_LEDGER,
) -> LedgerAudit:
    """Audit a ledger: every metric lists assumptions; every soft metric
    names a hard backup that exists in the ledger and IS hard.

    Construction already refuses assumption-free specs and soft specs
    without a backup, so the audit re-verifies defensively: a hand-built
    mapping cannot slip a bad entry past it. A soft metric whose backup
    names a missing metric, or a SOFT metric, is flagged.
    """
    findings: list[LedgerFinding] = []
    for name in sorted(ledger):
        spec = ledger[name]
        if spec.name != name:
            findings.append(
                LedgerFinding(
                    metric=name,
                    status=LedgerStatus.FLAGGED_BAD_BACKUP,
                    detail=f"spec filed under {name!r} is named {spec.name!r}",
                )
            )
            continue
        if not spec.assumptions:
            findings.append(
                LedgerFinding(
                    metric=name,
                    status=LedgerStatus.FLAGGED_NO_ASSUMPTIONS,
                    detail="assumption-free metrics are not accepted",
                )
            )
            continue
        if spec.kind is MetricKind.SOFT:
            backup = spec.hard_backup
            if not backup:
                findings.append(
                    LedgerFinding(
                        metric=name,
                        status=LedgerStatus.FLAGGED_NO_BACKUP,
                        detail="soft metric without hard backup",
                    )
                )
            elif backup not in ledger:
                findings.append(
                    LedgerFinding(
                        metric=name,
                        status=LedgerStatus.FLAGGED_BAD_BACKUP,
                        detail=f"hard backup {backup!r} is not in the ledger",
                    )
                )
            elif ledger[backup].kind is not MetricKind.HARD:
                findings.append(
                    LedgerFinding(
                        metric=name,
                        status=LedgerStatus.FLAGGED_BAD_BACKUP,
                        detail=f"hard backup {backup!r} is not HARD",
                    )
                )
            else:
                findings.append(
                    LedgerFinding(
                        metric=name,
                        status=LedgerStatus.OK,
                        detail=f"backed by hard metric {backup!r}",
                    )
                )
        else:
            findings.append(
                LedgerFinding(
                    metric=name,
                    status=LedgerStatus.OK,
                    detail="hard metric: directly measured",
                )
            )
    return LedgerAudit(findings=tuple(findings))


def require_ledger_ok(audit: LedgerAudit) -> None:
    """Raise on a flagged ledger, naming the flagged metrics.

    Callers that gate on the ledger use this; report printers use
    :meth:`LedgerAudit.summary` and let the reader see the flag.
    """
    if not audit.all_ok:
        raise ValueError(
            "metric ledger has flagged metrics: " + ", ".join(audit.flagged)
        )


def _require_str(value: object, field_name: str, metric: str) -> str:
    """One non-empty string field from an untrusted mapping."""
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"metric {metric!r}: field {field_name!r} must be a non-empty string"
        )
    return value


def _as_assumptions(value: object, metric: str) -> tuple[MetricAssumption, ...]:
    """Parse the assumptions list of one metric spec, fail-closed."""
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"metric {metric!r}: field 'assumptions' must be a non-empty list"
        )
    parsed: list[MetricAssumption] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError(
                f"metric {metric!r}: each assumption must be a mapping"
            )
        parsed.append(
            MetricAssumption(
                text=_require_str(item.get("text"), "text", metric),
                falsifier=_require_str(item.get("falsifier"), "falsifier", metric),
            )
        )
    return tuple(parsed)


def spec_from_dict(payload: Mapping[str, object]) -> MetricSpec:
    """Parse one metric spec from an untrusted mapping, fail-closed.

    Missing assumptions, a soft metric without a backup, a hard metric
    with one, and a non-string field are all refused here — the same
    rules :class:`MetricSpec` enforces at construction.
    """
    if not isinstance(payload, dict):
        raise TypeError("a metric spec must be a mapping")
    name = _require_str(payload.get("name"), "name", "<unknown>")
    kind_raw = _require_str(payload.get("kind"), "kind", name)
    if kind_raw not in ("hard", "soft"):
        raise ValueError(
            f"metric {name!r}: field 'kind' must be 'hard' or 'soft', "
            f"got {kind_raw!r}"
        )
    kind = MetricKind(kind_raw)
    phenomenon = _require_str(payload.get("phenomenon"), "phenomenon", name)
    assumptions = _as_assumptions(payload.get("assumptions"), name)
    backup_raw = payload.get("hard_backup")
    if backup_raw is not None and (
        not isinstance(backup_raw, str) or not backup_raw
    ):
        raise ValueError(
            f"metric {name!r}: field 'hard_backup' must be a non-empty "
            "string or absent"
        )
    return MetricSpec(
        name=name,
        kind=kind,
        phenomenon=phenomenon,
        assumptions=assumptions,
        hard_backup=backup_raw,
    )


def ledger_from_dicts(
    payloads: Sequence[Mapping[str, object]],
) -> Mapping[str, MetricSpec]:
    """Build a ledger from untrusted spec mappings, fail-closed.

    Duplicate names are refused. After parsing, every SOFT spec's backup
    must resolve to a HARD metric in the same ledger — a soft metric
    whose backup is missing or soft is refused at build time, not just
    flagged at audit.
    """
    specs = [spec_from_dict(payload) for payload in payloads]
    ledger: dict[str, MetricSpec] = {}
    for spec in specs:
        if spec.name in ledger:
            raise ValueError(f"duplicate metric {spec.name!r} in ledger")
        ledger[spec.name] = spec
    for spec in specs:
        if spec.kind is MetricKind.SOFT:
            backup = spec.hard_backup
            assert backup is not None  # enforced by MetricSpec
            if backup not in ledger:
                raise ValueError(
                    f"soft metric {spec.name!r}: hard backup {backup!r} "
                    "is not in the ledger"
                )
            if ledger[backup].kind is not MetricKind.HARD:
                raise ValueError(
                    f"soft metric {spec.name!r}: hard backup {backup!r} "
                    "is not HARD"
                )
    return MappingProxyType(ledger)
