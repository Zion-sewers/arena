"""Metric-assumption ledger + hard-backup rule (arena#56).

Berman 'Financial Intelligence' (catalog G.11), INSPIRED: metrics reflect
assumptions, and every soft metric needs a hard verifying metric — the
"cash test". This module is the ledger for the metrics the repo's
reports print.

The hard/soft line
------------------
HARD: the number is a deterministic function (a count, a sum, or a
proportion) of recorded observations. Any judgment the count depends on
— a matcher verdict, a dimension tag, a ground-truth label — is named
in the metric's assumptions; the count itself adds no further
inference. SOFT: the number is produced by an inference beyond that
function — a judge's grade, a gap between a self-report and a
measurement, an assessment, or a proxy for a phenomenon. The Wilson
interval is the uncertainty carried WITH a hard proportion, not a
separate soft metric.

What "backed by" means, operationally
-------------------------------------
A soft metric's ``hard_backup`` names a HARD metric in the same ledger
that is independently measured from it — different observations, never
the soft number's own output. The check itself is the reader (or
tooling) comparing the two numbers: in ``calibrate`` the gap and the
agreement rate print side by side; the ledger guarantees the reference
resolves to a HARD entry, it does not perform the numeric comparison.
Whether the backup is independent is a property of the entries,
reviewed like any other content: ``label_agreement`` is counted
against ground-truth labels the scorer never sees, ``agreement_rate``
is counted from recorded judgments, not from the gap.

Two axes, not two verdicts
--------------------------
The ledger's HARD/SOFT says HOW a number is measured. ``validity.py``
grades say WHETHER to trust it in a particular report: replay
reliability is UNVALIDATED_PROXY, a single cost observation is
UNVALIDATED_PROXY. When the two lenses disagree, the validity grade
governs trust and the ledger governs what the number assumes. Both
tables print together in ``run`` and ``decide`` for this reason.

How the audit works
-------------------
The audit is fed the metrics a report actually prints, per command
(``REPORT_METRIC_SETS``), and flags any printed metric with no ledger
entry. It never audits the bare registry and calls the result clean:
``audit_report`` takes the report's metric names, ``audit_ledger``
audits a whole mapping. A soft metric without a resolvable HARD backup
is flagged; an assumption-free metric is refused at construction, at
parsing, and at audit.

Acceptance (arena#56), enforced here, not in prose:
- every reported metric lists its assumptions — ``MetricSpec`` refuses
  an empty or blank ``assumptions`` list at construction, and the
  audit prints the assumption texts (with falsifiers) in the report;
- a soft metric without hard backup is flagged — ``audit_report``
  flags it, and the fail-closed parsers refuse to build it;
- assumption-free metrics are NOT accepted — construction, parsing,
  and audit all refuse them (blank strings count as empty).

Honesty boundary: the ledger lists assumptions; it cannot verify they
are TRUE. Each assumption carries a ``falsifier`` — the observation
that would break it — so every assumption is checkable, the same rule
as the prespecification ``Assumption`` (src/experiment/prespec.py).
``MetricAssumption`` is a separate type from prespec's because a
metric assumption lives in the reporting layer and is rendered into
reports; prespec assumptions gate an analysis plan.

Scope: the ledger covers the metrics printed by ``run``, ``decide``,
``calibrate``, ``perturb``, and ``tiered``. NOT covered yet: alt-test
alignment scores and p-values, novelty-oracle ``novelty_score``, and
regression aggregate scores — those reports carry no ledger entry;
the "every metric" claim is scoped to the covered commands until they
are wired in. The judge-allocation command (``judges``) prints no
scored metric and is out of scope.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class MetricKind(str, Enum):
    """Whether a metric is directly measured or inferred."""

    HARD = "hard"
    SOFT = "soft"


def _stripped(value: str, field_name: str, metric: str) -> str:
    """One string field, stripped; blank counts as empty."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(
            f"metric {metric!r}: field {field_name!r} must be a "
            "non-empty, non-blank string"
        )
    return stripped


@dataclass(frozen=True)
class MetricAssumption:
    """One stated assumption behind a metric, with its falsifier.

    An assumption without a falsifier is a slogan, not a checkable
    claim: construction refuses empty or blank text or falsifier.
    Stored stripped, so a whitespace-only string can never pass.
    """

    text: str
    falsifier: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _stripped(self.text, "text", "?"))
        object.__setattr__(
            self, "falsifier", _stripped(self.falsifier, "falsifier", "?")
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
    carries no backup (it IS the checkable count). ``assumptions`` is
    copied to a tuple at construction, so a caller holding the source
    list cannot mutate a frozen spec assumption-free afterwards.
    """

    name: str
    kind: MetricKind
    phenomenon: str
    assumptions: tuple[MetricAssumption, ...]
    hard_backup: str | None = None

    def __post_init__(self) -> None:
        name = _stripped(self.name, "name", "?")
        object.__setattr__(self, "name", name)
        phenomenon = _stripped(self.phenomenon, "phenomenon", name)
        object.__setattr__(self, "phenomenon", phenomenon)
        if isinstance(self.assumptions, (str, bytes)) or not isinstance(
            self.assumptions, Sequence
        ):
            raise TypeError(
                f"metric {name!r}: field 'assumptions' must be a "
                "sequence of MetricAssumption"
            )
        assumptions = tuple(self.assumptions)
        for item in assumptions:
            if not isinstance(item, MetricAssumption):
                raise TypeError(
                    f"metric {name!r}: every assumption must be a "
                    "MetricAssumption"
                )
        if not assumptions:
            raise ValueError(
                f"metric {name!r} is assumption-free: assumption-free "
                "metrics are not accepted"
            )
        object.__setattr__(self, "assumptions", assumptions)
        backup = self.hard_backup
        if backup is not None:
            backup = _stripped(backup, "hard_backup", name)
            object.__setattr__(self, "hard_backup", backup)
        if self.kind is MetricKind.SOFT and not backup:
            raise ValueError(
                f"soft metric {name!r} needs a hard backup: name the "
                "HARD metric it is checkable against"
            )
        if self.kind is MetricKind.HARD and backup is not None:
            raise ValueError(
                f"hard metric {name!r} carries no backup: it IS the "
                "checkable count"
            )

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form for reports."""
        return {
            "name": self.name,
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
                    "with the proportion, not a separate soft metric",
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
                    "with the proportion, not a separate soft metric",
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
                    "HARD here means the number is a count of agreeing "
                    "runs; whether to trust it in a given report is the "
                    "validity grade's job — in replay the grade is "
                    "UNVALIDATED_PROXY",
                    "a replay-agreement number is quoted as evidence of "
                    "live-agent stability",
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
                _a(
                    "HARD here means the total is a sum of measured "
                    "costs; whether one observation is enough to trust "
                    "is the validity grade's job",
                    "a single-observation cost total is quoted as a "
                    "stable estimate",
                ),
            ),
        ),
        "agreement_rate": MetricSpec(
            name="agreement_rate",
            kind=MetricKind.HARD,
            phenomenon="measured fraction of correct judgments in the "
            "calibration observations",
            assumptions=(
                _a(
                    "Each observation is a recorded (confidence, "
                    "correct?) pair",
                    "agreement_rate is computed from unrecorded judgments",
                ),
                _a(
                    "The rate is a count over recorded observations, not "
                    "a model of the judge",
                    "the rate changes when the same observations are "
                    "re-counted",
                ),
            ),
        ),
        "label_agreement": MetricSpec(
            name="label_agreement",
            kind=MetricKind.HARD,
            phenomenon="measured proportion of proxy-vs-truth agreement "
            "on labeled cases",
            assumptions=(
                _a(
                    "The labels are ground truth, recorded before scoring",
                    "a label is changed after seeing the proxy's verdict",
                ),
                _a(
                    "Agreement is a count over labeled cases; the Wilson "
                    "interval is the uncertainty carried with it",
                    "agreement is quoted with no interval and no n",
                ),
            ),
        ),
        "exact_match": MetricSpec(
            name="exact_match",
            kind=MetricKind.HARD,
            phenomenon="output equals the expected string exactly",
            assumptions=(
                _a(
                    "String equality is the phenomenon; no judgment is "
                    "involved",
                    "two strings compare equal while differing",
                ),
                _a(
                    "The expected string is the recorded expectation for "
                    "the case",
                    "the expected string is edited after the run",
                ),
            ),
        ),
        "coverage_pct": MetricSpec(
            name="coverage_pct",
            kind=MetricKind.HARD,
            phenomenon="fraction of the repo's lines covered by the "
            "tiered suite",
            assumptions=(
                _a(
                    "Coverage is measured by running the suite, not "
                    "estimated",
                    "coverage changes without the suite running",
                ),
                _a(
                    "The denominator is the repo tree the tiered phase "
                    "ran on",
                    "coverage is quoted for a different tree than the one "
                    "measured",
                ),
            ),
        ),
        "mutation_score": MetricSpec(
            name="mutation_score",
            kind=MetricKind.HARD,
            phenomenon="fraction of seeded mutants the suite kills",
            assumptions=(
                _a(
                    "Each mutant is a real seeded fault in the target repo",
                    "a mutant that changes nothing counts as killed or "
                    "survived",
                ),
                _a(
                    "The score is a count over mutants run in this phase",
                    "the score includes mutants never executed",
                ),
            ),
        ),
        "calibration_gap": MetricSpec(
            name="calibration_gap",
            kind=MetricKind.SOFT,
            phenomenon="mean judge confidence minus measured agreement "
            "rate (ECE is the bin-weighted aggregate of the same gap)",
            assumptions=(
                _a(
                    "Confidence is self-reported by the judge, not "
                    "measured",
                    "a confidence value is quoted as a measured accuracy",
                ),
                _a(
                    "The gap is computed from the same recorded "
                    "observations as the agreement rate",
                    "gap and agreement_rate are computed from different "
                    "observation sets",
                ),
            ),
            hard_backup="agreement_rate",
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
            hard_backup="label_agreement",
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
                _a(
                    "The drops are differences of the printed per-regime "
                    "pass rates",
                    "a drop is quoted that does not equal the difference "
                    "of the printed regime rates",
                ),
            ),
            hard_backup="pass_rate",
        ),
        "contains": MetricSpec(
            name="contains",
            kind=MetricKind.SOFT,
            phenomenon="output states the required fact (substring "
            "presence)",
            assumptions=(
                _a(
                    "Substring presence is a proxy for stating the fact, "
                    "blind to negation",
                    "'not X' passes a contains check for X",
                ),
                _a(
                    "The proxy is trusted only when its label agreement "
                    "clears the bar",
                    "contains is quoted as trusted with no validation run",
                ),
            ),
            hard_backup="label_agreement",
        ),
        "regex": MetricSpec(
            name="regex",
            kind=MetricKind.SOFT,
            phenomenon="output matches the required shape",
            assumptions=(
                _a(
                    "A pattern match is not a semantic match",
                    "an output matches the pattern while failing the "
                    "requirement",
                ),
                _a(
                    "The proxy is trusted only when its label agreement "
                    "clears the bar",
                    "regex is quoted as trusted with no validation run",
                ),
            ),
            hard_backup="label_agreement",
        ),
        "json_schema": MetricSpec(
            name="json_schema",
            kind=MetricKind.SOFT,
            phenomenon="output is valid JSON with the required keys",
            assumptions=(
                _a(
                    "Required-keys only; ignores extra keys and value "
                    "semantics",
                    "an output passes the schema while its values are "
                    "wrong",
                ),
                _a(
                    "The proxy is trusted only when its label agreement "
                    "clears the bar",
                    "json_schema is quoted as trusted with no validation "
                    "run",
                ),
            ),
            hard_backup="label_agreement",
        ),
    }
)


#: The metrics each report command prints, as a closed expected set.
#: The CLI audits exactly this set per command; a printed metric with
#: no ledger entry is flagged, never silently clean. Adding a metric
#: to a report means adding it here AND to METRIC_LEDGER first — the
#: sync test (tests/test_metric_ledger.py) fails otherwise.
REPORT_METRIC_SETS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "run": frozenset(
            {
                "pass_rate",
                "exact_match",
                "contains",
                "regex",
                "json_schema",
                "scorer_validity",
                # The validity table's agreement column: populated when
                # --validity-labels is given, empty otherwise.
                "label_agreement",
            }
        ),
        "decide": frozenset(
            {
                "quality",
                "safety",
                "reliability",
                "cost",
                "exact_match",
                "contains",
                "regex",
                "json_schema",
                "scorer_validity",
                # The validity table's agreement column, as in run.
                "label_agreement",
            }
        ),
        "calibrate": frozenset({"agreement_rate", "calibration_gap"}),
        "perturb": frozenset({"pass_rate", "perturbation_stability"}),
        "tiered": frozenset({"coverage_pct", "mutation_score"}),
    }
)


class LedgerStatus(str, Enum):
    """The audit verdict for one metric in a ledger."""

    OK = "ok"
    FLAGGED_NO_ASSUMPTIONS = "flagged_no_assumptions"
    FLAGGED_NO_BACKUP = "flagged_no_backup"
    FLAGGED_BAD_BACKUP = "flagged_bad_backup"
    FLAGGED_NOT_IN_LEDGER = "flagged_not_in_ledger"
    FLAGGED_MISFILED = "flagged_misfiled"


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
    """The audit of a ledger (or of one report's metric set).

    ``findings`` and ``ledger`` are copied and frozen at construction,
    like Decision's verdicts: a frozen dataclass alone does not stop
    dict mutation. The audit renders from the ledger it audited, never
    from module global state. An audit with no findings audited
    nothing and is not clean.
    """

    findings: tuple[LedgerFinding, ...] = field(default_factory=tuple)
    ledger: Mapping[str, MetricSpec] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "ledger", MappingProxyType(dict(self.ledger)))

    @property
    def flagged(self) -> list[str]:
        """Metrics the audit flagged, in metric order."""
        return sorted(
            f.metric for f in self.findings if f.status is not LedgerStatus.OK
        )

    @property
    def all_ok(self) -> bool:
        """True only when at least one metric was audited and none flagged."""
        return bool(self.findings) and not self.flagged

    def summary(self, report: str | None = None) -> str:
        """Human-readable listing: every metric with its kind, backup,
        and assumption texts with falsifiers. A flagged metric is
        named with its remediation, never hidden."""
        title = "Metric ledger" + (f" ({report})" if report else "")
        lines = [
            (
                f"{title}: {len(self.findings) - len(self.flagged)}"
                f"/{len(self.findings)} metrics clean"
            )
        ]
        if not self.findings:
            lines.append("  no metrics audited — nothing was checked")
            return "\n".join(lines)
        for finding in sorted(self.findings, key=lambda f: f.metric):
            spec = self.ledger.get(finding.metric)
            if spec is None:
                lines.append(
                    f"  {finding.metric:22s} NOT IN LEDGER — {finding.detail}"
                )
                continue
            backup = spec.hard_backup or "—"
            verdict = (
                "OK"
                if finding.status is LedgerStatus.OK
                else "FLAGGED: " + finding.detail
            )
            lines.append(
                f"  {finding.metric:22s} {spec.kind.value:4s} "
                f"backup={backup:15s} {verdict}"
            )
            for assumption in spec.assumptions:
                lines.append(
                    f"    - {assumption.text} "
                    f"(breaks if: {assumption.falsifier})"
                )
        if self.flagged:
            lines.append(
                "  FLAGGED: " + ", ".join(self.flagged) + " — do not quote "
                "or gate on flagged metrics until each names a hard backup"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form for reports: findings plus the full
        audited specs, so assumption texts travel with the report."""
        return {
            "metrics": {
                name: spec.to_dict()
                for name, spec in sorted(self.ledger.items())
                if any(f.metric == name for f in self.findings)
            },
            "findings": [f.to_dict() for f in self.findings],
            "all_ok": self.all_ok,
            "flagged": self.flagged,
        }


def _audit_entry(
    name: str, spec: MetricSpec, ledger: Mapping[str, MetricSpec]
) -> LedgerFinding:
    """The audit verdict for one entry of a ledger."""
    if spec.name != name:
        return LedgerFinding(
            metric=name,
            status=LedgerStatus.FLAGGED_MISFILED,
            detail=f"spec filed under {name!r} is named {spec.name!r}",
        )
    if not spec.assumptions:
        return LedgerFinding(
            metric=name,
            status=LedgerStatus.FLAGGED_NO_ASSUMPTIONS,
            detail="assumption-free metrics are not accepted",
        )
    if spec.kind is MetricKind.SOFT:
        backup = spec.hard_backup
        if not backup:
            return LedgerFinding(
                metric=name,
                status=LedgerStatus.FLAGGED_NO_BACKUP,
                detail="soft metric without hard backup",
            )
        if backup not in ledger:
            return LedgerFinding(
                metric=name,
                status=LedgerStatus.FLAGGED_BAD_BACKUP,
                detail=f"hard backup {backup!r} is not in the ledger",
            )
        if ledger[backup].kind is not MetricKind.HARD:
            return LedgerFinding(
                metric=name,
                status=LedgerStatus.FLAGGED_BAD_BACKUP,
                detail=f"hard backup {backup!r} is not HARD",
            )
        return LedgerFinding(
            metric=name,
            status=LedgerStatus.OK,
            detail=f"backed by hard metric {backup!r}",
        )
    if spec.hard_backup is not None:
        return LedgerFinding(
            metric=name,
            status=LedgerStatus.FLAGGED_BAD_BACKUP,
            detail=f"hard metric carries a backup {spec.hard_backup!r}",
        )
    return LedgerFinding(
        metric=name,
        status=LedgerStatus.OK,
        detail="hard metric: directly measured",
    )


def audit_ledger(
    ledger: Mapping[str, MetricSpec] = METRIC_LEDGER,
) -> LedgerAudit:
    """Audit a whole ledger: every metric lists assumptions; every soft
    metric names a hard backup that exists in the ledger and IS hard.

    Construction already refuses assumption-free specs and soft specs
    without a backup, so the audit re-verifies defensively: a hand-built
    mapping cannot slip a bad entry past it. A value that is not a
    MetricSpec raises TypeError — the audit fails closed, never
    silently passes a malformed entry.
    """
    findings: list[LedgerFinding] = []
    for name in sorted(ledger):
        spec = ledger[name]
        if not isinstance(spec, MetricSpec):
            raise TypeError(
                f"ledger entry {name!r} is not a MetricSpec: "
                f"got {type(spec).__name__}"
            )
        findings.append(_audit_entry(name, spec, ledger))
    return LedgerAudit(findings=tuple(findings), ledger=ledger)


def audit_report(
    metric_names: Iterable[str],
    ledger: Mapping[str, MetricSpec] = METRIC_LEDGER,
) -> LedgerAudit:
    """Audit exactly the metrics one report prints.

    This is the enforcement point for "every reported metric lists its
    assumptions": a printed metric with no ledger entry is flagged
    FLAGGED_NOT_IN_LEDGER, never silently clean. The CLI passes
    ``REPORT_METRIC_SETS[command]``; a report that grows a metric
    without a ledger entry fails its audit.
    """
    findings: list[LedgerFinding] = []
    for name in sorted(set(metric_names)):
        spec = ledger.get(name)
        if spec is None:
            findings.append(
                LedgerFinding(
                    metric=name,
                    status=LedgerStatus.FLAGGED_NOT_IN_LEDGER,
                    detail="reported metric has no ledger entry",
                )
            )
        else:
            findings.append(_audit_entry(name, spec, ledger))
    return LedgerAudit(findings=tuple(findings), ledger=ledger)


def require_ledger_ok(audit: LedgerAudit) -> None:
    """Raise on a flagged (or empty) ledger audit, naming the flagged
    metrics. Callers that gate on the ledger use this; report printers
    use :meth:`LedgerAudit.summary` and let the reader see the flag."""
    if not audit.all_ok:
        if not audit.findings:
            raise ValueError("metric ledger audit checked no metrics")
        raise ValueError(
            "metric ledger has flagged metrics: " + ", ".join(audit.flagged)
        )


def _require_str(value: object, field_name: str, metric: str) -> str:
    """One non-empty, non-blank string field from an untrusted mapping."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"metric {metric!r}: field {field_name!r} must be a non-empty, "
            "non-blank string"
        )
    return value.strip()


def _as_assumptions(value: object, metric: str) -> tuple[MetricAssumption, ...]:
    """Parse the assumptions list of one metric spec, fail-closed."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            f"metric {metric!r}: field 'assumptions' must be a list of "
            "mappings"
        )
    if not value:
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
    with one, a non-string field, and unknown keys are all refused
    here — the same rules :class:`MetricSpec` enforces at construction.
    Wrong-typed inputs raise TypeError; invalid values raise ValueError.
    """
    if not isinstance(payload, dict):
        raise TypeError("a metric spec must be a mapping")
    known = {"name", "kind", "phenomenon", "assumptions", "hard_backup"}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"metric spec has unknown keys: {unknown}")
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
        not isinstance(backup_raw, str) or not backup_raw.strip()
    ):
        raise ValueError(
            f"metric {name!r}: field 'hard_backup' must be a non-empty, "
            "non-blank string or absent"
        )
    backup = backup_raw.strip() if isinstance(backup_raw, str) else None
    return MetricSpec(
        name=name,
        kind=kind,
        phenomenon=phenomenon,
        assumptions=assumptions,
        hard_backup=backup,
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
            if backup is None:
                raise ValueError(
                    f"soft metric {spec.name!r}: hard backup is missing"
                )
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
