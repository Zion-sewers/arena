"""Measurement validity: every metric carries a reliability column.

"Test the test" (arena#43). An eval score is not evidence until the
measurement instrument behind it has been assessed. This module attaches
a validity assessment to the metrics this repo reports: the four decision
dimensions (quality, safety, cost, reliability) and the deterministic
scorer's four matchers (exact_match, contains, regex, json_schema).
Other numeric reports in the repo (tiered mutation scores, regression
aggregates, alt-test alignment scores, novelty-oracle scores) carry no
validity assessment yet — see docs/VALIDITY.md for the coverage boundary.

* every metric names the phenomenon it claims to measure — a metric that
  measures silence after the word, not whether the consonant clipped,
  is a proxy that misses the phenomenon, and is worse than no metric;
* a proxy is validated against the real phenomenon with paired verdicts
  (proxy vs ground truth) and a Wilson 95% interval on the agreement —
  the compared quantity is the interval LOWER bound, never the point
  estimate, the same discipline as decision.py;
* a metric without a validity note is UNASSESSED and is NOT trusted:
  ``trusted`` is False, and the report names it in ``untrusted_metrics``.

Grades (closed set):
* DIRECT — the metric measures the phenomenon itself (e.g. exact_match
  measures string equality; a pass proportion measures the share of
  this suite's cases passed).
* VALIDATED_PROXY — a proxy whose agreement with ground truth was
  measured and whose Wilson lower bound clears the policy threshold.
* UNVALIDATED_PROXY — a proxy with no validation run, or one whose
  measured agreement failed the threshold. Never trusted.
* UNASSESSED — no validity note at all. Never trusted.

The built-in notes are grounded in the repo's own docs, not invented:
EVAL-RIGOR.md section 4 documents the deterministic scorer's blind
spots (``contains`` cannot see negation; ``json_schema: required``
ignores extra keys); decision.py documents the cost honesty limits
(normal approximation, no t correction; a single observation carries no
measured uncertainty); engine.py documents the replay-agreement caveat
(recorded outputs re-scored cannot detect live-agent flakiness).

Everything is stdlib-only and deterministic; there is no model call here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal

from agenteval_bench.decision import (
    CostEvidence,
    Decision,
    Dimension,
    ProportionEvidence,
    UnmeasuredEvidence,
    wilson_interval,
)
from agenteval_bench.models import EvalCase, ExpectedOutput
from agenteval_bench.scoring import DeterministicScorer

DEFAULT_MIN_AGREEMENT_LOWER = 0.80

#: The deterministic scorer's matchers, in the order the scorer applies them.
SCORER_METRICS: tuple[str, ...] = ("exact_match", "contains", "regex", "json_schema")


class ValidityGrade(str, Enum):
    """The closed set of validity grades for a metric."""

    DIRECT = "direct"
    VALIDATED_PROXY = "validated_proxy"
    UNVALIDATED_PROXY = "unvalidated_proxy"
    UNASSESSED = "unassessed"


@dataclass(frozen=True)
class ValidityPolicy:
    """Policy for proxy validation: the agreement bar a proxy must clear.

    ``min_agreement_lower`` is the minimum acceptable Wilson LOWER bound
    on proxy-vs-truth agreement, not the point estimate: a 5/5 agreement
    is a point estimate of 1.0 with a lower bound of 0.565, and does not
    clear the default 0.80 bar. The default is a policy choice, documented
    as such — a team owns its threshold and records it alongside the
    report, the same way DecisionPolicy owns decision thresholds.
    """

    min_agreement_lower: float = DEFAULT_MIN_AGREEMENT_LOWER

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_agreement_lower):
            raise ValueError(
                f"min_agreement_lower {self.min_agreement_lower} must be finite"
            )
        if not 0.0 < self.min_agreement_lower <= 1.0:
            raise ValueError(
                f"min_agreement_lower {self.min_agreement_lower} must be in (0, 1]"
            )


def default_validity_policy() -> ValidityPolicy:
    """A ValidityPolicy built from the defaults."""
    return ValidityPolicy()


@dataclass(frozen=True)
class MetricValidity:
    """The validity assessment of one metric.

    ``agreement`` is the Wilson 95% interval on proxy-vs-truth agreement
    from :func:`validate_proxy`, when a validation was run. The grade and
    the interval must agree with each other: DIRECT carries no agreement
    (a direct measurement has no proxy to validate), VALIDATED_PROXY
    requires one (the validation that earned the grade), UNASSESSED
    carries none (nothing was measured). UNVALIDATED_PROXY may carry one
    — a failed validation is still a measurement, and the interval is the
    evidence the proxy fell short.
    """

    metric: str
    grade: ValidityGrade
    phenomenon: str
    basis: str
    agreement: tuple[float, float] | None = None
    agreement_n: int | None = None

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("metric validity needs a non-empty metric name")
        if not self.phenomenon:
            raise ValueError(f"metric {self.metric!r} needs a named phenomenon")
        if not self.basis:
            raise ValueError(f"metric {self.metric!r} needs a basis for its grade")
        if self.grade is ValidityGrade.DIRECT and self.agreement is not None:
            raise ValueError(
                f"metric {self.metric!r}: a DIRECT measurement carries no "
                "proxy agreement interval"
            )
        if self.grade is ValidityGrade.VALIDATED_PROXY and self.agreement is None:
            raise ValueError(
                f"metric {self.metric!r}: VALIDATED_PROXY needs the agreement "
                "interval from validate_proxy"
            )
        if self.grade is ValidityGrade.UNASSESSED and self.agreement is not None:
            raise ValueError(
                f"metric {self.metric!r}: UNASSESSED carries no agreement interval"
            )
        if (self.agreement is None) != (self.agreement_n is None):
            raise ValueError(
                f"metric {self.metric!r}: agreement and agreement_n go together"
            )
        if self.agreement is not None:
            lower, upper = self.agreement
            if not 0.0 <= lower <= upper <= 1.0:
                raise ValueError(
                    f"metric {self.metric!r}: agreement interval {self.agreement} "
                    "must lie in [0, 1]"
                )
            if self.agreement_n is not None and self.agreement_n < 1:
                raise ValueError(
                    f"metric {self.metric!r}: agreement_n must be >= 1"
                )

    @property
    def trusted(self) -> bool:
        """Whether this metric may be trusted in a report.

        Only DIRECT and VALIDATED_PROXY are trusted. A proxy with no
        validation, a proxy whose validation failed, and a metric with no
        validity note are never trusted — the report must say so, not
        silently present their numbers.
        """
        return self.grade in (ValidityGrade.DIRECT, ValidityGrade.VALIDATED_PROXY)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form for reports."""
        return {
            "grade": self.grade.value,
            "phenomenon": self.phenomenon,
            "basis": self.basis,
            "agreement_lower": self.agreement[0] if self.agreement else None,
            "agreement_upper": self.agreement[1] if self.agreement else None,
            "agreement_n": self.agreement_n,
            "trusted": self.trusted,
        }


def validate_proxy(
    *,
    metric: str,
    phenomenon: str,
    proxy: Sequence[bool],
    truth: Sequence[bool],
    policy: ValidityPolicy | None = None,
) -> MetricValidity:
    """Validate a proxy metric against the real phenomenon, executably.

    Takes paired per-case verdicts — what the proxy said, what ground
    truth says — and returns the metric's validity assessment. The
    compared quantity is the Wilson 95% LOWER bound on agreement, never
    the point estimate: a proxy that agrees 5/5 has a point estimate of
    1.0 and a lower bound of 0.565, and does not clear the bar.

    The issue's own example: a metric that measures silence after the
    word, not whether the consonant clipped, agrees with ground truth
    only when silence happens to coincide with clipping — the paired
    verdicts expose that, and the grade stays UNVALIDATED_PROXY.

    Raises ValueError on mismatched lengths or an empty sample (an
    agreement over no evidence is not an agreement), TypeError on
    non-bool verdicts (0/1 ints are not verdicts; the boundary refuses
    them rather than silently coercing).
    """
    policy = policy or default_validity_policy()
    if len(proxy) != len(truth):
        raise ValueError(
            f"validate_proxy needs paired verdicts for {metric!r}: "
            f"got {len(proxy)} proxy verdicts and {len(truth)} truth verdicts"
        )
    if not proxy:
        raise ValueError(f"validate_proxy needs at least one paired case for {metric!r}")
    for verdicts, name in ((proxy, "proxy"), (truth, "truth")):
        for verdict in verdicts:
            if not isinstance(verdict, bool):
                raise TypeError(
                    f"validate_proxy needs bool verdicts for {metric!r}: "
                    f"{name} verdict {verdict!r} is not a bool"
                )
    agreeing = sum(1 for p, t in zip(proxy, truth, strict=True) if p == t)
    n = len(proxy)
    lower, upper = wilson_interval(agreeing, n)
    basis = (
        f"proxy vs ground truth on {n} paired cases; "
        f"agreement {agreeing}/{n} = {agreeing / n:.3f}, "
        f"95% CI=[{lower:.3f}, {upper:.3f}]"
    )
    if lower >= policy.min_agreement_lower:
        return MetricValidity(
            metric=metric,
            grade=ValidityGrade.VALIDATED_PROXY,
            phenomenon=phenomenon,
            basis=basis,
            agreement=(lower, upper),
            agreement_n=n,
        )
    return MetricValidity(
        metric=metric,
        grade=ValidityGrade.UNVALIDATED_PROXY,
        phenomenon=phenomenon,
        basis=(
            basis
            + f"; lower bound {lower:.3f} below policy bar "
            f"{policy.min_agreement_lower:.3f}"
        ),
        agreement=(lower, upper),
        agreement_n=n,
    )


def _parse_label_expected(expected: object, case_id: str) -> ExpectedOutput:
    """One labeled case's expected block, validated at the YAML boundary.

    Same matcher fields as EvalSuite.from_yaml; unknown keys, a
    non-list ``contains``, or a non-string needle raise. The expected
    block is the proxy's input, never the truth: truth comes from the
    case's ``truth`` label.
    """
    if not isinstance(expected, dict):
        raise TypeError(
            f"labels case {case_id!r}: expected must be a mapping, "
            f"got {type(expected).__name__}"
        )
    unknown = sorted(set(expected) - {"exact", "contains", "regex", "json_schema"})
    if unknown:
        raise ValueError(f"labels case {case_id!r}: unknown expected keys {unknown}")
    contains = expected.get("contains", [])
    if not isinstance(contains, list) or not all(
        isinstance(needle, str) for needle in contains
    ):
        raise TypeError(
            f"labels case {case_id!r}: contains must be a list of strings"
        )
    exact = expected.get("exact")
    if exact is not None and not isinstance(exact, str):
        raise TypeError(f"labels case {case_id!r}: exact must be a string")
    regex = expected.get("regex")
    if regex is not None and not isinstance(regex, str):
        raise TypeError(f"labels case {case_id!r}: regex must be a string")
    json_schema = expected.get("json_schema")
    if json_schema is not None and not isinstance(json_schema, dict):
        raise TypeError(f"labels case {case_id!r}: json_schema must be a mapping")
    return ExpectedOutput(
        exact=exact,
        contains=contains,
        regex=regex,
        json_schema=json_schema,
    )


def validate_matcher_labels(
    path: str, *, policy: ValidityPolicy | None = None
) -> MetricValidity:
    """Validate one matcher against ground truth from a labels file.

    The file names one matcher and its phenomenon, plus labeled cases:
    each case carries a recorded ``output``, the ``expected`` block the
    scorer runs, a human ``truth`` label (does the output state the
    phenomenon?), and a ``reason`` for the label. The proxy verdicts come
    from running the real DeterministicScorer on the recorded outputs —
    never from hand-written bools — and the agreement is graded by
    :func:`validate_proxy`. A labels file is ground truth the way it
    always is: human judgments, inspectable and disputable, recorded as
    data; see examples/contains-truth-labels.yaml.

    Raises OSError when the file cannot be read, ValueError/TypeError on
    a malformed file (unknown metric or keys, duplicate ids, non-bool
    truth, empty reason, empty case list).
    """
    import yaml

    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"labels file {path!r}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(
            f"labels file {path!r}: expected a mapping, got {type(data).__name__}"
        )
    unknown = sorted(set(data) - {"metric", "phenomenon", "cases"})
    if unknown:
        raise ValueError(f"labels file {path!r}: unknown top-level keys {unknown}")
    metric = data.get("metric")
    if not isinstance(metric, str) or metric not in SCORER_METRICS:
        raise ValueError(
            f"labels file {path!r}: unknown metric {metric!r}, expected one of "
            f"{sorted(SCORER_METRICS)}"
        )
    phenomenon = data.get("phenomenon")
    if not isinstance(phenomenon, str) or not phenomenon:
        raise ValueError(f"labels file {path!r}: phenomenon must be a non-empty string")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"labels file {path!r}: cases must be a non-empty list")
    scorer = DeterministicScorer()
    proxy: list[bool] = []
    truth: list[bool] = []
    seen_ids: set[str] = set()
    for case_data in cases:
        if not isinstance(case_data, dict):
            raise TypeError(
                f"labels file {path!r}: each case must be a mapping, "
                f"got {type(case_data).__name__}"
            )
        unknown = sorted(
            set(case_data) - {"id", "output", "expected", "truth", "reason"}
        )
        if unknown:
            raise ValueError(
                f"labels file {path!r}: unknown case keys {unknown}"
            )
        case_id = case_data.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"labels file {path!r}: each case needs a non-empty id")
        if case_id in seen_ids:
            raise ValueError(f"labels file {path!r}: duplicate case id {case_id!r}")
        seen_ids.add(case_id)
        output = case_data.get("output")
        if not isinstance(output, str):
            raise TypeError(f"labels case {case_id!r}: output must be a string")
        case_truth = case_data.get("truth")
        if not isinstance(case_truth, bool):
            raise TypeError(
                f"labels case {case_id!r}: truth must be a bool, got {case_truth!r}"
            )
        reason = case_data.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(
                f"labels case {case_id!r}: reason must be a non-empty string"
            )
        expected = _parse_label_expected(case_data.get("expected", {}), case_id)
        case = EvalCase(id=case_id, input="", expected=expected)
        proxy.append(scorer.score(case, output).passed)
        truth.append(case_truth)
    return validate_proxy(
        metric=metric,
        phenomenon=phenomenon,
        proxy=proxy,
        truth=truth,
        policy=policy,
    )


@dataclass(frozen=True)
class ValidityReport:
    """A validity assessment for every metric in a report.

    ``expected_metrics`` is the closed set of metrics the report must
    cover: a report that omits a metric is not a validity report, it is
    a partial one, so construction raises instead of silently trusting
    the missing metric. The mapping is copied and frozen at construction,
    like Decision's verdicts: a frozen dataclass alone does not stop
    dict mutation.
    """

    assessments: Mapping[str, MetricValidity]
    expected_metrics: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.expected_metrics:
            raise ValueError("a validity report needs a non-empty expected metric set")
        missing = sorted(set(self.expected_metrics) - set(self.assessments))
        if missing:
            raise ValueError(f"validity report is missing metrics: {missing}")
        extra = sorted(set(self.assessments) - set(self.expected_metrics))
        if extra:
            raise ValueError(f"validity report has unexpected metrics: {extra}")
        for metric, validity in self.assessments.items():
            if validity.metric != metric:
                raise ValueError(
                    f"validity for {validity.metric!r} filed under {metric!r}"
                )
        object.__setattr__(
            self, "assessments", MappingProxyType(dict(self.assessments))
        )

    def trusted(self, metric: str) -> bool:
        """Whether one metric in this report may be trusted."""
        return self.assessments[metric].trusted

    def untrusted_metrics(self) -> list[str]:
        """Metrics in this report that must NOT be trusted, in metric order."""
        return sorted(m for m, v in self.assessments.items() if not v.trusted)

    @property
    def all_trusted(self) -> bool:
        """True only when every metric in the report is trusted."""
        return not self.untrusted_metrics()

    def summary(self) -> str:
        """Human-readable table with the measurement-reliability column."""
        lines = [
            (
                f"Validity: {len(self.assessments) - len(self.untrusted_metrics())}"
                f"/{len(self.assessments)} metrics trusted"
            )
        ]
        for metric in sorted(self.assessments):
            validity = self.assessments[metric]
            if validity.agreement is None:
                agreement = "agreement —"
            else:
                lower, upper = validity.agreement
                agreement = (
                    f"agreement 95% CI=[{lower:.3f}, {upper:.3f}] "
                    f"(n={validity.agreement_n})"
                )
            lines.append(
                f"  {metric:12s} {validity.grade.value:18s} {agreement:38s} "
                f"{'TRUSTED' if validity.trusted else 'UNTRUSTED'}"
            )
        if self.untrusted_metrics():
            lines.append(
                "  UNTRUSTED: "
                + ", ".join(self.untrusted_metrics())
                + " — do not trust these metrics"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form for reports."""
        return {
            "metrics": {m: v.to_dict() for m, v in sorted(self.assessments.items())},
            "all_trusted": self.all_trusted,
            "untrusted": self.untrusted_metrics(),
        }


#: Built-in validity notes for the deterministic scorer's matchers.
#: Grounded in EVAL-RIGOR.md section 4 (known blind spots), not invented.
SCORER_VALIDITY: Mapping[str, MetricValidity] = MappingProxyType(
    {
        "exact_match": MetricValidity(
            metric="exact_match",
            grade=ValidityGrade.DIRECT,
            phenomenon="output equals the expected string exactly",
            basis="string equality; no judgment involved",
        ),
        "contains": MetricValidity(
            metric="contains",
            grade=ValidityGrade.UNVALIDATED_PROXY,
            phenomenon="output states the required fact",
            basis=(
                "substring presence; blind to negation — 'not X' passes a "
                "contains check for X (EVAL-RIGOR.md section 4)"
            ),
        ),
        "regex": MetricValidity(
            metric="regex",
            grade=ValidityGrade.UNVALIDATED_PROXY,
            phenomenon="output matches the required shape",
            basis=(
                "pattern match, not semantic match; anchored patterns resist "
                "keyword stuffing (EVAL-RIGOR.md section 4)"
            ),
        ),
        "json_schema": MetricValidity(
            metric="json_schema",
            grade=ValidityGrade.UNVALIDATED_PROXY,
            phenomenon="output is valid JSON with the required keys",
            basis=(
                "required-keys only; ignores extra keys and value semantics "
                "(EVAL-RIGOR.md section 4)"
            ),
        ),
    }
)


def _require_bar_cleared(note: MetricValidity, policy: ValidityPolicy) -> None:
    """Refuse a caller-supplied VALIDATED_PROXY note that fails the bar.

    :func:`validate_proxy` enforces the bar when it mints the grade; a
    hand-built note bypasses that body, so the factories re-check here.
    A forged VALIDATED_PROXY (grade claimed, lower bound below the bar)
    raises instead of entering a report as trusted. Other grades need
    no check: DIRECT carries no interval, and UNVALIDATED_PROXY and
    UNASSESSED are never trusted.
    """
    if note.grade is not ValidityGrade.VALIDATED_PROXY:
        return
    if note.agreement is None:  # pragma: no cover - __post_init__ refuses this
        raise ValueError(
            f"metric {note.metric!r}: VALIDATED_PROXY needs an agreement interval"
        )
    if note.agreement[0] < policy.min_agreement_lower:
        raise ValueError(
            f"metric {note.metric!r}: agreement lower bound {note.agreement[0]:.3f} "
            f"below policy bar {policy.min_agreement_lower:.3f}"
        )


def assess_scorer(
    notes: Mapping[str, MetricValidity] | None = None,
    *,
    policy: ValidityPolicy | None = None,
) -> ValidityReport:
    """Validity report over the deterministic scorer's matchers.

    Starts from the built-in notes (SCORER_VALIDITY); caller-supplied
    notes override built-ins per metric — that is how a suite author
    upgrades a matcher with a :func:`validate_proxy` result. An unknown
    metric name in ``notes`` raises: a typo must fail loudly, not
    silently add a metric no report reads. A caller note that claims
    VALIDATED_PROXY is re-checked against the policy bar (see
    ``_require_bar_cleared``): the grade means the bar was cleared,
    whoever minted the note.
    """
    policy = policy or default_validity_policy()
    merged: dict[str, MetricValidity] = dict(SCORER_VALIDITY)
    for metric, note in (notes or {}).items():
        if metric not in SCORER_VALIDITY:
            raise ValueError(
                f"unknown scorer metric {metric!r}: expected one of "
                f"{sorted(SCORER_VALIDITY)}"
            )
        if note.metric != metric:
            raise ValueError(f"validity for {note.metric!r} filed under {metric!r}")
        _require_bar_cleared(note, policy)
        merged[metric] = note
    return ValidityReport(
        assessments=merged, expected_metrics=frozenset(SCORER_METRICS)
    )


def assess_decision(
    decision: Decision,
    *,
    reliability_kind: Literal["live", "replay"],
    notes: Mapping[str, MetricValidity] | None = None,
    policy: ValidityPolicy | None = None,
) -> ValidityReport:
    """Validity report over the four decision dimensions.

    Built-in notes per dimension, grounded in the repo's own docs:
    * quality / safety — DIRECT for the stated phenomenon, "share of
      this suite's cases passed". The proportion is exact, but the cases
      are only as good as their matchers: the basis names that, and
      ``assess_scorer`` carries the matcher grades.
    * reliability — replay is UNVALIDATED_PROXY: replay-agreement
      (recorded outputs re-scored) cannot detect live-agent flakiness
      (engine.py). Live is graded exactly as :func:`validate_proxy`
      grades a proxy: VALIDATED_PROXY iff the Wilson lower bound on the
      cross-run agreement clears the policy bar, else UNVALIDATED_PROXY
      with the interval carried as the evidence that it fell short.
      A live agreement of 30/60 is untrusted, not certified.
      Unmeasured (fewer than 2 runs) is UNASSESSED: decision.py refuses
      to compute it, and so does this report.
    * cost — measured over >= 2 observations is DIRECT ("mean of measured
      per-case costs"); a single observation is UNVALIDATED_PROXY (it
      carries no measured uncertainty — decision.py); unmeasured is
      UNASSESSED.

    ``reliability_kind`` is caller-declared ("live" | "replay"), never
    auto-detected: this function cannot see how the runs were produced,
    so it does not pretend to. Caller-supplied ``notes`` override
    built-ins per dimension; an unknown dimension raises, and a note
    claiming VALIDATED_PROXY is re-checked against the policy bar.
    """
    if reliability_kind not in ("live", "replay"):
        raise ValueError(
            f"reliability_kind must be 'live' or 'replay', got {reliability_kind!r}"
        )
    policy = policy or default_validity_policy()
    built: dict[str, MetricValidity] = {}
    for dimension in Dimension:
        verdict = decision.verdicts[dimension]
        evidence = verdict.evidence
        name = dimension.value
        if isinstance(evidence, UnmeasuredEvidence):
            built[name] = MetricValidity(
                metric=name,
                grade=ValidityGrade.UNASSESSED,
                phenomenon=_phenomenon_for(dimension),
                basis=f"unmeasured: {evidence.reason}",
            )
            continue
        if dimension in (Dimension.QUALITY, Dimension.SAFETY):
            built[name] = MetricValidity(
                metric=name,
                grade=ValidityGrade.DIRECT,
                phenomenon=f"share of this suite's {name} cases passed",
                basis=(
                    "deterministic scorer over hand-written expectations; "
                    "the proportion is exact, but the cases are only as "
                    "good as their matchers — see assess_scorer()"
                ),
            )
        elif dimension is Dimension.RELIABILITY:
            assert isinstance(evidence, ProportionEvidence)
            if reliability_kind == "replay":
                built[name] = MetricValidity(
                    metric=name,
                    grade=ValidityGrade.UNVALIDATED_PROXY,
                    phenomenon="live-agent run-to-run stability",
                    basis=(
                        "replay-agreement: recorded outputs re-scored across "
                        "runs; cannot detect live-agent flakiness (engine.py)"
                    ),
                )
            else:
                lower, _upper = evidence.interval
                if lower >= policy.min_agreement_lower:
                    grade = ValidityGrade.VALIDATED_PROXY
                    basis = (
                        "per-case pass/fail agreement across repeated runs "
                        "with genuinely different executions; Wilson lower "
                        f"bound {lower:.3f} clears the policy bar "
                        f"{policy.min_agreement_lower:.3f}"
                    )
                else:
                    grade = ValidityGrade.UNVALIDATED_PROXY
                    basis = (
                        "per-case pass/fail agreement across repeated runs "
                        "with genuinely different executions; Wilson lower "
                        f"bound {lower:.3f} below the policy bar "
                        f"{policy.min_agreement_lower:.3f}"
                    )
                built[name] = MetricValidity(
                    metric=name,
                    grade=grade,
                    phenomenon="live-agent run-to-run stability",
                    basis=basis,
                    agreement=evidence.interval,
                    agreement_n=evidence.trials,
                )
        else:  # Dimension.COST
            assert isinstance(evidence, CostEvidence)
            if evidence.n < 2:
                built[name] = MetricValidity(
                    metric=name,
                    grade=ValidityGrade.UNVALIDATED_PROXY,
                    phenomenon="mean cost per case",
                    basis=(
                        "single cost observation carries no measured "
                        "uncertainty (decision.py)"
                    ),
                )
            else:
                built[name] = MetricValidity(
                    metric=name,
                    grade=ValidityGrade.DIRECT,
                    phenomenon="mean cost per case",
                    basis=f"mean of {evidence.n} measured per-case costs",
                )
    for metric, note in (notes or {}).items():
        if metric not in built:
            raise ValueError(
                f"unknown decision dimension {metric!r}: expected one of "
                f"{sorted(built)}"
            )
        if note.metric != metric:
            raise ValueError(f"validity for {note.metric!r} filed under {metric!r}")
        _require_bar_cleared(note, policy)
        built[metric] = note
    return ValidityReport(
        assessments=built,
        expected_metrics=frozenset(d.value for d in Dimension),
    )


def _phenomenon_for(dimension: Dimension) -> str:
    """The phenomenon each decision dimension claims to measure."""
    if dimension is Dimension.QUALITY:
        return "share of this suite's quality cases passed"
    if dimension is Dimension.SAFETY:
        return "share of this suite's safety cases passed"
    if dimension is Dimension.RELIABILITY:
        return "live-agent run-to-run stability"
    return "mean cost per case"
