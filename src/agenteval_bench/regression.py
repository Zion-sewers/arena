"""Regression workflow: production failures -> frozen benchmark cases -> gated release.

The BetterBench maintenance phase made executable:

1. **Promote** a production failure (retry trace, correction pattern, failed
   tool call, escalation) into a replayable benchmark case with workflow
   metadata (reference answer, prior score, evaluator notes). The promoted
   case is published as a *new* snapshot version — the old version stays
   frozen.
2. **Replay** a candidate agent/config against the frozen set and compare
   aggregate scores against the version's pinned baseline. Releases gate on
   this comparison, not on spot checks.

The gate fails closed: any per-case score regression or an aggregate
pass-rate below the baseline fails the release, with the regressed cases
named in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from agenteval_bench.engine import AgentFn, EvalRunner
from agenteval_bench.models import EvalCase, EvalSuite, ExpectedOutput, RubricCriterion
from agenteval_bench.snapshots import (
    BaselineRecord,
    BenchmarkVersion,
    SnapshotStore,
    VersionMeta,
)

FailureKind = Literal["retry_trace", "correction_pattern", "failed_tool_call", "escalation"]

_FAILURE_KINDS: tuple[str, ...] = (
    "retry_trace",
    "correction_pattern",
    "failed_tool_call",
    "escalation",
)

GateVerdict = Literal["GATE_PASS", "GATE_FAIL"]


@dataclass(frozen=True)
class ProductionFailure:
    """One production failure, captured for promotion into the benchmark."""

    failure_id: str
    kind: FailureKind
    trace_summary: str
    reference_answer: str
    prior_score: float
    evaluator_notes: str = ""

    def __post_init__(self) -> None:
        if not self.failure_id:
            raise ValueError("failure_id must be non-empty")
        if self.kind not in _FAILURE_KINDS:
            raise ValueError(
                f"kind must be one of {_FAILURE_KINDS}, got {self.kind!r}"
            )
        if not self.trace_summary:
            raise ValueError("trace_summary must be non-empty")
        if not self.reference_answer:
            raise ValueError("reference_answer must be non-empty")
        if not 0.0 <= self.prior_score <= 1.0:
            raise ValueError(f"prior_score must be in [0, 1], got {self.prior_score}")


@dataclass(frozen=True)
class PromotionRecord:
    """Audit record of one promoted failure, stored in the new version's manifest.

    ``provenance`` is the trace-mining link (arena#41): trace id, incident
    id, detected pattern, evidence, owner, cadence. It is optional and
    omitted from :meth:`to_dict` when absent, so existing records are
    byte-identical to before.
    """

    failure_id: str
    kind: FailureKind
    promoted_case_id: str
    prior_score: float
    evaluator_notes: str
    promoted_at: str
    provenance: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.kind not in _FAILURE_KINDS:
            raise ValueError(
                f"kind must be one of {_FAILURE_KINDS}, got {self.kind!r}"
            )
        if self.provenance is not None and not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in self.provenance.items()
        ):
            raise ValueError("provenance must map strings to strings")

    def to_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "failure_id": self.failure_id,
            "kind": self.kind,
            "promoted_case_id": self.promoted_case_id,
            "prior_score": self.prior_score,
            "evaluator_notes": self.evaluator_notes,
            "promoted_at": self.promoted_at,
        }
        if self.provenance is not None:
            doc["provenance"] = dict(self.provenance)
        return doc

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PromotionRecord:
        if not isinstance(d, dict):
            raise TypeError(f"promotion record must be a mapping, got {type(d).__name__}")

        def req(key: str) -> Any:
            if key not in d:
                raise ValueError(f"promotion record missing {key!r}")
            return d[key]

        kind = str(req("kind"))
        if kind not in _FAILURE_KINDS:
            raise ValueError(
                f"promotion record kind must be one of {_FAILURE_KINDS}, got {kind!r}"
            )
        provenance_raw = d.get("provenance")
        provenance: dict[str, str] | None = None
        if provenance_raw is not None:
            if not isinstance(provenance_raw, dict) or not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in provenance_raw.items()
            ):
                raise ValueError("promotion record provenance must map strings to strings")
            provenance = dict(provenance_raw)
        return cls(
            failure_id=str(req("failure_id")),
            kind=cast(FailureKind, kind),
            promoted_case_id=str(req("promoted_case_id")),
            prior_score=float(req("prior_score")),
            evaluator_notes=str(d.get("evaluator_notes", "")),
            promoted_at=str(d.get("promoted_at", "")),
            provenance=provenance,
        )


def promote_failure(
    failure: ProductionFailure, *, promoted_at: str | None = None
) -> tuple[EvalCase, PromotionRecord]:
    """Turn a production failure into a replayable benchmark case.

    The case id is namespaced (``prod:<kind>:<failure_id>``) so promoted
    cases never collide with designed cases. The reference answer becomes
    the exact-match expectation; the evaluator's notes ride along as a
    rubric criterion description and in the promotion record.
    """
    case_id = f"prod:{failure.kind}:{failure.failure_id}"
    case = EvalCase(
        id=case_id,
        input=failure.trace_summary,
        expected=ExpectedOutput(exact=failure.reference_answer),
        rubric=[
            RubricCriterion(
                criterion="evaluator-acceptance",
                description=failure.evaluator_notes or "promoted production failure",
            )
        ],
    )
    record = PromotionRecord(
        failure_id=failure.failure_id,
        kind=failure.kind,
        promoted_case_id=case_id,
        prior_score=failure.prior_score,
        evaluator_notes=failure.evaluator_notes,
        promoted_at=promoted_at or datetime.now(UTC).isoformat(),
    )
    return case, record


@dataclass(frozen=True)
class CaseDelta:
    case_id: str
    baseline_score: float
    candidate_score: float


@dataclass(frozen=True)
class ComparisonReport:
    """Outcome of replaying a candidate against a frozen snapshot version."""

    suite_name: str
    version_id: str
    digest: str
    candidate_agent: str
    seed: int
    baseline_pass_rate: float
    candidate_pass_rate: float
    case_deltas: tuple[CaseDelta, ...]
    regressed_cases: tuple[str, ...]
    verdict: GateVerdict
    reasons: tuple[str, ...]


def compare_against_frozen(
    store: SnapshotStore,
    suite_name: str,
    version_id: str,
    agent_fn: AgentFn,
    *,
    seed: int | None = None,
    min_pass_rate: float | None = None,
    candidate_agent: str = "candidate",
) -> ComparisonReport:
    """Replay ``agent_fn`` against a frozen snapshot and gate on the comparison.

    The frozen version is hash-verified on load — a mutated snapshot fails
    closed before any scoring happens. ``seed`` defaults to the version's
    pinned baseline seed so the replay runs under the exact RNG stream the
    baseline was pinned with (apples-to-apples); pass an explicit seed to
    override. The gate fails if the candidate's aggregate pass-rate falls
    below ``min_pass_rate`` (default: the pinned baseline pass-rate — no
    aggregate regression tolerated) or if any individual case scores below
    its baseline (no per-case regression).
    """
    version, suite = store.load(suite_name, version_id)
    effective_seed = version.baseline.seed if seed is None else seed
    result = EvalRunner().run(suite, agent_fn, seed=effective_seed)

    baseline = version.baseline
    threshold = baseline.pass_rate if min_pass_rate is None else min_pass_rate

    scored = [r for r in result.results if not r.details.get("skipped")]
    by_id = {r.case_id: r.score for r in scored}
    deltas: list[CaseDelta] = []
    regressed: list[str] = []
    for case in suite.cases:
        if case.skip:
            continue
        base_score = baseline.per_case.get(case.id, 0.0)
        cand_score = by_id.get(case.id, 0.0)
        deltas.append(CaseDelta(case.id, base_score, cand_score))
        if cand_score < base_score:
            regressed.append(case.id)

    reasons: list[str] = []
    if result.pass_rate < threshold:
        reasons.append(
            f"aggregate pass_rate {result.pass_rate:.1%} below threshold {threshold:.1%}"
        )
    if regressed:
        reasons.append(f"per-case regressions: {', '.join(regressed)}")

    verdict: GateVerdict = "GATE_FAIL" if reasons else "GATE_PASS"
    return ComparisonReport(
        suite_name=suite_name,
        version_id=version_id,
        digest=version.digest,
        candidate_agent=candidate_agent,
        seed=effective_seed,
        baseline_pass_rate=baseline.pass_rate,
        candidate_pass_rate=result.pass_rate,
        case_deltas=tuple(deltas),
        regressed_cases=tuple(regressed),
        verdict=verdict,
        reasons=tuple(reasons),
    )


def promote_failures_to_version(
    store: SnapshotStore,
    suite: EvalSuite,
    version_id: str,
    meta: VersionMeta,
    failures: list[ProductionFailure],
    baseline: BaselineRecord,
    *,
    prev_version_id: str | None = None,
    promoted_at: str | None = None,
) -> tuple[BenchmarkVersion, EvalSuite]:
    """Promote production failures into a NEW snapshot version.

    The flagship BetterBench maintenance step as one call: each failure
    becomes a replayable case appended to a copy of ``suite`` (the caller's
    suite is not mutated), the new version is published with the promotion
    records in its manifest, and the previous version stays frozen.
    """
    new_cases: list[EvalCase] = []
    records: list[dict[str, Any]] = []
    for failure in failures:
        case, record = promote_failure(failure, promoted_at=promoted_at)
        new_cases.append(case)
        records.append(record.to_dict())
    new_suite = EvalSuite(
        name=suite.name,
        version=suite.version,
        cases=[*suite.cases, *new_cases],
        cost_bound=suite.cost_bound,
    )
    version = store.publish(
        new_suite,
        version_id,
        meta,
        baseline,
        prev_version_id=prev_version_id,
        promotions=records,
    )
    return version, new_suite


__all__ = [
    "CaseDelta",
    "ComparisonReport",
    "FailureKind",
    "GateVerdict",
    "ProductionFailure",
    "PromotionRecord",
    "compare_against_frozen",
    "promote_failure",
    "promote_failures_to_version",
]
