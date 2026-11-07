"""Analysis pre-specification protocol (agenteval-bench#48).

Source mechanism: Spiegelhalter, *The Art of Statistics* (catalog G.2) —
INSPIRED. Pre-specify analyses. Separate exploratory from confirmatory.
State assumptions, uncertainty, sample sizes, reproducibility. The
10 questions as a review gate.

What this module is: the analysis protocol for agent-produced analyses —
an agent reads eval results and writes an analysis ("model X improved",
"failure mode Y dominates"). The protocol binds the analysis plan BEFORE
the data is touched, splits exploratory from confirmatory, and runs the
10-question review gate over the reported finding. It is the analysis
counterpart to the experiment discipline in registry.py (#32): that
module pre-registers controlled *experiments*; this one pre-registers
*analyses* of data.

The ten questions are this repo's operationalization (adaptation) of the
book's themes — the issue names the themes but lists no question texts,
so no question below is a verbatim quote. Each question maps to a
machine-checkable field:

  Q1  Is the analysis plan registered?
  Q2  Was the plan written before the data was touched?
  Q3  Is the analysis question stated in advance?
  Q4  Is the analysis labeled confirmatory or exploratory, and does the
      finding carry the same label?
  Q5  Is the tested hypothesis one of the pre-specified hypotheses
      (confirmatory), or explicitly absent (exploratory)?
  Q6  Is the sample size stated, and does the observed n match the plan?
  Q7  Are the assumptions stated, each with what would break it?
  Q8  Is the uncertainty on the headline number stated as an interval,
      not a point?
  Q9  Is the decision rule stated in advance (what counts as confirmed)?
  Q10 Can someone else reproduce this from the plan + data + code?

Verdicts (closed set): CONFIRMED (confirmatory mode, all ten pass),
EXPLORATORY_ONLY (exploratory mode, all ten pass — the finding is a
finding, never a conclusion), REFUSED (any question fails, violations
named). Post-hoc storytelling is NOT confirmatory analysis: a
confirmatory finding must quote a registered hypothesis verbatim, and a
reworded or unregistered hypothesis matches nothing and is refused as
POST_HOC_CONFIRMATION. Hypothesis matching is exact by design, the same
discipline as #32's exact look fractions.

Enforcement boundary (stated honestly): the reviewer is a consistency
checker over reported evidence, like #32 — it does not see raw data.
The hash chain binds the plan contents immutably, and registered_at is
recorded, but nothing proves the plan was registered before data
collection except the reported timestamps (Q2) and the workflow rule
(require the digest before the run starts). A fabricated-but-consistent
report verifies clean. What the chain guarantees is that a plan cannot
be silently rewritten after registration, and that a post-hoc hypothesis
cannot be smuggled into a confirmatory verdict.

Amendment rule: there is no amendment path. Changing a plan means
registering a NEW plan; the old record stays, and findings on the old
digest remain reviewable against the old plan.

The registry is a single-writer file (same boundary as #32): concurrent
register_analysis_plan calls can chain to the same stale predecessor,
and the next read fails the whole file closed. One writer, or serialize
externally.

Everything is stdlib-only and deterministic; there is no model call here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

#: Confirmatory: the finding tests a pre-specified hypothesis and may
#: conclude. Exploratory: the finding describes what was seen and may
#: never conclude.
AnalysisMode = Literal["confirmatory", "exploratory"]

#: Closed verdict set for the review gate.
ReviewVerdict = Literal["CONFIRMED", "EXPLORATORY_ONLY", "REFUSED"]

#: The ten review-gate questions: (id, question text). Adaptation, not a
#: verbatim quote — see the module docstring.
THE_TEN_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("Q1", "Is the analysis plan registered?"),
    ("Q2", "Was the plan written before the data was touched?"),
    ("Q3", "Is the analysis question stated in advance?"),
    ("Q4", ("Is the analysis labeled confirmatory or exploratory, and "
           "does the finding carry the same label?")),
    ("Q5", ("Is the tested hypothesis one of the pre-specified hypotheses "
           "(confirmatory), or explicitly absent (exploratory)?")),
    ("Q6", ("Is the sample size stated, and does the observed n match "
           "the plan?")),
    ("Q7", "Are the assumptions stated, each with what would break it?"),
    ("Q8", ("Is the uncertainty on the headline number stated as an "
           "interval, not a point?")),
    ("Q9", ("Is the decision rule stated in advance (what counts as "
           "confirmed)?")),
    ("Q10", "Can someone else reproduce this from the plan + data + code?"),
)


class AnalysisViolation(ValueError):
    """The analysis finding violates its pre-registered plan (fail-closed)."""


class AnalysisRegistryCorrupted(ValueError):
    """The analysis-plan registry's hash chain does not verify."""


class AnalysisRegistryMalformed(AnalysisRegistryCorrupted):
    """A registry record is valid JSON but has the wrong shape."""


@dataclass(frozen=True)
class Assumption:
    """One stated assumption, with its falsifier.

    ``falsifier`` names the observation that would break the assumption
    ("if the control arm's n drops below 100, the comparison is void").
    An assumption without a falsifier is a slogan, not a checkable
    commitment, so construction refuses it.
    """

    text: str
    falsifier: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("assumption text must be non-empty")
        if not self.falsifier:
            raise ValueError(
                "assumption falsifier must be non-empty: "
                "state what observation would break this assumption"
            )


@dataclass(frozen=True)
class AnalysisPlan:
    """Pre-registered analysis plan.

    Confirmatory plans test pre-specified hypotheses against a
    pre-specified decision rule; exploratory plans describe what will
    be looked at and carry no hypothesis and no decision rule. The mode
    split is enforced at construction: a confirmatory plan without
    hypotheses or without a decision rule is refused, and an exploratory
    plan WITH hypotheses or a decision rule is refused — the split is
    structural, not a label the caller can fudge later.

    n_planned is the planned sample size. For confirmatory analyses it
    is a promise (the finding's n must land within n_tolerance); for
    exploratory analyses it is a budget (the finding's n must clear
    min_n, tolerance not enforced). data_source names where the data
    will come from, before it is touched. experiment_digest optionally
    links the analysis to a registered experiment plan (#32 digest) —
    a link, not a verification: the registries are separate files.
    """

    name: str
    mode: AnalysisMode
    question: str
    hypotheses: tuple[str, ...]
    assumptions: tuple[Assumption, ...]
    n_planned: int
    decision_rule: str
    data_source: str
    experiment_digest: str = ""
    n_tolerance: float = 0.02
    min_n: int = 30

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("plan name must be non-empty")
        if self.mode not in ("confirmatory", "exploratory"):
            raise ValueError(f"mode must be confirmatory/exploratory, got {self.mode!r}")
        if not self.question:
            raise ValueError("plan question must be non-empty")
        if not self.data_source:
            raise ValueError("data_source must be non-empty: name the data before touching it")
        if not self.assumptions:
            raise ValueError("at least one assumption is required")
        if self.n_planned < self.min_n:
            raise ValueError(
                f"n_planned={self.n_planned} below minimum {self.min_n}"
            )
        if not 0.0 <= self.n_tolerance < 1.0:
            raise ValueError(f"n_tolerance must be in [0, 1), got {self.n_tolerance}")
        if self.mode == "confirmatory":
            if not self.hypotheses or any(not h for h in self.hypotheses):
                raise ValueError("confirmatory plan needs >= 1 non-empty hypothesis")
            if not self.decision_rule:
                raise ValueError("confirmatory plan needs a decision_rule")
        else:
            if self.hypotheses:
                raise ValueError(
                    "exploratory plan carries no hypotheses: "
                    "a hypothesis stated in advance is a confirmatory plan"
                )
            if self.decision_rule:
                raise ValueError(
                    "exploratory plan carries no decision_rule: "
                    "there is nothing to decide, only something to describe"
                )


@dataclass(frozen=True)
class AnalysisFinding:
    """Everything an analysis reports at the end.

    plan_digest binds the finding to its registered plan. hypothesis is
    the registered hypothesis quoted verbatim (confirmatory) or ""
    (exploratory — an exploratory finding carries no hypothesis).
    interval is the uncertainty on the headline number as (lo, hi);
    required for confirmatory, optional for exploratory. data_digest
    identifies the data the analysis ran on; data_accessed_at is the
    ISO timestamp the data was first touched; code_version is the
    version or commit of the analysis code. Together the last three
    are the reproducibility record Q10 checks.
    """

    plan_digest: str
    mode: AnalysisMode
    hypothesis: str
    claim: str
    n_observed: int
    interval: tuple[float, float] | None
    data_digest: str
    data_accessed_at: str
    code_version: str

    def __post_init__(self) -> None:
        if not self.plan_digest:
            raise ValueError("plan_digest must be non-empty")
        if self.mode not in ("confirmatory", "exploratory"):
            raise ValueError(f"mode must be confirmatory/exploratory, got {self.mode!r}")
        if not self.claim:
            raise ValueError("claim must be non-empty")
        if self.n_observed < 1:
            raise ValueError(f"n_observed must be >= 1, got {self.n_observed}")
        if not self.data_digest:
            raise ValueError("data_digest must be non-empty")
        if not self.code_version:
            raise ValueError("code_version must be non-empty")
        try:
            _parse_time(self.data_accessed_at)
        except ValueError as e:
            raise ValueError(
                f"data_accessed_at must be an ISO timestamp, got {self.data_accessed_at!r}"
            ) from e
        if self.interval is not None:
            lo, hi = self.interval
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"interval bounds must be finite, got {self.interval!r}")
            if lo > hi:
                raise ValueError(f"interval lower bound exceeds upper, got {self.interval!r}")
        if self.mode == "confirmatory":
            if not self.hypothesis:
                raise ValueError("confirmatory finding needs a hypothesis")
            if self.interval is None:
                raise ValueError(
                    "confirmatory finding needs an uncertainty interval, not a point"
                )
        else:
            if self.hypothesis:
                raise ValueError(
                    "exploratory finding carries no hypothesis: "
                    "a tested hypothesis is a confirmatory finding"
                )


@dataclass(frozen=True)
class QuestionResult:
    """One review-gate question: pass/fail plus the evidence for it."""

    question_id: str
    question: str
    passed: bool
    evidence: str


@dataclass(frozen=True)
class ReviewGateReport:
    """Outcome of the 10-question review gate.

    verdict is CONFIRMED (confirmatory, all ten pass), EXPLORATORY_ONLY
    (exploratory, all ten pass), or REFUSED (any question fails).
    questions carries all ten results in order, so a refusal names
    exactly which questions failed and why.
    """

    verdict: ReviewVerdict
    questions: tuple[QuestionResult, ...]
    violations: tuple[str, ...]
    notes: tuple[str, ...]
    plan_digest: str


def _parse_time(text: str) -> datetime:
    """Parse an ISO timestamp; naive timestamps are read as UTC."""
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _canonical_plan(plan: AnalysisPlan) -> Mapping[str, object]:
    return {
        "name": plan.name,
        "mode": plan.mode,
        "question": plan.question,
        "hypotheses": list(plan.hypotheses),
        "assumptions": [
            {"text": a.text, "falsifier": a.falsifier} for a in plan.assumptions
        ],
        "n_planned": plan.n_planned,
        "decision_rule": plan.decision_rule,
        "data_source": plan.data_source,
        "experiment_digest": plan.experiment_digest,
        "n_tolerance": plan.n_tolerance,
        "min_n": plan.min_n,
    }


def _record_digest(prev_digest: str, canonical: Mapping[str, object]) -> str:
    payload = json.dumps(
        {"prev": prev_digest, "analysis_plan": canonical},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_records(registry_path: str) -> list[dict[str, object]]:
    try:
        with open(registry_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []
    records = [json.loads(line) for line in lines]
    prev = "GENESIS"
    try:
        for rec in records:
            if not isinstance(rec, dict):
                raise AnalysisRegistryMalformed(
                    f"record has wrong shape: {str(rec)[:80]}"
                )
            payload = rec.get("analysis_plan")
            if not isinstance(payload, dict):
                raise AnalysisRegistryMalformed(
                    f"record has wrong shape: {str(rec)[:80]}"
                )
            digest = _record_digest(prev, payload)
            if not isinstance(rec.get("digest"), str) or digest != rec["digest"]:
                raise AnalysisRegistryCorrupted(
                    f"hash chain broken at analysis plan "
                    f"{payload.get('name', '?')!r}: "
                    "the registry was rewritten after registration"
                )
            prev = digest
    except (AttributeError, TypeError, KeyError) as e:
        raise AnalysisRegistryMalformed(f"malformed registry record: {e}") from e
    return records


def _append_record(canonical: Mapping[str, object], registry_path: str) -> str:
    """Append one hash-chained analysis-plan record; return its digest."""
    parent = os.path.dirname(registry_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    records = _read_records(registry_path)
    prev = "GENESIS"
    if records:
        last_digest = records[-1]["digest"]
        assert isinstance(last_digest, str)
        prev = last_digest
    digest = _record_digest(prev, canonical)
    record = {
        "digest": digest,
        "registered_at": datetime.now(UTC).isoformat(),
        "analysis_plan": canonical,
    }
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return digest


def register_analysis_plan(plan: AnalysisPlan, registry_path: str) -> str:
    """Append a plan to the hash-chained registry; return its digest."""
    return _append_record(_canonical_plan(plan), registry_path)


def find_analysis_plan(plan: AnalysisPlan, registry_path: str) -> str | None:
    """Digest of a registered plan equal to ``plan``.

    Returns None when no record matches. Equality is on the canonical
    mapping, so a plan file that round-trips through JSON matches the
    record it was registered from. A post-hoc rewrite — a different
    hypothesis, a widened sample, a dropped assumption — matches
    nothing and is refused by the review gate.
    """
    canonical = dict(_canonical_plan(plan))
    for rec in _read_records(registry_path):
        if rec.get("analysis_plan") == canonical:
            digest = rec["digest"]
            assert isinstance(digest, str)
            return digest
    return None


def load_analysis_plan(digest: str, registry_path: str) -> tuple[AnalysisPlan, str]:
    """Load a registered plan by digest, with its registered_at timestamp.

    Unknown digests raise KeyError; a broken chain raises
    AnalysisRegistryCorrupted through _read_records.
    """
    for rec in _read_records(registry_path):
        if rec["digest"] != digest:
            continue
        payload = rec["analysis_plan"]
        if not isinstance(payload, dict):
            raise AnalysisRegistryMalformed(
                f"plan record has wrong shape: {str(rec)[:80]}"
            )
        registered_at = rec.get("registered_at")
        if not isinstance(registered_at, str):
            raise AnalysisRegistryMalformed(
                f"plan record has no registered_at: {str(rec)[:80]}"
            )
        plan = AnalysisPlan(
            name=payload["name"],
            mode=payload["mode"],
            question=payload["question"],
            hypotheses=tuple(payload["hypotheses"]),
            assumptions=tuple(
                Assumption(text=a["text"], falsifier=a["falsifier"])
                for a in payload["assumptions"]
            ),
            n_planned=payload["n_planned"],
            decision_rule=payload["decision_rule"],
            data_source=payload["data_source"],
            experiment_digest=payload.get("experiment_digest", ""),
            n_tolerance=payload["n_tolerance"],
            min_n=payload["min_n"],
        )
        return plan, registered_at
    raise KeyError(f"no registered analysis plan with digest {digest[:12]}...")


def plan_to_dict(plan: AnalysisPlan) -> dict[str, object]:
    """Canonical mapping, JSON-ready (round-trips through plan_from_dict)."""
    return dict(_canonical_plan(plan))


def plan_from_dict(doc: Mapping[str, object]) -> AnalysisPlan:
    """Build a plan from a JSON doc (field names match plan_to_dict)."""
    try:
        return AnalysisPlan(
            name=str(doc["name"]),
            mode=str(doc["mode"]),  # type: ignore[arg-type]
            question=str(doc["question"]),
            hypotheses=tuple(str(h) for h in doc["hypotheses"]),  # type: ignore[union-attr]
            assumptions=tuple(
                Assumption(text=str(a["text"]), falsifier=str(a["falsifier"]))
                for a in doc["assumptions"]  # type: ignore[union-attr]
            ),
            n_planned=int(doc["n_planned"]),  # type: ignore[arg-type]
            decision_rule=str(doc["decision_rule"]),
            data_source=str(doc["data_source"]),
            experiment_digest=str(doc.get("experiment_digest", "")),
            n_tolerance=float(doc.get("n_tolerance", 0.02)),
            min_n=int(doc.get("min_n", 30)),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"invalid analysis plan doc: {e}") from e


def finding_to_dict(finding: AnalysisFinding) -> dict[str, object]:
    """JSON-ready mapping (round-trips through finding_from_dict)."""
    return {
        "plan_digest": finding.plan_digest,
        "mode": finding.mode,
        "hypothesis": finding.hypothesis,
        "claim": finding.claim,
        "n_observed": finding.n_observed,
        "interval": list(finding.interval) if finding.interval is not None else None,
        "data_digest": finding.data_digest,
        "data_accessed_at": finding.data_accessed_at,
        "code_version": finding.code_version,
    }


def finding_from_dict(doc: Mapping[str, object]) -> AnalysisFinding:
    """Build a finding from a JSON doc (field names match finding_to_dict)."""
    try:
        raw_interval = doc.get("interval")
        interval: tuple[float, float] | None
        if raw_interval is None:
            interval = None
        else:
            pair = (float(raw_interval[0]), float(raw_interval[1]))  # type: ignore[index]
            interval = pair
        return AnalysisFinding(
            plan_digest=str(doc["plan_digest"]),
            mode=str(doc["mode"]),  # type: ignore[arg-type]
            hypothesis=str(doc.get("hypothesis", "")),
            claim=str(doc["claim"]),
            n_observed=int(doc["n_observed"]),  # type: ignore[arg-type]
            interval=interval,
            data_digest=str(doc["data_digest"]),
            data_accessed_at=str(doc["data_accessed_at"]),
            code_version=str(doc["code_version"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"invalid analysis finding doc: {e}") from e


def _n_within(plan: AnalysisPlan, n_observed: int) -> bool:
    lo = plan.n_planned * (1.0 - plan.n_tolerance)
    hi = plan.n_planned * (1.0 + plan.n_tolerance)
    return lo <= n_observed <= hi


def _question(
    question_id: str,
    passed: bool,
    evidence: str,
    violations: list[str],
    code: str | None,
) -> QuestionResult:
    if not passed and code is not None:
        violations.append(code)
    text = dict(THE_TEN_QUESTIONS)[question_id]
    return QuestionResult(
        question_id=question_id, question=text, passed=passed, evidence=evidence
    )


def review_analysis(
    plan_digest: str, registry_path: str, finding: AnalysisFinding
) -> ReviewGateReport:
    """Run the 10-question review gate over a finding.

    Never raises on violations — it names them. Use check_analysis for
    the raising (CI-gating) form. An unregistered or rewritten plan is
    UNREGISTERED_PLAN; an unreadable registry is REGISTRY_UNREADABLE.
    """
    try:
        plan, registered_at = load_analysis_plan(plan_digest, registry_path)
    except AnalysisRegistryMalformed as e:
        return ReviewGateReport(
            verdict="REFUSED",
            questions=(),
            violations=("REGISTRY_UNREADABLE",),
            notes=(f"{type(e).__name__}: {e}",),
            plan_digest=plan_digest,
        )
    except (KeyError, AnalysisRegistryCorrupted):
        return ReviewGateReport(
            verdict="REFUSED",
            questions=(),
            violations=("UNREGISTERED_PLAN",),
            notes=(),
            plan_digest=plan_digest,
        )
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        return ReviewGateReport(
            verdict="REFUSED",
            questions=(),
            violations=("REGISTRY_UNREADABLE",),
            notes=(f"{type(e).__name__}: {e}",),
            plan_digest=plan_digest,
        )

    violations: list[str] = []
    notes: list[str] = []
    questions: list[QuestionResult] = []

    # Q1 — registered: load_analysis_plan already proved it.
    questions.append(
        _question(
            "Q1", True, f"plan {plan.name!r} found in the registry", violations, None
        )
    )

    # Q2 — plan before data: reported timestamps, same boundary as #32.
    try:
        registered_time = _parse_time(registered_at)
        accessed_time = _parse_time(finding.data_accessed_at)
        before_data = registered_time <= accessed_time
        evidence_q2 = (
            f"registered_at={registered_at}, "
            f"data_accessed_at={finding.data_accessed_at}"
        )
    except ValueError:
        before_data = False
        evidence_q2 = (
            f"unparseable timestamp: registered_at={registered_at!r}, "
            f"data_accessed_at={finding.data_accessed_at!r}"
        )
    questions.append(
        _question("Q2", before_data, evidence_q2, violations, "PLAN_AFTER_DATA")
    )

    # Q3 — question stated in advance (construction-enforced; quote it).
    questions.append(
        _question(
            "Q3",
            bool(plan.question),
            f"question: {plan.question!r}",
            violations,
            "QUESTION_UNSTATED",
        )
    )

    # Q4 — mode labeled, finding carries the same label.
    mode_match = finding.mode == plan.mode
    questions.append(
        _question(
            "Q4",
            mode_match,
            f"plan mode={plan.mode}, finding mode={finding.mode}",
            violations,
            "MODE_MISMATCH",
        )
    )

    # Q5 — hypothesis pre-specified (exact match, deliberate — see docstring).
    if plan.mode == "confirmatory":
        prespecified = finding.hypothesis in plan.hypotheses
        evidence_q5 = (
            f"hypothesis {finding.hypothesis!r} "
            + ("matches a registered hypothesis" if prespecified else
               f"matches none of {len(plan.hypotheses)} registered hypotheses")
        )
        questions.append(
            _question("Q5", prespecified, evidence_q5, violations, "POST_HOC_CONFIRMATION")
        )
    else:
        questions.append(
            _question(
                "Q5",
                finding.hypothesis == "",
                "exploratory finding carries no hypothesis, as required",
                violations,
                "POST_HOC_CONFIRMATION",
            )
        )

    # Q6 — sample size stated; confirmatory additionally within tolerance.
    if finding.n_observed < plan.min_n:
        questions.append(
            _question(
                "Q6",
                False,
                f"n_observed={finding.n_observed} below min_n={plan.min_n}",
                violations,
                "SAMPLE_TOO_SMALL",
            )
        )
    elif plan.mode == "confirmatory" and not _n_within(plan, finding.n_observed):
        questions.append(
            _question(
                "Q6",
                False,
                f"n_observed={finding.n_observed} outside tolerance of "
                f"n_planned={plan.n_planned} (+/-{plan.n_tolerance:.0%})",
                violations,
                "SAMPLE_DEVIATION",
            )
        )
    else:
        scope = (
            f"within tolerance of n_planned={plan.n_planned}"
            if plan.mode == "confirmatory"
            else f"stated (n={finding.n_observed}); tolerance not enforced "
                 "for exploratory"
        )
        questions.append(
            _question("Q6", True, f"n_observed={finding.n_observed}, {scope}",
                      violations, None)
        )

    # Q7 — assumptions stated, each with a falsifier.
    assumptions_ok = bool(plan.assumptions) and all(
        a.text and a.falsifier for a in plan.assumptions
    )
    questions.append(
        _question(
            "Q7",
            assumptions_ok,
            f"{len(plan.assumptions)} assumption(s) stated, each with a falsifier",
            violations,
            "ASSUMPTIONS_UNSTATED",
        )
    )

    # Q8 — uncertainty as an interval, not a point.
    if plan.mode == "confirmatory":
        interval_ok = finding.interval is not None
        evidence_q8 = (
            f"interval={finding.interval}"
            if interval_ok
            else "no interval: the headline number is a bare point"
        )
        questions.append(
            _question("Q8", interval_ok, evidence_q8, violations, "UNCERTAINTY_MISSING")
        )
    else:
        evidence_q8 = (
            f"interval={finding.interval} stated"
            if finding.interval is not None
            else "no interval stated; optional for exploratory"
        )
        questions.append(_question("Q8", True, evidence_q8, violations, None))

    # Q9 — decision rule stated in advance.
    if plan.mode == "confirmatory":
        rule_ok = bool(plan.decision_rule)
        evidence_q9 = (
            f"decision_rule: {plan.decision_rule!r}"
            if rule_ok
            else "no decision rule: nothing was declared to count as confirmed"
        )
        questions.append(
            _question("Q9", rule_ok, evidence_q9, violations, "DECISION_RULE_MISSING")
        )
    else:
        questions.append(
            _question(
                "Q9",
                plan.decision_rule == "",
                "no confirmatory decision rule, as required for exploratory",
                violations,
                "DECISION_RULE_MISSING",
            )
        )

    # Q10 — reproducible: data digest + code version recorded.
    reproducible = bool(finding.data_digest) and bool(finding.code_version)
    questions.append(
        _question(
            "Q10",
            reproducible,
            f"data_digest={finding.data_digest[:12]}..., "
            f"code_version={finding.code_version!r}",
            violations,
            "REPRODUCIBILITY_MISSING",
        )
    )

    if violations:
        verdict: ReviewVerdict = "REFUSED"
    elif finding.mode == "confirmatory":
        verdict = "CONFIRMED"
    else:
        verdict = "EXPLORATORY_ONLY"
        notes.append(
            "exploratory finding: a finding, never a conclusion — "
            "do not cite it as confirmation"
        )
    return ReviewGateReport(
        verdict=verdict,
        questions=tuple(questions),
        violations=tuple(violations),
        notes=tuple(notes),
        plan_digest=plan_digest,
    )


def check_analysis(
    plan_digest: str, registry_path: str, finding: AnalysisFinding
) -> ReviewGateReport:
    """review_analysis, but violations raise AnalysisViolation.

    This is the CI gate: call it at the end of an analysis run and any
    discipline violation fails the build.
    """
    report = review_analysis(plan_digest, registry_path, finding)
    if report.verdict == "REFUSED":
        raise AnalysisViolation(
            f"analysis violates its pre-registered plan "
            f"({report.plan_digest[:12]}...): {', '.join(report.violations)}"
        )
    return report
