"""Pre-registration ledger and the experiment verification seam.

register_plan appends a hash-chained JSONL record: each record commits to
the previous record's digest, so a plan cannot be silently rewritten
after data arrives. verify_experiment replays the reported evidence
through a fresh SequentialGate and re-derives every interim decision from
the raw z-statistics — a fabricated reject/accept flag is caught as
DECISION_MISMATCH, an off-schedule peek as UNREGISTERED_PEEK, an
unregistered or rewritten plan as UNREGISTERED_PLAN, a moved sample size
as SAMPLE_DEVIATION, and a missing CUPED adjustment as CUPED_REQUIRED.

check_experiment raises ExperimentViolation on any violation: wire it
into CI and violations fail the build. That is the "violations fail CI"
acceptance criterion of #32, enforced by a running check, not prose.

Early stopping is a first-class compliant path: evidence may declare
stopped_at, the information fraction where the experiment stopped after
an interim reject. Verification then checks the looks exactly through
the stopping look, re-derives the reject from z, and checks n_final
against n_planned * stopped_at. Stopping early without a reject is
refused as EARLY_STOP_WITHOUT_REJECT.

Design registration (agenteval-bench#46): the same hash chain also
carries experiment DESIGN records — which metric is the OEC, the
guardrail budgets, the expected allocation, the surprise threshold.
A design record commits to the previous record's digest exactly like
a plan record, so a design cannot be silently rewritten after data
arrives either. `register_design` appends one; `find_design` checks
whether a design was registered. The ship gate (#46) refuses any
design that is not in the chain. Amendment rule (stated, not hidden):
the chain is append-only, so correcting a design means registering a
NEW design record; the old record stays, and the ship verdict records
which design digest was used, so an audit can check it was the
current one. Timing is workflow-bound, same as #32: the chain binds
contents immutably, it does not prove the design was registered
before data collection — but now there IS a recorded design whose
precedence could be proven, where before there was none.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from experiment.alpha_spend import PeekRefused, SequentialGate
from experiment.cuped import CupedResult
from experiment.types import CovariateSpec, ExperimentPlan


class ExperimentViolation(ValueError):
    """The experiment evidence violates its pre-registered plan (fail-closed)."""


class RegistryCorrupted(ValueError):
    """The plan registry's hash chain does not verify."""


class RegistryMalformed(RegistryCorrupted):
    """A registry record is valid JSON but has the wrong shape."""


@dataclass(frozen=True)
class LookEvidence:
    """Reported interim result: fraction, z-statistic, claimed decision.

    ``metric`` names the metric the z-statistic belongs to. Empty
    means unnamed (pre-#46 evidence): the sequential gate does not
    need it, and #32 verification ignores it. The #46 ship gate
    requires every look to name the design's OEC metric — a z with no
    metric provenance cannot ship as an OEC win.
    """

    fraction: float
    z: float
    rejected: bool
    metric: str = ""


@dataclass(frozen=True)
class ExperimentEvidence:
    """Everything the experiment reports at the end.

    stopped_at: information fraction where the experiment stopped early
    following an interim reject. None means the experiment ran the full
    schedule (stopped_at=1.0 is equivalent to None — pass the literal
    1.0; an arithmetically derived fraction takes the early-stop path).
    An early stop is compliant ONLY if the stopping look really rejected
    (re-derived from z, not just claimed) — stopping early without a
    reject is EARLY_STOP_WITHOUT_REJECT.
    """

    n_final: int
    looks: tuple[LookEvidence, ...]
    cuped: CupedResult | None = None
    stopped_at: float | None = None


@dataclass(frozen=True)
class ExperimentVerdict:
    """Outcome of verification: valid, or the named violations."""

    valid: bool
    violations: tuple[str, ...]
    notes: tuple[str, ...]
    plan_digest: str


def _canonical_plan(plan: ExperimentPlan) -> Mapping[str, object]:
    return {
        "name": plan.name,
        "n_planned": plan.n_planned,
        "alpha": plan.alpha,
        "sides": plan.sides,
        "looks": list(plan.looks),
        "covariate": (
            {"name": plan.covariate.name, "source": plan.covariate.source}
            if plan.covariate is not None
            else None
        ),
        "n_tolerance": plan.n_tolerance,
        "min_n": plan.min_n,
    }


def _record_digest(
    prev_digest: str, canonical: Mapping[str, object], kind_key: str = "plan"
) -> str:
    payload = json.dumps(
        {"prev": prev_digest, kind_key: canonical},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record_kind(record: dict[str, object]) -> str:
    """The record's kind: "plan" or "design"; "" when malformed."""
    has_plan = isinstance(record.get("plan"), dict)
    has_design = isinstance(record.get("design"), dict)
    if has_plan and not has_design:
        return "plan"
    if has_design and not has_plan:
        return "design"
    return ""


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
                raise RegistryMalformed(f"record has wrong shape: {str(rec)[:80]}")
            kind = _record_kind(rec)
            if not kind:
                raise RegistryMalformed(f"record has wrong shape: {str(rec)[:80]}")
            kind_payload = rec[kind]
            if not isinstance(kind_payload, dict):
                raise RegistryMalformed(f"record has wrong shape: {str(rec)[:80]}")
            digest = _record_digest(prev, kind_payload, kind)
            if not isinstance(rec.get("digest"), str) or digest != rec["digest"]:
                raise RegistryCorrupted(
                    f"hash chain broken at {kind} "
                    f"{kind_payload.get('name', kind_payload.get('plan_digest', '?'))!r}: "
                    "the registry was rewritten after registration"
                )
            prev = digest
    except (AttributeError, TypeError, KeyError) as e:
        raise RegistryMalformed(f"malformed registry record: {e}") from e
    return records


def _append_record(
    kind_key: str, canonical: Mapping[str, object], registry_path: str
) -> str:
    """Append one hash-chained record of either kind; return its digest."""
    parent = os.path.dirname(registry_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    records = _read_records(registry_path)
    prev = "GENESIS"
    if records:
        last_digest = records[-1]["digest"]
        assert isinstance(last_digest, str)
        prev = last_digest
    digest = _record_digest(prev, canonical, kind_key)
    record = {
        "digest": digest,
        "registered_at": datetime.now(UTC).isoformat(),
        kind_key: canonical,
    }
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return digest


def register_plan(plan: ExperimentPlan, registry_path: str) -> str:
    """Append a plan to the hash-chained registry; return its digest."""
    return _append_record("plan", _canonical_plan(plan), registry_path)


def register_design(
    canonical_design: Mapping[str, object], registry_path: str
) -> str:
    """Append a design to the hash-chained registry; return its digest.

    Takes the canonical design mapping (see
    ``experiment.oec.ExperimentDesign.to_dict``) rather than the
    dataclass, so this module stays free of the OEC layer: the caller
    validates the design before registering it. A design record binds
    the OEC, the guardrail budgets, the expected allocation, and the
    surprise threshold to the plan digest it names.
    """
    if not isinstance(canonical_design, dict) or not canonical_design:
        raise ValueError("design record needs a non-empty canonical mapping")
    return _append_record("design", canonical_design, registry_path)


def load_design(digest: str, registry_path: str) -> Mapping[str, object]:
    """Load a registered design's canonical mapping by digest.

    Unknown digests raise KeyError; a broken chain raises
    RegistryCorrupted through _read_records.
    """
    for rec in _read_records(registry_path):
        if rec["digest"] == digest and _record_kind(rec) == "design":
            design = rec["design"]
            assert isinstance(design, dict)
            return design
    raise KeyError(f"no registered design with digest {digest[:12]}...")


def find_design(
    canonical_design: Mapping[str, object], registry_path: str
) -> str | None:
    """Digest of a registered design equal to ``canonical_design``.

    Returns None when no design record matches. Equality is on the
    canonical mapping (dict equality, key order irrelevant), so a
    design file that round-trips through JSON matches the record it
    was registered from. A post-hoc rewrite — a different OEC, a
    wider guardrail budget, a moved surprise threshold — matches
    nothing and is refused by the ship gate.
    """
    for rec in _read_records(registry_path):
        if _record_kind(rec) != "design":
            continue
        if rec["design"] == dict(canonical_design):
            digest = rec["digest"]
            assert isinstance(digest, str)
            return digest
    return None


def load_plan(digest: str, registry_path: str) -> ExperimentPlan:
    """Load a registered plan by digest; unknown digests raise KeyError.

    Design records in the same chain are skipped: a design digest is
    not a plan digest, and vice versa.
    """
    for rec in _read_records(registry_path):
        if rec["digest"] != digest or _record_kind(rec) != "plan":
            continue
        p = rec["plan"]
        if not isinstance(p, dict):
            raise RegistryMalformed(f"plan record has wrong shape: {str(rec)[:80]}")
        cov = p["covariate"]
        return ExperimentPlan(
            name=p["name"],
            n_planned=p["n_planned"],
            alpha=p["alpha"],
            sides=p["sides"],
            looks=tuple(p["looks"]),
            covariate=(
                CovariateSpec(name=cov["name"], source=cov["source"])
                if cov is not None
                else None
            ),
            n_tolerance=p["n_tolerance"],
            min_n=p["min_n"],
        )
    raise KeyError(f"no registered plan with digest {digest[:12]}...")


def verify_experiment(
    plan_digest: str, registry_path: str, evidence: ExperimentEvidence
) -> ExperimentVerdict:
    """Verify reported evidence against its pre-registered plan.

    Never raises on violations — it names them. Use check_experiment for
    the raising (CI-gating) form. An early stop (stopped_at set) is
    compliant only when the stopping look really rejected: the whole
    point of the spending schedule is that a valid interim reject ends
    the experiment.
    """
    try:
        plan = load_plan(plan_digest, registry_path)
    except RegistryMalformed as e:
        return ExperimentVerdict(
            valid=False,
            violations=("REGISTRY_UNREADABLE",),
            notes=(f"{type(e).__name__}: {e}",),
            plan_digest=plan_digest,
        )
    except (KeyError, RegistryCorrupted):
        return ExperimentVerdict(
            valid=False,
            violations=("UNREGISTERED_PLAN",),
            notes=(),
            plan_digest=plan_digest,
        )
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        return ExperimentVerdict(
            valid=False,
            violations=("REGISTRY_UNREADABLE",),
            notes=(f"{type(e).__name__}: {e}",),
            plan_digest=plan_digest,
        )
    violations: list[str] = []
    notes: list[str] = []
    # Replay every reported look through a fresh gate: the gate refuses
    # off-schedule peeks, and re-deriving reject/accept from z catches
    # fabricated decision flags.
    gate = SequentialGate(plan)
    # stopped_at=1.0 is the full schedule under another name: the
    # experiment ran every look, so it takes the full-schedule path.
    if evidence.stopped_at is not None and evidence.stopped_at != 1.0:
        _verify_early_stop(plan, gate, evidence, violations)
    else:
        _verify_full_schedule(plan, gate, evidence, violations)
    if plan.covariate is not None and evidence.cuped is None:
        violations.append("CUPED_REQUIRED")
    if (
        evidence.cuped is not None
        and evidence.cuped.n_control + evidence.cuped.n_treated != evidence.n_final
    ):
        violations.append("CUPED_COVERAGE_MISMATCH")
    if plan.covariate is None and evidence.cuped is not None:
        notes.append("cuped supplied without a planned covariate: advisory only")
    return ExperimentVerdict(
        valid=not violations,
        violations=tuple(violations),
        notes=tuple(notes),
        plan_digest=plan_digest,
    )


def _n_within(plan: ExperimentPlan, n_final: int, expected: int) -> bool:
    lo = expected * (1.0 - plan.n_tolerance)
    hi = expected * (1.0 + plan.n_tolerance)
    return lo <= n_final <= hi


def _replay_looks(
    gate: SequentialGate, looks: tuple[LookEvidence, ...], violations: list[str]
) -> bool:
    """Replay looks through the gate; True if all replayed cleanly."""
    for ev in looks:
        try:
            decision = gate.look(ev.fraction, ev.z)
        except PeekRefused:
            violations.append("UNREGISTERED_PEEK")
            return False
        if decision.reject != ev.rejected:
            violations.append("DECISION_MISMATCH")
            return False
    return True


def _verify_full_schedule(
    plan: ExperimentPlan,
    gate: SequentialGate,
    evidence: ExperimentEvidence,
    violations: list[str],
) -> None:
    if not _n_within(plan, evidence.n_final, plan.n_planned):
        violations.append("SAMPLE_DEVIATION")
    if len(evidence.looks) != len(plan.looks):
        violations.append("LOOK_SCHEDULE_MISMATCH")
    else:
        _replay_looks(gate, evidence.looks, violations)


def _verify_early_stop(
    plan: ExperimentPlan,
    gate: SequentialGate,
    evidence: ExperimentEvidence,
    violations: list[str],
) -> None:
    stop = evidence.stopped_at
    assert stop is not None
    if stop not in plan.looks:
        violations.append("UNREGISTERED_PEEK")
        return
    expected_looks = plan.looks[: plan.looks.index(stop) + 1]
    if tuple(ev.fraction for ev in evidence.looks) != expected_looks:
        violations.append("LOOK_SCHEDULE_MISMATCH")
        return
    if not _n_within(plan, evidence.n_final, round(plan.n_planned * stop)):
        violations.append("SAMPLE_DEVIATION")
    # The stopping look must really have rejected: stopping early on
    # an accept is not a stopping rule, it is peeking and quitting.
    if _replay_looks(gate, evidence.looks, violations) and not evidence.looks[-1].rejected:
        violations.append("EARLY_STOP_WITHOUT_REJECT")


def check_experiment(
    plan_digest: str, registry_path: str, evidence: ExperimentEvidence
) -> ExperimentVerdict:
    """verify_experiment, but violations raise ExperimentViolation.

    This is the CI gate: call it at the end of an experiment run and any
    discipline violation fails the build.
    """
    verdict = verify_experiment(plan_digest, registry_path, evidence)
    if not verdict.valid:
        raise ExperimentViolation(
            f"experiment violates its pre-registered plan "
            f"({verdict.plan_digest[:12]}...): {', '.join(verdict.violations)}"
        )
    return verdict
