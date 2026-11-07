"""Experiment discipline: pre-registration, no-peeking enforcement, CUPED.

Implements agenteval-bench#32 (Kohavi, Tang & Xu, *Trustworthy Online
Controlled Experiments*): model-comparison experiments pre-register their
sample size and stopping rule; interim peeks are allowed only on the
registered look schedule and spend alpha through a Lan-DeMets
O'Brien-Fleming spending function; CUPED variance reduction applies where
the plan declares a pre-experiment covariate. Violations are refused
fail-closed so CI can gate on them.

Enforcement boundary (stated honestly): discipline is enforced at the
`registry.verify_experiment` / `registry.check_experiment` seam, which
replays evidence through a fresh `SequentialGate` and re-derives every
decision. The pure math helpers (`obrien_fleming_spend`,
`group_sequential_boundaries`, `cuped_adjust`) stay directly callable —
tests pin the seam, and the seam is what CI runs. The verifier is a
consistency checker over reported evidence (fractions vs plan, decisions
vs reported z's, arm n's vs n_final); it does not see raw data.

Aggregation discipline (agenteval-bench#36, Kohavi ch. 18) is enforced at
the `simpson.check_aggregation` / `simpson.assert_aggregation` /
`simpson.win_claim` seam: pooled win reporting is refused on allocation
shift across slices, and a win claim that flips direction in any stratum
of an exhaustive decomposition is rejected. There is no pooled-only
path to a win claim — `disaggregate` is mandatory output on every path.

Ship discipline (agenteval-bench#46, Kohavi ch. 5) is enforced at the
`oec.ship_decision` / `oec.check_ship` seam: an agent change ships only
when its OEC and guardrails were pre-registered against a registered
plan, the SRM and A/A trust tests pass BEFORE the OEC result is read,
no guardrail regressed beyond its pre-registered budget, the OEC won
significantly in the declared direction, and a surprising win
(Twyman's law) carries confirming replication evidence.
"""

from __future__ import annotations

from experiment.alpha_spend import (
    LookDecision,
    PeekRefused,
    SequentialGate,
    group_sequential_boundaries,
    inv_normal_cdf,
    obrien_fleming_spend,
    spending_report,
)
from experiment.cuped import CovariateViolation, CupedResult, cuped_adjust
from experiment.oec import (
    AAEvidence,
    AaVerdict,
    ExperimentDesign,
    GuardrailVerdict,
    MetricDeclaration,
    MetricDirection,
    MetricObservation,
    MetricRole,
    ReplicationEvidence,
    ShipEvidence,
    ShipRefused,
    ShipStatus,
    ShipVerdict,
    SrmVerdict,
    TrustStatus,
    check_aa,
    check_guardrail,
    check_ship,
    check_srm,
    chi2_sf,
    default_design,
    design_from_json,
    evidence_from_json,
    ship_decision,
    two_proportion_z,
    two_sided_p,
)
from experiment.registry import (
    ExperimentEvidence,
    ExperimentVerdict,
    ExperimentViolation,
    LookEvidence,
    check_experiment,
    load_plan,
    register_plan,
    verify_experiment,
)
from experiment.simpson import (
    DEFAULT_ALLOCATION_TOLERANCE,
    AggregationReport,
    AggregationVerdict,
    AggregationViolation,
    Stratum,
    StratumReport,
    WinVerdict,
    assert_aggregation,
    check_aggregation,
    disaggregate,
    pooled_rates,
    stratum_delta,
    win_claim,
)
from experiment.types import CovariateSpec, ExperimentPlan

__all__ = [
    "DEFAULT_ALLOCATION_TOLERANCE",
    "AAEvidence",
    "AaVerdict",
    "AggregationReport",
    "AggregationVerdict",
    "AggregationViolation",
    "CovariateSpec",
    "CovariateViolation",
    "CupedResult",
    "ExperimentDesign",
    "ExperimentEvidence",
    "ExperimentPlan",
    "ExperimentVerdict",
    "ExperimentViolation",
    "GuardrailVerdict",
    "LookDecision",
    "LookEvidence",
    "MetricDeclaration",
    "MetricDirection",
    "MetricObservation",
    "MetricRole",
    "PeekRefused",
    "ReplicationEvidence",
    "SequentialGate",
    "ShipEvidence",
    "ShipRefused",
    "ShipStatus",
    "ShipVerdict",
    "SrmVerdict",
    "Stratum",
    "StratumReport",
    "TrustStatus",
    "WinVerdict",
    "assert_aggregation",
    "check_aa",
    "check_aggregation",
    "check_experiment",
    "check_guardrail",
    "check_ship",
    "check_srm",
    "chi2_sf",
    "cuped_adjust",
    "default_design",
    "design_from_json",
    "disaggregate",
    "evidence_from_json",
    "group_sequential_boundaries",
    "inv_normal_cdf",
    "load_plan",
    "obrien_fleming_spend",
    "pooled_rates",
    "register_plan",
    "ship_decision",
    "spending_report",
    "stratum_delta",
    "two_proportion_z",
    "two_sided_p",
    "verify_experiment",
    "win_claim",
]
