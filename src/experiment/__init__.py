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

Analysis pre-specification (agenteval-bench#48, Spiegelhalter *The Art
of Statistics*) is enforced at the `prespec.review_analysis` /
`prespec.check_analysis` seam: an agent-produced analysis is reviewed
against its pre-registered analysis plan through the 10-question gate
(plan registered before data touched, exploratory/confirmatory split,
hypotheses pre-specified, sample sizes and assumptions stated,
uncertainty as an interval, decision rule declared, reproducible).
A confirmatory finding must quote a registered hypothesis verbatim —
post-hoc storytelling is refused as POST_HOC_CONFIRMATION, never
presented as a conclusion.
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
    design_from_dict,
    design_from_json,
    evidence_from_json,
    load_registered_design,
    newcombe_interval,
    register_design,
    ship_decision,
    two_proportion_z,
    two_sided_p,
)
from experiment.prespec import (
    AnalysisFinding,
    AnalysisPlan,
    AnalysisRegistryCorrupted,
    AnalysisRegistryMalformed,
    AnalysisViolation,
    Assumption,
    QuestionResult,
    ReviewGateReport,
    check_analysis,
    find_analysis_plan,
    finding_from_dict,
    finding_to_dict,
    load_analysis_plan,
    plan_from_dict,
    plan_to_dict,
    register_analysis_plan,
    review_analysis,
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
    "AnalysisFinding",
    "AnalysisPlan",
    "AnalysisRegistryCorrupted",
    "AnalysisRegistryMalformed",
    "AnalysisViolation",
    "Assumption",
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
    "QuestionResult",
    "ReplicationEvidence",
    "ReviewGateReport",
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
    "check_analysis",
    "check_experiment",
    "check_guardrail",
    "check_ship",
    "check_srm",
    "chi2_sf",
    "cuped_adjust",
    "default_design",
    "design_from_dict",
    "design_from_json",
    "disaggregate",
    "evidence_from_json",
    "find_analysis_plan",
    "finding_from_dict",
    "finding_to_dict",
    "group_sequential_boundaries",
    "inv_normal_cdf",
    "load_analysis_plan",
    "load_plan",
    "load_registered_design",
    "newcombe_interval",
    "obrien_fleming_spend",
    "plan_from_dict",
    "plan_to_dict",
    "pooled_rates",
    "register_analysis_plan",
    "register_design",
    "register_plan",
    "review_analysis",
    "ship_decision",
    "spending_report",
    "stratum_delta",
    "two_proportion_z",
    "two_sided_p",
    "verify_experiment",
    "win_claim",
]
