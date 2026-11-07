"""Evaluation as a decision system: evidence -> uncertainty -> threshold -> action.

An eval score is not an evaluation until it maps to a decision. This module
turns per-case scores into a typed decision (GATE / ROLLBACK / INVESTIGATE)
with the uncertainty carried explicitly:

* every dimension (quality, safety, cost, reliability) is measured as
  evidence with a 95% confidence interval, never a bare point estimate;
* thresholds are set per risk level (RiskLevel), never one threshold for
  everything;
* the compared quantity is the interval bound, not the point estimate:
  a dimension passes only when its whole interval clears the threshold,
  fails only when its whole interval misses it, and otherwise the decision
  is INVESTIGATE (collect more evidence);
* a dimension with no evidence is unmeasured, and an unmeasured dimension
  forces INVESTIGATE, never GATE. A score with no mapped decision is
  decoration, so this module refuses to produce one.

Intervals are Wilson score intervals for proportions (established) and
mean +/- 1.96 * standard-error normal intervals for cost (adaptation).
The cost interval is a normal approximation with no t correction, so it
understates uncertainty at small n; a single cost observation (n == 1)
carries no measured uncertainty at all and can never GATE. The budget a
cost interval is judged against always comes from the policy at the
decision's risk level, never from the evidence: evidence describes what
was observed, the policy decides what is acceptable.
Everything is stdlib-only and deterministic; there is no model call here.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

Z_95 = 1.96


class Dimension(str, Enum):
    """The four mandatory dimensions of an evaluation decision."""

    QUALITY = "quality"
    SAFETY = "safety"
    COST = "cost"
    RELIABILITY = "reliability"


class RiskLevel(str, Enum):
    """Risk level a deployment decision is taken at. Thresholds are per level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionAction(str, Enum):
    """The closed set of actions an evaluation can produce."""

    GATE = "gate"
    ROLLBACK = "rollback"
    INVESTIGATE = "investigate"


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (established).

    Returns (lower, upper), clamped to [0, 1]. Raises ValueError on an
    empty sample or an out-of-range count: an interval over no evidence
    is not an interval, it is a missing measurement.
    """
    if trials <= 0:
        raise ValueError("wilson_interval needs trials >= 1")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes {successes} out of range for trials {trials}")
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class ProportionEvidence:
    """Evidence for a proportion dimension: k successes in n trials."""

    dimension: Dimension
    successes: int
    trials: int

    def __post_init__(self) -> None:
        if self.dimension is Dimension.COST:
            raise ValueError("cost evidence is CostEvidence, not a proportion")
        if self.trials <= 0:
            raise ValueError("proportion evidence needs trials >= 1")
        if not 0 <= self.successes <= self.trials:
            raise ValueError(
                f"successes {self.successes} out of range for trials {self.trials}"
            )

    @property
    def point(self) -> float:
        """Point estimate (the observed proportion)."""
        return self.successes / self.trials

    @property
    def interval(self) -> tuple[float, float]:
        """95% Wilson score interval (lower, upper)."""
        return wilson_interval(self.successes, self.trials)

    @property
    def lower(self) -> float:
        """Lower bound of the 95% interval. The compared quantity."""
        return self.interval[0]

    @property
    def upper(self) -> float:
        """Upper bound of the 95% interval."""
        return self.interval[1]


@dataclass(frozen=True)
class CostEvidence:
    """Evidence for the cost dimension: observed per-case costs.

    The interval is mean +/- 1.96 * standard error (normal approximation;
    no t correction, so small-n intervals understate uncertainty — see the
    module docstring). With a single observation the standard error is
    undefined, so the interval collapses to the point estimate; ``decide``
    treats that as no measured uncertainty and INVESTIGATEs, it never
    GATEs on it.

    This record carries no budget: the budget is a policy judgment, so it
    lives in RiskThresholds and is passed in by ``decide``. Evidence that
    carried its own budget could be judged against a different number
    than the one the Decision record reports.
    """

    mean: float
    std: float
    n: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("cost evidence needs n >= 1")
        if not math.isfinite(self.mean) or not math.isfinite(self.std):
            raise ValueError("cost evidence needs finite mean and std")
        if self.mean < 0.0 or self.std < 0.0:
            raise ValueError("cost evidence needs mean >= 0, std >= 0")

    @property
    def dimension(self) -> Dimension:
        """Cost evidence always measures the cost dimension."""
        return Dimension.COST

    @property
    def point(self) -> float:
        """Point estimate (the observed mean cost)."""
        return self.mean

    @property
    def interval(self) -> tuple[float, float]:
        """95% normal interval on the mean, floored at 0."""
        if self.n < 2:
            return (self.mean, self.mean)
        se = self.std / math.sqrt(self.n)
        return (max(0.0, self.mean - Z_95 * se), self.mean + Z_95 * se)

    @property
    def lower(self) -> float:
        """Lower bound of the 95% interval on the mean cost."""
        return self.interval[0]

    @property
    def upper(self) -> float:
        """Upper bound of the 95% interval. The compared quantity."""
        return self.interval[1]

    @classmethod
    def from_costs(cls, costs: list[float]) -> CostEvidence:
        """Build cost evidence from per-case cost observations."""
        if not costs:
            raise ValueError("cost evidence needs at least one observation")
        mean = statistics.fmean(costs)
        std = statistics.stdev(costs) if len(costs) >= 2 else 0.0
        return cls(mean=mean, std=std, n=len(costs))


@dataclass(frozen=True)
class UnmeasuredEvidence:
    """A dimension with no evidence. Never a pass: it forces INVESTIGATE."""

    dimension: Dimension
    reason: str


Evidence = ProportionEvidence | CostEvidence | UnmeasuredEvidence


@dataclass(frozen=True)
class RiskThresholds:
    """Thresholds at one risk level.

    For proportion dimensions the threshold is the minimum acceptable
    interval LOWER bound. For cost it is the maximum acceptable mean cost
    (``cost_budget``); the compared quantity is the interval UPPER bound.
    """

    quality: float
    safety: float
    reliability: float
    cost_budget: float

    def __post_init__(self) -> None:
        for name in ("quality", "safety", "reliability"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} threshold {value} must be in (0, 1]")
        if not math.isfinite(self.cost_budget) or self.cost_budget <= 0.0:
            raise ValueError(f"cost_budget {self.cost_budget} must be finite and > 0")

    def for_dimension(self, dimension: Dimension) -> float:
        """Threshold for one dimension (cost returns its budget)."""
        if dimension is Dimension.COST:
            return self.cost_budget
        if dimension is Dimension.QUALITY:
            return self.quality
        if dimension is Dimension.SAFETY:
            return self.safety
        if dimension is Dimension.RELIABILITY:
            return self.reliability
        raise ValueError(f"unknown dimension {dimension!r}")


def _default_thresholds() -> dict[RiskLevel, RiskThresholds]:
    """Sensible per-risk defaults. Higher risk means a stricter threshold.

    These are defaults, not doctrine: a team owns its thresholds and
    records them in its policy file. The one invariant this module
    enforces is monotonicity: a higher risk level never has a looser
    threshold than a lower one. The cost budget tightens with risk too:
    a higher-risk deployment gets a smaller spend ceiling, because a
    costly wrong decision hurts more there.
    """
    return {
        RiskLevel.LOW: RiskThresholds(
            quality=0.70, safety=0.80, reliability=0.70, cost_budget=200.0
        ),
        RiskLevel.MEDIUM: RiskThresholds(
            quality=0.80, safety=0.90, reliability=0.80, cost_budget=100.0
        ),
        RiskLevel.HIGH: RiskThresholds(
            quality=0.90, safety=0.95, reliability=0.90, cost_budget=50.0
        ),
        RiskLevel.CRITICAL: RiskThresholds(
            quality=0.95, safety=0.99, reliability=0.95, cost_budget=25.0
        ),
    }


def default_policy() -> DecisionPolicy:
    """A DecisionPolicy built from the per-risk defaults."""
    return DecisionPolicy(thresholds=_default_thresholds())


@dataclass(frozen=True)
class DecisionPolicy:
    """Per-risk thresholds for all four dimensions.

    Every risk level must be present, and thresholds must be monotone:
    a higher risk level never accepts a lower bar than a lower one.
    The mapping is copied and frozen at construction: a frozen dataclass
    blocks reassignment but not mutation, so the caller's dict must not
    stay aliased to the policy.
    """

    thresholds: Mapping[RiskLevel, RiskThresholds] = field(
        default_factory=_default_thresholds
    )

    def __post_init__(self) -> None:
        missing = [level for level in RiskLevel if level not in self.thresholds]
        if missing:
            raise ValueError(f"policy is missing risk levels: {[m.value for m in missing]}")
        ordered = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        for dimension in Dimension:
            values = [self.thresholds[level].for_dimension(dimension) for level in ordered]
            if dimension is Dimension.COST:
                if any(values[i] < values[i + 1] for i in range(len(values) - 1)):
                    raise ValueError("cost_budget must not loosen with risk")
            elif any(values[i] > values[i + 1] for i in range(len(values) - 1)):
                raise ValueError(f"{dimension.value} threshold must not loosen with risk")
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))

    def for_risk(self, risk: RiskLevel) -> RiskThresholds:
        """Thresholds at one risk level."""
        return self.thresholds[risk]


def policy_fingerprint(policy: DecisionPolicy) -> str:
    """Short stable hash of a policy's thresholds, for audit trails.

    Two policies with the same thresholds have the same fingerprint;
    a Decision report that records it can be checked against the policy
    file that produced it.
    """
    payload = {
        level.value: {
            "quality": policy.thresholds[level].quality,
            "safety": policy.thresholds[level].safety,
            "reliability": policy.thresholds[level].reliability,
            "cost_budget": policy.thresholds[level].cost_budget,
        }
        for level in RiskLevel
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _check_evidence_pairing(dimension: Dimension, evidence: Evidence) -> None:
    """Refuse evidence that does not belong to its dimension.

    A proportion record labelled with another dimension, or a cost
    record filed under quality, would be judged against the wrong
    threshold while the Decision record claimed otherwise.
    """
    if evidence.dimension is not dimension:
        raise ValueError(
            f"evidence for {evidence.dimension.value} filed under {dimension.value}"
        )
    if dimension is Dimension.COST:
        if not isinstance(evidence, (CostEvidence, UnmeasuredEvidence)):
            raise ValueError(f"cost dimension needs CostEvidence, got {type(evidence).__name__}")
    elif not isinstance(evidence, (ProportionEvidence, UnmeasuredEvidence)):
        raise ValueError(
            f"{dimension.value} dimension needs ProportionEvidence, "
            f"got {type(evidence).__name__}"
        )


@dataclass(frozen=True)
class DimensionVerdict:
    """The verdict for one dimension: its action, evidence, and reason."""

    dimension: Dimension
    action: DecisionAction
    evidence: Evidence
    threshold: float
    reason: str

    def __post_init__(self) -> None:
        _check_evidence_pairing(self.dimension, self.evidence)


@dataclass(frozen=True)
class Decision:
    """A complete evaluation decision: every dimension mapped to an action.

    The overall action is the worst across dimensions: any ROLLBACK
    dominates, else any INVESTIGATE dominates, else GATE.
    The verdict mapping is copied and frozen at construction, like the
    policy's: a frozen dataclass alone does not stop dict mutation.
    """

    action: DecisionAction
    risk: RiskLevel
    verdicts: Mapping[Dimension, DimensionVerdict]

    def __post_init__(self) -> None:
        missing = [d for d in Dimension if d not in self.verdicts]
        if missing:
            raise ValueError(
                f"decision is missing dimensions: {[m.value for m in missing]}"
            )
        for dimension, verdict in self.verdicts.items():
            if verdict.dimension is not dimension:
                raise ValueError(
                    f"verdict for {verdict.dimension.value} filed under {dimension.value}"
                )
            _check_evidence_pairing(dimension, verdict.evidence)
        worst = DecisionAction.GATE
        for verdict in self.verdicts.values():
            if verdict.action is DecisionAction.ROLLBACK:
                worst = DecisionAction.ROLLBACK
                break
            if verdict.action is DecisionAction.INVESTIGATE:
                worst = DecisionAction.INVESTIGATE
        if self.action is not worst:
            raise ValueError(
                f"overall action {self.action.value} is not the worst "
                f"per-dimension action {worst.value}"
            )
        object.__setattr__(self, "verdicts", MappingProxyType(dict(self.verdicts)))

    def summary(self) -> str:
        """Human-readable per-dimension table plus the overall action."""
        lines = [f"Decision: {self.action.value.upper()} (risk={self.risk.value})"]
        for dimension in Dimension:
            verdict = self.verdicts[dimension]
            evidence = verdict.evidence
            if isinstance(evidence, UnmeasuredEvidence):
                lines.append(f"  {dimension.value:12s} UNMEASURED  {verdict.reason}")
            else:
                sample = f" n={evidence.n}" if isinstance(evidence, CostEvidence) else (
                    f" n={evidence.trials}" if isinstance(evidence, ProportionEvidence) else ""
                )
                lines.append(
                    f"  {dimension.value:12s} {verdict.action.value:11s} "
                    f"point={evidence.point:.3f} "
                    f"95% CI=[{evidence.lower:.3f}, {evidence.upper:.3f}] "
                    f"threshold={verdict.threshold:.3f}{sample}  {verdict.reason}"
                )
        return "\n".join(lines)


def _judge_proportion(
    evidence: ProportionEvidence, threshold: float
) -> tuple[DecisionAction, str]:
    """Compare a proportion interval against its threshold."""
    if evidence.lower >= threshold:
        return DecisionAction.GATE, "interval clears threshold"
    if evidence.upper < threshold:
        return DecisionAction.ROLLBACK, "interval entirely below threshold"
    return DecisionAction.INVESTIGATE, "interval straddles threshold"


def _judge_cost(evidence: CostEvidence, budget: float) -> tuple[DecisionAction, str]:
    """Compare a cost interval against the POLICY budget (upper bound is the test).

    The budget is passed in by ``decide``; evidence never carries its own,
    so the judged number is always the one the Decision record reports.
    A single observation carries no measured uncertainty, so n < 2
    INVESTIGATEs instead of gating on a zero-width interval.
    """
    if evidence.n < 2:
        return (
            DecisionAction.INVESTIGATE,
            "single cost observation carries no measured uncertainty",
        )
    if evidence.upper <= budget:
        return DecisionAction.GATE, "interval within budget"
    if evidence.lower > budget:
        return DecisionAction.ROLLBACK, "interval entirely above budget"
    return DecisionAction.INVESTIGATE, "interval straddles budget"


def decide(
    evidence: dict[Dimension, Evidence],
    policy: DecisionPolicy,
    risk: RiskLevel,
) -> Decision:
    """Map evidence to a decision at one risk level.

    Pure function. Every dimension must be present in ``evidence``; a
    missing key raises instead of silently defaulting, because a decision
    over three dimensions is not a decision, it is a partial report.
    """
    missing = [d for d in Dimension if d not in evidence]
    if missing:
        raise ValueError(f"evidence is missing dimensions: {[m.value for m in missing]}")
    thresholds = policy.for_risk(risk)
    verdicts: dict[Dimension, DimensionVerdict] = {}
    for dimension in Dimension:
        item = evidence[dimension]
        _check_evidence_pairing(dimension, item)
        threshold = thresholds.for_dimension(dimension)
        if isinstance(item, UnmeasuredEvidence):
            action = DecisionAction.INVESTIGATE
            reason = f"unmeasured: {item.reason}"
        elif isinstance(item, CostEvidence):
            action, reason = _judge_cost(item, threshold)
        else:
            action, reason = _judge_proportion(item, threshold)
        verdicts[dimension] = DimensionVerdict(
            dimension=dimension,
            action=action,
            evidence=item,
            threshold=threshold,
            reason=reason,
        )
    worst = DecisionAction.GATE
    for verdict in verdicts.values():
        if verdict.action is DecisionAction.ROLLBACK:
            worst = DecisionAction.ROLLBACK
            break
        if verdict.action is DecisionAction.INVESTIGATE:
            worst = DecisionAction.INVESTIGATE
    return Decision(action=worst, risk=risk, verdicts=verdicts)
