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
Everything is stdlib-only and deterministic; there is no model call here.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum

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


# Dimensions a single eval case can be tagged with. Reliability is a
# cross-run property (agreement across seeded runs), and cost comes from a
# per-case cost function, so neither is a case tag.
CASE_DIMENSIONS: frozenset[str] = frozenset({Dimension.QUALITY.value, Dimension.SAFETY.value})


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
    """Evidence for the cost dimension: observed per-case costs vs a budget.

    The interval is mean +/- 1.96 * standard error. With a single
    observation the standard error is undefined, so the interval collapses
    to the point estimate and the record says so (n == 1): one cost sample
    carries no measured uncertainty.
    """

    mean: float
    std: float
    n: int
    budget: float

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("cost evidence needs n >= 1")
        if self.mean < 0.0 or self.std < 0.0 or self.budget <= 0.0:
            raise ValueError("cost evidence needs mean >= 0, std >= 0, budget > 0")

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
    def from_costs(cls, costs: list[float], budget: float) -> CostEvidence:
        """Build cost evidence from per-case cost observations."""
        if not costs:
            raise ValueError("cost evidence needs at least one observation")
        mean = statistics.fmean(costs)
        std = statistics.stdev(costs) if len(costs) >= 2 else 0.0
        return cls(mean=mean, std=std, n=len(costs), budget=budget)


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
        if self.cost_budget <= 0.0:
            raise ValueError(f"cost_budget {self.cost_budget} must be > 0")

    def for_dimension(self, dimension: Dimension) -> float:
        """Threshold for one dimension (cost returns its budget)."""
        if dimension is Dimension.COST:
            return self.cost_budget
        if dimension is Dimension.QUALITY:
            return self.quality
        if dimension is Dimension.SAFETY:
            return self.safety
        return self.reliability


def _default_thresholds() -> dict[RiskLevel, RiskThresholds]:
    """Sensible per-risk defaults. Higher risk means a stricter threshold.

    These are defaults, not doctrine: a team owns its thresholds and
    records them in its policy file. The one invariant this module
    enforces is monotonicity: a higher risk level never has a looser
    threshold than a lower one.
    """
    return {
        RiskLevel.LOW: RiskThresholds(
            quality=0.70, safety=0.80, reliability=0.70, cost_budget=100.0
        ),
        RiskLevel.MEDIUM: RiskThresholds(
            quality=0.80, safety=0.90, reliability=0.80, cost_budget=100.0
        ),
        RiskLevel.HIGH: RiskThresholds(
            quality=0.90, safety=0.95, reliability=0.90, cost_budget=100.0
        ),
        RiskLevel.CRITICAL: RiskThresholds(
            quality=0.95, safety=0.99, reliability=0.95, cost_budget=100.0
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
    """

    thresholds: dict[RiskLevel, RiskThresholds] = field(default_factory=_default_thresholds)

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

    def for_risk(self, risk: RiskLevel) -> RiskThresholds:
        """Thresholds at one risk level."""
        return self.thresholds[risk]


@dataclass(frozen=True)
class DimensionVerdict:
    """The verdict for one dimension: its action, evidence, and reason."""

    dimension: Dimension
    action: DecisionAction
    evidence: Evidence
    threshold: float
    reason: str


@dataclass(frozen=True)
class Decision:
    """A complete evaluation decision: every dimension mapped to an action.

    The overall action is the worst across dimensions: any ROLLBACK
    dominates, else any INVESTIGATE dominates, else GATE.
    """

    action: DecisionAction
    risk: RiskLevel
    verdicts: dict[Dimension, DimensionVerdict]

    def __post_init__(self) -> None:
        missing = [d for d in Dimension if d not in self.verdicts]
        if missing:
            raise ValueError(
                f"decision is missing dimensions: {[m.value for m in missing]}"
            )
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

    def summary(self) -> str:
        """Human-readable per-dimension table plus the overall action."""
        lines = [f"Decision: {self.action.value.upper()} (risk={self.risk.value})"]
        for dimension in Dimension:
            verdict = self.verdicts[dimension]
            evidence = verdict.evidence
            if isinstance(evidence, UnmeasuredEvidence):
                lines.append(f"  {dimension.value:12s} UNMEASURED  {verdict.reason}")
            else:
                lines.append(
                    f"  {dimension.value:12s} {verdict.action.value:11s} "
                    f"point={evidence.point:.3f} "
                    f"95% CI=[{evidence.lower:.3f}, {evidence.upper:.3f}] "
                    f"threshold={verdict.threshold:.3f}  {verdict.reason}"
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


def _judge_cost(evidence: CostEvidence) -> tuple[DecisionAction, str]:
    """Compare a cost interval against its budget (upper bound is the test)."""
    if evidence.upper <= evidence.budget:
        return DecisionAction.GATE, "interval within budget"
    if evidence.lower > evidence.budget:
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
        threshold = thresholds.for_dimension(dimension)
        if isinstance(item, UnmeasuredEvidence):
            action = DecisionAction.INVESTIGATE
            reason = f"unmeasured: {item.reason}"
        elif isinstance(item, CostEvidence):
            action, reason = _judge_cost(item)
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
