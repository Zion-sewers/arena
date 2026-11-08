"""Version promotion gate: evidence gain, not accuracy gain (arena#51).

Source: Roberts/Yaida, "The Principles of Deep Learning Theory" (catalog
G.24). Promotion of a challenger version over the champion requires
improvement in Bayesian evidence, not just benchmark accuracy. Two
consequences follow, and this module enforces both as typed rules:

* The compared quantity is the log-evidence gain, log p(D|challenger) -
  log p(D|champion), judged by its interval lower bound against a policy
  margin. An accuracy gain never promotes on its own: a challenger that
  scores higher on the benchmark while its evidence does not clear the
  margin is REJECTED, with the accuracy gain named in the reason.
* Complexity is reported as r = L/n (depth over width, the microscopic
  complexity of Roberts/Yaida) for both versions. The evidence already
  carries the complexity penalty — the marginal likelihood integrates
  over parameters, so a more complex version must earn its keep in
  evidence — which is why this gate reports r as an audit trail instead
  of inventing a penalty coefficient: no coefficient in the literature
  maps a unit of r to a unit of log-evidence, and a made-up one would be
  a number with no provenance.

Log-evidence is a MEASURED input here, supplied by the caller's
training/eval pipeline (e.g. a Laplace-approximated log marginal
likelihood on a held-out split), exactly as successes/trials are
supplied to decision.py: this module is the decision discipline over
measured evidence, not an estimator. Unmeasured evidence never
promotes: a version with no log-evidence measurement HOLDs, however
high its benchmark accuracy. Evidence from different splits is
incomparable and refused outright — a gate that compared numbers
measured on different data would be grading the split, not the version.

The gain interval uses independent standard errors,
SE_diff = sqrt(se_challenger^2 + se_champion^2). Evidence for two
versions is usually estimated on the same split, so the errors are in
fact correlated; correlation would only NARROW the interval, so the
independent-SE interval is conservative — it is harder, never easier,
to clear the margin. That is the fail-closed direction for a gate.

The margin is a policy choice, default 0.0 (any positive evidence gain,
after accounting for measurement error, promotes). Teams that want a
stronger bar can use the Jeffreys scale for the log Bayes factor:
1 = substantial, 3 = strong, 5 = decisive (established).

This is a version-promotion gate (champion vs challenger). It is not
the failure-promotion of regression.py (promoting production failures
into benchmarks) — a different concern that happens to share a word.

Everything is stdlib-only and deterministic; there is no model call here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from agenteval_bench.decision import wilson_interval

Z_95 = 1.96


def _check_int(value: int, field_name: str) -> int:
    """One integer field: a real int, never a bool, never a float."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {value!r}")
    return value


def _check_finite(value: float, field_name: str) -> float:
    """One float field: a finite number, never a bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return number


@dataclass(frozen=True)
class VersionSpec:
    """Identity + architecture of one version: name, depth L, width n."""

    name: str
    depth: int  # L: number of layers
    width: int  # n: hidden width

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("version name must be a non-empty string")
        _check_int(self.depth, "depth")
        _check_int(self.width, "width")
        if self.depth < 1:
            raise ValueError(f"depth L must be >= 1, got {self.depth}")
        if self.width < 1:
            raise ValueError(f"width n must be >= 1, got {self.width}")

    @property
    def complexity(self) -> float:
        """Microscopic complexity r = L/n (Roberts/Yaida)."""
        return self.depth / self.width


@dataclass(frozen=True)
class VersionEvidence:
    """Measured evidence for one version, on one held-out split.

    ``log_evidence`` is the log marginal likelihood log p(D|version)
    on ``evidence_split``, with its standard error. Both are None when
    the evidence was never measured: an unmeasured version can never
    promote, however high its benchmark accuracy. Accuracy is the
    benchmark pass count (k successes in n trials) — reported for the
    record, never sufficient for promotion.
    """

    version: VersionSpec
    log_evidence: float | None
    log_evidence_se: float | None
    accuracy_successes: int
    accuracy_trials: int
    evidence_split: str

    def __post_init__(self) -> None:
        if (self.log_evidence is None) != (self.log_evidence_se is None):
            raise ValueError(
                "log_evidence and log_evidence_se must be set together or both be None (unmeasured)"
            )
        if self.log_evidence is not None:
            _check_finite(self.log_evidence, "log_evidence")
        if self.log_evidence_se is not None:
            se = _check_finite(self.log_evidence_se, "log_evidence_se")
            if se < 0.0:
                raise ValueError(f"log_evidence_se must be >= 0, got {se}")
        if not isinstance(self.evidence_split, str) or not self.evidence_split:
            raise ValueError("evidence_split must be a non-empty string")
        # Validates the accuracy counts (raises on empty sample or
        # out-of-range successes); the interval itself is computed on demand.
        wilson_interval(self.accuracy_successes, self.accuracy_trials)

    @property
    def evidence_measured(self) -> bool:
        """True when a log-evidence measurement exists for this version."""
        return self.log_evidence is not None

    @property
    def accuracy(self) -> float:
        """Benchmark accuracy point estimate."""
        return self.accuracy_successes / self.accuracy_trials

    @property
    def accuracy_interval(self) -> tuple[float, float]:
        """95% Wilson score interval on the benchmark accuracy."""
        return wilson_interval(self.accuracy_successes, self.accuracy_trials)


@dataclass(frozen=True)
class PromotionPolicy:
    """The promotion bar: minimum log-evidence gain (default 0.0).

    The gain is judged by its interval LOWER bound, so the margin is
    cleared only when the gain survives measurement error. Jeffreys
    scale for the log Bayes factor: 1 substantial, 3 strong, 5 decisive.
    """

    evidence_margin: float = 0.0

    def __post_init__(self) -> None:
        margin = _check_finite(self.evidence_margin, "evidence_margin")
        if margin < 0.0:
            raise ValueError(f"evidence_margin must be >= 0, got {margin}")


class PromotionAction(str, Enum):
    """The closed set of actions a promotion gate can produce.

    PROMOTE: evidence gain clears the margin. HOLD: evidence gain is
    inconclusive or unmeasured — collect evidence, do not promote.
    REJECT: evidence gain is absent — an accuracy gain does not promote.
    """

    PROMOTE = "promote"
    HOLD = "hold"
    REJECT = "reject"


@dataclass(frozen=True)
class PromotionVerdict:
    """The verdict of one promotion gate: action, numbers, and reason.

    ``evidence_gain`` and ``gain_interval`` are None when either
    version's evidence is unmeasured — there is no gain to report, and
    the reason says so. Accuracy gain is always reported: a REJECT
    names the accuracy gain it refused to promote on, so the refusal
    is auditable instead of silent.
    """

    action: PromotionAction
    champion: VersionEvidence
    candidate: VersionEvidence
    evidence_gain: float | None
    gain_interval: tuple[float, float] | None
    accuracy_gain: float
    reason: str

    def summary(self) -> str:
        """Human-readable verdict: action, evidence gain, accuracy, r=L/n."""
        champ = self.champion
        cand = self.candidate
        lines = [
            (
                f"Promotion: {self.action.value.upper()} "
                f"({cand.version.name} vs champion {champ.version.name})"
            ),
            (
                f"  complexity r=L/n: champion {champ.version.complexity:.4f} "
                f"(L={champ.version.depth}, n={champ.version.width}) vs "
                f"candidate {cand.version.complexity:.4f} "
                f"(L={cand.version.depth}, n={cand.version.width})"
            ),
        ]
        if self.evidence_gain is None or self.gain_interval is None:
            lines.append("  evidence gain: UNMEASURED")
        else:
            lo, hi = self.gain_interval
            lines.append(
                f"  evidence gain: {self.evidence_gain:+.3f} 95% CI=[{lo:+.3f}, {hi:+.3f}]"
            )
        lines.append(
            f"  accuracy gain: {self.accuracy_gain:+.3f} "
            f"(champion {champ.accuracy:.3f} -> "
            f"candidate {cand.accuracy:.3f})"
        )
        lines.append(f"  reason: {self.reason}")
        return "\n".join(lines)


def promote(
    champion: VersionEvidence,
    candidate: VersionEvidence,
    policy: PromotionPolicy | None = None,
) -> PromotionVerdict:
    """Gate a challenger version against the champion.

    Pure function. PROMOTE iff both evidences are measured, on the same
    split, and the lower bound of the evidence-gain interval clears the
    policy margin. Accuracy is reported but never sufficient: an
    accuracy gain with no evidence gain is REJECTED, and unmeasured
    evidence HOLDs however high the accuracy.
    """
    active_policy = policy if policy is not None else PromotionPolicy()
    if champion.version.name == candidate.version.name:
        raise ValueError(f"cannot promote {candidate.version.name!r} against itself")
    if champion.evidence_split != candidate.evidence_split:
        raise ValueError(
            "evidence splits differ "
            f"({champion.evidence_split!r} vs {candidate.evidence_split!r}): "
            "evidence measured on different data is incomparable"
        )
    accuracy_gain = candidate.accuracy - champion.accuracy
    champion_evidence = champion.log_evidence
    candidate_evidence = candidate.log_evidence
    champion_se = champion.log_evidence_se
    candidate_se = candidate.log_evidence_se
    if (
        champion_evidence is None
        or candidate_evidence is None
        or champion_se is None
        or candidate_se is None
    ):
        return PromotionVerdict(
            action=PromotionAction.HOLD,
            champion=champion,
            candidate=candidate,
            evidence_gain=None,
            gain_interval=None,
            accuracy_gain=accuracy_gain,
            reason=(
                "evidence unmeasured: promotion requires measured "
                "evidence gain; accuracy-only promotion is not allowed"
            ),
        )
    gain = candidate_evidence - champion_evidence
    gain_se = math.sqrt(champion_se * champion_se + candidate_se * candidate_se)
    gain_interval = (gain - Z_95 * gain_se, gain + Z_95 * gain_se)
    margin = active_policy.evidence_margin
    if gain_interval[0] > margin:
        return PromotionVerdict(
            action=PromotionAction.PROMOTE,
            champion=champion,
            candidate=candidate,
            evidence_gain=gain,
            gain_interval=gain_interval,
            accuracy_gain=accuracy_gain,
            reason=(
                f"evidence gain interval lower bound {gain_interval[0]:+.3f} "
                f"clears margin {margin:.3f}"
            ),
        )
    if gain_interval[1] > margin:
        return PromotionVerdict(
            action=PromotionAction.HOLD,
            champion=champion,
            candidate=candidate,
            evidence_gain=gain,
            gain_interval=gain_interval,
            accuracy_gain=accuracy_gain,
            reason=(
                f"evidence gain interval [{gain_interval[0]:+.3f}, "
                f"{gain_interval[1]:+.3f}] straddles margin {margin:.3f}: "
                "collect more evidence"
            ),
        )
    return PromotionVerdict(
        action=PromotionAction.REJECT,
        champion=champion,
        candidate=candidate,
        evidence_gain=gain,
        gain_interval=gain_interval,
        accuracy_gain=accuracy_gain,
        reason=(
            f"evidence gain interval [{gain_interval[0]:+.3f}, "
            f"{gain_interval[1]:+.3f}] does not clear margin {margin:.3f}; "
            f"accuracy gain {accuracy_gain:+.3f} does not promote"
        ),
    )
