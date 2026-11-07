"""Agent change = controlled experiment: OEC + guardrails + trust gates.

Implements agenteval-bench#46 (Kohavi, Tang & Xu, *Trustworthy Online
Controlled Experiments*): every agent change ships as a pre-registered
experiment whose Overall Evaluation Criterion, goal/driver/guardrail
metric roles, and trust tests are declared BEFORE the change, and whose
ship decision is a pure function over reported evidence.

Rules (from the issue's acceptance criteria, executable form):
* OEC + guardrails are pre-registered before the change — the design
  names exactly one OEC metric and every guardrail with its max allowed
  regression, bound to a registered plan digest, AND the design itself
  is registered in the hash chain (``register_design``) before the run.
  Evidence naming a different OEC, or a design rewritten after
  registration, cannot ship (OEC_MISMATCH / DESIGN_NOT_REGISTERED):
  the ship gate matches the supplied design against its registered
  record by canonical content, so widening a guardrail budget or
  moving the surprise threshold after results are known matches no
  record. Amendment rule: the chain is append-only, so correcting a
  design means registering a NEW design record; the old record stays,
  and the ship verdict records which design digest was used.
* SRM / A/A run before results are read — the ship gate evaluates the
  sample-ratio-mismatch test, the A/A null-split test, and the guardrail
  checks BEFORE it examines the OEC win. Trust-test code is ordered
  that way in :func:`ship_decision`, and a test pins the order: a
  trust-failing experiment with a huge OEC win reports the trust
  failure, never the win.
* Surprising wins require replication — Twyman's law: a final z beyond
  the pre-registered surprise threshold (|z| > ``surprise_z``) ships
  only with confirming replication evidence (same direction,
  significant at alpha, fresh run_id, minimum n). Without it the
  verdict is NEEDS_REPLICATION, not SHIP.
* Shipping on a single exciting number is NOT allowed — SHIP requires
  all of: a registered plan whose discipline verification passes, a
  registered design, a significant OEC win in the declared direction,
  passing SRM and A/A, unregressed guardrails, and (for surprising
  wins) replication. Anything else is NO_SHIP or NEEDS_REPLICATION,
  with named reasons.
* Metric provenance — every reported look names the metric its z is
  computed on, and the gate refuses a win whose looks do not name the
  design's OEC metric (OEC_METRIC_UNPROVEN). A z with no metric
  behind it cannot ship as an OEC win.

Measurement discipline (same as validity.py / decision.py / #45): the
grade rests on interval BOUNDS, never point estimates. A guardrail
holds only when the LOWER bound of the good-direction delta clears the
pre-registered max regression; SRM and A/A are p-value tests against
the registered alpha, not eyeballed counts. A trust test with too
little data is INSUFFICIENT, never a pass. Guardrail bounds are
Newcombe intervals (Wilson-based, score method), not Wald: near
p = 0/1 the Wald interval collapses to zero width and a guardrail
with zero observed events passes on one sample, so the gate uses the
Newcombe difference interval, which fails closed on thin data.

Everything is stdlib-only and deterministic. The module makes no
network call: arena has no live experiment transport, so the design
and the evidence are REPORTED records supplied by the caller (JSON for
the ``ship`` CLI, dataclasses in-process). The verifier is a
consistency checker over reported evidence — the same honesty boundary
as #32's registry: it binds the plan and re-derives every decision
from the reported z's and counts, but it cannot see raw data, so a
fabricated-but-internally-consistent report verifies clean. What the
registry binds is the plan (immutably, via the hash chain), not the
data. Replication independence is likewise workflow-bound: the gate
checks significance, direction, minimum n, and a distinct run_id, but
a replication run_id minted from the same data verifies clean — the
defense is workflow (the replication run_id must come from a
separately recorded run), not cryptography.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from experiment.alpha_spend import group_sequential_boundaries, inv_normal_cdf
from experiment.registry import (
    ExperimentEvidence,
    LookEvidence,
    RegistryCorrupted,
    RegistryMalformed,
    find_design,
    load_design,
    load_plan,
    verify_experiment,
)
from experiment.registry import (
    register_design as _register_design_record,
)

#: SRM alert level (Kohavi convention): a sample-ratio mismatch is
#: flagged when the chi-square p-value falls below 0.001, not 0.05 —
#: with thousands of assignments, a 5% level would cry wolf on every
#: routine imbalance.
SRM_ALPHA = 0.001

#: A chi-square cell with fewer than 5 expected assignments carries no
#: usable SRM signal; the verdict is INSUFFICIENT, never a pass.
SRM_MIN_EXPECTED = 5.0

#: The closed set of ship verdicts. NEEDS_REPLICATION is distinct from
#: NO_SHIP: the experiment is clean and the win is real, but Twyman's
#: law demands a second look before it ships.
ShipStatus = Literal["ship", "no_ship", "needs_replication"]

#: The closed set of trust-test verdicts. INSUFFICIENT means "too
#: little data to run the test" — a weak trust test is not a clean one.
TrustStatus = Literal["pass", "fail", "insufficient"]

#: Metric roles. Exactly one metric per design is the OEC; goal and
#: driver metrics support the decision; guardrails must not regress.
MetricRole = Literal["oec", "goal", "driver", "guardrail"]

#: Which direction of a metric is good. The OEC win, the guardrail
#: delta, and the replication direction are all read in this direction,
#: never assumed.
MetricDirection = Literal["higher_better", "lower_better"]


class ShipRefused(ValueError):
    """The ship gate refused to ship (``check_ship`` raising form).

    A ValueError subclass so callers that already guard on value errors
    fail closed the same way. NEEDS_REPLICATION also raises here: the
    change may not ship yet, and the reasons name the replication that
    would unblock it.
    """


def _is_int(value: object) -> bool:
    """True for real ints (bools are ints in Python, and are refused)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    """True for real numbers (bools refused)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class MetricDeclaration:
    """One metric's pre-registered role in the experiment.

    ``max_regression`` is the largest good-direction drop the metric
    may suffer (absolute delta, >= 0). Only guardrails carry one: a
    decision metric (OEC / goal / driver) with a regression budget is a
    contradiction in terms, and is refused at construction.
    """

    name: str
    role: MetricRole
    direction: MetricDirection
    max_regression: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name must be non-empty")
        if self.role == "guardrail":
            if self.max_regression is None:
                raise ValueError(
                    f"guardrail {self.name!r} needs a pre-registered "
                    "max_regression: a guardrail without a budget cannot "
                    "be checked"
                )
            if not math.isfinite(self.max_regression) or self.max_regression < 0:
                raise ValueError(
                    f"guardrail {self.name!r} max_regression must be a "
                    f"finite number >= 0, got {self.max_regression!r}"
                )
        elif self.max_regression is not None:
            raise ValueError(
                f"{self.role} metric {self.name!r} must not carry a "
                "max_regression: only guardrails have regression budgets"
            )


@dataclass(frozen=True)
class ExperimentDesign:
    """The pre-registered design of one agent-change experiment.

    ``plan_digest`` binds the design to a registered
    :class:`~experiment.types.ExperimentPlan` (n, alpha, look schedule,
    covariate) in the hash-chained registry — the OEC and guardrails
    are registered alongside that plan, before the change. ``metrics``
    names exactly one OEC. ``expected_allocation`` is the randomization
    split the SRM test checks against (e.g. (0.5, 0.5)). ``surprise_z``
    is the Twyman threshold: a final |z| beyond it is a surprising win
    and needs replication. ``min_aa_n`` is the per-arm floor below
    which the A/A trust test has no power; ``min_replication_n`` is
    the floor for a replication run.
    """

    plan_digest: str
    metrics: tuple[MetricDeclaration, ...]
    expected_allocation: tuple[float, ...]
    surprise_z: float = 3.0
    alpha: float = 0.05
    min_aa_n: int = 100
    min_replication_n: int = 30

    def __post_init__(self) -> None:
        if not self.plan_digest:
            raise ValueError("design plan_digest must be non-empty")
        if not self.metrics:
            raise ValueError("design needs at least one metric")
        names = [m.name for m in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError(f"design metric names must be unique, got {names}")
        oecs = [m for m in self.metrics if m.role == "oec"]
        if len(oecs) != 1:
            raise ValueError(f"design needs exactly one OEC metric, got {len(oecs)}")
        if len(self.expected_allocation) < 2:
            raise ValueError(
                f"expected_allocation needs at least 2 arms, got {self.expected_allocation}"
            )
        for share in self.expected_allocation:
            if not math.isfinite(share) or not 0.0 < share < 1.0:
                raise ValueError(f"allocation shares must be in (0, 1), got {share!r}")
        if not math.isclose(sum(self.expected_allocation), 1.0, abs_tol=1e-9):
            raise ValueError(
                f"expected_allocation must sum to 1.0, got {sum(self.expected_allocation)}"
            )
        if not math.isfinite(self.surprise_z) or self.surprise_z <= 0:
            raise ValueError(f"surprise_z must be finite and > 0, got {self.surprise_z!r}")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.min_aa_n < 1:
            raise ValueError(f"min_aa_n must be >= 1, got {self.min_aa_n}")
        if self.min_replication_n < 1:
            raise ValueError(f"min_replication_n must be >= 1, got {self.min_replication_n}")

    @property
    def oec(self) -> MetricDeclaration:
        """The design's single OEC metric."""
        return next(m for m in self.metrics if m.role == "oec")

    @property
    def guardrails(self) -> tuple[MetricDeclaration, ...]:
        """The design's guardrail metrics, in declaration order."""
        return tuple(m for m in self.metrics if m.role == "guardrail")

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form of the design."""
        return {
            "plan_digest": self.plan_digest,
            "metrics": [
                {
                    "name": m.name,
                    "role": m.role,
                    "direction": m.direction,
                    "max_regression": m.max_regression,
                }
                for m in self.metrics
            ],
            "expected_allocation": list(self.expected_allocation),
            "surprise_z": self.surprise_z,
            "alpha": self.alpha,
            "min_aa_n": self.min_aa_n,
            "min_replication_n": self.min_replication_n,
        }


@dataclass(frozen=True)
class MetricObservation:
    """One metric's reported arm counts (a proportion metric).

    Proportion metrics only for v1 — rates are the repo's lingua franca
    (decision.py ProportionEvidence, simpson strata). A continuous
    metric needs a mean/variance evidence type this module does not own,
    and is refused by the JSON loader rather than approximated.
    """

    name: str
    control_successes: int
    control_n: int
    treated_successes: int
    treated_n: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("observation name must be non-empty")
        for label, value in (
            ("control_successes", self.control_successes),
            ("control_n", self.control_n),
            ("treated_successes", self.treated_successes),
            ("treated_n", self.treated_n),
        ):
            if not _is_int(value):
                raise TypeError(f"observation {label} must be an int, got {value!r}")
        if self.control_n < 1 or self.treated_n < 1:
            raise ValueError(
                f"observation {self.name!r} needs n >= 1 per arm, got "
                f"control_n={self.control_n}, treated_n={self.treated_n}"
            )
        if not 0 <= self.control_successes <= self.control_n:
            raise ValueError(
                f"observation {self.name!r} control_successes "
                f"{self.control_successes} out of range for n={self.control_n}"
            )
        if not 0 <= self.treated_successes <= self.treated_n:
            raise ValueError(
                f"observation {self.name!r} treated_successes "
                f"{self.treated_successes} out of range for n={self.treated_n}"
            )

    @property
    def control_rate(self) -> float:
        """Observed control rate, derived from the counts."""
        return self.control_successes / self.control_n

    @property
    def treated_rate(self) -> float:
        """Observed treated rate, derived from the counts."""
        return self.treated_successes / self.treated_n


@dataclass(frozen=True)
class AAEvidence:
    """The A/A null split: the same agent on both sides.

    Both arms run the CONTROL (unchanged) agent, split by the same
    randomizer as the experiment. Any significant difference is a
    broken randomizer or a leaky pipeline, not a real effect — so a
    significant A/A fails the ship gate outright.
    """

    a_successes: int
    a_n: int
    b_successes: int
    b_n: int

    def __post_init__(self) -> None:
        for label, value in (
            ("a_successes", self.a_successes),
            ("a_n", self.a_n),
            ("b_successes", self.b_successes),
            ("b_n", self.b_n),
        ):
            if not _is_int(value):
                raise TypeError(f"A/A {label} must be an int, got {value!r}")
        if self.a_n < 1 or self.b_n < 1:
            raise ValueError(f"A/A needs n >= 1 per arm, got a_n={self.a_n}, b_n={self.b_n}")
        if not 0 <= self.a_successes <= self.a_n:
            raise ValueError(f"A/A a_successes {self.a_successes} out of range for n={self.a_n}")
        if not 0 <= self.b_successes <= self.b_n:
            raise ValueError(f"A/A b_successes {self.b_successes} out of range for n={self.b_n}")


@dataclass(frozen=True)
class ReplicationEvidence:
    """One replication run of a surprising win, from fresh data.

    ``z`` is the replication's final z-statistic (same sign convention
    as the experiment: positive means treated higher). ``run_id`` must
    differ from the original experiment's run_id — a replication that
    re-reports the same run is not a replication.
    """

    z: float
    n: int
    run_id: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.z):
            raise ValueError(f"replication z must be finite, got {self.z!r}")
        if not _is_int(self.n) or self.n < 1:
            raise ValueError(f"replication n must be an int >= 1, got {self.n!r}")
        if not self.run_id:
            raise ValueError("replication run_id must be non-empty")


@dataclass(frozen=True)
class ShipEvidence:
    """Everything the ship gate is asked to rule on.

    ``oec_metric`` names the metric the reported final z belongs to —
    it must equal the design's declared OEC, or the evidence is
    shipping on a metric the design never named OEC. ``srm_counts``
    are the observed per-arm assignment counts, in the same order as
    the design's expected allocation. ``aa`` may be None (then the
    gate refuses: an experiment with no A/A ran no trust test).
    ``guardrails`` carries one observation per declared guardrail;
    observations for undeclared metrics are refused, not ignored.
    ``secondary`` carries one observation per declared goal/driver
    metric: goal and driver metrics do not gate the ship, but a design
    that declares them must report them — a pre-registered design
    that never reports its declared metrics is an incomplete report,
    and the gate refuses it. Their good-direction deltas are reported
    in the verdict notes. ``replication`` is None unless the win was
    surprising.
    """

    plan_digest: str
    oec_metric: str
    run_id: str
    experiment: ExperimentEvidence
    srm_counts: tuple[int, ...]
    aa: AAEvidence | None
    guardrails: tuple[MetricObservation, ...] = ()
    secondary: tuple[MetricObservation, ...] = ()
    replication: ReplicationEvidence | None = None

    def __post_init__(self) -> None:
        if not self.plan_digest:
            raise ValueError("evidence plan_digest must be non-empty")
        if not self.oec_metric:
            raise ValueError("evidence oec_metric must be non-empty")
        if not self.run_id:
            raise ValueError("evidence run_id must be non-empty")
        if not self.srm_counts:
            raise ValueError("evidence needs srm_counts for at least 1 arm")
        for count in self.srm_counts:
            if not _is_int(count) or count < 0:
                raise TypeError(f"srm_counts must be non-negative ints, got {count!r}")
        names = [o.name for o in self.guardrails]
        if len(set(names)) != len(names):
            raise ValueError(f"guardrail observations must name each metric once, got {names}")
        secondary_names = [o.name for o in self.secondary]
        if len(set(secondary_names)) != len(secondary_names):
            raise ValueError(
                f"secondary observations must name each metric once, got {secondary_names}"
            )
        if set(names) & set(secondary_names):
            raise ValueError(
                "a metric cannot be both a guardrail and a secondary observation: "
                f"{sorted(set(names) & set(secondary_names))}"
            )


# ---------------------------------------------------------------------------
# Statistics helpers (stdlib only)
# ---------------------------------------------------------------------------


def _gser(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) by series expansion."""
    if x == 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) by continued fraction."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(chi2: float, df: int) -> float:
    """Survival function of the chi-square distribution (stdlib only).

    P(X > chi2) for X ~ chi^2(df). Validated against known values
    (df=1 via erfc, df=2 = exp(-x/2), df=3 spot values) in the tests.
    """
    if not math.isfinite(chi2) or chi2 < 0:
        raise ValueError(f"chi2 must be finite and >= 0, got {chi2!r}")
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    if chi2 == 0.0:
        return 1.0
    a = df / 2.0
    x = chi2 / 2.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """Pooled two-proportion z-statistic for (k1/n1) vs (k2/n2).

    Positive z means the FIRST proportion is higher. Raises on
    degenerate arms (n < 1, counts out of range). When the pooled rate
    is 0 or 1 both arms are identical, so z is 0.0 — no difference to
    detect — rather than a division by zero.
    """
    for label, k, n in (("k1", k1, n1), ("k2", k2, n2)):
        if not _is_int(k) or not _is_int(n):
            raise TypeError(f"{label}/n must be ints, got {k!r}/{n!r}")
        if n < 1:
            raise ValueError(f"two_proportion_z needs n >= 1, got n={n}")
        if not 0 <= k <= n:
            raise ValueError(f"{label}={k} out of range for n={n}")
    pooled = (k1 + k2) / (n1 + n2)
    if pooled <= 0.0 or pooled >= 1.0:
        return 0.0
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    return (k1 / n1 - k2 / n2) / se


def two_sided_p(z: float) -> float:
    """Two-sided p-value for a standard-normal z-statistic."""
    if not math.isfinite(z):
        raise ValueError(f"z must be finite, got {z!r}")
    return math.erfc(abs(z) / math.sqrt(2.0))


def _wilson_interval(successes: int, trials: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (established).

    A local copy of :func:`agenteval_bench.decision.wilson_interval`:
    the experiment package stays stdlib-only and must not import the
    app layer (`agenteval_bench/__init__` eagerly pulls in the engine,
    scorers, and judges). The formula is identical, and a test pins
    agreement between the two across a grid, so a drift in either is
    caught. Returns (lower, upper), clamped to [0, 1].
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


def newcombe_interval(
    control_successes: int,
    control_n: int,
    treated_successes: int,
    treated_n: int,
    alpha: float,
) -> tuple[float, float]:
    """Newcombe interval for (treated rate - control rate).

    Method 10 (Newcombe 1998): the difference interval built from the
    Wilson score interval of each arm,
    ``lower = delta - sqrt((pt - lt)^2 + (uc - pc)^2)`` and symmetrically
    for the upper bound. The Wilson intervals are taken at the
    one-sided (1 - alpha) level, so the lower bound is the one-sided
    (1 - alpha) bound on the difference. Unlike the Wald interval this
    one does not collapse to a point at zero events: 0/1 vs 0/1 has a
    lower bound near -0.73, not exactly 0.0, so a guardrail measured on
    trivial data fails instead of passing.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"newcombe alpha must be in (0, 1), got {alpha}")
    z = inv_normal_cdf(1.0 - alpha)
    pc = control_successes / control_n
    pt = treated_successes / treated_n
    lower_c, upper_c = _wilson_interval(control_successes, control_n, z)
    lower_t, upper_t = _wilson_interval(treated_successes, treated_n, z)
    delta = pt - pc
    lower = delta - math.sqrt((pt - lower_t) ** 2 + (upper_c - pc) ** 2)
    upper = delta + math.sqrt((upper_t - pt) ** 2 + (pc - lower_c) ** 2)
    return (lower, upper)


# ---------------------------------------------------------------------------
# Trust tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SrmVerdict:
    """SRM verdict: observed allocation vs the registered split."""

    status: TrustStatus
    chi2: float | None
    df: int | None
    p_value: float | None
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form of the verdict."""
        return {
            "status": self.status,
            "chi2": self.chi2,
            "df": self.df,
            "p_value": self.p_value,
            "reasons": list(self.reasons),
            "notes": list(self.notes),
        }


def check_srm(
    observed_counts: tuple[int, ...],
    expected_shares: tuple[float, ...],
    alpha: float = SRM_ALPHA,
) -> SrmVerdict:
    """Sample-ratio-mismatch test (Kohavi ch. 5 trust test).

    Chi-square goodness-of-fit of the observed per-arm assignment
    counts against the registered allocation shares. FAIL when the
    p-value falls below ``alpha`` (default 0.001): the randomization
    is broken, and nothing downstream may be interpreted. INSUFFICIENT
    when any arm's expected count is below 5 or the total is 0 — a
    trust test with no power is not a pass.
    """
    if len(observed_counts) != len(expected_shares):
        raise ValueError(
            f"SRM needs one count per expected share: got "
            f"{len(observed_counts)} counts for {len(expected_shares)} shares"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"srm alpha must be in (0, 1), got {alpha}")
    total = sum(observed_counts)
    df = len(observed_counts) - 1
    if total == 0:
        return SrmVerdict(
            status="insufficient",
            chi2=None,
            df=df,
            p_value=None,
            reasons=("no assignments observed: SRM cannot be tested",),
        )
    expected = [total * share for share in expected_shares]
    thin = [(i, e) for i, e in enumerate(expected) if e < SRM_MIN_EXPECTED]
    if thin:
        return SrmVerdict(
            status="insufficient",
            chi2=None,
            df=df,
            p_value=None,
            reasons=(
                "expected count below 5 in arm(s) "
                + ", ".join(f"{i} (expected {e:.1f})" for i, e in thin)
                + ": SRM has no power at this n",
            ),
        )
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed_counts, expected))
    p_value = chi2_sf(chi2, df)
    if p_value < alpha:
        return SrmVerdict(
            status="fail",
            chi2=chi2,
            df=df,
            p_value=p_value,
            reasons=(
                (
                    f"SRM FAILED: chi2={chi2:.2f} on {df} df, p={p_value:.2g} "
                    f"< alpha={alpha} — the randomization is broken; nothing "
                    "downstream may be interpreted"
                ),
            ),
        )
    return SrmVerdict(
        status="pass",
        chi2=chi2,
        df=df,
        p_value=p_value,
        notes=(f"SRM clean: chi2={chi2:.2f} on {df} df, p={p_value:.2g}",),
    )


@dataclass(frozen=True)
class AaVerdict:
    """A/A verdict: the null split must show no difference."""

    status: TrustStatus
    z: float | None
    p_value: float | None
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form of the verdict."""
        return {
            "status": self.status,
            "z": self.z,
            "p_value": self.p_value,
            "reasons": list(self.reasons),
            "notes": list(self.notes),
        }


def check_aa(evidence: AAEvidence, alpha: float, min_n: int) -> AaVerdict:
    """A/A trust test: the same agent on both sides must not differ.

    Two-proportion z on the null split, two-sided at ``alpha``. FAIL
    when the split is significantly different — the randomizer or the
    pipeline is broken. INSUFFICIENT when either arm is below
    ``min_n``: an A/A with no power is not a clean A/A.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"aa alpha must be in (0, 1), got {alpha}")
    if evidence.a_n < min_n or evidence.b_n < min_n:
        return AaVerdict(
            status="insufficient",
            z=None,
            p_value=None,
            reasons=(
                (
                    f"A/A arms below min_n={min_n} (a_n={evidence.a_n}, "
                    f"b_n={evidence.b_n}): the null split has no power"
                ),
            ),
        )
    z = two_proportion_z(
        evidence.a_successes,
        evidence.a_n,
        evidence.b_successes,
        evidence.b_n,
    )
    p_value = two_sided_p(z)
    critical = inv_normal_cdf(1.0 - alpha / 2.0)
    if abs(z) >= critical:
        return AaVerdict(
            status="fail",
            z=z,
            p_value=p_value,
            reasons=(
                (
                    f"A/A FAILED: null split differs significantly "
                    f"(z={z:.2f}, p={p_value:.2g} at alpha={alpha}) — the "
                    "randomizer or the pipeline is broken, not the agent"
                ),
            ),
        )
    return AaVerdict(
        status="pass",
        z=z,
        p_value=p_value,
        notes=(f"A/A clean: null split z={z:.2f}, p={p_value:.2g}",),
    )


@dataclass(frozen=True)
class GuardrailVerdict:
    """One guardrail's verdict: the lower bound clears the budget."""

    name: str
    passed: bool
    delta_good: float
    lower_bound: float
    max_regression: float
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form of the verdict."""
        return {
            "name": self.name,
            "passed": self.passed,
            "delta_good": self.delta_good,
            "lower_bound": self.lower_bound,
            "max_regression": self.max_regression,
            "reasons": list(self.reasons),
        }


def check_guardrail(
    declaration: MetricDeclaration,
    observation: MetricObservation,
    alpha: float,
) -> GuardrailVerdict:
    """Guardrail non-inferiority check, graded on the LOWER bound.

    The good-direction delta is (treated - control) for higher_better
    metrics, (control - treated) for lower_better ones, and the bound
    is the Newcombe interval for the difference of two proportions
    (Wilson score intervals per arm). The guardrail holds iff the
    one-sided (1 - alpha) LOWER bound on that delta clears the
    pre-registered max regression: ``lower >= -max_regression``. A
    point estimate above the budget with a lower bound below it FAILS
    — the repo's lower-bound discipline (#43/#45) applies to
    guardrails too, on the repo's own Wilson interval, not a Wald
    approximation: Wald intervals collapse to a point at zero events
    (0/1 vs 0/1 "passes" any budget), Newcombe intervals widen
    exactly when there is nothing to learn from, so thin data fails
    instead of passing.
    """
    if declaration.role != "guardrail":
        raise ValueError(
            f"check_guardrail needs a guardrail declaration, got "
            f"role {declaration.role!r} for {declaration.name!r}"
        )
    if declaration.name != observation.name:
        raise ValueError(
            f"guardrail observation {observation.name!r} does not match "
            f"declaration {declaration.name!r}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"guardrail alpha must be in (0, 1), got {alpha}")
    if declaration.max_regression is None:
        raise ValueError(
            f"guardrail {declaration.name!r} needs a max_regression: "
            "a guardrail without a budget cannot be checked"
        )
    lower_diff, upper_diff = newcombe_interval(
        observation.control_successes,
        observation.control_n,
        observation.treated_successes,
        observation.treated_n,
        alpha,
    )
    if declaration.direction == "higher_better":
        delta = observation.treated_rate - observation.control_rate
        lower = lower_diff
    else:
        delta = observation.control_rate - observation.treated_rate
        lower = -upper_diff
    passed = lower >= -declaration.max_regression
    reasons: tuple[str, ...] = ()
    if not passed:
        reasons = (
            (
                f"guardrail {declaration.name!r} REGRESSED: good-direction "
                f"delta {delta:+.3f} with lower bound {lower:+.3f} below the "
                f"pre-registered budget -{declaration.max_regression:.3f}"
            ),
        )
    return GuardrailVerdict(
        name=declaration.name,
        passed=passed,
        delta_good=delta,
        lower_bound=lower,
        max_regression=declaration.max_regression,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Ship gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShipVerdict:
    """The ship gate's verdict: SHIP, NO_SHIP, or NEEDS_REPLICATION.

    ``reasons`` names every failed condition (empty when SHIP);
    ``notes`` carry non-failing observations. The verdict embeds the
    trust-test verdicts so a NO_SHIP can be audited without re-running
    the gate. ``design_digest`` is the registered design that was
    used, so an audit can check it was the current design.
    """

    status: ShipStatus
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    srm: SrmVerdict | None = None
    aa: AaVerdict | None = None
    guardrails: tuple[GuardrailVerdict, ...] = ()
    oec_z: float | None = None
    surprising: bool = False
    plan_digest: str = ""
    design_digest: str = ""

    def __post_init__(self) -> None:
        if self.status == "ship" and self.reasons:
            raise ValueError("a SHIP verdict cannot carry reasons")
        if self.status in ("no_ship", "needs_replication") and not self.reasons:
            raise ValueError(f"a {self.status} verdict needs non-empty reasons")

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form of the verdict."""
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "notes": list(self.notes),
            "srm": self.srm.to_dict() if self.srm is not None else None,
            "aa": self.aa.to_dict() if self.aa is not None else None,
            "guardrails": [g.to_dict() for g in self.guardrails],
            "oec_z": self.oec_z,
            "surprising": self.surprising,
            "plan_digest": self.plan_digest,
            "design_digest": self.design_digest,
        }


def ship_decision(
    design: ExperimentDesign,
    evidence: ShipEvidence,
    registry_path: str,
) -> ShipVerdict:
    """Rule SHIP / NO_SHIP / NEEDS_REPLICATION for one agent change.

    Order is load-bearing, and is the issue's "trust gates before
    results are read" criterion in code:

    1. Design binding — the design itself must be registered in the
       hash-chained registry (``find_design`` on its canonical form).
       A caller-supplied design that is not in the chain grades
       nothing: rewriting a guardrail budget or a surprise threshold
       after results are known matches no record and is refused as
       DESIGN_NOT_REGISTERED. Then the plan is loaded and the
       experiment discipline verified through ``verify_experiment``.
    2. Design-validity checks at gate entry — the design alpha must
       equal the plan alpha, and the surprise threshold must exceed
       the plan's final sequential boundary. A misconfigured design
       is a design refusal, reported before any trust test runs.
    3. Trust tests, before results are read — SRM, then A/A, then
       guardrails, then secondary-metric completeness. A trust
       failure short-circuits with the trust reason; the OEC z is
       not even read on that path.
    4. Metric provenance — every reported look must name the
       design's OEC metric. A z with no metric behind it cannot ship
       as an OEC win.
    5. The OEC win, then Twyman's law — the final look must have
       rejected in the declared direction, and a surprising win
       needs confirming replication evidence.

    The verdict records the registered design digest that was used,
    so an audit can check it was the current design.
    """
    notes: list[str] = []
    design_digest = ""

    def refuse(
        reasons: list[str],
        srm: SrmVerdict | None = None,
        aa: AaVerdict | None = None,
        guardrails: tuple[GuardrailVerdict, ...] = (),
        oec_z: float | None = None,
        surprising: bool = False,
    ) -> ShipVerdict:
        return ShipVerdict(
            status="no_ship",
            reasons=tuple(reasons),
            notes=tuple(notes),
            srm=srm,
            aa=aa,
            guardrails=guardrails,
            oec_z=oec_z,
            surprising=surprising,
            plan_digest=design.plan_digest,
            design_digest=design_digest,
        )

    if evidence.plan_digest != design.plan_digest:
        return refuse(
            [
                (
                    f"PLAN_DIGEST_MISMATCH: evidence names plan "
                    f"{evidence.plan_digest[:12]}... but the design binds "
                    f"{design.plan_digest[:12]}..."
                )
            ]
        )
    try:
        plan = load_plan(design.plan_digest, registry_path)
    except KeyError:
        return refuse(
            [(f"UNREGISTERED_PLAN: no registered plan with digest {design.plan_digest[:12]}...")]
        )
    except (RegistryCorrupted, RegistryMalformed) as e:
        return refuse([f"REGISTRY_UNREADABLE: {type(e).__name__}: {e}"])
    found = find_design(design.to_dict(), registry_path)
    if found is None:
        return refuse(
            [
                (
                    "DESIGN_NOT_REGISTERED: this design (OEC, guardrail "
                    "budgets, allocation, surprise threshold) matches no "
                    "design record in the registry — register it with "
                    "`register-design` before the run, then ship against "
                    "the registered design. A design rewritten after "
                    "results are known is not a pre-registration."
                )
            ]
        )
    design_digest = found
    verdict = verify_experiment(design.plan_digest, registry_path, evidence.experiment)
    if not verdict.valid:
        notes.extend(verdict.notes)
        return refuse(
            [(f"experiment violates its pre-registered plan: {', '.join(verdict.violations)}")],
        )
    notes.extend(verdict.notes)

    # --- Design-validity checks at gate entry. ---
    if design.alpha != plan.alpha:
        return refuse(
            [
                (
                    f"ALPHA_MISMATCH: design alpha {design.alpha} != plan "
                    f"alpha {plan.alpha} — one alpha governs the whole "
                    "experiment; a design that grades at a looser level "
                    "than the plan it binds to is not a pre-registration"
                )
            ]
        )
    final_boundary = group_sequential_boundaries(plan.looks, plan.alpha, plan.sides)[-1]
    if design.surprise_z <= final_boundary:
        return refuse(
            [
                (
                    f"SURPRISE_THRESHOLD_INVALID: surprise_z={design.surprise_z} "
                    f"does not exceed the plan's final boundary "
                    f"{final_boundary:.2f} — every significant win would need "
                    "replication"
                )
            ]
        )

    # --- Trust tests, before results are read. ---
    if len(evidence.srm_counts) != len(design.expected_allocation):
        return refuse(
            [
                (
                    f"SRM_ARMS_MISMATCH: {len(evidence.srm_counts)} SRM counts "
                    f"for {len(design.expected_allocation)} allocated arms"
                )
            ]
        )
    srm = check_srm(evidence.srm_counts, design.expected_allocation)
    if srm.status != "pass":
        return refuse(
            [f"SRM_{srm.status.upper()}: " + "; ".join(srm.reasons)],
            srm=srm,
        )
    notes.extend(srm.notes)

    if evidence.aa is None:
        return refuse(
            [
                (
                    "AA_MISSING: no A/A null split reported — an experiment "
                    "with no A/A ran no trust test"
                )
            ],
            srm=srm,
        )
    aa = check_aa(evidence.aa, design.alpha, design.min_aa_n)
    if aa.status != "pass":
        return refuse(
            [f"AA_{aa.status.upper()}: " + "; ".join(aa.reasons)],
            srm=srm,
            aa=aa,
        )
    notes.extend(aa.notes)

    declared = {m.name: m for m in design.guardrails}
    observed = {o.name: o for o in evidence.guardrails}
    undeclared = sorted(set(observed) - set(declared))
    if undeclared:
        return refuse(
            [
                (
                    f"UNDECLARED_METRIC: guardrail observations for metrics the "
                    f"design never declared: {', '.join(undeclared)}"
                )
            ],
            srm=srm,
            aa=aa,
        )
    missing = sorted(set(declared) - set(observed))
    if missing:
        return refuse(
            [
                (
                    f"GUARDRAIL_UNMEASURED: declared guardrail(s) with no "
                    f"observation: {', '.join(missing)}"
                )
            ],
            srm=srm,
            aa=aa,
        )
    guardrail_verdicts: list[GuardrailVerdict] = []
    for name in sorted(declared):
        gv = check_guardrail(declared[name], observed[name], design.alpha)
        guardrail_verdicts.append(gv)
        if not gv.passed:
            return refuse(
                ["GUARDRAIL_REGRESSED: " + "; ".join(gv.reasons)],
                srm=srm,
                aa=aa,
                guardrails=tuple(guardrail_verdicts),
            )
    guardrails = tuple(guardrail_verdicts)

    # --- Secondary metrics: declared means reported. ---
    secondary_declared = {m.name: m for m in design.metrics if m.role in ("goal", "driver")}
    secondary_observed = {o.name: o for o in evidence.secondary}
    secondary_undeclared = sorted(set(secondary_observed) - set(secondary_declared))
    if secondary_undeclared:
        return refuse(
            [
                (
                    f"UNDECLARED_METRIC: secondary observations for metrics "
                    f"the design never declared as goal/driver: "
                    f"{', '.join(secondary_undeclared)}"
                )
            ],
            srm=srm,
            aa=aa,
            guardrails=guardrails,
        )
    secondary_missing = sorted(set(secondary_declared) - set(secondary_observed))
    if secondary_missing:
        return refuse(
            [
                (
                    f"SECONDARY_UNMEASURED: declared goal/driver metric(s) "
                    f"with no observation: {', '.join(secondary_missing)} — "
                    "a design that declares a metric must report it"
                )
            ],
            srm=srm,
            aa=aa,
            guardrails=guardrails,
        )
    for name in sorted(secondary_declared):
        decl = secondary_declared[name]
        obs = secondary_observed[name]
        if decl.direction == "higher_better":
            delta = obs.treated_rate - obs.control_rate
        else:
            delta = obs.control_rate - obs.treated_rate
        notes.append(
            f"secondary {name!r} ({decl.role}): good-direction delta "
            f"{delta:+.3f} (reported, does not gate)"
        )

    # --- Metric provenance, before the OEC win is read. ---
    if evidence.oec_metric != design.oec.name:
        return refuse(
            [
                (
                    f"OEC_MISMATCH: evidence reports its final z on "
                    f"{evidence.oec_metric!r}, but the design's OEC is "
                    f"{design.oec.name!r} — shipping on a metric the design "
                    "never named OEC is not allowed"
                )
            ],
            srm=srm,
            aa=aa,
            guardrails=guardrails,
        )
    looks = evidence.experiment.looks
    if not looks:
        return refuse(
            ["NO_LOOKS: the experiment reports no interim looks"],
            srm=srm,
            aa=aa,
            guardrails=guardrails,
        )
    unnamed = [look.fraction for look in looks if look.metric != design.oec.name]
    if unnamed:
        return refuse(
            [
                (
                    f"OEC_METRIC_UNPROVEN: look(s) at fraction(s) "
                    f"{', '.join(str(f) for f in unnamed)} do not name the "
                    f"OEC metric {design.oec.name!r} — a z with no metric "
                    "behind it cannot ship as an OEC win"
                )
            ],
            srm=srm,
            aa=aa,
            guardrails=guardrails,
        )

    # --- Results are read only now that every trust test passed. ---
    final = looks[-1]
    oec_z = final.z
    if not final.rejected:
        return refuse(
            [
                (
                    f"NO_SIGNIFICANT_OEC_WIN: final look z={oec_z:.2f} did not "
                    "reject — the OEC did not move"
                )
            ],
            srm=srm,
            aa=aa,
            guardrails=guardrails,
            oec_z=oec_z,
        )
    won = oec_z > 0 if design.oec.direction == "higher_better" else oec_z < 0
    if not won:
        return refuse(
            [
                (
                    f"SIGNIFICANT_OEC_HARM: final look rejected (z={oec_z:.2f}) "
                    f"in the WRONG direction for OEC {design.oec.name!r} "
                    f"({design.oec.direction}) — the change hurts the OEC"
                )
            ],
            srm=srm,
            aa=aa,
            guardrails=guardrails,
            oec_z=oec_z,
        )

    # --- Twyman's law: surprising wins need replication. ---
    surprising = abs(oec_z) > design.surprise_z
    if not surprising:
        return ShipVerdict(
            status="ship",
            notes=tuple(
                notes
                + [
                    (
                        f"OEC {design.oec.name!r} won (final z={oec_z:.2f}); "
                        "SRM, A/A, and guardrails clean"
                    )
                ]
            ),
            srm=srm,
            aa=aa,
            guardrails=guardrails,
            oec_z=oec_z,
            surprising=False,
            plan_digest=design.plan_digest,
            design_digest=design_digest,
        )
    replication = evidence.replication
    if replication is None:
        return ShipVerdict(
            status="needs_replication",
            reasons=(
                (
                    f"NEEDS_REPLICATION: surprising win (final z={oec_z:.2f} "
                    f"beyond surprise_z={design.surprise_z}) with no "
                    "replication evidence — Twyman's law: too-exciting "
                    "results need a second look before they ship. To ship, "
                    "supply replication evidence: a fresh run with z "
                    f"significant at alpha={design.alpha} in the winning "
                    f"direction, n >= {design.min_replication_n}, and a "
                    f"run_id different from {evidence.run_id!r} "
                    "(see docs/OEC.md section 3)."
                ),
            ),
            notes=tuple(notes),
            srm=srm,
            aa=aa,
            guardrails=guardrails,
            oec_z=oec_z,
            surprising=True,
            plan_digest=design.plan_digest,
            design_digest=design_digest,
        )
    if replication.run_id == evidence.run_id:
        return refuse(
            [
                (
                    "REPLICATION_SAME_RUN: replication run_id equals the "
                    "original run_id — a replication that re-reports the same "
                    "run is not a replication"
                )
            ],
            srm=srm,
            aa=aa,
            guardrails=guardrails,
            oec_z=oec_z,
            surprising=True,
        )
    if replication.n < design.min_replication_n:
        return refuse(
            [
                (
                    f"REPLICATION_TOO_SMALL: replication n={replication.n} below "
                    f"min_replication_n={design.min_replication_n}"
                )
            ],
            srm=srm,
            aa=aa,
            guardrails=guardrails,
            oec_z=oec_z,
            surprising=True,
        )
    rep_won = replication.z > 0 if design.oec.direction == "higher_better" else replication.z < 0
    # The replication is held to the same standard as the original
    # decision: a two-sided plan confirms at the two-sided level,
    # a one-sided plan at the one-sided level. The direction was
    # fixed by the original win before the replication data was
    # seen, so the test is directional, not post-hoc.
    if plan.sides == "one":
        rep_critical = inv_normal_cdf(1.0 - design.alpha)
    else:
        rep_critical = inv_normal_cdf(1.0 - design.alpha / 2.0)
    if not rep_won or abs(replication.z) < rep_critical:
        return refuse(
            [
                (
                    "REPLICATION_NOT_CONFIRMED: the replication run did not "
                    f"confirm the win (z={replication.z:.2f}, critical "
                    f"{rep_critical:.2f} at alpha={design.alpha}) — the surprise "
                    "does not survive a second look"
                )
            ],
            srm=srm,
            aa=aa,
            guardrails=guardrails,
            oec_z=oec_z,
            surprising=True,
        )
    return ShipVerdict(
        status="ship",
        notes=tuple(
            notes
            + [
                (
                    f"surprising OEC win (final z={oec_z:.2f}) confirmed by "
                    f"replication (z={replication.z:.2f}, n={replication.n}); "
                    "SRM, A/A, and guardrails clean"
                )
            ]
        ),
        srm=srm,
        aa=aa,
        guardrails=guardrails,
        oec_z=oec_z,
        surprising=True,
        plan_digest=design.plan_digest,
        design_digest=design_digest,
    )


def check_ship(
    design: ExperimentDesign,
    evidence: ShipEvidence,
    registry_path: str,
) -> ShipVerdict:
    """ship_decision, but anything other than SHIP raises ShipRefused.

    This is the CI gate: call it at the end of an agent-change run and
    any refusal fails the build, naming the reason.
    """
    verdict = ship_decision(design, evidence, registry_path)
    if verdict.status != "ship":
        raise ShipRefused(f"ship gate refuses ({verdict.status}): " + "; ".join(verdict.reasons))
    return verdict


# ---------------------------------------------------------------------------
# JSON loaders (the ``ship`` CLI input) — fail closed
# ---------------------------------------------------------------------------


def design_from_dict(data: Mapping[str, object]) -> ExperimentDesign:
    """Build an ExperimentDesign from its canonical mapping.

    The same validation the JSON loader applies, shared by the
    registry path: a design loaded from a registry record goes
    through this constructor, so a tampered record is refused the
    same way a tampered file is. The canonical mapping is what
    ``ExperimentDesign.to_dict`` produces and what the registry
    stores, so a registered design round-trips exactly.
    """
    if not isinstance(data, dict):
        raise TypeError("design must be a mapping")
    unknown = set(data) - {
        "plan_digest",
        "metrics",
        "expected_allocation",
        "surprise_z",
        "alpha",
        "min_aa_n",
        "min_replication_n",
    }
    if unknown:
        raise ValueError(f"design has unknown keys: {sorted(unknown)}")
    for key in ("plan_digest", "metrics", "expected_allocation"):
        if key not in data:
            raise ValueError(f"design missing required field {key!r}")
    if not isinstance(data["plan_digest"], str) or not data["plan_digest"]:
        raise TypeError("plan_digest must be a non-empty string")
    raw_metrics = data["metrics"]
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise TypeError("metrics must be a non-empty list")
    metrics: list[MetricDeclaration] = []
    for entry in raw_metrics:
        if not isinstance(entry, dict):
            raise TypeError(f"metric must be a mapping, got {entry!r}")
        entry_unknown = set(entry) - {"name", "role", "direction", "max_regression"}
        if entry_unknown:
            raise ValueError(f"metric has unknown keys: {sorted(entry_unknown)}")
        for key in ("name", "role", "direction"):
            if key not in entry:
                raise ValueError(f"metric missing required field {key!r}")
        if not isinstance(entry["name"], str) or not entry["name"]:
            raise TypeError(f"metric name must be a non-empty string, got {entry['name']!r}")
        if entry["role"] not in ("oec", "goal", "driver", "guardrail"):
            raise ValueError(
                f"metric role must be oec/goal/driver/guardrail, got {entry['role']!r}"
            )
        if entry["direction"] not in ("higher_better", "lower_better"):
            raise ValueError(
                f"metric direction must be higher_better/lower_better, got {entry['direction']!r}"
            )
        max_regression = entry.get("max_regression")
        if max_regression is not None and (
            not _is_number(max_regression) or isinstance(max_regression, bool)
        ):
            raise TypeError(f"max_regression must be a number or null, got {max_regression!r}")
        metrics.append(
            MetricDeclaration(
                name=entry["name"],
                role=entry["role"],
                direction=entry["direction"],
                max_regression=(float(max_regression) if max_regression is not None else None),
            )
        )
    raw_allocation = data["expected_allocation"]
    if not isinstance(raw_allocation, list) or not raw_allocation:
        raise TypeError("expected_allocation must be a non-empty list")
    allocation: list[float] = []
    for share in raw_allocation:
        if not _is_number(share):
            raise TypeError(f"allocation share must be a number, got {share!r}")
        allocation.append(float(share))
    kwargs: dict[str, object] = {}
    if "surprise_z" in data:
        if not _is_number(data["surprise_z"]):
            raise TypeError(f"surprise_z must be a number, got {data['surprise_z']!r}")
        kwargs["surprise_z"] = float(data["surprise_z"])
    if "alpha" in data:
        if not _is_number(data["alpha"]):
            raise TypeError(f"alpha must be a number, got {data['alpha']!r}")
        kwargs["alpha"] = float(data["alpha"])
    if "min_aa_n" in data:
        if not _is_int(data["min_aa_n"]):
            raise TypeError(f"min_aa_n must be an int, got {data['min_aa_n']!r}")
        kwargs["min_aa_n"] = data["min_aa_n"]
    if "min_replication_n" in data:
        if not _is_int(data["min_replication_n"]):
            raise TypeError(f"min_replication_n must be an int, got {data['min_replication_n']!r}")
        kwargs["min_replication_n"] = data["min_replication_n"]
    return ExperimentDesign(
        plan_digest=data["plan_digest"],
        metrics=tuple(metrics),
        expected_allocation=tuple(allocation),
        **kwargs,  # type: ignore[arg-type]
    )


def design_from_json(text: str) -> ExperimentDesign:
    """Load an ExperimentDesign from JSON (the ``ship`` CLI input).

    A mapping with ``plan_digest`` (string), ``metrics`` (list of
    ``name`` / ``role`` / ``direction`` / ``max_regression`` mappings)
    and ``expected_allocation`` (list of shares); ``surprise_z``,
    ``alpha``, ``min_aa_n`` and ``min_replication_n`` are optional.
    The loader fails closed: unknown keys, a missing required field,
    a duplicate metric name, a wrong-typed field, or an out-of-range
    value all raise ValueError/TypeError, so a typo cannot silently
    become a registered design.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    return design_from_dict(data)


def register_design(design: ExperimentDesign, registry_path: str) -> str:
    """Append a design to the hash-chained registry; return its digest.

    The design's plan must already be registered: a design bound to
    an unregistered plan is refused, so the registration order is
    plan first, design second. The design record commits to the
    previous record's digest, so the OEC, the guardrail budgets, the
    expected allocation, and the surprise threshold cannot be
    silently rewritten after data arrives. Amendment rule: the chain
    is append-only, so correcting a design means registering a NEW
    design record; the old record stays, and the ship verdict records
    which design digest was used.
    """
    try:
        load_plan(design.plan_digest, registry_path)
    except KeyError as e:
        raise ValueError(
            f"cannot register a design bound to unregistered plan "
            f"{design.plan_digest[:12]}...: register the plan first"
        ) from e
    return _register_design_record(design.to_dict(), registry_path)


def load_registered_design(digest: str, registry_path: str) -> ExperimentDesign:
    """Load a registered design by digest; unknown digests raise KeyError."""
    return design_from_dict(load_design(digest, registry_path))


def _looks_from_json(raw: object) -> tuple[LookEvidence, ...]:
    if not isinstance(raw, list) or not raw:
        raise TypeError("looks must be a non-empty list")
    looks: list[LookEvidence] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise TypeError(f"look must be a mapping, got {entry!r}")
        unknown = set(entry) - {"fraction", "z", "rejected", "metric"}
        if unknown:
            raise ValueError(f"look has unknown keys: {sorted(unknown)}")
        for key in ("fraction", "z", "rejected"):
            if key not in entry:
                raise ValueError(f"look missing required field {key!r}")
        if not _is_number(entry["fraction"]):
            raise TypeError(f"look fraction must be a number, got {entry['fraction']!r}")
        if not _is_number(entry["z"]):
            raise TypeError(f"look z must be a number, got {entry['z']!r}")
        if not isinstance(entry["rejected"], bool):
            raise TypeError(f"look rejected must be a bool, got {entry['rejected']!r}")
        metric = entry.get("metric", "")
        if not isinstance(metric, str):
            raise TypeError(f"look metric must be a string, got {metric!r}")
        looks.append(
            LookEvidence(
                fraction=float(entry["fraction"]),
                z=float(entry["z"]),
                rejected=entry["rejected"],
                metric=metric,
            )
        )
    return tuple(looks)


def _observation_from_json(entry: object) -> MetricObservation:
    if not isinstance(entry, dict):
        raise TypeError(f"guardrail observation must be a mapping, got {entry!r}")
    unknown = set(entry) - {
        "name",
        "control_successes",
        "control_n",
        "treated_successes",
        "treated_n",
    }
    if unknown:
        raise ValueError(f"observation has unknown keys: {sorted(unknown)}")
    for key in (
        "name",
        "control_successes",
        "control_n",
        "treated_successes",
        "treated_n",
    ):
        if key not in entry:
            raise ValueError(f"observation missing required field {key!r}")
    if not isinstance(entry["name"], str) or not entry["name"]:
        raise TypeError(f"observation name must be a non-empty string, got {entry['name']!r}")
    counts: dict[str, int] = {}
    for key in ("control_successes", "control_n", "treated_successes", "treated_n"):
        if not _is_int(entry[key]):
            raise TypeError(f"observation {key} must be an int, got {entry[key]!r}")
        counts[key] = entry[key]
    return MetricObservation(
        name=entry["name"],
        control_successes=counts["control_successes"],
        control_n=counts["control_n"],
        treated_successes=counts["treated_successes"],
        treated_n=counts["treated_n"],
    )


def evidence_from_json(text: str) -> ShipEvidence:
    """Load a ShipEvidence from JSON (the ``ship`` CLI input).

    A mapping with ``plan_digest``, ``oec_metric``, ``run_id``,
    ``experiment`` (``n_final`` + ``looks``), ``srm_counts``, and
    ``aa`` (mapping or null); ``guardrails`` (list) and ``replication``
    (mapping or null) are optional. CUPED evidence cannot be expressed
    here: a plan that declares a covariate needs a CupedResult, which
    this JSON carries no field for, so such evidence fails verification
    with CUPED_REQUIRED — the CLI refuses rather than silently
    verifying a plan it cannot fully check. The loader fails closed on
    unknown keys, missing fields, and wrong-typed values.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise TypeError("evidence JSON must be a mapping")
    unknown = set(data) - {
        "plan_digest",
        "oec_metric",
        "run_id",
        "experiment",
        "srm_counts",
        "aa",
        "guardrails",
        "secondary",
        "replication",
    }
    if unknown:
        raise ValueError(f"evidence has unknown keys: {sorted(unknown)}")
    for key in ("plan_digest", "oec_metric", "run_id", "experiment", "srm_counts", "aa"):
        if key not in data:
            raise ValueError(f"evidence missing required field {key!r}")
    for key in ("plan_digest", "oec_metric", "run_id"):
        if not isinstance(data[key], str) or not data[key]:
            raise TypeError(f"{key} must be a non-empty string, got {data[key]!r}")
    raw_experiment = data["experiment"]
    if not isinstance(raw_experiment, dict):
        raise TypeError(f"experiment must be a mapping, got {raw_experiment!r}")
    exp_unknown = set(raw_experiment) - {"n_final", "looks", "stopped_at"}
    if exp_unknown:
        raise ValueError(f"experiment has unknown keys: {sorted(exp_unknown)}")
    for key in ("n_final", "looks"):
        if key not in raw_experiment:
            raise ValueError(f"experiment missing required field {key!r}")
    if not _is_int(raw_experiment["n_final"]):
        raise TypeError(f"n_final must be an int, got {raw_experiment['n_final']!r}")
    stopped_at = raw_experiment.get("stopped_at")
    if stopped_at is not None and not _is_number(stopped_at):
        raise TypeError(f"stopped_at must be a number or null, got {stopped_at!r}")
    experiment = ExperimentEvidence(
        n_final=raw_experiment["n_final"],
        looks=_looks_from_json(raw_experiment["looks"]),
        stopped_at=(float(stopped_at) if stopped_at is not None else None),
    )
    raw_srm = data["srm_counts"]
    if not isinstance(raw_srm, list) or not raw_srm:
        raise TypeError("srm_counts must be a non-empty list")
    srm_counts: list[int] = []
    for count in raw_srm:
        if not _is_int(count) or count < 0:
            raise TypeError(f"srm_counts must be non-negative ints, got {count!r}")
        srm_counts.append(count)
    raw_aa = data["aa"]
    aa: AAEvidence | None = None
    if raw_aa is not None:
        if not isinstance(raw_aa, dict):
            raise TypeError(f"aa must be a mapping or null, got {raw_aa!r}")
        aa_unknown = set(raw_aa) - {"a_successes", "a_n", "b_successes", "b_n"}
        if aa_unknown:
            raise ValueError(f"aa has unknown keys: {sorted(aa_unknown)}")
        for key in ("a_successes", "a_n", "b_successes", "b_n"):
            if key not in raw_aa:
                raise ValueError(f"aa missing required field {key!r}")
            if not _is_int(raw_aa[key]):
                raise TypeError(f"aa {key} must be an int, got {raw_aa[key]!r}")
        aa = AAEvidence(
            a_successes=raw_aa["a_successes"],
            a_n=raw_aa["a_n"],
            b_successes=raw_aa["b_successes"],
            b_n=raw_aa["b_n"],
        )
    raw_guardrails = data.get("guardrails", [])
    if not isinstance(raw_guardrails, list):
        raise TypeError(f"guardrails must be a list, got {raw_guardrails!r}")
    guardrails: list[MetricObservation] = []
    for entry in raw_guardrails:
        guardrails.append(_observation_from_json(entry))
    raw_secondary = data.get("secondary", [])
    if not isinstance(raw_secondary, list):
        raise TypeError(f"secondary must be a list, got {raw_secondary!r}")
    secondary: list[MetricObservation] = []
    for entry in raw_secondary:
        secondary.append(_observation_from_json(entry))
    raw_replication = data.get("replication")
    replication: ReplicationEvidence | None = None
    if raw_replication is not None:
        if not isinstance(raw_replication, dict):
            raise TypeError(f"replication must be a mapping or null, got {raw_replication!r}")
        rep_unknown = set(raw_replication) - {"z", "n", "run_id"}
        if rep_unknown:
            raise ValueError(f"replication has unknown keys: {sorted(rep_unknown)}")
        for key in ("z", "n", "run_id"):
            if key not in raw_replication:
                raise ValueError(f"replication missing required field {key!r}")
        if not _is_number(raw_replication["z"]):
            raise TypeError(f"replication z must be a number, got {raw_replication['z']!r}")
        if not _is_int(raw_replication["n"]):
            raise TypeError(f"replication n must be an int, got {raw_replication['n']!r}")
        if not isinstance(raw_replication["run_id"], str) or not raw_replication["run_id"]:
            raise TypeError(
                f"replication run_id must be a non-empty string, got {raw_replication['run_id']!r}"
            )
        replication = ReplicationEvidence(
            z=float(raw_replication["z"]),
            n=raw_replication["n"],
            run_id=raw_replication["run_id"],
        )
    return ShipEvidence(
        plan_digest=data["plan_digest"],
        oec_metric=data["oec_metric"],
        run_id=data["run_id"],
        experiment=experiment,
        srm_counts=tuple(srm_counts),
        aa=aa,
        guardrails=tuple(guardrails),
        secondary=tuple(secondary),
        replication=replication,
    )


def default_design(plan_digest: str) -> ExperimentDesign:
    """A minimal design for the demo: one OEC, one guardrail, 50/50 split."""
    return ExperimentDesign(
        plan_digest=plan_digest,
        metrics=(
            MetricDeclaration("task_success", "oec", "higher_better"),
            MetricDeclaration("safety_violations", "guardrail", "lower_better", 0.02),
        ),
        expected_allocation=(0.5, 0.5),
    )
