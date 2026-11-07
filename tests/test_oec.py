"""Adversarial tests for the OEC ship gate (agenteval-bench#46).

Every mechanism the issue demands is pinned by a test that FAILS if
the mechanism is removed or weakened: OEC + guardrails pre-registered
(OEC switching refused), SRM / A/A trust tests running BEFORE the OEC
result is read, guardrails graded on interval lower bounds (never
point estimates), surprising wins needing replication (Twyman's law),
and no ship on a single exciting number.
"""

from __future__ import annotations

import json
import math
import os
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agenteval_bench import cli
from agenteval_bench.decision import wilson_interval as decision_wilson_interval
from experiment.alpha_spend import inv_normal_cdf
from experiment.oec import (
    SRM_ALPHA,
    AAEvidence,
    ExperimentDesign,
    MetricDeclaration,
    MetricObservation,
    ReplicationEvidence,
    ShipEvidence,
    ShipRefused,
    ShipVerdict,
    _wilson_interval,
    check_aa,
    check_guardrail,
    check_ship,
    check_srm,
    chi2_sf,
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
from experiment.registry import (
    ExperimentEvidence,
    LookEvidence,
    find_design,
    load_plan,
    register_plan,
    verify_experiment,
)
from experiment.registry import (
    register_design as register_design_record,
)
from experiment.types import ExperimentPlan

PLAN = ExperimentPlan(name="reranker-v2-vs-v1", n_planned=800, alpha=0.05, looks=(0.5, 1.0))


def _registry(tmp_path: str) -> tuple[str, str]:
    path = os.path.join(tmp_path, "plans.jsonl")
    return path, register_plan(PLAN, path)


def _registered_design(registry: str, digest: str, **kwargs: object) -> ExperimentDesign:
    """A design registered in the hash chain (the ship gate requires it)."""
    design = _design(digest, **kwargs)
    register_design(design, registry)
    return design


def _design(digest: str, **kwargs: object) -> ExperimentDesign:
    base: dict[str, object] = {
        "plan_digest": digest,
        "metrics": (
            MetricDeclaration("task_success", "oec", "higher_better"),
            MetricDeclaration("safety_violations", "guardrail", "lower_better", 0.02),
        ),
        "expected_allocation": (0.5, 0.5),
    }
    base.update(kwargs)
    return ExperimentDesign(**base)  # type: ignore[arg-type]


#: Default A/A split and guardrail observation for the test helper's
#: base evidence (module-level singletons: B008 forbids calls in
#: defaults).
_DEFAULT_AA = AAEvidence(250, 500, 260, 500)
_DEFAULT_GUARDRAILS = (MetricObservation("safety_violations", 24, 400, 16, 400),)


def _evidence(
    digest: str,
    final_z: float = 2.5,
    final_rejected: bool = True,
    srm_counts: tuple[int, ...] = (398, 402),
    aa: AAEvidence | None = _DEFAULT_AA,
    guardrails: tuple[MetricObservation, ...] = _DEFAULT_GUARDRAILS,
    replication: ReplicationEvidence | None = None,
    oec_metric: str = "task_success",
    run_id: str = "run-1",
    n_final: int = 800,
    looks: tuple[LookEvidence, ...] | None = None,
    stopped_at: float | None = None,
    secondary: tuple[MetricObservation, ...] = (),
) -> ShipEvidence:
    return ShipEvidence(
        plan_digest=digest,
        oec_metric=oec_metric,
        run_id=run_id,
        experiment=ExperimentEvidence(
            n_final=n_final,
            looks=(
                looks
                if looks is not None
                else (
                    LookEvidence(0.5, 1.0, False, metric="task_success"),
                    LookEvidence(1.0, final_z, final_rejected, metric="task_success"),
                )
            ),
            stopped_at=stopped_at,
        ),
        srm_counts=srm_counts,
        aa=aa,
        guardrails=guardrails,
        secondary=secondary,
        replication=replication,
    )


# ---------------------------------------------------------------------------
# chi-square survival function
# ---------------------------------------------------------------------------


def test_chi2_sf_known_values() -> None:
    # df=1: the 95th percentile; df=2: closed form exp(-x/2);
    # df=3: the 90th percentile. A wrong gamma implementation fails
    # these, not a reviewer's eye.
    assert math.isclose(chi2_sf(3.841458820694124, 1), 0.05, rel_tol=1e-6)
    assert math.isclose(chi2_sf(2.0, 2), math.exp(-1.0), rel_tol=1e-9)
    assert math.isclose(chi2_sf(6.251388631319988, 3), 0.10, rel_tol=1e-6)
    assert chi2_sf(0.0, 1) == 1.0
    assert chi2_sf(100.0, 1) < 1e-20


def test_chi2_sf_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        chi2_sf(-1.0, 1)
    with pytest.raises(ValueError):
        chi2_sf(1.0, 0)


def test_chi2_sf_null_p_values_are_uniform() -> None:
    # 1000 null experiments: the fraction of p-values below 0.05 must
    # sit near 0.05. Seeded, so deterministic; bounds are generous
    # (binomial sd ~0.007) so a correct implementation never flakes.
    rng = random.Random(20261107)
    below = 0
    trials = 1000
    for _ in range(trials):
        treated = sum(rng.random() < 0.5 for _ in range(800))
        verdict = check_srm((treated, 800 - treated), (0.5, 0.5))
        assert verdict.p_value is not None
        if verdict.p_value < 0.05:
            below += 1
    assert 0.02 <= below / trials <= 0.09


@given(
    chi2=st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
    df=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=60, deadline=None)
def test_chi2_sf_is_a_probability_and_decreases_in_chi2(chi2: float, df: int) -> None:
    p = chi2_sf(chi2, df)
    assert 0.0 <= p <= 1.0
    assert chi2_sf(chi2 + 1.0, df) <= p


def test_two_proportion_z_known_values() -> None:
    assert two_proportion_z(50, 100, 50, 100) == 0.0
    # 60/100 vs 40/100: pooled 0.5, se = sqrt(0.25 * 0.02) = 0.0707...
    z = two_proportion_z(60, 100, 40, 100)
    assert math.isclose(z, 0.2 / math.sqrt(0.25 * 0.02), rel_tol=1e-9)
    assert math.isclose(two_sided_p(1.959964), 0.05, rel_tol=1e-5)
    with pytest.raises(ValueError):
        two_proportion_z(101, 100, 50, 100)


# ---------------------------------------------------------------------------
# Design validation
# ---------------------------------------------------------------------------


def test_design_needs_exactly_one_oec() -> None:
    with pytest.raises(ValueError, match="exactly one OEC"):
        _design(
            "d",
            metrics=(
                MetricDeclaration("a", "oec", "higher_better"),
                MetricDeclaration("b", "oec", "higher_better"),
            ),
        )
    with pytest.raises(ValueError, match="exactly one OEC"):
        _design(
            "d",
            metrics=(MetricDeclaration("a", "goal", "higher_better"),),
        )


def test_design_rejects_duplicate_metric_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        _design(
            "d",
            metrics=(
                MetricDeclaration("a", "oec", "higher_better"),
                MetricDeclaration("a", "guardrail", "lower_better", 0.02),
            ),
        )


def test_guardrail_without_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="max_regression"):
        MetricDeclaration("g", "guardrail", "lower_better")
    with pytest.raises(ValueError, match="max_regression"):
        MetricDeclaration("o", "oec", "higher_better", 0.02)


def test_design_rejects_bad_allocation() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        _design("d", expected_allocation=(0.5, 0.6))
    with pytest.raises(ValueError, match="at least 2 arms"):
        _design("d", expected_allocation=(1.0,))


# ---------------------------------------------------------------------------
# SRM
# ---------------------------------------------------------------------------


def test_srm_pass_fail_and_insufficient() -> None:
    assert check_srm((398, 402), (0.5, 0.5)).status == "pass"
    failed = check_srm((350, 450), (0.5, 0.5))
    assert failed.status == "fail"
    assert failed.p_value is not None and failed.p_value < 0.001
    # 8 total assignments: expected 4 per arm — below the power floor.
    thin = check_srm((4, 4), (0.5, 0.5))
    assert thin.status == "insufficient"
    assert check_srm((0, 0), (0.5, 0.5)).status == "insufficient"
    with pytest.raises(ValueError):
        check_srm((400,), (0.5, 0.5))


def test_srm_shares_matter_not_just_counts() -> None:
    # The same counts pass under the split they match and fail under
    # the split they do not: the test checks the REGISTERED split.
    assert check_srm((450, 350), (0.5625, 0.4375)).status == "pass"
    assert check_srm((450, 350), (0.5, 0.5)).status == "fail"


def test_srm_alpha_is_pinned_at_0_001() -> None:
    # The distinguishing case from the round-1 review: (430, 370)
    # gives chi2=4.5, p=0.0339 — it passes at the Kohavi 0.001 level
    # and would fail at 0.05. If SRM_ALPHA ever drifts to 0.05, this
    # test fails: the constant is load-bearing, not decorative.
    assert SRM_ALPHA == 0.001
    verdict = check_srm((430, 370), (0.5, 0.5))
    assert verdict.status == "pass"
    assert verdict.p_value is not None
    assert verdict.p_value == pytest.approx(0.0339, abs=0.001)
    # p < 0.05, so the same counts would FAIL at a 0.05 level: the
    # 0.001 constant is what keeps this a pass.
    assert verdict.p_value < 0.05


# ---------------------------------------------------------------------------
# A/A
# ---------------------------------------------------------------------------


def test_aa_pass_fail_and_insufficient() -> None:
    assert check_aa(AAEvidence(250, 500, 260, 500), 0.05, 100).status == "pass"
    failed = check_aa(AAEvidence(200, 500, 300, 500), 0.05, 100)
    assert failed.status == "fail"
    assert failed.p_value is not None and failed.p_value < 1e-9
    # n=30/arm is below the power floor: not a clean A/A.
    thin = check_aa(AAEvidence(15, 30, 15, 30), 0.05, 100)
    assert thin.status == "insufficient"


# ---------------------------------------------------------------------------
# Guardrails (lower-bound discipline)
# ---------------------------------------------------------------------------


def test_guardrail_passes_when_lower_bound_clears_budget() -> None:
    decl = MetricDeclaration("safety", "guardrail", "lower_better", 0.02)
    verdict = check_guardrail(decl, MetricObservation("safety", 24, 400, 16, 400), 0.05)
    assert verdict.passed
    assert verdict.delta_good > 0  # treated is BETTER here


def test_guardrail_point_above_bound_below_fails() -> None:
    # The trap the lower-bound discipline exists for: the point
    # estimate of the good-direction delta is inside the budget, but
    # the lower bound is not — the guardrail FAILS. At n=40/arm the
    # interval is wide: delta +0.025 sits inside the 0.02 budget as a
    # point, but its Newcombe lower bound (about -0.088) does not clear
    # -0.02.
    decl = MetricDeclaration("safety", "guardrail", "lower_better", 0.02)
    trap = MetricObservation("safety", 2, 40, 1, 40)
    trap_verdict = check_guardrail(decl, trap, 0.05)
    assert trap_verdict.delta_good == pytest.approx(0.025)
    assert trap_verdict.lower_bound < -0.02
    assert not trap_verdict.passed
    # Same rates at n=400/arm: the interval tightens and it passes.
    # The grade follows the measurement, not the point.
    tight = MetricObservation("safety", 20, 400, 10, 400)
    tight_verdict = check_guardrail(decl, tight, 0.05)
    assert tight_verdict.delta_good == pytest.approx(0.025)
    assert tight_verdict.passed


def test_guardrail_zero_events_fail_closed_on_thin_data() -> None:
    # The round-1 review's B1/B3: zero observed events must NOT pass
    # on trivial data. Under the old Wald interval 0/1 vs 0/1 had a
    # zero-width bound and passed; under Newcombe the lower bound is
    # about -0.73 and it fails. At n=400 the same zero events pass:
    # 400 clean samples are real evidence, 1 is not.
    decl = MetricDeclaration("safety", "guardrail", "lower_better", 0.02)
    one = check_guardrail(decl, MetricObservation("safety", 0, 1, 0, 1), 0.05)
    assert not one.passed
    assert one.lower_bound == pytest.approx(-0.73, abs=0.01)
    forty = check_guardrail(decl, MetricObservation("safety", 0, 40, 0, 40), 0.05)
    assert not forty.passed
    four_hundred = check_guardrail(decl, MetricObservation("safety", 0, 400, 0, 400), 0.05)
    assert four_hundred.passed


def test_newcombe_interval_known_values() -> None:
    # 0/1 vs 0/1: the interval is wide, not a point — the failure
    # mode the Wald interval hid.
    lower, upper = newcombe_interval(0, 1, 0, 1, 0.05)
    assert lower == pytest.approx(-0.73, abs=0.01)
    assert upper == pytest.approx(0.73, abs=0.01)
    # The demo's default guardrail: 24/400 vs 16/400, treated-minus-
    # control delta -0.02, Newcombe lower about -0.046.
    lower, _ = newcombe_interval(24, 400, 16, 400, 0.05)
    assert lower == pytest.approx(-0.046, abs=0.002)
    with pytest.raises(ValueError):
        newcombe_interval(1, 10, 1, 10, 1.5)


def test_wilson_interval_agrees_with_decision_module() -> None:
    # The oec module carries a local copy of decision.wilson_interval
    # (the experiment package must stay stdlib-only). The two must
    # agree across a grid, so a drift in either is caught here.
    z = inv_normal_cdf(0.95)
    for k, n in [(0, 1), (0, 40), (0, 400), (1, 40), (2, 40), (24, 400), (200, 400), (400, 400)]:
        local = _wilson_interval(k, n, z)
        reference = decision_wilson_interval(k, n, z)
        assert local[0] == pytest.approx(reference[0], abs=1e-12)
        assert local[1] == pytest.approx(reference[1], abs=1e-12)


def test_guardrail_regression_fails_in_both_directions() -> None:
    low = MetricDeclaration("safety", "guardrail", "lower_better", 0.02)
    assert not check_guardrail(low, MetricObservation("safety", 8, 400, 40, 400), 0.05).passed
    high = MetricDeclaration("quality", "guardrail", "higher_better", 0.02)
    assert not check_guardrail(high, MetricObservation("quality", 320, 400, 280, 400), 0.05).passed


def test_guardrail_refuses_mismatched_declaration() -> None:
    decl = MetricDeclaration("safety", "guardrail", "lower_better", 0.02)
    with pytest.raises(ValueError):
        check_guardrail(decl, MetricObservation("other", 1, 10, 1, 10), 0.05)
    with pytest.raises(ValueError):
        check_guardrail(
            MetricDeclaration("q", "goal", "higher_better"),
            MetricObservation("q", 1, 10, 1, 10),
            0.05,
        )


@given(
    treated=st.integers(min_value=0, max_value=400),
)
@settings(max_examples=60, deadline=None)
def test_guardrail_is_monotone_in_treated_successes(treated: int) -> None:
    # Holding control fixed, more treated successes (higher_better)
    # can never lower the good-direction lower bound: the grade is
    # monotone in the measurement, so a better result cannot fail a
    # guardrail a worse result passed.
    decl = MetricDeclaration("quality", "guardrail", "higher_better", 0.02)
    better = MetricObservation("quality", 200, 400, treated, 400)
    worse = MetricObservation("quality", 200, 400, max(0, treated - 1), 400)
    assert (
        check_guardrail(decl, better, 0.05).lower_bound
        >= check_guardrail(decl, worse, 0.05).lower_bound
    )


# ---------------------------------------------------------------------------
# Ship gate
# ---------------------------------------------------------------------------


def test_ship_compliant_path(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    verdict = ship_decision(design, _evidence(digest), registry)
    assert verdict.status == "ship"
    assert verdict.reasons == ()
    assert verdict.srm is not None and verdict.srm.status == "pass"
    assert verdict.aa is not None and verdict.aa.status == "pass"
    assert verdict.oec_z == pytest.approx(2.5)
    assert not verdict.surprising
    assert verdict.design_digest != ""


def test_ship_refuses_unregistered_plan(tmp_path: str) -> None:
    registry, _ = _registry(str(tmp_path))
    verdict = ship_decision(_design("deadbeef" * 8), _evidence("deadbeef" * 8), registry)
    assert verdict.status == "no_ship"
    assert any("UNREGISTERED_PLAN" in r for r in verdict.reasons)


def test_ship_trust_tests_run_before_results_are_read(tmp_path: str) -> None:
    # The order pin (A7): a trust-failing experiment with a HUGE OEC
    # win reports the trust failure — the win is never read.
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    verdict = ship_decision(
        design,
        _evidence(digest, final_z=4.0, srm_counts=(350, 450)),
        registry,
    )
    assert verdict.status == "no_ship"
    assert any("SRM_FAIL" in r for r in verdict.reasons)
    assert verdict.oec_z is None  # the OEC z was never read


def test_ship_refuses_missing_aa(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    verdict = ship_decision(design, _evidence(digest, aa=None), registry)
    assert verdict.status == "no_ship"
    assert any("AA_MISSING" in r for r in verdict.reasons)


def test_ship_refuses_oec_switch(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    verdict = ship_decision(
        design,
        _evidence(digest, oec_metric="latency_p99"),
        registry,
    )
    assert verdict.status == "no_ship"
    assert any("OEC_MISMATCH" in r for r in verdict.reasons)


def test_ship_refuses_no_win_and_harm(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    no_win = ship_decision(
        design,
        _evidence(digest, final_z=1.0, final_rejected=False),
        registry,
    )
    assert no_win.status == "no_ship"
    assert any("NO_SIGNIFICANT_OEC_WIN" in r for r in no_win.reasons)
    harm = ship_decision(
        design,
        _evidence(digest, final_z=-2.5, final_rejected=True),
        registry,
    )
    assert harm.status == "no_ship"
    assert any("SIGNIFICANT_OEC_HARM" in r for r in harm.reasons)


def test_ship_refuses_undeclared_and_unmeasured_guardrails(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    undeclared = ship_decision(
        design,
        _evidence(
            digest,
            guardrails=(
                MetricObservation("safety_violations", 24, 400, 16, 400),
                MetricObservation("surprise_metric", 1, 10, 1, 10),
            ),
        ),
        registry,
    )
    assert undeclared.status == "no_ship"
    assert any("UNDECLARED_METRIC" in r for r in undeclared.reasons)
    unmeasured = ship_decision(design, _evidence(digest, guardrails=()), registry)
    assert unmeasured.status == "no_ship"
    assert any("GUARDRAIL_UNMEASURED" in r for r in unmeasured.reasons)


def test_surprise_needs_replication_and_replication_ships(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    needs = ship_decision(design, _evidence(digest, final_z=4.0), registry)
    assert needs.status == "needs_replication"
    assert needs.surprising
    assert any("fresh run" in r for r in needs.reasons)
    replicated = ship_decision(
        design,
        _evidence(
            digest,
            final_z=4.0,
            replication=ReplicationEvidence(z=2.5, n=400, run_id="run-2"),
        ),
        registry,
    )
    assert replicated.status == "ship"
    assert replicated.surprising


def test_surprise_with_bad_replication_is_no_ship(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    same_run = ship_decision(
        design,
        _evidence(
            digest,
            final_z=4.0,
            replication=ReplicationEvidence(z=2.5, n=400, run_id="run-1"),
        ),
        registry,
    )
    assert same_run.status == "no_ship"
    assert any("REPLICATION_SAME_RUN" in r for r in same_run.reasons)
    too_small = ship_decision(
        design,
        _evidence(
            digest,
            final_z=4.0,
            replication=ReplicationEvidence(z=2.5, n=10, run_id="run-2"),
        ),
        registry,
    )
    assert too_small.status == "no_ship"
    assert any("REPLICATION_TOO_SMALL" in r for r in too_small.reasons)
    opposite = ship_decision(
        design,
        _evidence(
            digest,
            final_z=4.0,
            replication=ReplicationEvidence(z=-2.5, n=400, run_id="run-2"),
        ),
        registry,
    )
    assert opposite.status == "no_ship"
    assert any("REPLICATION_NOT_CONFIRMED" in r for r in opposite.reasons)
    weak = ship_decision(
        design,
        _evidence(
            digest,
            final_z=4.0,
            replication=ReplicationEvidence(z=0.5, n=400, run_id="run-2"),
        ),
        registry,
    )
    assert weak.status == "no_ship"
    assert any("REPLICATION_NOT_CONFIRMED" in r for r in weak.reasons)


def test_surprise_threshold_below_final_boundary_is_invalid(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest, surprise_z=1.5)
    verdict = ship_decision(design, _evidence(digest), registry)
    assert verdict.status == "no_ship"
    assert any("SURPRISE_THRESHOLD_INVALID" in r for r in verdict.reasons)


def test_surprise_threshold_checked_before_results(tmp_path: str) -> None:
    # A misconfigured design is refused at gate entry even when the
    # evidence shows no win at all: the design check must not wait
    # for a significant result to notice it.
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest, surprise_z=1.5)
    verdict = ship_decision(
        design,
        _evidence(digest, final_z=1.0, final_rejected=False),
        registry,
    )
    assert verdict.status == "no_ship"
    assert any("SURPRISE_THRESHOLD_INVALID" in r for r in verdict.reasons)
    assert not any("NO_SIGNIFICANT_OEC_WIN" in r for r in verdict.reasons)


def test_check_ship_raises_and_passes(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    assert check_ship(design, _evidence(digest), registry).status == "ship"
    with pytest.raises(ShipRefused, match="SRM_FAIL"):
        check_ship(design, _evidence(digest, srm_counts=(350, 450)), registry)


def test_ship_verdict_forbids_forged_records() -> None:
    with pytest.raises(ValueError, match="cannot carry reasons"):
        ShipVerdict(status="ship", reasons=("x",))
    with pytest.raises(ValueError, match="needs non-empty reasons"):
        ShipVerdict(status="no_ship", reasons=())


# ---------------------------------------------------------------------------
# Round 2: design registration, metric provenance, alpha binding,
# secondary metrics, sides-aware replication, early stop
# ---------------------------------------------------------------------------


def test_ship_refuses_unregistered_design(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    # The design was never registered: the evidence is clean, the
    # design is well-formed, and the gate still refuses it.
    verdict = ship_decision(_design(digest), _evidence(digest), registry)
    assert verdict.status == "no_ship"
    assert any("DESIGN_NOT_REGISTERED" in r for r in verdict.reasons)
    assert verdict.oec_z is None


def test_ship_refuses_rewritten_design(tmp_path: str) -> None:
    # The round-1 review's B1/B3 reproduction: register a design,
    # then rewrite it after the fact — a 25x wider guardrail budget,
    # and separately a relabelled OEC — and both rewrites must refuse,
    # because neither matches a registered record.
    registry, digest = _registry(str(tmp_path))
    _registered_design(registry, digest)
    wider = _design(
        digest,
        metrics=(
            MetricDeclaration("task_success", "oec", "higher_better"),
            MetricDeclaration("safety_violations", "guardrail", "lower_better", 0.20),
        ),
    )
    verdict = ship_decision(wider, _evidence(digest), registry)
    assert verdict.status == "no_ship"
    assert any("DESIGN_NOT_REGISTERED" in r for r in verdict.reasons)
    relabelled = _design(
        digest,
        metrics=(
            MetricDeclaration("latency_p99", "oec", "lower_better"),
            MetricDeclaration("safety_violations", "guardrail", "lower_better", 0.02),
        ),
    )
    verdict = ship_decision(relabelled, _evidence(digest, oec_metric="latency_p99"), registry)
    assert verdict.status == "no_ship"
    assert any("DESIGN_NOT_REGISTERED" in r for r in verdict.reasons)


def _design_digest(registry: str, design: ExperimentDesign) -> str:
    """The digest under which a design was registered (test helper)."""
    found = find_design(design.to_dict(), registry)
    assert found is not None
    return found


def test_design_record_does_not_break_plan_loading(tmp_path: str) -> None:
    # Design records share the registry file with plan records;
    # loading and verifying the plan must ignore them.
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    assert load_plan(digest, registry).name == "reranker-v2-vs-v1"
    assert load_registered_design(_design_digest(registry, design), registry) == design
    verdict = verify_experiment(digest, registry, _evidence(digest).experiment)
    assert verdict.valid


def test_register_design_requires_registered_plan(tmp_path: str) -> None:
    registry, _ = _registry(str(tmp_path))
    with pytest.raises(ValueError, match="unregistered plan"):
        register_design(_design("deadbeef" * 8), registry)
    with pytest.raises(KeyError):
        load_registered_design("deadbeef" * 8, registry)


def test_design_from_dict_round_trip(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    assert design_from_dict(design.to_dict()) == design
    assert design_from_dict(json.loads(json.dumps(design.to_dict()))) == design


def test_ship_refuses_look_without_metric(tmp_path: str) -> None:
    # Metric provenance: looks that do not name the OEC metric cannot
    # ship as an OEC win, even when the z is decisive.
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    unnamed = ship_decision(
        design,
        _evidence(
            digest,
            looks=(
                LookEvidence(0.5, 1.0, False),
                LookEvidence(1.0, 2.5, True),
            ),
        ),
        registry,
    )
    assert unnamed.status == "no_ship"
    assert any("OEC_METRIC_UNPROVEN" in r for r in unnamed.reasons)
    wrong = ship_decision(
        design,
        _evidence(
            digest,
            looks=(
                LookEvidence(0.5, 1.0, False, metric="latency_p99"),
                LookEvidence(1.0, 2.5, True, metric="latency_p99"),
            ),
        ),
        registry,
    )
    assert wrong.status == "no_ship"
    assert any("OEC_METRIC_UNPROVEN" in r for r in wrong.reasons)


def test_ship_refuses_alpha_mismatch(tmp_path: str) -> None:
    # One alpha governs the whole experiment: a design grading at
    # 0.01 against a plan registered at 0.05 is not a pre-registration.
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest, alpha=0.01)
    verdict = ship_decision(design, _evidence(digest), registry)
    assert verdict.status == "no_ship"
    assert any("ALPHA_MISMATCH" in r for r in verdict.reasons)


def test_replication_critical_value_is_sides_aware(tmp_path: str) -> None:
    # z=1.8 confirms at the one-sided 0.05 level (critical 1.645)
    # and fails at the two-sided level (critical 1.96). The direction
    # was fixed by the original win, so the test is directional, but
    # the level follows the plan's sides.
    one_sided_plan = ExperimentPlan(
        name="one-sided", n_planned=800, alpha=0.05, looks=(0.5, 1.0), sides="one"
    )
    registry = os.path.join(str(tmp_path), "one.jsonl")
    one_digest = register_plan(one_sided_plan, registry)
    one_design = _registered_design(registry, one_digest)
    ships = ship_decision(
        one_design,
        _evidence(
            one_digest,
            final_z=4.0,
            replication=ReplicationEvidence(z=1.8, n=400, run_id="run-2"),
        ),
        registry,
    )
    assert ships.status == "ship"
    registry2, digest2 = _registry(str(tmp_path))
    two_design = _registered_design(registry2, digest2)
    refused = ship_decision(
        two_design,
        _evidence(
            digest2,
            final_z=4.0,
            replication=ReplicationEvidence(z=1.8, n=400, run_id="run-2"),
        ),
        registry2,
    )
    assert refused.status == "no_ship"
    assert any("REPLICATION_NOT_CONFIRMED" in r for r in refused.reasons)


def _one_sided_registry(tmp_path: str) -> tuple[str, str]:
    plan = ExperimentPlan(
        name="one-sided-latency", n_planned=800, alpha=0.05, looks=(0.5, 1.0), sides="one"
    )
    registry = os.path.join(str(tmp_path), "one.jsonl")
    return registry, register_plan(plan, registry)


def _lower_better_design(digest: str) -> ExperimentDesign:
    return _design(
        digest,
        metrics=(
            MetricDeclaration("latency_p99", "oec", "lower_better"),
            MetricDeclaration("safety_violations", "guardrail", "lower_better", 0.02),
        ),
    )


def test_register_design_refuses_one_sided_lower_better_oec(tmp_path: str) -> None:
    # B1 (round-2 implementation review): a one-sided sequential test
    # only ever rejects for positive z, while a lower_better win needs
    # a negative z — the combination can never ship, so registration
    # refuses it instead of hash-chaining a dead design.
    registry, digest = _one_sided_registry(str(tmp_path))
    with pytest.raises(ValueError, match="one-sided"):
        register_design(_lower_better_design(digest), registry)


def test_ship_names_one_sided_lower_better_instead_of_mismatch(tmp_path: str) -> None:
    # Defense in depth for a design registered before the registration
    # refusal existed (seeded here through the raw record writer): the
    # gate must blame the incoherent design, never misreport honest
    # evidence as DECISION_MISMATCH.
    registry, digest = _one_sided_registry(str(tmp_path))
    design = _lower_better_design(digest)
    register_design_record(design.to_dict(), registry)
    for final_z, rejected in ((-2.5, True), (-2.5, False), (2.5, True)):
        verdict = ship_decision(
            design,
            _evidence(
                digest,
                final_z=final_z,
                final_rejected=rejected,
                oec_metric="latency_p99",
                looks=(
                    LookEvidence(0.5, -1.0, False, metric="latency_p99"),
                    LookEvidence(1.0, final_z, rejected, metric="latency_p99"),
                ),
            ),
            registry,
        )
        assert verdict.status == "no_ship"
        assert any("ONE_SIDED_LOWER_BETTER" in r for r in verdict.reasons)
        assert not any("DECISION_MISMATCH" in r for r in verdict.reasons)


def test_two_sided_lower_better_oec_ships(tmp_path: str) -> None:
    # Control for the one-sided refusal: the same lower_better OEC on
    # a two-sided plan ships on honest evidence.
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest, metrics=_lower_better_design(digest).metrics)
    verdict = ship_decision(
        design,
        _evidence(
            digest,
            final_z=-2.5,
            oec_metric="latency_p99",
            looks=(
                LookEvidence(0.5, -1.0, False, metric="latency_p99"),
                LookEvidence(1.0, -2.5, True, metric="latency_p99"),
            ),
        ),
        registry,
    )
    assert verdict.status == "ship"


def test_evidence_loader_refuses_nonfinite_looks(tmp_path: str) -> None:
    # A2 (round-2 implementation review): the JSON loader is the CLI
    # trust boundary — an infinite z is not a plausible statistic and
    # must never reach the gate.
    registry, digest = _registry(str(tmp_path))
    _registered_design(registry, digest)
    evidence = _evidence(digest)
    doc: dict[str, object] = {
        "plan_digest": evidence.plan_digest,
        "oec_metric": evidence.oec_metric,
        "run_id": evidence.run_id,
        "experiment": {
            "n_final": evidence.experiment.n_final,
            "looks": [
                {
                    "fraction": look.fraction,
                    "z": look.z,
                    "rejected": look.rejected,
                    "metric": look.metric,
                }
                for look in evidence.experiment.looks
            ],
        },
        "srm_counts": list(evidence.srm_counts),
        "aa": {"a_successes": 250, "a_n": 500, "b_successes": 260, "b_n": 500},
        "guardrails": [
            {
                "name": "safety_violations",
                "control_successes": 24,
                "control_n": 400,
                "treated_successes": 16,
                "treated_n": 400,
            }
        ],
        "replication": None,
    }
    infinite_z = json.loads(json.dumps(doc))
    infinite_z["experiment"]["looks"][1]["z"] = float("inf")  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        evidence_from_json(json.dumps(infinite_z))
    infinite_fraction = json.loads(json.dumps(doc))
    infinite_fraction["experiment"]["looks"][0]["fraction"] = float("inf")  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        evidence_from_json(json.dumps(infinite_fraction))


def test_secondary_metrics_must_be_reported(tmp_path: str) -> None:
    # A design that declares a goal metric must report it: the
    # observation does not gate, but its absence refuses the ship.
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(
        registry,
        digest,
        metrics=(
            MetricDeclaration("task_success", "oec", "higher_better"),
            MetricDeclaration("session_length", "goal", "higher_better"),
            MetricDeclaration("safety_violations", "guardrail", "lower_better", 0.02),
        ),
    )
    missing = ship_decision(design, _evidence(digest), registry)
    assert missing.status == "no_ship"
    assert any("SECONDARY_UNMEASURED" in r for r in missing.reasons)
    reported = ship_decision(
        design,
        _evidence(
            digest,
            secondary=(MetricObservation("session_length", 200, 400, 220, 400),),
        ),
        registry,
    )
    assert reported.status == "ship"
    assert any("secondary 'session_length'" in n for n in reported.notes)


def test_secondary_undeclared_metric_is_refused(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    verdict = ship_decision(
        design,
        _evidence(
            digest,
            secondary=(MetricObservation("surprise_metric", 200, 400, 220, 400),),
        ),
        registry,
    )
    assert verdict.status == "no_ship"
    assert any("UNDECLARED_METRIC" in r for r in verdict.reasons)


def test_early_stop_ships(tmp_path: str) -> None:
    # Stopped at t=0.5 on a decisive look (z=3.0 past the 2.77
    # boundary), n_final=400: the early-stop path ships.
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(registry, digest)
    verdict = ship_decision(
        design,
        _evidence(
            digest,
            n_final=400,
            looks=(LookEvidence(0.5, 3.0, True, metric="task_success"),),
            stopped_at=0.5,
            srm_counts=(198, 202),
        ),
        registry,
    )
    assert verdict.status == "ship"


# ---------------------------------------------------------------------------
# JSON loaders
# ---------------------------------------------------------------------------


def test_json_round_trip_and_fail_closed(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    design = _design(digest)
    loaded_design = design_from_json(json.dumps(design.to_dict()))
    assert loaded_design == design
    register_design(loaded_design, registry)
    evidence = _evidence(digest)
    doc = {
        "plan_digest": evidence.plan_digest,
        "oec_metric": evidence.oec_metric,
        "run_id": evidence.run_id,
        "experiment": {
            "n_final": evidence.experiment.n_final,
            "looks": [
                {
                    "fraction": l.fraction,
                    "z": l.z,
                    "rejected": l.rejected,
                    "metric": l.metric,
                }
                for l in evidence.experiment.looks
            ],
        },
        "srm_counts": list(evidence.srm_counts),
        "aa": {
            "a_successes": 250,
            "a_n": 500,
            "b_successes": 260,
            "b_n": 500,
        },
        "guardrails": [
            {
                "name": "safety_violations",
                "control_successes": 24,
                "control_n": 400,
                "treated_successes": 16,
                "treated_n": 400,
            }
        ],
        "replication": None,
    }
    loaded = evidence_from_json(json.dumps(doc))
    assert loaded == evidence
    assert ship_decision(loaded_design, loaded, registry).status == "ship"
    # Fail closed: unknown keys, missing fields, wrong types.
    bad: dict[str, object] = dict(doc)
    bad["surprise_field"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        evidence_from_json(json.dumps(bad))
    bad_design = json.loads(json.dumps(design.to_dict()))
    bad_design["metrics"][0]["role"] = "side_quest"
    with pytest.raises(ValueError, match="role"):
        design_from_json(json.dumps(bad_design))
    with pytest.raises(ValueError, match="invalid JSON"):
        design_from_json("{not json")


def test_json_round_trip_with_stopped_at_and_secondary(tmp_path: str) -> None:
    registry, digest = _registry(str(tmp_path))
    design = _registered_design(
        registry,
        digest,
        metrics=(
            MetricDeclaration("task_success", "oec", "higher_better"),
            MetricDeclaration("session_length", "goal", "higher_better"),
            MetricDeclaration("safety_violations", "guardrail", "lower_better", 0.02),
        ),
    )
    evidence = _evidence(
        digest,
        n_final=400,
        looks=(LookEvidence(0.5, 3.0, True, metric="task_success"),),
        stopped_at=0.5,
        srm_counts=(198, 202),
        secondary=(MetricObservation("session_length", 200, 400, 220, 400),),
    )
    doc = {
        "plan_digest": evidence.plan_digest,
        "oec_metric": evidence.oec_metric,
        "run_id": evidence.run_id,
        "experiment": {
            "n_final": 400,
            "stopped_at": 0.5,
            "looks": [
                {"fraction": 0.5, "z": 3.0, "rejected": True, "metric": "task_success"},
            ],
        },
        "srm_counts": [198, 202],
        "aa": {
            "a_successes": 250,
            "a_n": 500,
            "b_successes": 260,
            "b_n": 500,
        },
        "guardrails": [
            {
                "name": "safety_violations",
                "control_successes": 24,
                "control_n": 400,
                "treated_successes": 16,
                "treated_n": 400,
            }
        ],
        "secondary": [
            {
                "name": "session_length",
                "control_successes": 200,
                "control_n": 400,
                "treated_successes": 220,
                "treated_n": 400,
            }
        ],
        "replication": None,
    }
    loaded = evidence_from_json(json.dumps(doc))
    assert loaded == evidence
    assert ship_decision(design, loaded, registry).status == "ship"


def test_evidence_json_rejects_non_list_observations(tmp_path: str) -> None:
    _registry_path, digest = _registry(str(tmp_path))
    doc = _evidence_doc(digest)
    doc["guardrails"] = {"name": "safety_violations"}
    with pytest.raises(TypeError, match="guardrails must be a list"):
        evidence_from_json(json.dumps(doc))
    doc = _evidence_doc(digest)
    doc["secondary"] = {"name": "session_length"}
    with pytest.raises(TypeError, match="secondary must be a list"):
        evidence_from_json(json.dumps(doc))


# ---------------------------------------------------------------------------
# CLI: `agenteval-bench ship` (covers cmd_ship for the coverage floor)
# ---------------------------------------------------------------------------


def _evidence_doc(
    digest: str,
    srm_counts: list[int] | None = None,
    final_z: float = 2.5,
    replication: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "plan_digest": digest,
        "oec_metric": "task_success",
        "run_id": "run-1",
        "experiment": {
            "n_final": 800,
            "looks": [
                {"fraction": 0.5, "z": 1.0, "rejected": False, "metric": "task_success"},
                {"fraction": 1.0, "z": final_z, "rejected": True, "metric": "task_success"},
            ],
        },
        "srm_counts": srm_counts if srm_counts is not None else [398, 402],
        "aa": {
            "a_successes": 250,
            "a_n": 500,
            "b_successes": 260,
            "b_n": 500,
        },
        "guardrails": [
            {
                "name": "safety_violations",
                "control_successes": 24,
                "control_n": 400,
                "treated_successes": 16,
                "treated_n": 400,
            }
        ],
        "replication": replication,
    }


def _write_json(doc: dict[str, object], path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc))
    return path


def _ship_files(
    tmp_path: str,
    digest: str,
    registry: str,
    srm_counts: list[int] | None = None,
    final_z: float = 2.5,
) -> tuple[str, str, str]:
    design_path = _write_json(_design(digest).to_dict(), os.path.join(tmp_path, "design.json"))
    evidence_path = _write_json(
        _evidence_doc(digest, srm_counts=srm_counts, final_z=final_z),
        os.path.join(tmp_path, "evidence.json"),
    )
    # The ship gate grades only registered designs.
    register_design(_design(digest), registry)
    return design_path, evidence_path, registry


def _run_cli(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    monkeypatch.setattr("sys.argv", ["agenteval-bench", *argv])
    try:
        cli.main()
        captured = capsys.readouterr()
        return 0, captured.out, captured.err
    except SystemExit as e:
        captured = capsys.readouterr()
        return int(e.code or 0), captured.out, captured.err


def test_cli_ship_happy_path(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry, digest = _registry(str(tmp_path))
    design_path, evidence_path, registry = _ship_files(str(tmp_path), digest, registry)
    code, out, _ = _run_cli(
        ["ship", "--design", design_path, "--evidence", evidence_path, "--registry", registry],
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert "verdict: SHIP" in out
    assert "SRM: pass" in out
    assert "A/A: pass" in out
    assert "guardrail safety_violations: pass" in out


def test_cli_ship_json_and_out(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry, digest = _registry(str(tmp_path))
    design_path, evidence_path, registry = _ship_files(str(tmp_path), digest, registry)
    out_path = os.path.join(str(tmp_path), "report.json")
    code, out, _ = _run_cli(
        [
            "ship",
            "--design",
            design_path,
            "--evidence",
            evidence_path,
            "--registry",
            registry,
            "--json",
            "--out",
            out_path,
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert json.loads(out)["status"] == "ship"
    with open(out_path, encoding="utf-8") as f:
        assert json.load(f)["status"] == "ship"


def test_cli_ship_no_ship_and_needs_replication(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry, digest = _registry(str(tmp_path))
    design_path, evidence_path, registry = _ship_files(
        str(tmp_path), digest, registry, srm_counts=[350, 450]
    )
    code, out, _ = _run_cli(
        ["ship", "--design", design_path, "--evidence", evidence_path, "--registry", registry],
        monkeypatch,
        capsys,
    )
    assert code == 1
    assert "NO SHIP" in out
    assert "SRM_FAIL" in out
    # Same design, new evidence with a surprising win: replication needed.
    evidence_path = _write_json(
        _evidence_doc(digest, final_z=4.0),
        os.path.join(str(tmp_path), "evidence-surprise.json"),
    )
    code, out, _ = _run_cli(
        ["ship", "--design", design_path, "--evidence", evidence_path, "--registry", registry],
        monkeypatch,
        capsys,
    )
    assert code == 1
    assert "NEEDS REPLICATION" in out


def test_cli_ship_bad_input_exits_2(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry, digest = _registry(str(tmp_path))
    design_path, evidence_path, registry = _ship_files(str(tmp_path), digest, registry)
    code, _, _ = _run_cli(["ship"], monkeypatch, capsys)
    assert code == 2
    code, _, _ = _run_cli(["ship", "--design", design_path], monkeypatch, capsys)
    assert code == 2
    bad_path = os.path.join(str(tmp_path), "bad.json")
    with open(bad_path, "w", encoding="utf-8") as f:
        f.write("{not json")
    code, _, _ = _run_cli(
        ["ship", "--design", bad_path, "--evidence", evidence_path, "--registry", registry],
        monkeypatch,
        capsys,
    )
    assert code == 2
    code, _, _ = _run_cli(
        ["ship", "--design", design_path, "--evidence", bad_path, "--registry", registry],
        monkeypatch,
        capsys,
    )
    assert code == 2
    code, _, _ = _run_cli(
        [
            "ship",
            "--design",
            design_path,
            "--evidence",
            evidence_path,
            "--registry",
            registry,
            "--out",
            "/nonexistent-dir/report.json",
        ],
        monkeypatch,
        capsys,
    )
    assert code == 2


def test_cli_help_lists_ship(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = _run_cli(["--help"], monkeypatch, capsys)
    assert code == 0
    assert "ship --design" in out
    assert "register-design" in out


def test_cli_register_design_round_trip(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The two-step flow: register-design first, then ship grades the
    # registered design. Shipping the same design before registering
    # it is refused.
    registry, digest = _registry(str(tmp_path))
    design_path = _write_json(_design(digest).to_dict(), os.path.join(tmp_path, "design.json"))
    evidence_path = _write_json(
        _evidence_doc(digest),
        os.path.join(tmp_path, "evidence.json"),
    )
    code, out, _ = _run_cli(
        ["ship", "--design", design_path, "--evidence", evidence_path, "--registry", registry],
        monkeypatch,
        capsys,
    )
    assert code == 1
    assert "DESIGN_NOT_REGISTERED" in out
    code, out, _ = _run_cli(
        ["register-design", "--design", design_path, "--registry", registry],
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert "registered design" in out
    code, out, _ = _run_cli(
        ["ship", "--design", design_path, "--evidence", evidence_path, "--registry", registry],
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert "verdict: SHIP" in out


def test_cli_register_design_bad_input_exits_2(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry, _digest = _registry(str(tmp_path))
    code, _, _ = _run_cli(["register-design"], monkeypatch, capsys)
    assert code == 2
    bad_path = os.path.join(str(tmp_path), "bad.json")
    with open(bad_path, "w", encoding="utf-8") as f:
        f.write("{not json")
    code, _, _ = _run_cli(
        ["register-design", "--design", bad_path, "--registry", registry],
        monkeypatch,
        capsys,
    )
    assert code == 2
    # A design bound to an unregistered plan cannot register.
    design_path = _write_json(
        _design("deadbeef" * 8).to_dict(), os.path.join(tmp_path, "design.json")
    )
    code, _, _ = _run_cli(
        ["register-design", "--design", design_path, "--registry", registry],
        monkeypatch,
        capsys,
    )
    assert code == 2


def test_cli_ship_missing_registry_exits_2(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry, digest = _registry(str(tmp_path))
    design_path, evidence_path, _ = _ship_files(str(tmp_path), digest, registry)
    missing = os.path.join(str(tmp_path), "no-such-registry.jsonl")
    code, _, err = _run_cli(
        ["ship", "--design", design_path, "--evidence", evidence_path, "--registry", missing],
        monkeypatch,
        capsys,
    )
    assert code == 2
    assert "registry" in err.lower() or "not found" in err.lower()
