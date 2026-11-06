"""Adversarial tests for Simpson-safe aggregation (agenteval-bench#36).

Every mechanism the issue demands is pinned by a test that FAILS if the
mechanism is removed or weakened: the Kohavi ch.18 reversal is named and
the pooled win claim is rejected; pooled reporting is refused on
allocation shift even without a reversal; disaggregation is returned on
every path including refusals; and there is no pooled-only path to a win
claim (empty strata, non-exhaustive decomposition).
"""

from __future__ import annotations

import pytest

from experiment.simpson import (
    AggregationViolation,
    Stratum,
    assert_aggregation,
    check_aggregation,
    disaggregate,
    pooled_rates,
    win_claim,
)


def kohavi() -> tuple[Stratum, Stratum]:
    friday = Stratum("friday", 1000, 99000, 23, 2000)
    saturday = Stratum("saturday", 5000, 5000, 60, 50)
    return friday, saturday


def test_kohavi_reversal_named_and_win_rejected() -> None:
    friday, saturday = kohavi()
    verdict = check_aggregation((friday, saturday))
    assert not verdict.valid
    assert set(verdict.violations) == {"ALLOCATION_SHIFT", "SIMPSON_REVERSAL"}
    rep = verdict.report
    assert all(r.delta > 0 for r in rep.strata)  # treated wins each period
    assert rep.pooled_delta < 0  # but loses pooled: the paradox
    claim = win_claim(
        (friday, saturday), rep.pooled_n_treated, rep.pooled_n_control
    )
    assert not claim.accepted
    assert "SIMPSON_REVERSAL" in claim.violations


def test_allocation_shift_refused_even_without_reversal() -> None:
    c = Stratum("c", 100, 9900, 5, 100)
    d = Stratum("d", 5000, 5000, 150, 100)
    verdict = check_aggregation((c, d))
    assert not verdict.valid
    assert "ALLOCATION_SHIFT" in verdict.violations
    assert "SIMPSON_REVERSAL" not in verdict.violations
    # The mechanism is gated, not just the symptom.


def test_stable_allocation_consistent_direction_accepts() -> None:
    a = Stratum("a", 5000, 5000, 115, 101)
    b = Stratum("b", 5000, 5000, 60, 50)
    verdict = check_aggregation((a, b))
    assert verdict.valid, verdict.violations
    claim = win_claim((a, b), verdict.report.pooled_n_treated,
                      verdict.report.pooled_n_control)
    assert claim.accepted


def test_no_pooled_only_path() -> None:
    with pytest.raises(ValueError):
        disaggregate(())


def test_non_exhaustive_decomposition_named() -> None:
    a = Stratum("a", 5000, 5000, 115, 101)
    b = Stratum("b", 5000, 5000, 60, 50)
    claim = win_claim((a, b), pooled_n_treated=9999, pooled_n_control=10000)
    assert not claim.accepted
    assert "NON_EXHAUSTIVE_DECOMPOSITION" in claim.violations


def test_refusal_still_returns_disaggregation() -> None:
    friday, saturday = kohavi()
    verdict = check_aggregation((friday, saturday))
    assert not verdict.valid
    assert len(verdict.report.strata) == 2
    assert verdict.report.pooled_treated_share != pytest.approx(0.5)
    with pytest.raises(AggregationViolation) as excinfo:
        assert_aggregation((friday, saturday))
    assert len(excinfo.value.report.strata) == 2  # type: ignore[attr-defined]


def test_zero_delta_stratum_is_neutral_not_a_flip() -> None:
    e = Stratum("e", 5000, 5000, 100, 100)
    f = Stratum("f", 5000, 5000, 120, 100)
    verdict = check_aggregation((e, f))
    assert verdict.valid, verdict.violations


def test_tolerance_boundary_exact() -> None:
    g = Stratum("g", 5000, 5000, 120, 100)
    h = Stratum("h", 5625, 4375, 135, 88)  # share 0.5625: spread 0.0625 exact
    assert check_aggregation((g, h), allocation_tolerance=0.0625).valid
    bad = check_aggregation((g, h), allocation_tolerance=0.0624)
    assert "ALLOCATION_SHIFT" in bad.violations


def test_stratum_contract_rejects_bad_counts() -> None:
    with pytest.raises(ValueError):
        Stratum("x", 0, 100, 0, 5)  # empty arm
    with pytest.raises(ValueError):
        Stratum("x", 100, 100, 101, 5)  # successes exceed n
    with pytest.raises(ValueError):
        disaggregate((Stratum("a", 100, 100, 5, 5), Stratum("a", 100, 100, 5, 5)))


def test_pooled_rates_are_descriptive_only() -> None:
    friday, saturday = kohavi()
    pt, pc = pooled_rates((friday, saturday))
    assert pt < pc  # the reversal, as description — quoting it as a win is refused
    with pytest.raises(AggregationViolation):
        assert_aggregation((friday, saturday))


def test_demo_stable_runs() -> None:
    """The Simpson demo is executable specification: its internal asserts
    (two strata, win claim accepted) must hold. A flipped comparison
    inside demo_stable survived the suite before this test because the
    demo ran only as a CI step, never under pytest (arena#39 nightly).
    """
    from experiment.demo_simpson import demo_stable

    demo_stable()
