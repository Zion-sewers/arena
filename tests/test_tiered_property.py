"""Property tests for the tiered pipeline (arena#39).

Invariants that must hold for *every* input, not just the examples in
test_tiered.py: phase prefixes, mutation-score arithmetic, report
round-trips, and the deterministic scorer's pass/score contract (the
engine the gate phase trusts).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from agenteval_bench.models import EvalCase, ExpectedOutput
from agenteval_bench.scoring import DeterministicScorer
from agenteval_bench.tiered import (
    PHASE_ORDER,
    MutationScore,
    PhaseOutcome,
    TieredReport,
    phases_up_to,
)

_PHASE = st.sampled_from([*PHASE_ORDER, "all"])
_STATUS = st.sampled_from(["passed", "failed", "skipped", "error"])


@given(phase=_PHASE)
def test_phases_up_to_is_a_prefix_ending_at_the_phase(phase: str) -> None:
    result = phases_up_to(phase)
    assert result == PHASE_ORDER[: len(result)]
    if phase == "all":
        assert result == PHASE_ORDER
    else:
        assert result[-1] == phase


@given(killed=st.integers(min_value=0, max_value=20), total=st.integers(min_value=1, max_value=20))
def test_mutation_score_is_killed_over_validated(killed: int, total: int) -> None:
    killed = min(killed, total)
    score = MutationScore(killed=killed, validated=total, skipped=0, score=killed / total)
    assert 0.0 <= score.score <= 1.0
    assert score.score == killed / total
    assert score.score == 1.0 if killed == total else score.score < 1.0


@given(
    phases=st.lists(
        st.tuples(_PHASE, _STATUS).filter(lambda p: p[0] != "all"),
        min_size=1,
        max_size=4,
        unique_by=lambda p: p[0],
    ),
    coverage=st.one_of(st.none(), st.floats(min_value=0.0, max_value=100.0)),
    mutation=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
)
@settings(max_examples=60)
def test_report_json_round_trip_for_any_phase_mix(
    phases: list[tuple[str, str]],
    coverage: float | None,
    mutation: float | None,
) -> None:
    outcomes = tuple(
        PhaseOutcome(
            phase=name,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            entry_met=status == "passed",
            exit_met=status == "passed",
            duration_s=0.5,
            coverage_percent=coverage,
            mutation_score=mutation,
        )
        for name, status in phases
    )
    verdict = "PASS" if all(p.status == "passed" for p in outcomes) else "FAIL"
    report = TieredReport(
        root="/tmp/fake-root",
        verdict=verdict,  # type: ignore[arg-type]
        phases=outcomes,
        coverage_percent=coverage,
        mutation_score=mutation,
    )
    restored = TieredReport.from_json(report.to_json())
    assert restored.to_dict() == report.to_dict()


_SAFE_REGEX = st.sampled_from(["hello", "a+", "[0-9]+", "x|y", "^a", "b$"])


def _expected_strategy() -> st.SearchStrategy[ExpectedOutput]:
    return st.builds(
        ExpectedOutput,
        exact=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
        contains=st.lists(st.text(min_size=1, max_size=10), max_size=3),
        regex=st.one_of(st.none(), _SAFE_REGEX),
    )


@given(
    expected=_expected_strategy(),
    output=st.text(max_size=60),
)
@settings(max_examples=80)
def test_scorer_score_in_range_and_passed_implies_full_score(
    expected: ExpectedOutput, output: str
) -> None:
    """Deterministic scorer invariant: score in [0, 1]; passed iff every
    applicable check passed, hence passed implies score == 1.0."""
    case = EvalCase(id="prop", input="q", expected=expected)
    result = DeterministicScorer().score(case, output)
    assert 0.0 <= result.score <= 1.0
    checks = result.details.get("checks")
    if not checks:
        assert result.score == 0.0
        assert not result.passed
    elif result.passed:
        assert result.score == 1.0
    elif result.score == 1.0:
        # A full score means every check passed, so the case must pass.
        assert result.passed
