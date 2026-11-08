"""Adversarial tests for code-length selection + capacity meter (arena#50).

Every behavior change ships with a running test — prose is not enforcement.
The pins that matter here are the traps:

* parameter count is not complexity: a 10M-parameter artifact of
  repeated rules must measure shorter — and win — against a 100k-parameter
  artifact of random weights at equal fit, and mutating
  ``parameter_count`` must never change a selection;
* the band rule refuses a large fit loss for code while the pure MDL
  rule accepts it — the two rules must disagree exactly there;
* the capacity meter stops only on growth WITHOUT gain: growth with
  gain continues, flat capacity continues, too few readings continue;
* the measurement is canonical (key order cannot change it) and
  compressive (repetition measures shorter than randomness).
"""

from __future__ import annotations

import json
import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agenteval_bench.code_length import (
    COMPRESSOR,
    CapacityReading,
    ModelVersion,
    SelectionReport,
    assess_capacity,
    mean_log_loss_bits,
    measure_description_bits,
    select_by_total_length,
    select_version,
)


def _version(
    name: str,
    description_bits: int,
    mean_log_loss: float,
    n: int = 500,
    parameter_count: int | None = None,
    compressor: str = COMPRESSOR,
    held_out_set: str | None = None,
) -> ModelVersion:
    return ModelVersion(
        name=name,
        description_bits=description_bits,
        held_out_n=n,
        held_out_mean_log_loss_bits=mean_log_loss,
        parameter_count=parameter_count,
        compressor=compressor,
        held_out_set=held_out_set,
    )


# --- measurement -------------------------------------------------------


def test_measure_is_canonical_under_key_order() -> None:
    table = {"b": 1, "a": [1, 2, 3], "c": {"z": "x", "y": "w"}}
    reordered = {"c": {"y": "w", "z": "x"}, "a": [1, 2, 3], "b": 1}
    assert measure_description_bits(table) == measure_description_bits(reordered)


@given(st.dictionaries(st.text(min_size=1, max_size=8), st.integers(), max_size=12))
@settings(max_examples=50)
def test_measure_is_canonical_for_arbitrary_mappings(
    mapping: dict[str, int],
) -> None:
    shuffled = dict(reversed(list(mapping.items())))
    assert measure_description_bits(mapping) == measure_description_bits(shuffled)


def test_measure_is_deterministic() -> None:
    artifact = {"rules": [{"match": "x", "route": "local"}] * 50}
    first = measure_description_bits(artifact)
    assert measure_description_bits(artifact) == first
    assert first > 0


def test_measure_repetition_shorter_than_randomness() -> None:
    rng = random.Random(11)
    repeated = {"rules": [{"match": "summarize", "route": "local"}] * 1000}
    randomish = {f"w-{i:05d}": rng.random() for i in range(1000)}
    assert measure_description_bits(repeated) < measure_description_bits(randomish)


def test_measure_refuses_non_serializable_artifact() -> None:
    with pytest.raises(TypeError, match="not JSON-serialisable"):
        measure_description_bits({"fn": object()})


def test_measure_refuses_nan_and_infinity() -> None:
    # The docstring promises loud refusal; NaN/Infinity are not JSON.
    with pytest.raises(TypeError, match="not JSON-serialisable"):
        measure_description_bits({"x": float("nan")})
    with pytest.raises(TypeError, match="not JSON-serialisable"):
        measure_description_bits({"x": float("inf")})


def test_measure_empty_artifact_is_positive() -> None:
    # zlib overhead: even an empty artifact has a code length.
    assert measure_description_bits({}) > 0


# --- ModelVersion validation --------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"name": 123},
        {"description_bits": -1},
        {"description_bits": 100.5},
        {"held_out_n": 0},
        {"held_out_n": 10.5},
        {"held_out_n": True},
        {"held_out_mean_log_loss_bits": -0.1},
        {"held_out_mean_log_loss_bits": float("nan")},
        {"held_out_mean_log_loss_bits": float("inf")},
        {"held_out_mean_log_loss_bits": "0.5"},
        {"parameter_count": -5},
        {"parameter_count": 1.5},
        {"compressor": ""},
        {"held_out_set": ""},
    ],
)
def test_model_version_refuses_bad_inputs(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "name": "v",
        "description_bits": 100,
        "held_out_n": 10,
        "held_out_mean_log_loss_bits": 0.5,
    }
    base.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        ModelVersion(**base)  # type: ignore[arg-type]


def test_total_description_length_arithmetic() -> None:
    version = _version("v", description_bits=1200, mean_log_loss=0.4, n=500)
    assert version.held_out_total_log_loss_bits == pytest.approx(200.0)
    assert version.total_description_bits == pytest.approx(1400.0)


def test_held_out_total_log_loss_refuses_overflow() -> None:
    version = _version("v", description_bits=100, mean_log_loss=0.4, n=10**400)
    with pytest.raises(ValueError, match="overflows"):
        _ = version.held_out_total_log_loss_bits


def test_from_artifact_measures_and_records_compressor() -> None:
    artifact = {"rules": [{"match": "x", "route": "local"}] * 20}
    version = ModelVersion.from_artifact(
        "v", artifact, held_out_n=10, held_out_mean_log_loss_bits=0.5
    )
    assert version.description_bits == measure_description_bits(artifact)
    assert version.compressor == COMPRESSOR


def test_mean_log_loss_bits_from_scored_pairs() -> None:
    # Certain predictions cost 0; a 0.5 prediction on a positive label costs 1 bit.
    assert mean_log_loss_bits([(1.0, True), (1.0, True)]) == pytest.approx(0.0)
    assert mean_log_loss_bits([(0.5, True), (0.5, False)]) == pytest.approx(1.0)
    assert mean_log_loss_bits([(0.25, True)]) == pytest.approx(2.0)


def test_mean_log_loss_bits_refuses_bad_pairs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mean_log_loss_bits([])
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        mean_log_loss_bits([(0.0, True)])
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        mean_log_loss_bits([(1.5, True)])
    with pytest.raises(ValueError, match="infinite log-loss"):
        mean_log_loss_bits([(1.0, False)])


# --- selection ----------------------------------------------------------


def test_parameter_count_cannot_change_selection() -> None:
    versions = [
        _version("a", 9000, 0.40, parameter_count=10_000_000),
        _version("b", 1200, 0.40, parameter_count=100),
    ]
    first = select_version(versions)
    mutated = [
        _version("a", 9000, 0.40, parameter_count=3),
        _version("b", 1200, 0.40, parameter_count=99_000_000),
    ]
    second = select_version(mutated)
    assert first.winner.name == second.winner.name == "b"
    assert first.winner.total_description_bits == second.winner.total_description_bits


def test_repeated_rules_beat_random_weights_at_equal_fit() -> None:
    rng = random.Random(7)
    repeated_bits = measure_description_bits(
        {"rules": [{"match": "summarize", "route": "local"}] * 1000}
    )
    random_bits = measure_description_bits({f"w-{i:05d}": rng.random() for i in range(1000)})
    report = select_version(
        [
            _version("big-repeated", repeated_bits, 0.40, parameter_count=10_000_000),
            _version("small-random", random_bits, 0.40, parameter_count=100_000),
        ]
    )
    assert report.winner.name == "big-repeated"
    assert report.fit_deficit_bits == 0.0


def test_equal_performance_picks_shorter_code_and_names_the_trade() -> None:
    report = select_version(
        [
            _version("long", 9000, 0.395),
            _version("short", 1200, 0.400),
        ],
        equivalence_band_bits=0.01,
    )
    assert report.winner.name == "short"
    assert report.fit_deficit_bits == pytest.approx(0.005)
    # Total saving: best total 9197.5 - winner total 1400.0 = 7797.5,
    # not the description-only difference of 7800.0.
    assert report.code_saving_bits == 7797.5
    assert report.best_fit.name == "long"
    assert "short" in report.rationale and "7797.5" in report.rationale


def test_band_rule_refuses_large_fit_loss_that_mdl_accepts() -> None:
    versions = [
        _version("long", 9000, 0.395),
        _version("short", 1200, 0.400),
        _version("tiny", 100, 0.450),
    ]
    banded = select_version(versions, equivalence_band_bits=0.01)
    assert banded.winner.name == "short"
    assert banded.equivalent == (versions[0], versions[1])
    unbanded = select_by_total_length(versions)
    assert unbanded.winner.name == "tiny"
    assert unbanded.equivalence_band_bits is None


def test_selection_outside_band_keeps_the_better_fit() -> None:
    report = select_version(
        [_version("fit", 9000, 0.30), _version("short", 1200, 0.40)],
        equivalence_band_bits=0.01,
    )
    assert report.winner.name == "fit"
    assert report.equivalent == (report.best_fit,)


def test_selection_with_a_single_version() -> None:
    only = _version("only", 500, 0.30)
    report = select_version([only])
    assert report.winner is only
    assert report.fit_deficit_bits == 0.0
    assert report.code_saving_bits == 0.0


def test_selection_tie_break_is_deterministic() -> None:
    report = select_version([_version("zeta", 1000, 0.40), _version("alpha", 1000, 0.40)])
    assert report.winner.name == "alpha"
    again = select_version([_version("alpha", 1000, 0.40), _version("zeta", 1000, 0.40)])
    assert again.winner.name == "alpha"


def test_selection_refuses_empty_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="no versions"):
        select_version([])
    with pytest.raises(ValueError, match="unique"):
        select_version([_version("dup", 100, 0.4), _version("dup", 200, 0.3)])
    with pytest.raises(ValueError, match="equivalence_band_bits"):
        select_version([_version("v", 100, 0.4)], equivalence_band_bits=-0.1)


def test_selection_refuses_mixed_held_out_n() -> None:
    # The trap that escaped: totals over different n compare set size,
    # not quality — a worse fit on n=1000 must not lose to n=100.
    with pytest.raises(ValueError, match="held_out_n"):
        select_version([_version("a", 100, 0.40, n=100), _version("b", 100, 0.39, n=1000)])
    with pytest.raises(ValueError, match="held_out_n"):
        select_by_total_length([_version("a", 100, 0.40, n=100), _version("b", 100, 0.39, n=1000)])


def test_selection_refuses_mixed_compressor() -> None:
    with pytest.raises(ValueError, match="compressor"):
        select_version(
            [
                _version("a", 100, 0.40),
                _version("b", 100, 0.40, compressor="gzip-9"),
            ]
        )


def test_selection_refuses_mixed_held_out_set() -> None:
    with pytest.raises(ValueError, match="held_out_set"):
        select_version(
            [
                _version("a", 100, 0.40, held_out_set="split-a"),
                _version("b", 100, 0.40, held_out_set="split-b"),
            ]
        )
    with pytest.raises(ValueError, match="held_out_set"):
        select_version(
            [
                _version("a", 100, 0.40, held_out_set="split-a"),
                _version("b", 100, 0.40),
            ]
        )


def test_rationale_qualifies_shortest_total_code() -> None:
    # Winner is the best fit but NOT the global shortest total: a
    # shorter-total version sits outside the band. The rationale must
    # not call the winner the shortest total code, unqualified.
    report = select_version(
        [_version("fit", 9000, 0.30), _version("tiny", 100, 0.45)],
        equivalence_band_bits=0.01,
    )
    assert report.winner.name == "fit"
    assert "among the equivalent versions" in report.rationale
    assert report.rationale.count("shortest total code") == 1


def test_selection_report_refuses_bad_reports() -> None:
    versions = [_version("a", 100, 0.4), _version("b", 200, 0.3)]
    base: dict[str, object] = {
        "winner": versions[0],
        "candidates": tuple(versions),
        "equivalent": tuple(versions),
        "best_fit": versions[1],
        "fit_deficit_bits": 0.1,
        "code_saving_bits": 100.0,
        "equivalence_band_bits": 0.01,
        "rule": "test",
    }
    with pytest.raises(ValueError, match="code_saving_bits"):
        SelectionReport(**{**base, "code_saving_bits": -1.0})  # type: ignore[arg-type]
    phantom = _version("ghost", 50, 0.2)
    with pytest.raises(ValueError, match="equivalent"):
        SelectionReport(**{**base, "equivalent": (phantom,)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="equivalence_band_bits"):
        SelectionReport(**{**base, "equivalence_band_bits": math.inf})  # type: ignore[arg-type]


def test_selection_report_to_dict_round_trips() -> None:
    report = select_version([_version("a", 9000, 0.395), _version("b", 1200, 0.400)])
    doc = report.to_dict()
    assert doc["winner"] == "b"
    assert doc["rule"] == "equal-performance code length"
    assert doc["compressor"] == COMPRESSOR
    assert len(doc["candidates"]) == 2  # type: ignore[arg-type]
    assert doc["equivalent"] == ["a", "b"]
    json.dumps(doc, allow_nan=False)


def test_unbanded_report_round_trips_strict_json() -> None:
    report = select_by_total_length([_version("a", 9000, 0.395), _version("b", 1200, 0.400)])
    doc = report.to_dict()
    assert doc["equivalence_band_bits"] is None
    parsed = json.loads(json.dumps(doc, allow_nan=False))
    assert parsed["winner"] == "b"


# --- capacity meter ------------------------------------------------------


def _stalling_series() -> list[CapacityReading]:
    """Capacity +100/step; held-out improves to step 2, then flat."""
    return [
        CapacityReading(
            step=0,
            capacity_bits=1000,
            held_out_mean_log_loss_bits=0.40,
            train_mean_log_loss_bits=0.38,
        ),
        CapacityReading(
            step=1,
            capacity_bits=1100,
            held_out_mean_log_loss_bits=0.39,
            train_mean_log_loss_bits=0.36,
        ),
        CapacityReading(
            step=2,
            capacity_bits=1200,
            held_out_mean_log_loss_bits=0.38,
            train_mean_log_loss_bits=0.34,
        ),
        CapacityReading(
            step=3,
            capacity_bits=1300,
            held_out_mean_log_loss_bits=0.38,
            train_mean_log_loss_bits=0.32,
        ),
        CapacityReading(
            step=4,
            capacity_bits=1400,
            held_out_mean_log_loss_bits=0.38,
            train_mean_log_loss_bits=0.30,
        ),
        CapacityReading(
            step=5,
            capacity_bits=1500,
            held_out_mean_log_loss_bits=0.38,
            train_mean_log_loss_bits=0.28,
        ),
    ]


def test_meter_stops_on_growth_without_gain() -> None:
    verdict = assess_capacity(_stalling_series())
    assert verdict.decision == "stop"
    assert verdict.capacity_growth_bits == 200
    assert verdict.held_out_best_before_bits == pytest.approx(0.38)
    assert verdict.held_out_best_in_window_bits == pytest.approx(0.38)
    assert verdict.max_generalization_gap_bits == pytest.approx(0.10)
    assert verdict.window[0].step == 3 and verdict.window[-1].step == 5
    assert "200 bits" in verdict.reason


def test_meter_stops_at_the_first_stalled_window() -> None:
    readings = _stalling_series()
    assert assess_capacity(readings[:5]).decision == "continue"
    assert assess_capacity(readings[:6]).decision == "stop"


def test_meter_continues_when_generalisation_keeps_improving() -> None:
    improving = [
        CapacityReading(
            step=s,
            capacity_bits=1000 + 100 * s,
            held_out_mean_log_loss_bits=0.40 - 0.02 * s,
            train_mean_log_loss_bits=0.38 - 0.02 * s,
        )
        for s in range(6)
    ]
    verdict = assess_capacity(improving)
    assert verdict.decision == "continue"
    assert verdict.capacity_growth_bits == 200


def test_meter_continues_when_capacity_is_flat() -> None:
    flat = [
        CapacityReading(
            step=s,
            capacity_bits=1000,
            held_out_mean_log_loss_bits=0.40,
            train_mean_log_loss_bits=0.38,
        )
        for s in range(6)
    ]
    verdict = assess_capacity(flat)
    assert verdict.decision == "continue"
    assert verdict.capacity_growth_bits == 0


def test_meter_continues_when_capacity_shrinks() -> None:
    shrinking = [
        CapacityReading(
            step=s,
            capacity_bits=1500 - 100 * s,
            held_out_mean_log_loss_bits=0.40,
            train_mean_log_loss_bits=0.38,
        )
        for s in range(6)
    ]
    verdict = assess_capacity(shrinking)
    assert verdict.decision == "continue"
    assert verdict.capacity_growth_bits == -200


def test_meter_continues_with_too_few_readings() -> None:
    verdict = assess_capacity(_stalling_series()[:3])
    assert verdict.decision == "continue"
    assert verdict.held_out_best_before_bits is None
    assert "no baseline" in verdict.reason


def test_meter_stops_when_improvement_is_exactly_at_the_bar() -> None:
    # Improvement 0.125 == bar 0.125: "more than the bar" is required,
    # so exactly-at-the-bar does not count as improvement.
    readings = [
        CapacityReading(
            step=0,
            capacity_bits=1000,
            held_out_mean_log_loss_bits=0.5,
            train_mean_log_loss_bits=0.5,
        ),
        CapacityReading(
            step=1,
            capacity_bits=1100,
            held_out_mean_log_loss_bits=0.5,
            train_mean_log_loss_bits=0.5,
        ),
        CapacityReading(
            step=2,
            capacity_bits=1200,
            held_out_mean_log_loss_bits=0.375,
            train_mean_log_loss_bits=0.375,
        ),
        CapacityReading(
            step=3,
            capacity_bits=1300,
            held_out_mean_log_loss_bits=0.375,
            train_mean_log_loss_bits=0.375,
        ),
        CapacityReading(
            step=4,
            capacity_bits=1400,
            held_out_mean_log_loss_bits=0.375,
            train_mean_log_loss_bits=0.375,
        ),
    ]
    verdict = assess_capacity(readings, patience=3, min_improvement_bits=0.125)
    assert verdict.decision == "stop"


def test_meter_reports_the_generalisation_gap() -> None:
    reading = CapacityReading(
        step=0, capacity_bits=100, held_out_mean_log_loss_bits=0.50, train_mean_log_loss_bits=0.30
    )
    assert reading.generalization_gap_bits == pytest.approx(0.20)


def test_meter_refuses_bad_series() -> None:
    with pytest.raises(ValueError, match="no readings"):
        assess_capacity([])
    with pytest.raises(ValueError, match="strictly increasing"):
        assess_capacity(
            [
                CapacityReading(
                    step=1,
                    capacity_bits=100,
                    held_out_mean_log_loss_bits=0.4,
                    train_mean_log_loss_bits=0.4,
                ),
                CapacityReading(
                    step=1,
                    capacity_bits=200,
                    held_out_mean_log_loss_bits=0.4,
                    train_mean_log_loss_bits=0.4,
                ),
            ]
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        assess_capacity(list(reversed(_stalling_series())))
    with pytest.raises(ValueError, match="patience"):
        assess_capacity(_stalling_series(), patience=1)


def test_meter_verdict_to_dict() -> None:
    verdict = assess_capacity(_stalling_series())
    doc = verdict.to_dict()
    assert doc["decision"] == "stop"
    assert doc["window_steps"] == [3, 4, 5]
    assert doc["capacity_growth_bits"] == 200
