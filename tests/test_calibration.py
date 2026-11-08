"""Tests for judge calibration (arena#45): "confidence is not evidence".

Every acceptance criterion has a named test, and the property tests at the
bottom pin the invariants for every generated observation set: ECE stays
in [0, 1], the bins account for every observation exactly once, and
calibration is deterministic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agenteval_bench.calibration import (
    CalibrationBin,
    CalibrationPolicy,
    ConfidenceObservation,
    JudgeCalibration,
    UncalibratedJudgeError,
    calibrate,
    may_gate,
    observations_from_alt_test,
    observations_from_json,
    require_calibrated,
)
from agenteval_bench.cli import cmd_calibrate
from alt_test.types import AltTestItem, Label

JUDGE = "sharp-judge"


def _obs(
    judge: str = JUDGE,
    n: int = 1,
    confidence: float = 0.5,
    agreed: bool = True,
    start: int = 0,
) -> list[ConfidenceObservation]:
    return [
        ConfidenceObservation(
            judge=judge,
            item_id=f"item-{start + i}",
            stated_confidence=confidence,
            agreed=agreed,
        )
        for i in range(n)
    ]


def _mixed(
    judge: str = JUDGE,
    confidence: float = 0.5,
    n_agree: int = 0,
    n_disagree: int = 0,
    start: int = 0,
) -> list[ConfidenceObservation]:
    return _obs(judge, n_agree, confidence, True, start) + _obs(
        judge, n_disagree, confidence, False, start + n_agree
    )


# --- Boundary validation -------------------------------------------------


def test_confidence_out_of_range_is_refused() -> None:
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        ConfidenceObservation(JUDGE, "i1", 1.01, True)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        ConfidenceObservation(JUDGE, "i1", -0.01, True)


def test_confidence_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        ConfidenceObservation(JUDGE, "i1", math.nan, True)
    with pytest.raises(ValueError, match="finite"):
        ConfidenceObservation(JUDGE, "i1", math.inf, True)


def test_agreed_must_be_a_bool_not_an_int() -> None:
    # 1 is truthy but it is not a measured agreement: the boundary refuses it.
    with pytest.raises(TypeError, match="must be a bool"):
        ConfidenceObservation(JUDGE, "i1", 0.5, 1)  # type: ignore[arg-type]


def test_empty_judge_or_item_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ConfidenceObservation("", "i1", 0.5, True)
    with pytest.raises(ValueError, match="non-empty"):
        ConfidenceObservation(JUDGE, "", 0.5, True)


def test_policy_bounds_are_refused() -> None:
    with pytest.raises(ValueError, match="min_total"):
        CalibrationPolicy(min_total=0)
    with pytest.raises(ValueError, match="tolerance"):
        CalibrationPolicy(tolerance=1.5)


def test_mixed_judge_observations_are_refused_not_mixed_in() -> None:
    observations = _obs(JUDGE, 60, 0.5, True) + _obs("other-judge", 1, 0.5, True)
    with pytest.raises(ValueError, match="mixed into calibration"):
        calibrate(JUDGE, observations)


def test_duplicate_observations_are_refused_by_calibrate() -> None:
    # Copies of one observation cannot inflate n: one item is one
    # measurement, wherever the observations came from.
    observations = [ConfidenceObservation(JUDGE, "same-item", 0.5, True)] * 50
    with pytest.raises(ValueError, match="duplicate observation"):
        calibrate(JUDGE, observations)


def test_last_bin_range_is_closed() -> None:
    record = calibrate(JUDGE, _mixed(confidence=1.0, n_agree=50, n_disagree=50))
    assert record.status == "miscalibrated"
    assert "[0.9, 1.0]" in record.reasons[0]
    assert "[0.9, 1.0)" not in record.reasons[0]


def test_bin_range_label_keeps_non_decile_edges_honest() -> None:
    from agenteval_bench.calibration import bin_range_label

    assert bin_range_label(1 / 3, 2 / 3) == "[0.3333, 0.6667)"
    assert bin_range_label(0.9, 1.0) == "[0.9, 1.0]"
    assert bin_range_label(0.0, 0.1) == "[0, 0.1)"


def test_record_field_validation() -> None:
    good = calibrate(JUDGE, _mixed(confidence=0.5, n_agree=100, n_disagree=100))
    with pytest.raises(ValueError, match="non-empty"):
        JudgeCalibration(
            judge="",
            status="calibrated",
            n_total=200,
            ece=0.0,
            bins=good.bins,
        )
    with pytest.raises(ValueError, match="n_total must be >= 0"):
        JudgeCalibration(
            judge=JUDGE,
            status="uncalibrated",
            n_total=-1,
            ece=None,
        )
    with pytest.raises(ValueError, match="ece must be in \\[0, 1\\]"):
        JudgeCalibration(
            judge=JUDGE,
            status="calibrated",
            n_total=200,
            ece=1.5,
            bins=good.bins,
        )


# --- The grade ------------------------------------------------------------


def test_well_calibrated_judge_passes() -> None:
    # 200 observations at confidence 0.5, 100 agreed: the Wilson lower
    # bound (0.43) sits within tolerance of the stated 0.5.
    record = calibrate(JUDGE, _mixed(confidence=0.5, n_agree=100, n_disagree=100))
    assert record.status == "calibrated"
    assert record.n_total == 200
    assert record.ece == 0.0
    assert record.reasons == ()
    assert may_gate(record) is True
    require_calibrated(record)  # passes silently


def test_perfect_agreement_at_confidence_one_is_calibrated() -> None:
    record = calibrate(JUDGE, _obs(n=50, confidence=1.0, agreed=True))
    assert record.status == "calibrated"
    assert record.ece == 0.0
    assert len(record.bins) == 1 and record.bins[0].bin_index == 9


def test_confidence_one_at_fifty_percent_agreement_is_miscalibrated() -> None:
    # The headline trap: maximal self-reported confidence, coin-flip
    # measured agreement. The gate must refuse it.
    record = calibrate(JUDGE, _mixed(confidence=1.0, n_agree=50, n_disagree=50))
    assert record.status == "miscalibrated"
    assert record.bins[0].bin_index == 9
    assert record.bins[0].gap > 0.10
    assert may_gate(record) is False
    with pytest.raises(UncalibratedJudgeError, match="miscalibrated"):
        require_calibrated(record)


def test_gate_error_names_the_judge_status_and_reasons() -> None:
    record = calibrate(JUDGE, _mixed(confidence=1.0, n_agree=50, n_disagree=50))
    with pytest.raises(UncalibratedJudgeError) as exc:
        require_calibrated(record)
    assert JUDGE in str(exc.value)
    assert "miscalibrated" in str(exc.value)
    assert "never gate anything" in str(exc.value)


def test_five_out_of_five_is_uncalibrated_not_calibrated() -> None:
    # The small-n trap: a point estimate of 1.0 with a Wilson lower bound
    # of 0.565 is not a measurement that clears the bar.
    record = calibrate(JUDGE, _obs(n=5, confidence=1.0, agreed=True))
    assert record.status == "uncalibrated"
    assert record.ece is None
    assert may_gate(record) is False
    with pytest.raises(UncalibratedJudgeError, match="uncalibrated"):
        require_calibrated(record)


def test_no_observations_is_uncalibrated_by_definition() -> None:
    record = calibrate(JUDGE, [])
    assert record.status == "uncalibrated"
    assert record.n_total == 0
    assert record.ece is None
    assert "no observations" in record.reasons[0]


def test_thin_bin_is_uncalibrated_not_a_pass() -> None:
    # 50 observations total, but one bin holds a single observation:
    # the thin bin is unmeasured, so the judge is uncalibrated.
    observations = _mixed(confidence=0.9, n_agree=49, n_disagree=0) + _obs(
        n=1, confidence=0.5, agreed=True, start=100
    )
    record = calibrate(JUDGE, observations)
    assert record.status == "uncalibrated"
    assert "bin 5" in record.reasons[0]


def test_confidence_zero_bin_is_pinned() -> None:
    # Confidence 0.0 lands in bin 0; zero claimed, zero agreed: calibrated.
    record = calibrate(JUDGE, _obs(n=50, confidence=0.0, agreed=False))
    assert record.status == "calibrated"
    assert record.bins[0].bin_index == 0
    assert record.bins[0].mean_confidence == 0.0


def test_ece_is_a_diagnostic_not_the_grade() -> None:
    # Agreement 0.5 at stated 0.5 in every bin: ECE is 0 and the judge
    # passes. Raise the stated confidence to 0.6 with the same 0.5
    # agreement and the lower-bound grade fails even though the ECE
    # (0.1) is small — a small point-estimate error cannot smuggle an
    # under-measured judge through.
    good = calibrate(JUDGE, _mixed(confidence=0.5, n_agree=100, n_disagree=100))
    assert good.ece == 0.0 and good.status == "calibrated"
    bad = calibrate(JUDGE, _mixed(confidence=0.6, n_agree=100, n_disagree=100))
    assert bad.ece == pytest.approx(0.1)
    assert bad.status == "miscalibrated"


def test_conservative_judge_is_noted_not_failed() -> None:
    # Agreement far above stated confidence is safe, not a failure:
    # the bin passes and the conservatism is a note.
    record = calibrate(JUDGE, _obs(n=100, confidence=0.1, agreed=True))
    assert record.status == "calibrated"
    assert record.notes, "conservative bins must be noted"
    assert "conservative" in record.notes[0]


def test_record_round_trips_through_json() -> None:
    record = calibrate(JUDGE, _mixed(confidence=0.5, n_agree=100, n_disagree=100))
    parsed = json.loads(json.dumps(record.to_dict()))
    assert parsed["judge"] == JUDGE
    assert parsed["status"] == "calibrated"
    assert parsed["n_total"] == 200
    assert parsed["ece"] == 0.0
    assert len(parsed["bins"]) == 1
    assert parsed["policy"]["tolerance"] == 0.10
    assert parsed["policy"]["min_total"] == 50


def test_policy_is_recorded_on_the_grade() -> None:
    policy = CalibrationPolicy(min_total=10, bins=5, min_bin_n=1, tolerance=0.5)
    record = calibrate(JUDGE, _mixed(confidence=0.5, n_agree=10, n_disagree=10), policy)
    assert record.policy == policy
    assert record.policy.tolerance == 0.5


def test_empty_and_negative_bins_are_refused() -> None:
    # The arch re-review FAIL: an empty bin let a zero-observation
    # CALIBRATED record through (0 == 0 on the n_total check).
    with pytest.raises(ValueError, match="n >= 1"):
        CalibrationBin(9, 0.9, 1.0, 0, 0.0, 0, True)
    with pytest.raises(ValueError, match="n >= 1"):
        CalibrationBin(9, 0.9, 1.0, -5, 0.0, -5, True)
    with pytest.raises(ValueError, match="out of range"):
        CalibrationBin(9, 0.9, 1.0, 50, 50.0, 51, True)
    with pytest.raises(ValueError, match="out of range"):
        CalibrationBin(9, 0.9, 1.0, 50, 51.0, 50, True)


def test_bin_derives_its_interval_from_the_counts() -> None:
    # A forged interval has nowhere to live: the interval is recomputed
    # from (agreed, n) on every read.
    from agenteval_bench.decision import wilson_interval

    b = CalibrationBin(9, 0.9, 1.0, 50, 45.0, 25, True)
    assert b.mean_confidence == pytest.approx(0.9)
    assert b.agreement_rate == pytest.approx(0.5)
    assert b.agreement_interval == wilson_interval(25, 50)
    assert b.gap == pytest.approx(0.9 - wilson_interval(25, 50)[0])
    assert b.to_dict()["agreed"] == 25


def test_ece_mismatch_is_refused() -> None:
    good = calibrate(JUDGE, _mixed(confidence=0.5, n_agree=100, n_disagree=100))
    with pytest.raises(ValueError, match="does not match the bins"):
        JudgeCalibration(
            judge=JUDGE,
            status="calibrated",
            n_total=200,
            ece=0.999,
            bins=good.bins,
        )


def test_hand_built_record_below_policy_floors_is_refused() -> None:
    small_policy = CalibrationPolicy(min_total=10, bins=5, min_bin_n=1, tolerance=0.5)
    small = calibrate(
        JUDGE, _mixed(confidence=0.5, n_agree=10, n_disagree=10), small_policy
    )
    assert small.status == "calibrated"
    with pytest.raises(ValueError, match="below the record policy's min_total"):
        JudgeCalibration(
            judge=JUDGE,
            status="calibrated",
            n_total=small.n_total,
            ece=small.ece,
            bins=small.bins,
        )


def test_forged_calibrated_record_is_refused_at_construction() -> None:
    # The arch FAIL: a hand-built CALIBRATED record with no measurement
    # behind it must not reach the gate.
    with pytest.raises(ValueError, match="without a measurement"):
        JudgeCalibration(judge=JUDGE, status="calibrated", n_total=0, ece=None)
    good = calibrate(JUDGE, _mixed(confidence=0.5, n_agree=100, n_disagree=100))
    failed_bin = CalibrationBin(
        bin_index=9,
        lower_edge=0.9,
        upper_edge=1.0,
        n=50,
        confidence_sum=50.0,
        agreed=25,
        passed=False,
    )
    with pytest.raises(ValueError, match="failed bin"):
        JudgeCalibration(
            judge=JUDGE,
            status="calibrated",
            n_total=50,
            ece=0.5,
            bins=(failed_bin,),
        )
    with pytest.raises(ValueError, match="cannot carry reasons"):
        JudgeCalibration(
            judge=JUDGE,
            status="calibrated",
            n_total=200,
            ece=0.0,
            bins=(good.bins[0],),
            reasons=("forged",),
        )
    with pytest.raises(ValueError, match="does not match"):
        JudgeCalibration(
            judge=JUDGE,
            status="calibrated",
            n_total=199,
            ece=0.0,
            bins=good.bins,
        )


def test_forged_miscalibrated_and_uncalibrated_records_are_refused() -> None:
    good = calibrate(JUDGE, _mixed(confidence=0.5, n_agree=100, n_disagree=100))
    with pytest.raises(ValueError, match="non-empty reasons"):
        JudgeCalibration(
            judge=JUDGE,
            status="miscalibrated",
            n_total=200,
            ece=0.1,
            bins=good.bins,
        )
    with pytest.raises(ValueError, match="cannot carry bins"):
        JudgeCalibration(
            judge=JUDGE,
            status="uncalibrated",
            n_total=200,
            ece=None,
            bins=good.bins,
        )


def test_package_exports_calibration_names() -> None:
    from agenteval_bench import calibrate as package_calibrate
    from agenteval_bench import may_gate as package_may_gate

    assert package_calibrate is calibrate
    assert package_may_gate is may_gate


# --- Alt-test bridge ------------------------------------------------------


def _alt_items() -> list[AltTestItem]:
    return [
        AltTestItem(
            id="a",
            task="discrete",
            annotator_labels={"h1": "yes", "h2": "yes", "h3": "no"},
            judge_labels={JUDGE: "yes"},
        ),
        AltTestItem(
            id="b",
            task="discrete",
            annotator_labels={"h1": "yes", "h2": "no", "h3": "no"},
            judge_labels={JUDGE: "yes"},
        ),
        AltTestItem(
            id="c",
            task="discrete",
            annotator_labels={"h1": "yes", "h2": "yes", "h3": "yes"},
            judge_labels={},
        ),
        AltTestItem(
            id="d",
            task="discrete",
            annotator_labels={"h1": "yes", "h2": "no", "h3": "yes"},
            judge_labels={JUDGE: "yes"},
        ),
    ]


def test_bridge_derives_agreement_from_annotator_majority() -> None:
    bridged = observations_from_alt_test(
        _alt_items(), {"a": 0.9, "b": 0.8, "d": 0.7}, JUDGE
    )
    assert [o.item_id for o in bridged.observations] == ["a", "b", "d"]
    assert bridged.observations[0].agreed is True  # "yes" == majority "yes"
    assert bridged.observations[1].agreed is False  # "yes" vs majority "no"
    assert bridged.observations[2].agreed is True  # "yes" == majority "yes"
    assert bridged.skipped == ()
    # Item c carries no judge label: not scored, not reported as skipped.


def test_bridge_names_tied_majority_and_missing_confidence() -> None:
    items = [
        AltTestItem(
            id="tie",
            task="discrete",
            annotator_labels={"h1": "yes", "h2": "yes", "h3": "no", "h4": "no"},
            judge_labels={JUDGE: "yes"},
        ),
        AltTestItem(
            id="noconf",
            task="discrete",
            annotator_labels={"h1": "yes", "h2": "yes", "h3": "yes"},
            judge_labels={JUDGE: "yes"},
        ),
    ]
    bridged = observations_from_alt_test(items, {}, JUDGE)
    assert bridged.observations == ()
    assert [s.item_id for s in bridged.skipped] == ["tie", "noconf"]
    assert "tied" in bridged.skipped[0].reason
    assert "no confidence" in bridged.skipped[1].reason


def test_bridge_refuses_items_with_fewer_than_three_annotators() -> None:
    # A 1- or 2-annotator "majority" is not a majority (alt-test needs >= 3).
    items = [
        AltTestItem(
            id="thin",
            task="discrete",
            annotator_labels={"h1": "yes", "h2": "yes"},
            judge_labels={JUDGE: "yes"},
        )
    ]
    with pytest.raises(ValueError, match="need at least 3"):
        observations_from_alt_test(items, {"thin": 0.9}, JUDGE)


def test_bridge_ignores_thin_items_the_judge_never_labelled() -> None:
    # A thin item the judge never labelled is never scored, so it blocks
    # nothing; only labelled items need a usable majority reference.
    items = [
        AltTestItem(
            id="thin-unlabelled",
            task="discrete",
            annotator_labels={"h1": "yes"},
            judge_labels={},
        ),
        AltTestItem(
            id="good",
            task="discrete",
            annotator_labels={"h1": "yes", "h2": "yes", "h3": "no"},
            judge_labels={JUDGE: "yes"},
        ),
    ]
    bridged = observations_from_alt_test(items, {"good": 0.9}, JUDGE)
    assert [o.item_id for o in bridged.observations] == ["good"]


def test_bridge_refuses_wrong_typed_labels() -> None:
    # A float judge label on a discrete task is refused at the boundary,
    # not silently counted as disagreement.
    items = [
        AltTestItem(
            id="badtype",
            task="discrete",
            annotator_labels={"h1": "yes", "h2": "yes", "h3": "yes"},
            judge_labels={JUDGE: 1.0},
        )
    ]
    with pytest.raises(TypeError, match="string labels"):
        observations_from_alt_test(items, {"badtype": 0.9}, JUDGE)
    bool_items = [
        AltTestItem(
            id="booltype",
            task="discrete",
            annotator_labels={
                "h1": cast(Label, True),
                "h2": cast(Label, True),
                "h3": cast(Label, True),
            },
            judge_labels={JUDGE: "yes"},
        )
    ]
    with pytest.raises(TypeError, match="must not be a bool"):
        observations_from_alt_test(bool_items, {"booltype": 0.9}, JUDGE)


def test_bridge_refuses_non_discrete_tasks() -> None:
    items = [
        AltTestItem(
            id="x",
            task="continuous",
            annotator_labels={"h1": 1.0, "h2": 1.0, "h3": 1.0},
            judge_labels={JUDGE: 1.0},
        )
    ]
    with pytest.raises(ValueError, match="discrete tasks only"):
        observations_from_alt_test(items, {"x": 0.9}, JUDGE)


# --- JSON loader ----------------------------------------------------------


def test_loader_accepts_mapping_and_bare_list() -> None:
    judge, observations = observations_from_json(
        json.dumps(
            {
                "judge": JUDGE,
                "observations": [
                    {"item": "i1", "confidence": 0.9, "agreed": True},
                    {"item": "i2", "confidence": 0.9, "agreed": False},
                ],
            }
        )
    )
    assert judge == JUDGE and len(observations) == 2
    judge2, observations2 = observations_from_json(
        json.dumps([{"judge": JUDGE, "item": "i1", "confidence": 0.9, "agreed": True}])
    )
    assert judge2 == JUDGE and len(observations2) == 1


def test_loader_fails_closed() -> None:
    with pytest.raises(ValueError, match="duplicate observation"):
        observations_from_json(
            json.dumps(
                {
                    "judge": JUDGE,
                    "observations": [
                        {"item": "i1", "confidence": 0.9, "agreed": True},
                        {"item": "i1", "confidence": 0.9, "agreed": True},
                    ],
                }
            )
        )
    with pytest.raises(ValueError, match="unknown keys"):
        observations_from_json(
            json.dumps({"judge": JUDGE, "observations": [], "surprise": 1})
        )
    with pytest.raises(ValueError, match="missing required field"):
        observations_from_json(json.dumps({"judge": JUDGE, "observations": [{"item": "i1"}]}))
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        observations_from_json(
            json.dumps(
                {
                    "judge": JUDGE,
                    "observations": [{"item": "i1", "confidence": 1.5, "agreed": True}],
                }
            )
        )
    with pytest.raises(ValueError, match="one judge per document"):
        observations_from_json(
            json.dumps(
                [
                    {"judge": "a", "item": "i1", "confidence": 0.5, "agreed": True},
                    {"judge": "b", "item": "i2", "confidence": 0.5, "agreed": True},
                ]
            )
        )
    with pytest.raises(ValueError, match="empty"):
        observations_from_json(json.dumps({"judge": JUDGE, "observations": []}))


# --- CLI ------------------------------------------------------------------


def _write_observations(tmp_path: Path, n_agree: int, n_disagree: int, confidence: float) -> Path:
    path = tmp_path / "observations.json"
    observations = [
        {"item": f"i{i}", "confidence": confidence, "agreed": True}
        for i in range(n_agree)
    ] + [
        {"item": f"j{i}", "confidence": confidence, "agreed": False}
        for i in range(n_disagree)
    ]
    path.write_text(
        json.dumps({"judge": JUDGE, "observations": observations}), encoding="utf-8"
    )
    return path


def test_cli_calibrated_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_observations(tmp_path, 100, 100, 0.5)
    code = cmd_calibrate(["--observations", str(path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "calibrated" in out
    assert "may gate" in out
    assert "once a live judge transport calls require_calibrated" in out


def test_cli_miscalibrated_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_observations(tmp_path, 50, 50, 1.0)
    code = cmd_calibrate(["--observations", str(path)])
    assert code == 2
    out = capsys.readouterr().out
    assert "miscalibrated" in out
    assert "must not gate" in out


def test_cli_uncalibrated_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_observations(tmp_path, 3, 2, 1.0)
    code = cmd_calibrate(["--observations", str(path)])
    assert code == 2
    assert "uncalibrated" in capsys.readouterr().out


def test_cli_writes_json_report(tmp_path: Path) -> None:
    path = _write_observations(tmp_path, 100, 100, 0.5)
    out = tmp_path / "report.json"
    code = cmd_calibrate(["--observations", str(path), "--json", "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    # Round-5 (arena#56): the CLI payload qualifies status when the
    # ledger flags the calibration gap; the record grade stays in
    # "grade". The record-level to_dict still says "calibrated"
    # (see test_record_round_trips_through_json).
    assert report["status"] == "calibrated (flagged: calibration_gap unbacked)"
    assert report["grade"] == "calibrated"
    assert report["n_total"] == 200


def test_cli_json_stdout_stays_parseable_with_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_observations(tmp_path, 100, 100, 0.5)
    out = tmp_path / "report.json"
    code = cmd_calibrate(["--observations", str(path), "--json", "--out", str(out)])
    assert code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)  # must not raise "Extra data"
    # Round-5 (arena#56): status is qualified when the ledger flags
    # the gap; the grade is preserved alongside it.
    assert parsed["status"] == "calibrated (flagged: calibration_gap unbacked)"
    assert parsed["grade"] == "calibrated"
    assert f"Report: {out}" in captured.err


def test_cli_bad_input_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert cmd_calibrate(["--observations", str(path)]) == 1
    assert cmd_calibrate([]) == 1


def test_cli_policy_flags_change_the_grade(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Confidence 1.0 at 50% agreement is miscalibrated under the default
    # tolerance — but a lax --tolerance 1.0 grades it calibrated, and the
    # record says so. The flags are real policy, not decoration.
    path = _write_observations(tmp_path, 50, 50, 1.0)
    assert cmd_calibrate(["--observations", str(path)]) == 2
    assert cmd_calibrate(["--observations", str(path), "--tolerance", "1.0"]) == 0
    out = capsys.readouterr().out
    assert "calibrated" in out
    assert "policy (non-default)" in out


def test_cli_bad_policy_exits_one(tmp_path: Path) -> None:
    path = _write_observations(tmp_path, 100, 100, 0.5)
    assert cmd_calibrate(["--observations", str(path), "--tolerance", "abc"]) == 1
    assert cmd_calibrate(["--observations", str(path), "--tolerance", "2"]) == 1
    assert cmd_calibrate(["--observations", str(path), "--min-total", "0"]) == 1


# --- Properties -----------------------------------------------------------


def _policy_for_properties() -> CalibrationPolicy:
    # Small floors so generated sets are usually graded, not uncalibrated.
    return CalibrationPolicy(min_total=10, bins=5, min_bin_n=1, tolerance=0.10)


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            st.booleans(),
        ),
        min_size=10,
        max_size=120,
    )
)
def test_properties_hold_for_any_observation_set(
    pairs: list[tuple[float, bool]],
) -> None:
    observations = [
        ConfidenceObservation(JUDGE, f"item-{i}", confidence, agreed)
        for i, (confidence, agreed) in enumerate(pairs)
    ]
    record = calibrate(JUDGE, observations, _policy_for_properties())
    assert record.status in ("calibrated", "miscalibrated", "uncalibrated")
    assert record.n_total == len(pairs)
    if record.ece is not None:
        assert 0.0 <= record.ece <= 1.0
    assert sum(b.n for b in record.bins) == len(pairs)
    assert [b.bin_index for b in record.bins] == sorted(b.bin_index for b in record.bins)
    assert may_gate(record) == (record.status == "calibrated")
    if record.status == "calibrated":
        require_calibrated(record)
    else:
        with pytest.raises(UncalibratedJudgeError):
            require_calibrated(record)


@settings(max_examples=25, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            st.booleans(),
        ),
        min_size=10,
        max_size=60,
    )
)
def test_calibration_is_deterministic(pairs: list[tuple[float, bool]]) -> None:
    observations = [
        ConfidenceObservation(JUDGE, f"item-{i}", confidence, agreed)
        for i, (confidence, agreed) in enumerate(pairs)
    ]
    policy = _policy_for_properties()
    assert calibrate(JUDGE, observations, policy) == calibrate(
        JUDGE, observations, policy
    )
