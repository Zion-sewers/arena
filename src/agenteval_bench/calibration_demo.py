"""Seeded end-to-end demo of judge calibration (arena#45).

Runs the full "confidence is not evidence" protocol offline on recorded
observations plus one real CLI invocation on a temporary observations
file. Exit 0 only if every acceptance expectation holds:
- the documented trap: a judge stating confidence 1.0 on every item
  while agreeing with the annotator majority only half the time is
  MISCALIBRATED, and the gate refuses it;
- a judge whose stated confidence matches its measured agreement is
  CALIBRATED, and the gate allows it;
- the small-n trap: 5/5 agreement at confidence 1.0 is UNCALIBRATED,
  never a pass — a point estimate of 1.0 with a Wilson lower bound of
  0.565 does not clear the bar;
- a thin bin (one observation in a bin, fifty total) is UNCALIBRATED;
- the alt-test bridge derives agreement from the annotator majority,
  names a tied majority and a missing confidence as skipped items, and
  ignores items the judge never labelled;
- calibration is deterministic: the same observations calibrate identically
  twice, and the loader fails closed on a duplicate observation;
- the CLI `calibrate` command on an observations file exits 0 when
  calibrated and 2 when miscalibrated, and writes the report JSON.

The fixture confidences are recorded data, not measurements: no live
judge runs in this demo, and the module itself makes no network call.
Any failure raises, so CI can gate on this demo directly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agenteval_bench.calibration import (
    ConfidenceObservation,
    calibrate,
    may_gate,
    observations_from_alt_test,
    observations_from_json,
    require_calibrated,
)
from agenteval_bench.cli import cmd_calibrate
from alt_test.types import AltTestItem

JUDGE = "demo-judge"


def _expect(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"  ok    {label}")


def _mixed(confidence: float, n_agree: int, n_disagree: int) -> list[ConfidenceObservation]:
    return [
        ConfidenceObservation(JUDGE, f"agree-{i}", confidence, True)
        for i in range(n_agree)
    ] + [
        ConfidenceObservation(JUDGE, f"disagree-{i}", confidence, False)
        for i in range(n_disagree)
    ]


def main() -> int:
    print("Judge calibration demo (arena#45)")

    # 1. The trap: confidence 1.0, coin-flip measured agreement.
    trap = calibrate(JUDGE, _mixed(1.0, 50, 50))
    _expect(trap.status == "miscalibrated", "confidence 1.0 at 50% agreement is miscalibrated")
    _expect(not may_gate(trap), "miscalibrated judge may not gate")
    try:
        require_calibrated(trap)
    except ValueError as e:
        _expect("never gate anything" in str(e), "gate refusal names the rule")
    else:  # pragma: no cover - the refusal is the expectation
        raise AssertionError("require_calibrated must refuse a miscalibrated judge")

    # 2. A judge whose confidence matches its measured agreement passes.
    good = calibrate(JUDGE, _mixed(0.5, 100, 100))
    _expect(good.status == "calibrated", "confidence 0.5 at 50% agreement is calibrated")
    _expect(good.ece == 0.0, "perfectly matched confidence has ECE 0")
    _expect(may_gate(good), "calibrated judge may gate")
    require_calibrated(good)

    # 3. The small-n trap: 5/5 is not a measurement.
    small = calibrate(
        JUDGE, [ConfidenceObservation(JUDGE, f"s{i}", 1.0, True) for i in range(5)]
    )
    _expect(small.status == "uncalibrated", "5/5 agreement is uncalibrated, not calibrated")
    _expect(small.ece is None, "no ECE without a measurement")

    # 4. A thin bin is unmeasured, so the judge is uncalibrated.
    thin = calibrate(
        JUDGE,
        _mixed(0.9, 49, 0)
        + [ConfidenceObservation(JUDGE, "thin-1", 0.5, True)],
    )
    _expect(thin.status == "uncalibrated", "a one-observation bin is uncalibrated")

    # 5. The alt-test bridge: majority agreement, named skips.
    items = [
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
            id="tie",
            task="discrete",
            annotator_labels={"h1": "yes", "h2": "no"},
            judge_labels={JUDGE: "yes"},
        ),
        AltTestItem(
            id="noconf",
            task="discrete",
            annotator_labels={"h1": "yes", "h2": "yes", "h3": "yes"},
            judge_labels={JUDGE: "yes"},
        ),
    ]
    bridged = observations_from_alt_test(items, {"a": 0.9, "b": 0.8, "tie": 0.7}, JUDGE)
    _expect(len(bridged.observations) == 2, "bridge derives two observations")
    _expect(bridged.observations[0].agreed is True, "judge agrees with the majority on a")
    _expect(bridged.observations[1].agreed is False, "judge disagrees with the majority on b")
    _expect(
        [s.item_id for s in bridged.skipped] == ["tie", "noconf"],
        "tied majority and missing confidence are named skips",
    )

    # 6. Determinism, and the loader fails closed on a duplicate.
    again = calibrate(JUDGE, _mixed(0.5, 100, 100))
    _expect(good == again, "the same observations calibrate identically twice")
    try:
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
    except ValueError as e:
        _expect("duplicate" in str(e), "duplicate observation is refused")
    else:  # pragma: no cover - the refusal is the expectation
        raise AssertionError("the loader must refuse a duplicate observation")

    # 7. The CLI on real files: 0 when calibrated, 2 when miscalibrated.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good_path = root / "good.json"
        good_path.write_text(
            json.dumps(
                {
                    "judge": JUDGE,
                    "observations": [
                        {"item": f"g{i}", "confidence": 0.5, "agreed": i % 2 == 0}
                        for i in range(200)
                    ],
                }
            ),
            encoding="utf-8",
        )
        report_path = root / "report.json"
        code = cmd_calibrate(
            ["--observations", str(good_path), "--json", "--out", str(report_path)]
        )
        _expect(code == 0, "CLI exits 0 for a calibrated judge")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _expect(report["status"] == "calibrated", "CLI report JSON carries the status")
        bad_path = root / "bad.json"
        bad_path.write_text(
            json.dumps(
                {
                    "judge": JUDGE,
                    "observations": [
                        {"item": f"b{i}", "confidence": 1.0, "agreed": i % 2 == 0}
                        for i in range(100)
                    ],
                }
            ),
            encoding="utf-8",
        )
        _expect(
            cmd_calibrate(["--observations", str(bad_path)]) == 2,
            "CLI exits 2 for a miscalibrated judge",
        )

    print("All calibration demo checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
