# CALIBRATION.md — judge calibration: confidence is not evidence

Executable form: `src/agenteval_bench/calibration.py`, demo
`src/agenteval_bench/calibration_demo.py`, tests `tests/test_calibration.py`,
CLI `agenteval-bench calibrate --observations <file.json>`.

## The trap

A judge says "confidence 0.95". That number is a claim about its own
accuracy, not a measurement of it. A gate that trusts the number lets a
confident-but-wrong judge through, and the pipeline reports agreement
that carries no information. Self-reported confidence is NOT evidence:
the only evidence is measured agreement — how often the judge's labels
actually agreed with a reference, at that stated confidence.

Source: eval_* series ch.5/ch.10 (catalog G.41) — logical consistency,
chain validity, judge calibration; inference-time control without gold
annotations.

## Rules

- Calibration is per confidence bin: stated confidences are bucketed
  (10 equal-width bins over [0, 1] by default; 1.0 lands in the last
  bin), and each bin compares its mean stated confidence with its
  measured agreement rate.
- The grade rests on the Wilson 95% interval LOWER bound on agreement,
  never on the point estimate — the same discipline as `validity.py`
  and `decision.py`. A bin fails when its mean stated confidence
  exceeds the lower bound by more than the policy tolerance (default
  0.10). The overconfident direction is the dangerous one.
- ECE (expected calibration error) is reported as a diagnostic only.
  It never decides the grade, so a small point-estimate error cannot
  smuggle an under-measured judge through.
- Too little data is UNCALIBRATED, never trusted: fewer than
  `min_total` observations (default 50), any non-empty bin below
  `min_bin_n` (default 5), or zero observations. 5/5 agreement at
  confidence 1.0 is a point estimate of 1.0 with a lower bound of
  0.565, and does not clear the bar.
- Statuses (closed set): `calibrated`, `miscalibrated` (measured, and
  the measurement refutes the confidence), `uncalibrated` (no usable
  measurement). Only `calibrated` may gate.
- The gate is structural: `may_gate` / `require_calibrated` take ONLY
  the calibration record. A stated confidence cannot be passed to the
  gate at all, so no call site can substitute a high confidence number
  for a measurement. `require_calibrated` refuses anything else with
  a typed `UncalibratedJudgeError` naming the judge, its status, and
  every failed condition.

## Observations format

A JSON document: a mapping with a `judge` string and an `observations`
list, or a bare list (judge taken from the entries, which must then all
name one judge). Each observation carries `item`, `confidence` in
[0, 1], and `agreed` (whether the judge's label agreed with the
reference on that item).

```json
{"judge": "sharp-judge", "observations": [
  {"item": "a", "confidence": 0.9, "agreed": true},
  {"item": "b", "confidence": 0.9, "agreed": false}
]}
```

The loader fails closed: duplicate (judge, item) pairs, unknown keys,
a missing required field, or a confidence outside [0, 1] all raise
`ValueError`, so a typo cannot silently become an observation with no
measured agreement. A copyable example ships at
`examples/calibration-observations.json`.

## Measured agreement: the alt-test bridge

`observations_from_alt_test(items, confidences, judge)` derives
observations from recorded alt-test labels: agreement is the judge's
label vs the majority annotator label on the same item (discrete tasks
only — a continuous or text task is refused, not approximated, because
agreement there needs a similarity threshold this module does not own).
Items the judge never labelled are not scored on it (alt-test
convention). Items with a tied annotator majority, or with a label but
no supplied confidence, are named in `skipped` — nothing is silently
dropped.

## CLI

```bash
agenteval-bench calibrate --observations observations.json [--json] [--out report.json]
```

Exit 0 = calibrated (the judge's scores may gate); exit 2 =
miscalibrated or uncalibrated (the scores must not gate); exit 1 =
unusable observations file. The report JSON carries `judge`, `status`,
`n_total`, `ece`, per-bin tables, `reasons`, and `notes`.

## Coverage boundary

This module calibrates recorded (label, stated-confidence) pairs; it
does not call any judge. Arena has no live judge transport (README:
"LLM-as-judge is planned, not implemented"), so observations are
supplied by the caller and a judge with no supplied confidence is
uncalibrated by definition — the module never invents one. When a live
judge transport lands, it must call `require_calibrated` before its
scores gate anything; that wiring is a named follow-up, not part of
this unit.
