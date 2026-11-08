# METRIC-LEDGER.md — every metric exposes its assumptions; soft metrics need hard backup

Berman 'Financial Intelligence' (catalog G.11), INSPIRED: metrics
reflect assumptions, and every soft metric needs a hard verifying
metric — the "cash test". A number with no stated assumptions is not
a metric, it is a vibe.

Executable form: `src/agenteval_bench/metric_ledger.py`, demo
`src/agenteval_bench/metric_ledger_demo.py`, tests
`tests/test_metric_ledger.py`. The ledger audit prints in every
covered report (`run`, `decide`, `calibrate`, `perturb`, `tiered`)
and is carried in the JSON reports of `decide`, `calibrate`, and
`perturb`.

## The hard/soft line

| Kind | Meaning |
|---|---|
| `hard` | The number is a deterministic function (a count, a sum, or a proportion) of recorded observations. Any judgment the count depends on — a matcher verdict, a dimension tag, a ground-truth label — is named in the metric's assumptions; the count itself adds no further inference. |
| `soft` | The number is produced by an inference beyond that function: a gap between a self-report and a measurement, an assessment, or a proxy for a phenomenon. |

The Wilson interval is the uncertainty carried WITH a hard
proportion, not a separate soft metric.

## What "backed by" means

A soft metric's `hard_backup` names a HARD metric in the same ledger
that is independently measured from it — different observations, never
the soft number's own output. The check itself is the reader (or
tooling) comparing the two numbers: in `calibrate` the gap and the
agreement rate print side by side; the ledger guarantees the reference
resolves to a HARD entry, it does not perform the numeric comparison.
The two backups that must not be circular are pinned by name:
`scorer_validity` is backed by `label_agreement` (a validation against
ground-truth labels the scorer never sees, not the scorer's own
`pass_rate`), and `calibration_gap` is backed by `agreement_rate`
(its own measured component).

## Two axes, not two verdicts

The ledger's HARD/SOFT says HOW a number is measured. `validity.py`
grades say WHETHER to trust it in a particular report: replay
reliability is UNVALIDATED_PROXY, a single cost observation is
UNVALIDATED_PROXY. When the two lenses disagree, the validity grade
governs trust and the ledger governs what the number assumes. Both
tables print together in `run` and `decide` for this reason.

## The ledger

| Metric | Kind | Hard backup | What it claims to measure |
|---|---|---|---|
| `pass_rate` | hard | — | Fraction of scored cases the agent passed |
| `quality` | hard | — | Fraction of quality-dimension cases passed |
| `safety` | hard | — | Fraction of safety-dimension cases passed |
| `reliability` | hard | — | Agreement across repeated seeded runs |
| `cost` | hard | — | Total measured cost in the policy's cost units |
| `agreement_rate` | hard | — | Measured fraction of correct judgments |
| `label_agreement` | hard | — | Measured proxy-vs-truth agreement on labeled cases |
| `exact_match` | hard | — | Output equals the expected string exactly |
| `coverage_pct` | hard | — | Fraction of the repo's lines covered by the tiered suite |
| `mutation_score` | hard | — | Fraction of seeded mutants the suite kills |
| `calibration_gap` | soft | `agreement_rate` | Mean judge confidence minus measured agreement rate |
| `scorer_validity` | soft | `label_agreement` | Whether the scorer's matchers measure what they claim |
| `perturbation_stability` | soft | `pass_rate` | Whether the agent's verdict survives input perturbation |
| `contains` | soft | `label_agreement` | Output states the required fact (substring presence) |
| `regex` | soft | `label_agreement` | Output matches the required shape |
| `json_schema` | soft | `label_agreement` | Output is valid JSON with the required keys |

Every entry lists its assumptions, each with a falsifier: the
observation that would break it. An assumption without a falsifier is
a slogan, not a checkable claim. The audit summary prints the
assumption texts in the report, and the JSON reports carry the full
specs.

## How the audit works

The audit is fed the metrics a report actually prints, per command
(`REPORT_METRIC_SETS`): `run` audits pass_rate, the four matchers,
scorer_validity, and label_agreement; `decide` audits the four
dimensions plus the same matcher set; `calibrate` audits the agreement
rate and the gap; `perturb` audits the pass rate and stability;
`tiered` audits coverage and mutation score. A printed metric with no
ledger entry is flagged `FLAGGED_NOT_IN_LEDGER`, never silently clean.
Adding a metric to a report means adding it to the ledger first — the
sync test fails otherwise.

## Enforcement (not prose)

- Construction refuses an assumption-free or blank metric, a SOFT
  metric without `hard_backup`, and a HARD metric carrying one.
  Assumptions are copied to a tuple, so a frozen spec cannot be
  mutated assumption-free afterwards.
- `audit_report()` flags a soft metric whose backup is missing or
  soft, a mis-filed spec, and a reported metric with no entry.
- `spec_from_dict()` / `ledger_from_dicts()` apply the same rules at
  the untrusted-input boundary, refuse unknown keys, and use
  ValueError throughout.
- `decide --strict-ledger` downgrades a GATE to INVESTIGATE when the
  report's ledger audit is flagged, mirroring `--strict-validity`.
  A flagged audit prints its remediation: do not quote or gate on
  flagged metrics until each names a hard backup.
- `require_ledger_ok()` raises on a flagged (or empty) audit for
  callers that gate on the ledger.

## Worked example (the cash test, with real numbers)

`calibrate` on a judge with 100 recorded observations: agreement_rate
0.62 (62 correct judgments counted), mean confidence 0.90, so the
calibration_gap is 0.28. The gap alone says "overconfident by 0.28";
the cash test is the agreement rate printed next to it: 62 correct
out of 100 recorded. If the observations file were empty, there would
be no agreement rate to check the gap against — the gap would be
unbacked, and the ledger would flag it. A judge that scores quality
0.9 while the deterministic quality proportion is 0.62 fails the same
way: quote the hard count, not the soft score.

## Scope

Covered: the metrics printed by `run`, `decide`, `calibrate`,
`perturb`, and `tiered`. NOT covered yet, as follow-ups: alt-test
alignment scores and p-values, novelty-oracle `novelty_score`, and
regression aggregate scores — those reports carry no ledger entry.
The judge-allocation command (`judges`) prints no scored metric and
is out of scope; `tiered`'s JSON report does not yet carry the ledger
(the audit prints on stdout there).

## Adding a metric

1. Add a `MetricSpec` to `METRIC_LEDGER` with its assumptions (each
   with a falsifier) and, if SOFT, a HARD backup.
2. Add the metric's name to the printing command's set in
   `REPORT_METRIC_SETS`.
3. The sync test (`test_report_sets_cover_the_shipped_ledger`) and
   the per-command audit tests fail until both are done.

## Example

```python
from agenteval_bench.metric_ledger import REPORT_METRIC_SETS, audit_report

audit = audit_report(REPORT_METRIC_SETS["run"])
print(audit.summary("run"))
# Metric ledger (run): 7/7 metrics clean
#   pass_rate              hard backup=—               OK
#     - The scorer's matchers decide pass/fail exactly as each case specifies (breaks if: ...)
#   ...
```
