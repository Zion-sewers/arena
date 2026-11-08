# METRIC-LEDGER.md — every metric exposes its assumptions; soft metrics need hard backup

Berman 'Financial Intelligence' (catalog G.11), INSPIRED: metrics
reflect assumptions, and every soft metric needs a hard verifying
metric — the "cash test". A number with no stated assumptions is not
a metric, it is a vibe.

Executable form: `src/agenteval_bench/metric_ledger.py`, demo
`src/agenteval_bench/metric_ledger_demo.py`, tests
`tests/test_metric_ledger.py` (unit) and
`tests/test_metric_ledger_cli.py` (per-command CLI coverage). The
ledger audit prints in every covered report and is carried in the
JSON reports of `decide`, `calibrate`, `perturb`, `tiered`,
`simpson`, `diverge`, `ship`, and `alt-test`.

## The hard/soft line

| Kind | Meaning |
|---|---|
| `hard` | The number is a deterministic function (a count, a sum, or a proportion) of recorded observations. Any judgment the count depends on — a matcher verdict, a dimension tag, a ground-truth label — is named in the metric's assumptions; the count itself adds no further inference. |
| `soft` | The number is produced by an inference beyond that function: a gap between a self-report and a measurement, an assessment, or a proxy for a phenomenon. |

The Wilson interval is the uncertainty carried WITH a hard
proportion, not a separate soft metric.

## What "backed by" means

A soft metric's `hard_backup` names a HARD metric in the same ledger
that is independently measured from it — different observations,
never an input to the soft number's own computation and never the
soft number's own output. The check itself is the reader (or
tooling) comparing the two numbers; the ledger guarantees the
reference resolves to a HARD entry measured in the same report, it
does not perform the numeric comparison. The two backups that are
genuinely independent: the soft matchers (`contains`, `regex`,
`json_schema`) are backed by `label_agreement` (a validation
against ground-truth labels the scorer never sees), and
`p_conventional` is backed by `conventional_share` (counted from
the candidate cluster, while the declared p is a stated belief).

Where no independent measurement exists, the honest filing is
UNBACKED, not a circular backup: `calibration_gap` (the gap is
computed from the agreement rate itself) and
`perturbation_stability` (the drops are differences of the pass
rates themselves) carry `hard_backup=None`, and the audit flags them
`FLAGGED_NO_BACKUP` in every report that prints them. A circular
reference is not a backup, and a flagged soft metric is still a
printed metric: the flag, not silence, is what the issue requires.

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
| `reliability` | hard | — | Agreement across repeated runs (replay-agreement in `decide`) |
| `cost` | hard | — | Total measured cost in the policy's cost units |
| `agreement_rate` | hard | — | Measured fraction of correct judgments |
| `label_agreement` | hard | — | Measured proxy-vs-truth agreement on labeled cases |
| `exact_match` | hard | — | Output equals the expected string exactly |
| `coverage_pct` | hard | — | Fraction of the repo's lines covered by the tiered suite |
| `mutation_score` | hard | — | Fraction of seeded mutants the suite kills |
| `simpson_stratum_rate` | hard | — | Per-stratum treated/control success rates |
| `simpson_pooled_delta` | hard | — | Difference of the pooled treated/control rates (descriptive only) |
| `pairwise_diversity` | hard | — | Minimum pairwise (token-level) diversity in a candidate set |
| `conventional_share` | hard | — | Measured share of the conventional answer's cluster |
| `oec_z` | hard | — | Final z-score of the overall evaluation criterion |
| `srm_chi2` | hard | — | Chi-square for sample-ratio mismatch vs the design |
| `aa_z` | hard | — | Z-score of the A/A split |
| `guardrail_delta` | hard | — | Treated-vs-control guardrail difference |
| `alt_win_rate` | hard | — | A judge's winning rate (rho/omega) on recorded comparisons |
| `alt_coverage` | hard | — | Fraction of (judge, annotator) cells meeting the floor |
| `gate_catch_rate` | hard | — | Fraction of sampled flywheel payloads the gate caught |
| `calibration_gap` | soft | UNBACKED (flagged) | Mean judge confidence minus measured agreement rate |
| `perturbation_stability` | soft | UNBACKED (flagged) | Whether the agent's verdict survives input perturbation |
| `p_conventional` | soft | `conventional_share` | Declared probability the conventional answer is right |
| `contains` | soft | `label_agreement` | Output states the required fact (substring presence) |
| `regex` | soft | `label_agreement` | Output matches the required shape |
| `json_schema` | soft | `label_agreement` | Output is valid JSON with the required keys |

Every entry lists its assumptions, each with a falsifier: the
observation that would break it. An assumption without a falsifier is
a slogan, not a checkable claim. The audit summary prints the
assumption texts in the report, and the JSON reports carry the full
specs.

`scorer_validity` is deliberately NOT in the ledger: it is a table
of per-matcher validity grades printed by `validity.py`, not a
printed number, so auditing it as a metric would certify a number
the report never shows.

## How the audit works

The audit is derived from the live report object at each print site
— never from a static table of what the command is supposed to
print. Each command builds the `measured` mapping (metric name ->
whether this report measured it) from the values it rendered, and
`audit_measured` audits exactly that set:

- `run` audits `pass_rate` always, plus `label_agreement` and the
  one validated matcher only when `--validity-labels` measured them.
  The validity table prints grades, not matcher rates.
- `decide` audits only the dimensions with measured evidence; an
  UNMEASURED dimension prints no number and is never audited clean.
- `calibrate` audits `agreement_rate` and `calibration_gap` (flagged,
  unbacked). ECE is the bin-weighted aggregate of the same gap.
- `perturb` audits `pass_rate` for regimes that were measured and
  scored, and `perturbation_stability` only when both drops exist
  (flagged, unbacked).
- `tiered` audits `coverage_pct` / `mutation_score` when the phase
  measured them (the original live-derivation pattern).
- `simpson` audits `simpson_stratum_rate` and
  `simpson_pooled_delta`; `diverge` audits `pairwise_diversity`,
  `conventional_share`, and `p_conventional`; `ship` audits `oec_z`,
  `srm_chi2`, `aa_z`, and `guardrail_delta`, each only when the
  verdict carries it; `alt-test` audits `alt_win_rate` and
  `alt_coverage`; `snapshot` (show / publish / compare) audits
  `pass_rate`; `flywheel` audits `gate_catch_rate` when a rate was
  measured, and says so when none was.

A printed metric with no ledger entry is flagged
`FLAGGED_NOT_IN_LEDGER`, never silently clean. A soft metric whose
backup was not measured in THIS report is flagged
`FLAGGED_NO_BACKUP` even when the registry names one. A metric the
report did not print is never audited, so the "N/N clean" headline
is a true statement about the report in front of the reader. Adding
a metric to a report means adding it to the ledger first — the
per-command CLI tests (`tests/test_metric_ledger_cli.py`) fail until
both are done: they run each command and assert its printed metrics
are covered by the audit it prints.

## Enforcement (not prose)

- Construction refuses an assumption-free or blank metric and a HARD
  metric carrying a backup. `kind` is coerced to `MetricKind`: a
  plain `"soft"` string becomes SOFT, a garbage kind raises — a
  hand-built spec can never slip a non-Enum kind past the audit as
  clean. Assumptions are copied to a tuple, so a frozen spec cannot
  be mutated assumption-free afterwards.
- A SOFT metric without a backup constructs and is flagged by the
  audit (`FLAGGED_NO_BACKUP`); the parsers accept it the same way.
- `audit_measured()` flags a soft metric whose backup is missing,
  soft, or unmeasured in this report, a mis-filed spec, and a
  reported metric with no entry. It refuses a bare string.
- `spec_from_dict()` / `ledger_from_dicts()` apply the same rules at
  the untrusted-input boundary, refuse unknown keys (including
  inside assumption mappings) and duplicate assumptions, and use
  ValueError for invalid values, TypeError for wrong types.
- `--strict-ledger` exists on `run`, `decide`, `calibrate`,
  `perturb`, and `tiered`: a flagged audit refuses the report
  (run/tiered exit 2, calibrate exit 2, perturb downgrades STABLE to
  UNSTABLE, decide downgrades GATE to INVESTIGATE), mirroring
  `--strict-validity`. Default off. Note `calibrate --strict-ledger`
  always refuses: the gap is flagged by design (unbacked).
- A flagged audit prints its remediation: do not quote or gate on
  flagged metrics until each is resolved.
- `require_ledger_ok()` raises on a flagged (or empty) audit for
  callers that gate on the ledger.

## Worked example (the cash test, with real numbers)

`calibrate` on a judge with 100 recorded observations: agreement_rate
0.62 (62 correct judgments counted), mean confidence 0.90, so the
calibration_gap is 0.28. The gap alone says "overconfident by 0.28";
the cash test is the agreement rate printed next to it: 62 correct
out of 100 recorded. The gap is filed UNBACKED because it is
computed from that same agreement rate — the agreement rate is the
number the reader checks the confidence against, not a backup the
ledger certifies. If the observations file were empty, there would
be no agreement rate to check the gap against — the gap would be
unmeasured, and the ledger would flag it. A judge that scores quality
0.9 while the deterministic quality proportion is 0.62 fails the same
way: quote the hard count, not the soft score.

## Scope

Covered: every metric printed by `run`, `decide`, `calibrate`,
`perturb`, `tiered`, `simpson`, `diverge`, `ship`, `alt-test`,
`snapshot`, and `flywheel` — each prints its audit, and the JSON
reports carry `metric_ledger`. NOT covered, with an in-report notice
instead of silence: `tracebench` (gate verdict and case counts, not
scored metrics) and `baseline` (frozen per-case scores, not an
aggregate metric). The judge-allocation command (`judges`) prints no
scored metric and is out of scope. `novelty_score` has no CLI print
site and is not in the ledger until one exists.

## Adding a metric

1. Add a `MetricSpec` to `METRIC_LEDGER` with its assumptions (each
   with a falsifier) and, if SOFT, a HARD backup that is
   independently measured — or file it unbacked on purpose and say
   why in the assumptions.
2. Derive the audited set from the live report object at the print
   site (`audit_measured`), never from a static table.
3. Add a CLI-level test in `tests/test_metric_ledger_cli.py` that
   runs the command and asserts its printed metrics are covered by
   the audit it prints.

## Example

```python
from agenteval_bench.metric_ledger import audit_measured

audit = audit_measured({"pass_rate": True})
print(audit.summary("run"))
# Metric ledger (run): 1/1 metrics clean
#   pass_rate              hard backup=—               OK
#     - The scorer's matchers decide pass/fail exactly as each case specifies (breaks if: ...)
#   ...
```
