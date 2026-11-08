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
`simpson`, `diverge`, `ship`, `alt-test`, `tracebench regress`, and
`bugsmith` (manifest.json).

## The hard/soft line

| Kind | Meaning |
|---|---|
| `hard` | The number is a descriptive statistic computed only from recorded observations — a count, a sum, a proportion, or a weighted mean of recorded values — with no inferential step. Any judgment the count depends on — a matcher verdict, a dimension tag, a ground-truth label — is named in the metric's assumptions; the count itself adds no further inference. |
| `soft` | The number's computation consumes an inferred quantity about something unobserved — a judge's grade, an interval bound, a label, an assessment, or a proxy for a phenomenon. Determinism is not the test: a confidence bound is a deterministic function of the sample and still an inference. The test is what the number is computed FROM. A gap against a recorded measurement is HARD: ECE is the bin-weighted mean absolute difference between mean judge confidence and the measured agreement rate, both recorded. A gap against an inferred quantity is SOFT: the calibration gap compares mean confidence against the Wilson lower bound of agreement, a 95% interval estimate of the true agreement rate, not a recorded value. |

The Wilson interval is the uncertainty carried WITH a hard
proportion, not a separate soft metric — unless a number is computed
from the bound itself, as the calibration gap is.

## Components are not metrics

A report prints numbers that are not metrics: case totals, seeds,
phase durations, timestamps, the corpus trace count, the
denominator of a rate. These are inputs to, denominators of, or
bookkeeping around an audited metric — a component of it — and the
ledger does not audit them as separate entries. The rule is
mechanical, not vibes: if the number is a count, a duration, or a
timestamp that the report does not present as a result, it is a
component; if the report presents it as a result (a rate, a score,
a z, a delta, a share), it is a metric and needs an entry. The
growth-scan test enforces the boundary at the observable surface
(see Enforcement): every decimal, integer-percent, fraction,
integer+unit count, or labeled bare-integer rendering in the
detector's enumerated shapes (decimals, `N%`, `N/M`, `z=`/`chi2=`,
integers followed by a listed unit noun, bare integers after one of
the listed labels `score`/`rate`/`total`/`count`/`recorded`) must
match a declared pattern tied to an audited metric or a declared
component. The scan is format-aware, not format-blind: a new metric
printed in an unlisted shape (an unlabeled bare integer, a count
with an unlisted noun, a word percent) passes the scan, so the
enumerated shapes are the documented boundary — widen the detector
before relying on it for a new rendering shape.

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
computed from the agreement rate itself),
`perturbation_stability` (the drops are differences of the pass
rates themselves), and `alt_conservative_score` (the score is a mean
of Wilson bounds computed from the same wins the win rate is
computed from, so the win rate cannot independently check it) carry
`hard_backup=None`, and the audit flags them `FLAGGED_NO_BACKUP` in
every report that prints them. A circular
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
| `ece` | hard | — | Bin-weighted mean absolute difference between mean judge confidence and measured agreement rate (expected calibration error) |
| `label_agreement` | hard | — | Measured proxy-vs-truth agreement on labeled cases |
| `exact_match` | hard | — | Output equals the expected string exactly |
| `coverage_pct` | hard | — | Fraction of the repo's lines covered by the tiered suite |
| `mutation_score` | hard | — | Fraction of seeded mutants the suite kills |
| `simpson_stratum_rate` | hard | — | Per-stratum treated/control success rates |
| `treated_share` | hard | — | Share of treated cases within one stratum |
| `simpson_pooled_delta` | hard | — | Difference of the pooled treated/control rates (descriptive only) |
| `pairwise_diversity` | hard | — | Minimum pairwise (token-level) diversity in a candidate set |
| `conventional_share` | hard | — | Measured share of the conventional answer's cluster |
| `oec_z` | hard | — | Final z-score of the overall evaluation criterion |
| `srm_chi2` | hard | — | Chi-square for sample-ratio mismatch vs the design |
| `aa_z` | hard | — | Z-score of the A/A split |
| `guardrail_delta` | hard | — | Treated-vs-control guardrail difference |
| `alt_win_rate` | hard | — | A judge's winning rate (rho/omega) on recorded comparisons |
| `alt_annotator_rho` | hard | — | Per-annotator mean win probability rho_f (mean of n recorded binary wins) |
| `alt_coverage` | hard | — | Fraction of (judge, annotator) cells meeting the floor |
| `prior_score` | hard | — | A case's recorded score in a previous snapshot version (promotion provenance) |
| `regression_counts` | hard | — | Cases mined, new, updated, and retired in one scheduled tracebench regression run |
| `regression_streak` | hard | — | Consecutive failed scheduled regression runs (the rollback streak) |
| `gate_catch_rate` | hard | — | Fraction of sampled flywheel payloads the gate caught |
| `validated_share` | hard | — | Fraction of mined bugsmith candidate reports that break at least one test |
| `alt_conservative_score` | soft | UNBACKED (flagged) | Conservative score: mean Wilson 95% lower bound on the win rate (covered cells when ranked, all cells as the diagnostic when masked) — a mean of bounds, so SOFT under the ledger rule |
| `calibration_gap` | soft | UNBACKED (flagged) | Per-bin mean judge confidence minus the Wilson lower bound of measured agreement (ECE is a different aggregate and has its own entry) |
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
- `calibrate` audits from the record it printed: `agreement_rate`
  and `calibration_gap` (flagged, unbacked) only when bins exist,
  `ece` only when the record carries one. With too few
  observations no bin and no ECE is printed, so the audit is empty
  (see below). ECE is a bin-weighted mean absolute difference
  between agreement and confidence — a different aggregate from
  the per-bin gap, which compares confidence against the Wilson
  lower bound of agreement — and has its own entry.
- `perturb` audits `pass_rate` for regimes that were measured and
  scored, and `perturbation_stability` only when both drops exist
  (flagged, unbacked).
- `tiered` audits `coverage_pct` / `mutation_score` when the phase
  measured them (the original live-derivation pattern). The gate
  phase measures neither, so its audit is empty (see below).
- `simpson` audits `simpson_stratum_rate`, `treated_share`, and
  `simpson_pooled_delta`; `diverge` audits `pairwise_diversity`,
  `conventional_share`, and `p_conventional`; `ship` audits `oec_z`,
  `srm_chi2`, `aa_z`, and `guardrail_delta`, each only when the
  verdict carries it; `alt-test` audits all four — `alt_win_rate`,
  `alt_annotator_rho`, `alt_conservative_score`, `alt_coverage` —
  on a ranked board, and only `alt_conservative_score` and
  `alt_coverage` on a refused ranking; `snapshot` (show / publish /
  compare) audits `pass_rate`, plus `prior_score` when promotion
  lines print one; `flywheel` audits `gate_catch_rate` when a rate
  was measured, and says so when none was. `tracebench regress`
  audits `pass_rate` (read back from the comparison in the report
  it writes), `regression_counts`, and `regression_streak`, and
  carries the ledger in `report.json`.

A printed metric with no ledger entry is flagged
`FLAGGED_NOT_IN_LEDGER`, never silently clean. A soft metric whose
backup was not measured in THIS report is flagged
`FLAGGED_NO_BACKUP` even when the registry names one. A metric the
report did not print is never audited, so the "N/N clean" headline
is a true statement about the report in front of the reader.

## Empty audit: pass-through, not "not clean"

A report that measured nothing prints no "0/0 metrics clean" and
never pairs a verdict with a failing ledger. An empty audit says
`no metrics measured — nothing audited`, its `all_ok` is True
(nothing is flagged), and `--strict-ledger` passes through on it.
This is a deliberate semantic choice, recorded in round 4: the
gate phase, a too-few-observations calibration, and a trust
refusal legitimately measure nothing, and refusing them for naming
zero flagged metrics would be a false positive. The audit is a
check on what was measured; silence about an unmeasured metric is
the honest filing, and the report says so in its own text.

## The flagged-verdict rule

A report may print a flagged metric — that is what the flag is
for — but the artifact must not certify what it flags. The
headline carries the flag next to the grade (`calibrate` appends
"calibration gap unbacked (see ledger): do not quote it as verified";
`perturb`
appends "(stability label unbacked — see ledger)" to the verdict
line; `alt-test` prints "Ranking ordered by
alt_conservative_score, which is unbacked (see ledger): do not quote
the ranking as verified" with the ranking itself), the gating sentence scopes gating to the measured grade
(the bin-check grade for calibrate), never to the flagged number,
and the JSON carries a `status_note` / `verdict_note` / `ranking_note` next to the
grade. `calibrate` JSON qualifies `status` itself when flagged
(`"calibrated (flagged: calibration_gap unbacked)"`) and, under
`--strict-ledger`, writes `status` as `"refused (ledger gate)"`
with the bin-check grade preserved in `grade`. The grade itself stays: the bin-check grade and the
drop-threshold verdict are measured facts, and the live gate
(`require_calibrated` / `check_stability`) is the Python API, not
the CLI text. Under `--strict-ledger` the artifact itself is
downgraded (`perturb` STABLE becomes UNSTABLE in stdout, JSON, and
exit code; `calibrate` refuses with exit 2 and its written file
says refused, not calibrated).

The validity table is graded, not audited: `run` and `decide` print
a scope line in their own text stating the headline covers the
printed aggregate (or dimensions) only, and `decide` carries a
`metric_ledger_scope` field in its JSON next to the
`scorer_validity` object it does not cover. The table's grades are
not ledger metrics, and the headline never certifies them. Adding
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
  `audit_measured` and `_audit_entry` re-validate every spec
  through `_normalized_spec` (isinstance check against
  `MetricSpec`, kind coercion to `MetricKind`, every assumption item
  re-checked to be a `MetricAssumption`): a hand-built spec
  with a duck-typed shape raises TypeError, a garbage kind raises
  ValueError, a bypass-built spec with junk assumption items raises
  TypeError, and a bypassed-but-valid kind is coerced and audited
  normally — the audit can never certify a spec it never validated.
- The growth-scan test (`_assert_stdout_covered` in
  `tests/test_metric_ledger_cli.py`) parses each command's stdout:
  every line rendering a number must match a declared rendering
  pattern tied to an audited metric or a declared component, every
  declared pattern must match at least one line, and every named
  metric must appear in the audit the output carries. A command
  that grows a printed metric without a ledger entry fails here —
  the audit alone cannot catch it, because the audit can only flag
  metrics it is told about. The arch reviewer's exact mutation
  line is pinned as a failing case, along with its integer-percent
  (`87%`), fraction (`4/5`), and integer-count (`12 tests`) twins.
- The dispatch test (`test_dispatch_every_command_is_classified`
  in the same file) parses `main()` plus the `tracebench` and
  `snapshot` subcommand dispatch and requires every reachable
  command to have a scan test or an explicit out-of-scope notice:
  a new command, or a new subcommand, fails the test until it is
  classified, so coverage cannot silently go stale.
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
- `require_ledger_ok()` raises only on a flagged audit (an empty
  audit passes through — see above), for callers that gate on the
  ledger.
- CI enforces the ledger on the live path: the tiered expanded
  step runs with `--strict-ledger`, and a `run --strict-ledger`
  step runs the support-agent suite, so a flagged metric fails the
  build, not just a local run. A second CI step exercises the
  flagging strict path itself — it asserts `perturb --strict-ledger`
  exits 1 (STABLE downgraded to UNSTABLE) and `calibrate
  --strict-ledger` exits 2 (refused) — so the gate is a live gate,
  not only a tripwire for future misfiling.

## Worked example (the cash test, with real numbers)

`calibrate` on a judge with 100 recorded observations in a single
bin: agreement_rate 0.62 (62 correct judgments counted), mean
confidence 0.90. The per-bin gap compares that 0.90 against the
Wilson lower bound of the agreement rate (0.52 for 62/100), so the
gap is about 0.38. The gap alone says "overconfident"; the cash test
is the agreement rate printed next to it: 62 correct out of 100
recorded. The gap is filed UNBACKED because it is computed from
that same agreement rate — the agreement rate is the number the
reader checks the confidence against, not a backup the ledger
certifies. ECE (0.28 here, the bin-weighted mean absolute
difference between agreement and confidence) is a different
aggregate from the gap and has its own entry: quoting the ECE
number as the per-bin gap would be the exact confusion the two
entries exist to prevent. If the
observations file were empty, there would be no agreement rate to
check the gap against — the gap would be unmeasured, and the
ledger would say "no metrics measured", not "0/0 clean". A judge
that scores quality 0.9 while the deterministic quality proportion
is 0.62 fails the same way: quote the hard count, not the soft score.

## Scope

Covered: every metric printed by `run`, `decide`, `calibrate`,
`perturb`, `tiered`, `simpson`, `diverge`, `ship`, `alt-test`,
`snapshot` (show / publish / compare / verify / list / check /
repair), `flywheel`, `bugsmith`, and `tracebench regress` — each prints its
audit, and the JSON reports carry `metric_ledger`. NOT covered, with
an in-report notice instead of silence: `tracebench mine` (gate
verdict and case counts, not scored metrics) and `baseline` (frozen
per-case scores, not an aggregate metric). The judge-allocation
command (`judges`) prints no scored metric and is out of scope.
`replay` prints case counts and a bit-exact verdict, no metric;
`register-design`, `prespec`, and `register-analysis-plan` print a
registration digest and a review verdict, no metric — each carries
a scan test proving no numeric metric line. Coverage is enforced by
the dispatch test (`tests/test_metric_ledger_cli.py`): it enumerates
every command `main()` can reach and requires each to have a scan
test or an explicit out-of-scope notice, so a new command fails the
test until it is classified.
`novelty_score` has no CLI print site and is not in the ledger until
one exists.

## Adding a metric

1. Add a `MetricSpec` to `METRIC_LEDGER` with its assumptions (each
   with a falsifier) and, if SOFT, a HARD backup that is
   independently measured — or file it unbacked on purpose and say
   why in the assumptions.
2. Derive the audited set from the live report object at the print
   site (`audit_measured`), never from a static table.
3. Add a CLI-level test in `tests/test_metric_ledger_cli.py` that
   runs the command and asserts its printed metrics are covered by
   the audit it prints — and declare the metric's stdout rendering
   in the growth-scan patterns (`_assert_stdout_covered`), so a
   future print of the same number without an entry fails the scan.

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
