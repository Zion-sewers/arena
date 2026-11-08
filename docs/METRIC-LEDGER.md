# METRIC-LEDGER.md — every metric exposes its assumptions; soft metrics need hard backup

Berman 'Financial Intelligence' (catalog G.11), INSPIRED: metrics
reflect assumptions, and every soft metric needs a hard verifying
metric — the "cash test". A number with no stated assumptions is not
a metric, it is a vibe.

Executable form: `src/agenteval_bench/metric_ledger.py`, demo
`src/agenteval_bench/metric_ledger_demo.py`, tests
`tests/test_metric_ledger.py`. The ledger audit prints in every `run`
and `decide` report and is carried in the `decide` JSON report under
`metric_ledger`.

## The hard/soft line

| Kind | Meaning |
|---|---|
| `hard` | The number is a count or a sum of recorded observations, with no inference step between the phenomenon and the number. `pass_rate` is a count of scored cases; `cost` is a sum of measured per-case costs. |
| `soft` | The number is produced by an inference: a judge's grade, an interval, a label, an assessment. A soft metric is only reportable with a named HARD backup: the hard count it must be checkable against. |

A HARD metric's assumptions name what its count depends on (for
example, `pass_rate` depends on the scorer's matchers deciding
pass/fail exactly as each case specifies). A SOFT metric's backup is
the "cash test": the judge says quality is 0.9, the deterministic
scorer passes X% of quality cases — if the two cannot be checked
against each other, the soft number is flagged.

## The ledger

| Metric | Kind | Hard backup | What it claims to measure |
|---|---|---|---|
| `pass_rate` | hard | — | Fraction of scored cases the agent passed |
| `quality` | hard | — | Fraction of quality-dimension cases passed |
| `safety` | hard | — | Fraction of safety-dimension cases passed |
| `reliability` | hard | — | Agreement across repeated seeded runs |
| `cost` | hard | — | Total measured cost in the policy's cost units |
| `judge_score` | soft | `quality` | A judge's graded score for the agent's output |
| `calibration_gap` | soft | `pass_rate` | Mean judge confidence minus measured agreement rate |
| `scorer_validity` | soft | `pass_rate` | Whether the scorer's matchers measure what they claim |
| `perturbation_stability` | soft | `pass_rate` | Whether the agent's verdict survives input perturbation |

Every entry lists its assumptions, each with a falsifier: the
observation that would break it. An assumption without a falsifier is
a slogan, not a checkable claim.

## Enforcement (not prose)

- Construction refuses an assumption-free metric (`ValueError`).
- Construction refuses a SOFT metric without `hard_backup`, and a
  HARD metric carrying one.
- `audit_ledger()` flags a soft metric whose backup is missing from
  the ledger or names a SOFT metric, and a spec filed under the wrong
  key.
- `spec_from_dict()` / `ledger_from_dicts()` apply the same rules at
  the untrusted-input boundary: missing assumptions, bad kind,
  duplicate names, and unresolvable backups are all refused.
- `require_ledger_ok()` raises on a flagged audit for callers that
  gate on the ledger.

## Scope

Covered: the metrics in the `run` and `decide` reports. NOT covered
yet: tiered `MutationScore` (kill rate), regression aggregate scores
vs pinned baselines, alt-test alignment scores and p-values,
novelty-oracle `novelty_score`. Those reports carry no ledger entry;
the "every metric" claim is scoped to the covered set until they are
wired in. Bugsmith and flywheel produce no eval-metric report and are
out of scope.

## Example

```python
from agenteval_bench.metric_ledger import audit_ledger

audit = audit_ledger()
print(audit.summary())
# Metric ledger: 9/9 metrics clean
#   pass_rate              hard backup=—          assumptions=3 OK
#   judge_score            soft backup=quality    assumptions=3 OK
#   ...
```
