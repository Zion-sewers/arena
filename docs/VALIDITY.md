# VALIDITY.md — measurement validity: test the test

Every metric in an eval report carries a measurement-reliability column.
A proxy that misses the phenomenon is worse than no metric: the source
example is IVR testing, where the metric measured silence after the word,
not whether the consonant clipped. Evals report their own
confidence/validity, and a metric without a validity note is NOT trusted.

Executable form: `src/agenteval_bench/validity.py`, demo
`src/agenteval_bench/validity_demo.py`, tests `tests/test_validity.py`.

## Grades (closed set)

| Grade | Meaning |
|---|---|
| `direct` | The metric measures the phenomenon itself. |
| `validated_proxy` | A proxy validated against ground truth; the Wilson 95% lower bound on agreement clears the policy bar (default 0.80). |
| `unvalidated_proxy` | A proxy with no validation run, or one whose measured agreement failed the bar. Never trusted. |
| `unassessed` | No validity note at all. Never trusted. |

`trusted` is True only for `direct` and `validated_proxy`.

## Validating a proxy

```python
from agenteval_bench.validity import validate_proxy

validity = validate_proxy(
    metric="silence_after_word",
    phenomenon="whether the consonant clipped",
    proxy=[True] * 12 + [False] * 8,   # what the proxy said, per case
    truth=[True] * 12 + [True] * 8,    # what ground truth says, per case
)
validity.grade  # unvalidated_proxy: 12/20 agreement, lower bound 0.387
```

The compared quantity is the Wilson 95% interval LOWER bound on
agreement, never the point estimate: 5/5 agreement is a point estimate
of 1.0 with a lower bound of 0.565, and does not clear the 0.80 bar.
Mismatched lengths, an empty sample, and non-bool verdicts raise at the
boundary. The bar is a policy choice (`ValidityPolicy`,
`min_agreement_lower`, default 0.80) — a team owns its threshold.

## Reports

`ValidityReport` covers a closed metric set; construction raises if a
metric is missing (a report that omits a metric is not a validity
report) or unexpected. `summary()` prints the reliability column:

```
Validity: 3/4 metrics trusted
  cost         direct             agreement —                            TRUSTED
  quality      direct             agreement —                            TRUSTED
  reliability  unvalidated_proxy  agreement —                            UNTRUSTED
  safety       direct             agreement —                            TRUSTED
  UNTRUSTED: reliability — do not trust these metrics
```

`to_dict()` is the JSON form written into `decide --out` reports under
the `validity` key.

## Built-in notes

`assess_scorer()` grades the deterministic scorer's matchers from
EVAL-RIGOR.md section 4: `exact_match` is DIRECT (string equality);
`contains`, `regex`, and `json_schema` are UNVALIDATED_PROXY (blind to
negation; pattern match, not semantic match; required-keys only).
Supply a `validate_proxy` result as a note to upgrade a matcher.

`assess_decision(decision, reliability_kind="live" | "replay")` grades
the four decision dimensions. Quality and safety are DIRECT for the
stated phenomenon ("share of this suite's cases passed"); the proportion
is exact, but the cases are only as good as their matchers. Reliability
in replay mode is UNVALIDATED_PROXY (replay-agreement cannot detect
live-agent flakiness); live with a measured agreement is
VALIDATED_PROXY, carrying the cross-run agreement interval. Cost over
>= 2 observations is DIRECT; a single observation is UNVALIDATED_PROXY
(no measured uncertainty); unmeasured is UNASSESSED.

`reliability_kind` is caller-declared, never auto-detected: the function
cannot see how the runs were produced, so it does not pretend to.

## What would falsify this doc

- If a metric's phenomenon cannot be stated precisely, the grade is
  decoration — restate the phenomenon or drop the metric, do not keep a
  DIRECT grade on a vague claim.
- If `validate_proxy` is never called with real ground truth within
  ~6 months (only the demo's hand-built pairs), the proxy-validation
  path is ceremony — keep the built-in grades, delete the unused path.
- If a lighter-weight check (e.g. a fixed validation split with a
  pre-registered agreement bar) empirically catches the same proxy
  failures with less machinery, replace `validate_proxy` with it.
