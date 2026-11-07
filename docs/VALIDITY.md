# VALIDITY.md — measurement validity: test the test

Every metric this repo reports carries a measurement-reliability column.
A proxy that misses the phenomenon is worse than no metric: the source
example is IVR testing, where the metric measured silence after the word,
not whether the consonant clipped. Evals report their own
confidence/validity, and a metric without a validity note is NOT trusted.

Executable form: `src/agenteval_bench/validity.py`, demo
`src/agenteval_bench/validity_demo.py`, tests `tests/test_validity.py`.

## Coverage

Covered: the four decision dimensions (quality, safety, cost,
reliability) in `decide` reports, and the deterministic scorer's four
matchers (exact_match, contains, regex, json_schema) in `run` and
`decide` reports. NOT covered yet: tiered `MutationScore` (kill rate),
regression aggregate scores vs pinned baselines, alt-test alignment
scores and p-values, novelty-oracle `novelty_score`. Those reports
carry no validity assessment; the "every metric" claim is scoped to the
covered set until they are wired in. Bugsmith and flywheel produce no
eval-metric report and are out of scope.

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

## Real validation: labels files

Hand-built pairs show what the mechanism does; a real validation runs
the scorer on recorded outputs against human truth labels:

```
agenteval-bench run --suite examples/adversarial-scorer.yaml --validity-labels examples/contains-truth-labels.yaml
```

`validate_matcher_labels(path)` loads a labels file — one matcher, its
phenomenon, and labeled cases (recorded `output`, the `expected` block
the scorer runs, a human `truth` label, and a `reason`). The proxy
verdicts come from running the real DeterministicScorer; the agreement
is graded by `validate_proxy`. Malformed files (unknown metric or keys,
duplicate ids, non-bool truth, empty reason, empty case list) raise.

The shipped labels file validates `contains` for the refund phenomenon
("output states the required fact") on four outputs quoted from the
adversarial suite and its comments: agreement 1/4, Wilson 95% CI [0.046,
0.699] — the matcher fails the phenomenon, and the interval
says so honestly. The ambiguous negation case is excluded with its
reason recorded in the file. One real labeled run exists; the corpus
grows by adding labels files, not by inventing pairs.

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
the `validity` key. The decision report and the scorer report travel
together: `decide` prints both tables and writes both sections
(`validity` + `scorer_validity`), so a quality/safety DIRECT grade —
which is about the proportion being exact arithmetic — is never read
apart from the matcher grades underneath it.

## Enforcing trust

"NOT trusted" is enforced two ways. First, the flag: every report
carries machine-readable `all_trusted` / `untrusted`, and a consumer
can gate on them. Second, the switch: `decide --strict-validity`
downgrades a GATE to INVESTIGATE (exit 2) when any metric is untrusted,
recording the evidence action as `evidence_action` in the report — the
downgrade is visible, never silent.

The switch defaults OFF, deliberately: in `decide` mode reliability is
replay-agreement, which is structurally untrusted, so a strict default
would make `decide` unable to gate at all. Strict mode is for
live-assessed decisions and for callers that want the validity gate to
bite; the default report still names every untrusted metric out loud.

## Built-in notes

`assess_scorer()` grades the deterministic scorer's matchers from
EVAL-RIGOR.md section 4: `exact_match` is DIRECT (string equality);
`contains`, `regex`, and `json_schema` are UNVALIDATED_PROXY (blind to
negation; pattern match, not semantic match; required-keys only).
Supply a `validate_proxy` result as a note to upgrade a matcher — a
note claiming VALIDATED_PROXY is re-checked against the policy bar, so
a forged grade raises instead of entering a report as trusted.

`assess_decision(decision, reliability_kind="live" | "replay")` grades
the four decision dimensions. Quality and safety are DIRECT for the
stated phenomenon ("share of this suite's cases passed"); the proportion
is exact, but the cases are only as good as their matchers, which is why
the scorer report travels with the decision report. Reliability in
replay mode is UNVALIDATED_PROXY (replay-agreement cannot detect
live-agent flakiness); live is graded exactly as `validate_proxy`
grades a proxy — VALIDATED_PROXY only when the Wilson lower bound on
the cross-run agreement clears the policy bar, else UNVALIDATED_PROXY
with the interval carried as the evidence. Cost over >= 2 observations
is DIRECT; a single observation is UNVALIDATED_PROXY (no measured
uncertainty); unmeasured is UNASSESSED.

`reliability_kind` is caller-declared, never auto-detected: the function
cannot see how the runs were produced, so it does not pretend to.

## What would falsify this doc

- If a metric's phenomenon cannot be stated precisely, the grade is
  decoration — restate the phenomenon or drop the metric, do not keep a
  DIRECT grade on a vague claim.
- If the labels corpus does not grow past the single shipped file, the
  proxy-validation path stays thin: one real run proves the mechanism,
  it does not validate the matchers. Add labels files or shrink the
  claim.
- If a lighter-weight check (e.g. a fixed validation split with a
  pre-registered agreement bar) empirically catches the same proxy
  failures with less machinery, replace `validate_proxy` with it.
