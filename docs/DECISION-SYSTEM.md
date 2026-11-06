# DECISION-SYSTEM.md — evaluation as a decision system (arena#42)

Evaluation is a decision system, not a score: evidence → uncertainty →
threshold → action. A reported score that maps to no decision is not an
evaluation, it is decoration. This doc defines the decision system that
`src/agenteval_bench/decision.py` implements; the code is the enforcement,
this doc is the explanation.

## 1. The four dimensions

Every decision measures all four, every time:

| Dimension | Evidence | Compared quantity |
|---|---|---|
| quality | pass proportion over quality-tagged cases | 95% Wilson lower bound |
| safety | pass proportion over safety-tagged cases | 95% Wilson lower bound |
| cost | mean per-case cost vs the policy budget | 95% normal upper bound |
| reliability | per-case pass/fail agreement across ≥2 seeded runs | 95% Wilson lower bound |

A dimension with no evidence is **unmeasured**, and an unmeasured
dimension forces INVESTIGATE, never GATE. The common ways a dimension
ends up unmeasured: a suite with no safety cases, no cost function
supplied, or `reliability_runs=1` (agreement across one run is 1.0 by
construction, so the module refuses to compute it).

## 2. Uncertainty is the compared quantity, not the point estimate

For a proportion dimension the decision compares the **lower bound** of
the 95% Wilson score interval against the threshold:

- lower bound ≥ threshold → the dimension passes;
- upper bound < threshold → the dimension fails confidently;
- otherwise the interval straddles the threshold → INVESTIGATE.

For cost the mirror rule applies: the **upper bound** of the mean-cost
interval is compared against the budget. A point estimate that clears a
threshold while its interval does not is not a pass; it is a request for
more cases. `examples/decision-suite.yaml` is the worked example: 6/6
quality and 4/4 safety, all passing, still INVESTIGATEs at medium risk
because n is small.

## 3. Thresholds are per risk level

`DecisionPolicy` carries one `RiskThresholds` set per `RiskLevel`
(low/medium/high/critical). A higher risk level never has a looser
threshold; the policy constructor rejects a non-monotone policy, and it
rejects a policy missing a risk level. The built-in defaults
(`decision.default_policy`) are a starting point, not doctrine: a team
owns its thresholds and records them in its policy file
(`examples/decision-policy.yaml`).

## 4. The decision

The overall action is the worst across dimensions:

- any dimension ROLLBACK → ROLLBACK (an interval entirely on the wrong
  side of its threshold);
- else any dimension INVESTIGATE → INVESTIGATE (a straddling interval or
  an unmeasured dimension);
- else GATE.

Every `Decision` record carries the per-dimension verdicts with their
evidence (point estimate, interval, threshold, reason), so a decision is
auditable from the record alone. `Decision` also enforces its own
invariant at construction: the overall action must be the worst
per-dimension action.

## 5. Wiring it up

Python API:

```python
from agenteval_bench import EvalRunner, default_policy, RiskLevel

decision = EvalRunner().run_decision(
    suite, agent_fn, default_policy(), RiskLevel.HIGH,
    cost_fn=lambda case, output: case.cost,  # per-case cost units
    reliability_runs=2,
)
print(decision.summary())
```

CLI (replay mode — recorded outputs and costs, reproducible in CI):

```bash
agenteval-bench decide --suite examples/decision-suite.yaml \
    --policy examples/decision-policy.yaml --risk medium \
    [--out report.json]
# exit 0 GATE, 1 ROLLBACK, 2 INVESTIGATE, 3 usage error
```

Cases declare their dimension in the suite YAML (`dimension: quality`
or `dimension: safety`; default `quality`). Anything else is rejected
at load time: a typo must fail loudly, not silently score as quality.
Reliability is a cross-run property and cost comes from the cost
function, so neither is a case tag. Cases may also carry a recorded
`cost:` value, which the CLI uses as its cost function.

## 6. What this replaces, and what it does not

`EvalRunner.run_ci(threshold=...)` is superseded by this system: it
takes a threshold and ignores it, comparing nothing. It is left
untouched as a legacy path so existing callers do not change behavior
silently; new gates should use `run_decision` / `decide`.

What this is not: it does not judge output quality beyond the
deterministic scorer (no LLM-as-judge in this repo), it does not measure
costs the caller does not supply, and it does not turn a small sample
into a confident verdict. The honest failure mode is INVESTIGATE.

## What would falsify this doc

- If a dimension's interval rule is ever replaced by a point-estimate
  comparison, the straddle test in `tests/test_decision.py` fails first;
  delete the rule and the test together, never one without the other.
- If per-risk thresholds are collapsed to a single global threshold, the
  same-evidence-gates-low/investigates-critical test fails first.
- If `decide` is ever wired to a live agent with no recorded cost path,
  cost will be unmeasured and every decision will be INVESTIGATE — that
  is the system refusing to decide on partial evidence, not a bug.
