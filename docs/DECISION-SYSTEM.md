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
| reliability | per-case pass/fail agreement across ≥2 repeated runs | 95% Wilson lower bound |

A dimension with no evidence is **unmeasured**, and an unmeasured
dimension forces INVESTIGATE, never GATE. The common ways a dimension
ends up unmeasured: a suite with no safety cases, no cost function
supplied, or `reliability_runs=1` (agreement across one run is 1.0 by
construction, so the module refuses to compute it).

Reliability has one more caveat, and it lives in the consumer, not the
core: the CLI replays recorded outputs, so its reliability runs re-score
one recording. That agreement is **replay-agreement**, not live-agent
reliability: it cannot detect a flaky live agent whose recorded outputs
happen to pass. The CLI labels it as such in its output and its report
JSON. The Python API measures real reliability when the agent is live
and varies across calls; pass `seeded_agent_fn` to let the run seed
perturb the agent, which is what makes the repeated runs independent
samples instead of replays of one execution.

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
rejects a policy missing a risk level. That holds for the cost budget
too: it tightens with risk (200 / 100 / 50 / 25 in the defaults), because
a costly wrong decision hurts more at higher risk. The built-in defaults
(`decision.default_policy`) are a starting point, not doctrine: a team
owns its thresholds and records them in its policy file
(`examples/decision-policy.yaml`). The policy loader rejects unknown
keys, non-numeric values, and non-finite numbers, so a typo'd threshold
fails loudly instead of silently falling back to a default. A report
records the policy's fingerprint (`decision.policy_fingerprint`), so a
past decision can be checked against the policy file that produced it.

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
    seeded_agent_fn=lambda prompt, seed: agent(prompt, seed),  # optional
)
print(decision.summary())
```

CLI (replay mode — recorded outputs and costs, reproducible in CI):

```bash
agenteval-bench decide --suite examples/decision-suite.yaml \
    --policy examples/decision-policy.yaml --risk medium \
    [--reliability-runs 2] [--out report.json]
# exit 0 GATE, 1 ROLLBACK, 2 INVESTIGATE, 3 usage error
```

Three shipped example suites show all three outcomes through the CLI:
`decision-suite.yaml` INVESTIGATEs at medium (exit 2, small n),
`decision-suite-gate.yaml` GATEs at medium (exit 0, 60+40 cases),
`decision-suite-rollback.yaml` ROLLBACKs at medium (exit 1, safety fails).
The report JSON carries the action, risk, suite name and digest, seed,
reliability-run count, policy fingerprint, and per-dimension evidence
(point, interval, successes/trials or n, threshold, reason) — enough to
reconstruct why from the file alone. Its reliability entry is labelled
`kind: "replay-agreement"` (see §1).

Cases declare their dimension in the suite YAML (`dimension: quality`
or `dimension: safety`; default `quality`). Anything else is rejected
at load time: a typo must fail loudly, not silently score as quality.
Reliability is a cross-run property and cost comes from the cost
function, so neither is a case tag. Cases may also carry a recorded
`cost:` value, which the CLI uses as its cost function.

## 6. What this replaces, and what it does not

`EvalRunner.run_ci(threshold=...)` is deprecated (it emits a
`DeprecationWarning`): it takes a threshold and ignores it, and the
`run --ci` gate compares a point-estimate pass rate with no dimensions,
no uncertainty, and no unmeasured rule. It is kept so existing callers
do not change behavior silently, and `run --ci` now prints a legacy
warning. New gates should use `run_decision` / `decide`. Exit codes
differ per subcommand and are documented in `docs/spec.md` §6: `run`
uses 0 clean / 1 bad input / 2 gate failed, while `decide` uses 0 GATE /
1 ROLLBACK / 2 INVESTIGATE / 3 usage error — a wrapper must not read
one command's codes with the other's table.

Named follow-up, not done here: migrate the tiered pipeline's gate
(`tiered.py:_run_gate`, which shells out to `run --ci`) to `decide`, so
the repo's own CI gate runs on the decision system instead of the
legacy point-estimate gate. Until then this system is a real, exercised
gate available to callers, not yet the repo's default gate.

What this is not: it does not judge output quality beyond the
deterministic scorer (no LLM-as-judge in this repo), it does not measure
costs the caller does not supply, and it does not turn a small sample
into a confident verdict. The honest failure mode is INVESTIGATE. The
cost interval is a normal approximation with no t correction, so it
understates uncertainty at small n; a single cost observation can never
GATE. The policy fingerprint, not a version field, is what ties a
decision report to its policy.

## What would falsify this doc

- If a dimension's interval rule is ever replaced by a point-estimate
  comparison, the straddle test in `tests/test_decision.py` fails first;
  delete the rule and the test together, never one without the other.
- If per-risk thresholds are collapsed to a single global threshold, the
  same-evidence-gates-low/investigates-critical test fails first.
- If `decide` is ever wired to a live agent with no recorded cost path,
  cost will be unmeasured and every decision will be INVESTIGATE — that
  is the system refusing to decide on partial evidence, not a bug.
- If the CLI's replay-agreement label is ever dropped while the CLI
  still replays, the reliability column becomes a live-agent claim the
  evidence cannot support; restore the label or mark replay reliability
  unmeasured.
