# OEC.md — agent change = controlled experiment: OEC + guardrails + trust gates

How to ship an agent change in this repo without fooling yourself.
Mechanism source: Kohavi, Tang & Xu, *Trustworthy Online Controlled
Experiments* (ch. 5, trust tests; ch. 1, OEC). Distinct from #32's
variance-reduction discipline (pre-registration, no peeking, CUPED)
and #36's aggregation discipline (Simpson-safe pooling): this governs
whether a change may SHIP at all.

## 1. Pre-register the OEC + guardrails before the change

Before data arrives, write down WHICH metric decides the change and
which metrics must not regress:

```python
from experiment import ExperimentDesign, MetricDeclaration, register_plan
from experiment.types import ExperimentPlan

plan = ExperimentPlan(name="reranker-v2-vs-v1", n_planned=800,
                      alpha=0.05, looks=(0.5, 1.0))
digest = register_plan(plan, "plans/experiments.jsonl")

design = ExperimentDesign(
    plan_digest=digest,
    metrics=(
        MetricDeclaration("task_success", "oec", "higher_better"),
        MetricDeclaration("safety_violations", "guardrail",
                          "lower_better", 0.02),  # max regression: 2pp
    ),
    expected_allocation=(0.5, 0.5),
    surprise_z=3.0,
)
```

The design binds to the registered plan digest, so the OEC and the
guardrail budgets are registered alongside the plan, before the
change. Exactly one metric is the OEC; a guardrail without a
pre-registered max regression is refused at construction; a decision
metric carrying a regression budget is refused too. Evidence that
reports its final z on a different metric is `OEC_MISMATCH` — there
is no path that ships on a metric the design never named OEC.

## 2. Trust gates run before results are read

The ship gate evaluates, in this order, and short-circuits:

1. **Plan binding + discipline** — the evidence replays through
   `verify_experiment` (#32): unregistered plan, off-schedule peek,
   fabricated decision, sample deviation, or missing CUPED all refuse
   the ship before anything else is examined.
2. **SRM** — sample-ratio mismatch: chi-square goodness-of-fit of the
   observed per-arm assignment counts against the registered
   allocation, flagged at p < 0.001 (Kohavi convention). An SRM
   failure blocks interpretation of everything else. Too few
   assignments to test (any expected count < 5) is INSUFFICIENT, never
   a pass.
3. **A/A** — the same (unchanged) agent on both sides of the null
   split must not differ significantly at the registered alpha. A
   significant A/A means the randomizer or the pipeline is broken.
   Arms below `min_aa_n` (default 100) have no power: INSUFFICIENT,
   never a pass.
4. **Guardrails** — each declared guardrail is a one-sided
   non-inferiority check graded on the LOWER bound of the
   good-direction delta (the repo's lower-bound discipline, #43/#45):
   the guardrail holds iff the lower bound clears the pre-registered
   max regression. A point estimate inside the budget with a lower
   bound outside it FAILS. A declared guardrail with no observation is
   UNMEASURED; an observation for an undeclared metric is refused.
5. **OEC win** — only now is the result read: the final look must
   have rejected, in the OEC's declared direction. A non-significant
   final look is `NO_SIGNIFICANT_OEC_WIN`; a significant rejection in
   the wrong direction is `SIGNIFICANT_OEC_HARM`.

A trust-failing experiment with a huge OEC win reports the trust
failure, never the win. The order is enforced in code
(`experiment.oec.ship_decision`) and pinned by a test, not a doc.

## 3. Surprising wins require replication (Twyman's law)

A final |z| beyond the pre-registered `surprise_z` (default 3.0) is a
surprising win: too-exciting results need a second look. The gate
returns `NEEDS_REPLICATION` — not SHIP — until replication evidence
arrives: a second run's z, significant one-sided at the registered
alpha in the same direction, with n >= `min_replication_n` (default
30) and a run_id distinct from the original run. A replication that
re-reports the same run, is too small, or does not confirm blocks the
ship (`REPLICATION_SAME_RUN` / `REPLICATION_TOO_SMALL` /
`REPLICATION_NOT_CONFIRMED`).

The surprise threshold must exceed the plan's final sequential
boundary (else every significant win would need replication); the gate
checks this against `group_sequential_boundaries` for the loaded plan
and refuses the design otherwise (`SURPRISE_THRESHOLD_INVALID`).

## 4. Shipping on a single exciting number is NOT allowed

```python
from experiment import check_ship

check_ship(design, evidence, "plans/experiments.jsonl")  # raises ShipRefused otherwise
```

SHIP requires all of: registered plan + clean discipline, passing
SRM and A/A, unregressed guardrails, a significant OEC win in the
declared direction, and (for surprising wins) confirming replication.
`ship_decision` returns the verdict without raising
(`ship` / `no_ship` / `needs_replication` + named reasons);
`check_ship` is the raising CI-gate form. From the command line:

```bash
agenteval-bench ship --design examples/ship-design.json \
    --evidence examples/ship-evidence.json \
    --registry examples/ship-registry.jsonl
# exit 0 SHIP, 1 NO_SHIP / NEEDS_REPLICATION, 2 bad input
```

The shipped examples are a compliant run (final z=2.5, SRM/A/A
clean, guardrail unregressed) and exit 0. Flip `srm_counts` to
`[350, 450]` and the same command exits 1 naming `SRM_FAIL`.

## Honesty boundaries (stated, not hidden)

- The gate is a **consistency checker over reported evidence** —
  reported counts against the registered split, reported decisions
  against reported z's, reported guardrail counts against the
  registered budgets. It does not see raw data, so a
  fabricated-but-internally-consistent report verifies clean. What
  the registry binds is the *plan* (immutably, via the hash chain),
  not the data. Same boundary as #32.
- Replication independence is **workflow-bound**: the gate checks
  significance, direction, minimum n, and a distinct run_id, but a
  replication run_id minted from the same data verifies clean. The
  defense is workflow — the replication run_id must come from a
  separately recorded run — not cryptography.
- CUPED evidence cannot be expressed in the ship CLI's JSON: a plan
  that declares a covariate needs a `CupedResult`, which the JSON
  carries no field for, so such evidence fails verification with
  `CUPED_REQUIRED`. The CLI refuses rather than silently verifying a
  plan it cannot fully check; use the in-process seam
  (`ship_decision` with a `CupedResult` in the evidence) instead.
- Guardrails are **proportion metrics only** for v1 (rates are the
  repo's lingua franca: decision.py, simpson strata). A continuous
  guardrail needs a mean/variance evidence type this module does not
  own, and is refused by the JSON loader rather than approximated.
- The hash chain delivers immutability, not precedence: `registered_at`
  is recorded but nothing proves the design was registered *before*
  data collection (same as #32). The defense is workflow — require the
  digest before the run starts — not cryptography.
- The chi-square survival function is a stdlib-only regularized
  incomplete gamma (Numerical Recipes series + continued fraction),
  validated against known values (df=1 via erfc, df=2 = exp(-x/2),
  df=3 spot values) and a seeded null-uniformity check in the tests.
- Evidence levels: the SRM/A/A/guardrail/replication rules are
  **adaptations** of Kohavi et al. ch. 5; the chi-square p-values and
  the demo's drill outcomes are **measured** on the repo's own code
  (rerun `python -m experiment.demo_oec`); the Kohavi ch. 5
  conventions (SRM alpha 0.001) are **established** from the book.

## Coverage boundary (stated, not hidden)

- There is no live experiment transport in this repo (README:
  "LLM-as-judge is planned, not implemented"), so the design and the
  evidence are REPORTED records supplied by the caller. The consumers
  of this module are the `ship_decision` seam, the `ship` CLI, and the
  demo. Wiring the gate into a live traffic splitter is a named
  follow-up for when such a transport exists; it is not faked here.
  Same coverage-boundary pattern #44/#45 used.
