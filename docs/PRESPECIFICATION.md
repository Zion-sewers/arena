# PRESPECIFICATION.md — pre-specification; exploratory/confirmatory split; 10-question review gate

How to write an agent-produced analysis in this repo without fooling
yourself. Mechanism source: Spiegelhalter, *The Art of Statistics*
(catalog G.2) — INSPIRED. The analysis protocol for agent-produced
analyses: an agent reads eval results and writes an analysis ("model X
improved", "failure mode Y dominates"). The protocol binds the analysis
plan BEFORE the data is touched, splits exploratory from confirmatory,
and runs the 10-question review gate over the reported finding.

This is the analysis counterpart to `docs/EXPERIMENT-DISCIPLINE.md`:
that doc pre-registers controlled *experiments* (#32); this one
pre-registers *analyses* of data. An analysis of experiment results may
link its plan to the experiment's registered digest
(`experiment_digest`), but the registries are separate files and the
link is metadata, not a cross-registry verification.

## 1. Pre-register the analysis plan

Before data arrives, write down the question, the hypotheses, the
assumptions (each with its falsifier), the sample size, and the decision
rule:

```python
from experiment import (
    AnalysisPlan, Assumption, register_analysis_plan,
)

plan = AnalysisPlan(
    name="reranker-v2-failure-modes",
    mode="confirmatory",
    question="Does reranker v2 reduce the hallucinated-citation failure mode?",
    hypotheses=(
        "reranker v2 cuts hallucinated citations by at least 10pp vs v1",
    ),
    assumptions=(
        Assumption(
            text="the eval suite's citation cases represent production traffic",
            falsifier="suite citation cases differ from production by >20pp",
        ),
    ),
    n_planned=200,
    decision_rule="confirm iff the 95% interval on the citation-error "
                  "delta excludes 0 in v2's favour",
    data_source="golden replay suite, seed 42",
)
digest = register_analysis_plan(plan, "plans/analyses.jsonl")
```

The registry is hash-chained: each record commits to the previous
record's digest, so a plan cannot be silently rewritten after data
arrives. A rewritten registry is detected (`AnalysisRegistryCorrupted`)
and the finding is refused as `UNREGISTERED_PLAN`.

The mode split is structural, not a label. A confirmatory plan without
hypotheses or without a decision rule is refused at construction, and
an exploratory plan WITH hypotheses or a decision rule is refused —
a hypothesis stated in advance IS a confirmatory plan, and there is
nothing to decide in an exploration, only something to describe.

## 2. The exploratory/confirmatory split

| | confirmatory | exploratory |
|---|---|---|
| Hypotheses | >= 1, pre-specified | none — stating one makes it confirmatory |
| Decision rule | required | forbidden |
| Sample promise | n_observed within tolerance of n_planned | n_observed stated (>= min_n); tolerance not enforced |
| Uncertainty interval | required | optional |
| Verdict on a clean review | CONFIRMED | EXPLORATORY_ONLY — a finding, never a conclusion |

An exploratory finding can only ever yield `EXPLORATORY_ONLY`. Relabeling
it as confirmatory is refused (`MODE_MISMATCH`): the finding must carry
the same mode the plan registered.

## 3. The 10-question review gate

```python
from experiment import AnalysisFinding, check_analysis

report = check_analysis(digest, "plans/analyses.jsonl", AnalysisFinding(
    plan_digest=digest,
    mode="confirmatory",
    hypothesis="reranker v2 cuts hallucinated citations by at least 10pp vs v1",
    claim="v2 cuts hallucinated citations by 14pp",
    n_observed=200,
    interval=(0.08, 0.20),
    data_digest="sha256:golden-suite-seed42",
    data_accessed_at="2026-11-07T19:00:00+00:00",
    code_version="arena@dbf6909",
))  # raises AnalysisViolation naming the violation otherwise
```

The ten questions (this repo's operationalization — **adaptation**, not
a verbatim quote of the book; the issue names the themes but lists no
question texts):

1. **Is the analysis plan registered?** The digest must be in the chain
   (`UNREGISTERED_PLAN`).
2. **Was the plan written before the data was touched?**
   `registered_at <= data_accessed_at` on the reported timestamps
   (`PLAN_AFTER_DATA`).
3. **Is the analysis question stated in advance?**
   (`QUESTION_UNSTATED`).
4. **Is the analysis labeled confirmatory or exploratory, and does the
   finding carry the same label?** (`MODE_MISMATCH`).
5. **Is the tested hypothesis one of the pre-specified hypotheses
   (confirmatory), or explicitly absent (exploratory)?** A confirmatory
   finding must quote a registered hypothesis verbatim; a reworded or
   unregistered hypothesis matches nothing and is refused as
   `POST_HOC_CONFIRMATION`. Hypothesis matching is exact by design,
   the same discipline as #32's exact look fractions.
6. **Is the sample size stated, and does the observed n match the plan?**
   Below `min_n` is `SAMPLE_TOO_SMALL`; a confirmatory n outside
   tolerance is `SAMPLE_DEVIATION`.
7. **Are the assumptions stated, each with what would break it?**
   (`ASSUMPTIONS_UNSTATED`).
8. **Is the uncertainty on the headline number stated as an interval,
   not a point?** (`UNCERTAINTY_MISSING`).
9. **Is the decision rule stated in advance (what counts as confirmed)?**
   (`DECISION_RULE_MISSING`).
10. **Can someone else reproduce this from the plan + data + code?**
    The finding must carry a data digest and a code version
    (`REPRODUCIBILITY_MISSING`).

The gate prints all ten results with evidence on every path, including
refusals, so a refusal names exactly which questions failed and why.

## 4. The review gate from the command line

```bash
agenteval-bench register-analysis-plan --plan plan.json [--registry plans/analyses.jsonl]
agenteval-bench prespec --finding finding.json [--plan plan.json] [--registry plans/analyses.jsonl] [--out report.json]
```

`prespec` exits 0 only if the review passes (`CONFIRMED` or
`EXPLORATORY_ONLY` — an exploratory-only pass is a gate pass, not a
confirmation); exit 1 names the violations; exit 2 is unusable input.
With `--plan`, the plan file is cross-checked against the registry
first: the finding's digest must load to the same plan as the file, so
a plan edited after registration is unusable input, never a refused
finding.

## 5. Violations fail CI

```python
from experiment import check_analysis

report = check_analysis(digest, "plans/analyses.jsonl", finding)
```

`check_analysis` raises `AnalysisViolation` on any violation; wire it
into CI and violations fail the build. CI runs the full drill battery
via `python -m experiment.demo_prespec`, which exits non-zero unless
the compliant paths (confirmatory and exploratory) verify clean and
every refusal fires.

## Honesty boundaries (stated, not hidden)

- The reviewer is a **consistency checker over reported evidence** —
  reported timestamps against the registry, reported hypotheses against
  the plan, reported n against n_planned, reported interval shape. It
  does not see raw data, so a fabricated-but-internally-consistent
  report verifies clean. What the registry binds is the *plan*
  (immutably, via the hash chain), not the data.
- The hash chain delivers immutability, not precedence: `registered_at`
  is recorded but nothing proves the plan was registered *before* data
  collection except the reported timestamps (Q2) and the workflow rule
  (require the digest before the run starts). The executable check that
  survives timestamp forgery is Q5: no hypothesis text, no confirmation.
- A vacuous decision rule ("we will decide somehow") passes Q9: the
  gate checks the rule is present and quotes it verbatim in the report
  for the human reviewer. Presence is enforced; meaning is reviewed.
- An absurdly wide or narrow interval passes Q8: the gate checks the
  interval is present and well-formed (finite, lo <= hi). The interval
  is printed in the report for the reviewer.
- There is **no amendment path**: changing a plan means registering a
  NEW plan; the old record stays, and findings on the old digest remain
  reviewable against the old plan.
- The registry is a **single-writer** file: concurrent
  `register_analysis_plan` calls can chain to the same stale
  predecessor, and the next read fails the whole file closed
  (`AnalysisRegistryCorrupted`). One writer, or serialize externally.
- The prespec registry is a **separate chain** from the experiment
  registry (`registry.py`): the same construction, but a separate file,
  so this change cannot regress the #32/#46 seams. `find_analysis_plan`
  is the equality check against that chain.
- Evidence levels: the ten questions and the mode-split enforcement are
  **adaptations** of the book's themes; the gate mechanics are
  **established** repo patterns (hash-chained registry, exact-match
  discipline, and the review/check raising pair from #32/#36/#46).
