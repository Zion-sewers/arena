# agenteval-bench — Specification

> Authoritative design and contract document for `agenteval-bench`.
> Status: v0.1.0 (MVP shipped). Unimplemented items are marked **`planned`** with
> estimated milestones. This document supersedes the older roadmap notes; the
> living roadmap now lives in [`../TODO.md`](../TODO.md).

---

## 1. Motivation

Teams deploying LLM agents have no systematic way to answer: *"Is my agent getting
better or worse?"* Evaluation is fragmented — LangSmith traces, manual notebooks,
ad-hoc scripts — and none of it is reproducible in CI. There is no unified CLI that
treats agent evaluation like a test suite, with deterministic assertions and
regression gates.

`agenteval-bench` exists to make agent evaluation **repeatable, auditable, and
CI-native**, starting from a deterministic core that needs no model calls and
expanding toward LLM-judged and regression-tracked evaluation.

## 2. Goals

- **G1 — Deterministic by default.** Core scoring must be reproducible without LLM
  calls, so CI results are stable and diffable.
- **G2 — Declarative suites.** Eval definitions live in versioned YAML that is
  schema-validated and portable across machines and CI.
- **G3 — CI gating.** A single command fails the build when quality degrades below a
  configurable threshold, with stable exit codes.
- **G4 — Cost awareness.** Token budgets are explicit and bounded per run.
- **G5 — Extensibility.** Scoring and agent wiring are pluggable so the tool grows
  toward LLM-as-judge and richer reporting without rewriting the core.

## 3. Non-goals

- **NG1 — Web UI / dashboard.** No hosted UI in the MVP; reports are file artifacts.
- **NG2 — Distributed execution.** Single-process, single-host evaluation.
- **NG3 — Arbitrary custom scorer plugins** beyond the documented extension points.
- **NG4 — Agent trace replay / observability.** We score recorded text outputs, not
  full agent trajectories.
- **NG5 — Training or fine-tuning.** This is an evaluation tool, not a training loop.

## 4. Architecture

### 4.1 Module layout

```
src/agenteval_bench/
  __init__.py     # Public API surface (re-exports)
  models.py       # Data models: EvalSuite, EvalCase, EvalResult, RunResult,
                  #   ExpectedOutput, RubricCriterion, CostBound
  scoring.py      # DeterministicScorer (exact, contains, regex, json_schema)
  engine.py       # EvalRunner — orchestrates suite execution
  decision.py     # Decision system: evidence -> uncertainty -> threshold -> action
                  #   (Dimension/RiskLevel/DecisionAction, Wilson intervals, decide)
  decision_demo.py # Seeded end-to-end demo of the decision system
  validity.py     # Measurement validity: every metric carries a reliability
                  #   column (ValidityGrade, validate_proxy, ValidityReport)
  validity_demo.py # Seeded end-to-end demo of measurement validity
  judges.py       # Judge decorrelation: HOST != FAMILY (Judge, allocate,
                  #   AllocationReport; family-keyed allocation, unreachable
                  #   judges reported as holes, VARIES-BY-MODEL never a
                  #   second slot)
  judges_demo.py  # Seeded end-to-end demo of judge decorrelation
  calibration.py  # Judge calibration: confidence is not evidence
                  #   (ConfidenceObservation, calibrate, JudgeCalibration;
                  #   per-bin stated confidence vs Wilson lower bound on
                  #   measured agreement; only CALIBRATED may gate)
                  #   First agenteval_bench -> alt_test import (types only):
                  #   the bridge reads recorded AltTestItem labels.
  calibration_demo.py # Seeded end-to-end demo of judge calibration
  diverge.py      # Anti-mode-collapse: /diverge with 5 candidates +
                  #   p_conventional (Candidate, DivergeSet, DiversityReport,
                  #   check_p_conventional, select, require_high_stakes_decision;
                  #   pair similarity = max(content-token, hard-token) Jaccard,
                  #   weakest-link grade, Wilson interval on the modal share;
                  #   single-sample generation refused on high stakes)
  diverge_demo.py # Seeded end-to-end demo of /diverge
  regression.py   # Production failures -> frozen versions -> release gate
  snapshots.py    # Immutable benchmark snapshots (hash-verified store)
  replay.py       # Deterministic seeded replay logs (bit-exact re-runs)
  tiered.py       # Tiered eval pipeline (gate -> smoke -> expanded -> nightly)
  tracebench.py   # Trace-mined benchmarks: mine traces -> living benchmark
                  #   (owner + cadence per benchmark, corpus_kind on every
                  #   artifact) -> scheduled regression runner with
                  #   gate/rollback/investigation actions; the loop is closed
                  #   across runs via --state-dir (previous version +
                  #   failure streak chained, changed oracles published as
                  #   updates, mined cases retired, base cases carried
                  #   forward) and the drift gate fails the run when the
                  #   benchmark moves under the candidate; the corpus
                  #   boundary fails closed on malformed lines, both output
                  #   locations are preflighted writable before publishing,
                  #   the manifest describes the benchmark actually gated,
                  #   and a stale previous version dispatches an
                  #   investigation action
  tracebench_demo.py  # Demo: mine the sample corpus, regress oracle +
                  #   degraded candidates (gate -> investigation -> rollback),
                  #   then the closed loop (three state-chained weeks) and
                  #   the drift gate (oracle change fails, clears next week);
                  #   also the scheduled workflow's canary — it runs there
                  #   so the schedule exercises the red path every week
  cli.py          # CLI entry point (run --suite --ci --threshold; decide --suite --policy)
tests/            # pytest suite (engine + CLI gateway)
examples/         # Golden replay suites (YAML)
docs/             # Specifications, archive, roadmap references
.github/          # CI + code-review workflows
```

### 4.2 Component responsibilities

| Component            | Responsibility                                                        |
|----------------------|-----------------------------------------------------------------------|
| `EvalSuite.from_yaml`| Parse + validate YAML into typed `EvalSuite`/`EvalCase` models.       |
| `DeterministicScorer`| Apply configured matchers; return `EvalResult` with details.          |
| `EvalRunner`         | Iterate cases, invoke `agent_fn`, collect `EvalResult`s into `RunResult`. |
| `ValidityReport`     | Validity assessment per metric; a metric without a note is UNASSESSED and untrusted. |
| `AllocationReport`   | Family-decorrelated judge allocation; unreachable judges are holes, never dropped. |
| `JudgeCalibration`   | Per-bin stated confidence vs measured agreement; only CALIBRATED may gate. |
| `ShipVerdict`        | OEC + guardrails + trust gates (SRM, A/A, replication); SHIP only when all pass. |
| `Selection`          | /diverge selection record: winner vs conventional scores, margin, closed outcome, reason from measured facts; only a diverse selection with a non-overconfident p may decide on high stakes. |
| `cli.main`           | Parse argv, load suite, build replay `agent_fn`, print summary + gate. |

### 4.3 Data-flow diagram

```mermaid
flowchart LR
    A[Eval YAML] --> B[EvalSuite.from_yaml]
    B --> C[EvalSuite]
    C --> D[EvalRunner.run]
    D --> E{agent_fn}
    E -->|live| F[Agent output]
    E -->|replay| G[case.output]
    F --> H[DeterministicScorer.score]
    G --> H
    H --> I[EvalResult]
    I --> J[RunResult]
    J --> K[summary / CI gate]
```

## 5. Data model (contract)

All models are `dataclasses`. Field defaults are shown.

| Model              | Fields                                                                        | Notes |
|--------------------|-------------------------------------------------------------------------------|-------|
| `ExpectedOutput`   | `exact: str \| None`, `contains: list[str]`, `regex: str \| None`, `json_schema: dict \| None` | At least one matcher should be set; none ⇒ failure. |
| `RubricCriterion`  | `criterion: str`, `weight: float = 1.0`, `description: str = ""`              | Used by LLM-as-judge (**`planned`**). |
| `CostBound`        | `max_input_tokens: int = 1500`, `max_output_tokens: int = 120`               | Loaded from `cost_bound`; enforced by caller (**`planned`**). |
| `EvalCase`         | `id: str`, `input: str`, `expected: ExpectedOutput`, `rubric: list`, `skip: bool = False`, `output: str \| None = None` | `output` enables CLI replay. |
| `EvalSuite`        | `name: str`, `version: int = 1`, `cases: list[EvalCase]`, `cost_bound: CostBound` | `from_yaml(path)` is the only constructor today. |
| `EvalResult`       | `case_id: str`, `passed: bool`, `score: float`, `details: dict`              | `score` ∈ [0.0, 1.0]. |
| `RunResult`        | `suite_name: str`, `results: list[EvalResult]`, `total`, `passed`, `failed`, `skipped`, `pass_rate: float`, `summary()` | Aggregate of a run. |

### 5.1 Scoring semantics

For a single `EvalCase`, `DeterministicScorer.score`:

1. Runs **every** configured matcher (`exact`, `contains`, `regex`, `json_schema`).
2. `passed = all(configured matchers passed)`.
3. `score = (sum of weights for passing matchers) / (sum of all configured weights)`.
4. If **no** matcher is configured, returns `passed=False, score=0.0` with
   `details.error = "No expected output defined"`.

Matcher rules:

- `exact`: `agent_output.strip() == expected.exact.strip()`.
- `contains`: `all(needle in agent_output for needle in expected.contains)`.
- `regex`: `re.search(expected.regex, agent_output) is not None`.
- `json_schema`: output parses as JSON; if a dict, **all** `required` keys present
  (non-dict JSON satisfies the matcher only when `required` is empty).

### 5.2 `EvalRunner` semantics

- `run(suite, agent_fn)`: iterates cases in order; skipped cases contribute a result
  with `details.skipped = True` and are **excluded** from `pass_rate` denominator.
- `pass_rate = passed / len(scored)` where `scored = results without skipped`.
- `run_ci(suite, agent_fn, threshold=1.0)`: DEPRECATED (emits a
  `DeprecationWarning`). It delegates to `run` and returns the
  `RunResult`; the CLI performs the threshold comparison and `sys.exit(2)` on
  failure. It gates on a point-estimate pass rate with no dimensions, no
  uncertainty, and no unmeasured rule; new gates use `run_decision`.
- `run_decision(suite, agent_fn, policy, risk, cost_fn=None, reliability_runs=2)`:
  maps a suite to a typed GATE / ROLLBACK / INVESTIGATE `Decision` over the
  four dimensions (quality, safety, cost, reliability) with 95% intervals
  compared against per-risk thresholds; an unmeasured dimension forces
  INVESTIGATE. Full contract in `docs/DECISION-SYSTEM.md`. This supersedes
  the `run_ci` threshold path for new gates.

## 6. API contract

### 6.1 CLI

```
agenteval-bench run --suite <FILE> [--ci] [--threshold FLOAT]
```

| Flag          | Default | Meaning                                                |
|---------------|---------|--------------------------------------------------------|
| `--suite`     | —       | Path to YAML suite (required; exit `1` if missing).    |
| `--ci`        | off     | Enable the pass-rate gate.                             |
| `--threshold` | `1.0`   | Minimum `pass_rate` to pass the gate (CI only).        |

Behavior:

- Cases without a recorded `output` are **skipped** in CLI mode (the CLI cannot call
  a live agent) and reported with a "Note".
- Exit codes: `0` clean / gate passed, `1` bad input, `2` gate failed.

```
agenteval-bench decide --suite <FILE> [--policy <FILE>] [--risk <LEVEL>]
                       [--seed 42] [--reliability-runs 2] [--out report.json]
```

Maps a replay suite to a decision (evidence -> uncertainty -> threshold ->
action) over the four dimensions; see `docs/DECISION-SYSTEM.md`. Exit
codes are per subcommand and must not be read with the other command's
table:

| Subcommand | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `run` | clean / gate passed | bad input | gate failed | — |
| `decide` | GATE | ROLLBACK | INVESTIGATE | usage error |
| `calibrate` | calibrated | bad input | mis/uncalibrated | — |
| `ship` | SHIP | NO_SHIP / NEEDS_REPLICATION | bad input | — |
| `diverge` | selection made | bad input | refused (paraphrase set / overconfident p / single sample on high stakes) | — |
| `register-design` | registered | — | bad input | — |

> The `compare` and `report` subcommands described in earlier drafts are **`planned`**
> (see §9) and are **not** implemented in v0.1.0; invoking an unknown command exits `1`.

### 6.2 Python API

```python
from agenteval_bench import EvalSuite, EvalRunner, DeterministicScorer

suite = EvalSuite.from_yaml("eval.yaml")
runner = EvalRunner()                       # or EvalRunner(DeterministicScorer())
result = runner.run(suite, agent_fn=lambda q: "...")   # Callable[[str], str]
print(result.summary())
```

`agent_fn` contract: `Callable[[str], str]`. Advanced signatures
(`Callable[[EvalInput], EvalOutput]`) are **`planned`**.

## 7. Extension points

- **Custom scorer.** Subclass or duck-type `DeterministicScorer.score(case, output)`
  returning an `EvalResult`; pass it to `EvalRunner(scorer=...)`.
- **Agent adapter.** Wrap any framework (LangChain, OpenAI, Haystack) behind a
  `Callable[[str], str]`; replay suites bypass this entirely via `case.output`.
- **Output sinks.** Consume `RunResult.results` directly to emit JSON/Markdown
  reports (**`planned`**: built-in `report`).

## 8. Security considerations

- **S1 — Untrusted YAML.** Suites are loaded with `yaml.safe_load`; no arbitrary
  object construction. Treat eval YAML as code (CODEOWNERS review applies).
- **S2 — Prompt injection via `expected`.** Matcher inputs are plain strings; no
  eval/interp. LLM-as-judge (**`planned`**) must sandbox judge prompts and pin
  `temperature=0` to reduce nondeterminism.
- **S3 — Cost/DoS.** `cost_bound` limits are advisory in v0.1.0 (not yet enforced by
  the runner). Enforcement is **`planned`** with a dry-run cost estimate.
- **S4 — Output handling.** Recorded `output` may contain PII; store replay suites
  outside public repos or sanitize before commit.
- **S5 — Exit-code contract.** CI must rely only on documented exit codes (`0/1/2`);
  never parse stdout for gating.

## 9. Performance requirements

- **P1 — Deterministic scoring is O(output length)** per matcher; suite runtime is
  dominated by `agent_fn` latency.
- **P2 — Replay mode must score a 100-case suite in well under 1s** on CI runners
  (no network). Current golden suite (4 cases) runs in milliseconds.
- **P3 — No global state**; `EvalRunner`/`DeterministicScorer` are safe to construct
  per-run.
- **P4 — Memory** stays flat with suite size (streaming iteration, no full-matrix
  retention beyond `RunResult.results`).

| Feature                | Status      | Est. milestone |
|------------------------|-------------|----------------|
| Deterministic scoring  | ✅ shipped  | v0.1.0         |
| Measurement-validity column (decide + scorer) | ✅ shipped | v0.1.0  |
| Judge decorrelation (HOST != FAMILY) | ✅ shipped | v0.1.0  |
| Judge calibration (confidence is not evidence) | ✅ shipped (protocol only; no live consumer yet) | v0.1.0  |
| OEC ship gate (OEC + guardrails + trust gates) | ✅ shipped (protocol only; no live consumer yet) | v0.1.0  |
| Anti-mode-collapse /diverge (5 candidates + p_conventional) | ✅ shipped (protocol only; no live consumer yet) | v0.1.0  |
| CLI `run` + CI gate    | ✅ shipped  | v0.1.0         |
| Golden replay suites   | ✅ shipped  | v0.1.0         |
| alt-test benchmark     | ✅ shipped  | v0.1.0         |
| Trace-mined benchmarks | ✅ shipped  | v0.2           |
| LLM-as-judge scoring   | `planned`   | v0.2           |
| `compare` regression   | `planned`   | v0.2           |
| `report` (md + json)   | `planned`   | v0.2           |
| `cost_bound` enforced  | `planned`   | v0.2           |
| typer CLI + rich output| `planned`   | v0.3           |
| Custom scorer plugins  | `planned`   | v0.3           |
| Historical dashboard   | `planned`   | v0.3+          |

## 10. Failure modes (PREMORTEM)

1. **Flaky LLM-judge scoring** — inconsistent grades across runs.
   *Mitigation:* deterministic scorers are primary; judge is opt-in with N=3
   majority vote and `temperature=0`.
2. **Cost overrun on large suites** — token burn from judge calls.
   *Mitigation:* `cost_bound` enforced per-row; dry-run estimate; fast CI fail.
3. **YAML schema drift** — mis-scoring from malformed suites.
   *Mitigation:* strict validation on load; reject unknown keys; versioned schema.
4. **Regression false positives** — variance triggers spurious alerts.
   *Mitigation:* configurable threshold (default 5% degradation target for
   `compare`); `skip` flag for known-flaky cases.
5. **Agent-function coupling** — narrow `agent_fn` interface.
   *Mitigation:* `Callable[[str], str]` minimum; richer signatures `planned`;
   documented adapter patterns.

## 11. Acceptance criteria

- [x] Core eval engine loads YAML suites and scores outputs deterministically.
- [x] CLI `run` with `--ci`/`--threshold` gates CI on pass rate.
- [x] Golden replay suite passes in this repo's own CI.
- [x] README accurate and AEO-optimized; spec, roadmap, and repo hygiene mature.
- [ ] LLM-judge integration with cost bounding — *planned (v0.2)*. When the transport lands, it must call `require_calibrated` before judge scores gate anything (see `docs/CALIBRATION.md`).
- [ ] `compare` command for run-vs-run regression — *planned (v0.2)*.
- [ ] `report` generation (markdown + JSON) — *planned (v0.2)*.
