# Tiered eval pipeline

The standard eval runner for the agent company (arena#39). Four phases,
executed in order; each phase runs only when every earlier phase passed.

```text
gate -> smoke -> expanded -> nightly
```

Run it:

```bash
uv pip install -e ".[dev]"   # hypothesis is a dev-only dependency
agenteval-bench tiered --phase <gate|smoke|expanded|nightly|all> \
    [--mutants 8] [--min-mutation-score 0.5] [--min-coverage 80] \
    [--seed 42] [--root .] [--out tiered-report.json]
```

`--phase X` runs X and every earlier phase. Exit 0 = all run phases
passed, exit 2 = a phase failed, exit 1 = usage error. `--out` writes
the JSON report (per-phase status, commands run, coverage, mutation
score, and the requested phases — so a partial run's `PASS` is never
mistaken for a full-pipeline `PASS`). Unknown flags are a usage error,
not silently ignored.

## Phases

| Phase | What runs | Entry criterion | Exit criterion |
|---|---|---|---|
| gate | Golden replay suite via `run --ci --threshold 0.9`; fresh replay log verified bit-exact via `replay --check` | First phase | Pass rate >= 90% AND replay bit-exact |
| smoke | Core pytest subset: `test_engine.py`, `test_cli.py`, `test_replay.py` | Gate passed | Core subset green |
| expanded | Full pytest suite with coverage floor; includes the mock-rejection scan and the property tests | Gate + smoke passed | Suite green AND coverage of `src/agenteval_bench` >= floor |
| nightly | Mutation scoring (bugsmith procedural generator + LocalRunner, seeded) | Gate + smoke + expanded passed | Mutation score computed AND >= floor |

The runner never writes into the repo tree: replay logs, coverage data
and bugsmith work dirs live under a temporary directory. The only
caller-chosen write is `--out`.

## Coverage is not quality evidence on its own

The coverage floor is pinned at the **measured** 80% of
`src/agenteval_bench` (2026-11-06), not an aspirational number. Raise it
deliberately when coverage genuinely rises. On a full nightly run,
coverage is reported **alongside** the mutation score in the same
report; a coverage percentage alone is never accepted as quality
evidence. Partial runs (`--phase gate/smoke/expanded`, i.e. every PR
CI run) carry `mutation_score: null` — coverage paired with nothing,
not with a fabricated score.

## Mutation score

Nightly injects seeded AST mutants (bugsmith's procedural generator:
binop swaps, comparison flips, boolop flips, off-by-one, constant
negation) into throwaway copies of the repo and runs the repo's own suite
against each mutant via `LocalRunner`. Score = killed / validated, where
a mutant is killed when at least one test fails on it. Unvalidatable
mutants (patch does not apply, infrastructure failure) are counted as
skipped, never as killed or survived, and are flagged `skipped: true`
in the report details so they cannot be misread as survivors.

Scope note: the two numbers describe different scopes. Coverage is
measured on `src/agenteval_bench` only; the mutation generator draws
from the evaluated packages (`src/agenteval_bench`,
`src/novelty_oracle`, `src/alt_test`, `src/experiment`) and excludes
`src/bugsmith` — the scorer must not score itself, or a mutant in the
scoring path could fabricate the score. Pairing the two as one quality
story is valid only with that scope in mind.

The score floor is 0.5, pre-registered before the first measurement
as the minimum acceptable kill rate — it is a policy minimum, not the
measured value. The measured value on the merged tree (seed 42,
2026-11-06) is 8/8 = 100%; see below. A score that cannot be
computed — red clean-tree baseline, zero validated mutants — fails the
nightly phase outright rather than reporting 0.0 or 1.0.

The first measurement did not pass at once, and the floor was not
lowered to make it pass: the initial run scored 3/8 = 38% and failed
nightly, honestly. Every survivor was a fair kill (unpinned constants,
demo-only code), so the suite was strengthened instead: targeted tests
pinned battery counts, placeholder zeros, dusted-mechanism amounts
(extracted to a testable helper), demo stability under pytest, and
decision-log rounding. Re-measurement scored 7/8 = 88%; the last
survivor was the `a == 1` winner check in `_noisy_copy`'s payment dust,
killed by a test asserting nonzero noise lands on the winner's payment
(exactly +0.05 with noise_std 0.1 on values (1.0, 0.5, 0.25)).
Final re-measurement: 8/8 = 100%, 298 tests green, coverage 82.5%.

The scope then changed, so the draw changed: excluding `src/bugsmith`
(the scorer must not score itself) redrew the 8 mutants at the same
seed, and the new draw first scored 2/8 = 25% and failed nightly,
honestly. Its six survivors were again fair kills — negated default
seeds and constants (`AltTestConfig.seed`, `make_dataset` seed,
`RubricCriterion.weight`, the skipped-case score, the demo compare,
the CLI exit code) — so the suite was strengthened with one pinning
test per survivor instead of lowering the floor. Final re-measurement
on the rescoped draw: 8/8 = 100%, 309 tests green, coverage 82.6%.

The numbers above are a snapshot of the merged tree on 2026-11-06
(309 tests, coverage 82.6%, mutation 8/8 = 100%, seed 42). They will
go stale as the tree changes: the seeded generator's draw changes
when files are added, so the same seed will not reproduce the same 8
mutants later. The enforced invariant is the floor (0.5), not the
snapshot; the nightly workflow re-measures on schedule (daily 02:00
UTC) and uploads the fresh report as an artifact, so the current
number always lives there, not in this file.

Mutation scoring is slow by construction (one full suite run per
mutant, ~70 s each here), so it lives only in nightly, never in a PR
gate. Nightly scores the working tree as-is: the gate's entry
criterion is "first phase", not a clean tree, so a dirty tree scores
uncommitted code — that is the intended dev use (score your changes
before committing), and CI always runs on a clean checkout.

## Mocks

Mocks on business logic, the database, files, or external services are
rejected. Enforcement is the test
`tests/test_no_mocks_on_boundaries.py`: no `unittest.mock` / `mock`
import, no `mock.*` attribute use, no `mocker` fixture anywhere in
`tests/`. The repo's test style is recorded replay outputs and
hand-written stub functions; if a future test genuinely needs a double,
write a stub or record data instead of importing mock machinery.

## Copy-runnable suite

The nightly mutation run executes the suite inside throwaway copies
that carry no `.git` directory (bugsmith's harness copies the tree
without it). Tests that need git therefore skip when they are not
inside a git work tree (`tests/test_gitignore_coverage.py`); in any real
checkout they run in full. Any new test that shells out to git must do
the same.

## CI wiring

- PR CI (`.github/workflows/ci.yml`): a single `tiered --phase expanded`
  step runs gate, then smoke, then expanded (prefix semantics — one
  step subsumes the old eval-gate step and the old raw pytest step, so
  no step runs twice). It writes the JSON report (`--out
  tiered-expanded-report.json`, uploaded as an artifact even on
  failure); the report carries `mutation_score: null` because mutation
  scoring is nightly-only.
- Nightly (`.github/workflows/tiered-nightly.yml`, daily 02:00 UTC):
  `tiered --phase nightly --mutants 8`, which runs all four phases and
  uploads the JSON report as an artifact (`if: always()`, so a failed
  nightly still leaves its evidence).
