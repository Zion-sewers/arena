# Fix-commit flywheel (arena#40)

A meta-agent loop that turns fix-commits in agent repos into proposed
blocking rules, rendered as PR payloads. It never edits the target's
code directly. Posting the payloads via the GitHub API is the fleet
runner's job — a follow-up, not shipped here: the CLI writes payload
files out and says so.

## Running it

`agenteval-bench flywheel --repo owner/name --target-root .` runs the
loop against live history (the `gh` CLI transport) with the local
runner. Payloads land under `<target>/.flywheel/payloads/<rule-kind>/`
(`PR.md` plus the wiring files); the gate-catch ledger defaults to
`<target>/.flywheel/gate-catch-ledger.jsonl`, a durable path, so the
leading metric persists across runs.

## The loop

Detect -> Sample -> Reflect -> Propose -> Validate -> PR

- **Detect** (`flywheel.detect`): fix-commits are read through an injected
  transport. The live transport shells out to the fleet's `gh` CLI
  (authentication lives there, not in this repo); tests and the demo use
  hand-written stub transports carrying recorded payloads.
- **Sample** (`flywheel.detect.sample_commits`): seeded sample, sorted by
  sha first so the sample does not depend on listing order.
- **Reflect** (`flywheel.reflect`): deterministic aggregation over a closed
  rule-kind registry. A kind is proposed when at least two sampled
  fix-commits signal it. A kind already enforced in the target is skipped
  with the enforcer named.
- **Propose** (`flywheel.types.DraftRule`): an unvalidated proposal. It
  must never reach a PR payload in this form.
- **Validate** (`flywheel.validate.dry_run`): the dry-run gate. The rule
  predicate runs over the clean target tree and the validation corpus
  (false positives refuse promotion), and deterministic faults of the
  kind's class are injected into throwaway copies of the vendored mcalc
  fixture and validated as real bugs by the bugsmith harness
  (fail_to_pass evidence from real suite runs). Promotion requires at
  least one injected fault, every injected fault caught, zero false
  positives, and — for kinds whose faults must be behavioral — every
  injected fault breaking a test. For stylistic kinds the installed
  enforcer test is additionally run on the injected tree and must fail
  there. Reports are consistency-checked at construction: the numbers
  must be derived from the listed cases, so no hand-built report can
  authorize a payload with evidence it does not carry.
- **PR** (`flywheel.propose.render_pr_payload`): renders a postable PR
  payload (title, body with dry-run evidence, wiring files) from a
  ValidatedRule only. The flywheel never writes into the target tree;
  the CLI writes the payload out under `.flywheel/payloads/` and the
  fleet runner posts it via the GitHub API (follow-up).

## The leading metric

Every dry-run, promotable or not, is appended to the gate-catch ledger
(JSONL; the CLI defaults it to `<target>/.flywheel/gate-catch-ledger.jsonl`).
The leading metric is the pooled gate-catch rate across ledgered
dry-runs (`flywheel.propose.leading_gate_catch_rate`: total caught /
total injected), so a 1-case run does not count the same as a 2-case
run. A vacuous dry-run (zero injected faults) scores 0.0, never 100%.
Ledger entries are validated on read (counts, ranges, and
catch_rate == caught/injected consistency); a corrupt line fails loudly
rather than bending the metric.

## Rule kinds (closed registry)

| kind | what it forbids | live consumer when promoted |
| --- | --- | --- |
| `no-bare-except` | bare `except:` swallowing BaseException | ruff `E722` select in pyproject.toml (CI runs `ruff check`) |
| `no-mock-on-boundary` | `unittest.mock` in tests patching production code | `tests/test_no_mocks_on_boundaries.py` (CI runs it) |

The registry is closed on purpose: free-form rule synthesis needs a model
transport this repo does not have, and this module does not simulate one.

## Scope and honesty

- Dogfood target: arena itself is target #1; the target repo is a
  parameter, so other agent repos are the same loop with a different
  target, not a rewrite. Multi-repo rollout is a follow-up.
- The registry is closed at two kinds, and it exhausts: once a kind's
  rule lands in a target, the loop never proposes that kind there again.
  On arena today only `no-bare-except` can fire (`no-mock-on-boundary`
  is already enforced). A loop that proposes one rule and then goes
  quiet is the honest current shape; new kinds, and free-form rule
  synthesis, are follow-ups, not claimed here.
- The dry-run corpus is the vendored mcalc fixture, not the target repo's
  own history: the PR body says so, with the injected case list.
- `FixCommit.files` is empty from the GitHub list endpoint; Reflect
  signals use commit messages only. Per-commit file enrichment is a
  follow-up.
- Detect reads a single page of the newest 100 commits; older history
  is invisible to it until pagination lands (follow-up).
- The bare-except wiring selects `E722` only, never a group (`E`,
  `F`): the dry-run validates the bare-except predicate and nothing
  else, so the deployed check must not turn on unvalidated rules.
  Wiring extends an existing `[tool.ruff.lint]` select with `E722`
  when it is a plain list; any other shape is refused loudly.
- The mock predicate and the enforcer test its wiring installs are the
  same detector, same file set (top-level `tests/*.py`): if they drift,
  probe tests fail. For stylistic kinds the dry-run additionally runs
  the installed enforcer test on the injected tree and requires it to
  fail — the thing validated is the thing the PR installs.
- A wiring failure on one draft does not kill the loop: the draft is
  recorded skipped and the other drafts still render.
- The loop has no human step: `run_loop` takes no callback and never
  reads stdin. The demo runs with stdin closed as the negative control.
- Manual spot-checking is not the loop: spot-checks may happen around
  the loop, but the loop completes on its own or fails loudly.

## Demo

`python -m flywheel.demo` runs the full loop offline on recorded
fix-commits (exit 0 only if the payload is produced, the mock kind is
skipped as already enforced, the target digest is unchanged, and the
gate-catch rate is ledgered). CI runs it as the flywheel step.
