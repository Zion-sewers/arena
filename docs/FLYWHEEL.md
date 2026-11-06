# Fix-commit flywheel (arena#40)

A meta-agent loop that turns fix-commits in agent repos into proposed
blocking rules, opened as PRs. It never edits the target's code directly.

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
  injected fault breaking a test.
- **PR** (`flywheel.propose.render_pr_payload`): renders a postable PR
  payload (title, body with dry-run evidence, wiring files) from a
  ValidatedRule only. The flywheel never writes into the target tree;
  the fleet runner posts the payload via the GitHub API.

## The leading metric

Every dry-run, promotable or not, is appended to the gate-catch ledger
(JSONL, caller-chosen path). The leading metric is the mean gate-catch
rate across ledgered dry-runs (`flywheel.propose.leading_gate_catch_rate`).
A vacuous dry-run (zero injected faults) scores 0.0, never 100%.

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
- The dry-run corpus is the vendored mcalc fixture, not the target repo's
  own history: the PR body says so, with the injected case list.
- `FixCommit.files` is empty from the GitHub list endpoint; Reflect
  signals use commit messages only. Per-commit file enrichment is a
  follow-up.
- The loop has no human step: `run_loop` takes no callback and never
  reads stdin. The demo runs with stdin closed as the negative control.
- Manual spot-checking is not the loop: spot-checks may happen around the
  loop, but the loop completes on its own or fails loudly.

## Demo

`python -m flywheel.demo` runs the full loop offline on recorded
fix-commits (exit 0 only if the payload is produced, the mock kind is
skipped as already enforced, the target digest is unchanged, and the
gate-catch rate is ledgered). CI runs it as the flywheel step.
