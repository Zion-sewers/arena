# Persona symmetry gate (arena#52)

Source: Bronstein et al., *Geometric Deep Learning* (catalog G.25).
Symmetry → equivariance → stability. Each persona contract states which
transformations it is equivariant/invariant to. Test: permuting
irrelevant inputs must give identical outputs.

## The personas

The four coffee-break personas (`.github/agent-coffee/COFFEE.md`):
builder, critic, guardian, dreamer. A persona's deterministic surface is
its call: the system prompt, the user message, and max_tokens, plus the
model the `jev` router picked. The live model call is not part of the
gate: model outputs are non-deterministic, and the repo's first principle
forbids the core from requiring a model to produce stable results.

## The contracts

`src/agenteval_bench/symmetry.py` holds one `PersonaContract` per persona
in `PERSONA_CONTRACTS` (the single source of truth for the prompts). Every
persona declares the same four symmetries — they are properties of the
shared call construction; the personas differ in their prompts:

| Transformation | Kind | Claim |
|---|---|---|
| `MODEL_SUBSTITUTION` | invariant | The payload (system prompt, user message, max_tokens) is identical whichever model the router picks. "No persona names a model." |
| `WHITESPACE` | invariant | Trailing spaces, leading/trailing blank lines, doubled blank lines in the topic or context items do not change the call. The build path canonicalizes them away. |
| `CONTEXT_ORDER` | invariant | Any permutation of the context items does not change the call. Context items are an explicit sequence of *independent* items; the build path sorts them into a canonical order. |
| `TOPIC_SUBSTITUTION` | equivariant | Replacing the topic replaces exactly its occurrence in the user message; system prompt, max_tokens, and model are unchanged. |

A contract with no declared symmetries is refused at construction, and a
transformation may only be declared in a pair the gate can check
(`MODEL_SUBSTITUTION` / `WHITESPACE` / `CONTEXT_ORDER` as invariant,
`TOPIC_SUBSTITUTION` as equivariant). A declared symmetry the gate cannot
check would be a promise with no enforcement behind it.

## What the gate proves, and what it does not

The gate proves the symmetry of everything the repo controls: the call
construction and the contract machinery. It proves it exactly (byte
equality), not by similarity — "it seemed robust" is not a symmetry
argument.

It does not prove the live model's text output is symmetric. That would
need a model in the loop and determinism the roadmap forbids. A test
that called a live model and asserted byte-equality would be flaky
theater, worse than the boundary.

Two things keep the gate honest instead of vacuous:

- **Negative controls.** The test suite runs the gate against stand-in
  builders that embed the model name, join context unsorted, skip
  canonicalization, or drop the topic. Each FAILs with the violated
  transformation named. A gate that cannot fail is theater.
- **Property tests.** Hypothesis generates topics, context-item lists,
  and models; the symmetries must hold for every generated input, not
  just the sample. Permuting irrelevant inputs must give identical
  outputs — quantified, not anecdoted.

One deliberate omission: pairwise-composition checks. For exact
invariance, invariance under t1 and t2 implies invariance under t1∘t2,
so a composition check cannot fail when the singles pass. Including it
would be a check that cannot fail.

## Scope notes

- `CONTEXT_ORDER` applies to `context_items`, an explicit sequence of
  independent items. The coffee ritual's free-text `TOPIC_BODY` path
  stays free text and claims no order symmetry; `run_coffee.py` is
  untouched by this change.
- `MODEL_SUBSTITUTION` is vacuous for the current prompts (they never
  named a model). Its value is as a regression guard: the negative
  control proves the gate detects a violation if a future prompt does
  name one.

## Sync obligation

The prompts in `PERSONA_CONTRACTS` must stay byte-identical to
`PERSONA_PROMPTS` in `.github/agent-coffee/run_coffee.py`, and
`max_tokens` must match the literal in its `chat()` call. That script
runs standalone in GitHub Actions without the package installed, so it
cannot import this module; `tests/test_symmetry.py` parses its prompts
and fails CI on drift. If a prompt changes, update the registry (single
source of truth), never the test.

## Running it

```sh
agenteval-bench symmetry [--out report.json]   # exit 0 all pass, 1 violation
PYTHONPATH=src python -m agenteval_bench.symmetry_demo
PYTHONPATH=src python -m pytest tests/test_symmetry.py
```
