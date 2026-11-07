# Anti-mode-collapse: /diverge with 5 candidates + p_conventional

Source: arena#47 (claude-setup rule out-of-distribution.md, catalog A.6,
INSPIRED). Deliberate divergence from the conventional answer: 5 candidates
plus an explicit p_conventional estimate. Fights mode collapse in agent
generation.

## The problem

A generator asked once returns the mode: the single most likely answer.
Asked five times at temperature it often returns the same answer five
times in different words, and a selector then "chooses" between
paraphrases. The choice was already made by the mode; the selection
step is theater. /diverge makes the divergence real before any
selection happens.

## The protocol

```
record 5 candidates (with caller-supplied scores)
  -> score the set for genuine diversity (not paraphrases)
  -> check declared p_conventional against the measured modal share
  -> select on score; the record says why the conventional answer won or lost
  -> high-stakes decisions need a diverse selection with a non-overconfident p
```

### 1. Candidates are recorded, never invented

Arena has no live generation transport (like LLM-as-judge, generation
is planned, not implemented). Candidates arrive as recorded outputs
supplied by the caller, with a score each — e.g. from
`DeterministicScorer`. The module never invents a candidate, a score,
or a p_conventional. A candidate without a score is refused by the
loader rather than defaulted to 0.

### 2. Genuine diversity, not paraphrases

Pair similarity combines two signals, both stdlib-only and auditable
by reading them:

- **Content-token Jaccard**: lowercased word tokens minus stopwords.
  A paraphrase rewords the glue; the content tokens are what must
  differ.
- **Hard-token Jaccard**: digit-bearing tokens (numbers, versions,
  identifiers with digits). A paraphrase preserves the numbers and
  names while rewording the glue around them; a genuinely different
  answer changes them.

Pair similarity is the **max** of the two when both sides carry hard
tokens, else the content Jaccard. Max, not mean: either signal can
convict a paraphrase, and averaging would let a reworded text with the
same numbers slip through on a low content score.

A pair above `max_pair_similarity` (default 0.80) is a near-duplicate.
The set grade is the **weakest link**: one paraphrase pair fails the
whole set, and `select()` refuses it with typed `NotDiverseError`.
Near-duplicate edges form clusters; the largest cluster is the modal
cluster, and its size over n is the **modal share** — the measured
support for the answer the candidates converge on.

Known limitation, recorded here rather than hidden: a semantic
paraphrase that also changes every number and name is undetectable
stdlib-only. But that changes the answer, not just the wording, and
the hard-token anchor catches the common case (same figures, new
glue).

### 3. p_conventional is a claim, checked against measurement

The caller declares which candidate is conventional
(`conventional_index`: the answer the generator would have returned on
its own) and `p_conventional`: their estimate that the conventional
answer is correct. The module checks the declaration against the
measured modal share through the Wilson 95% interval — the same
discipline as `validity.py` / `calibration.py`: grade on the interval,
never the point estimate.

- `p` above the interval UPPER bound + tolerance (default 0.15) is
  **overconfident** and blocks the high-stakes gate.
- `p` below the LOWER bound − tolerance is **underconfident**.
- Otherwise **consistent**. With n=5 the interval is wide, so weak
  evidence yields consistent, never a false conviction.

A conventional candidate outside the modal cluster is a recorded
finding: the candidates converge on a different answer than the one
declared conventional. It is named in the p-check notes and in the
selection reason, never smoothed over.

### 4. Selection records why the conventional answer won or lost

`select()` picks the argmax score. Ties break toward the candidate in
the **smallest cluster** — the most distinctive answer wins a tie,
the deliberate anti-mode-collapse tie-break — then the lowest index,
so the choice is deterministic with no RNG. The `Selection` record
carries the winner, the conventional candidate, both scores, the
margin, a closed outcome (`conventional_won` / `conventional_lost` /
`tie`) and a reason string built from the measured facts. The record
IS the explanation; there is no prose beside it to drift from it.

### 5. Single-sample generation is not sufficient for high stakes

- `build_diverge_set(..., stakes="high")` with fewer than 5
  candidates raises typed `SingleSampleError`.
- `may_decide_high_stakes(selection)` is True only for a diverse
  selection with a non-overconfident p; `require_high_stakes_decision`
  raises typed `HighStakesRefused` otherwise. The gate takes ONLY the
  selection record: a bare candidate, a single sample, or a score has
  no path through it.

## CLI

```bash
agenteval-bench diverge --candidates examples/diverge-candidates.json
# exit 0 selection made, 2 refused, 1 bad input
```

Input shape (`examples/diverge-candidates.json`): a mapping with
`prompt`, `candidates` (list of `{"text", "score", "label"?}`),
`conventional_index`, `p_conventional`, optional `stakes`
(`"low"` | `"high"`, default `"low"`). The loader fails closed on
unknown keys, missing fields, wrong types and non-finite numbers.

## Honesty and coverage boundaries

- **Protocol only, no live consumer yet.** The gate seam
  (`require_high_stakes_decision`) and the CLI are the consumers.
  When a live generation transport lands, it must call
  `build_diverge_set` with `stakes="high"` before a high-stakes choice
  is decided from a single sample — the same shape as
  `require_calibrated` for judge scores (see `docs/CALIBRATION.md`).
- Scores are caller-supplied measurements. The module records and
  quotes them; it never vouches for them. A gamed scorer is a scorer
  problem, and the selection record makes the scores auditable.
- Diversity is token-level, stdlib-only. It convicts rewording that
  preserves numbers and names; it does not do semantic similarity.
  Do not present a diverse verdict as a semantic-diversity claim.
- The p-check convicts only a p the measurement actually refutes
  (outside the Wilson interval ± tolerance). A consistent verdict is
  not a claim the p is right.
