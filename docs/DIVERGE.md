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
step is theater. /diverge makes the *check* real before any
selection happens: it refuses paraphrase sets and overconfident p
claims. The divergence itself still depends entirely on the caller
supplying token-level-diverse candidates — a caller supplying five
paraphrases gets a refusal, not token-level-diverse candidates.
Generation is not implemented; candidates are caller-supplied.

## The protocol

```
record 5 candidates (with caller-supplied scores)
  -> score the set for token-level diversity (near-duplicate detection, not semantic diversity)
  -> check declared p_conventional against the measured share of the conventional answer's cluster
  -> select on score; the record says why the conventional answer won or lost
  -> high-stakes decisions need a token-level-diverse 5-candidate selection, graded under the canonical policy, with a non-overconfident p
```

### 1. Candidates are recorded, never invented

Arena has no live generation transport (like LLM-as-judge, generation
is planned, not implemented). Candidates arrive as recorded outputs
supplied by the caller, with a score each — e.g. from
`DeterministicScorer`. The module never invents a candidate, a score,
or a p_conventional. A candidate without a score is refused by the
loader rather than defaulted to 0.

### 2. Genuine diversity, not paraphrases — at token level, not semantic

Pair similarity combines two signals, both stdlib-only and auditable
by reading them:

- **Content-token Jaccard**: lowercased word tokens minus stopwords.
  A near-verbatim rewording keeps the content tokens; a different
  answer changes them.
- **Hard-token Jaccard**: digit-bearing tokens (numbers, versions,
  identifiers with digits). It applies only when BOTH sides carry at
  least three hard tokens: Jaccard on tiny hard-token sets has almost
  no gradation, so one shared number — or two — is coincidence, not
  paraphrase evidence. Shared numbers are paraphrase evidence only
  in bulk. Residual, disclosed here: two genuinely different answers
  that share three numbers can still be refused.

Pair similarity is the **max** of the two when the hard-token floor
is met, else the content Jaccard. Max, not mean: either signal can
convict a paraphrase, and averaging would let a reworded text with
the same numbers slip through on a low content score.

A pair above `max_pair_similarity` (default 0.80) is a near-duplicate.
Texts identical after normalization are always maximally similar
(1.0): the comparison lowercases and reads the word-token sequence,
so case, surrounding whitespace and punctuation variants of one
answer ("the" vs "The", "the ", "the.") are the same answer, not five
diverse candidates. Distinct candidates with no content tokens share
no content (similarity 0.0), so they are not convicted as paraphrases
of each other. The set grade is the **weakest link**: one near-verbatim pair fails
the whole set, and `select()` refuses it with typed
`NotDiverseError`. Near-duplicate edges form clusters; the largest
cluster is the modal cluster, and its size over n is the **modal
share** — the measured support for the answer the candidates
converge on.

Known limitation, recorded here rather than hidden: this is
near-duplicate detection at token level, NOT semantic diversity. A
fluent synonym rewrite that replaces the content words ("hash map"
to "hash table", "60s" written out as "one minute") passes as
diverse — token overlap cannot see it, and no stdlib-only signal
reliably can without false-positiving on genuinely different
answers. Do not present a diverse verdict as a semantic-diversity
claim: it means no near-verbatim pair, nothing more. The check
convicts rewording that preserves most content tokens; it does not
do semantic similarity.

### 3. p_conventional is a claim, checked against measurement

The caller declares which candidate is conventional
(`conventional_index`: the answer the generator would have returned on
its own) and `p_conventional`: their estimate that the conventional
answer is the answer the divergence CONVERGES on (the modal answer).
That definition is deliberate: the candidate set measures convergence,
not correctness, so only a convergence claim can be checked against
it. A p of 0.85 for a conventional answer nobody's candidates
converge on is overconfident even if the answer happens to be right —
the claim is about what the divergence shows, and the divergence
does not show it. The module checks the declaration against the
measured share of the conventional answer's OWN cluster through the
Wilson 95% interval — the same discipline as `validity.py` /
`calibration.py`: grade on the interval, never the point estimate.

- `p` above the interval UPPER bound + tolerance (default 0.15) is
  **overconfident** and blocks the high-stakes gate.
- `p` below the LOWER bound − tolerance is **underconfident**.
- Otherwise **consistent**. With n=5 the interval is wide, so weak
  evidence yields consistent, never a false conviction.

The interval is built on the conventional cluster, not the modal
cluster: when the conventional answer is a minority, the modal share
belongs to a rival answer, and validating confidence in the
conventional answer against a rival's support would license exactly
the overconfidence the check exists to catch. Three candidates
converging on a rival while the conventional answer stands alone
with a declared p of 0.95 is overconfident, full stop.

When the conventional cluster is large enough that the interval
upper bound + tolerance reaches 1.00 (a share of 0.60 at n=5), no
declared p can be refuted, and the check says so in its notes
instead of pretending to have tested the claim.

A conventional candidate outside the modal cluster is a recorded
finding: the candidates converge on a different answer than the one
declared conventional. It is named in the p-check notes and in the
selection reason, never smoothed over.

The pipeline is fingerprint-linked: every diversity report, p-check
and selection carries the SHA-256 fingerprint of the candidate set
it was computed from (numbers coerced to float, so int and float
spellings of the same set fingerprint alike), and a step refuses a
measurement whose fingerprint does not match the set in hand. Every
record validates its own invariants at construction, recomputed from
the fields it carries: the report re-derives its near-duplicate
flags, clusters, modal cluster, diversities and per-candidate means
from the pairs; the p-check re-derives its interval and verdict from
the conventional share, n and tolerance, and its modal and
conventional shares from the clusters it carries; the selection
re-derives its margin and outcome from the scores it carries, and
refuses a winner that is not the argmax. Each record carries what its
own checks need, and `select()` cross-checks the records against each
other before composing them — fingerprints, thresholds, clusters,
declared p, conventional index and n — so two measurements that
contradict each other, or either measurement against the set it
claims to cover, never compose into a valid-looking selection.
The gate itself reads only the Selection record: it enforces the
count (n == 5) and the canonical policy (0.80 / 0.15) from the
carried fields, and trusts `diverse` and `p_verdict` as echoes of
the cross-validated records — stamped with the canonical policy, and
bounded by the determined-forger concession below. The tie-break
means are carried by the report, not the selection, so the tie-break
choice among tied top scorers is trusted, not re-derived. Two limits
remain, stated plainly: a record that satisfies every invariant is
indistinguishable from a pipeline-built one — the validations catch
inconsistent records, not a determined forger who builds a consistent
one under the canonical policy — and the tie-break choice among tied
top scorers is trusted, not re-derived, since the record does not
carry the per-candidate means the tie-break reads.

### 4. Selection records why the conventional answer won or lost

`select()` picks the argmax score. Ties break toward the candidate
with the LOWEST mean pairwise similarity — the answer least like the
others, the most distinctive — then the lowest index, so the choice
is deterministic with no RNG. Mean similarity is a measured quantity
that varies inside a diverse set (unlike cluster size, which is 1
for every candidate once the set is diverse), so the tie-break is a
real mechanism. The `Selection` record carries the winner, the
conventional candidate, both scores, the runner-up score, the margin
(winner over runner-up, the same quantity in every outcome) and a
reason string built from the measured facts. When no two candidates
converge, the reason says so instead of naming an arbitrary modal
cluster as a finding. When the conventional candidate ties at the
top and wins the tie-break, the outcome is `tie`, not a win on score
— the reason names the tie-break, since a margin of +0.00 is not a
win on score. The record IS the explanation; there is no
prose beside it to drift from it.

### 5. Single-sample generation is not sufficient for high stakes

- `build_diverge_set(..., stakes="high")` with fewer than 5
  candidates raises typed `SingleSampleError`, and a high-stakes
  policy must ask for exactly 5 — a policy that shrinks the count
  would let a 2-candidate set through the gate on the library seam.
- `may_decide_high_stakes(selection)` is True only for a token-level-diverse
  selection covering exactly 5 candidates, graded under the canonical
  policy (max_pair_similarity 0.80, p tolerance 0.15), with a
  non-overconfident p; `require_high_stakes_decision` raises typed
  `HighStakesRefused` otherwise. The gate enforces the count and the
  policy itself from the selection record, so a short set built
  low-stakes cannot decide high stakes — and neither can a set graded
  under a laxer policy, which selects fine but is refused at the gate.
  The gate takes ONLY the selection record: a bare candidate, a
  single sample, or a score has no path through it.

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

Refusal examples, so a stranger can see the gate bite without
reading tests:

```bash
agenteval-bench diverge --candidates examples/diverge-paraphrase.json
# exit 2: a near-verbatim pair fails the diversity bar
agenteval-bench diverge --candidates examples/diverge-overconfident.json
# exit 2: p 0.95 is overconfident against the conventional share
```

The human output prints the Wilson interval plus one
plain-language sentence translating the verdict, so the interval
math is never the only explanation on screen.

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
- Diversity is token-level, stdlib-only. It convicts near-verbatim
  rewording; it does not do semantic similarity. A fluent synonym
  rewrite passes as diverse, and the similarity is English-only
  (the stopword list is English function words and the token pattern
  is ASCII, so non-English candidates get their glue counted as
  content). Do not present a diverse verdict as a semantic-diversity
  claim.
- The p-check convicts only a p the measurement actually refutes
  (outside the Wilson interval ± tolerance on the conventional
  cluster's share). A consistent verdict is not a claim the p is
  right, and at a conventional share of 0.60 or above the check
  cannot refute any p at n=5 — the notes say so when that is the case.
  The p is a convergence claim, not a correctness claim: it is never
  validated as a probability that the conventional answer is correct.
