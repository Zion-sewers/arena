# PROMOTION-GATE.md — evidence-gate against benchmark overfitting (arena#51)

Source: Roberts/Yaida, *The Principles of Deep Learning Theory*
(catalog G.24). Implementation: `src/agenteval_bench/promotion.py`
(`VersionSpec`, `VersionEvidence`, `PromotionPolicy`, `promote`).

## The rule in one sentence

A challenger version promotes over the champion only when its Bayesian
evidence improves — log p(D|challenger) − log p(D|champion) clears a
policy margin after accounting for measurement error. A benchmark
accuracy gain never promotes on its own.

## Theory → code mapping

| Theory (Roberts/Yaida) | Code |
|---|---|
| Promotion requires improvement in Bayesian evidence, not accuracy | `promote()` PROMOTEs iff the lower bound of the evidence-gain interval clears `PromotionPolicy.evidence_margin` |
| Microscopic complexity r = L/n (depth over width) resolves the generalization puzzle | `VersionSpec.complexity` = `depth / width`, reported for both versions in every `PromotionVerdict.summary()` |
| Criticality of initialization / hyperparameters | Out of scope for the gate: the gate judges measured evidence, it does not estimate it. The evidence supplier (training/eval pipeline) owns criticality. |
| Accuracy-only promotion is not allowed | Accuracy gain with no evidence gain REJECTs, with the accuracy gain named in the reason; unmeasured evidence HOLDs however high the accuracy |

## What the gate does not do

- **It does not compute evidence.** Log-evidence (log marginal
  likelihood, e.g. Laplace-approximated on a held-out split) is a
  measured input supplied by the caller's pipeline, exactly as
  successes/trials are supplied to `decision.py`. A gate that estimated
  evidence from benchmark scores would be the overfitting it exists to
  prevent.
- **It does not invent a complexity penalty.** The marginal likelihood
  already penalizes complexity (it integrates over parameters — the
  Occam factor, established). No coefficient in the literature maps a
  unit of r to a unit of log-evidence, so the gate reports r as an
  audit trail instead of making one up.
- **It has no live consumer yet.** Like the other protocol-only gates
  in the spec table, it is the decision discipline waiting for a
  release pipeline to call it. Do not claim otherwise.

## Uncertainty handling

- The gain interval is gain ± 1.96 · √(se_challenger² + se_champion²).
  The independent-SE formula is conservative: evidence for two versions
  is usually estimated on the same split, so the errors are correlated,
  and correlation would only narrow the interval. The conservative
  direction is the fail-closed direction for a gate.
- The compared quantity is the interval LOWER bound, mirroring
  `decision.py`: a point estimate that clears the margin while its
  interval straddles it HOLDs.
- Evidence from different splits is incomparable and refused outright
  (`ValueError`); an empty split name is refused (unidentified
  provenance fails closed).
- The margin is a policy choice, default 0.0. Jeffreys scale for the
  log Bayes factor: 1 substantial, 3 strong, 5 decisive.

## Relation to the other gates

- `decision.py` gates a DEPLOYMENT decision on per-dimension evidence
  intervals; this gate decides VERSION promotion on evidence gain.
  A version can pass the promotion gate and still fail `decide` at a
  given risk level — the two gates compose, neither replaces the other.
- `docs/EVAL-RIGOR.md` §1 requires baseline-vs-candidate comparison
  with confidence intervals; the promotion gate's Wilson-interval
  accuracy reporting is consistent with it, and the evidence-gain rule
  is strictly stronger than §1's "report the CI" discipline.
