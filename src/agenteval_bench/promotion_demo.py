"""Seeded end-to-end demo of the version promotion gate (arena#51).

Runs the evidence-gain promotion rule offline on hand-built version
evidence. Exit 0 only if every acceptance expectation holds:
- an evidence gain that clears the margin PROMOTEs;
- an accuracy gain with an evidence loss REJECTs (benchmark overfitting
  does not promote, however large the accuracy gain);
- an evidence-gain interval that straddles the margin HOLDs;
- unmeasured evidence HOLDs even at 100% benchmark accuracy;
- complexity is reported as r = L/n for both versions.

Any failure raises, so CI can gate on this demo directly.
"""

from __future__ import annotations

import sys

from agenteval_bench.promotion import (
    PromotionAction,
    PromotionPolicy,
    VersionEvidence,
    VersionSpec,
    promote,
)


def _version(
    name: str,
    depth: int,
    width: int,
    log_evidence: float | None,
    log_evidence_se: float | None,
    successes: int,
    trials: int,
    split: str = "heldout-v3",
) -> VersionEvidence:
    return VersionEvidence(
        version=VersionSpec(name=name, depth=depth, width=width),
        log_evidence=log_evidence,
        log_evidence_se=log_evidence_se,
        accuracy_successes=successes,
        accuracy_trials=trials,
        evidence_split=split,
    )


def main() -> int:
    # 1. Evidence gain clears the margin -> PROMOTE. Accuracy is up too,
    # but the verdict rides on the evidence gain (+5.0 nats, SE 0.5 each).
    champion = _version("v1", 12, 768, 0.0, 0.5, 80, 100)
    candidate = _version("v2", 12, 768, 5.0, 0.5, 88, 100)
    verdict = promote(champion, candidate)
    print(verdict.summary())
    assert verdict.action is PromotionAction.PROMOTE, verdict.summary()
    assert verdict.evidence_gain == 5.0

    # 2. Accuracy up (+0.15), evidence down (-4.0) -> REJECT. This is the
    # overfitting case the gate exists for: the benchmark gain is named
    # in the reason, and it does not promote.
    overfit = _version("v2-overfit", 24, 768, -4.0, 0.5, 95, 100)
    verdict = promote(champion, overfit)
    print(verdict.summary())
    assert verdict.action is PromotionAction.REJECT, verdict.summary()
    assert abs(verdict.accuracy_gain - 0.15) < 1e-9
    assert "accuracy gain +0.150 does not promote" in verdict.reason

    # 3. Evidence gain interval straddles the margin -> HOLD.
    # Gain +1.0 with SE 0.5 each: interval [-0.39, +2.39] straddles 0.
    noisy = _version("v2-noisy", 12, 768, 1.0, 0.5, 85, 100)
    verdict = promote(champion, noisy)
    print(verdict.summary())
    assert verdict.action is PromotionAction.HOLD, verdict.summary()

    # 4. No evidence measured, 100% benchmark accuracy -> HOLD, never PROMOTE.
    unmeasured = _version("v2-blind", 12, 768, None, None, 100, 100)
    verdict = promote(champion, unmeasured)
    print(verdict.summary())
    assert verdict.action is PromotionAction.HOLD, verdict.summary()
    assert verdict.evidence_gain is None
    assert "accuracy-only promotion is not allowed" in verdict.reason

    # 5. Complexity is reported as r = L/n: champion 12/768, candidate
    # 24/768. The verdict summary carries both.
    assert champion.version.complexity == 12 / 768
    assert overfit.version.complexity == 24 / 768
    summary = promote(champion, overfit).summary()
    assert "champion 0.0156 (L=12, n=768)" in summary, summary
    assert "candidate 0.0312 (L=24, n=768)" in summary, summary

    # 6. A policy margin is respected: the same +5.0 gain that PROMOTEs
    # at margin 0 HOLDs at margin 6 (lower bound 3.61 < 6).
    strict = PromotionPolicy(evidence_margin=6.0)
    verdict = promote(champion, candidate, strict)
    print(verdict.summary())
    assert verdict.action is PromotionAction.HOLD, verdict.summary()

    print("promotion demo: all 6 expectations hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
