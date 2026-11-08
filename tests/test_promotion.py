"""Tests for the version promotion gate (arena#51).

Evidence gain, not accuracy gain. Every behavior change ships with an
adversarial test: the tests below pin the failure modes the design must
refuse, not just the happy paths. No mocks anywhere (repo-wide ban).
"""

from __future__ import annotations

import math

import pytest

from agenteval_bench.promotion import (
    PromotionAction,
    PromotionPolicy,
    PromotionVerdict,
    VersionEvidence,
    VersionSpec,
    promote,
)

SPLIT = "heldout-v3"


# ---------------------------------------------------------------- helpers


def _version(
    name: str = "v2",
    depth: int = 12,
    width: int = 768,
    log_evidence: float | None = 5.0,
    log_evidence_se: float | None = 0.5,
    successes: int = 88,
    trials: int = 100,
    split: str = SPLIT,
) -> VersionEvidence:
    return VersionEvidence(
        version=VersionSpec(name=name, depth=depth, width=width),
        log_evidence=log_evidence,
        log_evidence_se=log_evidence_se,
        accuracy_successes=successes,
        accuracy_trials=trials,
        evidence_split=split,
    )


def _champion() -> VersionEvidence:
    return _version(name="v1", log_evidence=0.0, log_evidence_se=0.5, successes=80, trials=100)


# ---------------------------------------------------------------- VersionSpec


def test_complexity_is_depth_over_width() -> None:
    assert VersionSpec("v1", 12, 768).complexity == 12 / 768
    assert VersionSpec("v2", 24, 768).complexity == 24 / 768
    assert VersionSpec("v3", 1, 1).complexity == 1.0


@pytest.mark.parametrize("depth,width", [(0, 768), (-1, 768), (12, 0), (12, -8)])
def test_bad_dimensions_rejected(depth: int, width: int) -> None:
    with pytest.raises(ValueError):
        VersionSpec("v", depth, width)


@pytest.mark.parametrize("depth,width", [(True, 768), (12, False), (12.0, 768)])
def test_bool_or_float_dimensions_rejected(depth: object, width: object) -> None:
    with pytest.raises(TypeError):
        VersionSpec("v", depth, width)  # type: ignore[arg-type]


def test_empty_version_name_rejected() -> None:
    with pytest.raises(ValueError):
        VersionSpec("", 12, 768)


def test_whitespace_only_name_or_split_rejected() -> None:
    with pytest.raises(ValueError):
        VersionSpec(" ", 12, 768)
    with pytest.raises(ValueError):
        _version(split=" ")


# ---------------------------------------------------------------- VersionEvidence


def test_evidence_and_se_must_come_together() -> None:
    with pytest.raises(ValueError):
        _version(log_evidence=5.0, log_evidence_se=None)
    with pytest.raises(ValueError):
        _version(log_evidence=None, log_evidence_se=0.5)


def test_negative_se_rejected() -> None:
    with pytest.raises(ValueError):
        _version(log_evidence_se=-0.1)


@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_nonfinite_evidence_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        _version(log_evidence=bad)


def test_empty_split_rejected() -> None:
    with pytest.raises(ValueError):
        _version(split="")


@pytest.mark.parametrize("successes,trials", [(0, 0), (101, 100), (-1, 100), (5, -3)])
def test_bad_accuracy_counts_rejected(successes: int, trials: int) -> None:
    with pytest.raises(ValueError):
        _version(successes=successes, trials=trials)


@pytest.mark.parametrize("successes,trials", [(True, 100), (80.5, 100), (80, 100.0), (80, True)])
def test_bool_or_float_accuracy_counts_rejected(successes: object, trials: object) -> None:
    # A True success count would silently become 1 success and corrupt
    # the reported accuracy and accuracy_gain in every verdict.
    with pytest.raises(TypeError):
        _version(successes=successes, trials=trials)  # type: ignore[arg-type]


def test_accuracy_interval_is_wilson() -> None:
    evidence = _version(successes=95, trials=100)
    lo, hi = evidence.accuracy_interval
    assert evidence.accuracy == 0.95
    # Exact Wilson interval for 95/100 at z=1.96 (hardcoded literals,
    # not recomputed from the implementation under test).
    assert lo == pytest.approx(0.888248, abs=1e-6)
    assert hi == pytest.approx(0.978457, abs=1e-6)


def test_unmeasured_flag() -> None:
    assert _version().evidence_measured is True
    assert _version(log_evidence=None, log_evidence_se=None).evidence_measured is False


# ---------------------------------------------------------------- PromotionPolicy


def test_negative_margin_rejected() -> None:
    with pytest.raises(ValueError):
        PromotionPolicy(evidence_margin=-1.0)


@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_nonfinite_margin_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        PromotionPolicy(evidence_margin=bad)


# ---------------------------------------------------------------- promote: PROMOTE


def test_evidence_gain_clears_margin_promotes() -> None:
    verdict = promote(_champion(), _version())
    assert verdict.action is PromotionAction.PROMOTE
    assert verdict.evidence_gain == 5.0
    assert verdict.gain_interval is not None
    lo, hi = verdict.gain_interval
    # SE_diff = sqrt(0.5^2 + 0.5^2); interval = 5.0 +/- 1.96 * SE_diff.
    assert lo == pytest.approx(5.0 - 1.96 * math.sqrt(0.5))
    assert hi == pytest.approx(5.0 + 1.96 * math.sqrt(0.5))
    assert lo > 0.0
    assert verdict.accuracy_gain == pytest.approx(0.08)


def test_zero_se_exact_evidence_promotes() -> None:
    champion = _version(name="v1", log_evidence=0.0, log_evidence_se=0.0)
    candidate = _version(log_evidence=2.0, log_evidence_se=0.0)
    verdict = promote(champion, candidate)
    assert verdict.action is PromotionAction.PROMOTE
    assert verdict.gain_interval == (2.0, 2.0)


def test_accuracy_loss_does_not_veto_evidence_gain() -> None:
    # The gate is about evidence: a candidate that is WORSE on the
    # benchmark but BETTER in evidence still promotes. Accuracy is
    # reported, never decisive in either direction.
    candidate = _version(successes=70, trials=100)
    verdict = promote(_champion(), candidate)
    assert verdict.action is PromotionAction.PROMOTE
    assert verdict.accuracy_gain == pytest.approx(-0.10)


# ---------------------------------------------------------------- promote: REJECT


def test_accuracy_gain_with_evidence_loss_rejects() -> None:
    # The overfitting case: +0.15 accuracy, -4.0 evidence. An
    # accuracy-only rule would promote this; the evidence gate must not.
    candidate = _version(log_evidence=-4.0, log_evidence_se=0.5, successes=95)
    verdict = promote(_champion(), candidate)
    assert verdict.action is PromotionAction.REJECT
    assert verdict.accuracy_gain == pytest.approx(0.15)
    assert verdict.evidence_gain == pytest.approx(-4.0)
    assert "accuracy gain +0.150 does not promote" in verdict.reason


def test_accuracy_gain_with_zero_evidence_gain_rejects() -> None:
    champion = _version(name="v1", log_evidence=0.0, log_evidence_se=0.0, successes=80)
    candidate = _version(log_evidence=0.0, log_evidence_se=0.0, successes=99)
    verdict = promote(champion, candidate)
    assert verdict.action is PromotionAction.REJECT
    assert verdict.accuracy_gain == pytest.approx(0.19)


def test_margin_is_respected_on_reject() -> None:
    # Gain +2.0 (exact, SE 0) against margin 3.0: upper bound below margin.
    candidate = _version(log_evidence=2.0, log_evidence_se=0.0)
    verdict = promote(
        _version(name="v1", log_evidence=0.0, log_evidence_se=0.0),
        candidate,
        PromotionPolicy(evidence_margin=3.0),
    )
    assert verdict.action is PromotionAction.REJECT


# ---------------------------------------------------------------- promote: HOLD


def test_straddling_gain_interval_holds() -> None:
    # Gain +1.0, SE 0.5 each: interval [-0.39, +2.39] straddles margin 0.
    candidate = _version(log_evidence=1.0, log_evidence_se=0.5, successes=85)
    verdict = promote(_champion(), candidate)
    assert verdict.action is PromotionAction.HOLD
    assert verdict.gain_interval is not None
    assert verdict.gain_interval[0] < 0.0 < verdict.gain_interval[1]


def test_gain_exactly_at_margin_rejects() -> None:
    # Strict inequality, and no uncertainty left to collect: a gain
    # measured exactly at the margin (SE 0) does not clear it, and the
    # point interval does not straddle it either — so REJECT, not HOLD.
    # (With SE > 0 the same boundary would HOLD: the interval would
    # straddle the margin and more evidence could move it.)
    candidate = _version(log_evidence=2.0, log_evidence_se=0.0)
    verdict = promote(
        _version(name="v1", log_evidence=0.0, log_evidence_se=0.0),
        candidate,
        PromotionPolicy(evidence_margin=2.0),
    )
    assert verdict.action is PromotionAction.REJECT


def test_lower_bound_exactly_at_margin_with_uncertainty_holds() -> None:
    # Same boundary, but with measurement error: the interval straddles
    # the margin, so the verdict is HOLD (collect more evidence), not
    # REJECT. Gain chosen so the lower bound lands exactly on margin 0.
    gain = 1.96 * math.sqrt(0.5)
    candidate = _version(log_evidence=gain, log_evidence_se=0.5)
    verdict = promote(_champion(), candidate)
    assert verdict.action is PromotionAction.HOLD
    assert verdict.gain_interval is not None
    assert verdict.gain_interval[0] == pytest.approx(0.0)


def test_unmeasured_evidence_never_promotes() -> None:
    # 100% benchmark accuracy, no evidence measurement: HOLD, and the
    # verdict carries no gain to misread as zero.
    candidate = _version(log_evidence=None, log_evidence_se=None, successes=100, trials=100)
    verdict = promote(_champion(), candidate)
    assert verdict.action is PromotionAction.HOLD
    assert verdict.evidence_gain is None
    assert verdict.gain_interval is None
    assert "accuracy-only promotion is not allowed" in verdict.reason


def test_unmeasured_champion_holds() -> None:
    champion = _version(name="v1", log_evidence=None, log_evidence_se=None, successes=80)
    verdict = promote(champion, _version())
    assert verdict.action is PromotionAction.HOLD


def test_margin_blocks_an_otherwise_clear_gain() -> None:
    verdict = promote(_champion(), _version(), PromotionPolicy(evidence_margin=6.0))
    assert verdict.action is PromotionAction.HOLD


# ---------------------------------------------------------------- promote: refusal guards


def test_same_version_name_refused() -> None:
    with pytest.raises(ValueError, match="against itself"):
        promote(_version(name="v1"), _version(name="v1"))


def test_different_splits_refused() -> None:
    with pytest.raises(ValueError, match="splits differ"):
        promote(_champion(), _version(split="heldout-v4"))


def test_default_policy_is_zero_margin() -> None:
    assert PromotionPolicy().evidence_margin == 0.0


# ---------------------------------------------------------------- verdict record


def test_verdict_summary_reports_complexity_and_reason() -> None:
    candidate = _version(name="v2-wide", depth=24, width=768, log_evidence=-4.0)
    verdict = promote(_champion(), candidate)
    assert isinstance(verdict, PromotionVerdict)
    summary = verdict.summary()
    assert "REJECT" in summary
    assert "champion 0.0156 (L=12, n=768)" in summary
    assert "candidate 0.0312 (L=24, n=768)" in summary
    assert "accuracy gain: +0.080" in summary
    assert "evidence gain: -4.000" in summary
    assert "accuracy gain +0.080 does not promote" in summary
    # Audit artifact: split, SEs, and accuracy intervals are printed,
    # not just held on the objects.
    assert "evidence split: heldout-v3" in summary
    assert "evidence SEs: champion 0.500, candidate 0.500" in summary
    assert "accuracy 95% CIs: champion [0.711, 0.867]" in summary
    assert "candidate [0.802, 0.930]" in summary


def test_verdict_summary_unmeasured_names_no_gain() -> None:
    candidate = _version(log_evidence=None, log_evidence_se=None, successes=100)
    verdict = promote(_champion(), candidate)
    summary = verdict.summary()
    assert "evidence gain: UNMEASURED" in summary
    assert "evidence split: heldout-v3" in summary
    assert "evidence SEs" not in summary
