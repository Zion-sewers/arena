"""Judge calibration: "confidence is not evidence" (arena#45).

A judge's self-reported confidence is a claim about its own accuracy. The
claim is checked against MEASURED agreement: on items where the judge's
label can be compared with a reference (annotator labels, see
:func:`observations_from_alt_test`), did the judge agree as often as its
confidence said it would?

Rules (from the issue's acceptance criteria, executable form):
* confidence is calibrated against measured agreement — per confidence bin,
  the observed agreement rate is compared with the stated confidence;
* uncalibrated judge scores never gate anything — :func:`may_gate` is True
  only for a CALIBRATED judge, and :func:`require_calibrated` refuses
  anything else with a typed error;
* self-reported confidence is NOT evidence — the gate takes ONLY the
  calibration record. Stated confidence is not even an argument to the
  gate, so no call site can pass a high confidence number in place of a
  measurement. The grade record itself enforces status↔evidence
  consistency at construction (see ``JudgeCalibration``), so a
  hand-built record claiming CALIBRATED without the measurement behind
  it is refused before it can reach the gate.

Measurement discipline (same as validity.py / decision.py): the grade rests
on the Wilson 95% interval LOWER bound on agreement, never on the point
estimate. A bin fails when its mean stated confidence exceeds the lower
bound on measured agreement by more than the policy tolerance — the
overconfident direction is the dangerous one. ECE (expected calibration
error) is reported as a diagnostic only; it never decides the grade, so a
small ECE over point estimates cannot smuggle an under-measured judge
through. A judge with too little data is UNCALIBRATED, never trusted:
5/5 agreement at confidence 1.0 is a point estimate of 1.0 with a lower
bound of 0.565, and does not clear the bar.

Everything is stdlib-only and deterministic. The module makes no network
call: arena has no live judge transport (README: "LLM-as-judge is planned,
not implemented"), so observations are RECORDED (label, stated-confidence)
pairs supplied by the caller. A judge with no supplied confidence is
uncalibrated by definition — the module never invents one.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agenteval_bench.decision import wilson_interval
from alt_test.types import AltTestItem

#: The closed set of calibration statuses. UNCALIBRATED means "no usable
#: measurement" — it is never trusted, and it is distinct from
#: MISCALIBRATED ("measured, and the measurement refutes the confidence").
CalibrationStatus = Literal["calibrated", "miscalibrated", "uncalibrated"]


class UncalibratedJudgeError(ValueError):
    """A judge whose scores were asked to gate something without calibration.

    Raised by :func:`require_calibrated` for any status other than
    CALIBRATED. A ValueError subclass so callers that already guard on
    value errors fail closed the same way.
    """


@dataclass(frozen=True)
class ConfidenceObservation:
    """One measured (confidence, agreement) pair for one judge.

    ``stated_confidence`` is the confidence the judge reported for its
    label on ``item_id``; ``agreed`` is whether that label agreed with the
    reference on the item. The observation carries no label and no score:
    calibration is about the confidence claim, not about the label.
    """

    judge: str
    item_id: str
    stated_confidence: float
    agreed: bool

    def __post_init__(self) -> None:
        if not self.judge:
            raise ValueError("observation judge must be non-empty")
        if not self.item_id:
            raise ValueError("observation item_id must be non-empty")
        if not isinstance(self.agreed, bool):
            raise TypeError(
                f"observation agreed must be a bool, got {self.agreed!r}"
            )
        if not math.isfinite(self.stated_confidence):
            raise ValueError(
                f"stated confidence must be finite, got {self.stated_confidence!r}"
            )
        if not 0.0 <= self.stated_confidence <= 1.0:
            raise ValueError(
                f"stated confidence must be in [0, 1], got {self.stated_confidence}"
            )


@dataclass(frozen=True)
class CalibrationPolicy:
    """Policy for what counts as calibrated.

    ``min_total`` is the minimum observations for any grade other than
    UNCALIBRATED. ``bins`` is the number of equal-width confidence bins
    over [0, 1]; ``min_bin_n`` is the minimum observations in a non-empty
    bin for the bin to count as measured. ``tolerance`` is how far a bin's
    mean stated confidence may exceed the Wilson LOWER bound on its
    measured agreement before the bin — and the judge — fails.
    """

    min_total: int = 50
    bins: int = 10
    min_bin_n: int = 5
    tolerance: float = 0.10

    def __post_init__(self) -> None:
        if self.min_total < 1:
            raise ValueError(f"min_total must be >= 1, got {self.min_total}")
        if self.bins < 1:
            raise ValueError(f"bins must be >= 1, got {self.bins}")
        if self.min_bin_n < 1:
            raise ValueError(f"min_bin_n must be >= 1, got {self.min_bin_n}")
        if not math.isfinite(self.tolerance):
            raise ValueError(f"tolerance must be finite, got {self.tolerance!r}")
        if not 0.0 <= self.tolerance <= 1.0:
            raise ValueError(f"tolerance must be in [0, 1], got {self.tolerance}")

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form of the policy."""
        return {
            "min_total": self.min_total,
            "bins": self.bins,
            "min_bin_n": self.min_bin_n,
            "tolerance": self.tolerance,
        }


def default_calibration_policy() -> CalibrationPolicy:
    """A CalibrationPolicy built from the defaults."""
    return CalibrationPolicy()


@dataclass(frozen=True)
class CalibrationBin:
    """One confidence bin: stated confidence vs measured agreement.

    ``gap`` is ``mean_confidence - agreement_lower``: positive means the
    judge claimed more than the measurement supports. ``passed`` is True
    when the gap is within the policy tolerance. Empty bins are never
    constructed — a bin with no observations carries no claim.
    """

    bin_index: int
    lower_edge: float
    upper_edge: float
    n: int
    mean_confidence: float
    agreement_rate: float
    agreement_interval: tuple[float, float]
    gap: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form of the bin."""
        return {
            "bin": self.bin_index,
            "range": [self.lower_edge, self.upper_edge],
            "n": self.n,
            "mean_confidence": self.mean_confidence,
            "agreement_rate": self.agreement_rate,
            "agreement_interval": list(self.agreement_interval),
            "gap": self.gap,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class JudgeCalibration:
    """The calibration record of one judge.

    ``ece`` is the expected calibration error over point estimates — a
    diagnostic, never the grade. ``status`` is the grade; ``reasons`` name
    every failed condition (empty when CALIBRATED); ``notes`` carry
    non-failing observations such as conservative (underconfident) bins.
    ``policy`` is the grading policy the status was computed under, so a
    CALIBRATED record minted under a lax custom policy is distinguishable
    from a default-policy one.

    The record enforces status↔evidence consistency in ``__post_init__``
    (the same discipline as ``Decision`` in decision.py and
    ``AllocationReport`` in judges.py): a hand-built record that claims
    CALIBRATED without the measurement behind it is refused at
    construction, so the gate cannot be passed a forged record.
    """

    judge: str
    status: CalibrationStatus
    n_total: int
    ece: float | None
    bins: tuple[CalibrationBin, ...] = ()
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    policy: CalibrationPolicy = field(default_factory=default_calibration_policy)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, CalibrationPolicy):
            raise TypeError(
                f"calibration policy must be a CalibrationPolicy, "
                f"got {self.policy!r}"
            )
        if self.status == "calibrated":
            if not self.bins:
                raise ValueError(
                    "a CALIBRATED record needs non-empty bins: a grade "
                    "without a measurement is a forgery"
                )
            if not all(b.passed for b in self.bins):
                raise ValueError(
                    "a CALIBRATED record cannot carry a failed bin"
                )
            if self.reasons:
                raise ValueError("a CALIBRATED record cannot carry reasons")
            if self.ece is None:
                raise ValueError("a CALIBRATED record needs a measured ECE")
        elif self.status == "miscalibrated":
            if not self.bins:
                raise ValueError(
                    "a MISCALIBRATED record needs non-empty bins"
                )
            if not self.reasons:
                raise ValueError(
                    "a MISCALIBRATED record needs non-empty reasons"
                )
            if not any(not b.passed for b in self.bins):
                raise ValueError(
                    "a MISCALIBRATED record needs at least one failed bin"
                )
            if self.ece is None:
                raise ValueError("a MISCALIBRATED record needs a measured ECE")
        elif self.status == "uncalibrated":
            if self.bins:
                raise ValueError(
                    "an UNCALIBRATED record cannot carry bins: too little "
                    "data is not a measurement"
                )
            if self.ece is not None:
                raise ValueError("an UNCALIBRATED record cannot carry an ECE")
        else:  # pragma: no cover - Literal type, unreachable in typed code
            raise ValueError(f"unknown calibration status {self.status!r}")
        if self.status in ("calibrated", "miscalibrated"):
            if self.n_total != sum(b.n for b in self.bins):
                raise ValueError(
                    f"n_total {self.n_total} does not match the bins' "
                    f"total {sum(b.n for b in self.bins)}"
                )
            for b in self.bins:
                if b.passed != (
                    b.mean_confidence - b.agreement_interval[0]
                    <= self.policy.tolerance
                ):
                    raise ValueError(
                        f"bin {b.bin_index} passed={b.passed} contradicts "
                        "its gap under the record's policy"
                    )

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form of the record."""
        return {
            "judge": self.judge,
            "status": self.status,
            "n_total": self.n_total,
            "ece": self.ece,
            "bins": [b.to_dict() for b in self.bins],
            "reasons": list(self.reasons),
            "notes": list(self.notes),
            "policy": self.policy.to_dict(),
        }


def _bin_index(confidence: float, bins: int) -> int:
    """Bin index for a confidence in [0, 1]; 1.0 lands in the last bin."""
    index = int(confidence * bins)
    return min(index, bins - 1)


def bin_range_label(lower_edge: float, upper_edge: float) -> str:
    """Human-readable bin range: the last bin is closed on top.

    Bin 9 of a 10-bin grid covers [0.9, 1.0] — confidence 1.0 lands in
    it — so it prints with a closed bracket, not the half-open ``)``
    the other bins use.
    """
    if upper_edge >= 1.0:
        return f"[{lower_edge:.1f}, {upper_edge:.1f}]"
    return f"[{lower_edge:.1f}, {upper_edge:.1f})"


def calibrate(
    judge: str,
    observations: Sequence[ConfidenceObservation],
    policy: CalibrationPolicy | None = None,
) -> JudgeCalibration:
    """Calibrate one judge's stated confidence against measured agreement.

    Every observation must belong to ``judge``; observations for another
    judge are refused, not silently mixed in. The grade:
    * UNCALIBRATED when there are no observations, fewer than
      ``min_total``, or any non-empty bin holds fewer than ``min_bin_n``
      observations — too little evidence is not evidence;
    * MISCALIBRATED when a measured bin's mean stated confidence exceeds
      the Wilson lower bound on its agreement by more than ``tolerance``;
    * CALIBRATED otherwise.
    """
    policy = policy or default_calibration_policy()
    if not judge:
        raise ValueError("judge must be non-empty")
    seen: set[tuple[str, str]] = set()
    for obs in observations:
        if obs.judge != judge:
            raise ValueError(
                f"observation for judge {obs.judge!r} mixed into calibration "
                f"of {judge!r}"
            )
        pair = (obs.judge, obs.item_id)
        if pair in seen:
            raise ValueError(
                f"duplicate observation for judge {obs.judge!r} item "
                f"{obs.item_id!r}: one item is one measurement, copies "
                "cannot inflate n"
            )
        seen.add(pair)
    n_total = len(observations)
    if n_total == 0:
        return JudgeCalibration(
            judge=judge,
            status="uncalibrated",
            n_total=0,
            ece=None,
            reasons=(
                (
                    "no observations: a judge with no measured agreement is "
                    "uncalibrated by definition"
                ),
            ),
            policy=policy,
        )
    if n_total < policy.min_total:
        return JudgeCalibration(
            judge=judge,
            status="uncalibrated",
            n_total=n_total,
            ece=None,
            reasons=(
                f"only {n_total} observations, policy needs {policy.min_total}",
            ),
            policy=policy,
        )

    grouped: dict[int, list[ConfidenceObservation]] = {}
    for obs in observations:
        grouped.setdefault(_bin_index(obs.stated_confidence, policy.bins), []).append(obs)

    bins: list[CalibrationBin] = []
    reasons: list[str] = []
    notes: list[str] = []
    ece = 0.0
    for index in sorted(grouped):
        group = grouped[index]
        n = len(group)
        if n < policy.min_bin_n:
            return JudgeCalibration(
                judge=judge,
                status="uncalibrated",
                n_total=n_total,
                ece=None,
                reasons=(
                    (
                        f"bin {index} "
                        f"{bin_range_label(index / policy.bins, (index + 1) / policy.bins)} "
                        f"holds only {n} observations, policy needs {policy.min_bin_n}"
                    ),
                ),
                policy=policy,
            )
        mean_confidence = sum(o.stated_confidence for o in group) / n
        agreed = sum(1 for o in group if o.agreed)
        agreement_rate = agreed / n
        interval = wilson_interval(agreed, n)
        gap = mean_confidence - interval[0]
        passed = gap <= policy.tolerance
        if not passed:
            reasons.append(
                f"bin {index} "
                f"{bin_range_label(index / policy.bins, (index + 1) / policy.bins)}: "
                f"mean confidence {mean_confidence:.2f} exceeds the Wilson "
                f"lower bound on measured agreement {interval[0]:.2f} by "
                f"{gap:.2f} (tolerance {policy.tolerance:.2f})"
            )
        elif interval[0] - mean_confidence > policy.tolerance:
            notes.append(
                f"bin {index} "
                f"{bin_range_label(index / policy.bins, (index + 1) / policy.bins)} "
                "is conservative: measured agreement lower bound "
                f"{interval[0]:.2f} exceeds mean confidence "
                f"{mean_confidence:.2f} by {interval[0] - mean_confidence:.2f}"
            )
        ece += (n / n_total) * abs(agreement_rate - mean_confidence)
        bins.append(
            CalibrationBin(
                bin_index=index,
                lower_edge=index / policy.bins,
                upper_edge=(index + 1) / policy.bins,
                n=n,
                mean_confidence=mean_confidence,
                agreement_rate=agreement_rate,
                agreement_interval=interval,
                gap=gap,
                passed=passed,
            )
        )

    if reasons:
        return JudgeCalibration(
            judge=judge,
            status="miscalibrated",
            n_total=n_total,
            ece=ece,
            bins=tuple(bins),
            reasons=tuple(reasons),
            notes=tuple(notes),
            policy=policy,
        )
    return JudgeCalibration(
        judge=judge,
        status="calibrated",
        n_total=n_total,
        ece=ece,
        bins=tuple(bins),
        reasons=(),
        notes=tuple(notes),
        policy=policy,
    )


def may_gate(calibration: JudgeCalibration) -> bool:
    """Whether this judge's scores may gate anything: CALIBRATED only.

    The function takes ONLY the calibration record. A stated confidence
    cannot be passed here, so no call site can substitute a high
    confidence number for a measurement.
    """
    return calibration.status == "calibrated"


def require_calibrated(calibration: JudgeCalibration) -> None:
    """Refuse to gate on an uncalibrated or miscalibrated judge.

    Raises :class:`UncalibratedJudgeError` naming the judge, its status,
    and every failed condition. CALIBRATED passes silently.
    """
    if calibration.status == "calibrated":
        return
    detail = (
        f"judge {calibration.judge!r} is {calibration.status} "
        f"({calibration.n_total} observations)"
    )
    if calibration.reasons:
        detail += ": " + "; ".join(calibration.reasons)
    raise UncalibratedJudgeError(
        f"refusing to gate on {detail} — uncalibrated judge scores never "
        "gate anything"
    )


@dataclass(frozen=True)
class SkippedAltTestItem:
    """An alt-test item that produced no calibration observation, named."""

    item_id: str
    reason: str


@dataclass(frozen=True)
class AltTestObservations:
    """Calibration observations derived from an alt-test dataset.

    ``skipped`` names every item that carried a label for the judge but
    no usable observation — a tied annotator majority, or a label with no
    supplied confidence. Nothing is silently dropped.
    """

    observations: tuple[ConfidenceObservation, ...] = ()
    skipped: tuple[SkippedAltTestItem, ...] = ()
    notes: tuple[str, ...] = ()


def _check_bridge_items(items: Sequence[AltTestItem], judge: str) -> None:
    """Enforce the alt-test data contract the bridge relies on.

    Mirrors ``alt_test.runner._validate`` / ``_check_label_types`` for the
    items the bridge reads, scoped to the target judge and the annotators
    (labels of other judges are not the bridge's business): every item
    needs at least 3 annotator labels — a 1- or 2-annotator "majority" is
    not a majority — and every label the bridge compares must be a string
    (discrete tasks only; a bool or a float label is refused at the
    boundary instead of silently counting as disagreement).
    """
    for item in items:
        n_ann = len(item.annotator_labels)
        if n_ann < 3:
            raise ValueError(
                f"item {item.id!r} has {n_ann} annotators, "
                "need at least 3 for a majority reference"
            )
        labels = list(item.annotator_labels.items())
        judge_label = item.judge_labels.get(judge)
        if judge_label is not None:
            labels.append((judge, judge_label))
        for name, label in labels:
            if isinstance(label, bool):
                raise TypeError(
                    f"item {item.id!r}: label {name!r} must not be a bool"
                )
            if not isinstance(label, str):
                raise TypeError(
                    f"item {item.id!r}: discrete task needs string labels, "
                    f"{name!r} got {label!r}"
                )


def observations_from_alt_test(
    items: Sequence[AltTestItem],
    confidences: Mapping[str, float],
    judge: str,
) -> AltTestObservations:
    """Derive calibration observations from recorded alt-test labels.

    ``confidences`` maps item id -> the confidence the judge stated for
    its label on that item, supplied by the caller. Measured agreement is
    the judge's label vs the majority annotator label on the same item
    (discrete tasks only — a continuous or text task is refused, not
    approximated, because agreement there needs a similarity threshold
    this function does not own). Items where the judge has no label are
    not scored on it (alt-test convention) and are not reported as
    skipped; items with a label but a tied annotator majority, or with
    no supplied confidence, are named in ``skipped``.
    """
    if not judge:
        raise ValueError("judge must be non-empty")
    for item in items:
        if item.task != "discrete":
            raise ValueError(
                f"observations_from_alt_test supports discrete tasks only; "
                f"item {item.id!r} is {item.task!r}"
            )
    _check_bridge_items(items, judge)
    observations: list[ConfidenceObservation] = []
    skipped: list[SkippedAltTestItem] = []
    for item in items:
        label = item.judge_labels.get(judge)
        if label is None:
            continue
        counts = Counter(item.annotator_labels.values())
        top = max(counts.values())
        majority = [value for value, count in counts.items() if count == top]
        if len(majority) != 1:
            skipped.append(
                SkippedAltTestItem(
                    item_id=item.id,
                    reason=f"tied annotator majority ({top} votes each); "
                    "agreement is not determinable",
                )
            )
            continue
        confidence = confidences.get(item.id)
        if confidence is None:
            skipped.append(
                SkippedAltTestItem(
                    item_id=item.id,
                    reason="judge labelled the item but stated no confidence; "
                    "a missing confidence is not evidence",
                )
            )
            continue
        observations.append(
            ConfidenceObservation(
                judge=judge,
                item_id=item.id,
                stated_confidence=confidence,
                agreed=label == majority[0],
            )
        )
    notes: list[str] = []
    if skipped:
        notes.append(
            f"{len(skipped)} item(s) skipped (see skipped); "
            "calibration covers the observed items only"
        )
    return AltTestObservations(
        observations=tuple(observations),
        skipped=tuple(skipped),
        notes=tuple(notes),
    )


def observations_from_json(text: str) -> tuple[str, tuple[ConfidenceObservation, ...]]:
    """Load calibration observations from JSON (the ``calibrate`` CLI input).

    A mapping with a ``judge`` string and an ``observations`` list, or a
    bare list (judge taken from the observations, which must then all
    name one judge). Each observation carries ``item``, ``confidence``
    and ``agreed``. The loader fails closed: duplicate (judge, item)
    pairs, unknown keys, a missing required field, or a confidence
    outside [0, 1] all raise ValueError, and wrong-typed fields raise
    TypeError, so a typo cannot silently become an observation with no
    measured agreement.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    raw_judge: str | None = None
    raw_observations: list[Any] = []
    if isinstance(data, list):
        raw_observations = data
    elif isinstance(data, dict):
        unknown = set(data) - {"judge", "observations"}
        if unknown:
            raise ValueError(f"unknown keys: {sorted(unknown)}")
        raw_judge = data.get("judge")
        if raw_judge is not None and not isinstance(raw_judge, str):
            raise TypeError(f"judge must be a string, got {raw_judge!r}")
        if data.get("observations") is None:
            raise ValueError("missing required field 'observations'")
        raw_observations = data["observations"]
    else:
        raise TypeError("observations JSON must be a mapping or a list")
    if not isinstance(raw_observations, list):
        raise TypeError(
            f"observations must be a list, got {type(raw_observations).__name__}"
        )
    observations: list[ConfidenceObservation] = []
    seen: set[tuple[str, str]] = set()
    for entry in raw_observations:
        if not isinstance(entry, dict):
            raise TypeError(f"observation must be a mapping, got {entry!r}")
        unknown = set(entry) - {"judge", "item", "confidence", "agreed"}
        if unknown:
            raise ValueError(f"observation has unknown keys: {sorted(unknown)}")
        for key in ("item", "confidence", "agreed"):
            if key not in entry:
                raise ValueError(f"observation missing required field {key!r}")
        entry_judge = entry.get("judge", raw_judge)
        if not isinstance(entry_judge, str) or not entry_judge:
            raise TypeError(
                "observation judge must be a non-empty string "
                "(from the entry or the document)"
            )
        if raw_judge is not None and entry_judge != raw_judge:
            raise ValueError(
                f"observation judge {entry_judge!r} does not match "
                f"document judge {raw_judge!r}"
            )
        if not isinstance(entry["item"], str) or not entry["item"]:
            raise TypeError(f"observation item must be a non-empty string, got {entry['item']!r}")
        if not isinstance(entry["confidence"], (int, float)) or isinstance(
            entry["confidence"], bool
        ):
            raise TypeError(
                f"observation confidence must be a number, got {entry['confidence']!r}"
            )
        if not isinstance(entry["agreed"], bool):
            raise TypeError(
                f"observation agreed must be a bool, got {entry['agreed']!r}"
            )
        pair = (entry_judge, entry["item"])
        if pair in seen:
            raise ValueError(f"duplicate observation for judge {entry_judge!r} item {entry['item']!r}")
        seen.add(pair)
        observations.append(
            ConfidenceObservation(
                judge=entry_judge,
                item_id=entry["item"],
                stated_confidence=float(entry["confidence"]),
                agreed=entry["agreed"],
            )
        )
    judges = {o.judge for o in observations}
    if len(judges) > 1:
        raise ValueError(
            f"observations name {len(judges)} judges {sorted(judges)}; "
            "calibrate one judge per document"
        )
    if raw_judge is not None:
        judge = raw_judge
    elif judges:
        judge = next(iter(judges))
    else:
        raise ValueError("no observations and no judge: nothing to calibrate")
    if not observations:
        raise ValueError("observations list is empty: nothing to calibrate")
    return judge, tuple(observations)
