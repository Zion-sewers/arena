"""Model/version selection by code length; capacity meter (arena#50).

Source: Polyanskiy/Wu 'Information Theory: From Coding to Learning'
(catalog G.22) — INSPIRED. Two-part code length as the model-selection
criterion (universal compression = formal Occam). Metric entropy ->
uniform convergence rates. Minimax-Bayes duality. No pipeline stage
'invents' evidence (data processing cannot create information).

The selection rule, in executable form:

* Every version carries TWO measured numbers, both in bits: the
  description length of the version itself (``description_bits``) and
  the held-out log-loss of the data under the version
  (``held_out_mean_log_loss_bits`` per item over ``held_out_n`` items).
  The total description length is their sum — the two-part code.
* At equal performance (held-out mean log-loss within
  ``equivalence_band_bits`` of the best-fitting version), the more
  compressible version wins: :func:`select_version` picks the minimum
  total description length among the equivalent versions.
* :func:`select_by_total_length` is the pure MDL rule: minimum total
  description length over all versions, no equivalence band. The two
  rules agree whenever the best-fitting version is also the most
  compressible; they differ exactly when a worse-fitting version saves
  more code than it costs in fit, and the report names that trade in
  bits both ways.

The equivalence band is a POINT-ESTIMATE comparison with no sampling
claim: it compares mean held-out log-losses as numbers, and "equal
performance" means the point estimates lie within the band, not that
the versions are statistically indistinguishable. The standard error
of a mean log-loss over n held-out items can exceed the default
0.01-bit band, especially at small n, so a band that is wide enough
at n=10000 may declare noise "equal" at n=100. Callers who need an
uncertainty-aware comparison should scale the band with n (for
example with the standard error of the mean log-loss) instead of
relying on the default.

What "description length" means here, honestly (established MDL
practice): the length of the version's fitted artifact under a FIXED,
stated compressor — canonical JSON serialisation
(:func:`measure_description_bits`, zlib level 9). That number is an
upper bound on the version's Kolmogorov complexity, not the complexity
itself; the comparison is valid because every version is measured
under the SAME compressor, so a shorter measured code is a genuinely
more compressible version. The compressor is carried on the version
itself (``ModelVersion.compressor``) and enforced equal across
candidates in :func:`_require_versions` — a version measured under a
different compressor is refused, not silently compared.

One measurement caveat, stated plainly: JSON decimal encoding makes
the measured length sensitive to float formatting, not only to weight
values — two artifacts differing only in float precision can measure
differently. Quantise or round weights before measuring when the
comparison must be about the weights, not their print precision.

Parameter count alone is NOT a complexity argument (acceptance
criterion 3). It is enforced structurally, not by convention:
selection functions never accept a parameter count — ``ModelVersion``
carries ``parameter_count`` as a diagnostic-only field that no
selection path reads, and the tests pin that mutating it cannot change
a selection. The measurement trap is the proof: a 10M-parameter
routing table of repeated rules compresses to fewer bits than a
100k-parameter table of random weights, and the shorter code wins.

The capacity meter (:func:`assess_capacity`) is the training-time
half: a checkpoint series of measured capacity (description bits of
the fitted artifact at that checkpoint) against measured
generalisation (held-out mean log-loss). Capacity growth without
generalisation improvement stops training: over a patience window the
meter returns STOP when capacity strictly grew while the best
held-out log-loss in the window failed to beat the best before the
window by more than ``min_improvement_bits``. The meter REPORTS a
STOP; it never drives a training loop itself, and no loop shipped in
this repo stops on it — acceptance criterion 2 is delivered here as a
STOP report plus a demo loop that honours it, not as a wired
training-loop control (this is an evaluation tool, spec NG5). A STOP
names the window, the capacity growth in bits, and both held-out
bests, so the stop is auditable, not a bare flag. The reported
generalisation gap is a diagnostic only; it does not enter the stop
decision.

Measurements are caller-measured, never invented here: held-out
log-loss must come from held-out data scored under the version (a
probabilistic judge/router output). There is no accuracy-derived
shortcut — accuracy without probabilities has no log-loss, and this
module refuses to invent one. The worked path from a real artifact
is :meth:`ModelVersion.from_artifact` (measures the bits itself and
records the compressor) plus :func:`mean_log_loss_bits` (computes
mean held-out log-loss from scored (probability, label) pairs); the
raw-number constructor remains as an explicitly marked escape hatch
for pre-measured versions. A two-part-code comparison is only
defined on the SAME held-out set: candidates must share one
``held_out_n`` (and one ``held_out_set`` id when given), enforced in
:func:`_require_versions`, because totals over different n compare
set size, not quality.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

#: The single compressor every description length in this module is
#: measured under. Fixed so versions are comparable; see module docstring.
COMPRESSOR = "zlib-9 over canonical JSON (sort_keys, compact separators)"

#: Default equivalence band in mean held-out log-loss bits per item:
#: versions within this of the best fit count as "equal performance".
#: A POINT-ESTIMATE band with no sampling claim (see module docstring):
#: the standard error of a mean over n held-out items can exceed it,
#: so scale the band with n when the comparison needs uncertainty.
DEFAULT_EQUIVALENCE_BAND_BITS = 0.01

#: Default patience: the number of consecutive checkpoints in the
#: stop window. A window of 3 means two consecutive stalls after the
#: first growth step are not enough — the meter waits for the third.
DEFAULT_PATIENCE = 3

#: Default improvement bar: a held-out gain smaller than this many
#: bits per item does not count as generalisation improvement.
DEFAULT_MIN_IMPROVEMENT_BITS = 0.001

#: The closed set of capacity-meter decisions. STOP is a report, not a
#: control action: the training loop honours it.
StopDecision = Literal["continue", "stop"]


def measure_description_bits(artifact: Mapping[str, object]) -> int:
    """Measure an artifact's description length in bits.

    The artifact is serialised canonically (sorted keys, compact
    separators, ASCII) and compressed with zlib level 9; the length is
    8 bits per compressed byte. Key order cannot change the result
    (canonicalisation), and a repetitive artifact measures shorter than
    a random one of the same size (compression, not counting). Values
    must be JSON-serialisable; anything else refuses loudly instead of
    silently dropping fields into a shorter code.
    """
    if not isinstance(artifact, Mapping):
        raise TypeError(
            f"cannot measure description length: artifact must be a mapping, got {type(artifact).__name__}"
        )
    try:
        canonical = json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"cannot measure description length: artifact is not JSON-serialisable: {e}"
        ) from e
    return 8 * len(zlib.compress(canonical.encode("utf-8"), 9))


@dataclass(frozen=True)
class ModelVersion:
    """One model/version candidate with its two measured numbers.

    ``description_bits`` is the measured length of the version itself
    (see :func:`measure_description_bits`); ``held_out_n`` and
    ``held_out_mean_log_loss_bits`` are the measured held-out fit.
    ``parameter_count`` is diagnostic only — no selection function
    reads it (acceptance criterion 3: parameter count alone is not a
    complexity argument). A version with no measured held-out fit
    cannot be constructed: ``held_out_n`` >= 1 is enforced.
    ``compressor`` names the compressor the description was measured
    under; ``held_out_set`` names the held-out set the fit was scored
    on. Both are enforced equal across candidates, so a comparison
    can never silently mix compressors or held-out sets. Prefer
    :meth:`from_artifact`, which measures the bits itself; the
    raw-number constructor is the escape hatch for pre-measured
    versions.
    """

    name: str
    description_bits: int
    held_out_n: int
    held_out_mean_log_loss_bits: float
    parameter_count: int | None = None
    compressor: str = COMPRESSOR
    held_out_set: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("model version name must be a non-empty str")
        if isinstance(self.description_bits, bool) or not isinstance(self.description_bits, int):
            raise TypeError(f"description_bits must be an int, got {self.description_bits!r}")
        if self.description_bits < 0:
            raise ValueError(f"description_bits must be >= 0, got {self.description_bits}")
        if isinstance(self.held_out_n, bool) or not isinstance(self.held_out_n, int):
            raise TypeError(f"held_out_n must be an int, got {self.held_out_n!r}")
        if self.held_out_n < 1:
            raise ValueError(f"held_out_n must be >= 1, got {self.held_out_n}")
        if isinstance(self.held_out_mean_log_loss_bits, bool) or not isinstance(
            self.held_out_mean_log_loss_bits, (int, float)
        ):
            raise TypeError(
                "held_out_mean_log_loss_bits must be a number, "
                f"got {self.held_out_mean_log_loss_bits!r}"
            )
        if not math.isfinite(self.held_out_mean_log_loss_bits):
            raise ValueError(
                "held_out_mean_log_loss_bits must be finite, "
                f"got {self.held_out_mean_log_loss_bits}"
            )
        if self.held_out_mean_log_loss_bits < 0:
            raise ValueError(
                f"held_out_mean_log_loss_bits must be >= 0, got {self.held_out_mean_log_loss_bits}"
            )
        if self.parameter_count is not None and (
            isinstance(self.parameter_count, bool) or not isinstance(self.parameter_count, int)
        ):
            raise ValueError(
                f"parameter_count must be an int or None, got {self.parameter_count!r}"
            )
        if self.parameter_count is not None and self.parameter_count < 0:
            raise ValueError(f"parameter_count must be >= 0, got {self.parameter_count}")
        if not isinstance(self.compressor, str) or not self.compressor:
            raise ValueError("compressor must be a non-empty str")
        if self.held_out_set is not None and (
            not isinstance(self.held_out_set, str) or not self.held_out_set
        ):
            raise ValueError("held_out_set must be a non-empty str or None")

    @classmethod
    def from_artifact(
        cls,
        name: str,
        artifact: Mapping[str, object],
        held_out_n: int,
        held_out_mean_log_loss_bits: float,
        parameter_count: int | None = None,
        held_out_set: str | None = None,
    ) -> ModelVersion:
        """Build a version by MEASURING its artifact, not trusting a number.

        ``description_bits`` is derived by calling
        :func:`measure_description_bits` on the artifact, and
        ``compressor`` is recorded as the module's single compressor —
        the measured path is the default, the raw-number constructor
        is the escape hatch for pre-measured versions.
        """
        return cls(
            name=name,
            description_bits=measure_description_bits(artifact),
            held_out_n=held_out_n,
            held_out_mean_log_loss_bits=held_out_mean_log_loss_bits,
            parameter_count=parameter_count,
            compressor=COMPRESSOR,
            held_out_set=held_out_set,
        )

    @property
    def held_out_total_log_loss_bits(self) -> float:
        """Total held-out code length: mean log-loss times n, in bits."""
        try:
            return self.held_out_mean_log_loss_bits * self.held_out_n
        except OverflowError as e:
            raise ValueError(
                f"held_out total log-loss overflows for held_out_n={self.held_out_n}"
            ) from e

    @property
    def total_description_bits(self) -> float:
        """Two-part code length: model bits + data-given-model bits."""
        try:
            return self.description_bits + self.held_out_total_log_loss_bits
        except OverflowError as e:
            raise ValueError(
                f"total description length overflows for description_bits={self.description_bits}"
            ) from e


def mean_log_loss_bits(pairs: Sequence[tuple[float, bool]]) -> float:
    """Mean held-out log-loss in bits from scored (probability, label) pairs.

    Each pair is (predicted probability of the positive label, whether
    the label is positive). Probabilities must lie in (0, 1]; a 0 or
    1-outside probability is refused, not clipped — clipping would
    invent a finite loss for a prediction the model ruled out.
    """
    scored = tuple(pairs)
    if not scored:
        raise ValueError("mean log-loss needs at least one scored pair")
    total = 0.0
    for p, label in scored:
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise TypeError(f"predicted probability must be a number, got {p!r}")
        if not isinstance(label, bool):
            raise TypeError(f"label must be a bool, got {label!r}")
        if not 0.0 < p <= 1.0:
            raise ValueError(f"predicted probability must lie in (0, 1], got {p}")
        q = p if label else 1.0 - p
        if q <= 0.0:
            raise ValueError(
                "a prediction of probability 1 for the wrong label "
                "has infinite log-loss and is refused, not clipped"
            )
        total += -math.log2(q)
    return total / len(scored)


@dataclass(frozen=True)
class SelectionReport:
    """A version selection: the winner, the trade, the audit trail.

    ``winner`` is the selected version. ``fit_deficit_bits`` is how
    much worse the winner fits than the best-fitting version, in mean
    bits per item (0 when the winner is itself the best fit).
    ``code_saving_bits`` is how much shorter the winner's code is than
    the best-fitting version's, in total bits (0 when the winner is
    the best fit). ``equivalent`` holds the versions that counted as
    equal performance under the band; for the pure rule, which has no
    band, it holds just the winner — nothing else was judged equal
    performance, and the full considered set is ``candidates``.
    ``rule`` names which selection rule ran. ``equivalence_band_bits``
    is None for the pure rule, which has no band. The rationale states
    the trade in one sentence so the decision is auditable without
    re-running the selection.
    """

    winner: ModelVersion
    candidates: tuple[ModelVersion, ...]
    equivalent: tuple[ModelVersion, ...]
    best_fit: ModelVersion
    fit_deficit_bits: float
    code_saving_bits: float
    equivalence_band_bits: float | None
    rule: str

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("a selection report needs at least one candidate")
        if self.winner not in self.candidates:
            raise ValueError("the winner must be one of the candidates")
        if self.best_fit not in self.candidates:
            raise ValueError("the best fit must be one of the candidates")
        if not set(self.equivalent) <= set(self.candidates):
            raise ValueError("equivalent versions must be among the candidates")
        if not math.isfinite(self.fit_deficit_bits) or self.fit_deficit_bits < 0:
            raise ValueError(
                f"fit_deficit_bits must be finite and >= 0, got {self.fit_deficit_bits}"
            )
        if not math.isfinite(self.code_saving_bits) or self.code_saving_bits < 0:
            raise ValueError(
                f"code_saving_bits must be finite and >= 0, got {self.code_saving_bits}"
            )
        if self.equivalence_band_bits is not None and (
            not math.isfinite(self.equivalence_band_bits) or self.equivalence_band_bits < 0
        ):
            raise ValueError(
                "equivalence_band_bits must be finite and >= 0 or None, "
                f"got {self.equivalence_band_bits}"
            )

    @property
    def rationale(self) -> str:
        """One-sentence audit trail for the selection."""
        if self.winner == self.best_fit:
            return (
                f"{self.winner.name} wins: best fit "
                f"({self.winner.held_out_mean_log_loss_bits:.4f} bits/item) and "
                f"shortest total code among the equivalent versions "
                f"({self.winner.total_description_bits:.1f} bits)"
            )
        return (
            f"{self.winner.name} wins: fits "
            f"{self.fit_deficit_bits:.4f} bits/item worse than "
            f"{self.best_fit.name} but saves "
            f"{self.code_saving_bits:.1f} code bits, "
            f"total {self.winner.total_description_bits:.1f} bits vs "
            f"{self.best_fit.total_description_bits:.1f}"
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form of the selection.

        Strict JSON: the pure rule has no band, so the field is None,
        never infinity.
        """
        return {
            "winner": self.winner.name,
            "rule": self.rule,
            "compressor": self.winner.compressor,
            "equivalence_band_bits": self.equivalence_band_bits,
            "fit_deficit_bits": self.fit_deficit_bits,
            "code_saving_bits": self.code_saving_bits,
            "rationale": self.rationale,
            "candidates": [
                {
                    "name": v.name,
                    "description_bits": v.description_bits,
                    "held_out_n": v.held_out_n,
                    "held_out_mean_log_loss_bits": v.held_out_mean_log_loss_bits,
                    "total_description_bits": v.total_description_bits,
                    "parameter_count": v.parameter_count,
                    "compressor": v.compressor,
                    "held_out_set": v.held_out_set,
                }
                for v in self.candidates
            ],
            "equivalent": [v.name for v in self.equivalent],
        }


def _best_fit(versions: Sequence[ModelVersion]) -> ModelVersion:
    """The best-fitting version: lowest mean held-out log-loss.

    Ties break by shorter total code, then name, so the choice is
    deterministic — a tie in fit is not a coin flip.
    """
    return min(
        versions,
        key=lambda v: (v.held_out_mean_log_loss_bits, v.total_description_bits, v.name),
    )


def _shortest_total(
    versions: Sequence[ModelVersion],
) -> ModelVersion:
    """The most compressible version: lowest total description length.

    Ties break by better fit, then name, for determinism.
    """
    return min(
        versions,
        key=lambda v: (v.total_description_bits, v.held_out_mean_log_loss_bits, v.name),
    )


def _require_versions(versions: Sequence[ModelVersion]) -> tuple[ModelVersion, ...]:
    """Validate a candidate list: non-empty, unique names, one held-out set.

    A two-part-code comparison is only defined on the same data, so
    candidates must share one ``held_out_n`` and one ``held_out_set``
    id (a version with no set id cannot be compared against one with
    a named set), and one compressor — otherwise the totals compare
    set size or compressor, not quality.
    """
    ordered = tuple(versions)
    if not ordered:
        raise ValueError("selecting among no versions is not a selection")
    names = [v.name for v in ordered]
    if len(set(names)) != len(names):
        raise ValueError(f"version names must be unique, got {names}")
    ns = {v.held_out_n for v in ordered}
    if len(ns) != 1:
        raise ValueError(
            f"versions must share one held_out_n (same held-out set), got {sorted(ns)}"
        )
    compressors = {v.compressor for v in ordered}
    if len(compressors) != 1:
        raise ValueError(f"versions must share one compressor, got {sorted(compressors)}")
    held_out_sets = {v.held_out_set for v in ordered}
    if len(held_out_sets) != 1:
        raise ValueError(
            f"versions must share one held_out_set id, got {sorted(s or '' for s in held_out_sets)}"
        )
    return ordered


def select_version(
    versions: Sequence[ModelVersion],
    equivalence_band_bits: float = DEFAULT_EQUIVALENCE_BAND_BITS,
) -> SelectionReport:
    """Select by code length at equal performance (the issue's rule).

    Versions within ``equivalence_band_bits`` (mean held-out log-loss
    bits per item) of the best fit count as equal performance; among
    them the minimum total description length wins. The band is a
    point-estimate comparison with no sampling claim (see the module
    docstring): "equal performance" means the point estimates lie
    within the band. The report names the fit deficit the winner
    accepted and the code it saved, so a selection that trades fit
    for compression is visible, not silent.
    """
    if not math.isfinite(equivalence_band_bits) or equivalence_band_bits < 0:
        raise ValueError(
            f"equivalence_band_bits must be finite and >= 0, got {equivalence_band_bits}"
        )
    ordered = _require_versions(versions)
    best = _best_fit(ordered)
    equivalent = tuple(
        v
        for v in ordered
        if v.held_out_mean_log_loss_bits <= best.held_out_mean_log_loss_bits + equivalence_band_bits
    )
    winner = _shortest_total(equivalent)
    return SelectionReport(
        winner=winner,
        candidates=ordered,
        equivalent=equivalent,
        best_fit=best,
        fit_deficit_bits=winner.held_out_mean_log_loss_bits - best.held_out_mean_log_loss_bits,
        code_saving_bits=float(best.total_description_bits - winner.total_description_bits),
        equivalence_band_bits=equivalence_band_bits,
        rule="equal-performance code length",
    )


def select_by_total_length(
    versions: Sequence[ModelVersion],
) -> SelectionReport:
    """Select by minimum total description length (the pure MDL rule).

    No equivalence band: a version wins iff its two-part code is
    shortest, whatever its fit. This is the rule
    :func:`select_version` reduces to when the band covers every
    version; it is exposed separately so callers that want the
    unbanded trade state that choice explicitly. The report's
    ``equivalent`` holds just the winner: with no band, nothing else
    was judged equal performance.
    """
    ordered = _require_versions(versions)
    best = _best_fit(ordered)
    winner = _shortest_total(ordered)
    return SelectionReport(
        winner=winner,
        candidates=ordered,
        equivalent=(winner,),
        best_fit=best,
        fit_deficit_bits=winner.held_out_mean_log_loss_bits - best.held_out_mean_log_loss_bits,
        code_saving_bits=float(best.total_description_bits - winner.total_description_bits),
        equivalence_band_bits=None,
        rule="minimum total description length",
    )


@dataclass(frozen=True)
class CapacityReading:
    """One training checkpoint: measured capacity vs measured fit.

    ``capacity_bits`` is the measured description length of the fitted
    artifact at this checkpoint (see :func:`measure_description_bits`)
    — a capacity that grew because the artifact genuinely got more
    complex, not because a counter went up. ``held_out_*`` is the
    generalisation measurement; ``train_*`` is the fit on the training
    data. The generalisation gap (held-out minus train) is derived,
    never supplied, so a reading cannot claim a small gap while
    reporting numbers that say otherwise.
    """

    step: int
    capacity_bits: int
    held_out_mean_log_loss_bits: float
    train_mean_log_loss_bits: float

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int):
            raise TypeError(f"step must be an int, got {self.step!r}")
        if self.step < 0:
            raise ValueError(f"step must be >= 0, got {self.step}")
        if isinstance(self.capacity_bits, bool) or not isinstance(self.capacity_bits, int):
            raise TypeError(f"capacity_bits must be an int, got {self.capacity_bits!r}")
        if self.capacity_bits < 0:
            raise ValueError(f"capacity_bits must be >= 0, got {self.capacity_bits}")
        for label, value in (
            ("held_out_mean_log_loss_bits", self.held_out_mean_log_loss_bits),
            ("train_mean_log_loss_bits", self.train_mean_log_loss_bits),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} must be a number, got {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite, got {value}")
            if value < 0:
                raise ValueError(f"{label} must be >= 0, got {value}")

    @property
    def generalization_gap_bits(self) -> float:
        """Held-out minus train log-loss: the overfitting signal, in bits."""
        return self.held_out_mean_log_loss_bits - self.train_mean_log_loss_bits


@dataclass(frozen=True)
class CapacityVerdict:
    """A capacity-meter decision over a checkpoint series.

    ``decision`` is STOP when, over the last ``patience`` checkpoints,
    capacity strictly grew while the best held-out log-loss in the
    window failed to beat the best before the window by more than
    ``min_improvement_bits`` — capacity growth without generalisation
    improvement. ``capacity_growth_bits`` is the window's capacity
    growth (last minus first); ``max_generalization_gap_bits`` is the
    widest train/held-out gap inside the window, reported as a
    diagnostic — it does not enter the stop decision. The reason names
    the window steps and both held-out bests, so the stop is auditable.
    """

    decision: StopDecision
    reason: str
    window: tuple[CapacityReading, ...]
    capacity_growth_bits: int
    held_out_best_before_bits: float | None
    held_out_best_in_window_bits: float
    max_generalization_gap_bits: float
    patience: int
    min_improvement_bits: float

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form of the capacity verdict."""
        return {
            "decision": self.decision,
            "reason": self.reason,
            "window_steps": [r.step for r in self.window],
            "capacity_growth_bits": self.capacity_growth_bits,
            "held_out_best_before_bits": self.held_out_best_before_bits,
            "held_out_best_in_window_bits": self.held_out_best_in_window_bits,
            "max_generalization_gap_bits": self.max_generalization_gap_bits,
            "patience": self.patience,
            "min_improvement_bits": self.min_improvement_bits,
        }


def assess_capacity(
    readings: Sequence[CapacityReading],
    patience: int = DEFAULT_PATIENCE,
    min_improvement_bits: float = DEFAULT_MIN_IMPROVEMENT_BITS,
) -> CapacityVerdict:
    """Meter a checkpoint series: stop on capacity growth without gain.

    Needs at least ``patience + 1`` readings: the window is the last
    ``patience`` checkpoints and the baseline is the best held-out
    log-loss BEFORE the window. With fewer readings there is no window
    and no baseline, so the verdict is CONTINUE with the reason saying
    so — a meter that stops on a single checkpoint would be guessing,
    not measuring. Steps must be strictly increasing; a series that
    goes backwards in time is corrupt and refused.
    """
    if patience < 2:
        raise ValueError(f"patience must be >= 2, got {patience}")
    if not math.isfinite(min_improvement_bits) or min_improvement_bits < 0:
        raise ValueError(
            f"min_improvement_bits must be finite and >= 0, got {min_improvement_bits}"
        )
    ordered = tuple(readings)
    if not ordered:
        raise ValueError("a capacity meter with no readings cannot decide")
    steps = [r.step for r in ordered]
    if steps != sorted(steps) or len(set(steps)) != len(steps):
        raise ValueError(f"readings must be in strictly increasing step order, got {steps}")
    window = ordered[-patience:]
    best_in_window = min(r.held_out_mean_log_loss_bits for r in window)
    growth = window[-1].capacity_bits - window[0].capacity_bits
    max_gap = max(r.generalization_gap_bits for r in window)
    if len(ordered) <= patience:
        return CapacityVerdict(
            decision="continue",
            reason=(
                f"only {len(ordered)} reading(s) for a patience-{patience} window: "
                "no baseline before the window yet"
            ),
            window=window,
            capacity_growth_bits=growth,
            held_out_best_before_bits=None,
            held_out_best_in_window_bits=best_in_window,
            max_generalization_gap_bits=max_gap,
            patience=patience,
            min_improvement_bits=min_improvement_bits,
        )
    best_before = min(r.held_out_mean_log_loss_bits for r in ordered[:-patience])
    improvement = best_before - best_in_window
    if growth > 0 and improvement <= min_improvement_bits:
        return CapacityVerdict(
            decision="stop",
            reason=(
                f"capacity grew {growth} bits over steps "
                f"{window[0].step}-{window[-1].step} while held-out log-loss "
                f"improved only {improvement:.4f} bits/item "
                f"(best {best_in_window:.4f} vs {best_before:.4f} before, "
                f"bar {min_improvement_bits:.4f})"
            ),
            window=window,
            capacity_growth_bits=growth,
            held_out_best_before_bits=best_before,
            held_out_best_in_window_bits=best_in_window,
            max_generalization_gap_bits=max_gap,
            patience=patience,
            min_improvement_bits=min_improvement_bits,
        )
    return CapacityVerdict(
        decision="continue",
        reason=(
            f"capacity grew {growth} bits over steps "
            f"{window[0].step}-{window[-1].step} and held-out log-loss "
            f"improved {improvement:.4f} bits/item "
            f"(best {best_in_window:.4f} vs {best_before:.4f} before)"
        ),
        window=window,
        capacity_growth_bits=growth,
        held_out_best_before_bits=best_before,
        held_out_best_in_window_bits=best_in_window,
        max_generalization_gap_bits=max_gap,
        patience=patience,
        min_improvement_bits=min_improvement_bits,
    )
