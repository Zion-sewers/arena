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

What "description length" means here, honestly (established MDL
practice): the length of the version's fitted artifact under a FIXED,
stated compressor — canonical JSON serialisation
(:func:`measure_description_bits`, zlib level 9). That number is an
upper bound on the version's Kolmogorov complexity, not the complexity
itself; the comparison is valid because every version is measured
under the SAME compressor, so a shorter measured code is a genuinely
more compressible version. A version measured under a different
compressor is not comparable — the report carries no compressor field
because there is only one, named here.

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
window by more than ``min_improvement_bits``. The meter REPORTS; it
never drives a training loop itself — the loop honours the STOP
decision (the demo shows a loop that breaks on it). A STOP names the
window, the capacity growth in bits, and both held-out bests, so the
stop is auditable, not a bare flag.

Measurements are caller-measured, never invented here: held-out
log-loss must come from held-out data scored under the version (a
probabilistic judge/router output). There is no accuracy-derived
shortcut — accuracy without probabilities has no log-loss, and this
module refuses to invent one.
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
    try:
        canonical = json.dumps(
            artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=True
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
    """

    name: str
    description_bits: int
    held_out_n: int
    held_out_mean_log_loss_bits: float
    parameter_count: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("model version name must be non-empty")
        if self.description_bits < 0:
            raise ValueError(
                f"description_bits must be >= 0, got {self.description_bits}"
            )
        if self.held_out_n < 1:
            raise ValueError(f"held_out_n must be >= 1, got {self.held_out_n}")
        if not math.isfinite(self.held_out_mean_log_loss_bits):
            raise ValueError(
                "held_out_mean_log_loss_bits must be finite, "
                f"got {self.held_out_mean_log_loss_bits}"
            )
        if self.held_out_mean_log_loss_bits < 0:
            raise ValueError(
                "held_out_mean_log_loss_bits must be >= 0, "
                f"got {self.held_out_mean_log_loss_bits}"
            )
        if self.parameter_count is not None and self.parameter_count < 0:
            raise ValueError(
                f"parameter_count must be >= 0, got {self.parameter_count}"
            )

    @property
    def held_out_total_log_loss_bits(self) -> float:
        """Total held-out code length: mean log-loss times n, in bits."""
        return self.held_out_mean_log_loss_bits * self.held_out_n

    @property
    def total_description_bits(self) -> float:
        """Two-part code length: model bits + data-given-model bits."""
        return self.description_bits + self.held_out_total_log_loss_bits


@dataclass(frozen=True)
class SelectionReport:
    """A version selection: the winner, the trade, the audit trail.

    ``winner`` is the selected version. ``fit_deficit_bits`` is how
    much worse the winner fits than the best-fitting version, in mean
    bits per item (0 when the winner is itself the best fit).
    ``code_saving_bits`` is how much shorter the winner's code is than
    the best-fitting version's, in total bits (0 when the winner is
    the best fit). ``equivalent`` holds the versions that counted as
    equal performance under the band; ``rule`` names which selection
    rule ran. The rationale states the trade in one sentence so the
    decision is auditable without re-running the selection.
    """

    winner: ModelVersion
    candidates: tuple[ModelVersion, ...]
    equivalent: tuple[ModelVersion, ...]
    best_fit: ModelVersion
    fit_deficit_bits: float
    code_saving_bits: float
    equivalence_band_bits: float
    rule: str

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("a selection report needs at least one candidate")
        if self.winner not in self.candidates:
            raise ValueError("the winner must be one of the candidates")
        if self.best_fit not in self.candidates:
            raise ValueError("the best fit must be one of the candidates")
        if self.fit_deficit_bits < 0:
            raise ValueError(
                f"fit_deficit_bits must be >= 0, got {self.fit_deficit_bits}"
            )

    @property
    def rationale(self) -> str:
        """One-sentence audit trail for the selection."""
        if self.winner == self.best_fit:
            return (
                f"{self.winner.name} wins: best fit "
                f"({self.winner.held_out_mean_log_loss_bits:.4f} bits/item) and "
                f"shortest total code "
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
        """JSON-serializable form of the selection."""
        return {
            "winner": self.winner.name,
            "rule": self.rule,
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
    """Validate a candidate list: non-empty, unique names."""
    ordered = tuple(versions)
    if not ordered:
        raise ValueError("selecting among no versions is not a selection")
    names = [v.name for v in ordered]
    if len(set(names)) != len(names):
        raise ValueError(f"version names must be unique, got {names}")
    return ordered


def select_version(
    versions: Sequence[ModelVersion],
    equivalence_band_bits: float = DEFAULT_EQUIVALENCE_BAND_BITS,
) -> SelectionReport:
    """Select by code length at equal performance (the issue's rule).

    Versions within ``equivalence_band_bits`` (mean held-out log-loss
    bits per item) of the best fit count as equal performance; among
    them the minimum total description length wins. The report names
    the fit deficit the winner accepted and the code it saved, so a
    selection that trades fit for compression is visible, not silent.
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
        if v.held_out_mean_log_loss_bits <= best.held_out_mean_log_loss_bits
        + equivalence_band_bits
    )
    winner = _shortest_total(equivalent)
    return SelectionReport(
        winner=winner,
        candidates=ordered,
        equivalent=equivalent,
        best_fit=best,
        fit_deficit_bits=winner.held_out_mean_log_loss_bits
        - best.held_out_mean_log_loss_bits,
        code_saving_bits=float(best.description_bits - winner.description_bits),
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
    unbanded trade state that choice explicitly.
    """
    ordered = _require_versions(versions)
    best = _best_fit(ordered)
    winner = _shortest_total(ordered)
    return SelectionReport(
        winner=winner,
        candidates=ordered,
        equivalent=ordered,
        best_fit=best,
        fit_deficit_bits=winner.held_out_mean_log_loss_bits
        - best.held_out_mean_log_loss_bits,
        code_saving_bits=float(best.description_bits - winner.description_bits),
        equivalence_band_bits=float("inf"),
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
        if self.step < 0:
            raise ValueError(f"step must be >= 0, got {self.step}")
        if self.capacity_bits < 0:
            raise ValueError(f"capacity_bits must be >= 0, got {self.capacity_bits}")
        for label, value in (
            ("held_out_mean_log_loss_bits", self.held_out_mean_log_loss_bits),
            ("train_mean_log_loss_bits", self.train_mean_log_loss_bits),
        ):
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
    widest train/held-out gap inside the window. The reason names the
    window steps and both held-out bests, so the stop is auditable.
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
