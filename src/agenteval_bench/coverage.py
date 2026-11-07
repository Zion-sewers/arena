"""Coverage requirements before learning or ranking (arena#49).

The binding constraint is per-(state, action) coverage, not total count:
~50 observations per cell minimum (source: Perplexity sample-size
research, catalog B-47). Sparse cells are masked out. Conservative
fallback where data is thin.

Rules (from the issue's acceptance criteria, executable form):
* a minimum interaction count is enforced before any ranking — a judge
  is rankable only when EVERY one of its cells meets the per-cell floor;
* sparse regions fall back to a conservative default — a cell's reported
  rate is the Wilson 95% interval LOWER bound on its win rate (the
  established discipline of decision.py / validity.py / calibration.py),
  never the point estimate, so thin data is punished by the interval
  width, not by a hand-picked penalty number;
* a leaderboard over tiny samples is NOT a ranking — when no judge is
  rankable, :func:`require_ranking` refuses with a typed error naming
  the masked judges and their thinnest cells, and the CLI exits 1.

A cell is one (state, action) pair. In the alt-test bridge the state is
the left-out annotator (the reference population) and the action is the
judge under test; observations are the items the pair shares. The module
never invents observations: wins are recovered exactly from the
report's per-cell (rho_f, n) — rho_f is the mean of n binary wins, so
``round(rho_f * n)`` is exact, and a cell whose wins cannot be
recovered is a corrupt report and refuses loudly instead of guessing.
That binary-wins invariant lives in the runner (``w_f`` is 0.0/1.0)
and is stated on ``AnnotatorComparison.rho_f`` in alt_test/types.py.

Pairs the runner drops before the report — fewer than
``min_pair_items`` shared items — never reach this layer as cells at
all. They are not forgotten: the verdict carries ``m_annotators``, so
a judge with fewer comparisons than annotators is charged with that
many unreported sparse cells and masked, never ranked on its reported
pairs alone.

``run_alt_test`` is untouched: the coverage layer consumes its report
and never rewrites it. A masked judge is still SHOWN (with its
conservative diagnostic score and its thinnest cells named), never
silently dropped — the same holes-reported-never-dropped discipline as
judges.py. Masked judges simply never enter the ranked order.

Follow-up (named, not claimed): the same cell machinery can mask sparse
bins in calibration.py, where a bin below the floor is UNCALIBRATED
today but its observations still enter the ECE diagnostic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from agenteval_bench.decision import wilson_interval
from alt_test.types import AltTestConfig, AltTestReport

#: The per-cell floor from the source (catalog B-47): ~50 observations
#: per (state, action) cell minimum.
DEFAULT_MIN_PER_CELL = 50

#: The total floor, read from the alt-test config default rather than
#: restated: if the runner default ever changes, the coverage default
#: follows it instead of drifting apart (the calibration.py pattern).
DEFAULT_MIN_TOTAL = AltTestConfig().min_items

#: The closed set of cell statuses. SPARSE means "masked from ranking"
#: — it is never a rank, and it is distinct from "no data" (a cell with
#: n = 0 never reaches this module through the alt-test bridge: the
#: runner only reports pairs that share items. The generic
#: assess_observations path does accept n = 0, which is sparse.)
CellStatus = Literal["covered", "sparse"]


class RankingRefused(ValueError):
    """A leaderboard was asked to rank on too little coverage.

    Raised by :func:`require_ranking` when the total is below the floor
    or when no judge has every cell covered. A ValueError subclass so
    callers that already guard on value errors fail closed the same way.
    """


@dataclass(frozen=True)
class CoveragePolicy:
    """Policy for what counts as enough coverage to rank.

    ``min_per_cell`` is the minimum shared observations in a single
    (state, action) cell for the cell to count as measured.
    ``min_total`` is the minimum items in the dataset for any ranking
    at all — a dataset below this floor is not a ranking even when
    every cell in it happens to be covered.
    """

    min_per_cell: int = DEFAULT_MIN_PER_CELL
    min_total: int = DEFAULT_MIN_TOTAL

    def __post_init__(self) -> None:
        if self.min_per_cell < 2:
            raise ValueError(
                f"min_per_cell must be >= 2 (a comparison needs n >= 2), got {self.min_per_cell}"
            )
        if self.min_total < 1:
            raise ValueError(f"min_total must be positive, got {self.min_total}")


def default_coverage_policy() -> CoveragePolicy:
    """The source floor: 50 observations per cell, 50 items total."""
    return CoveragePolicy()


@dataclass(frozen=True)
class CellObservation:
    """One cell's raw counts: ``n`` shared observations, ``wins`` of them won.

    ``state`` is the reference population (an annotator in the alt-test
    bridge), ``action`` is the subject being ranked (a judge). Counts are
    validated at the boundary: a negative count or wins above n is a
    corrupt observation, refused before it can enter any score.
    """

    state: str
    action: str
    n: int
    wins: int

    def __post_init__(self) -> None:
        if not self.state:
            raise ValueError("cell state must be non-empty")
        if not self.action:
            raise ValueError("cell action must be non-empty")
        if self.n < 0:
            raise ValueError(f"cell n must be >= 0, got {self.n}")
        if not 0 <= self.wins <= self.n:
            raise ValueError(f"cell wins {self.wins} out of range for n={self.n}")


@dataclass(frozen=True)
class CellCoverage:
    """One cell's coverage verdict under a policy.

    ``status`` is derived from ``n`` against ``min_per_cell`` at
    construction, so a hand-built record claiming COVERED on a thin cell
    is refused before it can reach the ranking — the same
    status↔evidence consistency calibration.py enforces. The
    conservative rate is the Wilson 95% LOWER bound on the win rate;
    the point rate is carried for diagnosis only and never ranks.
    """

    state: str
    action: str
    n: int
    wins: int
    min_per_cell: int
    status: CellStatus

    def __post_init__(self) -> None:
        if not self.state:
            raise ValueError("cell state must be non-empty")
        if not self.action:
            raise ValueError("cell action must be non-empty")
        if self.n < 0:
            raise ValueError(f"cell n must be >= 0, got {self.n}")
        if not 0 <= self.wins <= self.n:
            raise ValueError(f"cell wins {self.wins} out of range for n={self.n}")
        if self.min_per_cell < 2:
            raise ValueError(f"min_per_cell must be >= 2, got {self.min_per_cell}")
        expected: CellStatus = "covered" if self.n >= self.min_per_cell else "sparse"
        if self.status != expected:
            raise ValueError(
                f"cell ({self.state!r}, {self.action!r}) with n={self.n} "
                f"is {expected} under min_per_cell={self.min_per_cell}, "
                f"not {self.status}"
            )

    @property
    def point_rate(self) -> float:
        """Win rate as a point estimate — diagnosis only, never ranks."""
        return self.wins / self.n if self.n else 0.0

    @property
    def conservative_rate(self) -> float:
        """Wilson 95% LOWER bound on the win rate — the conservative fallback.

        Thin data gets a wide interval and therefore a low lower bound by
        construction: 5/5 wins gives ~0.57, 50/50 gives ~0.93. A cell
        with no observations has no interval; its conservative rate is
        0.0, the most conservative value available. The bound is clamped
        to the point rate: the Wilson formula's centre and half-width
        cancel exactly at 0 wins in real arithmetic, but in floats the
        subtraction leaves dust (6.9e-18 at n=47), and a conservative
        fallback must never exceed the measured rate, not even by dust.
        """
        if self.n == 0:
            return 0.0
        lower, _upper = wilson_interval(self.wins, self.n)
        return min(lower, self.point_rate)


def assess_cell(state: str, action: str, n: int, wins: int, min_per_cell: int) -> CellCoverage:
    """Assess one cell against the per-cell floor."""
    status: CellStatus = "covered" if n >= min_per_cell else "sparse"
    return CellCoverage(
        state=state,
        action=action,
        n=n,
        wins=wins,
        min_per_cell=min_per_cell,
        status=status,
    )


@dataclass(frozen=True)
class JudgeCoverage:
    """One judge's coverage verdict: its cells, rankability, scores.

    ``rankable`` is True iff the judge has at least one cell, every
    cell is covered, and no pair went unreported — a single sparse cell
    masks the judge from the ranking, whatever its omega, and so does a
    single pair the runner dropped before the report (fewer than
    ``min_pair_items`` shared items): the judge was never measured on
    that cell at all, so ranking on the reported pairs alone would
    certify a judge on a partial population. ``unreported_cells`` is
    that count; it is 0 for a fully-reported judge.
    ``conservative_score`` is the mean Wilson lower bound over COVERED
    cells only (masked cells never enter a ranked score); it is None
    when the judge has no covered cell. ``diagnostic_score`` is the
    mean Wilson lower bound over ALL cells — masked cells enter at
    their (wide-interval) lower bounds, so a masked judge is still
    shown honestly, never hidden.
    """

    judge: str
    cells: tuple[CellCoverage, ...]
    unreported_cells: int = 0

    def __post_init__(self) -> None:
        if not self.judge:
            raise ValueError("judge coverage judge must be non-empty")
        if self.unreported_cells < 0:
            raise ValueError(f"unreported_cells must be >= 0, got {self.unreported_cells}")
        floors = {cell.min_per_cell for cell in self.cells}
        if len(floors) > 1:
            raise ValueError(
                f"judge {self.judge!r} mixes cells assessed under different floors {sorted(floors)}"
            )
        for cell in self.cells:
            if cell.action != self.judge:
                raise ValueError(
                    f"cell action {cell.action!r} does not belong to judge {self.judge!r}"
                )

    @property
    def rankable(self) -> bool:
        """True iff cells exist, every cell is covered, none unreported."""
        return (
            bool(self.cells)
            and self.unreported_cells == 0
            and all(c.status == "covered" for c in self.cells)
        )

    @property
    def covered_cells(self) -> tuple[CellCoverage, ...]:
        """Cells meeting the per-cell floor."""
        return tuple(c for c in self.cells if c.status == "covered")

    @property
    def masked_cells(self) -> tuple[CellCoverage, ...]:
        """Sparse cells: masked from the ranking, named, never dropped."""
        return tuple(c for c in self.cells if c.status == "sparse")

    @property
    def coverage_fraction(self) -> float:
        """Fraction of the judge's cells (reported + unreported) covered."""
        total = len(self.cells) + self.unreported_cells
        return len(self.covered_cells) / total if total else 0.0

    @property
    def thinnest_cell(self) -> CellCoverage | None:
        """The judge's sparsest cell (fewest observations), or None."""
        if not self.cells:
            return None
        return min(self.cells, key=lambda c: (c.n, c.state))

    @property
    def conservative_score(self) -> float | None:
        """Mean Wilson lower bound over covered cells; None if none covered."""
        covered = self.covered_cells
        if not covered:
            return None
        return sum(c.conservative_rate for c in covered) / len(covered)

    @property
    def diagnostic_score(self) -> float | None:
        """Mean Wilson lower bound over all cells; None if no cells."""
        if not self.cells:
            return None
        return sum(c.conservative_rate for c in self.cells) / len(self.cells)


@dataclass(frozen=True)
class RankedLeaderboard:
    """A coverage-gated leaderboard: ranked judges, masked judges, verdict.

    ``ranked`` holds the rankable judges in rank order (conservative
    score descending, judge name breaking ties for determinism).
    ``masked`` holds every other judge, in judge-name order, with the
    reason attached to each cell. ``is_ranking`` is True iff at least
    one judge is rankable AND the dataset meets the total floor — a
    leaderboard with no rankable judge, or on too few items in total,
    is not a ranking, it is a refused claim.
    """

    ranked: tuple[JudgeCoverage, ...]
    masked: tuple[JudgeCoverage, ...]
    n_items: int
    min_per_cell: int
    min_total: int

    def __post_init__(self) -> None:
        if self.n_items < 0:
            raise ValueError(f"n_items must be >= 0, got {self.n_items}")
        for judge in self.ranked:
            if not judge.rankable:
                raise ValueError(f"judge {judge.judge!r} is in ranked but is not rankable")
        for judge in self.masked:
            if judge.rankable:
                raise ValueError(f"judge {judge.judge!r} is in masked but is rankable")

    @property
    def is_ranking(self) -> bool:
        """True iff at least one judge is rankable and the total floor holds."""
        return bool(self.ranked) and self.n_items >= self.min_total

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form of the coverage verdict."""
        return {
            "is_ranking": self.is_ranking,
            "min_per_cell": self.min_per_cell,
            "min_total": self.min_total,
            "n_items": self.n_items,
            "ranked": [
                {
                    "judge": j.judge,
                    "conservative_score": j.conservative_score,
                    "covered_cells": len(j.covered_cells),
                    "total_cells": len(j.cells) + j.unreported_cells,
                }
                for j in self.ranked
            ],
            "masked": [
                {
                    "judge": j.judge,
                    "diagnostic_score": j.diagnostic_score,
                    "covered_cells": len(j.covered_cells),
                    "total_cells": len(j.cells) + j.unreported_cells,
                    "unreported_cells": j.unreported_cells,
                    "thinnest_cell": (
                        {
                            "state": j.thinnest_cell.state,
                            "n": j.thinnest_cell.n,
                        }
                        if j.thinnest_cell is not None
                        else None
                    ),
                }
                for j in self.masked
            ],
        }


def _recover_wins(judge: str, state: str, rho_f: float, n: int) -> int:
    """Recover a cell's integer win count from (rho_f, n).

    rho_f is the mean of n binary wins, so rho_f * n is an integer up to
    float error. A value that is not within 1e-9 of an integer means the
    report is corrupt (or was not produced by this runner); refusing
    loudly beats guessing a count into a ranking. Non-finite rho_f
    (inf/nan) refuses as ValueError too: ``round(inf)`` raises
    OverflowError, which no caller catches, so it is converted here.
    """
    if not math.isfinite(rho_f):
        raise ValueError(
            f"cannot recover wins for judge {judge!r} vs {state!r}: rho_f={rho_f} is not finite"
        )
    raw = rho_f * n
    try:
        wins = round(raw)
    except OverflowError as e:
        raise ValueError(
            f"cannot recover wins for judge {judge!r} vs {state!r}: "
            f"rho_f={rho_f} over n={n} overflows"
        ) from e
    if abs(raw - wins) > 1e-9:
        raise ValueError(
            f"cannot recover wins for judge {judge!r} vs {state!r}: "
            f"rho_f={rho_f} over n={n} is not an integer win count"
        )
    return wins


def assess_ranking(
    report: AltTestReport, policy: CoveragePolicy | None = None
) -> RankedLeaderboard:
    """Gate an alt-test report on per-cell coverage.

    Each (judge, annotator) comparison is one cell; the cell's n is the
    shared-item count and its wins are recovered from rho_f. A verdict
    with fewer comparisons than annotators has unreported cells: the
    runner dropped those pairs (fewer than min_pair_items shared
    items), and the judge is charged with one unreported sparse cell
    per missing comparison — it is masked, never ranked on its
    reported pairs alone. Judges with every cell covered and none
    unreported are ranked by conservative score; every other judge is
    masked with its thinnest cell named. The report itself is never
    modified.
    """
    policy = policy or default_coverage_policy()
    assessed: list[JudgeCoverage] = []
    for verdict in report.verdicts:
        cells = tuple(
            assess_cell(
                state=c.annotator,
                action=verdict.judge,
                n=c.n,
                wins=_recover_wins(verdict.judge, c.annotator, c.rho_f, c.n),
                min_per_cell=policy.min_per_cell,
            )
            for c in verdict.comparisons
        )
        unreported = max(0, verdict.m_annotators - len(verdict.comparisons))
        assessed.append(
            JudgeCoverage(judge=verdict.judge, cells=cells, unreported_cells=unreported)
        )
    ranked = tuple(
        sorted(
            (j for j in assessed if j.rankable),
            key=lambda j: (
                -(j.conservative_score if j.conservative_score is not None else 0.0),
                j.judge,
            ),
        )
    )
    masked = tuple(sorted((j for j in assessed if not j.rankable), key=lambda j: j.judge))
    return RankedLeaderboard(
        ranked=ranked,
        masked=masked,
        n_items=report.n_items,
        min_per_cell=policy.min_per_cell,
        min_total=policy.min_total,
    )


def require_ranking(board: RankedLeaderboard) -> RankedLeaderboard:
    """Return the board iff it is a ranking; refuse otherwise.

    Refuses when the dataset is below the total floor, and when no
    judge is rankable. The message names every masked judge and its
    thinnest cell, so the fix (more shared items per pair) is visible
    in the refusal itself.
    """
    if board.n_items < board.min_total:
        raise RankingRefused(
            f"not a ranking: {board.n_items} items is below the min_total={board.min_total} floor"
        )
    if not board.is_ranking:
        if not board.ranked and not board.masked:
            raise RankingRefused(
                "not a ranking: the report has no judges at all "
                f"(min_per_cell={board.min_per_cell})"
            )
        reasons = "; ".join(
            f"{j.judge} thinnest cell "
            f"{j.thinnest_cell.state if j.thinnest_cell is not None else '?'} "
            f"n={j.thinnest_cell.n if j.thinnest_cell is not None else 0} "
            f"(need >= {board.min_per_cell})"
            + (f", {j.unreported_cells} unreported pair(s)" if j.unreported_cells else "")
            for j in board.masked
        )
        raise RankingRefused(
            f"not a ranking: no judge has every cell covered "
            f"(min_per_cell={board.min_per_cell}); masked: {reasons}"
        )
    return board


def render_ranked(board: RankedLeaderboard) -> str:
    """Human-readable coverage-gated leaderboard."""
    lines = [
        (
            "coverage-gated ranking "
            f"(min_per_cell={board.min_per_cell} min_total={board.min_total} "
            f"n_items={board.n_items})"
        ),
        "",
        f"{'judge':<24}{'conservative':>13}{'cells':>8}  status",
    ]
    for judge in board.ranked:
        score = judge.conservative_score if judge.conservative_score is not None else 0.0
        cells = f"{len(judge.covered_cells)}/{len(judge.cells)}"
        lines.append(f"{judge.judge:<24}{score:>13.3f}{cells:>8}  RANKED")
    for judge in board.masked:
        score = judge.diagnostic_score if judge.diagnostic_score is not None else 0.0
        total = len(judge.cells) + judge.unreported_cells
        cells = f"{len(judge.covered_cells)}/{total}"
        thinnest = judge.thinnest_cell
        thin = f"thinnest {thinnest.state} n={thinnest.n}" if thinnest is not None else "no cells"
        unreported = (
            f", {judge.unreported_cells} unreported pair(s)" if judge.unreported_cells else ""
        )
        lines.append(f"{judge.judge:<24}{score:>13.3f}{cells:>8}  MASKED ({thin}{unreported})")
    if not board.is_ranking:
        lines += [
            "",
            (
                "NOT A RANKING: no judge has every cell covered. "
                "Collect more shared items per (judge, annotator) pair."
            ),
        ]
    return "\n".join(lines) + "\n"


def assess_observations(
    observations: Sequence[CellObservation], policy: CoveragePolicy | None = None
) -> tuple[CellCoverage, ...]:
    """Assess raw (state, action) counts — the generic learning gate.

    The same cell machinery without the alt-test bridge: callers that
    gate learning (not just ranking) pass their own per-cell counts and
    get back the covered/sparse verdict per cell. Sparse cells are the
    caller's to mask; this function never invents a fallback value.
    """
    policy = policy or default_coverage_policy()
    return tuple(
        assess_cell(
            state=o.state,
            action=o.action,
            n=o.n,
            wins=o.wins,
            min_per_cell=policy.min_per_cell,
        )
        for o in observations
    )
