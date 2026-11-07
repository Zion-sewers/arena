"""Typed contracts for the Alternative Annotator Test (alt-test) benchmark.

Implements the procedure of Calderon, Reichart & Dror, "The Alternative
Annotator Test for LLM-as-a-Judge" (arXiv:2501.10970, CC0-licensed;
reference implementation at github.com/nitaytech/AltTest — studied, not
forked). Per-annotator leave-one-out alignment, paired t-tests against a
cost-benefit margin epsilon, Benjamini-Yekutieli FDR control, winning
rate omega, and Average Advantage Probability rho.

Judge labels are RECORDED DATA, never live LLM calls: the benchmark reads
a dataset of items annotated by >=3 humans plus one label column per
judge under test, exactly like the replay command scores recorded agent
outputs without a live agent. This module makes no network calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

TaskType = Literal["discrete", "continuous", "text"]

#: Label values: category strings for discrete/text tasks, floats for continuous.
Label = str | float

#: Name of the multiple-comparison procedure used (paper-faithful).
FDR_PROCEDURE = "Benjamini-Yekutieli"

#: omega decision threshold: omega >= 0.5 justifies replacing annotators.
OMEGA_THRESHOLD = 0.5


@dataclass(frozen=True)
class AltTestItem:
    """One annotated item.

    annotator_labels maps annotator id -> label. Every item must carry
    labels from at least ``min_annotators`` annotators (default 3).
    judge_labels maps judge name -> that judge's recorded label for the
    item. A judge absent from an item is simply not scored on it.
    """

    id: str
    task: TaskType
    annotator_labels: Mapping[str, Label]
    judge_labels: Mapping[str, Label] = field(default_factory=dict)


@dataclass(frozen=True)
class AltTestConfig:
    """Knobs of the alt-test procedure."""

    epsilon: float = 0.1  #: cost-benefit margin; H0: rho_h - rho_f >= epsilon.
    #: Bounded to [0, 0.5] per the paper's recommended range: beyond 0.5 the
    #: margin exceeds the plausible advantage gap and every judge passes.
    q: float = 0.05  #: target false discovery rate for the BY procedure
    min_annotators: int = 3  #: minimum annotators per item
    min_items: int = 50  #: minimum items in the dataset
    min_pair_items: int = 10  #: minimum shared items per (judge, annotator) pair
    t_test_min_n: int = 30  #: below this n the Wilcoxon signed-rank path is used
    seed: int = 42  #: recorded for reproducibility; the test itself is deterministic

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon <= 0.5:
            raise ValueError(f"epsilon must be in [0, 0.5], got {self.epsilon}")
        if not 0.0 < self.q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {self.q}")
        if self.min_annotators < 3:
            raise ValueError("alt-test requires at least 3 annotators")
        if self.min_items < 1:
            raise ValueError("min_items must be positive")
        if self.min_pair_items < 2:
            raise ValueError(
                f"min_pair_items must be >= 2 (a test needs n >= 2), "
                f"got {self.min_pair_items}"
            )
        if self.t_test_min_n < 2:
            raise ValueError(f"t_test_min_n must be >= 2, got {self.t_test_min_n}")


@dataclass(frozen=True)
class AnnotatorComparison:
    """Alt-test outcome of one judge against one left-out annotator.

    ``rho_f`` is the mean of ``n`` binary wins (1.0 when the judge's
    alignment on an item is at least the annotator's, else 0.0), so
    ``round(rho_f * n)`` recovers the integer win count exactly. The
    coverage layer (arena#49) relies on this: it reconstructs per-cell
    wins from (rho_f, n) rather than carrying a second count. A future
    runner change to graded (non-binary) wins must update this contract
    first — the reconstruction refuses loudly instead of guessing.
    """

    annotator: str
    n: int  #: shared items used in this comparison
    rho_f: float  #: P(judge alignment >= annotator alignment)
    rho_h: float  #: P(annotator alignment >= judge alignment)
    p_value: float  #: one-sided p-value for H0: rho_h - rho_f >= epsilon
    rejected: bool  #: H0 rejected after BY-FDR control
    test_used: Literal["paired-t", "wilcoxon"]


@dataclass(frozen=True)
class JudgeVerdict:
    """Per-judge alt-test verdict — one row of the leaderboard."""

    judge: str
    rho: float  #: Average Advantage Probability = mean over annotators of rho_f
    omega: float  #: winning rate = fraction of annotators with rejected H0
    justified: bool  #: omega >= 0.5
    comparisons: tuple[AnnotatorComparison, ...]
    n_items: int
    m_annotators: int


@dataclass(frozen=True)
class AltTestReport:
    """Full benchmark report: the winning-rate leaderboard for LLM judges."""

    verdicts: tuple[JudgeVerdict, ...]  #: sorted by rho descending
    epsilon: float
    q: float
    fdr_procedure: str = FDR_PROCEDURE
    omega_threshold: float = OMEGA_THRESHOLD
    n_items: int = 0
    input_digest: str = ""
    seed: int = 42
    sim_name: str = "default_text_sim"  #: similarity used for the text task
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """JSON-serializable form of the report."""
        return {
            "leaderboard": [
                {
                    "judge": v.judge,
                    "rho": v.rho,
                    "omega": v.omega,
                    "justified": v.justified,
                    "n_items": v.n_items,
                    "m_annotators": v.m_annotators,
                    "comparisons": [
                        {
                            "annotator": c.annotator,
                            "n": c.n,
                            "rho_f": c.rho_f,
                            "rho_h": c.rho_h,
                            "p_value": c.p_value,
                            "rejected": c.rejected,
                            "test_used": c.test_used,
                        }
                        for c in v.comparisons
                    ],
                }
                for v in self.verdicts
            ],
            "epsilon": self.epsilon,
            "q": self.q,
            "fdr_procedure": self.fdr_procedure,
            "omega_threshold": self.omega_threshold,
            "n_items": self.n_items,
            "input_digest": self.input_digest,
            "seed": self.seed,
            "sim_name": self.sim_name,
            "notes": list(self.notes),
        }
