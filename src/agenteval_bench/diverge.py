"""Anti-mode-collapse: /diverge with 5 candidates + p_conventional (arena#47).

A generation step that forces diverse candidates before selection on
high-stakes choices. Mode collapse is when a generator returns the same
answer five times in different words and the selector then "chooses"
between paraphrases: the choice was already made by the mode.

Rules (from the issue's acceptance criteria, executable form):
* candidates are scored for genuine diversity, not paraphrases — pair
  similarity combines content-token overlap with hard-token overlap
  (numbers and digit-bearing tokens, the things a paraphrase preserves
  and a genuinely different answer changes); the set grade is the
  WEAKEST link, so one paraphrase pair fails the set;
* the selection records why the conventional answer won or lost — the
  Selection record carries both scores, the margin, a closed outcome
  and a reason built from the measured facts;
* single-sample generation is NOT sufficient for high-stakes choices —
  building a high-stakes set with fewer than ``n_candidates`` raises
  typed ``SingleSampleError``, and the high-stakes gate refuses a
  selection whose set was not diverse or whose declared
  ``p_conventional`` was overconfident against the measured modal share.

Measurement discipline (same as validity.py / calibration.py): the
declared ``p_conventional`` is a CLAIM checked against measurement —
the modal share (largest near-duplicate cluster / n) through the
Wilson 95% interval, never against the point estimate. A declared p
above the interval UPPER bound + tolerance is overconfident and blocks
the high-stakes gate; weak evidence (n=5, so intervals are wide)
yields CONSISTENT, never a false conviction.

Everything is stdlib-only and deterministic. The module makes no
network call and has no live generation transport to call: arena has
no live generation (README: "LLM-as-judge is planned, not
implemented", and the same holds for generation), so candidates are
RECORDED outputs supplied by the caller, like calibration
observations. Scores are caller-supplied measurements (e.g. from
DeterministicScorer); the module records and quotes them, it never
invents or vouches for them. A candidate with no supplied score has no
score — the loader refuses it rather than defaulting to 0.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from agenteval_bench.decision import wilson_interval

#: Function words stripped before content comparison. A paraphrase
#: rewords the glue; the content tokens are what must differ.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
        "not",
        "no",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "so",
        "such",
        "than",
        "too",
        "very",
        "just",
        "also",
        "into",
        "over",
        "under",
        "between",
        "through",
        "after",
        "before",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "what",
        "how",
        "why",
    }
)

#: Word-token pattern: lowercase alphanumerics. Deterministic, no
#: stemming, no embeddings — a stdlib-only similarity has to be
#: auditable by reading it, not by trusting a model.
_TOKEN_RE: re.Pattern[str] = re.compile(r"[a-z0-9]+")

#: Closed set of p_conventional check verdicts. CONSISTENT means the
#: declared p survives the interval check; it is not a claim the p is
#: right, only that the measurement does not refute it.
PCheckVerdict = Literal["consistent", "overconfident", "underconfident"]

#: Closed set of selection outcomes: did the conventional answer win,
#: lose, or tie on score (tie = the conventional candidate shares the
#: top score and the winner was picked by the diversity tie-break).
SelectionOutcome = Literal["conventional_won", "conventional_lost", "tie"]

#: Closed set of stakes. Only "high" triggers the single-sample
#: refusal; "low" still enforces the exactly-5 contract, it just
#: reports the shape failure as a plain ValueError.
Stakes = Literal["low", "high"]


class SingleSampleError(ValueError):
    """A high-stakes choice was given fewer than n_candidates.

    Single-sample generation is not sufficient for high-stakes choices:
    there is nothing to diverge from and nothing to select between.
    A ValueError subclass so callers that already guard on value
    errors fail closed the same way.
    """


class NotDiverseError(ValueError):
    """A selection was requested on a set that fails the diversity bar.

    A paraphrase set must never silently select: the choice would be
    between rewordings of one answer, which is mode collapse wearing a
    selection record. Re-diverge instead.
    """


class HighStakesRefused(ValueError):
    """A high-stakes decision was requested on an unfit selection.

    Raised by :func:`require_high_stakes_decision` when the selection
    is not diverse or its declared p_conventional was overconfident
    against the measured modal share.
    """


@dataclass(frozen=True)
class Candidate:
    """One recorded generation output offered to /diverge.

    ``text`` is the candidate answer as generated; ``score`` is a
    caller-supplied quality measurement for it (e.g. from
    DeterministicScorer). ``label`` is an optional caller name for the
    candidate, used in reports; it plays no role in grading.
    """

    text: str
    score: float
    label: str = ""

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("candidate text must be non-empty")
        if not math.isfinite(self.score):
            raise ValueError(f"candidate score must be finite, got {self.score!r}")


@dataclass(frozen=True)
class DivergePolicy:
    """Policy for what counts as a genuine divergence.

    ``n_candidates`` is the exact candidate count /diverge requires
    (5: the issue's number). ``max_pair_similarity`` is the highest
    pair similarity a set tolerates: any pair above it is a
    near-duplicate and fails the set. ``p_conventional_tolerance`` is
    how far the declared p_conventional may sit outside the Wilson 95%
    interval on the measured modal share before the p-check fails.
    """

    n_candidates: int = 5
    max_pair_similarity: float = 0.80
    p_conventional_tolerance: float = 0.15

    def __post_init__(self) -> None:
        if self.n_candidates < 2:
            raise ValueError(f"n_candidates must be >= 2, got {self.n_candidates}")
        if not 0.0 < self.max_pair_similarity < 1.0:
            raise ValueError(
                f"max_pair_similarity must be in (0, 1), got {self.max_pair_similarity}"
            )
        if not 0.0 <= self.p_conventional_tolerance < 1.0:
            raise ValueError(
                f"p_conventional_tolerance must be in [0, 1), got {self.p_conventional_tolerance}"
            )


def default_diverge_policy() -> DivergePolicy:
    """The default policy: 5 candidates, 0.80 pair cap, 0.15 p tolerance."""
    return DivergePolicy()


@dataclass(frozen=True)
class DivergeSet:
    """One /diverge call: 5 candidates plus the conventional claim.

    ``conventional_index`` names which candidate is the conventional
    answer (the one the generator would have returned on its own).
    ``p_conventional`` is the caller's declared estimate that the
    conventional answer is correct — a claim, checked later against
    the measured modal share, never trusted on its own.
    """

    prompt: str
    candidates: tuple[Candidate, ...]
    conventional_index: int
    p_conventional: float
    policy: DivergePolicy = field(default_factory=default_diverge_policy)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("diverge prompt must be non-empty")
        if len(self.candidates) != self.policy.n_candidates:
            raise ValueError(
                f"/diverge needs exactly {self.policy.n_candidates} "
                f"candidates, got {len(self.candidates)}"
            )
        if not 0 <= self.conventional_index < len(self.candidates):
            raise ValueError(
                f"conventional_index {self.conventional_index} out of "
                f"range for {len(self.candidates)} candidates"
            )
        if not math.isfinite(self.p_conventional):
            raise ValueError(f"p_conventional must be finite, got {self.p_conventional!r}")
        if not 0.0 <= self.p_conventional <= 1.0:
            raise ValueError(f"p_conventional must be in [0, 1], got {self.p_conventional}")


def build_diverge_set(
    prompt: str,
    candidates: Sequence[Candidate],
    conventional_index: int,
    p_conventional: float,
    stakes: Stakes = "low",
    policy: DivergePolicy | None = None,
) -> DivergeSet:
    """Build a DivergeSet, refusing too few candidates on high stakes.

    On high stakes a short candidate list raises typed
    :class:`SingleSampleError` (a single sample is not sufficient for
    a high-stakes choice); on low stakes the same short list is a
    plain shape ValueError from the DivergeSet constructor. More than
    n candidates is always a shape ValueError: /diverge is 5, not
    "at least 5".
    """
    policy = policy or default_diverge_policy()
    if len(candidates) < policy.n_candidates and stakes == "high":
        raise SingleSampleError(
            f"high-stakes choice needs {policy.n_candidates} candidates, "
            f"got {len(candidates)}: single-sample generation is not "
            "sufficient for high-stakes choices"
        )
    return DivergeSet(
        prompt=prompt,
        candidates=tuple(candidates),
        conventional_index=conventional_index,
        p_conventional=p_conventional,
        policy=policy,
    )


def content_tokens(text: str) -> frozenset[str]:
    """Lowercased word tokens minus stopwords: what the answer says."""
    return frozenset(t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS)


def hard_tokens(text: str) -> frozenset[str]:
    """Digit-bearing tokens: numbers, versions, identifiers with digits.

    A paraphrase preserves the numbers and names while rewording the
    glue around them; a genuinely different answer changes them. Hard
    tokens are the paraphrase anchor.
    """
    return frozenset(t for t in _TOKEN_RE.findall(text.lower()) if any(c.isdigit() for c in t))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard similarity; two empty sets are identical (1.0)."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def pair_similarity(left: str, right: str) -> float:
    """Similarity of two candidate texts in [0, 1].

    The max of content-token Jaccard and hard-token Jaccard when both
    sides carry hard tokens, else content-token Jaccard. Max, not mean:
    either signal can convict a paraphrase, and averaging would let a
    reworded text with the same numbers slip through on a low content
    score.
    """
    content = _jaccard(content_tokens(left), content_tokens(right))
    left_hard = hard_tokens(left)
    right_hard = hard_tokens(right)
    if not left_hard or not right_hard:
        return content
    return max(content, _jaccard(left_hard, right_hard))


@dataclass(frozen=True)
class PairSimilarity:
    """One pair's similarity measurement."""

    i: int
    j: int
    similarity: float
    near_duplicate: bool

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form of this pair measurement."""
        return {
            "i": self.i,
            "j": self.j,
            "similarity": self.similarity,
            "near_duplicate": self.near_duplicate,
        }


@dataclass(frozen=True)
class DiversityReport:
    """The diversity measurement of a whole candidate set.

    ``diverse`` is the grade: True only when NO pair is a
    near-duplicate (weakest link — one paraphrase pair fails the set).
    ``clusters`` are the connected components over near-duplicate
    edges; ``modal_cluster`` is the largest cluster (ties: lowest
    first index) and ``modal_share`` its size over n — the measured
    support for the answer the candidates converge on.
    """

    n: int
    pairs: tuple[PairSimilarity, ...]
    min_pair_diversity: float
    mean_pair_diversity: float
    diverse: bool
    clusters: tuple[tuple[int, ...], ...]
    modal_cluster: tuple[int, ...]
    modal_share: float
    max_pair_similarity: float

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form of this diversity measurement."""
        return {
            "n": self.n,
            "pairs": [p.to_dict() for p in self.pairs],
            "min_pair_diversity": self.min_pair_diversity,
            "mean_pair_diversity": self.mean_pair_diversity,
            "diverse": self.diverse,
            "clusters": [list(c) for c in self.clusters],
            "modal_cluster": list(self.modal_cluster),
            "modal_share": self.modal_share,
            "max_pair_similarity": self.max_pair_similarity,
        }


def _clusters(n: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Connected components over near-duplicate edges (union-find)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    grouped: dict[int, list[int]] = {}
    for x in range(n):
        grouped.setdefault(find(x), []).append(x)
    return sorted((sorted(members) for members in grouped.values()), key=lambda m: m[0])


def analyze_diversity(diverge_set: DivergeSet) -> DiversityReport:
    """Score a candidate set for genuine diversity, not paraphrases."""
    texts = [c.text for c in diverge_set.candidates]
    n = len(texts)
    pairs: list[PairSimilarity] = []
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = pair_similarity(texts[i], texts[j])
            near = sim > diverge_set.policy.max_pair_similarity
            pairs.append(PairSimilarity(i=i, j=j, similarity=sim, near_duplicate=near))
            if near:
                edges.append((i, j))
    diversities = [1.0 - p.similarity for p in pairs]
    clusters = _clusters(n, edges)
    modal = max(clusters, key=lambda c: (len(c), -c[0]))
    return DiversityReport(
        n=n,
        pairs=tuple(pairs),
        min_pair_diversity=min(diversities),
        mean_pair_diversity=sum(diversities) / len(diversities),
        diverse=not edges,
        clusters=tuple(tuple(c) for c in clusters),
        modal_cluster=tuple(modal),
        modal_share=len(modal) / n,
        max_pair_similarity=diverge_set.policy.max_pair_similarity,
    )


@dataclass(frozen=True)
class PConventionalCheck:
    """Declared p_conventional checked against the measured modal share.

    ``verdict`` is OVERCONFIDENT only when the declared p sits above
    the Wilson 95% UPPER bound on the modal share + tolerance, and
    UNDERCONFIDENT only when it sits below the LOWER bound - tolerance.
    With n=5 the interval is wide, so weak evidence yields CONSISTENT:
    the check convicts only a p the measurement actually refutes.
    ``conventional_in_modal`` is False when the declared conventional
    candidate is not in the modal cluster — the candidates converge on
    a different answer than the one declared conventional, recorded
    here and in the selection reason, never smoothed over.
    """

    declared_p: float
    modal_share: float
    agreement_interval: tuple[float, float]
    verdict: PCheckVerdict
    conventional_in_modal: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form of this p_conventional check."""
        return {
            "declared_p": self.declared_p,
            "modal_share": self.modal_share,
            "agreement_interval": list(self.agreement_interval),
            "verdict": self.verdict,
            "conventional_in_modal": self.conventional_in_modal,
            "notes": list(self.notes),
        }


def check_p_conventional(diverge_set: DivergeSet, report: DiversityReport) -> PConventionalCheck:
    """Check the declared p_conventional against the measured modal share."""
    modal_size = len(report.modal_cluster)
    interval = wilson_interval(modal_size, report.n)
    tolerance = diverge_set.policy.p_conventional_tolerance
    p = diverge_set.p_conventional
    if p > interval[1] + tolerance:
        verdict: PCheckVerdict = "overconfident"
    elif p < interval[0] - tolerance:
        verdict = "underconfident"
    else:
        verdict = "consistent"
    in_modal = diverge_set.conventional_index in report.modal_cluster
    notes: list[str] = []
    if not in_modal:
        notes.append(
            f"conventional candidate {diverge_set.conventional_index} is not "
            f"in the modal cluster {list(report.modal_cluster)}: the "
            "candidates converge on a different answer than the one "
            "declared conventional"
        )
    if verdict == "overconfident":
        notes.append(
            f"declared p_conventional {p:.2f} exceeds the Wilson 95% "
            f"upper bound {interval[1]:.2f} on the measured modal share "
            f"{report.modal_share:.2f} by more than the "
            f"{tolerance:.2f} tolerance"
        )
    elif verdict == "underconfident":
        notes.append(
            f"declared p_conventional {p:.2f} sits below the Wilson 95% "
            f"lower bound {interval[0]:.2f} on the measured modal share "
            f"{report.modal_share:.2f} by more than the "
            f"{tolerance:.2f} tolerance"
        )
    return PConventionalCheck(
        declared_p=p,
        modal_share=report.modal_share,
        agreement_interval=interval,
        verdict=verdict,
        conventional_in_modal=in_modal,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class Selection:
    """The recorded selection: why the conventional answer won or lost.

    ``reason`` is built from the measured facts only: winner score vs
    conventional score, the margin, whether the winner sits in the
    modal cluster, the diversity grade and the p-check verdict. The
    record IS the explanation — there is no prose beside it to drift
    from it.
    """

    winner_index: int
    conventional_index: int
    outcome: SelectionOutcome
    winner_score: float
    conventional_score: float
    margin: float
    reason: str
    diverse: bool
    p_verdict: PCheckVerdict

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form of this selection record."""
        return {
            "winner_index": self.winner_index,
            "conventional_index": self.conventional_index,
            "outcome": self.outcome,
            "winner_score": self.winner_score,
            "conventional_score": self.conventional_score,
            "margin": self.margin,
            "reason": self.reason,
            "diverse": self.diverse,
            "p_verdict": self.p_verdict,
        }


def select(
    diverge_set: DivergeSet,
    report: DiversityReport,
    p_check: PConventionalCheck,
) -> Selection:
    """Select the winning candidate from a diverse set.

    Winner is the argmax score. Ties break toward the candidate in the
    SMALLEST cluster — the most distinctive answer wins a tie, which is
    the deliberate anti-mode-collapse tie-break — then the lowest
    index, so the choice is deterministic with no RNG. A set that
    fails the diversity bar raises :class:`NotDiverseError`: a
    paraphrase set never silently selects.
    """
    if not report.diverse:
        dupes = [p for p in report.pairs if p.near_duplicate]
        raise NotDiverseError(
            "cannot select from a non-diverse set: "
            + ", ".join(f"candidates {p.i}/{p.j} similarity {p.similarity:.2f}" for p in dupes)
            + "; re-diverge instead of choosing between paraphrases"
        )
    scores = [c.score for c in diverge_set.candidates]
    top = max(scores)
    tied = [i for i, s in enumerate(scores) if s == top]
    cluster_size = {idx: len(cluster) for cluster in report.clusters for idx in cluster}
    winner = min(tied, key=lambda i: (cluster_size[i], i))
    conv = diverge_set.conventional_index
    conv_score = scores[conv]
    if winner == conv:
        outcome: SelectionOutcome = "conventional_won"
    elif conv in tied:
        outcome = "tie"
    else:
        outcome = "conventional_lost"
    margin = scores[winner] - conv_score
    winner_in_modal = winner in report.modal_cluster
    if outcome == "conventional_won":
        others = [s for i, s in enumerate(scores) if i != conv]
        runner_up = max(others)
        why = (
            f"conventional candidate {conv} won on score "
            f"({conv_score:.2f}, margin +{conv_score - runner_up:.2f} "
            "over the runner-up)"
        )
    elif outcome == "tie":
        why = (
            f"conventional candidate {conv} tied on top score "
            f"({conv_score:.2f}); winner candidate {winner} picked by "
            "the diversity tie-break (smallest cluster, then lowest index)"
        )
    else:
        why = (
            f"conventional candidate {conv} lost on score "
            f"({conv_score:.2f} vs winner {scores[winner]:.2f}, "
            f"margin {margin:.2f})"
        )
    modal_note = (
        "winner is in the modal cluster"
        if winner_in_modal
        else "winner is outside the modal cluster"
    )
    reason = (
        f"{why}. {modal_note} (modal cluster "
        f"{list(report.modal_cluster)}, share {report.modal_share:.2f}). "
        f"Set diverse (min pairwise diversity "
        f"{report.min_pair_diversity:.2f}). p_conventional "
        f"{p_check.declared_p:.2f} is {p_check.verdict} against the "
        f"measured modal share {report.modal_share:.2f}."
    )
    return Selection(
        winner_index=winner,
        conventional_index=conv,
        outcome=outcome,
        winner_score=scores[winner],
        conventional_score=conv_score,
        margin=margin,
        reason=reason,
        diverse=report.diverse,
        p_verdict=p_check.verdict,
    )


def may_decide_high_stakes(selection: Selection) -> bool:
    """True only for a diverse selection with a non-overconfident p.

    A high-stakes decision needs both: genuine diversity (the choice
    was between different answers) and a p_conventional the
    measurement does not refute.
    """
    return selection.diverse and selection.p_verdict != "overconfident"


def require_high_stakes_decision(selection: Selection) -> Selection:
    """Return the selection, or refuse it for a high-stakes decision.

    The gate takes ONLY the selection record. A caller that wants to
    decide on high stakes must pass a selection; a bare candidate, a
    single sample, or a score has no path through this function.
    """
    if not may_decide_high_stakes(selection):
        raise HighStakesRefused(
            "high-stakes decision refused: "
            + (
                "selection is not diverse"
                if not selection.diverse
                else "p_conventional was overconfident against the measured modal share"
            )
            + f" (outcome {selection.outcome}, p verdict "
            f"{selection.p_verdict})"
        )
    return selection


_DIVERGE_KEYS: frozenset[str] = frozenset(
    {"prompt", "stakes", "conventional_index", "p_conventional", "candidates"}
)
_CANDIDATE_KEYS: frozenset[str] = frozenset({"text", "score", "label"})


def candidates_from_json(
    text: str,
) -> tuple[str, list[Candidate], int, float, Stakes]:
    """Parse a /diverge input file, failing closed on bad shape.

    Expected shape: a mapping with ``prompt`` (str), ``candidates``
    (list of ``{"text", "score", "label"?}``), ``conventional_index``
    (int), ``p_conventional`` (number in [0, 1]) and optional
    ``stakes`` ("low" | "high", default "low"). Unknown keys, missing
    fields, wrong types and non-finite numbers are all refused — a
    malformed divergence file is unusable input, never a silent
    default.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid diverge JSON: {e}") from None
    if not isinstance(data, Mapping):
        raise TypeError(f"diverge input must be a mapping, got {type(data).__name__}")
    unknown = set(data) - _DIVERGE_KEYS
    if unknown:
        raise ValueError(f"diverge input has unknown keys: {sorted(unknown)}")
    for key in ("prompt", "candidates", "conventional_index", "p_conventional"):
        if key not in data:
            raise ValueError(f"diverge input is missing required key {key!r}")
    prompt = data["prompt"]
    if not isinstance(prompt, str):
        raise TypeError(f"diverge prompt must be a string, got {type(prompt).__name__}")
    stakes_raw = data.get("stakes", "low")
    if stakes_raw not in ("low", "high"):
        raise ValueError(f"diverge stakes must be 'low' or 'high', got {stakes_raw!r}")
    stakes: Stakes = "high" if stakes_raw == "high" else "low"
    conv_raw = data["conventional_index"]
    if isinstance(conv_raw, bool) or not isinstance(conv_raw, int):
        raise TypeError(f"diverge conventional_index must be an int, got {conv_raw!r}")
    p_raw = data["p_conventional"]
    if isinstance(p_raw, bool) or not isinstance(p_raw, (int, float)):
        raise TypeError(f"diverge p_conventional must be a number, got {p_raw!r}")
    p_conventional = float(p_raw)
    raw_candidates = data["candidates"]
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        raise TypeError(f"diverge candidates must be a list, got {type(raw_candidates).__name__}")
    candidates: list[Candidate] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise TypeError(f"candidate {index} must be a mapping, got {type(raw).__name__}")
        unknown_c = set(raw) - _CANDIDATE_KEYS
        if unknown_c:
            raise ValueError(f"candidate {index} has unknown keys: {sorted(unknown_c)}")
        if "text" not in raw or "score" not in raw:
            raise ValueError(f"candidate {index} needs 'text' and 'score' keys")
        text_raw = raw["text"]
        score_raw = raw["score"]
        if not isinstance(text_raw, str):
            raise TypeError(
                f"candidate {index} text must be a string, got {type(text_raw).__name__}"
            )
        if isinstance(score_raw, bool) or not isinstance(score_raw, (int, float)):
            raise TypeError(f"candidate {index} score must be a number, got {score_raw!r}")
        label_raw = raw.get("label", "")
        if not isinstance(label_raw, str):
            raise TypeError(
                f"candidate {index} label must be a string, got {type(label_raw).__name__}"
            )
        candidates.append(Candidate(text=text_raw, score=float(score_raw), label=label_raw))
    return prompt, candidates, conv_raw, p_conventional, stakes
