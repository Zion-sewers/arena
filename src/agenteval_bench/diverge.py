"""Anti-mode-collapse: /diverge with 5 candidates + p_conventional (arena#47).

A checking step that refuses paraphrase sets before selection on
high-stakes choices. There is no live generation transport: candidates
are recorded outputs supplied by the caller, and /diverge checks them,
it does not generate diverse candidates. Mode collapse is when a
generator returns the same answer five times in different words and
the selector then "chooses" between paraphrases: the choice was
already made by the mode.

Rules (from the issue's acceptance criteria, executable form):
* candidates are scored for token-level diversity (near-duplicate
  detection, not semantic diversity) — pair similarity is
  content-token Jaccard, with a hard-token (numbers and digit-bearing
  tokens) Jaccard that applies only when BOTH sides carry at least
  three hard tokens: Jaccard on tiny hard-token sets has almost no
  gradation (two shared numbers score 1.0), so a smaller anchor
  convicts genuinely different answers that merely share a number.
  The set grade is the WEAKEST link, so one near-verbatim pair fails
  the set. What this measures is near-duplicate detection at token
  level, NOT semantic diversity: a fluent synonym rewrite that
  replaces the content words passes, and that limit is named in
  docs/DIVERGE.md rather than hidden;
* the selection records why the conventional answer won or lost — the
  Selection record carries both scores, the full score vector (so a
  non-argmax winner is refused at construction), the runner-up score,
  the margin (winner over runner-up), a closed outcome and a reason
  built from the measured facts; ties break toward the candidate with
  the LOWEST mean pairwise similarity (least like the others), then
  the lowest index — and when the conventional candidate ties at the
  top and wins the tie-break, the outcome is ``tie``, not a win on
  score, and the reason names the tie-break. Every record validates
  its own invariants in ``__post_init__`` (a forged margin or outcome
  is refused at construction), and every report, check and selection
  carries the fingerprint of the candidate set it was computed from —
  a report from a different set is refused, never silently mixed in.
  The boundary is stated plainly: a hand-built record that is
  internally consistent under the canonical policy is
  indistinguishable from a pipeline-built one, and the tie-break
  choice among tied top scorers is trusted rather than re-derived —
  the record does not carry the per-candidate means the tie-break
  reads;
* single-sample generation is NOT sufficient for high-stakes choices —
  building a high-stakes set with fewer than ``n_candidates`` raises
  typed ``SingleSampleError``, a high-stakes policy must ask for
  exactly 5, and the high-stakes gate itself refuses any selection
  whose set covers anything other than 5 candidates, whose set was
  not diverse, whose declared ``p_conventional`` was overconfident
  against the measured share of the conventional answer's own cluster,
  or whose diversity and p-check grades were computed under anything
  but the canonical policy (max_pair_similarity 0.80, p tolerance
  0.15): a lax policy cannot launder a set through the gate. The gate
  enforces the count and the policy from the selection record, so the
  obligation survives build time: a short set built low-stakes
  cannot decide high stakes, and neither can a set graded under lax
  thresholds.

Measurement discipline (same as validity.py / calibration.py): the
declared ``p_conventional`` is a CLAIM checked against measurement.
``p_conventional`` is the declared probability that the conventional
answer is the answer the divergence CONVERGES on (the modal answer),
not a probability that it is correct — the only quantity the
candidate set measures is convergence, so only a convergence claim
can be checked against it. The check runs the declared p against the
Wilson 95% interval on the conventional cluster's share (size of the
cluster containing the conventional candidate / n), never the point
estimate. A declared p above the interval UPPER bound + tolerance is
overconfident and blocks the high-stakes gate; weak evidence (n=5,
so intervals are wide) yields CONSISTENT, never a false conviction —
and when the conventional cluster is large enough that no p can be
refuted, the check says so in its notes instead of pretending to
have tested anything.

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

import hashlib
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
    against the measured share of the conventional answer's cluster.
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
    """Policy for what counts as a divergence at token level.

    A set is diverse when no pair of candidates is a near-verbatim
    paraphrase under the thresholds below — token-level divergence,
    not semantic divergence. Only the canonical defaults (0.80 pair
    cap, 0.15 p tolerance) may decide high stakes: a set graded under
    a laxer policy selects fine, but the high-stakes gate refuses it.

    ``n_candidates`` is the exact candidate count /diverge requires
    (5: the issue's number). ``max_pair_similarity`` is the highest
    pair similarity a set tolerates: any pair above it is a
    near-duplicate and fails the set. ``p_conventional_tolerance`` is
    how far the declared p_conventional may sit outside the Wilson 95%
    interval on the measured share of the conventional answer's
    cluster before the p-check fails.
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
    conventional answer is the answer the divergence converges on
    (the modal answer) — a claim, checked later against the measured
    share of the conventional answer's own cluster, never trusted on
    its own, and never a claim that the answer is correct.
    """

    prompt: str
    candidates: tuple[Candidate, ...]
    conventional_index: int
    p_conventional: float
    policy: DivergePolicy = field(default_factory=default_diverge_policy)

    def fingerprint(self) -> str:
        """Fingerprint binding this set to the reports built from it.

        SHA-256 over the prompt, every candidate (text, score, label),
        the conventional index, the declared p and the policy. A
        DiversityReport, PConventionalCheck or Selection carries this
        fingerprint, and the pipeline steps refuse a measurement whose
        fingerprint does not match the set in hand — so a report from
        a different candidate set can never silently mix into a
        selection from this one. Numbers are coerced to float (and
        -0.0 to 0.0) before hashing, so int and float spellings of the
        same set — score 1 vs 1.0, p 1 vs 1.0 — fingerprint alike.
        """
        payload = {
            "prompt": self.prompt,
            "candidates": [
                {
                    "text": c.text,
                    "score": _fp_number(c.score),
                    "label": c.label,
                }
                for c in self.candidates
            ],
            "conventional_index": self.conventional_index,
            "p_conventional": _fp_number(self.p_conventional),
            "policy": {
                "n_candidates": self.policy.n_candidates,
                "max_pair_similarity": _fp_number(self.policy.max_pair_similarity),
                "p_conventional_tolerance": _fp_number(self.policy.p_conventional_tolerance),
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

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


def _fp_number(value: float) -> float:
    """Canonical float for fingerprinting.

    ``json.dumps(1)`` is ``"1"`` but ``json.dumps(1.0)`` is ``"1.0"``,
    so int and float spellings of the same set would fingerprint
    differently. Coercing to float (and -0.0 to 0.0) makes the
    fingerprint a function of the set, not of its spelling.
    """
    value = float(value)
    return 0.0 if value == 0 else value


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
    "at least 5". A high-stakes policy must ask for exactly 5: a
    policy that shrinks the count to 2 would let a 2-candidate set
    through the high-stakes gate, which defeats the issue's "5
    candidates" contract on the library seam, not just the CLI.
    ``stakes`` is a closed pair: anything else is a caller error,
    never silently treated as low — a typo'd "High" on a short list
    must not skip the typed SingleSampleError.
    """
    if stakes not in ("low", "high"):
        raise ValueError(f"diverge stakes must be 'low' or 'high', got {stakes!r}")
    policy = policy or default_diverge_policy()
    if stakes == "high" and policy.n_candidates != 5:
        raise SingleSampleError(
            f"high-stakes /diverge needs exactly 5 candidates per "
            f"arena#47, policy asks for {policy.n_candidates}"
        )
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
    """Jaccard similarity; two empty sets share no content (0.0).

    Scoring two content-free candidates as identical (1.0) convicted
    glue-only pairs ("the" vs "a") as near-duplicates: they share no
    content words, so there is nothing to be a paraphrase of.
    """
    if not left and not right:
        return 0.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def pair_similarity(left: str, right: str) -> float:
    """Similarity of two candidate texts in [0, 1].

    Content-token Jaccard, with hard-token Jaccard taking the max
    only when BOTH sides carry at least three hard tokens. Max, not
    mean: either signal can convict a paraphrase, and averaging would
    let a reworded text with the same numbers slip through on a low
    content score. The three-hard-token floor is load-bearing:
    Jaccard on tiny hard-token sets has almost no gradation, so a
    single shared number ("Retry after 30 seconds" vs "Delete after
    30 days") — and even two shared numbers — used to score two
    genuinely different answers as identical (1.0) and refuse the
    set. Shared numbers are paraphrase evidence only in bulk; with
    fewer hard tokens, content Jaccard alone decides. Residual,
    disclosed in docs/DIVERGE.md: two genuinely different answers
    that share three numbers can still be refused. Two candidates
    with no content tokens at all share no content (similarity 0.0
    for that signal), so distinct glue-only pairs are not convicted
    as paraphrases — but texts identical after normalization are
    always maximally similar (1.0): five copies of one answer are a
    paraphrase set, even when the answer is only glue, and case,
    surrounding whitespace or punctuation do not make a copy a
    different answer.
    """
    # Identity is judged after the function's own normalization
    # (lowercased word tokens): "the" vs "The", "the " and "the."
    # are the same answer five times, not five diverse candidates.
    # Distinct content-free texts ("the" vs "a") stay 0.0: their
    # token sequences differ, so they fall through to the Jaccard
    # signals below.
    if _TOKEN_RE.findall(left.lower()) == _TOKEN_RE.findall(right.lower()):
        return 1.0
    content = _jaccard(content_tokens(left), content_tokens(right))
    left_hard = hard_tokens(left)
    right_hard = hard_tokens(right)
    if len(left_hard) < 3 or len(right_hard) < 3:
        return content
    return max(content, _jaccard(left_hard, right_hard))


@dataclass(frozen=True)
class PairSimilarity:
    """One pair's similarity measurement."""

    i: int
    j: int
    similarity: float
    near_duplicate: bool

    def __post_init__(self) -> None:
        if not 0 <= self.i < self.j:
            raise ValueError(f"pair indices must satisfy 0 <= i < j, got i={self.i}, j={self.j}")
        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError(f"pair similarity must be in [0, 1], got {self.similarity}")

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
    ``mean_similarity_by_candidate`` is each candidate's mean pair
    similarity against the other four: the tie-break reads it, so a
    tie is decided by measured distinctiveness, not by index alone.
    ``set_fingerprint`` binds this report to the candidate set it was
    computed from; a pipeline step refuses a report whose fingerprint
    does not match the set in hand. ``max_pair_similarity`` is the
    threshold the pairs were graded under, carried so a pipeline step
    can refuse a report graded under a different policy than the
    set's: the fingerprint already binds the policy, but the threshold
    is what the grades mean, so it must be visible on the record
    rather than hidden inside the fingerprint.

    The record validates its own invariants — and it recomputes
    them from the pairs, it does not trust the flags: the
    near-duplicate flags must match similarity vs threshold, the
    clusters must be the connected components over the near-duplicate
    edges, the modal cluster must be the largest (ties: lowest first
    index), and the min/mean diversities and per-candidate means must
    match the pairs. A hand-built report that claims ``diverse``
    while holding a near-duplicate pair, or whose clusters do not
    partition the candidates, or whose modal share does not match
    its modal cluster, is refused at construction — and so is one
    whose tie-break means are forged, since ``select()`` reads them.
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
    set_fingerprint: str
    mean_similarity_by_candidate: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError(f"diversity report needs n >= 2, got {self.n}")
        if len(self.pairs) != self.n * (self.n - 1) // 2:
            raise ValueError(
                f"diversity report for n={self.n} needs "
                f"{self.n * (self.n - 1) // 2} pairs, got {len(self.pairs)}"
            )
        if len(self.mean_similarity_by_candidate) != self.n:
            raise ValueError(
                "mean similarity needs one entry per candidate, got "
                f"{len(self.mean_similarity_by_candidate)} for n={self.n}"
            )
        for name, value in (
            ("min_pair_diversity", self.min_pair_diversity),
            ("mean_pair_diversity", self.mean_pair_diversity),
            ("modal_share", self.modal_share),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not all(0.0 <= s <= 1.0 for s in self.mean_similarity_by_candidate):
            raise ValueError("per-candidate mean similarities must be in [0, 1]")
        if not 0.0 < self.max_pair_similarity < 1.0:
            raise ValueError(
                f"max_pair_similarity must be in (0, 1), got {self.max_pair_similarity}"
            )
        # Pair coverage: exactly one pair per (i, j), i < j.
        expected_pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        if sorted((p.i, p.j) for p in self.pairs) != expected_pairs:
            raise ValueError("pairs must cover every (i, j) with i < j exactly once")
        # near_duplicate flags are recomputed from similarity vs
        # threshold, never trusted: a pair at 0.99 similarity cannot
        # claim to be a non-duplicate, and a pair at 0.10 cannot claim
        # to be one.
        for pair in self.pairs:
            if pair.near_duplicate != (pair.similarity > self.max_pair_similarity):
                raise ValueError(
                    f"pair {pair.i}/{pair.j} near_duplicate flag contradicts "
                    f"its similarity {pair.similarity:.2f} vs threshold "
                    f"{self.max_pair_similarity:.2f}"
                )
        if self.diverse != (not any(p.near_duplicate for p in self.pairs)):
            raise ValueError(
                "diverse flag contradicts the pair measurements: a set "
                "with a near-duplicate pair cannot be diverse"
            )
        flat = sorted(i for cluster in self.clusters for i in cluster)
        if flat != list(range(self.n)):
            raise ValueError("clusters must partition the candidate indices")
        # Clusters must be the connected components over the
        # near-duplicate edges, and the modal cluster the largest
        # (ties: lowest first index) — recomputed, not trusted.
        edges = [(p.i, p.j) for p in self.pairs if p.near_duplicate]
        expected_clusters = _clusters(self.n, edges)
        if sorted(tuple(c) for c in self.clusters) != sorted(tuple(c) for c in expected_clusters):
            raise ValueError(
                "clusters must be the connected components over the near-duplicate edges"
            )
        expected_modal = max(expected_clusters, key=lambda c: (len(c), -c[0]))
        if tuple(self.modal_cluster) != tuple(expected_modal):
            raise ValueError(
                f"modal_cluster must be the largest cluster {expected_modal}, "
                f"got {list(self.modal_cluster)}"
            )
        if self.modal_cluster not in self.clusters:
            raise ValueError("modal_cluster must be one of the clusters")
        if abs(self.modal_share - len(self.modal_cluster) / self.n) > 1e-12:
            raise ValueError("modal_share must equal len(modal_cluster) / n")
        # Min/mean diversity and per-candidate means are recomputed
        # from the pairs: the tie-break reads the means, so forged
        # means must not steer a tie.
        similarities = [p.similarity for p in self.pairs]
        if abs(self.min_pair_diversity - min(1.0 - s for s in similarities)) > 1e-12:
            raise ValueError("min_pair_diversity must match the pair similarities")
        if (
            abs(self.mean_pair_diversity - sum(1.0 - s for s in similarities) / len(similarities))
            > 1e-12
        ):
            raise ValueError("mean_pair_diversity must match the pair similarities")
        for i in range(self.n):
            own = [p.similarity for p in self.pairs if p.i == i or p.j == i]
            if abs(self.mean_similarity_by_candidate[i] - sum(own) / len(own)) > 1e-12:
                raise ValueError(f"mean similarity for candidate {i} must match the pairs")
        if not self.set_fingerprint:
            raise ValueError("set_fingerprint must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form of this diversity measurement."""
        return {
            "n": self.n,
            "pairs": [p.to_dict() for p in self.pairs],
            "min_pair_diversity": self.min_pair_diversity,
            "mean_pair_diversity": self.mean_pair_diversity,
            "diverse": self.diverse,
            "diversity_scope": "token-level near-duplicate",
            "clusters": [list(c) for c in self.clusters],
            "modal_cluster": list(self.modal_cluster),
            "modal_share": self.modal_share,
            "max_pair_similarity": self.max_pair_similarity,
            "set_fingerprint": self.set_fingerprint,
            "mean_similarity_by_candidate": list(self.mean_similarity_by_candidate),
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
    """Score a candidate set for token-level diversity, not paraphrases."""
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
    mean_by_candidate = tuple(
        sum(p.similarity for p in pairs if p.i == i or p.j == i) / (n - 1) for i in range(n)
    )
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
        set_fingerprint=diverge_set.fingerprint(),
        mean_similarity_by_candidate=mean_by_candidate,
    )


@dataclass(frozen=True)
class PConventionalCheck:
    """Declared p_conventional checked against the measured convergence.

    ``p_conventional`` is the declared probability that the
    conventional answer is the answer the divergence converges on.
    The check runs it against the Wilson 95% interval on
    ``conventional_share`` — the share of the cluster containing the
    conventional candidate — NOT the modal share: when the
    conventional answer is a minority, the modal share belongs to a
    rival answer, and validating confidence in the conventional
    answer against a rival's support licenses exactly the
    overconfidence the check exists to catch.

    ``verdict`` is OVERCONFIDENT only when the declared p sits above
    the Wilson 95% UPPER bound on the conventional share + tolerance,
    and UNDERCONFIDENT only when it sits below the LOWER bound -
    tolerance. With n=5 the interval is wide, so weak evidence yields
    CONSISTENT: the check convicts only a p the measurement actually
    refutes. When the conventional cluster is large enough that the
    upper bound + tolerance reaches 1.0, no declared p can be
    refuted, and the notes say so instead of pretending the check
    tested anything.
    ``conventional_in_modal`` is False when the declared conventional
    candidate is not in the modal cluster — the candidates converge on
    a different answer than the one declared conventional, recorded
    here and in the selection reason, never smoothed over.
    ``tolerance`` and ``n`` are carried so the verdict is checkable
    from the record alone: the verdict must follow from the interval
    ± tolerance, the interval must be the Wilson 95% interval on the
    conventional share, and the conventional share can never exceed
    the modal share (the conventional cluster is one cluster; the
    modal cluster is the largest). A check with a flipped verdict,
    a forged interval, or a conventional share above the modal share
    is refused at construction.
    ``set_fingerprint`` binds this check to the candidate set it was
    computed from. ``conventional_index`` and ``clusters`` are carried
    so the modal share and the conventional share are re-derivable
    from the record alone: the modal share must equal the largest
    cluster over n, the conventional share the conventional
    candidate's own cluster over n, and ``conventional_in_modal``
    must say whether that cluster is the modal one. A check with a
    forged modal share or a flipped conventional_in_modal is refused
    at construction.
    """

    declared_p: float
    modal_share: float
    conventional_share: float
    agreement_interval: tuple[float, float]
    verdict: PCheckVerdict
    conventional_in_modal: bool
    notes: tuple[str, ...]
    set_fingerprint: str
    tolerance: float
    n: int
    conventional_index: int
    clusters: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("declared_p", self.declared_p),
            ("modal_share", self.modal_share),
            ("conventional_share", self.conventional_share),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.n < 2:
            raise ValueError(f"p-check needs n >= 2, got {self.n}")
        if not 0.0 <= self.tolerance < 1.0:
            raise ValueError(f"tolerance must be in [0, 1), got {self.tolerance}")
        if self.conventional_share > self.modal_share:
            raise ValueError(
                "conventional share cannot exceed the modal share: "
                f"{self.conventional_share} > {self.modal_share}"
            )
        if not 0 <= self.conventional_index < self.n:
            raise ValueError(
                f"conventional_index {self.conventional_index} out of range for n={self.n}"
            )
        # The modal share, the conventional share and the in-modal
        # flag are re-derived from the carried clusters, never
        # trusted: a forged modal share or a flipped
        # conventional_in_modal is refused here.
        flat = sorted(i for cluster in self.clusters for i in cluster)
        if flat != list(range(self.n)):
            raise ValueError("clusters must partition the candidate indices")
        if any(len(c) == 0 for c in self.clusters):
            raise ValueError("clusters must not contain an empty cluster")
        modal = max(self.clusters, key=lambda c: (len(c), -min(c)))
        if abs(self.modal_share - len(modal) / self.n) > 1e-12:
            raise ValueError(
                "modal_share must equal len(modal_cluster) / n: "
                f"{self.modal_share} != {len(modal)}/{self.n}"
            )
        conventional_cluster = next(
            cluster for cluster in self.clusters if self.conventional_index in cluster
        )
        if abs(self.conventional_share - len(conventional_cluster) / self.n) > 1e-12:
            raise ValueError(
                "conventional_share must equal the conventional "
                "candidate's cluster share: "
                f"{self.conventional_share} != {len(conventional_cluster)}/{self.n}"
            )
        if self.conventional_in_modal != (set(conventional_cluster) == set(modal)):
            raise ValueError(
                "conventional_in_modal contradicts the clusters: the "
                "conventional candidate is "
                + (
                    "in the modal cluster"
                    if set(conventional_cluster) == set(modal)
                    else "outside the modal cluster"
                )
            )
        low, high = self.agreement_interval
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                f"agreement interval must satisfy 0 <= low <= high <= 1, "
                f"got {self.agreement_interval!r}"
            )
        # The interval must be the Wilson 95% interval on the
        # conventional share (k = round(share * n) of n candidates).
        expected_low, expected_high = wilson_interval(
            round(self.conventional_share * self.n), self.n
        )
        if abs(low - expected_low) > 1e-12 or abs(high - expected_high) > 1e-12:
            raise ValueError(
                "agreement interval must be the Wilson 95% interval on "
                f"the conventional share {self.conventional_share:.2f} "
                f"over n={self.n}"
            )
        # The verdict must follow from the interval ± tolerance.
        if self.declared_p > high + self.tolerance:
            expected_verdict: PCheckVerdict = "overconfident"
        elif self.declared_p < low - self.tolerance:
            expected_verdict = "underconfident"
        else:
            expected_verdict = "consistent"
        if self.verdict != expected_verdict:
            raise ValueError(
                f"verdict {self.verdict!r} contradicts the interval "
                f"[{low:.2f}, {high:.2f}] ± tolerance {self.tolerance:.2f} "
                f"for declared p {self.declared_p:.2f}: expected "
                f"{expected_verdict!r}"
            )
        if self.verdict not in ("consistent", "overconfident", "underconfident"):
            raise ValueError(f"unknown p-check verdict: {self.verdict!r}")
        if not all(isinstance(note, str) for note in self.notes):
            raise ValueError("p-check notes must all be strings")
        if not self.set_fingerprint:
            raise ValueError("set_fingerprint must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form of this p_conventional check."""
        return {
            "declared_p": self.declared_p,
            "modal_share": self.modal_share,
            "conventional_share": self.conventional_share,
            "agreement_interval": list(self.agreement_interval),
            "verdict": self.verdict,
            "conventional_in_modal": self.conventional_in_modal,
            "notes": list(self.notes),
            "set_fingerprint": self.set_fingerprint,
            "tolerance": self.tolerance,
            "n": self.n,
            "conventional_index": self.conventional_index,
            "clusters": [list(c) for c in self.clusters],
        }


def check_p_conventional(diverge_set: DivergeSet, report: DiversityReport) -> PConventionalCheck:
    """Check the declared p_conventional against the measured convergence.

    The interval is built on the conventional candidate's own cluster
    share. A report from a different candidate set is refused: the
    fingerprint must match, so a measurement computed on one set can
    never silently validate a claim about another.
    """
    if report.set_fingerprint != diverge_set.fingerprint():
        raise ValueError(
            "diversity report is for a different candidate set: "
            "refusing to check p_conventional against a foreign measurement"
        )
    if report.n != len(diverge_set.candidates):
        raise ValueError(
            "diversity report covers n="
            f"{report.n} candidates but the set has "
            f"{len(diverge_set.candidates)}: refusing to check "
            "p_conventional against a measurement on a different-sized set"
        )
    if report.max_pair_similarity != diverge_set.policy.max_pair_similarity:
        raise ValueError(
            "diversity report was graded under a different "
            f"max_pair_similarity ({report.max_pair_similarity:.2f}) than "
            f"the set's policy ({diverge_set.policy.max_pair_similarity:.2f}): "
            "the clusters the p-check reads depend on that threshold"
        )
    conventional_cluster = next(
        cluster for cluster in report.clusters if diverge_set.conventional_index in cluster
    )
    conventional_share = len(conventional_cluster) / report.n
    interval = wilson_interval(len(conventional_cluster), report.n)
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
            f"upper bound {interval[1]:.2f} on the measured share "
            f"{conventional_share:.2f} of the conventional answer's "
            f"cluster by more than the {tolerance:.2f} tolerance"
        )
    elif verdict == "underconfident":
        notes.append(
            f"declared p_conventional {p:.2f} sits below the Wilson 95% "
            f"lower bound {interval[0]:.2f} on the measured share "
            f"{conventional_share:.2f} of the conventional answer's "
            f"cluster by more than the {tolerance:.2f} tolerance"
        )
    elif interval[1] + tolerance >= 1.0:
        notes.append(
            f"at this convergence (conventional share "
            f"{conventional_share:.2f}) the check cannot refute any "
            f"declared p: the Wilson 95% upper bound {interval[1]:.2f} "
            f"+ tolerance {tolerance:.2f} reaches 1.00 with only "
            f"n={report.n} candidates"
        )
    return PConventionalCheck(
        declared_p=p,
        modal_share=report.modal_share,
        conventional_share=conventional_share,
        agreement_interval=interval,
        verdict=verdict,
        conventional_in_modal=in_modal,
        notes=tuple(notes),
        set_fingerprint=diverge_set.fingerprint(),
        tolerance=tolerance,
        n=report.n,
        conventional_index=diverge_set.conventional_index,
        clusters=report.clusters,
    )


def p_check_summary(check: PConventionalCheck) -> str:
    """One plain-language sentence translating the p-check verdict.

    The CLI prints this after the Wilson interval, so a stranger who
    never reads the interval math still learns what the verdict means
    for the decision and what to do next.
    """
    if check.verdict == "overconfident":
        return (
            f"the candidates converge on the conventional answer only "
            f"{check.conventional_share:.2f} of the time, so a declared p "
            f"of {check.declared_p:.2f} claims more convergence than the "
            f"candidates show; re-diverge or lower the declared p"
        )
    if check.verdict == "underconfident":
        return (
            f"the candidates converge on the conventional answer "
            f"{check.conventional_share:.2f} of the time, so a declared p "
            f"of {check.declared_p:.2f} understates the convergence the "
            f"candidates show"
        )
    return (
        f"the candidates converge on the conventional answer "
        f"{check.conventional_share:.2f} of the time, which does not "
        f"refute a declared p of {check.declared_p:.2f}"
    )


@dataclass(frozen=True)
class Selection:
    """The recorded selection: why the conventional answer won or lost.

    ``margin`` is the winner's score minus the RUNNER-UP score (the
    highest score among the non-winners; 0.0 on a tie at the top,
    within the 1e-12 tie tolerance), the same quantity in every outcome — never the
    winner-minus-conventional gap, which is 0.0 exactly when the
    conventional answer wins and means something else when it loses.
    ``runner_up_score`` is carried so the margin is checkable from the
    record alone. ``reason`` is built from the measured facts only:
    winner score vs conventional score, the margin, whether the winner
    sits in the modal cluster, the diversity grade and the p-check
    verdict. The record IS the explanation — there is no prose beside
    it to drift from it. ``set_fingerprint`` binds the selection to
    the candidate set it was computed from. ``scores`` is the full
    score vector, carried so the winner is checkable as the argmax
    from the record alone: a winner that does not hold the top score,
    or a runner-up that is not the best of the rest, is refused.
    ``max_pair_similarity`` and ``p_conventional_tolerance`` are the
    thresholds the diversity and p-check grades were computed under,
    carried so the high-stakes gate can pin the canonical policy
    (0.80 / 0.15) instead of trusting grades computed under thresholds
    it cannot see. The boundary, stated plainly: a hand-built record
    that is internally consistent under the canonical policy is
    indistinguishable from a pipeline-built one, and the tie-break
    choice among tied top scorers is trusted rather than re-derived —
    the record does not carry the per-candidate means the tie-break
    reads.

    The record validates its own invariants: a hand-built Selection
    whose margin is not winner-minus-runner-up, whose outcome does
    not match its scores, or whose indices are out of range is refused
    at construction — the high-stakes gate trusts this record, so the
    record must not be freely forgeable in an inconsistent state.
    """

    winner_index: int
    conventional_index: int
    n: int
    outcome: SelectionOutcome
    winner_score: float
    conventional_score: float
    runner_up_score: float
    margin: float
    reason: str
    diverse: bool
    p_verdict: PCheckVerdict
    set_fingerprint: str
    scores: tuple[float, ...]
    max_pair_similarity: float
    p_conventional_tolerance: float

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError(f"selection needs n >= 2, got {self.n}")
        if not 0 <= self.winner_index < self.n:
            raise ValueError(f"winner_index {self.winner_index} out of range for n={self.n}")
        if not 0 <= self.conventional_index < self.n:
            raise ValueError(
                f"conventional_index {self.conventional_index} out of range for n={self.n}"
            )
        for name, value in (
            ("winner_score", self.winner_score),
            ("conventional_score", self.conventional_score),
            ("runner_up_score", self.runner_up_score),
            ("margin", self.margin),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        # Tolerance-consistent with the tie-break in select(): a winner
        # picked by the tie-break can sit up to 1e-12 below the strict
        # top score, so the best of the rest can exceed the winner by
        # at most the tie tolerance (scores 0.0 vs 7.6e-268 are tied
        # at 1e-12). Anything beyond the tolerance is still a forged
        # non-argmax winner and is refused.
        if self.runner_up_score > self.winner_score + 1e-12:
            raise ValueError(
                "runner-up score cannot exceed the winner score: "
                f"{self.runner_up_score} > {self.winner_score}"
            )
        # Float subtraction is not exact (0.95 - 0.90 is
        # 0.04999999999999993, not 0.05), so a mathematically correct
        # margin written as a literal must not be refused as forged:
        # compare with the same 1e-12 tolerance the diversity report
        # uses for modal_share.
        if abs(self.margin - (self.winner_score - self.runner_up_score)) > 1e-12:
            raise ValueError(
                "margin must equal winner_score - runner_up_score: "
                f"{self.margin} != {self.winner_score} - {self.runner_up_score}"
            )
        if len(self.scores) != self.n:
            raise ValueError(
                f"selection needs one score per candidate, got {len(self.scores)} for n={self.n}"
            )
        if not all(math.isfinite(s) for s in self.scores):
            raise ValueError("selection scores must all be finite")
        if not 0.0 < self.max_pair_similarity < 1.0:
            raise ValueError(
                f"max_pair_similarity must be in (0, 1), got {self.max_pair_similarity}"
            )
        if not 0.0 <= self.p_conventional_tolerance < 1.0:
            raise ValueError(
                f"p_conventional_tolerance must be in [0, 1), got {self.p_conventional_tolerance}"
            )
        # The winner must be an argmax of the carried scores, and the
        # runner-up the best of the rest: a forged non-argmax winner
        # (a winner_index whose own score is not the top score) is
        # refused here, not at the gate.
        if abs(self.winner_score - max(self.scores)) > 1e-12:
            raise ValueError(
                "winner_score is not the top score: "
                f"{self.winner_score} != max(scores) {max(self.scores)}"
            )
        if abs(self.scores[self.winner_index] - self.winner_score) > 1e-12:
            raise ValueError(
                f"winner_index {self.winner_index} holds score "
                f"{self.scores[self.winner_index]}, not the winner_score {self.winner_score}"
            )
        expected_runner_up = max(s for i, s in enumerate(self.scores) if i != self.winner_index)
        if abs(self.runner_up_score - expected_runner_up) > 1e-12:
            raise ValueError(
                "runner_up_score must be the best score among the "
                f"non-winners: {self.runner_up_score} != {expected_runner_up}"
            )
        # The conventional score is read by the outcome derivation
        # below, so it must match the carried vector like every other
        # score: a forged conventional_score that flips lost to tie
        # (or tie to lost) is refused here.
        if abs(self.conventional_score - self.scores[self.conventional_index]) > 1e-12:
            raise ValueError(
                "conventional_score must equal scores[conventional_index]: "
                f"{self.conventional_score} != {self.scores[self.conventional_index]}"
            )
        tied_at_top = any(
            abs(s - self.winner_score) <= 1e-12
            for i, s in enumerate(self.scores)
            if i != self.winner_index
        )
        if self.winner_index == self.conventional_index:
            # A conventional candidate that ties at the top and wins
            # the tie-break did not win on score: the outcome is a
            # tie, and the reason must name the tie-break.
            expected: SelectionOutcome = "tie" if tied_at_top else "conventional_won"
        elif abs(self.conventional_score - self.winner_score) <= 1e-12:
            expected = "tie"
        else:
            expected = "conventional_lost"
        if self.outcome != expected:
            raise ValueError(
                f"outcome {self.outcome!r} contradicts the scores "
                f"(winner {self.winner_index} at {self.winner_score}, "
                f"conventional {self.conventional_index} at "
                f"{self.conventional_score}): expected {expected!r}"
            )
        if self.p_verdict not in ("consistent", "overconfident", "underconfident"):
            raise ValueError(f"unknown p verdict: {self.p_verdict!r}")
        if not self.reason.strip():
            raise ValueError("selection reason must be non-empty")
        if not self.set_fingerprint:
            raise ValueError("set_fingerprint must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form of this selection record."""
        return {
            "winner_index": self.winner_index,
            "conventional_index": self.conventional_index,
            "n": self.n,
            "outcome": self.outcome,
            "winner_score": self.winner_score,
            "conventional_score": self.conventional_score,
            "runner_up_score": self.runner_up_score,
            "margin": self.margin,
            "reason": self.reason,
            "diverse": self.diverse,
            "diversity_scope": "token-level near-duplicate",
            "p_verdict": self.p_verdict,
            "set_fingerprint": self.set_fingerprint,
            "scores": list(self.scores),
            "max_pair_similarity": self.max_pair_similarity,
            "p_conventional_tolerance": self.p_conventional_tolerance,
        }


def select(
    diverge_set: DivergeSet,
    report: DiversityReport,
    p_check: PConventionalCheck,
) -> Selection:
    """Select the winning candidate from a diverse set.

    Winner is the argmax score. Ties break toward the candidate with
    the LOWEST mean pairwise similarity — the answer least like the
    others, the most distinctive — then the lowest index, so the
    choice is deterministic with no RNG. The mean similarity is a
    measured quantity that varies inside a diverse set (unlike cluster
    size, which is 1 for every candidate once the set is diverse), so
    the tie-break is a real mechanism, not dead code. A set that
    fails the diversity bar raises :class:`NotDiverseError`: a
    paraphrase set never silently selects. A report or p-check from a
    different candidate set raises ValueError: the fingerprint must
    match, so mismatched measurements never compose into a
    valid-looking selection. The same holds inside one set: the p-check
    must agree with the report on the clusters, the declared p and the
    conventional index — a check graded on foreign clusters (or a
    foreign declared p) composes into a verdict on a measurement the
    report does not make, so it is refused here, not at the gate.
    """
    if report.set_fingerprint != diverge_set.fingerprint():
        raise ValueError(
            "diversity report is for a different candidate set: "
            "refusing to select from a foreign measurement"
        )
    if p_check.set_fingerprint != diverge_set.fingerprint():
        raise ValueError(
            "p_conventional check is for a different candidate set: "
            "refusing to select from a foreign measurement"
        )
    if report.max_pair_similarity != diverge_set.policy.max_pair_similarity:
        raise ValueError(
            "diversity report was graded under a different "
            f"max_pair_similarity ({report.max_pair_similarity:.2f}) than "
            f"the set's policy ({diverge_set.policy.max_pair_similarity:.2f}): "
            "refusing to select from a mismatched measurement"
        )
    if p_check.tolerance != diverge_set.policy.p_conventional_tolerance:
        raise ValueError(
            "p_conventional check was graded under a different "
            f"tolerance ({p_check.tolerance:.2f}) than the set's policy "
            f"({diverge_set.policy.p_conventional_tolerance:.2f}): "
            "refusing to select from a mismatched measurement"
        )
    # The p-check must agree with the report it composes with: same
    # clusters (the shares the verdict was computed on), same n, same
    # conventional index, same declared p. A check graded on foreign
    # clusters would lend its verdict to a measurement the report
    # does not make.
    if tuple(p_check.clusters) != tuple(report.clusters):
        raise ValueError(
            "p_conventional check clusters do not match the diversity "
            "report clusters: refusing to select from contradictory "
            "measurements"
        )
    if report.n != len(diverge_set.candidates):
        raise ValueError(
            "diversity report covers n="
            f"{report.n} candidates but the set has "
            f"{len(diverge_set.candidates)}: refusing to select from "
            "a measurement on a different-sized set"
        )
    if p_check.n != len(diverge_set.candidates):
        raise ValueError(
            "p_conventional check covers n="
            f"{p_check.n} candidates but the set has "
            f"{len(diverge_set.candidates)}: refusing to select from "
            "a measurement on a different-sized set"
        )
    if p_check.n != report.n:
        raise ValueError(
            f"p_conventional check covers n={p_check.n} but the "
            f"diversity report covers n={report.n}"
        )
    if p_check.conventional_index != diverge_set.conventional_index:
        raise ValueError(
            "p_conventional check names a different conventional "
            f"candidate ({p_check.conventional_index}) than the set "
            f"({diverge_set.conventional_index})"
        )
    if abs(p_check.declared_p - diverge_set.p_conventional) > 1e-12:
        raise ValueError(
            "p_conventional check declared_p "
            f"({p_check.declared_p:.2f}) does not match the set's "
            f"p_conventional ({diverge_set.p_conventional:.2f})"
        )
    if not report.diverse:
        dupes = [p for p in report.pairs if p.near_duplicate]
        raise NotDiverseError(
            "cannot select from a non-diverse set: "
            + ", ".join(f"candidates {p.i}/{p.j} similarity {p.similarity:.2f}" for p in dupes)
            + "; re-diverge instead of choosing between paraphrases"
        )
    scores = [c.score for c in diverge_set.candidates]
    top = max(scores)
    # Ties are float-tolerance ties (1e-12), the same tolerance the
    # Selection record uses: an exact-equality test here would let a
    # within-tolerance winner_score rebuild the false "won on score"
    # record the tie outcome exists to prevent.
    tied = [i for i, s in enumerate(scores) if abs(s - top) <= 1e-12]
    winner = min(tied, key=lambda i: (report.mean_similarity_by_candidate[i], i))
    conv = diverge_set.conventional_index
    conv_score = scores[conv]
    runner_up = max(s for i, s in enumerate(scores) if i != winner)
    margin = scores[winner] - runner_up
    if -1e-12 <= margin < 0:
        # A tie-break winner can sit up to 1e-12 below the strict top
        # score, so the margin can go microscopically negative; the
        # record quotes it as +0.00, never -0.00.
        margin = 0.0
    if winner == conv:
        # A tie at the top won by the conventional candidate on the
        # tie-break is a tie, not a win on score: the margin is 0.00
        # and the reason must say the tie-break decided it.
        outcome: SelectionOutcome = "tie" if len(tied) > 1 else "conventional_won"
    elif conv in tied:
        outcome = "tie"
    else:
        outcome = "conventional_lost"
    winner_in_modal = winner in report.modal_cluster
    if outcome == "conventional_won":
        why = (
            f"conventional candidate {conv} won on score "
            f"({conv_score:.2f}, margin +{margin:.2f} over the runner-up)"
        )
    elif outcome == "tie":
        why = (
            f"conventional candidate {conv} tied on top score "
            f"({conv_score:.2f}); winner candidate {winner} picked by "
            "the diversity tie-break (lowest mean pairwise similarity "
            f"{report.mean_similarity_by_candidate[winner]:.2f}, then lowest index)"
        )
    else:
        gap = scores[winner] - conv_score
        why = (
            f"conventional candidate {conv} lost on score "
            f"({conv_score:.2f} vs winner {scores[winner]:.2f}, "
            f"gap {gap:.2f}; winner's margin over the runner-up {margin:.2f})"
        )
    if max(len(cluster) for cluster in report.clusters) > 1:
        modal_note = (
            "winner is in the modal cluster"
            if winner_in_modal
            else "winner is outside the modal cluster"
        ) + f" (modal cluster {list(report.modal_cluster)}, share {report.modal_share:.2f})"
    else:
        # All clusters are singletons: there is no convergence to
        # report, and naming the modal cluster would present an
        # arbitrary lowest-index tie-break as a finding.
        modal_note = "no two candidates converge (all clusters are singletons)"
    reason = (
        f"{why}. {modal_note}. "
        f"Set diverse at token level (no near-verbatim pair, not "
        f"semantic diversity; min pairwise diversity "
        f"{report.min_pair_diversity:.2f}). p_conventional "
        f"{p_check.declared_p:.2f} is {p_check.verdict} against the "
        f"measured share of the conventional answer's cluster "
        f"({p_check.conventional_share:.2f})."
    )
    return Selection(
        winner_index=winner,
        conventional_index=conv,
        n=len(scores),
        outcome=outcome,
        winner_score=scores[winner],
        conventional_score=conv_score,
        runner_up_score=runner_up,
        margin=margin,
        reason=reason,
        diverse=report.diverse,
        p_verdict=p_check.verdict,
        set_fingerprint=diverge_set.fingerprint(),
        scores=tuple(scores),
        max_pair_similarity=diverge_set.policy.max_pair_similarity,
        p_conventional_tolerance=diverge_set.policy.p_conventional_tolerance,
    )


def may_decide_high_stakes(selection: Selection) -> bool:
    """True only for a token-level-diverse 5-candidate selection with a non-overconfident p.

    A high-stakes decision needs all four: exactly 5 candidates (the
    issue's number), token-level diversity (the choice was between
    answers that are not near-verbatim paraphrases), a
    p_conventional the measurement does not refute, and the canonical
    policy (max_pair_similarity 0.80, p tolerance 0.15). The count and
    the policy are enforced HERE, from the selection record — not at
    build time, where a caller could pass low stakes and decide high
    stakes anyway, and not on trust, where a caller could grade a set
    under a lax policy (a near-verbatim pair at 0.83 passes a 0.99
    cap; a declared p of 0.95 passes a 0.99 tolerance) and hand the
    gate a "diverse" / "consistent" record it cannot distinguish from
    a default-policy one. The selection carries the thresholds it was
    graded under, so the gate reads them instead of trusting them. A
    2-candidate selection, however diverse and however confident, may
    not decide — and neither may a lax-policy one.
    """
    canonical = default_diverge_policy()
    return (
        selection.n == 5
        and selection.diverse
        and selection.p_verdict != "overconfident"
        and selection.max_pair_similarity == canonical.max_pair_similarity
        and selection.p_conventional_tolerance == canonical.p_conventional_tolerance
    )


def require_high_stakes_decision(selection: Selection) -> Selection:
    """Return the selection, or refuse it for a high-stakes decision.

    The gate takes ONLY the selection record. A caller that wants to
    decide on high stakes must pass a selection; a bare candidate, a
    single sample, or a score has no path through this function.
    """
    if not may_decide_high_stakes(selection):
        if selection.n != 5:
            why = f"selection covers {selection.n} candidates, high-stakes /diverge needs exactly 5"
        elif (
            selection.max_pair_similarity != default_diverge_policy().max_pair_similarity
            or selection.p_conventional_tolerance
            != default_diverge_policy().p_conventional_tolerance
        ):
            why = (
                "selection was graded under a non-canonical policy "
                f"(max_pair_similarity {selection.max_pair_similarity:.2f}, "
                f"p tolerance {selection.p_conventional_tolerance:.2f}); "
                "high-stakes /diverge needs the canonical 0.80 / 0.15 thresholds"
            )
        elif not selection.diverse:
            why = "selection is not diverse"
        else:
            why = (
                "p_conventional was overconfident against the measured "
                "share of the conventional answer's cluster"
            )
        raise HighStakesRefused(
            "high-stakes decision refused: " + why + f" (outcome {selection.outcome}, p verdict "
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
    if not math.isfinite(p_conventional):
        raise ValueError(f"diverge p_conventional must be finite, got {p_raw!r}")
    if not 0.0 <= p_conventional <= 1.0:
        raise ValueError(f"diverge p_conventional must be in [0, 1], got {p_conventional}")
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
