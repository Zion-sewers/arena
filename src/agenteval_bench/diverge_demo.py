"""Seeded end-to-end demo of anti-mode-collapse /diverge (arena#47).

Runs the full /diverge protocol offline on recorded candidate sets
plus one real CLI invocation on a temporary candidates file. Exit 0
only if every acceptance expectation holds:
- a genuinely diverse set selects on score, and the record says why
  the conventional answer lost (highest score elsewhere, margin
  quoted);
- a paraphrase set (two candidates differing by one word, same
  numbers) fails the diversity bar and selection is refused, never
  silently made;
- an overconfident p_conventional (declared 0.95 while the candidates
  converge on nothing — modal share 0.20) is convicted by the Wilson
  interval and blocks the high-stakes gate;
- a single sample on high stakes is refused with SingleSampleError —
  one candidate is not a divergence;
- a tie on top score breaks toward the most distinctive candidate
  (smallest cluster), deterministically;
- the conventional answer can also win: the record says so, with the
  margin over the runner-up;
- the loader fails closed on an unknown key and on a missing score;
- the CLI `diverge` command on a candidates file exits 0 with a
  selection and writes the report JSON.

The fixture candidates are recorded data, not measurements: no live
generation runs in this demo, and the module itself makes no network
call. Any failure raises, so CI can gate on this demo directly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agenteval_bench.cli import cmd_diverge
from agenteval_bench.diverge import (
    Candidate,
    HighStakesRefused,
    NotDiverseError,
    SingleSampleError,
    analyze_diversity,
    build_diverge_set,
    candidates_from_json,
    check_p_conventional,
    may_decide_high_stakes,
    require_high_stakes_decision,
    select,
)

PROMPT = "How do I make this lookup fast?"


def _expect(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"  ok    {label}")


def _diverse_set() -> list[Candidate]:
    return [
        Candidate("Use a hash map keyed by user id for O(1) lookup", 0.70, "hashmap"),
        Candidate("Sort the list and binary search it in O(log n)", 0.90, "sort"),
        Candidate("Scan the list linearly each time O(n)", 0.60, "scan"),
        Candidate("Build a trie over the names for prefix lookup", 0.80, "trie"),
        Candidate("Cache results in Redis with a 60s TTL", 0.75, "cache"),
    ]


def main() -> int:
    print("Anti-mode-collapse /diverge demo (arena#47)")

    # 1. Diverse set: the non-conventional answer wins on score, and
    # the record says why.
    good = build_diverge_set(PROMPT, _diverse_set(), 0, 0.60, stakes="high")
    report = analyze_diversity(good)
    _expect(report.diverse, "five genuinely different answers are diverse")
    _expect(report.modal_share == 0.20, "no convergence: modal share is 1/5")
    p_check = check_p_conventional(good, report)
    _expect(p_check.verdict == "consistent", "p 0.60 survives the wide n=5 interval")
    selection = select(good, report, p_check)
    _expect(selection.outcome == "conventional_lost", "conventional answer lost")
    _expect(selection.winner_index == 1, "winner is the top scorer")
    _expect("margin 0.20" in selection.reason, "reason quotes the margin")
    _expect("lost on score" in selection.reason, "reason names why it lost")
    _expect(may_decide_high_stakes(selection), "diverse + consistent p may decide")
    require_high_stakes_decision(selection)

    # 2. The paraphrase trap: same answer twice, one word changed.
    trap = build_diverge_set(
        PROMPT,
        [
            Candidate("Use a hash map keyed by user id for O(1) lookup", 0.70),
            Candidate("Use a hash map keyed by user id for O(1) lookups", 0.90),
            Candidate("Sort the list and binary search it in O(log n)", 0.60),
            Candidate("Build a trie over the names for prefix lookup", 0.80),
            Candidate("Cache results in Redis with a 60s TTL", 0.75),
        ],
        0,
        0.20,
        stakes="high",
    )
    trap_report = analyze_diversity(trap)
    _expect(not trap_report.diverse, "a paraphrase pair fails the set")
    _expect(
        trap_report.min_pair_diversity == 0.0,
        "the paraphrase pair has zero diversity",
    )
    try:
        select(trap, trap_report, check_p_conventional(trap, trap_report))
    except NotDiverseError as e:
        _expect("re-diverge" in str(e), "selection refusal tells the caller to re-diverge")
    else:  # pragma: no cover - the refusal is the expectation
        raise AssertionError("select must refuse a paraphrase set")

    # 3. The overconfident p: declared 0.95, modal share 0.20.
    over = build_diverge_set(PROMPT, _diverse_set(), 0, 0.95, stakes="high")
    over_report = analyze_diversity(over)
    over_check = check_p_conventional(over, over_report)
    _expect(over_check.verdict == "overconfident", "p 0.95 is refuted by the modal share")
    over_selection = select(over, over_report, over_check)
    _expect(
        not may_decide_high_stakes(over_selection),
        "an overconfident p blocks the high-stakes gate",
    )
    try:
        require_high_stakes_decision(over_selection)
    except HighStakesRefused as e:
        _expect("overconfident" in str(e), "gate refusal names the overconfident p")
    else:  # pragma: no cover - the refusal is the expectation
        raise AssertionError("require_high_stakes_decision must refuse an overconfident p")

    # 4. Single sample on high stakes is refused, structurally.
    try:
        build_diverge_set(PROMPT, _diverse_set()[:1], 0, 0.50, stakes="high")
    except SingleSampleError as e:
        _expect("not sufficient" in str(e), "single sample refusal names the rule")
    else:  # pragma: no cover - the refusal is the expectation
        raise AssertionError("one candidate must not build a high-stakes set")

    # 5. A tie on top score breaks toward the most distinctive answer.
    # (The tie-break is unreachable on a paraphrase set — select()
    # refuses those first — so the fixture is diverse by construction.)
    tied = build_diverge_set(
        PROMPT,
        [
            Candidate("Use a hash map keyed by user id for O(1) lookup", 0.70),
            Candidate("Sort the list and binary search it in O(log n)", 0.90),
            Candidate("Build a trie over the names for prefix lookup", 0.90),
            Candidate("Cache results in Redis with a 60s TTL", 0.75),
            Candidate("Scan the list linearly each time O(n)", 0.60),
        ],
        2,
        0.40,
        stakes="low",
    )
    tied_report = analyze_diversity(tied)
    _expect(tied_report.diverse, "the tie fixture is diverse")
    tied_selection = select(tied, tied_report, check_p_conventional(tied, tied_report))
    _expect(tied_selection.outcome == "tie", "top-score tie with the conventional in it")
    _expect(tied_selection.winner_index == 1, "tie breaks to the lowest index")
    _expect("tie-break" in tied_selection.reason, "reason names the tie-break")

    # 6. The conventional answer can win: the record says so.
    winning = build_diverge_set(
        PROMPT,
        [
            Candidate("Use a hash map keyed by user id for O(1) lookup", 0.95),
            Candidate("Sort the list and binary search it in O(log n)", 0.90),
            Candidate("Scan the list linearly each time O(n)", 0.60),
            Candidate("Build a trie over the names for prefix lookup", 0.80),
            Candidate("Cache results in Redis with a 60s TTL", 0.75),
        ],
        0,
        0.60,
        stakes="high",
    )
    win_report = analyze_diversity(winning)
    win_selection = select(winning, win_report, check_p_conventional(winning, win_report))
    _expect(win_selection.outcome == "conventional_won", "conventional answer won")
    _expect("margin +0.05" in win_selection.reason, "reason quotes the winning margin")

    # 7. The loader fails closed on an unknown key and a missing score.
    try:
        candidates_from_json(
            json.dumps(
                {
                    "prompt": PROMPT,
                    "conventional_index": 0,
                    "p_conventional": 0.6,
                    "candidates": [{"text": "x", "score": 0.5}],
                    "surprise": True,
                }
            )
        )
    except ValueError as e:
        _expect("unknown keys" in str(e), "loader refuses unknown keys")
    else:  # pragma: no cover - the refusal is the expectation
        raise AssertionError("loader must refuse unknown keys")
    try:
        candidates_from_json(
            json.dumps(
                {
                    "prompt": PROMPT,
                    "conventional_index": 0,
                    "p_conventional": 0.6,
                    "candidates": [{"text": "x"}],
                }
            )
        )
    except ValueError as e:
        _expect("needs 'text' and 'score'" in str(e), "loader refuses a scoreless candidate")
    else:  # pragma: no cover - the refusal is the expectation
        raise AssertionError("loader must refuse a candidate without a score")

    # 8. The CLI on a candidates file exits 0 and writes the report.
    payload = {
        "prompt": PROMPT,
        "stakes": "high",
        "conventional_index": 0,
        "p_conventional": 0.60,
        "candidates": [
            {"text": c.text, "score": c.score, "label": c.label} for c in _diverse_set()
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        infile = Path(tmp) / "diverge.json"
        outfile = Path(tmp) / "diverge-report.json"
        infile.write_text(json.dumps(payload), encoding="utf-8")
        code = cmd_diverge(["--candidates", str(infile), "--out", str(outfile)])
        _expect(code == 0, "CLI exits 0 on a diverse high-stakes selection")
        written = json.loads(outfile.read_text(encoding="utf-8"))
        _expect(
            written["selection"]["outcome"] == "conventional_lost",
            "CLI report records the conventional loss",
        )
        _expect(written["report"]["diverse"] is True, "CLI report records diversity")

    print("diverge demo: all drills passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
