"""Tests for anti-mode-collapse /diverge (arena#47).

Every acceptance criterion has a named test, and the property tests
at the bottom pin the invariants for every generated candidate set:
similarity is symmetric and reflexive, diversity never exceeds 1,
and selection is deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agenteval_bench.cli import cmd_diverge
from agenteval_bench.diverge import (
    Candidate,
    DivergePolicy,
    DivergeSet,
    HighStakesRefused,
    NotDiverseError,
    Selection,
    SingleSampleError,
    Stakes,
    analyze_diversity,
    build_diverge_set,
    candidates_from_json,
    check_p_conventional,
    content_tokens,
    hard_tokens,
    may_decide_high_stakes,
    pair_similarity,
    require_high_stakes_decision,
    select,
)

PROMPT = "How do I make this lookup fast?"

HASHMAP = "Use a hash map keyed by user id for O(1) lookup"
SORT = "Sort the list and binary search it in O(log n)"
SCAN = "Scan the list linearly each time O(n)"
TRIE = "Build a trie over the names for prefix lookup"
CACHE = "Cache results in Redis with a 60s TTL"


def _set(
    scores: tuple[float, float, float, float, float] = (0.70, 0.90, 0.60, 0.80, 0.75),
    conventional: int = 0,
    p: float = 0.60,
    stakes: Stakes = "high",
) -> tuple[list[Candidate], DivergeSet]:
    candidates = [
        Candidate(HASHMAP, scores[0], "hashmap"),
        Candidate(SORT, scores[1], "sort"),
        Candidate(SCAN, scores[2], "scan"),
        Candidate(TRIE, scores[3], "trie"),
        Candidate(CACHE, scores[4], "cache"),
    ]
    return candidates, build_diverge_set(PROMPT, candidates, conventional, p, stakes=stakes)


def _diverse_selection(p: float = 0.60, stakes: Stakes = "high") -> Selection:
    _, diverge_set = _set(p=p, stakes=stakes)
    report = analyze_diversity(diverge_set)
    return select(diverge_set, report, check_p_conventional(diverge_set, report))


class TestSimilarity:
    def test_identical_texts_are_maximally_similar(self) -> None:
        assert pair_similarity(HASHMAP, HASHMAP) == 1.0

    def test_disjoint_texts_are_minimally_similar(self) -> None:
        assert pair_similarity(HASHMAP, SCAN) < 0.20

    def test_stopwords_do_not_create_similarity(self) -> None:
        # Two texts sharing only glue words ("use the ... of the ...")
        # must not read as similar: content is what counts.
        assert pair_similarity("Use the thing of the day", "Use the stuff of the week") < 0.5

    def test_hard_tokens_convict_a_reworded_paraphrase(self) -> None:
        # Same numbers, different glue: a paraphrase that keeps the
        # figures must not pass as a different answer.
        paraphrase = "Employ a hash table indexed by user id for O(1) lookup"
        assert pair_similarity(HASHMAP, paraphrase) > 0.80

    def test_different_numbers_are_different_answers(self) -> None:
        assert pair_similarity("Retry after 30 seconds", "Retry after 300 seconds") < 0.80

    def test_content_and_hard_token_extraction(self) -> None:
        assert "hash" in content_tokens(HASHMAP)
        assert "the" not in content_tokens(HASHMAP)
        assert "60s" in hard_tokens(CACHE)
        assert "hash" not in hard_tokens(HASHMAP)


class TestDiversity:
    def test_genuinely_different_answers_are_diverse(self) -> None:
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        assert report.diverse
        assert report.modal_share == 0.20
        assert len(report.clusters) == 5

    def test_one_paraphrase_pair_fails_the_whole_set(self) -> None:
        candidates = [
            Candidate(HASHMAP, 0.70),
            Candidate("Use a hash map keyed by user id for O(1) lookups", 0.90),
            Candidate(SORT, 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 0, 0.20, stakes="high")
        report = analyze_diversity(diverge_set)
        assert not report.diverse
        assert report.min_pair_diversity == 0.0
        assert report.modal_cluster == (0, 1)
        assert report.modal_share == 0.40

    def test_three_way_paraphrase_cluster_is_the_modal_cluster(self) -> None:
        candidates = [
            Candidate(HASHMAP, 0.70),
            Candidate("Use a hash map keyed by user id for O(1) lookups", 0.90),
            Candidate("Use a hash map keyed by user ID for O(1) lookup", 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 0, 0.20, stakes="high")
        report = analyze_diversity(diverge_set)
        assert report.modal_cluster == (0, 1, 2)
        assert report.modal_share == 0.60

    def test_diversity_grade_is_the_weakest_link(self) -> None:
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        assert report.min_pair_diversity <= report.mean_pair_diversity


class TestPConventional:
    def test_consistent_p_survives_the_wide_interval(self) -> None:
        _, diverge_set = _set(p=0.60)
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        assert check.verdict == "consistent"

    def test_overconfident_p_is_convicted(self) -> None:
        _, diverge_set = _set(p=0.95)
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        assert check.verdict == "overconfident"
        assert "upper bound" in check.notes[0]

    def test_underconfident_p_is_named(self) -> None:
        # Three of five candidates converge on one answer (modal share
        # 0.60); declaring p 0.05 for the conventional answer understates
        # the measured convergence.
        candidates = [
            Candidate(HASHMAP, 0.70),
            Candidate("Use a hash map keyed by user id for O(1) lookups", 0.90),
            Candidate("Use a hash map keyed by user ID for O(1) lookup", 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 0, 0.05, stakes="high")
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        assert check.verdict == "underconfident"

    def test_conventional_outside_the_modal_cluster_is_a_finding(self) -> None:
        candidates = [
            Candidate(HASHMAP, 0.70),
            Candidate("Use a hash map keyed by user id for O(1) lookups", 0.90),
            Candidate("Use a hash map keyed by user ID for O(1) lookup", 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 3, 0.40, stakes="high")
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        assert not check.conventional_in_modal
        assert "not in the modal cluster" in check.notes[0]


class TestSelection:
    def test_conventional_loses_and_the_reason_says_so(self) -> None:
        selection = _diverse_selection()
        assert selection.outcome == "conventional_lost"
        assert selection.winner_index == 1
        assert "lost on score" in selection.reason
        assert "margin 0.20" in selection.reason

    def test_conventional_wins_and_the_reason_quotes_the_margin(self) -> None:
        _, diverge_set = _set(scores=(0.95, 0.90, 0.60, 0.80, 0.75))
        report = analyze_diversity(diverge_set)
        won = select(diverge_set, report, check_p_conventional(diverge_set, report))
        assert won.outcome == "conventional_won"
        assert "margin +0.05" in won.reason

    def test_tie_breaks_toward_the_most_distinctive_answer(self) -> None:
        candidates = [
            Candidate(HASHMAP, 0.70),
            Candidate(SORT, 0.90),
            Candidate(TRIE, 0.90),
            Candidate(CACHE, 0.75),
            Candidate(SCAN, 0.60),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 2, 0.40, stakes="low")
        report = analyze_diversity(diverge_set)
        tied = select(diverge_set, report, check_p_conventional(diverge_set, report))
        assert tied.outcome == "tie"
        assert tied.winner_index == 1
        assert "tie-break" in tied.reason

    def test_selection_refuses_a_paraphrase_set(self) -> None:
        candidates = [
            Candidate(HASHMAP, 0.70),
            Candidate("Use a hash map keyed by user id for O(1) lookups", 0.90),
            Candidate(SORT, 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 0, 0.20, stakes="high")
        report = analyze_diversity(diverge_set)
        with pytest.raises(NotDiverseError, match="re-diverge"):
            select(diverge_set, report, check_p_conventional(diverge_set, report))


class TestHighStakesGate:
    def test_diverse_consistent_selection_may_decide(self) -> None:
        selection = _diverse_selection()
        assert may_decide_high_stakes(selection)
        assert require_high_stakes_decision(selection) is selection

    def test_overconfident_p_blocks_the_gate(self) -> None:
        selection = _diverse_selection(p=0.95)
        assert not may_decide_high_stakes(selection)
        with pytest.raises(HighStakesRefused, match="overconfident"):
            require_high_stakes_decision(selection)

    def test_single_sample_is_refused_on_high_stakes(self) -> None:
        candidates, _ = _set()
        with pytest.raises(SingleSampleError, match="not sufficient"):
            build_diverge_set(PROMPT, candidates[:1], 0, 0.50, stakes="high")

    def test_short_list_on_low_stakes_is_a_shape_error_not_a_sample_error(self) -> None:
        candidates, _ = _set()
        with pytest.raises(ValueError, match="exactly 5"):
            build_diverge_set(PROMPT, candidates[:3], 0, 0.50, stakes="low")

    def test_six_candidates_is_refused(self) -> None:
        candidates, _ = _set()
        with pytest.raises(ValueError, match="exactly 5"):
            build_diverge_set(PROMPT, candidates + [Candidate(SCAN, 0.5)], 0, 0.50, stakes="low")


class TestValidation:
    def test_empty_candidate_text_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Candidate("   ", 0.5)

    def test_non_finite_score_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Candidate(HASHMAP, float("nan"))

    def test_p_conventional_out_of_range_is_refused(self) -> None:
        candidates, _ = _set()
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            build_diverge_set(PROMPT, candidates, 0, 1.5, stakes="low")

    def test_conventional_index_out_of_range_is_refused(self) -> None:
        candidates, _ = _set()
        with pytest.raises(ValueError, match="out of range"):
            build_diverge_set(PROMPT, candidates, 5, 0.5, stakes="low")

    def test_empty_prompt_is_refused(self) -> None:
        candidates, _ = _set()
        with pytest.raises(ValueError, match="non-empty"):
            build_diverge_set("  ", candidates, 0, 0.5, stakes="low")

    def test_bad_policy_is_refused(self) -> None:
        with pytest.raises(ValueError):
            DivergePolicy(n_candidates=1)
        with pytest.raises(ValueError):
            DivergePolicy(max_pair_similarity=1.0)
        with pytest.raises(ValueError):
            DivergePolicy(p_conventional_tolerance=1.0)


class TestLoader:
    def test_loader_round_trip(self) -> None:
        payload = {
            "prompt": PROMPT,
            "stakes": "high",
            "conventional_index": 0,
            "p_conventional": 0.60,
            "candidates": [
                {"text": HASHMAP, "score": 0.70, "label": "hashmap"},
                {"text": SORT, "score": 0.90},
                {"text": SCAN, "score": 0.60},
                {"text": TRIE, "score": 0.80},
                {"text": CACHE, "score": 0.75},
            ],
        }
        prompt, candidates, conv, p, stakes = candidates_from_json(json.dumps(payload))
        assert prompt == PROMPT
        assert len(candidates) == 5
        assert conv == 0 and p == 0.60 and stakes == "high"

    def test_loader_defaults_stakes_to_low(self) -> None:
        payload = {
            "prompt": PROMPT,
            "conventional_index": 0,
            "p_conventional": 0.60,
            "candidates": [
                {"text": t, "score": s}
                for t, s in [(HASHMAP, 0.7), (SORT, 0.9), (SCAN, 0.6), (TRIE, 0.8), (CACHE, 0.75)]
            ],
        }
        _, _, _, _, stakes = candidates_from_json(json.dumps(payload))
        assert stakes == "low"

    def test_loader_refuses_unknown_keys(self) -> None:
        with pytest.raises(ValueError, match="unknown keys"):
            candidates_from_json(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "candidates": [],
                        "conventional_index": 0,
                        "p_conventional": 0.5,
                        "surprise": True,
                    }
                )
            )

    def test_loader_refuses_a_scoreless_candidate(self) -> None:
        with pytest.raises(ValueError, match="needs 'text' and 'score'"):
            candidates_from_json(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "candidates": [{"text": HASHMAP}],
                        "conventional_index": 0,
                        "p_conventional": 0.5,
                    }
                )
            )

    def test_loader_refuses_bad_json(self) -> None:
        with pytest.raises(ValueError, match="invalid diverge JSON"):
            candidates_from_json("{not json")

    def test_loader_refuses_a_non_mapping_document(self) -> None:
        with pytest.raises(TypeError, match="must be a mapping"):
            candidates_from_json(json.dumps(["not", "a", "mapping"]))

    def test_loader_refuses_missing_keys(self) -> None:
        with pytest.raises(ValueError, match="missing required key"):
            candidates_from_json(json.dumps({"prompt": PROMPT}))

    def test_loader_refuses_a_non_string_prompt(self) -> None:
        with pytest.raises(TypeError, match="prompt must be a string"):
            candidates_from_json(
                json.dumps({"prompt": 42, "candidates": [], "conventional_index": 0,
                            "p_conventional": 0.5})
            )

    def test_loader_refuses_bad_stakes(self) -> None:
        with pytest.raises(ValueError, match="stakes must be 'low' or 'high'"):
            candidates_from_json(
                json.dumps({"prompt": PROMPT, "stakes": "critical", "candidates": [],
                            "conventional_index": 0, "p_conventional": 0.5})
            )

    def test_loader_refuses_a_bool_conventional_index(self) -> None:
        with pytest.raises(TypeError, match="conventional_index must be an int"):
            candidates_from_json(
                json.dumps({"prompt": PROMPT, "candidates": [],
                            "conventional_index": True, "p_conventional": 0.5})
            )

    def test_loader_refuses_a_bool_p_conventional(self) -> None:
        with pytest.raises(TypeError, match="p_conventional must be a number"):
            candidates_from_json(
                json.dumps({"prompt": PROMPT, "candidates": [],
                            "conventional_index": 0, "p_conventional": False})
            )

    def test_loader_refuses_non_list_candidates(self) -> None:
        with pytest.raises(TypeError, match="candidates must be a list"):
            candidates_from_json(
                json.dumps({"prompt": PROMPT, "candidates": "five",
                            "conventional_index": 0, "p_conventional": 0.5})
            )

    def test_loader_refuses_a_non_mapping_candidate(self) -> None:
        with pytest.raises(TypeError, match="must be a mapping"):
            candidates_from_json(
                json.dumps({"prompt": PROMPT, "candidates": ["five"],
                            "conventional_index": 0, "p_conventional": 0.5})
            )

    def test_loader_refuses_unknown_candidate_keys(self) -> None:
        with pytest.raises(ValueError, match="unknown keys"):
            candidates_from_json(
                json.dumps({"prompt": PROMPT,
                            "candidates": [{"text": HASHMAP, "score": 0.7, "rank": 1}],
                            "conventional_index": 0, "p_conventional": 0.5})
            )

    def test_loader_refuses_a_non_string_candidate_text(self) -> None:
        with pytest.raises(TypeError, match="text must be a string"):
            candidates_from_json(
                json.dumps({"prompt": PROMPT,
                            "candidates": [{"text": 42, "score": 0.7}],
                            "conventional_index": 0, "p_conventional": 0.5})
            )

    def test_loader_refuses_a_non_string_candidate_label(self) -> None:
        with pytest.raises(TypeError, match="label must be a string"):
            candidates_from_json(
                json.dumps({"prompt": PROMPT,
                            "candidates": [{"text": HASHMAP, "score": 0.7, "label": 3}],
                            "conventional_index": 0, "p_conventional": 0.5})
            )

    def test_non_finite_p_conventional_is_refused(self) -> None:
        candidates, _ = _set()
        with pytest.raises(ValueError, match="finite"):
            build_diverge_set(PROMPT, candidates, 0, float("inf"), stakes="low")

    def test_loader_refuses_a_bool_score(self) -> None:
        with pytest.raises(TypeError, match="must be a number"):
            candidates_from_json(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "candidates": [{"text": HASHMAP, "score": True}],
                        "conventional_index": 0,
                        "p_conventional": 0.5,
                    }
                )
            )


class TestCli:
    def _write(self, tmp_path: Path, payload: dict[str, object]) -> str:
        path = tmp_path / "diverge.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def _payload(self, p: float = 0.60, stakes: str = "high") -> dict[str, object]:
        return {
            "prompt": PROMPT,
            "stakes": stakes,
            "conventional_index": 0,
            "p_conventional": p,
            "candidates": [
                {"text": HASHMAP, "score": 0.70},
                {"text": SORT, "score": 0.90},
                {"text": SCAN, "score": 0.60},
                {"text": TRIE, "score": 0.80},
                {"text": CACHE, "score": 0.75},
            ],
        }

    def test_cli_exits_0_on_a_diverse_selection(self, tmp_path: Path) -> None:
        infile = self._write(tmp_path, self._payload())
        out = str(tmp_path / "report.json")
        assert cmd_diverge(["--candidates", infile, "--out", out]) == 0
        written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert written["selection"]["outcome"] == "conventional_lost"
        assert written["may_decide_high_stakes"] is True

    def test_cli_exits_2_on_a_paraphrase_set(self, tmp_path: Path) -> None:
        payload = self._payload()
        candidates = cast(list[dict[str, object]], payload["candidates"])
        candidates[1] = {
            "text": "Use a hash map keyed by user id for O(1) lookups",
            "score": 0.90,
        }
        infile = self._write(tmp_path, payload)
        assert cmd_diverge(["--candidates", infile]) == 2

    def test_cli_exits_2_on_an_overconfident_p(self, tmp_path: Path) -> None:
        infile = self._write(tmp_path, self._payload(p=0.95))
        assert cmd_diverge(["--candidates", infile]) == 2

    def test_cli_exits_2_on_a_single_sample(self, tmp_path: Path) -> None:
        payload = self._payload()
        candidates = cast(list[dict[str, object]], payload["candidates"])
        payload["candidates"] = candidates[:1]
        infile = self._write(tmp_path, payload)
        assert cmd_diverge(["--candidates", infile]) == 2

    def test_cli_exits_1_on_bad_input(self, tmp_path: Path) -> None:
        infile = self._write(tmp_path, {"prompt": PROMPT})
        assert cmd_diverge(["--candidates", infile]) == 1
        assert cmd_diverge([]) == 1

    def test_cli_exits_1_on_a_missing_file(self) -> None:
        assert cmd_diverge(["--candidates", "/nonexistent/diverge.json"]) == 1

    def test_cli_exits_1_on_a_bad_policy(self, tmp_path: Path) -> None:
        infile = self._write(tmp_path, self._payload())
        assert cmd_diverge(["--candidates", infile, "--max-pair-similarity", "2"]) == 1

    def test_cli_json_output_is_parseable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        infile = self._write(tmp_path, self._payload())
        assert cmd_diverge(["--candidates", infile, "--json"]) == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["selection"]["winner_index"] == 1


def test_demo_main_runs_end_to_end() -> None:
    """The demo runs green and returns 0.

    Running it under pytest also covers the demo module, which the
    tiered coverage floor counts but no test previously executed.
    """
    from agenteval_bench.diverge_demo import main

    assert main() == 0


# --- Property tests: invariants over every generated candidate set. ---

_text_alphabet = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=12
)


@given(st.lists(_text_alphabet, min_size=1, max_size=6))
@settings(max_examples=50)
def test_similarity_is_symmetric_and_reflexive(words: list[str]) -> None:
    left = " ".join(words)
    right = " ".join(reversed(words))
    assert pair_similarity(left, right) == pair_similarity(right, left)
    assert pair_similarity(left, left) == 1.0
    assert 0.0 <= pair_similarity(left, right) <= 1.0


@given(st.lists(_text_alphabet, min_size=2, max_size=8))
@settings(max_examples=50)
def test_diversity_stays_in_range(words: list[str]) -> None:
    texts = [" ".join(words[i:]) or "x" for i in range(5)]
    candidates = [Candidate(t, float(i) / 5.0) for i, t in enumerate(texts)]
    diverge_set = build_diverge_set(PROMPT, candidates, 0, 0.5, stakes="low")
    report = analyze_diversity(diverge_set)
    assert 0.0 <= report.min_pair_diversity <= 1.0
    assert 0.0 <= report.mean_pair_diversity <= 1.0
    assert 0.0 < report.modal_share <= 1.0
    assert sum(len(c) for c in report.clusters) == 5


@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=5, max_size=5))
@settings(max_examples=50)
def test_selection_is_deterministic(scores: list[float]) -> None:
    candidates = [Candidate(t, s) for t, s in zip([HASHMAP, SORT, SCAN, TRIE, CACHE], scores)]
    diverge_set = build_diverge_set(PROMPT, candidates, 0, 0.5, stakes="low")
    report = analyze_diversity(diverge_set)
    first = select(diverge_set, report, check_p_conventional(diverge_set, report))
    second = select(diverge_set, report, check_p_conventional(diverge_set, report))
    assert first == second
    assert first.winner_score == max(scores)
