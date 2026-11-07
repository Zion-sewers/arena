"""Tests for anti-mode-collapse /diverge (arena#47).

Every acceptance criterion has a named test, and the property tests
at the bottom pin the invariants for every generated candidate set:
similarity is symmetric and reflexive, diversity never exceeds 1,
and selection is deterministic.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agenteval_bench.cli import cmd_diverge
from agenteval_bench.decision import wilson_interval
from agenteval_bench.diverge import (
    Candidate,
    DivergePolicy,
    DivergeSet,
    HighStakesRefused,
    NotDiverseError,
    PairSimilarity,
    PConventionalCheck,
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
    p_check_summary,
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

# A longer near-verbatim pair: the one-word change keeps content
# Jaccard at 0.83, above the 0.80 bar, so the pair is convicted by
# content overlap alone — no single shared number does the work.
HASHMAP_LONG = "Use a hash map keyed by user id for O(1) lookup in the response cache"
HASHMAP_LONG_PARA = "Use a hash map keyed by user id for O(1) lookups in the response cache"
HASHMAP_LONG_ID = "Use a hash map keyed by user ID for O(1) lookup in the response cache"


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

    def test_content_free_candidates_share_no_content(self) -> None:
        # Two DISTINCT candidates with no content tokens at all
        # ("the" vs "a") share nothing to be a paraphrase of:
        # empty-vs-empty content similarity is 0.0, so a distinct
        # glue-only pair is not convicted as a near-duplicate and a
        # set of distinct glue-only candidates stays diverse. But
        # identical texts are always maximally similar: five copies
        # of "the" are one answer five times, and the set is refused.
        assert pair_similarity("the", "a") == 0.0
        assert pair_similarity("the", "the") == 1.0
        candidates = [
            Candidate("the", 0.70),
            Candidate("a", 0.90),
            Candidate("of the", 0.60),
            Candidate("to be", 0.80),
            Candidate("in on", 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 0, 0.50, stakes="low")
        assert analyze_diversity(diverge_set).diverse
        identical = [Candidate("the", s) for s in (0.70, 0.90, 0.60, 0.80, 0.75)]
        identical_set = build_diverge_set(PROMPT, identical, 0, 0.50, stakes="low")
        assert not analyze_diversity(identical_set).diverse

    def test_near_verbatim_rewrite_is_convicted_by_content_tokens(self) -> None:
        # A one-word change in a long text keeps content Jaccard at
        # 0.83: convicted by content overlap alone, with no help from
        # any shared number.
        assert pair_similarity(HASHMAP_LONG, HASHMAP_LONG_PARA) > 0.80

    def test_different_answers_sharing_one_number_are_not_paraphrases(self) -> None:
        # Regression: Jaccard on singleton hard-token sets used to
        # score these genuinely different answers as identical (1.0)
        # because both carry the single hard token "30". A lone shared
        # number is coincidence, not paraphrase evidence.
        assert pair_similarity("Retry after 30 seconds", "Delete after 30 days") < 0.80

    def test_synonym_rewrite_is_not_convicted_by_token_overlap(self) -> None:
        # Documented limit (see docs/DIVERGE.md): token-level
        # similarity cannot see a fluent synonym rewrite. "Employ a
        # hash table indexed by user id" is the same answer as the
        # hash-map text in different words, and it passes as diverse.
        # This test pins the actual power of the check so the claim
        # can never silently widen again.
        paraphrase = "Employ a hash table indexed by user id for O(1) lookup"
        assert pair_similarity(HASHMAP, paraphrase) < 0.80
        assert pair_similarity(CACHE, "Store results in Redis expiring after one minute") < 0.80

    def test_two_shared_numbers_do_not_convict_different_answers(self) -> None:
        # The hard-token anchor needs three shared numbers: Jaccard on
        # two-element sets is 0 or 1, so two shared numbers used to
        # score genuinely different answers as identical.
        assert pair_similarity("Set timeout 30 retries 3", "Set limit 30 attempts 3") < 0.80

    def test_three_shared_numbers_still_anchor_a_paraphrase(self) -> None:
        # Same three numbers, reworded glue, content overlap below the
        # bar: the hard-token anchor still convicts.
        assert (
            pair_similarity(
                "Use timeout 30 retries 2 backoff 5",
                "Set timeout 30 retries 2 backoff 5 now",
            )
            > 0.80
        )

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
            Candidate(HASHMAP_LONG, 0.70),
            Candidate(HASHMAP_LONG_PARA, 0.90),
            Candidate(SORT, 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 0, 0.20, stakes="high")
        report = analyze_diversity(diverge_set)
        assert not report.diverse
        assert report.min_pair_diversity < 0.20
        assert report.modal_cluster == (0, 1)
        assert report.modal_share == 0.40

    def test_three_way_paraphrase_cluster_is_the_modal_cluster(self) -> None:
        candidates = [
            Candidate(HASHMAP_LONG, 0.70),
            Candidate(HASHMAP_LONG_PARA, 0.90),
            Candidate(HASHMAP_LONG_ID, 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 0, 0.20, stakes="high")
        report = analyze_diversity(diverge_set)
        assert report.modal_cluster == (0, 1, 2)
        assert report.modal_share == 0.60

    def test_mean_similarity_names_the_most_distinctive_candidate(self) -> None:
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        assert len(report.mean_similarity_by_candidate) == 5
        assert all(0.0 <= s <= 1.0 for s in report.mean_similarity_by_candidate)
        # The cache answer shares no content token with any other
        # candidate, so it is the least similar on average.
        assert report.mean_similarity_by_candidate[4] == min(report.mean_similarity_by_candidate)

    def test_a_report_from_another_set_is_refused(self) -> None:
        _, first_set = _set()
        _, second_set = _set(scores=(0.70, 0.90, 0.60, 0.80, 0.71))
        first_report = analyze_diversity(first_set)
        with pytest.raises(ValueError, match="different candidate set"):
            check_p_conventional(second_set, first_report)
        with pytest.raises(ValueError, match="different candidate set"):
            select(second_set, first_report, check_p_conventional(first_set, first_report))

    def test_a_self_contradictory_report_is_refused_at_construction(self) -> None:
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        with pytest.raises(ValueError, match="contradicts the pair measurements"):
            replace(report, diverse=False)

    def test_a_forged_near_duplicate_flag_is_refused(self) -> None:
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        forged_pair = replace(report.pairs[0], near_duplicate=True)
        forged_pairs = (forged_pair,) + report.pairs[1:]
        with pytest.raises(ValueError, match="near_duplicate flag contradicts"):
            replace(report, pairs=forged_pairs)

    def test_forged_means_are_refused(self) -> None:
        # The tie-break reads the per-candidate means, so forged means
        # must not steer a tie: a report whose means do not match its
        # pairs is refused at construction.
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        with pytest.raises(ValueError, match="mean similarity for candidate"):
            replace(report, mean_similarity_by_candidate=(0.0, 0.0, 0.0, 0.0, 0.0))

    def test_forged_min_diversity_is_refused(self) -> None:
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        with pytest.raises(ValueError, match="min_pair_diversity must match"):
            replace(report, min_pair_diversity=1.0)

    def test_forged_clusters_are_refused(self) -> None:
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        with pytest.raises(ValueError, match="connected components"):
            replace(report, clusters=((0, 1), (2,), (3,), (4,)))

    def test_forged_modal_cluster_is_refused(self) -> None:
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        with pytest.raises(ValueError, match="modal_cluster must be the largest"):
            replace(report, modal_cluster=(4,))

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
            Candidate(HASHMAP_LONG, 0.70),
            Candidate(HASHMAP_LONG_PARA, 0.90),
            Candidate(HASHMAP_LONG_ID, 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 0, 0.05, stakes="high")
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        assert check.verdict == "underconfident"

    def test_conventional_minority_high_p_is_convicted(self) -> None:
        # Three of five candidates converge on a RIVAL answer while the
        # conventional answer is a singleton: declaring p 0.95 for the
        # conventional answer must be overconfident, because the check
        # runs on the conventional answer's own cluster (share 0.20),
        # not the rival's modal share (0.60), which would have blessed it.
        candidates = [
            Candidate(HASHMAP_LONG, 0.70),
            Candidate(HASHMAP_LONG_PARA, 0.90),
            Candidate(HASHMAP_LONG_ID, 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 4, 0.95, stakes="high")
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        assert check.conventional_share == 0.20
        assert check.verdict == "overconfident"

    def test_wide_interval_at_high_convergence_cannot_refute_any_p(self) -> None:
        # At a conventional share of 0.60 the Wilson upper bound +
        # tolerance reaches 1.00, so no declared p can be overconfident.
        # The check must say so in its notes instead of pretending to
        # have tested the claim.
        candidates = [
            Candidate(HASHMAP_LONG, 0.70),
            Candidate(HASHMAP_LONG_PARA, 0.90),
            Candidate(HASHMAP_LONG_ID, 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 0, 1.0, stakes="high")
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        assert check.verdict == "consistent"
        assert "cannot refute" in check.notes[0]

    def test_p_check_summary_is_plain_language(self) -> None:
        _, diverge_set = _set(p=0.95)
        report = analyze_diversity(diverge_set)
        over = check_p_conventional(diverge_set, report)
        assert "0.95" in p_check_summary(over)
        assert "re-diverge" in p_check_summary(over)
        _, consistent_set = _set(p=0.60)
        consistent = check_p_conventional(consistent_set, analyze_diversity(consistent_set))
        assert "does not refute" in p_check_summary(consistent)

    def test_a_forged_verdict_is_refused(self) -> None:
        _, diverge_set = _set(p=0.60)
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        assert check.verdict == "consistent"
        with pytest.raises(ValueError, match="contradicts the interval"):
            replace(check, verdict="overconfident")

    def test_a_forged_interval_is_refused(self) -> None:
        _, diverge_set = _set(p=0.60)
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        with pytest.raises(ValueError, match="must be the Wilson 95% interval"):
            replace(check, agreement_interval=(0.0, 1.0))

    def test_a_conventional_share_above_the_modal_share_is_refused(self) -> None:
        _, diverge_set = _set(p=0.60)
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        with pytest.raises(ValueError, match="cannot exceed the modal share"):
            replace(check, conventional_share=0.99)

    def test_a_forged_modal_share_is_refused(self) -> None:
        # The modal share is re-derived from the carried clusters: a
        # check claiming 0.99 while the modal cluster holds 1 of 5
        # is refused at construction.
        _, diverge_set = _set(p=0.60)
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        with pytest.raises(ValueError, match="modal_share must equal"):
            replace(check, modal_share=0.99)

    def test_a_forged_conventional_in_modal_is_refused(self) -> None:
        # The minority fixture converges on a rival (0, 1, 2) while
        # the conventional candidate (4) stands alone: flipping the
        # flag to True contradicts the carried clusters.
        candidates = [
            Candidate(HASHMAP_LONG, 0.70),
            Candidate(HASHMAP_LONG_PARA, 0.90),
            Candidate(HASHMAP_LONG_ID, 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 4, 0.95, stakes="high")
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        assert not check.conventional_in_modal
        with pytest.raises(ValueError, match="conventional_in_modal contradicts"):
            replace(check, conventional_in_modal=True)

    def test_p_check_refuses_a_report_graded_under_another_policy(self) -> None:
        # The clusters the p-check reads depend on the diversity
        # threshold, so a report graded under a 0.99 cap must not
        # validate a claim about a default-policy set.
        _, diverge_set = _set(p=0.60)
        report = analyze_diversity(diverge_set)
        lax_report = replace(report, max_pair_similarity=0.99)
        with pytest.raises(ValueError, match="different max_pair_similarity"):
            check_p_conventional(diverge_set, lax_report)

    def test_p_check_refuses_an_empty_cluster(self) -> None:
        # A trailing empty cluster passes the partition check (it
        # contributes no indices) but must not reach the modal
        # derivation as an IndexError: every invariant violation in
        # this module is a ValueError.
        _, diverge_set = _set(p=0.60)
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        with pytest.raises(ValueError, match="must not contain an empty cluster"):
            replace(check, clusters=check.clusters + ((),))

    def test_conventional_outside_the_modal_cluster_is_a_finding(self) -> None:
        candidates = [
            Candidate(HASHMAP_LONG, 0.70),
            Candidate(HASHMAP_LONG_PARA, 0.90),
            Candidate(HASHMAP_LONG_ID, 0.60),
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
        # Margin is winner over runner-up (0.90 - 0.80); the gap to the
        # conventional answer (0.90 - 0.70) is quoted separately.
        assert "margin over the runner-up 0.10" in selection.reason
        assert "gap 0.20" in selection.reason
        assert selection.runner_up_score == 0.80
        assert selection.margin == pytest.approx(0.10)

    def test_conventional_wins_and_the_reason_quotes_the_margin(self) -> None:
        _, diverge_set = _set(scores=(0.95, 0.90, 0.60, 0.80, 0.75))
        report = analyze_diversity(diverge_set)
        won = select(diverge_set, report, check_p_conventional(diverge_set, report))
        assert won.outcome == "conventional_won"
        assert "margin +0.05" in won.reason

    def test_tie_breaks_toward_the_most_distinctive_answer(self) -> None:
        # SORT and TRIE tie on top score (0.90). The old tie-break
        # (smallest cluster) was dead code: every diverse set has
        # all-singleton clusters, so it always reduced to lowest index.
        # The real tie-break is lowest MEAN pairwise similarity: TRIE
        # shares almost nothing with the other four (mean 0.02) while
        # SORT shares more (mean 0.08), so TRIE wins the tie even
        # though it is not the lowest index — and because the
        # conventional candidate (SORT, index 1) is in the tie, the
        # outcome is "tie", not a conventional win.
        candidates = [
            Candidate(HASHMAP, 0.70),
            Candidate(SORT, 0.90),
            Candidate(TRIE, 0.90),
            Candidate(CACHE, 0.75),
            Candidate(SCAN, 0.60),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 1, 0.40, stakes="low")
        report = analyze_diversity(diverge_set)
        tied = select(diverge_set, report, check_p_conventional(diverge_set, report))
        assert tied.outcome == "tie"
        assert tied.winner_index == 2
        assert "tie-break" in tied.reason

    def test_conventional_winning_a_top_score_tie_is_recorded_as_a_tie(self) -> None:
        # The conventional candidate ties at the top (0.90) and wins
        # the tie-break (lowest mean similarity): the margin is 0.00,
        # so recording "conventional_won ... won on score" would be
        # a false explanation. The outcome is "tie" and the reason
        # names the tie-break.
        candidates = [
            Candidate(HASHMAP, 0.90),
            Candidate(SORT, 0.90),
            Candidate(SCAN, 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 0, 0.60, stakes="low")
        report = analyze_diversity(diverge_set)
        won_tie = select(diverge_set, report, check_p_conventional(diverge_set, report))
        assert won_tie.outcome == "tie"
        assert won_tie.winner_index == 0
        assert won_tie.margin == 0.0
        assert "tie-break" in won_tie.reason
        assert "won on score" not in won_tie.reason
        # The corrected outcome is enforced, not just produced: the
        # record refuses to be built as a win on score.
        with pytest.raises(ValueError, match="contradicts the scores"):
            replace(won_tie, outcome="conventional_won")

    def test_a_forged_selection_is_refused_at_construction(self) -> None:
        selection = _diverse_selection()
        # A margin that is not winner-minus-runner-up.
        with pytest.raises(ValueError, match="margin must equal"):
            replace(selection, margin=0.20)
        # An outcome that contradicts the scores (the conventional
        # answer lost, it cannot be recorded as the winner).
        with pytest.raises(ValueError, match="contradicts the scores"):
            replace(selection, outcome="conventional_won")
        # An index outside the candidate set.
        with pytest.raises(ValueError, match="out of range"):
            replace(selection, winner_index=5)

    def test_a_forged_conventional_score_is_refused_at_construction(self) -> None:
        # The outcome derivation reads conventional_score, so it must
        # match the carried vector like every other score: forging it
        # flips a lost conventional answer to tied, and a tied one to
        # lost, without touching the winner or the scores.
        selection = _diverse_selection()
        assert selection.outcome == "conventional_lost"
        with pytest.raises(ValueError, match="conventional_score must equal"):
            replace(selection, conventional_score=0.90, outcome="tie")
        with pytest.raises(ValueError, match="conventional_score must equal"):
            replace(selection, conventional_score=0.10)
        candidates = [
            Candidate(HASHMAP, 0.70),
            Candidate(SORT, 0.90),
            Candidate(TRIE, 0.90),
            Candidate(CACHE, 0.75),
            Candidate(SCAN, 0.60),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 1, 0.40, stakes="low")
        report = analyze_diversity(diverge_set)
        tied = select(diverge_set, report, check_p_conventional(diverge_set, report))
        assert tied.outcome == "tie"
        with pytest.raises(ValueError, match="conventional_score must equal"):
            replace(tied, conventional_score=0.60, outcome="conventional_lost")

    def test_a_within_tolerance_forged_winner_score_cannot_flip_a_tie(self) -> None:
        # The tie test is tolerance-consistent with the margin check:
        # a winner_score 5e-13 above a tied top score must not rebuild
        # the false "conventional_won ... won on score" record.
        candidates = [
            Candidate(HASHMAP, 0.90),
            Candidate(SORT, 0.90),
            Candidate(SCAN, 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        diverge_set = build_diverge_set(PROMPT, candidates, 0, 0.60, stakes="low")
        report = analyze_diversity(diverge_set)
        won_tie = select(diverge_set, report, check_p_conventional(diverge_set, report))
        assert won_tie.outcome == "tie"
        with pytest.raises(ValueError, match="contradicts the scores"):
            replace(
                won_tie,
                outcome="conventional_won",
                winner_score=0.9000000000005,
                runner_up_score=0.90,
                margin=0.0000000000005,
            )

    def test_a_non_argmax_selection_is_refused_at_construction(self) -> None:
        # The selection carries the full score vector, so a winner
        # that does not hold the top score is refused: the forgery
        # (winner_index 2, whose own score is 0.60, recorded with
        # the top score's 0.90) cannot be built, and neither can a
        # record whose scores contradict its winner or runner-up.
        selection = _diverse_selection()
        assert selection.winner_index == 1
        with pytest.raises(ValueError, match="not the winner_score"):
            replace(selection, winner_index=2)
        with pytest.raises(ValueError, match="not the top score"):
            replace(selection, scores=(0.70, 0.60, 0.60, 0.80, 0.75))
        with pytest.raises(ValueError, match="runner_up_score must be the best"):
            replace(selection, scores=(0.70, 0.90, 0.60, 0.85, 0.75))

    def test_a_rounded_literal_margin_is_accepted(self) -> None:
        # Float subtraction is not exact (0.95 - 0.90 is
        # 0.04999999999999993, not 0.05): a mathematically correct
        # margin written as a literal must not be refused as forged.
        _, diverge_set = _set(scores=(0.95, 0.90, 0.60, 0.80, 0.75))
        report = analyze_diversity(diverge_set)
        won = select(diverge_set, report, check_p_conventional(diverge_set, report))
        assert replace(won, margin=0.05).margin == 0.05

    def test_selection_refuses_a_paraphrase_set(self) -> None:
        candidates = [
            Candidate(HASHMAP_LONG, 0.70),
            Candidate(HASHMAP_LONG_PARA, 0.90),
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

    def test_high_stakes_needs_exactly_five_by_policy(self) -> None:
        # A policy that shrinks the count to 2 must not let a
        # 2-candidate set through the high-stakes gate: the issue's
        # contract is 5 candidates, on the library seam as well as
        # the CLI.
        policy = DivergePolicy(n_candidates=2)
        candidates = [Candidate(HASHMAP, 0.70), Candidate(SORT, 0.90)]
        with pytest.raises(SingleSampleError, match="exactly 5"):
            build_diverge_set(PROMPT, candidates, 0, 0.50, stakes="high", policy=policy)

    def test_gate_refuses_a_short_selection_built_directly(self) -> None:
        # The gate enforces the count from the selection record: a
        # 2-candidate set built low-stakes through the public
        # constructor selects fine, but it must not decide high stakes.
        policy = DivergePolicy(n_candidates=2)
        candidates = [Candidate(HASHMAP, 0.70), Candidate(SORT, 0.90)]
        small = build_diverge_set(PROMPT, candidates, 0, 0.50, stakes="low", policy=policy)
        report = analyze_diversity(small)
        selection = select(small, report, check_p_conventional(small, report))
        assert selection.n == 2
        assert not may_decide_high_stakes(selection)
        with pytest.raises(HighStakesRefused, match="exactly 5"):
            require_high_stakes_decision(selection)

    def test_gate_refuses_a_six_candidate_selection(self) -> None:
        policy = DivergePolicy(n_candidates=6)
        candidates = [
            Candidate(HASHMAP, 0.70),
            Candidate(SORT, 0.90),
            Candidate(SCAN, 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
            Candidate("Restart the service and retry the request", 0.65),
        ]
        wide = build_diverge_set(PROMPT, candidates, 0, 0.50, stakes="low", policy=policy)
        report = analyze_diversity(wide)
        selection = select(wide, report, check_p_conventional(wide, report))
        assert selection.n == 6
        assert not may_decide_high_stakes(selection)
        with pytest.raises(HighStakesRefused, match="exactly 5"):
            require_high_stakes_decision(selection)

    def test_gate_refuses_a_lax_policy_selection(self) -> None:
        # A paraphrase pair passes a 0.99 cap, so the set is diverse
        # and the p-check is consistent — but the grades were computed
        # under a lax policy the gate cannot trust, so the selection
        # may not decide high stakes.
        candidates = [
            Candidate(HASHMAP_LONG, 0.70),
            Candidate(HASHMAP_LONG_PARA, 0.90),
            Candidate(SORT, 0.60),
            Candidate(TRIE, 0.80),
            Candidate(CACHE, 0.75),
        ]
        lax = build_diverge_set(
            PROMPT,
            candidates,
            0,
            0.20,
            stakes="low",
            policy=DivergePolicy(max_pair_similarity=0.99),
        )
        report = analyze_diversity(lax)
        assert report.diverse
        selection = select(lax, report, check_p_conventional(lax, report))
        assert not may_decide_high_stakes(selection)
        with pytest.raises(HighStakesRefused, match="non-canonical"):
            require_high_stakes_decision(selection)

    def test_select_refuses_a_report_graded_under_another_policy(self) -> None:
        # A diverse report re-graded under a 0.99 threshold is still
        # internally consistent (no near-duplicate flags to contradict),
        # but it must not compose into a selection from a default-policy set.
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        lax_report = replace(report, max_pair_similarity=0.99)
        with pytest.raises(ValueError, match="different max_pair_similarity"):
            select(diverge_set, lax_report, check_p_conventional(diverge_set, report))

    def test_select_refuses_a_p_check_graded_under_another_tolerance(self) -> None:
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        lax_check = replace(check, tolerance=0.99)
        with pytest.raises(ValueError, match="different tolerance"):
            select(diverge_set, report, lax_check)

    def test_select_refuses_a_p_check_on_foreign_clusters(self) -> None:
        # A check graded on all-in-one clusters (share 1.00, p 0.95
        # consistent) must not compose with an all-singleton report
        # (real share 0.20): the verdict would vouch for a measurement
        # the report does not make.
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        low, high = wilson_interval(5, 5)
        forged = replace(
            check,
            clusters=((0, 1, 2, 3, 4),),
            modal_share=1.0,
            conventional_share=1.0,
            agreement_interval=(low, high),
            declared_p=0.95,
        )
        assert forged.verdict == "consistent"
        with pytest.raises(ValueError, match="clusters do not match"):
            select(diverge_set, report, forged)

    def test_select_refuses_a_p_check_with_a_foreign_declared_p(self) -> None:
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        forged = replace(check, declared_p=0.50)
        assert forged.verdict == "consistent"
        with pytest.raises(ValueError, match="does not match the set's"):
            select(diverge_set, report, forged)

    def test_select_refuses_a_p_check_naming_another_conventional_candidate(self) -> None:
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        check = check_p_conventional(diverge_set, report)
        forged = replace(check, conventional_index=1, conventional_in_modal=False)
        with pytest.raises(ValueError, match="different conventional candidate"):
            select(diverge_set, report, forged)

    def test_check_p_conventional_refuses_a_report_covering_another_n(self) -> None:
        # Round-5 H2/NEW1: a pair of hand-built records that agree
        # with each other on n=3 (fingerprint copied) must not
        # validate a claim about a 5-candidate set — the Wilson
        # interval at n=3 does not refute what n=5 refutes.
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        forged = replace(
            report,
            n=3,
            pairs=tuple(
                PairSimilarity(i, j, 0.0, False) for i in range(3) for j in range(i + 1, 3)
            ),
            clusters=((0,), (1,), (2,)),
            modal_cluster=(0,),
            modal_share=1 / 3,
            min_pair_diversity=1.0,
            mean_pair_diversity=1.0,
            mean_similarity_by_candidate=(0.0, 0.0, 0.0),
        )
        with pytest.raises(ValueError, match="different-sized set"):
            check_p_conventional(diverge_set, forged)

    def test_check_p_conventional_refuses_a_short_report_with_a_high_conventional_index(
        self,
    ) -> None:
        # Same n gap, second-order effect: with conventional_index 4
        # and a forged n=3 report, the pre-fix code raised raw
        # StopIteration from the cluster lookup; the n check converts
        # it into a clean ValueError.
        _, diverge_set = _set(conventional=4)
        report = analyze_diversity(diverge_set)
        forged = replace(
            report,
            n=3,
            pairs=tuple(
                PairSimilarity(i, j, 0.0, False) for i in range(3) for j in range(i + 1, 3)
            ),
            clusters=((0,), (1,), (2,)),
            modal_cluster=(0,),
            modal_share=1 / 3,
            min_pair_diversity=1.0,
            mean_pair_diversity=1.0,
            mean_similarity_by_candidate=(0.0, 0.0, 0.0),
        )
        with pytest.raises(ValueError, match="different-sized set"):
            check_p_conventional(diverge_set, forged)

    def test_check_p_conventional_refuses_the_p_laundering_n4_report(self) -> None:
        # Round-5 H2 replay: declared p 0.80 is overconfident on the
        # true n=5 measurement (Wilson(1,5) upper 0.6245 + 0.15) but
        # consistent on a forged n=4 one (Wilson(1,4) upper 0.6994 +
        # 0.15) — the check must refuse the foreign-n report before
        # any verdict is computed.
        _, diverge_set = _set(p=0.80)
        assert check_p_conventional(diverge_set, analyze_diversity(diverge_set)).verdict == (
            "overconfident"
        )
        report = analyze_diversity(diverge_set)
        forged = replace(
            report,
            n=4,
            pairs=tuple(
                PairSimilarity(i, j, 0.0, False) for i in range(4) for j in range(i + 1, 4)
            ),
            clusters=((0,), (1,), (2,), (3,)),
            modal_cluster=(0,),
            modal_share=0.25,
            min_pair_diversity=1.0,
            mean_pair_diversity=1.0,
            mean_similarity_by_candidate=(0.0, 0.0, 0.0, 0.0),
        )
        with pytest.raises(ValueError, match="different-sized set"):
            check_p_conventional(diverge_set, forged)

    def test_select_refuses_a_report_covering_another_n(self) -> None:
        # The n=3 forgery, end to end through select(): the forged
        # check is built to match the forged report (same n, same
        # clusters, same declared p), so only the n-vs-set check can
        # refuse the composition.
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        forged_report = replace(
            report,
            n=3,
            pairs=tuple(
                PairSimilarity(i, j, 0.0, False) for i in range(3) for j in range(i + 1, 3)
            ),
            clusters=((0,), (1,), (2,)),
            modal_cluster=(0,),
            modal_share=1 / 3,
            min_pair_diversity=1.0,
            mean_pair_diversity=1.0,
            mean_similarity_by_candidate=(0.0, 0.0, 0.0),
        )
        low, high = wilson_interval(1, 3)
        forged_check = PConventionalCheck(
            declared_p=0.60,
            modal_share=1 / 3,
            conventional_share=1 / 3,
            agreement_interval=(low, high),
            verdict="consistent",
            conventional_in_modal=True,
            notes=(),
            set_fingerprint=diverge_set.fingerprint(),
            tolerance=0.15,
            n=3,
            conventional_index=0,
            clusters=((0,), (1,), (2,)),
        )
        with pytest.raises(ValueError, match="different-sized set"):
            select(diverge_set, forged_report, forged_check)

    def test_select_refuses_a_report_covering_seven_candidates(self) -> None:
        # The n=7 variant of the same forgery: 21 zero-similarity
        # pairs, 7 singleton clusters, matching forged check.
        _, diverge_set = _set()
        report = analyze_diversity(diverge_set)
        forged_report = replace(
            report,
            n=7,
            pairs=tuple(
                PairSimilarity(i, j, 0.0, False) for i in range(7) for j in range(i + 1, 7)
            ),
            clusters=((0,), (1,), (2,), (3,), (4,), (5,), (6,)),
            modal_cluster=(0,),
            modal_share=1 / 7,
            min_pair_diversity=1.0,
            mean_pair_diversity=1.0,
            mean_similarity_by_candidate=(0.0,) * 7,
        )
        low, high = wilson_interval(1, 7)
        forged_check = PConventionalCheck(
            declared_p=0.60,
            modal_share=1 / 7,
            conventional_share=1 / 7,
            agreement_interval=(low, high),
            verdict="consistent",
            conventional_in_modal=True,
            notes=(),
            set_fingerprint=diverge_set.fingerprint(),
            tolerance=0.15,
            n=7,
            conventional_index=0,
            clusters=((0,), (1,), (2,), (3,), (4,), (5,), (6,)),
        )
        with pytest.raises(ValueError, match="different-sized set"):
            select(diverge_set, forged_report, forged_check)

    def test_select_refuses_a_short_report_when_the_top_score_is_outside_it(self) -> None:
        # Second-order effect: with the top score at index 4 and a
        # forged n=3 report, the pre-fix tie-break raised raw
        # IndexError from the per-candidate means lookup; the n check
        # converts it into a clean ValueError.
        _, diverge_set = _set(scores=(0.70, 0.60, 0.50, 0.40, 0.90))
        report = analyze_diversity(diverge_set)
        forged_report = replace(
            report,
            n=3,
            pairs=tuple(
                PairSimilarity(i, j, 0.0, False) for i in range(3) for j in range(i + 1, 3)
            ),
            clusters=((0,), (1,), (2,)),
            modal_cluster=(0,),
            modal_share=1 / 3,
            min_pair_diversity=1.0,
            mean_pair_diversity=1.0,
            mean_similarity_by_candidate=(0.0, 0.0, 0.0),
        )
        low, high = wilson_interval(1, 3)
        forged_check = PConventionalCheck(
            declared_p=0.60,
            modal_share=1 / 3,
            conventional_share=1 / 3,
            agreement_interval=(low, high),
            verdict="consistent",
            conventional_in_modal=True,
            notes=(),
            set_fingerprint=diverge_set.fingerprint(),
            tolerance=0.15,
            n=3,
            conventional_index=0,
            clusters=((0,), (1,), (2,)),
        )
        with pytest.raises(ValueError, match="different-sized set"):
            select(diverge_set, forged_report, forged_check)

    def test_gate_refuses_a_tolerance_only_lax_policy_selection(self) -> None:
        # A declared p of 0.95 is consistent only under a 0.99
        # tolerance; the gate pins the canonical 0.15, so the
        # selection may not decide high stakes.
        candidates, _ = _set()
        lax = build_diverge_set(
            PROMPT,
            candidates,
            0,
            0.95,
            stakes="low",
            policy=DivergePolicy(p_conventional_tolerance=0.99),
        )
        report = analyze_diversity(lax)
        selection = select(lax, report, check_p_conventional(lax, report))
        assert selection.p_verdict == "consistent"
        assert not may_decide_high_stakes(selection)
        with pytest.raises(HighStakesRefused, match="non-canonical"):
            require_high_stakes_decision(selection)

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

    def test_build_diverge_set_refuses_an_unknown_stakes_value(self) -> None:
        # The library seam validates stakes like the loader does: an
        # unknown value is a caller error, never silently low — a
        # typo'd "High" on a short list must not skip the typed
        # SingleSampleError.
        candidates, _ = _set()
        with pytest.raises(ValueError, match="stakes must be 'low' or 'high'"):
            build_diverge_set(PROMPT, candidates, 0, 0.50, stakes=cast(Stakes, "critical"))
        with pytest.raises(ValueError, match="stakes must be 'low' or 'high'"):
            build_diverge_set(PROMPT, candidates, 0, 0.50, stakes=cast(Stakes, "High"))
        with pytest.raises(SingleSampleError, match="not sufficient"):
            build_diverge_set(PROMPT, candidates[:1], 0, 0.50, stakes="high")


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
                json.dumps(
                    {"prompt": 42, "candidates": [], "conventional_index": 0, "p_conventional": 0.5}
                )
            )

    def test_loader_refuses_bad_stakes(self) -> None:
        with pytest.raises(ValueError, match="stakes must be 'low' or 'high'"):
            candidates_from_json(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "stakes": "critical",
                        "candidates": [],
                        "conventional_index": 0,
                        "p_conventional": 0.5,
                    }
                )
            )

    def test_loader_refuses_a_bool_conventional_index(self) -> None:
        with pytest.raises(TypeError, match="conventional_index must be an int"):
            candidates_from_json(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "candidates": [],
                        "conventional_index": True,
                        "p_conventional": 0.5,
                    }
                )
            )

    def test_loader_refuses_a_bool_p_conventional(self) -> None:
        with pytest.raises(TypeError, match="p_conventional must be a number"):
            candidates_from_json(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "candidates": [],
                        "conventional_index": 0,
                        "p_conventional": False,
                    }
                )
            )

    def test_loader_refuses_non_list_candidates(self) -> None:
        with pytest.raises(TypeError, match="candidates must be a list"):
            candidates_from_json(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "candidates": "five",
                        "conventional_index": 0,
                        "p_conventional": 0.5,
                    }
                )
            )

    def test_loader_refuses_a_non_mapping_candidate(self) -> None:
        with pytest.raises(TypeError, match="must be a mapping"):
            candidates_from_json(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "candidates": ["five"],
                        "conventional_index": 0,
                        "p_conventional": 0.5,
                    }
                )
            )

    def test_loader_refuses_unknown_candidate_keys(self) -> None:
        with pytest.raises(ValueError, match="unknown keys"):
            candidates_from_json(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "candidates": [{"text": HASHMAP, "score": 0.7, "rank": 1}],
                        "conventional_index": 0,
                        "p_conventional": 0.5,
                    }
                )
            )

    def test_loader_refuses_a_non_string_candidate_text(self) -> None:
        with pytest.raises(TypeError, match="text must be a string"):
            candidates_from_json(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "candidates": [{"text": 42, "score": 0.7}],
                        "conventional_index": 0,
                        "p_conventional": 0.5,
                    }
                )
            )

    def test_loader_refuses_a_non_string_candidate_label(self) -> None:
        with pytest.raises(TypeError, match="label must be a string"):
            candidates_from_json(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "candidates": [{"text": HASHMAP, "score": 0.7, "label": 3}],
                        "conventional_index": 0,
                        "p_conventional": 0.5,
                    }
                )
            )

    def test_non_finite_p_conventional_is_refused(self) -> None:
        candidates, _ = _set()
        with pytest.raises(ValueError, match="finite"):
            build_diverge_set(PROMPT, candidates, 0, float("inf"), stakes="low")

    def test_loader_refuses_a_non_finite_p_conventional(self) -> None:
        # json.loads accepts NaN; the loader must refuse it itself
        # rather than passing it downstream to the DivergeSet check.
        with pytest.raises(ValueError, match="finite"):
            candidates_from_json(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "candidates": [],
                        "conventional_index": 0,
                        "p_conventional": float("nan"),
                    }
                )
            )

    def test_loader_refuses_a_p_conventional_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            candidates_from_json(
                json.dumps(
                    {
                        "prompt": PROMPT,
                        "candidates": [],
                        "conventional_index": 0,
                        "p_conventional": 1.5,
                    }
                )
            )

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
        candidates[0] = {"text": HASHMAP_LONG, "score": 0.70}
        candidates[1] = {"text": HASHMAP_LONG_PARA, "score": 0.90}
        infile = self._write(tmp_path, payload)
        assert cmd_diverge(["--candidates", infile]) == 2

    def test_cli_exits_2_on_the_shipped_refusal_examples(self) -> None:
        examples = Path(__file__).resolve().parent.parent / "examples"
        assert cmd_diverge(["--candidates", str(examples / "diverge-paraphrase.json")]) == 2
        assert cmd_diverge(["--candidates", str(examples / "diverge-overconfident.json")]) == 2

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

    def test_cli_names_the_token_level_scope(self, tmp_path: Path) -> None:
        infile = self._write(tmp_path, self._payload())
        out = str(tmp_path / "report.json")
        assert cmd_diverge(["--candidates", infile, "--out", out]) == 0
        written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert written["diversity_scope"] == "token-level near-duplicate"

    def test_cli_human_output_names_the_token_level_scope(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        infile = self._write(tmp_path, self._payload())
        assert cmd_diverge(["--candidates", infile]) == 0
        assert "token-level only" in capsys.readouterr().out


class TestFingerprint:
    def test_fingerprint_is_canonical_across_number_spellings(self) -> None:
        # Int and float spellings of the same set — score 1 vs 1.0,
        # p 1 vs 1.0, score -0.0 vs 0.0 — must fingerprint alike, or
        # a report from one spelling is refused for the other.
        _, int_set = _set(scores=(1, 0, 0, 0, 0), p=1)
        _, float_set = _set(scores=(1.0, 0.0, 0.0, 0.0, 0.0), p=1.0)
        assert int_set.fingerprint() == float_set.fingerprint()
        _, neg_zero_set = _set(scores=(-0.0, 0.0, 0.0, 0.0, 0.0), p=0.0)
        _, pos_zero_set = _set(scores=(0.0, 0.0, 0.0, 0.0, 0.0), p=0.0)
        assert neg_zero_set.fingerprint() == pos_zero_set.fingerprint()


def test_verdict_domains_are_exported_at_top_level() -> None:
    from typing import get_args

    from agenteval_bench import (
        PCheckVerdict,
        SelectionOutcome,
        content_tokens,
        hard_tokens,
    )

    assert get_args(PCheckVerdict) == ("consistent", "overconfident", "underconfident")
    assert get_args(SelectionOutcome) == ("conventional_won", "conventional_lost", "tie")
    assert "hash" in content_tokens(HASHMAP)
    assert "60s" in hard_tokens(CACHE)


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
    # Tolerance, not exact equality: the tie-break in select() treats
    # scores within 1e-12 of the top as tied, so a tie-break winner
    # can sit up to 1e-12 below the strict max (e.g. 0.0 vs 7.6e-268).
    assert abs(first.winner_score - max(scores)) <= 1e-12


def test_selection_survives_sub_tolerance_score_spread() -> None:
    # Regression: hypothesis found scores [0, 0, 0, 7.6e-268, 0] —
    # all tied at 1e-12 — where the tie-break winner (score 0.0) had
    # a runner-up (7.6e-268) strictly above it, and Selection refused
    # its own pipeline-built record with ValueError.
    scores = [0.0, 0.0, 0.0, 7.591571019389426e-268, 0.0]
    candidates = [Candidate(t, s) for t, s in zip([HASHMAP, SORT, SCAN, TRIE, CACHE], scores)]
    diverge_set = build_diverge_set(PROMPT, candidates, 0, 0.5, stakes="low")
    report = analyze_diversity(diverge_set)
    first = select(diverge_set, report, check_p_conventional(diverge_set, report))
    second = select(diverge_set, report, check_p_conventional(diverge_set, report))
    assert first == second
    assert first.outcome == "tie"
    assert abs(first.winner_score - max(scores)) <= 1e-12
    assert first.runner_up_score - first.winner_score <= 1e-12
    assert first.margin >= -1e-12
    # The margin is clamped at 0.0: a tie-break winner sitting up to
    # 1e-12 below the strict top must not print as "-0.00".
    assert first.margin == 0.0
    assert "-0.00" not in first.reason


def test_glue_variants_differing_only_by_case_or_punctuation_are_identical() -> None:
    # Round-5 NEW2 sub-case: identity is judged after normalization
    # (lowercased word tokens), so case / whitespace / punctuation
    # variants of one glue answer are the same answer, not five
    # diverse candidates — while distinct glue stays 0.0.
    assert pair_similarity("the", "The") == 1.0
    assert pair_similarity("the", "THE") == 1.0
    assert pair_similarity("the", "the ") == 1.0
    assert pair_similarity("the", "the.") == 1.0
    assert pair_similarity("the", "the!") == 1.0
    assert pair_similarity("the", "a") == 0.0
    variants = build_diverge_set(
        PROMPT,
        [
            Candidate(t, s)
            for t, s in zip(("the", "The", "THE", "the ", "the."), (0.70, 0.90, 0.60, 0.80, 0.75))
        ],
        0,
        0.5,
        stakes="low",
    )
    report = analyze_diversity(variants)
    assert not report.diverse
    with pytest.raises(NotDiverseError):
        select(variants, report, check_p_conventional(variants, report))
