"""Adversarial tests for coverage-gated ranking (arena#49).

Every behavior change ships with a running test — prose is not enforcement.
The pins that matter here are the traps: a judge that wins every
comparison it enters on 20 items must never outrank a fully-covered
judge; a 5/5 cell must fall back far below a 50/50 cell at the same
100% point rate; and a leaderboard with no rankable judge must refuse
instead of ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agenteval_bench.cli import cmd_alt_test
from agenteval_bench.coverage import (
    CellCoverage,
    CellObservation,
    CoveragePolicy,
    JudgeCoverage,
    RankedLeaderboard,
    RankingRefused,
    _recover_wins,
    assess_cell,
    assess_observations,
    assess_ranking,
    default_coverage_policy,
    render_ranked,
    require_ranking,
)
from agenteval_bench.decision import wilson_interval
from alt_test.demo import make_dataset
from alt_test.runner import run_alt_test
from alt_test.types import AltTestConfig, AltTestItem, AltTestReport

POLICY = CoveragePolicy(min_per_cell=50, min_total=50)


def _write_jsonl(items: list[AltTestItem], path: Path) -> None:
    rows = [
        json.dumps(
            {
                "id": item.id,
                "task": item.task,
                "annotators": dict(item.annotator_labels),
                "judges": dict(item.judge_labels),
            }
        )
        for item in items
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def _thin_dataset(n_items: int = 120, n_labelled: int = 20) -> list[AltTestItem]:
    """Demo dataset plus a thin judge labelling only the first N items.

    The thin judge copies the majority-oracle label, so on its items it
    wins every comparison — the trap is that it must still not rank.
    """
    out: list[AltTestItem] = []
    for i, item in enumerate(make_dataset(n_items=n_items)):
        judges = dict(item.judge_labels)
        if i < n_labelled:
            judges["thin-judge"] = item.judge_labels["majority-oracle"]
        out.append(
            AltTestItem(
                id=item.id,
                task=item.task,
                annotator_labels=dict(item.annotator_labels),
                judge_labels=judges,
            )
        )
    return out


def _report(n_items: int = 120, n_labelled: int = 20) -> AltTestReport:
    return run_alt_test(
        _thin_dataset(n_items, n_labelled),
        AltTestConfig(epsilon=0.1, q=0.05, seed=7),
    )


# --- Policy ---------------------------------------------------------------


def test_policy_rejects_impossible_floors() -> None:
    with pytest.raises(ValueError, match="min_per_cell"):
        CoveragePolicy(min_per_cell=1)
    with pytest.raises(ValueError, match="min_total"):
        CoveragePolicy(min_total=0)
    assert default_coverage_policy() == CoveragePolicy(min_per_cell=50, min_total=50)


def test_cell_observation_rejects_corrupt_counts() -> None:
    with pytest.raises(ValueError, match="wins"):
        CellObservation("ann0", "judge", 10, 11)
    with pytest.raises(ValueError, match="n must be"):
        CellObservation("ann0", "judge", -1, 0)
    with pytest.raises(ValueError, match="state"):
        CellObservation("", "judge", 10, 5)


# --- Cell boundary --------------------------------------------------------


def test_cell_boundary_49_sparse_50_covered() -> None:
    assert assess_cell("s", "a", 49, 49, 50).status == "sparse"
    assert assess_cell("s", "a", 50, 50, 50).status == "covered"
    assert assess_cell("s", "a", 0, 0, 50).status == "sparse"


def test_cell_coverage_forgery_refused() -> None:
    # A hand-built record claiming COVERED on a thin cell is refused at
    # construction, before it can reach any ranking.
    with pytest.raises(ValueError, match="is sparse"):
        CellCoverage(state="s", action="a", n=10, wins=10, min_per_cell=50, status="covered")
    with pytest.raises(ValueError, match="is covered"):
        CellCoverage(state="s", action="a", n=60, wins=60, min_per_cell=50, status="sparse")


# --- Conservative fallback --------------------------------------------------


def test_conservative_fallback_is_wilson_lower_bound() -> None:
    cell = assess_cell("s", "a", 50, 40, 50)
    lower, _upper = wilson_interval(40, 50)
    assert cell.conservative_rate == pytest.approx(lower)
    assert cell.point_rate == pytest.approx(0.8)
    assert cell.conservative_rate < cell.point_rate


def test_thin_perfect_cell_falls_back_far_below_covered_perfect_cell() -> None:
    thin = assess_cell("s", "a", 5, 5, 50)
    covered = assess_cell("s", "a", 50, 50, 50)
    assert thin.point_rate == covered.point_rate == 1.0
    assert 0.5 < thin.conservative_rate < 0.6
    assert covered.conservative_rate > 0.9
    assert thin.conservative_rate < covered.conservative_rate


def test_empty_cell_conservative_rate_is_zero() -> None:
    assert assess_cell("s", "a", 0, 0, 50).conservative_rate == 0.0


def test_zero_wins_conservative_rate_is_exactly_zero() -> None:
    # Regression (hypothesis found it, CI confirmed it): the Wilson
    # centre/half-width cancel exactly at 0 wins in real arithmetic,
    # but in floats the subtraction leaves dust (6.9e-18 at n=47). A
    # conservative fallback must never exceed the measured rate.
    for n in (2, 10, 47, 50, 120, 200):
        assert assess_cell("s", "a", n, 0, 50).conservative_rate == 0.0


@settings(max_examples=50, deadline=None)
@given(st.integers(min_value=1, max_value=500))
def test_lower_bound_grows_with_n_at_fixed_perfect_rate(n: int) -> None:
    # The fallback punishes thin data by interval width: at a fixed 100%
    # point rate, more observations must never lower the fallback.
    assert (
        assess_cell("s", "a", n, n, 50).conservative_rate
        <= assess_cell("s", "a", n + 1, n + 1, 50).conservative_rate
    )


@settings(max_examples=50, deadline=None)
@given(
    st.integers(min_value=2, max_value=200),
    st.integers(min_value=0, max_value=200),
)
def test_conservative_rate_always_inside_unit_interval(n: int, wins: int) -> None:
    wins = min(wins, n)
    rate = assess_cell("s", "a", n, wins, 50).conservative_rate
    assert 0.0 <= rate <= 1.0
    assert rate <= wins / n


# --- Judge coverage -------------------------------------------------------


def test_judge_rankable_only_when_every_cell_covered() -> None:
    full = JudgeCoverage(
        "judge",
        tuple(assess_cell(f"ann{a}", "judge", 60, 50, 50) for a in range(5)),
    )
    assert full.rankable
    assert full.coverage_fraction == 1.0
    assert full.conservative_score is not None
    assert full.diagnostic_score is not None
    assert full.conservative_score == full.diagnostic_score

    one_sparse = JudgeCoverage(
        "judge",
        tuple(
            assess_cell(f"ann{a}", "judge", 60 if a else 20, 50 if a else 20, 50) for a in range(5)
        ),
    )
    assert not one_sparse.rankable
    assert one_sparse.coverage_fraction == pytest.approx(0.8)
    assert len(one_sparse.masked_cells) == 1
    assert one_sparse.masked_cells[0].state == "ann0"
    assert one_sparse.thinnest_cell is not None
    assert one_sparse.thinnest_cell.n == 20
    # The ranked score ignores the masked cell; the diagnostic includes it
    # at its (wide-interval) lower bound, so the two scores differ by
    # exactly the masked cell's contribution.
    assert one_sparse.conservative_score is not None
    assert one_sparse.diagnostic_score is not None
    assert one_sparse.conservative_score == pytest.approx(
        sum(c.conservative_rate for c in one_sparse.covered_cells) / 4
    )
    assert one_sparse.diagnostic_score == pytest.approx(
        sum(c.conservative_rate for c in one_sparse.cells) / 5
    )
    assert one_sparse.diagnostic_score != one_sparse.conservative_score


def test_judge_with_no_covered_cell_has_no_ranked_score() -> None:
    thin = JudgeCoverage(
        "judge", tuple(assess_cell(f"ann{a}", "judge", 10, 10, 50) for a in range(5))
    )
    assert not thin.rankable
    assert thin.conservative_score is None
    assert thin.diagnostic_score is not None
    assert thin.coverage_fraction == 0.0


def test_judge_cells_must_belong_to_the_judge() -> None:
    with pytest.raises(ValueError, match="does not belong"):
        JudgeCoverage("a", (assess_cell("s", "b", 60, 60, 50),))


def test_ranked_leaderboard_refuses_misplaced_judges() -> None:
    rankable = JudgeCoverage("a", tuple(assess_cell(f"ann{i}", "a", 60, 60, 50) for i in range(2)))
    masked = JudgeCoverage("b", tuple(assess_cell(f"ann{i}", "b", 10, 10, 50) for i in range(2)))
    with pytest.raises(ValueError, match="not rankable"):
        RankedLeaderboard(ranked=(masked,), masked=(), n_items=60, min_per_cell=50, min_total=50)
    with pytest.raises(ValueError, match="is rankable"):
        RankedLeaderboard(ranked=(), masked=(rankable,), n_items=60, min_per_cell=50, min_total=50)


# --- The ranking trap -----------------------------------------------------


def test_thin_perfect_judge_is_masked_not_ranked() -> None:
    report = _report()
    thin_verdict = next(v for v in report.verdicts if v.judge == "thin-judge")
    # The trap, confirmed ungated: the thin judge wins everything it enters.
    assert thin_verdict.rho == 1.0
    assert thin_verdict.omega == 1.0
    oracle = next(v for v in report.verdicts if v.judge == "majority-oracle")
    assert thin_verdict.rho == oracle.rho

    board = assess_ranking(report, POLICY)
    assert board.is_ranking
    assert "thin-judge" not in [j.judge for j in board.ranked]
    assert "thin-judge" in [j.judge for j in board.masked]
    assert board.ranked[0].judge == "majority-oracle"
    thin_cov = next(j for j in board.masked if j.judge == "thin-judge")
    assert thin_cov.thinnest_cell is not None
    assert thin_cov.thinnest_cell.n == 20
    # Masked judges are shown, never silently dropped.
    assert "thin-judge" in render_ranked(board)
    assert "MASKED" in render_ranked(board)
    assert "RANKED" in render_ranked(board)


def test_ranking_orders_by_conservative_score_not_point_rho() -> None:
    report = _report()
    board = assess_ranking(report, POLICY)
    scores: list[float] = []
    for judge in board.ranked:
        assert judge.conservative_score is not None
        scores.append(judge.conservative_score)
    assert scores == sorted(scores, reverse=True)
    # Every ranked judge beats every masked judge's diagnostic only in the
    # sense that matters: masked judges are not in the order at all.
    assert len(board.ranked) + len(board.masked) == len(report.verdicts)


def test_dropped_pair_masks_the_judge() -> None:
    # Replicates the implementation reviewer's probe A: annotator ann4
    # labels only 5 of 120 items, so the runner drops every (judge,
    # ann4) pair before the report. The gate must not rank on the
    # remaining pairs alone — each judge carries one unreported sparse
    # cell and is masked.
    items: list[AltTestItem] = []
    for i, item in enumerate(make_dataset(n_items=120)):
        ann = dict(item.annotator_labels)
        if i >= 5:
            ann.pop("ann4", None)
        items.append(
            AltTestItem(
                id=item.id,
                task=item.task,
                annotator_labels=ann,
                judge_labels=dict(item.judge_labels),
            )
        )
    report = run_alt_test(items, AltTestConfig(epsilon=0.1, q=0.05, seed=7))
    assert all(len(v.comparisons) == 4 for v in report.verdicts)
    board = assess_ranking(report, POLICY)
    assert board.ranked == ()
    assert not board.is_ranking
    for judge in board.masked:
        assert judge.unreported_cells == 1
        assert judge.coverage_fraction == pytest.approx(4 / 5)
    assert "unreported" in render_ranked(board)
    with pytest.raises(RankingRefused, match="unreported pair"):
        require_ranking(board)


def test_no_rankable_judge_means_no_ranking() -> None:
    report = _report()
    strict = assess_ranking(report, CoveragePolicy(min_per_cell=200, min_total=50))
    assert not strict.is_ranking
    assert strict.ranked == ()
    assert len(strict.masked) == len(report.verdicts)
    with pytest.raises(RankingRefused, match="no judge has every cell covered"):
        require_ranking(strict)


def test_total_floor_refuses_even_a_fully_covered_small_dataset() -> None:
    report = _report(n_items=60, n_labelled=60)
    board = assess_ranking(report, CoveragePolicy(min_per_cell=50, min_total=61))
    # Cells are covered, but the dataset is below the total floor, so
    # the board is not a ranking either — is_ranking folds the floor in.
    assert board.is_ranking is False
    assert board.ranked != ()
    with pytest.raises(RankingRefused, match="below the min_total"):
        require_ranking(board)


def test_assess_ranking_never_rewrites_the_report() -> None:
    report = _report()
    before = [(v.judge, v.rho, v.omega) for v in report.verdicts]
    assess_ranking(report, POLICY)
    after = [(v.judge, v.rho, v.omega) for v in report.verdicts]
    assert before == after


def test_board_serializes_to_json() -> None:
    board = assess_ranking(_report(), POLICY)
    doc = board.to_dict()
    assert doc["is_ranking"] is True
    assert doc["min_per_cell"] == 50
    assert doc["n_items"] == 120
    ranked = cast(list[dict[str, object]], doc["ranked"])
    masked = cast(list[dict[str, object]], doc["masked"])
    assert [r["judge"] for r in ranked] == [j.judge for j in board.ranked]
    assert [m["judge"] for m in masked] == ["thin-judge"]
    assert masked[0]["unreported_cells"] == 0
    thinnest = cast(dict[str, object], masked[0]["thinnest_cell"])
    assert thinnest["n"] == 20
    json.dumps(doc)  # must be JSON-serializable


def test_recover_wins_refuses_non_finite_rho() -> None:
    # A corrupt report carrying rho_f=inf must refuse as a typed
    # ValueError, not crash the CLI with an OverflowError traceback.
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="not finite"):
            _recover_wins("judge", "ann0", bad, 50)


def test_empty_report_refusal_names_the_emptiness() -> None:
    board = assess_ranking(AltTestReport(verdicts=(), epsilon=0.1, q=0.05, n_items=60), POLICY)
    assert not board.is_ranking
    with pytest.raises(RankingRefused, match="no judges at all"):
        require_ranking(board)


def test_judge_cells_must_share_one_floor() -> None:
    with pytest.raises(ValueError, match="different floors"):
        JudgeCoverage(
            "judge",
            (
                assess_cell("ann0", "judge", 60, 60, 50),
                assess_cell("ann1", "judge", 60, 60, 20),
            ),
        )


def test_generic_observations_gate() -> None:
    cells = assess_observations(
        [
            CellObservation("s0", "a", 60, 55),
            CellObservation("s1", "a", 10, 10),
            CellObservation("s0", "b", 60, 30),
        ],
        POLICY,
    )
    assert [c.status for c in cells] == ["covered", "sparse", "covered"]
    assert cells[1].conservative_rate < cells[0].conservative_rate


# --- CLI ------------------------------------------------------------------


def test_cli_ranks_full_dataset_and_refuses_thin_only(tmp_path: Path) -> None:
    full_path = tmp_path / "full.jsonl"
    thin_path = tmp_path / "thin.jsonl"
    out_path = tmp_path / "report.json"

    _write_jsonl(_thin_dataset(), full_path)
    assert cmd_alt_test(["--data", str(full_path), "--out", str(out_path)]) == 0
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["coverage"]["is_ranking"] is True
    assert doc["coverage"]["masked"][0]["judge"] == "thin-judge"

    thin_only = [
        AltTestItem(
            id=item.id,
            task=item.task,
            annotator_labels=dict(item.annotator_labels),
            judge_labels=({"thin-judge": item.judge_labels["majority-oracle"]} if i < 20 else {}),
        )
        for i, item in enumerate(make_dataset(n_items=60))
    ]
    _write_jsonl(thin_only, thin_path)
    assert cmd_alt_test(["--data", str(thin_path)]) == 1


def test_cli_refusal_prints_no_ungated_leaderboard_and_writes_no_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The gate runs before anything is printed or written: on refusal
    # stdout carries only the gated board (masked rows), and no --out
    # file exists — a refused ranking must not leak as an artifact.
    thin_path = tmp_path / "thin.jsonl"
    out_path = tmp_path / "report.json"
    thin_only = [
        AltTestItem(
            id=item.id,
            task=item.task,
            annotator_labels=dict(item.annotator_labels),
            judge_labels=({"thin-judge": item.judge_labels["majority-oracle"]} if i < 20 else {}),
        )
        for i, item in enumerate(make_dataset(n_items=60))
    ]
    _write_jsonl(thin_only, thin_path)
    assert cmd_alt_test(["--data", str(thin_path), "--out", str(out_path)]) == 1
    assert not out_path.exists()
    stdout = capsys.readouterr().out
    assert "thin-judge" in stdout  # the masked row is shown, not hidden
    assert "MASKED" in stdout
    assert "NOT A RANKING" in stdout
    assert "alt-test leaderboard" not in stdout
    assert "omega" not in stdout


def test_cli_bad_coverage_policy_exits_one(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    _write_jsonl(_thin_dataset(), path)
    assert cmd_alt_test(["--data", str(path), "--min-cell-n", "abc"]) == 1
    assert cmd_alt_test(["--data", str(path), "--min-cell-n", "1"]) == 1
    assert cmd_alt_test(["--data", str(path), "--min-total", "0"]) == 1


def test_cli_relaxed_floor_ranks_the_thin_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The floor is a policy knob, not a hard-coded constant: relaxing it
    # to 20 ranks the thin judge, which pins that the mask follows the
    # policy rather than the judge's name. Relaxing below the source
    # default of 50 warns on stderr: the ranking is no longer the
    # source-floor ranking.
    path = tmp_path / "data.jsonl"
    _write_jsonl(_thin_dataset(), path)
    out_path = tmp_path / "report.json"
    assert cmd_alt_test(["--data", str(path), "--min-cell-n", "20", "--out", str(out_path)]) == 0
    assert "Warning" in capsys.readouterr().err
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert "thin-judge" in [r["judge"] for r in doc["coverage"]["ranked"]]
    assert doc["coverage"]["masked"] == []


# --- Properties -----------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=200),
            st.integers(min_value=0, max_value=200),
        ),
        min_size=1,
        max_size=10,
    )
)
def test_masking_is_monotone_in_n(pairs: list[tuple[int, int]]) -> None:
    # More observations in a cell must never flip it from covered to sparse:
    # coverage is exactly the n >= floor comparison, so adding observations
    # can only move a cell sparse -> covered, never the reverse.
    for n, wins in pairs:
        wins = min(wins, n)
        assert (assess_cell("s", "a", n, wins, 50).status == "covered") == (n >= 50)
        assert assess_cell("s", "a", n + 50, wins + 50, 50).status == "covered"


@settings(max_examples=30, deadline=None)
@given(st.integers(min_value=2, max_value=120))
def test_ranking_is_deterministic(floor: int) -> None:
    report = _report()
    policy = CoveragePolicy(min_per_cell=floor, min_total=50)
    first = assess_ranking(report, policy)
    second = assess_ranking(report, policy)
    assert first == second
    assert [j.judge for j in first.ranked] == [j.judge for j in second.ranked]
