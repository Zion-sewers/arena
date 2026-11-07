"""Seeded end-to-end demo of coverage-gated ranking (arena#49).

Builds the alt-test demo dataset (120 items, 5 annotators, 4 judges),
adds a thin judge that labels only 20 items — and would win on all of
them — and shows the coverage gate masking it out of the ranking while
the fully-covered judges stay ranked. Then shows the refusal: a
dataset where no judge meets the per-cell floor is not a ranking, and
the CLI exits 1 naming the thinnest cells. Finally the dropped pair:
an annotator labelling only 5 items drops every pair against it, and
the judges are masked on the unreported cells.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agenteval_bench.cli import cmd_alt_test
from agenteval_bench.coverage import (
    CoveragePolicy,
    RankingRefused,
    assess_ranking,
    require_ranking,
)
from alt_test.demo import make_dataset
from alt_test.runner import run_alt_test
from alt_test.types import AltTestConfig, AltTestItem

THIN_JUDGE = "thin-judge"


def _expect(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"  ok    {label}")


def _with_thin_judge(items: list[AltTestItem], n_labelled: int) -> list[AltTestItem]:
    """Copy the dataset, giving the thin judge labels on the first N items.

    The thin judge copies the majority-oracle label, so on its 20 items
    it wins every comparison — the trap is that omega=1.0 on 20 items
    must still not rank it.
    """
    out: list[AltTestItem] = []
    for i, item in enumerate(items):
        judges = dict(item.judge_labels)
        if i < n_labelled:
            judges[THIN_JUDGE] = item.judge_labels["majority-oracle"]
        out.append(
            AltTestItem(
                id=item.id,
                task=item.task,
                annotator_labels=dict(item.annotator_labels),
                judge_labels=judges,
            )
        )
    return out


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


def main() -> int:
    print("Coverage-gated ranking demo (arena#49)")

    # 1. The trap: a judge that wins every comparison it enters, on 20 items.
    items = _with_thin_judge(make_dataset(), 20)
    report = run_alt_test(items, AltTestConfig(epsilon=0.1, q=0.05, seed=7))
    thin = next(v for v in report.verdicts if v.judge == THIN_JUDGE)
    _expect(thin.omega == 1.0, "thin judge wins every comparison it enters (omega=1.0)")
    board = assess_ranking(report, CoveragePolicy(min_per_cell=50, min_total=50))
    ranked_names = [j.judge for j in board.ranked]
    masked_names = [j.judge for j in board.masked]
    _expect(THIN_JUDGE in masked_names, "thin judge is MASKED, never ranked")
    _expect(THIN_JUDGE not in ranked_names, "masked judge never enters the ranked order")
    _expect(board.ranked[0].judge == "majority-oracle", "oracle still tops the ranking")
    _expect(board.is_ranking, "the leaderboard is still a ranking (others are covered)")
    thin_cov = next(j for j in board.masked if j.judge == THIN_JUDGE)
    thinnest = thin_cov.thinnest_cell
    _expect(
        thinnest is not None and thinnest.n == 20,
        "masking names the thinnest cell (n=20 < 50)",
    )
    require_ranking(board)

    # 2. The conservative fallback: same 100% win rate, thin vs covered.
    # The fallback is the Wilson lower bound, so the interval width does
    # the punishing: 5/5 wins falls back to ~0.57, 120/120 stays ~0.97.
    from agenteval_bench.coverage import assess_cell

    thin_cell = assess_cell("ann0", "demo-judge", 5, 5, min_per_cell=50)
    covered_cell = assess_cell("ann0", "demo-judge", 120, 120, min_per_cell=50)
    _expect(thin_cell.status == "sparse", "a 5-observation cell is sparse")
    _expect(covered_cell.status == "covered", "a 120-observation cell is covered")
    _expect(
        thin_cell.conservative_rate < 0.6,
        f"5/5 wins falls back low ({thin_cell.conservative_rate:.3f})",
    )
    _expect(
        covered_cell.conservative_rate > 0.9,
        f"120/120 wins stays high ({covered_cell.conservative_rate:.3f})",
    )
    _expect(
        thin_cell.conservative_rate < covered_cell.conservative_rate,
        "the fallback punishes thin data by interval width, not by fiat",
    )

    # 3. The refusal: no judge meets the floor, so there is no ranking.
    strict = assess_ranking(report, CoveragePolicy(min_per_cell=200, min_total=50))
    _expect(not strict.is_ranking, "a 200-per-cell floor ranks nobody")
    try:
        require_ranking(strict)
    except RankingRefused as e:
        _expect("no judge has every cell covered" in str(e), "refusal names the rule")
    else:  # pragma: no cover - the refusal is the expectation
        raise AssertionError("require_ranking must refuse when nobody is rankable")

    # 4. The CLI: full dataset ranks (exit 0); thin-only dataset refuses (exit 1).
    with tempfile.TemporaryDirectory() as tmp:
        full_path = Path(tmp) / "full.jsonl"
        _write_jsonl(items, full_path)
        code = cmd_alt_test(["--data", str(full_path)])
        _expect(code == 0, "CLI ranks the full dataset (exit 0)")
        thin_items = _with_thin_judge(make_dataset(n_items=60), 20)
        thin_only = [
            AltTestItem(
                id=item.id,
                task=item.task,
                annotator_labels=dict(item.annotator_labels),
                judge_labels=({THIN_JUDGE: item.judge_labels["majority-oracle"]} if i < 20 else {}),
            )
            for i, item in enumerate(thin_items)
        ]
        thin_path = Path(tmp) / "thin.jsonl"
        _write_jsonl(thin_only, thin_path)
        code = cmd_alt_test(["--data", str(thin_path)])
        _expect(code == 1, "CLI refuses the thin-only dataset (exit 1)")

    # 5. The dropped pair: an annotator labelling only 5 items drops
    # every pair against it, and the judges are masked on the unreported
    # cells — never ranked on their remaining pairs alone.
    dropped: list[AltTestItem] = []
    for i, item in enumerate(make_dataset(n_items=120)):
        ann = dict(item.annotator_labels)
        if i >= 5:
            ann.pop("ann4", None)
        dropped.append(
            AltTestItem(
                id=item.id,
                task=item.task,
                annotator_labels=ann,
                judge_labels=dict(item.judge_labels),
            )
        )
    dropped_report = run_alt_test(dropped, AltTestConfig(epsilon=0.1, q=0.05, seed=7))
    dropped_board = assess_ranking(dropped_report, CoveragePolicy(min_per_cell=50, min_total=50))
    _expect(dropped_board.ranked == (), "dropped pairs mask every judge")
    _expect(
        all(j.unreported_cells == 1 for j in dropped_board.masked),
        "each masked judge names 1 unreported pair",
    )
    try:
        require_ranking(dropped_board)
    except RankingRefused as e:
        _expect("unreported pair" in str(e), "refusal names the unreported pairs")
    else:  # pragma: no cover - the refusal is the expectation
        raise AssertionError("require_ranking must refuse on unreported pairs")

    print("demo checks passed: sparse cells masked, fallback conservative, refusal named")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
