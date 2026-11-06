"""Adversarial tests for the alt-test benchmark (agenteval-bench#31).

Every behavior change ships with a running test — prose is not enforcement.
The strongest pins come from the paper's own Theorem 1: a judge predicting
the majority vote (discrete) or the mean annotation (continuous) must reach
rho = 1.0 exactly. Any implementation whose alignment scoring drifts from
the paper fails these tests.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import tempfile

import pytest

from alt_test.runner import (
    InsufficientCoverage,
    alignment_score,
    dataset_digest,
    default_text_sim,
    load_jsonl,
    render_text,
    run_alt_test,
)
from alt_test.stats import (
    benjamini_yekutieli,
    paired_t_test_one_sided,
    regularized_incomplete_beta,
    student_t_cdf,
    wilcoxon_signed_rank_one_sided,
)
from alt_test.types import AltTestConfig, AltTestItem

LABELS = ["a", "b", "c"]


def _discrete_items(
    n: int = 60,
    n_ann: int = 5,
    seed: int = 11,
    judge_noise: float = 0.1,
    judge_name: str = "judge",
) -> list[AltTestItem]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        truth = rng.choice(LABELS)
        ann = {}
        for a in range(n_ann):
            ann[f"ann{a}"] = (
                rng.choice([c for c in LABELS if c != truth])
                if rng.random() < 0.15
                else truth
            )
        judge_label = (
            rng.choice(LABELS) if rng.random() < judge_noise else truth
        )
        items.append(
            AltTestItem(
                id=f"i{i}",
                task="discrete",
                annotator_labels=ann,
                judge_labels={judge_name: judge_label},
            )
        )
    return items


def _majority(items_labels: list[str]) -> str:
    return max(LABELS, key=lambda c: items_labels.count(c))


# ---------------------------------------------------------------------------
# Statistics: pinned against published values, not against themselves
# ---------------------------------------------------------------------------


def test_incomplete_beta_against_known_values():
    # I_0.5(1, 1) = 0.5 exactly (uniform CDF); I_x(2,3) has a closed form.
    assert regularized_incomplete_beta(0.5, 1.0, 1.0) == pytest.approx(0.5, abs=1e-12)
    # I_x(2,3) = 6x^2 - 8x^3 + 3x^4
    x = 0.3
    assert regularized_incomplete_beta(x, 2.0, 3.0) == pytest.approx(
        6 * x**2 - 8 * x**3 + 3 * x**4, abs=1e-10
    )


def test_student_t_cdf_against_published_table():
    # One-sided published values: P(T_10 <= 1.812) = 0.95; P(T_10 <= 2.764) = 0.99.
    assert student_t_cdf(1.812, 10) == pytest.approx(0.95, abs=1e-3)
    assert student_t_cdf(2.764, 10) == pytest.approx(0.99, abs=1e-3)
    assert student_t_cdf(0.0, 10) == 0.5
    # df=1 is Cauchy: P(T_1 <= 1) = 0.75.
    assert student_t_cdf(1.0, 1) == pytest.approx(0.75, abs=1e-6)


def test_paired_t_test_detects_clear_advantage():
    diffs = [-1.0] * 40 + [1.0] * 10  # judge mostly better (d = W_h - W_f < 0)
    p = paired_t_test_one_sided(diffs, epsilon=0.1)
    assert p < 1e-6


def test_paired_t_test_keeps_clear_disadvantage():
    diffs = [1.0] * 40 + [-1.0] * 10
    p = paired_t_test_one_sided(diffs, epsilon=0.1)
    assert p > 0.99


def test_paired_t_test_degenerate_tie_goes_to_judge():
    # d = 0 everywhere: rho_f == rho_h, so H0 (rho_f <= rho_h - eps) is
    # deterministically false for eps > 0 — the cost-benefit margin decides.
    assert paired_t_test_one_sided([0.0] * 50, epsilon=0.1) == 0.0
    # With epsilon = 0 the tie sits exactly on the boundary: no rejection.
    assert paired_t_test_one_sided([0.0] * 50, epsilon=0.0) == 1.0


def test_by_fdr_hand_computed():
    # m=4, q=0.05: c(4) = 1 + 1/2 + 1/3 + 1/4 = 2.0833.
    # thresholds: k=1 -> 0.006, k=2 -> 0.012, k=3 -> 0.018, k=4 -> 0.024.
    p = [0.005, 0.011, 0.05, 0.5]
    assert benjamini_yekutieli(p, 0.05) == [True, True, False, False]
    # BH would reject the third too (0.05 <= 3*0.05/4 = 0.0375? no: 0.05 > 0.0375).
    # Sanity: with q=0.5 the thresholds widen and more reject.
    assert benjamini_yekutieli(p, 0.5) == [True, True, True, False]


def test_wilcoxon_hand_computed():
    # Hand-computed independent pin. diffs=[0.5, 1.5, -0.5], eps=0:
    # |e| ranks: 0.5,0.5 tie -> 1.5,1.5; 1.5 -> 3. W+ = 1.5+3 = 4.5.
    # mu = 3, var = 3*4*7/24 - (2^3-2)/48 = 3.375, z = (4.5-3+0.5)/sqrt(3.375).
    # p = Phi(1.0887) ~= 0.862.
    p = wilcoxon_signed_rank_one_sided([0.5, 1.5, -0.5], epsilon=0.0)
    assert p == pytest.approx(0.862, abs=1e-3)


def test_report_records_sim_name():
    items = _discrete_items(n=60, seed=131)
    report = run_alt_test(items, AltTestConfig(seed=131))
    assert report.to_dict()["sim_name"] == "default_text_sim"

    def my_sim(a: str, b: str) -> float:
        return 1.0 if a == b else 0.0

    items_t = [
        AltTestItem(id=f"t{i}", task="text",
                    annotator_labels={f"a{k}": "hello world" for k in range(3)},
                    judge_labels={"j": "hello world"})
        for i in range(60)
    ]
    report_t = run_alt_test(items_t, AltTestConfig(seed=131), sim=my_sim)
    assert report_t.to_dict()["sim_name"] == "my_sim"


def test_config_rejects_min_pair_items_below_two():
    with pytest.raises(ValueError, match="min_pair_items"):
        AltTestConfig(min_pair_items=1)
    with pytest.raises(ValueError, match="t_test_min_n"):
        AltTestConfig(t_test_min_n=1)


def test_validate_rejects_mismatched_label_types():
    base = _discrete_items(n=60, seed=141)

    def _retasked(task, label_fn):
        return [
            AltTestItem(
                id=item.id, task=task,
                annotator_labels={k: label_fn(v) for k, v in item.annotator_labels.items()},
                judge_labels={k: label_fn(v) for k, v in item.judge_labels.items()},
            )
            for item in base
        ]

    with pytest.raises(TypeError, match="string labels"):
        run_alt_test(_retasked("text", lambda v: 1.5))
    with pytest.raises(TypeError, match="numeric labels"):
        run_alt_test(_retasked("continuous", lambda v: v))
    with pytest.raises(TypeError, match="must not be a bool"):
        items = _retasked("discrete", lambda v: v)
        first = items[0]
        items[0] = AltTestItem(
            id=first.id, task=first.task,
            annotator_labels={**first.annotator_labels, "annX": True},
            judge_labels=first.judge_labels,
        )
        run_alt_test(items)


def test_render_text_shows_compared_count_on_partial_coverage():
    rng = random.Random(151)
    items = []
    for i in range(60):
        ann = {f"ann{a}": rng.choice(LABELS) for a in range(4)}
        if i < 5:
            ann["ann4"] = rng.choice(LABELS)  # sparse fifth annotator
        items.append(
            AltTestItem(id=f"i{i}", task="discrete",
                        annotator_labels=ann,
                        judge_labels={"j": rng.choice(LABELS)})
        )
    report = run_alt_test(items, AltTestConfig(epsilon=0.1, q=0.05))
    (verdict,) = report.verdicts
    assert len(verdict.comparisons) == 4
    assert verdict.m_annotators == 5
    text = render_text(report)
    assert "4/5" in text
    assert any("ann4" in note for note in report.notes)


def test_wilcoxon_small_n_path_runs():
    rng = random.Random(3)
    diffs = [rng.choice([-1.0, 0.0, 1.0]) for _ in range(20)]
    p = wilcoxon_signed_rank_one_sided(diffs, epsilon=0.1)
    assert 0.0 <= p <= 1.0
    # All-zero diffs with epsilon > 0: the judge ties the annotator
    # everywhere, and the cost-benefit margin decides for the judge —
    # H0 (median(d) >= epsilon) is deterministically false.
    assert wilcoxon_signed_rank_one_sided([0.0] * 20, epsilon=0.1) < 1e-3
    # With epsilon = 0 the shifted diffs are all exactly zero: the data sit
    # on the H0 boundary, p = 1.0.
    assert wilcoxon_signed_rank_one_sided([0.0] * 20, epsilon=0.0) == 1.0


# ---------------------------------------------------------------------------
# Theorem 1 pins: the optimal judge reaches rho = 1.0 exactly
# ---------------------------------------------------------------------------


def test_majority_oracle_reaches_rho_one_discrete():
    rng = random.Random(21)
    items = []
    for i in range(60):
        ann = {f"ann{a}": rng.choice(LABELS) for a in range(5)}
        majority = _majority(list(ann.values()))
        items.append(
            AltTestItem(
                id=f"i{i}", task="discrete",
                annotator_labels=ann, judge_labels={"oracle": majority},
            )
        )
    report = run_alt_test(items, AltTestConfig(epsilon=0.1, q=0.05))
    (verdict,) = report.verdicts
    assert verdict.rho == 1.0
    assert verdict.omega == 1.0
    assert verdict.justified


def test_mean_oracle_reaches_rho_one_continuous():
    rng = random.Random(22)
    items = []
    for i in range(60):
        ann = {f"ann{a}": rng.uniform(1.0, 5.0) for a in range(5)}
        mean_label = sum(ann.values()) / len(ann)
        items.append(
            AltTestItem(
                id=f"i{i}", task="continuous",
                annotator_labels=ann, judge_labels={"oracle": mean_label},
            )
        )
    report = run_alt_test(items, AltTestConfig(epsilon=0.1, q=0.05))
    (verdict,) = report.verdicts
    assert verdict.rho == 1.0
    assert verdict.justified


# ---------------------------------------------------------------------------
# Behavioral pins
# ---------------------------------------------------------------------------


def test_coinflip_judge_is_not_justified():
    items = _discrete_items(n=120, seed=31, judge_noise=1.0, judge_name="coinflip")
    report = run_alt_test(items, AltTestConfig(epsilon=0.1, q=0.05, seed=31))
    (verdict,) = report.verdicts
    assert verdict.rho < 0.6
    assert verdict.omega == 0.0
    assert not verdict.justified


def test_sharp_judge_beats_noisy_judge_on_rho():
    rng = random.Random(41)
    items = []
    for i in range(120):
        truth = rng.choice(LABELS)
        ann = {
            f"ann{a}": truth if rng.random() > 0.15 else rng.choice(LABELS)
            for a in range(5)
        }
        judges = {
            "sharp": truth if rng.random() > 0.05 else rng.choice(LABELS),
            "noisy": truth if rng.random() > 0.45 else rng.choice(LABELS),
        }
        items.append(
            AltTestItem(id=f"i{i}", task="discrete",
                        annotator_labels=ann, judge_labels=judges)
        )
    report = run_alt_test(items, AltTestConfig(epsilon=0.1, q=0.05))
    by_name = {v.judge: v for v in report.verdicts}
    assert by_name["sharp"].rho > by_name["noisy"].rho
    assert report.verdicts[0].judge == "sharp"  # leaderboard sorted by rho


def test_epsilon_monotonicity_of_omega():
    items = _discrete_items(n=80, seed=51, judge_noise=0.25)
    omegas = [
        run_alt_test(items, AltTestConfig(epsilon=e, q=0.05)).verdicts[0].omega
        for e in (0.0, 0.1, 0.2, 0.3, 0.5)
    ]
    assert all(b >= a for a, b in itertools.pairwise(omegas)), omegas


def test_anti_correlated_judge_loses_everywhere():
    items = []
    for i in range(80):
        ann = {f"ann{a}": "a" for a in range(5)}  # unanimous annotators
        items.append(
            AltTestItem(id=f"i{i}", task="discrete",
                        annotator_labels=ann, judge_labels={"troll": "b"})
        )
    report = run_alt_test(items, AltTestConfig(epsilon=0.1, q=0.05))
    (verdict,) = report.verdicts
    assert verdict.rho == 0.0
    assert verdict.omega == 0.0
    assert not verdict.justified


# ---------------------------------------------------------------------------
# Alignment scoring units
# ---------------------------------------------------------------------------


def test_alignment_discrete_is_fraction_agreeing():
    assert alignment_score("a", ["a", "b", "a"], "discrete") == pytest.approx(2 / 3)


def test_alignment_continuous_is_negative_rmse():
    s = alignment_score(3.0, [3.0, 5.0], "continuous")
    assert s == pytest.approx(-math.sqrt(((0.0) ** 2 + (2.0) ** 2) / 2))


def test_alignment_text_uses_injected_sim():
    calls = []

    def sim(a: str, b: str) -> float:
        calls.append((a, b))
        return 0.42

    assert alignment_score("x", ["y", "z"], "text", sim) == pytest.approx(0.42)
    assert len(calls) == 2


def test_default_text_sim_is_deterministic_and_bounded():
    assert default_text_sim("hello world", "hello world") == 1.0
    s = default_text_sim("the cat sat", "the cat ran")
    assert 0.0 < s < 1.0
    assert default_text_sim("", "x") == 0.0


# ---------------------------------------------------------------------------
# Boundary refusals and input validation
# ---------------------------------------------------------------------------


def test_refuses_fewer_than_three_annotators():
    items = _discrete_items(n=60, n_ann=2)
    with pytest.raises(ValueError, match="annotators"):
        run_alt_test(items)


def test_refuses_fewer_than_min_items():
    items = _discrete_items(n=20)
    with pytest.raises(ValueError, match="at least 50 items"):
        run_alt_test(items, AltTestConfig(min_items=50))


def test_refuses_mixed_task_types():
    items = _discrete_items(n=60)
    mixed = list(items)
    first = mixed[0]
    mixed[0] = AltTestItem(
        id=first.id, task="continuous",
        annotator_labels={k: 1.0 for k in first.annotator_labels},
        judge_labels={k: 1.0 for k in first.judge_labels},
    )
    with pytest.raises(ValueError, match="single task type"):
        run_alt_test(mixed)


def test_insufficient_pair_coverage_raises_named_pair():
    items = _discrete_items(n=60, seed=71)
    # Strip the judge label from every item but five: coverage collapses.
    sparse = [
        AltTestItem(id=item.id, task=item.task,
                    annotator_labels=item.annotator_labels,
                    judge_labels=item.judge_labels if int(item.id[1:]) < 5 else {})
        for item in items
    ]
    with pytest.raises(InsufficientCoverage, match="judge"):
        run_alt_test(sparse)


def test_sparse_annotator_items_still_run():
    rng = random.Random(81)
    items = []
    for i in range(60):
        ann = {f"ann{a}": rng.choice(LABELS) for a in range(5) if rng.random() > 0.1}
        if len(ann) < 3:
            for a in range(3 - len(ann)):
                ann[f"fill{a}"] = rng.choice(LABELS)
        items.append(
            AltTestItem(id=f"i{i}", task="discrete",
                        annotator_labels=ann,
                        judge_labels={"j": rng.choice(LABELS)})
        )
    report = run_alt_test(items, AltTestConfig(epsilon=0.1, q=0.05))
    assert report.verdicts[0].n_items == 60


def test_load_jsonl_rejects_malformed_rows_with_line_numbers():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            f.write('{"id": "ok", "task": "discrete", "annotators": {"a": "x"}, "judges": {}}\n')
            f.write('{"id": "bad", "task": "sideways", "annotators": {"a": "x"}}\n')
        with pytest.raises(ValueError, match=":2:"):
            load_jsonl(path)
    finally:
        os.unlink(path)


def test_load_jsonl_rejects_bool_labels():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(
                '{"id": "b", "task": "discrete", '
                '"annotators": {"a1": true, "a2": "x", "a3": "x"}, "judges": {"j": "x"}}\n'
            )
        with pytest.raises(TypeError, match="str or number"):
            load_jsonl(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Report contract
# ---------------------------------------------------------------------------


def test_report_names_fdr_procedure_and_echoes_config():
    items = _discrete_items(n=60, seed=91)
    report = run_alt_test(items, AltTestConfig(epsilon=0.2, q=0.01, seed=91))
    d = report.to_dict()
    assert d["fdr_procedure"] == "Benjamini-Yekutieli"
    assert d["epsilon"] == 0.2
    assert d["q"] == 0.01
    assert d["seed"] == 91
    assert d["input_digest"] == ""
    text = render_text(report)
    assert "Benjamini-Yekutieli" in text
    assert "epsilon=0.2" in text


def test_reproducibility_identical_bytes():
    items = _discrete_items(n=60, seed=101)
    digest = dataset_digest(items)
    r1 = run_alt_test(items, AltTestConfig(seed=101), input_digest=digest).to_dict()
    r2 = run_alt_test(items, AltTestConfig(seed=101), input_digest=digest).to_dict()
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_dataset_digest_stable_and_sensitive():
    items = _discrete_items(n=60, seed=111)
    d1 = dataset_digest(items)
    assert d1 == dataset_digest(items)
    altered = list(items)
    first = altered[0]
    altered[0] = AltTestItem(
        id=first.id, task=first.task,
        annotator_labels={**first.annotator_labels, "ann0": "zzz"},
        judge_labels=first.judge_labels,
    )
    assert dataset_digest(altered) != d1
    assert len(d1) == 64


def _write_jsonl(rows: list[dict]) -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _run_cli(argv, monkeypatch, capsys):
    from agenteval_bench import cli

    monkeypatch.setattr("sys.argv", ["agenteval-bench", *argv])
    try:
        cli.main()
        captured = capsys.readouterr()
        return 0, captured.out
    except SystemExit as e:
        captured = capsys.readouterr()
        return e.code or 0, captured.out


def test_cli_alt_test_end_to_end(monkeypatch, capsys):
    rng = random.Random(121)
    rows = []
    for i in range(60):
        truth = rng.choice(LABELS)
        rows.append(
            {
                "id": f"i{i}",
                "task": "discrete",
                "annotators": {
                    f"ann{a}": truth if rng.random() > 0.15 else rng.choice(LABELS)
                    for a in range(4)
                },
                "judges": {"j": truth if rng.random() > 0.1 else rng.choice(LABELS)},
            }
        )
    data = _write_jsonl(rows)
    out = data + ".report.json"
    try:
        code, stdout = _run_cli(
            ["alt-test", "--data", data, "--epsilon", "0.1", "--out", out],
            monkeypatch, capsys,
        )
        assert code == 0, stdout
        assert "omega" in stdout and "rho" in stdout
        with open(out) as f:
            report = json.load(f)
        assert report["fdr_procedure"] == "Benjamini-Yekutieli"
        assert report["leaderboard"][0]["judge"] == "j"
        assert len(report["input_digest"]) == 64
    finally:
        os.unlink(data)
        if os.path.exists(out):
            os.unlink(out)


def test_cli_alt_test_requires_data(monkeypatch, capsys):
    code, _ = _run_cli(["alt-test"], monkeypatch, capsys)
    assert code == 1


def test_cli_alt_test_refuses_bad_epsilon(monkeypatch, capsys):
    data = _write_jsonl([])
    try:
        code, _ = _run_cli(
            ["alt-test", "--data", data, "--epsilon", "0.9"], monkeypatch, capsys
        )
        assert code == 1
    finally:
        os.unlink(data)


def test_alt_test_config_default_seed_is_42() -> None:
    """The default config seed is 42 (recorded for reproducibility).

    Kills the types.py mutant that negates the default seed to -42.
    The seed is a documented default; a negated default would silently
    change every run that relies on the default.
    """
    assert AltTestConfig(epsilon=0.1, q=0.05).seed == 42


def test_make_dataset_default_seed_is_seven() -> None:
    """make_dataset() defaults to seed 7: explicit seed 7 reproduces it.

    Kills the demo.py mutant that negates the default seed to -7. The
    pin is on the signature default itself, not on behavior: CPython's
    ``random.Random(-7)`` draws the identical stream to ``Random(7)``
    (int seeds are taken by absolute value), so the two datasets are
    behaviorally identical and only the declared default distinguishes
    them. The default seed is part of the demo's determinism contract.
    """
    import inspect

    from alt_test.demo import make_dataset

    assert inspect.signature(make_dataset).parameters["seed"].default == 7
    assert make_dataset(n_items=5) == make_dataset(seed=7, n_items=5)


def test_make_dataset_first_items_are_golden() -> None:
    """Seeded dataset items are pinned exactly (golden values, seed 7).

    Kills the demo.py compare-flip mutant (`rng.random() < noise` ->
    `>`): under the mutant the low-noise annotators take the wrong
    label most of the time, so the first items differ from the golden
    values below.
    """
    from alt_test.demo import make_dataset

    items = make_dataset(seed=7, n_items=3)
    assert items[0].annotator_labels == {
        "ann0": "neutral",
        "ann1": "neutral",
        "ann2": "entailment",
        "ann3": "neutral",
        "ann4": "entailment",
    }
    assert items[0].judge_labels["majority-oracle"] == "neutral"
    assert items[1].annotator_labels["ann0"] == "entailment"
    assert items[1].judge_labels["sharp-judge"] == "contradiction"
    assert items[2].annotator_labels == {
        "ann0": "entailment",
        "ann1": "entailment",
        "ann2": "entailment",
        "ann3": "entailment",
        "ann4": "entailment",
    }
