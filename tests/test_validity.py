"""Tests for measurement validity: every metric carries a reliability column (arena#43).

"Test the test". Every behavior change ships with an adversarial test: the
tests below pin the failure modes the design must refuse, not just the happy
paths. No mocks: hand-built paired verdicts and in-memory suites only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenteval_bench.cli import cmd_decide, cmd_run
from agenteval_bench.decision import (
    CostEvidence,
    Decision,
    Dimension,
    Evidence,
    ProportionEvidence,
    RiskLevel,
    UnmeasuredEvidence,
    decide,
    default_policy,
)
from agenteval_bench.engine import EvalRunner
from agenteval_bench.models import EvalCase, EvalSuite, ExpectedOutput
from agenteval_bench.validity import (
    MetricValidity,
    ValidityGrade,
    ValidityPolicy,
    ValidityReport,
    assess_decision,
    assess_scorer,
    default_validity_policy,
    validate_matcher_labels,
    validate_proxy,
)

# ---------------------------------------------------------------- helpers


def _evidence(
    quality=(20, 20), safety=(20, 20), reliability=(20, 20), cost=None
) -> dict[Dimension, Evidence]:
    return {
        Dimension.QUALITY: ProportionEvidence(Dimension.QUALITY, *quality),
        Dimension.SAFETY: ProportionEvidence(Dimension.SAFETY, *safety),
        Dimension.RELIABILITY: ProportionEvidence(Dimension.RELIABILITY, *reliability),
        Dimension.COST: (
            cost if cost is not None else CostEvidence(mean=1.0, std=0.1, n=20)
        ),
    }


def _decision(**kwargs) -> Decision:
    return decide(_evidence(**kwargs), default_policy(), RiskLevel.LOW)


def _case(case_id: str, dimension: str, answer: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        input=f"prompt-{case_id}",
        expected=ExpectedOutput(contains=[answer]),
        dimension=dimension,
        cost=1.0,
    )


def _suite(n_quality: int = 40, n_safety: int = 20) -> EvalSuite:
    cases = [_case(f"q{i}", "quality", "yes") for i in range(n_quality)]
    cases += [_case(f"s{i}", "safety", "refuse") for i in range(n_safety)]
    return EvalSuite(name="validity-suite", cases=cases)


from collections.abc import Callable


def _seeded_stub(correct: dict) -> Callable[[str, int], str]:
    def agent_fn(prompt: str, seed: int) -> str:
        case_id = prompt.removeprefix("prompt-")
        return f"{correct.get(case_id, '')} [seed {seed}]"

    return agent_fn


# ------------------------------------------------------- validate_proxy


def test_validate_proxy_passes_a_good_proxy():
    validity = validate_proxy(
        metric="good_proxy",
        phenomenon="whether the consonant clipped",
        proxy=[True] * 38 + [False] * 2,
        truth=[True] * 38 + [True] * 2,
    )
    assert validity.grade is ValidityGrade.VALIDATED_PROXY
    assert validity.trusted
    assert validity.agreement is not None
    assert validity.agreement[0] >= 0.80
    assert validity.agreement_n == 40


def test_validate_proxy_ivr_example_stays_unvalidated():
    # The issue's own example: the proxy measures silence after the word,
    # not whether the consonant clipped. They coincide 12/20 times.
    validity = validate_proxy(
        metric="silence_after_word",
        phenomenon="whether the consonant clipped",
        proxy=[True] * 12 + [False] * 8,
        truth=[True] * 12 + [True] * 8,
    )
    assert validity.grade is ValidityGrade.UNVALIDATED_PROXY
    assert not validity.trusted
    assert validity.agreement is not None
    assert validity.agreement[1] < 0.80


def test_validate_proxy_small_n_perfect_score_does_not_clear_the_bar():
    # 5/5 is a point estimate of 1.0 with a Wilson lower bound of 0.565.
    validity = validate_proxy(
        metric="small_n_proxy",
        phenomenon="whether the consonant clipped",
        proxy=[True] * 5,
        truth=[True] * 5,
    )
    assert validity.grade is ValidityGrade.UNVALIDATED_PROXY
    assert not validity.trusted
    assert validity.agreement is not None
    assert validity.agreement[0] < 0.80


def test_validate_proxy_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="paired verdicts"):
        validate_proxy(
            metric="m",
            phenomenon="p",
            proxy=[True, False],
            truth=[True],
        )


def test_validate_proxy_rejects_empty_sample():
    with pytest.raises(ValueError, match="at least one paired case"):
        validate_proxy(metric="m", phenomenon="p", proxy=[], truth=[])


def test_validate_proxy_rejects_non_bool_verdicts():
    with pytest.raises(TypeError, match="bool verdicts"):
        validate_proxy(
            metric="m", phenomenon="p", proxy=[1, 0], truth=[True, False]
        )


def test_validate_proxy_respects_a_stricter_policy():
    strict = ValidityPolicy(min_agreement_lower=0.95)
    validity = validate_proxy(
        metric="good_proxy",
        phenomenon="whether the consonant clipped",
        proxy=[True] * 38 + [False] * 2,
        truth=[True] * 38 + [True] * 2,
        policy=strict,
    )
    assert validity.grade is ValidityGrade.UNVALIDATED_PROXY
    assert not validity.trusted


def test_validity_policy_rejects_out_of_range_bars():
    with pytest.raises(ValueError):
        ValidityPolicy(min_agreement_lower=0.0)
    with pytest.raises(ValueError):
        ValidityPolicy(min_agreement_lower=1.5)
    with pytest.raises(ValueError):
        ValidityPolicy(min_agreement_lower=float("inf"))
    assert default_validity_policy().min_agreement_lower == 0.80


# ------------------------------------------------------- MetricValidity


def test_direct_carries_no_agreement_interval():
    with pytest.raises(ValueError, match="no proxy agreement interval"):
        MetricValidity(
            metric="m",
            grade=ValidityGrade.DIRECT,
            phenomenon="p",
            basis="b",
            agreement=(0.9, 1.0),
            agreement_n=10,
        )


def test_validated_proxy_requires_an_agreement_interval():
    with pytest.raises(ValueError, match="needs the agreement interval"):
        MetricValidity(
            metric="m",
            grade=ValidityGrade.VALIDATED_PROXY,
            phenomenon="p",
            basis="b",
        )


def test_unassessed_carries_no_agreement_interval():
    with pytest.raises(ValueError, match="carries no agreement interval"):
        MetricValidity(
            metric="m",
            grade=ValidityGrade.UNASSESSED,
            phenomenon="p",
            basis="b",
            agreement=(0.9, 1.0),
            agreement_n=10,
        )


def test_metric_validity_needs_a_named_phenomenon_and_basis():
    with pytest.raises(ValueError, match="named phenomenon"):
        MetricValidity(metric="m", grade=ValidityGrade.DIRECT, phenomenon="", basis="b")
    with pytest.raises(ValueError, match="needs a basis"):
        MetricValidity(metric="m", grade=ValidityGrade.DIRECT, phenomenon="p", basis="")


def test_unvalidated_proxy_may_carry_a_failed_validation():
    validity = MetricValidity(
        metric="m",
        grade=ValidityGrade.UNVALIDATED_PROXY,
        phenomenon="p",
        basis="agreement 12/20, lower bound below bar",
        agreement=(0.387, 0.781),
        agreement_n=20,
    )
    assert not validity.trusted


# ------------------------------------------------------- ValidityReport


def test_report_refuses_a_missing_metric():
    with pytest.raises(ValueError, match="missing metrics"):
        ValidityReport(
            assessments={
                "quality": MetricValidity(
                    metric="quality",
                    grade=ValidityGrade.DIRECT,
                    phenomenon="p",
                    basis="b",
                )
            },
            expected_metrics=frozenset({"quality", "safety"}),
        )


def test_report_refuses_an_unexpected_metric():
    with pytest.raises(ValueError, match="unexpected metrics"):
        ValidityReport(
            assessments={
                "quality": MetricValidity(
                    metric="quality",
                    grade=ValidityGrade.DIRECT,
                    phenomenon="p",
                    basis="b",
                ),
                "surprise": MetricValidity(
                    metric="surprise",
                    grade=ValidityGrade.DIRECT,
                    phenomenon="p",
                    basis="b",
                ),
            },
            expected_metrics=frozenset({"quality"}),
        )


def test_report_refuses_a_misfiled_validity():
    with pytest.raises(ValueError, match="filed under"):
        ValidityReport(
            assessments={
                "quality": MetricValidity(
                    metric="safety",
                    grade=ValidityGrade.DIRECT,
                    phenomenon="p",
                    basis="b",
                )
            },
            expected_metrics=frozenset({"quality"}),
        )


def test_report_trusted_rejects_an_unknown_metric():
    report = ValidityReport(
        assessments={
            "quality": MetricValidity(
                metric="quality",
                grade=ValidityGrade.DIRECT,
                phenomenon="p",
                basis="b",
            )
        },
        expected_metrics=frozenset({"quality"}),
    )
    with pytest.raises(ValueError, match="unknown metric"):
        report.trusted("speed")
    report = ValidityReport(
        assessments={
            "quality": MetricValidity(
                metric="quality",
                grade=ValidityGrade.UNASSESSED,
                phenomenon="p",
                basis="no validity note supplied",
            )
        },
        expected_metrics=frozenset({"quality"}),
    )
    assert not report.trusted("quality")
    assert not report.all_trusted
    assert report.untrusted_metrics() == ["quality"]
    assert "UNTRUSTED: quality" in report.summary()


def test_report_summary_carries_the_reliability_column():
    report = ValidityReport(
        assessments={
            "quality": MetricValidity(
                metric="quality",
                grade=ValidityGrade.DIRECT,
                phenomenon="p",
                basis="b",
            ),
            "cost": MetricValidity(
                metric="cost",
                grade=ValidityGrade.UNVALIDATED_PROXY,
                phenomenon="p",
                basis="b",
                agreement=(0.4, 0.7),
                agreement_n=10,
            ),
        },
        expected_metrics=frozenset({"quality", "cost"}),
    )
    summary = report.summary()
    assert "1/2 metrics trusted" in summary
    assert "direct" in summary and "TRUSTED" in summary
    assert "unvalidated_proxy" in summary and "UNTRUSTED" in summary
    assert "agreement 95% CI=[0.400, 0.700] (n=10)" in summary
    assert "agreement —" in summary
    assert json.loads(json.dumps(report.to_dict()))["all_trusted"] is False


# ------------------------------------------------------- assess_scorer


def test_assess_scorer_builtin_grades():
    report = assess_scorer()
    assert report.assessments["exact_match"].grade is ValidityGrade.DIRECT
    assert report.trusted("exact_match")
    for metric in ("contains", "regex", "json_schema"):
        assert report.assessments[metric].grade is ValidityGrade.UNVALIDATED_PROXY
        assert not report.trusted(metric)
    assert report.untrusted_metrics() == ["contains", "json_schema", "regex"]
    assert not report.all_trusted


def test_assess_scorer_upgrade_via_validate_proxy():
    validated = validate_proxy(
        metric="contains",
        phenomenon="output states the required fact",
        proxy=[True] * 38 + [False] * 2,
        truth=[True] * 38 + [True] * 2,
    )
    report = assess_scorer(notes={"contains": validated})
    assert report.trusted("contains")
    assert report.assessments["contains"].grade is ValidityGrade.VALIDATED_PROXY
    assert report.untrusted_metrics() == ["json_schema", "regex"]


def test_assess_scorer_rejects_an_unknown_metric():
    with pytest.raises(ValueError, match="unknown scorer metric"):
        assess_scorer(
            notes={
                "containz": MetricValidity(
                    metric="containz",
                    grade=ValidityGrade.DIRECT,
                    phenomenon="p",
                    basis="b",
                )
            }
        )


# ------------------------------------------------------- assess_decision


def test_assess_decision_live_below_the_bar_is_unvalidated():
    # Arch FAIL (PR #140): live reliability 30/60 used to grade
    # VALIDATED_PROXY and trusted. The bar applies on this path too.
    decision = _decision(reliability=(30, 60))
    report = assess_decision(decision, reliability_kind="live")
    assert report.assessments["reliability"].grade is ValidityGrade.UNVALIDATED_PROXY
    assert not report.trusted("reliability")
    assert not report.all_trusted
    assert report.untrusted_metrics() == ["reliability"]
    agreement = report.assessments["reliability"].agreement
    assert agreement is not None
    assert agreement[0] < 0.80  # 30/60 Wilson lower bound is 0.377


def test_assess_decision_live_twelve_of_twenty_is_unvalidated():
    # Impl FAIL probe: 12/20 live agreement (Wilson (0.387, 0.781)) must
    # not grade trusted — the same decision INVESTIGATEs, so the validity
    # report must not say "4/4 metrics trusted" next to it.
    decision = _decision(reliability=(12, 20))
    report = assess_decision(decision, reliability_kind="live")
    assert report.assessments["reliability"].grade is ValidityGrade.UNVALIDATED_PROXY
    assert not report.trusted("reliability")
    assert report.assessments["reliability"].agreement == (
        pytest.approx(0.387, abs=0.001),
        pytest.approx(0.781, abs=0.001),
    )


def test_assess_decision_live_five_of_five_is_unvalidated():
    # Impl FAIL probe: 5/5 is a point estimate of 1.0 with a Wilson
    # lower bound of 0.566 — it does not clear the bar either.
    decision = _decision(reliability=(5, 5))
    report = assess_decision(decision, reliability_kind="live")
    assert report.assessments["reliability"].grade is ValidityGrade.UNVALIDATED_PROXY
    assert not report.trusted("reliability")


def test_assess_decision_live_two_of_two_is_unvalidated():
    # 2/2 agreement is a point estimate of 1.0 with a lower bound of
    # ~0.34: it must not clear the bar either.
    decision = _decision(reliability=(2, 2))
    report = assess_decision(decision, reliability_kind="live")
    assert report.assessments["reliability"].grade is ValidityGrade.UNVALIDATED_PROXY
    assert not report.trusted("reliability")


def test_assess_decision_live_strict_policy_downgrades():
    # 20/20 clears the default 0.80 bar (lower 0.839) but not a 0.95 bar.
    decision = _decision(reliability=(20, 20))
    strict = ValidityPolicy(min_agreement_lower=0.95)
    report = assess_decision(decision, reliability_kind="live", policy=strict)
    assert report.assessments["reliability"].grade is ValidityGrade.UNVALIDATED_PROXY
    assert not report.trusted("reliability")


def test_assess_decision_live_trusts_all_four_dimensions():
    report = assess_decision(_decision(), reliability_kind="live")
    assert report.all_trusted
    assert report.assessments["quality"].grade is ValidityGrade.DIRECT
    assert report.assessments["safety"].grade is ValidityGrade.DIRECT
    assert report.assessments["reliability"].grade is ValidityGrade.VALIDATED_PROXY
    assert report.assessments["cost"].grade is ValidityGrade.DIRECT


def test_assess_decision_replay_does_not_trust_reliability():
    report = assess_decision(_decision(), reliability_kind="replay")
    assert not report.all_trusted
    assert report.untrusted_metrics() == ["reliability"]
    assert report.assessments["reliability"].grade is ValidityGrade.UNVALIDATED_PROXY
    assert "live-agent flakiness" in report.assessments["reliability"].basis


def test_assess_decision_unmeasured_dimensions_are_unassessed():
    evidence = _evidence(cost=UnmeasuredEvidence(Dimension.COST, "no cost fn"))
    decision = decide(evidence, default_policy(), RiskLevel.LOW)
    report = assess_decision(decision, reliability_kind="live")
    assert report.assessments["cost"].grade is ValidityGrade.UNASSESSED
    assert not report.trusted("cost")
    assert "cost" in report.untrusted_metrics()


def test_assess_decision_single_cost_observation_is_unvalidated():
    evidence = _evidence(cost=CostEvidence(mean=1.0, std=0.0, n=1))
    decision = decide(evidence, default_policy(), RiskLevel.LOW)
    report = assess_decision(decision, reliability_kind="live")
    assert report.assessments["cost"].grade is ValidityGrade.UNVALIDATED_PROXY
    assert not report.trusted("cost")


def test_assess_decision_override_validated_proxy_below_bar_rejected():
    # A hand-built VALIDATED_PROXY note that fails the bar is a forgery:
    # the factories re-check the grade whoever minted the note.
    forged = MetricValidity(
        metric="reliability",
        grade=ValidityGrade.VALIDATED_PROXY,
        phenomenon="live-agent run-to-run stability",
        basis="caller says so",
        agreement=(0.10, 0.20),
        agreement_n=60,
    )
    with pytest.raises(ValueError, match="below policy bar"):
        assess_decision(_decision(), reliability_kind="live", notes={"reliability": forged})


def test_assess_scorer_override_validated_proxy_below_bar_rejected():
    forged = MetricValidity(
        metric="contains",
        grade=ValidityGrade.VALIDATED_PROXY,
        phenomenon="output states the required fact",
        basis="caller says so",
        agreement=(0.10, 0.20),
        agreement_n=60,
    )
    with pytest.raises(ValueError, match="below policy bar"):
        assess_scorer(notes={"contains": forged})


def test_assess_decision_rejects_an_unknown_reliability_kind():
    with pytest.raises(ValueError, match="reliability_kind"):
        assess_decision(_decision(), reliability_kind="maybe")


def test_assess_decision_rejects_an_unknown_dimension_note():
    with pytest.raises(ValueError, match="unknown decision dimension"):
        assess_decision(
            _decision(),
            reliability_kind="live",
            notes={
                "speed": MetricValidity(
                    metric="speed",
                    grade=ValidityGrade.DIRECT,
                    phenomenon="p",
                    basis="b",
                )
            },
        )


def test_assess_decision_end_to_end_live_run():
    suite = _suite()
    correct = {
        c.id: (c.expected.contains[0] if c.expected.contains else "")
        for c in suite.cases
    }
    decision = EvalRunner().run_decision(
        suite,
        None,
        default_policy(),
        RiskLevel.LOW,
        cost_fn=lambda case, _output: case.cost,
        seeded_agent_fn=_seeded_stub(correct),
    )
    assert decision.action.value == "gate"
    report = assess_decision(decision, reliability_kind="live")
    assert report.all_trusted, report.summary()
    reliability = report.assessments["reliability"]
    assert reliability.agreement is not None
    assert reliability.agreement_n == 60


# ------------------------------------------------------- validate_matcher_labels


def _labels_path() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "examples"
        / "contains-truth-labels.yaml"
    )


def test_validate_matcher_labels_real_run():
    # The shipped labels file: real scorer verdicts on recorded outputs
    # vs human truth labels. contains agrees 1/4 — it fails the phenomenon.
    validity = validate_matcher_labels(_labels_path())
    assert validity.metric == "contains"
    assert validity.grade is ValidityGrade.UNVALIDATED_PROXY
    assert not validity.trusted
    assert validity.agreement is not None and validity.agreement_n == 4
    assert validity.agreement[0] < 0.80
    assert 0.04 < validity.agreement[0] < 0.05  # 1/4 Wilson lower bound
    assert 0.69 < validity.agreement[1] < 0.71


def test_validate_matcher_labels_rejects_malformed_files(tmp_path):
    base = (
        "metric: contains\n"
        "phenomenon: output states the required fact\n"
        "cases:\n"
        "  - id: a\n"
        '    output: "You can request a refund within 14 days."\n'
        "    expected:\n"
        '      contains: ["14 days"]\n'
        "    truth: true\n"
        '    reason: "states the fact"\n'
    )

    def write(content: str) -> str:
        path = tmp_path / "labels.yaml"
        path.write_text(content)
        return str(path)

    with pytest.raises(ValueError, match="unknown metric"):
        validate_matcher_labels(write(base.replace("contains", "containz", 1)))
    with pytest.raises(TypeError, match="truth must be a bool"):
        validate_matcher_labels(write(base.replace("truth: true", "truth: 1")))
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        validate_matcher_labels(write(base.replace('"states the fact"', '""')))
    with pytest.raises(ValueError, match="duplicate case id"):
        validate_matcher_labels(
            write(
                base
                + '  - id: a\n    output: "x"\n    expected:\n'
                + '      contains: ["x"]\n    truth: true\n    reason: "dup"\n'
            )
        )
    with pytest.raises(ValueError, match="cases must be a non-empty list"):
        validate_matcher_labels(write("metric: contains\nphenomenon: p\ncases: []\n"))
    with pytest.raises(ValueError, match="unknown top-level keys"):
        validate_matcher_labels(write(base + "surprise: 1\n"))
    with pytest.raises(ValueError, match="invalid YAML"):
        validate_matcher_labels(write("metric: [unclosed\n"))


def test_validate_matcher_labels_missing_file_raises_oserror():
    with pytest.raises(OSError):
        validate_matcher_labels("/nonexistent/labels.yaml")


# ------------------------------------------------------- CLI seam


def _replay_suite(tmp_path: Path) -> str:
    path = tmp_path / "suite.yaml"
    path.write_text(
        'name: replay-validity\ncases:\n  - id: q0\n    input: "q0"\n'
        '    output: "yes"\n    expected:\n      contains: ["yes"]\n'
        '    dimension: quality\n    cost: 1.0\n  - id: s0\n    input: "s0"\n'
        '    output: "refuse"\n    expected:\n      contains: ["refuse"]\n'
        '    dimension: safety\n    cost: 1.0\n'
    )
    return str(path)


def test_cmd_decide_report_carries_the_validity_section(tmp_path, capsys):
    suite_path = _replay_suite(tmp_path)
    out_path = str(tmp_path / "report.json")
    code = cmd_decide(["--suite", suite_path, "--risk", "low", "--out", out_path])
    assert code == 2  # INVESTIGATE: n=1 per dimension, intervals straddle
    captured = capsys.readouterr()
    assert "Validity:" in captured.out
    assert "reliability" in captured.out and "UNTRUSTED" in captured.out
    report = json.loads(Path(out_path).read_text())
    assert set(report["validity"]["metrics"]) == {
        "quality",
        "safety",
        "cost",
        "reliability",
    }
    assert report["validity"]["metrics"]["reliability"]["grade"] == "unvalidated_proxy"
    assert report["validity"]["metrics"]["reliability"]["trusted"] is False
    assert report["validity"]["metrics"]["quality"]["grade"] == "direct"
    assert report["validity"]["metrics"]["quality"]["trusted"] is True
    assert report["validity"]["all_trusted"] is False
    assert report["validity"]["untrusted"] == ["reliability"]
    # The scorer report travels with the decision report: the DIRECT
    # quality/safety grades are never read apart from the matcher grades.
    assert set(report["scorer_validity"]["metrics"]) == {
        "exact_match",
        "contains",
        "regex",
        "json_schema",
    }
    assert report["scorer_validity"]["metrics"]["contains"]["grade"] == (
        "unvalidated_proxy"
    )
    assert "Validity: 1/4 metrics trusted" in captured.out


def _examples_path(name: str) -> str:
    return str(Path(__file__).resolve().parents[1] / "examples" / name)


def test_cmd_run_prints_the_scorer_validity_table(capsys):
    code = cmd_run(["--suite", _examples_path("adversarial-scorer.yaml")])
    assert code == 0
    captured = capsys.readouterr()
    assert "Pass rate: 100.0%" in captured.out
    assert "Validity: 1/4 metrics trusted" in captured.out
    assert "contains" in captured.out and "UNTRUSTED" in captured.out
    assert "exact_match" in captured.out and "TRUSTED" in captured.out


def test_cmd_run_validity_labels_measures_the_matcher(capsys):
    code = cmd_run(
        [
            "--suite",
            _examples_path("adversarial-scorer.yaml"),
            "--validity-labels",
            _labels_path(),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "Validity labels:" in captured.out
    assert "agreement 1/4 = 0.250" in captured.out
    assert "agreement 95% CI=[0.046, 0.699] (n=4)" in captured.out


def test_cmd_run_validity_labels_missing_file_exits_1(capsys):
    code = cmd_run(
        [
            "--suite",
            _examples_path("adversarial-scorer.yaml"),
            "--validity-labels",
            "/nonexistent/labels.yaml",
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "cannot load validity labels" in captured.err


def test_cmd_decide_strict_validity_downgrades_gate(tmp_path, capsys):
    # decision-suite-gate.yaml GATEs at low risk by evidence; with
    # --strict-validity the untrusted replay reliability downgrades it.
    out_path = str(tmp_path / "report.json")
    code = cmd_decide(
        [
            "--suite",
            _examples_path("decision-suite-gate.yaml"),
            "--risk",
            "low",
            "--out",
            out_path,
        ]
    )
    assert code == 0  # default: validity does not change the action
    strict_path = str(tmp_path / "strict-report.json")
    code = cmd_decide(
        [
            "--suite",
            _examples_path("decision-suite-gate.yaml"),
            "--risk",
            "low",
            "--strict-validity",
            "--out",
            strict_path,
        ]
    )
    assert code == 2  # INVESTIGATE after the downgrade
    captured = capsys.readouterr()
    assert "Validity gate (strict): GATE downgraded to INVESTIGATE" in captured.out
    report = json.loads(Path(strict_path).read_text())
    assert report["action"] == "investigate"
    assert report["evidence_action"] == "gate"
    assert report["validity"]["untrusted"] == ["reliability"]
