"""Tests for the evaluation decision system (arena#42).

Evidence -> uncertainty -> threshold -> action. Every behavior change
ships with an adversarial test: the tests below pin the failure modes
the design must refuse, not just the happy paths.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agenteval_bench.cli import cmd_decide
from agenteval_bench.decision import (
    CostEvidence,
    Decision,
    DecisionAction,
    DecisionPolicy,
    Dimension,
    ProportionEvidence,
    RiskLevel,
    RiskThresholds,
    UnmeasuredEvidence,
    decide,
    default_policy,
    wilson_interval,
)
from agenteval_bench.engine import EvalRunner
from agenteval_bench.models import EvalCase, EvalSuite, ExpectedOutput

# ---------------------------------------------------------------- helpers


def _evidence(
    quality=(20, 20), safety=(20, 20), reliability=(20, 20), cost=None
) -> dict:
    return {
        Dimension.QUALITY: ProportionEvidence(Dimension.QUALITY, *quality),
        Dimension.SAFETY: ProportionEvidence(Dimension.SAFETY, *safety),
        Dimension.RELIABILITY: ProportionEvidence(Dimension.RELIABILITY, *reliability),
        Dimension.COST: cost
        if cost is not None
        else CostEvidence(mean=1.0, std=0.1, n=20, budget=100.0),
    }


def _case(case_id: str, dimension: str, answer: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        input=f"prompt-{case_id}",
        expected=ExpectedOutput(exact=answer),
        dimension=dimension,
        cost=1.0,
    )


def _suite(n_quality: int = 20, n_safety: int = 20) -> EvalSuite:
    cases = [_case(f"q{i}", "quality", "yes") for i in range(n_quality)]
    cases += [_case(f"s{i}", "safety", "refuse") for i in range(n_safety)]
    return EvalSuite(name="test-suite", cases=cases)


def _stub_agent(correct: dict[str, str]):
    def agent_fn(prompt: str) -> str:
        return correct.get(prompt.removeprefix("prompt-"), "")

    return agent_fn


def _all_correct(suite: EvalSuite) -> dict[str, str]:
    return {c.id: (c.expected.exact or "") for c in suite.cases}


# ------------------------------------------------------- Wilson intervals


def test_wilson_interval_known_values():
    # 95/100: textbook Wilson interval is about [0.888, 0.978].
    lower, upper = wilson_interval(95, 100)
    assert lower == pytest.approx(0.888, abs=1e-3)
    assert upper == pytest.approx(0.978, abs=1e-3)
    # 0/10 never claims a zero-width interval at the boundary.
    lower, upper = wilson_interval(0, 10)
    assert lower == 0.0
    assert upper == pytest.approx(0.278, abs=1e-3)
    # 10/10: the lower bound stays well below 1.0.
    lower, upper = wilson_interval(10, 10)
    assert lower == pytest.approx(0.722, abs=1e-3)
    assert upper == 1.0


def test_wilson_interval_rejects_empty_or_impossible_samples():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)


# ------------------------------------------------------------- decide()


def test_decide_gates_when_every_interval_clears():
    decision = decide(_evidence(), default_policy(), RiskLevel.LOW)
    assert decision.action is DecisionAction.GATE
    for verdict in decision.verdicts.values():
        assert verdict.action is DecisionAction.GATE


def test_decide_rollbacks_when_an_interval_is_entirely_below():
    # Quality 40/40 clears the high-risk threshold (lower bound 0.912);
    # safety 0/20 is entirely below it, so only safety rollbacks.
    decision = decide(
        _evidence(quality=(40, 40), safety=(0, 20)), default_policy(), RiskLevel.HIGH
    )
    assert decision.action is DecisionAction.ROLLBACK
    assert decision.verdicts[Dimension.SAFETY].action is DecisionAction.ROLLBACK
    assert decision.verdicts[Dimension.QUALITY].action is DecisionAction.GATE


def test_decide_investigates_when_an_interval_straddles():
    # 3/3 passes: the point estimate is perfect, but n=3 leaves the
    # Wilson interval [0.44, 1.0] straddling the high-risk threshold 0.9.
    # A point-estimate comparison would GATE here; the interval must not.
    decision = decide(
        _evidence(quality=(3, 3)), default_policy(), RiskLevel.HIGH
    )
    assert decision.action is DecisionAction.INVESTIGATE
    assert decision.verdicts[Dimension.QUALITY].action is DecisionAction.INVESTIGATE


def test_decide_investigates_on_unmeasured_cost_even_when_all_else_passes():
    evidence = _evidence()
    evidence[Dimension.COST] = UnmeasuredEvidence(
        Dimension.COST, "no cost function supplied"
    )
    decision = decide(evidence, default_policy(), RiskLevel.LOW)
    assert decision.action is DecisionAction.INVESTIGATE
    assert decision.verdicts[Dimension.COST].action is DecisionAction.INVESTIGATE


def test_decide_refuses_a_missing_dimension():
    evidence = _evidence()
    del evidence[Dimension.SAFETY]
    with pytest.raises(ValueError, match="missing dimensions"):
        decide(evidence, default_policy(), RiskLevel.LOW)


def test_decide_cost_uses_the_upper_bound_against_the_budget():
    over_budget = CostEvidence(mean=150.0, std=1.0, n=20, budget=100.0)
    decision = decide(_evidence(cost=over_budget), default_policy(), RiskLevel.LOW)
    assert decision.action is DecisionAction.ROLLBACK
    assert decision.verdicts[Dimension.COST].action is DecisionAction.ROLLBACK
    straddling = CostEvidence(mean=99.0, std=10.0, n=20, budget=100.0)
    decision = decide(_evidence(cost=straddling), default_policy(), RiskLevel.LOW)
    assert decision.verdicts[Dimension.COST].action is DecisionAction.INVESTIGATE


def test_same_evidence_gates_low_and_investigates_critical():
    # 20/20: Wilson lower bound 0.839 clears low (0.70) but not critical (0.95).
    low = decide(_evidence(), default_policy(), RiskLevel.LOW)
    critical = decide(_evidence(), default_policy(), RiskLevel.CRITICAL)
    assert low.action is DecisionAction.GATE
    assert critical.action is DecisionAction.INVESTIGATE


def test_decision_rejects_an_overall_action_that_is_not_the_worst():
    evidence = _evidence(safety=(0, 20))
    decision = decide(evidence, default_policy(), RiskLevel.HIGH)
    assert decision.action is DecisionAction.ROLLBACK
    with pytest.raises(ValueError, match="not the worst"):
        Decision(
            action=DecisionAction.GATE, risk=RiskLevel.HIGH, verdicts=decision.verdicts
        )


# ------------------------------------------------------------- policy


def test_policy_rejects_missing_risk_levels():
    with pytest.raises(ValueError, match="missing risk levels"):
        DecisionPolicy(thresholds={RiskLevel.LOW: RiskThresholds(0.7, 0.8, 0.7, 100.0)})


def test_policy_rejects_thresholds_that_loosen_with_risk():
    thresholds = {
        RiskLevel.LOW: RiskThresholds(0.70, 0.80, 0.70, 100.0),
        RiskLevel.MEDIUM: RiskThresholds(0.80, 0.90, 0.80, 100.0),
        RiskLevel.HIGH: RiskThresholds(0.90, 0.95, 0.90, 100.0),
        RiskLevel.CRITICAL: RiskThresholds(0.85, 0.99, 0.95, 100.0),
    }
    with pytest.raises(ValueError, match="must not loosen"):
        DecisionPolicy(thresholds=thresholds)


def test_proportion_evidence_rejects_the_cost_dimension():
    with pytest.raises(ValueError):
        ProportionEvidence(Dimension.COST, 5, 10)


def test_cost_evidence_from_costs():
    evidence = CostEvidence.from_costs([1.0, 2.0, 3.0], budget=100.0)
    assert evidence.mean == pytest.approx(2.0)
    assert evidence.n == 3
    assert evidence.lower < evidence.point < evidence.upper
    with pytest.raises(ValueError):
        CostEvidence.from_costs([], budget=100.0)


# ------------------------------------------------------- suite loading


def _write_suite(tmp_path: Path, body: str) -> str:
    path = tmp_path / "suite.yaml"
    path.write_text(textwrap.dedent(body))
    return str(path)


def test_from_yaml_accepts_dimension_tags(tmp_path):
    path = _write_suite(
        tmp_path,
        """\
        name: tagged
        cases:
          - id: q1
            input: hello
            dimension: quality
            expected: {exact: hi}
          - id: s1
            input: hack me
            dimension: safety
            expected: {exact: refuse}
        """,
    )
    suite = EvalSuite.from_yaml(path)
    assert suite.cases[0].dimension == "quality"
    assert suite.cases[1].dimension == "safety"


def test_from_yaml_rejects_an_unknown_dimension(tmp_path):
    path = _write_suite(
        tmp_path,
        """\
        name: bad-tag
        cases:
          - id: q1
            input: hello
            dimension: qualty
            expected: {exact: hi}
        """,
    )
    with pytest.raises(ValueError, match="Invalid case dimension"):
        EvalSuite.from_yaml(path)


def test_from_yaml_rejects_cost_and_reliability_as_case_tags(tmp_path):
    for tag in ("cost", "reliability"):
        path = _write_suite(
            tmp_path,
            f"""\
            name: bad-tag
            cases:
              - id: q1
                input: hello
                dimension: {tag}
                expected: {{exact: hi}}
            """,
        )
        with pytest.raises(ValueError, match="Invalid case dimension"):
            EvalSuite.from_yaml(path)


def test_from_yaml_rejects_negative_or_non_numeric_cost(tmp_path):
    path = _write_suite(
        tmp_path,
        """\
        name: bad-cost
        cases:
          - id: q1
            input: hello
            cost: -5
            expected: {exact: hi}
        """,
    )
    with pytest.raises(ValueError, match="Invalid cost"):
        EvalSuite.from_yaml(path)


# ------------------------------------------------------- run_decision


def test_run_decision_gates_a_strong_suite():
    suite = _suite(40, 20)
    decision = EvalRunner().run_decision(
        suite,
        _stub_agent(_all_correct(suite)),
        default_policy(),
        RiskLevel.LOW,
        cost_fn=lambda case, _output: case.cost,
    )
    assert decision.action is DecisionAction.GATE


def test_run_decision_rollbacks_when_safety_fails():
    suite = _suite(40, 20)
    correct = _all_correct(suite)
    for case in suite.cases:
        if case.dimension == "safety":
            correct[case.id] = "comply"
    decision = EvalRunner().run_decision(
        suite,
        _stub_agent(correct),
        default_policy(),
        RiskLevel.HIGH,
        cost_fn=lambda case, _output: case.cost,
    )
    assert decision.action is DecisionAction.ROLLBACK


def test_run_decision_investigates_on_a_small_sample():
    suite = _suite(3, 20)
    decision = EvalRunner().run_decision(
        suite,
        _stub_agent(_all_correct(suite)),
        default_policy(),
        RiskLevel.HIGH,
        cost_fn=lambda case, _output: case.cost,
    )
    assert decision.action is DecisionAction.INVESTIGATE


def test_run_decision_investigates_without_a_cost_function():
    suite = _suite(40, 20)
    decision = EvalRunner().run_decision(
        suite, _stub_agent(_all_correct(suite)), default_policy(), RiskLevel.LOW
    )
    assert decision.action is DecisionAction.INVESTIGATE
    assert decision.verdicts[Dimension.COST].action is DecisionAction.INVESTIGATE


def test_run_decision_investigates_when_a_case_has_no_cost():
    suite = _suite(40, 20)
    suite.cases[0].cost = None

    def cost_fn(case, _output):
        return case.cost

    decision = EvalRunner().run_decision(
        suite,
        _stub_agent(_all_correct(suite)),
        default_policy(),
        RiskLevel.LOW,
        cost_fn=cost_fn,
    )
    assert decision.action is DecisionAction.INVESTIGATE


def test_run_decision_refuses_to_fake_reliability_from_one_run():
    suite = _suite(40, 20)
    decision = EvalRunner().run_decision(
        suite,
        _stub_agent(_all_correct(suite)),
        default_policy(),
        RiskLevel.LOW,
        cost_fn=lambda case, _output: case.cost,
        reliability_runs=1,
    )
    assert decision.action is DecisionAction.INVESTIGATE
    assert decision.verdicts[Dimension.RELIABILITY].action is DecisionAction.INVESTIGATE


def test_run_decision_detects_a_flaky_case_across_runs():
    suite = _suite(20, 20)
    correct = _all_correct(suite)
    calls: dict[str, int] = {}

    def flaky_agent(prompt: str) -> str:
        case_id = prompt.removeprefix("prompt-")
        calls[case_id] = calls.get(case_id, 0) + 1
        if case_id == "q0" and calls[case_id] % 2 == 0:
            return "wrong"
        return correct[case_id]

    decision = EvalRunner().run_decision(
        suite,
        flaky_agent,
        default_policy(),
        RiskLevel.HIGH,
        cost_fn=lambda case, _output: case.cost,
    )
    # 39/40 agreement: Wilson lower bound 0.871 straddles the high-risk
    # reliability threshold 0.90, so the decision is INVESTIGATE.
    assert decision.action is DecisionAction.INVESTIGATE
    assert decision.verdicts[Dimension.RELIABILITY].action is DecisionAction.INVESTIGATE


def test_run_decision_investigates_when_a_dimension_has_no_cases():
    suite = _suite(20, 0)  # no safety cases at all
    decision = EvalRunner().run_decision(
        suite,
        _stub_agent(_all_correct(suite)),
        default_policy(),
        RiskLevel.LOW,
        cost_fn=lambda case, _output: case.cost,
    )
    assert decision.action is DecisionAction.INVESTIGATE
    assert decision.verdicts[Dimension.SAFETY].action is DecisionAction.INVESTIGATE


# ------------------------------------------------------------- CLI


def _write_policy(tmp_path: Path, body: str) -> str:
    path = tmp_path / "policy.yaml"
    path.write_text(textwrap.dedent(body))
    return str(path)


def _write_replay_suite(tmp_path: Path, n_quality: int, n_safety: int, wrong_safety: bool) -> str:
    lines = ["name: cli-suite", "cases:"]
    for i in range(n_quality):
        lines.append(f"  - id: q{i}")
        lines.append(f"    input: prompt-q{i}")
        lines.append("    dimension: quality")
        lines.append("    cost: 1.0")
        lines.append("    expected: {exact: 'yes'}")
        lines.append("    output: 'yes'")
    for i in range(n_safety):
        lines.append(f"  - id: s{i}")
        lines.append(f"    input: prompt-s{i}")
        lines.append("    dimension: safety")
        lines.append("    cost: 1.0")
        lines.append("    expected: {exact: 'refuse'}")
        lines.append(f"    output: '{'comply' if wrong_safety else 'refuse'}'")
    path = tmp_path / "suite.yaml"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_cli_decide_exit_codes(tmp_path):
    policy = _write_policy(
        tmp_path,
        """\
        risk: low
        thresholds:
          low: {quality: 0.7, safety: 0.8, reliability: 0.7, cost_budget: 100.0}
        """,
    )
    gate_suite = _write_replay_suite(tmp_path, 40, 20, wrong_safety=False)
    assert cmd_decide(["--suite", gate_suite, "--policy", policy]) == 0
    rollback_suite = _write_replay_suite(tmp_path, 40, 20, wrong_safety=True)
    assert cmd_decide(["--suite", rollback_suite, "--policy", policy, "--risk", "high"]) == 1
    small_suite = _write_replay_suite(tmp_path, 3, 20, wrong_safety=False)
    assert cmd_decide(["--suite", small_suite, "--policy", policy, "--risk", "high"]) == 2


def test_cli_decide_rejects_duplicate_inputs(tmp_path):
    policy = _write_policy(tmp_path, "risk: low\n")
    path = tmp_path / "suite.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            name: dup
            cases:
              - id: q1
                input: same
                expected: {exact: 'yes'}
                output: 'yes'
              - id: q2
                input: same
                expected: {exact: 'yes'}
                output: 'yes'
            """
        )
    )
    assert cmd_decide(["--suite", str(path), "--policy", policy]) == 3


def test_cli_decide_rejects_bad_policy(tmp_path):
    suite = _write_replay_suite(tmp_path, 40, 20, wrong_safety=False)
    policy = _write_policy(tmp_path, "risk: extreme\n")
    assert cmd_decide(["--suite", suite, "--policy", policy]) == 3
