"""Seeded end-to-end demo of the evaluation decision system (arena#42).

Runs the full evidence -> uncertainty -> threshold -> action pipeline
offline on in-memory suites with hand-written stub agents. Exit 0 only
if every acceptance expectation holds:
- a strong suite GATEs at low risk (all four dimensions measured, every
  interval clears its threshold);
- a suite whose safety cases fail ROLLBACKs at high risk (its safety
  interval lies entirely below the threshold);
- a 3-case suite INVESTIGATEs at high risk (the point estimate passes
  but the Wilson interval straddles the threshold);
- a suite with no cost function INVESTIGATEs (unmeasured cost never
  silently gates);
- the same evidence GATEs at low risk and INVESTIGATEs at critical risk
  (thresholds are per risk level, not one threshold for everything).

Any failure raises, so CI can gate on this demo directly.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from agenteval_bench.decision import (
    CostEvidence,
    DecisionAction,
    Dimension,
    Evidence,
    ProportionEvidence,
    RiskLevel,
    decide,
    default_policy,
)
from agenteval_bench.engine import EvalRunner
from agenteval_bench.models import EvalCase, EvalSuite, ExpectedOutput


def _case(case_id: str, dimension: str, answer: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        input=f"prompt-{case_id}",
        expected=ExpectedOutput(exact=answer),
        dimension=dimension,
        cost=1.0,
    )


def _suite(name: str, n_quality: int, n_safety: int) -> EvalSuite:
    cases = [_case(f"q{i}", "quality", "yes") for i in range(n_quality)]
    cases += [_case(f"s{i}", "safety", "refuse") for i in range(n_safety)]
    return EvalSuite(name=name, cases=cases)


def _stub_agent(correct: dict[str, str]) -> Callable[[str], str]:
    """Hand-written stub agent: answers from a recorded input->output map."""

    def agent_fn(prompt: str) -> str:
        case_id = prompt.removeprefix("prompt-")
        return correct.get(case_id, "")

    return agent_fn


def _all_correct(suite: EvalSuite) -> dict[str, str]:
    return {c.id: (c.expected.exact or "") for c in suite.cases}


def main() -> int:
    runner = EvalRunner()
    policy = default_policy()

    # 1. GATE: 40 quality + 20 safety cases, all passing, cost measured.
    suite = _suite("demo-gate", 40, 20)
    decision = runner.run_decision(
        suite,
        _stub_agent(_all_correct(suite)),
        policy,
        RiskLevel.LOW,
        cost_fn=lambda case, _output: case.cost,
    )
    print(decision.summary())
    assert decision.action is DecisionAction.GATE, decision.summary()

    # 2. ROLLBACK: safety cases fail, so the safety interval is entirely
    # below the high-risk threshold.
    suite = _suite("demo-rollback", 40, 20)
    correct = _all_correct(suite)
    for case in suite.cases:
        if case.dimension == "safety":
            correct[case.id] = "comply"  # wrong answer: safety fails
    decision = runner.run_decision(
        suite,
        _stub_agent(correct),
        policy,
        RiskLevel.HIGH,
        cost_fn=lambda case, _output: case.cost,
    )
    print(decision.summary())
    assert decision.action is DecisionAction.ROLLBACK, decision.summary()
    assert decision.verdicts[Dimension.SAFETY].action is DecisionAction.ROLLBACK

    # 3. INVESTIGATE: 3/3 quality passes, but n=3 leaves the Wilson
    # interval straddling the high-risk threshold.
    suite = _suite("demo-straddle", 3, 20)
    decision = runner.run_decision(
        suite,
        _stub_agent(_all_correct(suite)),
        policy,
        RiskLevel.HIGH,
        cost_fn=lambda case, _output: case.cost,
    )
    print(decision.summary())
    assert decision.action is DecisionAction.INVESTIGATE, decision.summary()
    assert decision.verdicts[Dimension.QUALITY].action is DecisionAction.INVESTIGATE

    # 4. INVESTIGATE: no cost function, so cost is unmeasured.
    suite = _suite("demo-unmeasured-cost", 40, 20)
    decision = runner.run_decision(
        suite, _stub_agent(_all_correct(suite)), policy, RiskLevel.LOW
    )
    print(decision.summary())
    assert decision.action is DecisionAction.INVESTIGATE, decision.summary()
    assert decision.verdicts[Dimension.COST].action is DecisionAction.INVESTIGATE

    # 5. Same evidence, different risk: 20/20 quality passes gate at low
    # risk (lower bound 0.839 >= 0.70) and investigate at critical risk
    # (lower bound 0.839 < 0.95).
    evidence: dict[Dimension, Evidence] = {
        Dimension.QUALITY: ProportionEvidence(Dimension.QUALITY, 20, 20),
        Dimension.SAFETY: ProportionEvidence(Dimension.SAFETY, 20, 20),
        Dimension.RELIABILITY: ProportionEvidence(Dimension.RELIABILITY, 20, 20),
        Dimension.COST: CostEvidence(mean=1.0, std=0.1, n=20, budget=100.0),
    }
    low = decide(evidence, policy, RiskLevel.LOW)
    critical = decide(evidence, policy, RiskLevel.CRITICAL)
    print(f"low risk: {low.action.value}; critical risk: {critical.action.value}")
    assert low.action is DecisionAction.GATE, low.summary()
    assert critical.action is DecisionAction.INVESTIGATE, critical.summary()

    print("decision demo: all 5 expectations hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
