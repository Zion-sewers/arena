"""Seeded end-to-end demo of measurement validity (arena#43).

Runs the full "test the test" pipeline offline on in-memory suites with
hand-written stub agents. Exit 0 only if every acceptance expectation
holds:
- the scorer's built-in report trusts exact_match (DIRECT) and refuses
  to trust contains/regex/json_schema (UNVALIDATED_PROXY);
- the issue's IVR example: a proxy that measures silence after the word,
  not whether the consonant clipped, agrees with ground truth only 12/20
  times and stays UNVALIDATED_PROXY and untrusted;
- a good proxy (38/40 agreement) clears the bar and is VALIDATED_PROXY;
- a perfect-looking proxy at small n (5/5) does NOT clear the bar: the
  Wilson lower bound, not the point estimate, is the compared quantity;
- a decision assessed as live trusts all four dimensions (quality and
  safety DIRECT, reliability VALIDATED_PROXY across seeded runs, cost
  DIRECT over measured observations);
- the same decision assessed as replay does NOT trust reliability
  (replay-agreement cannot detect live-agent flakiness);
- a report with an unassessed metric never trusts it, and a report that
  omits a metric refuses to be built at all.

Any failure raises, so CI can gate on this demo directly.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from agenteval_bench.decision import (
    Dimension,
    RiskLevel,
    default_policy,
)
from agenteval_bench.engine import EvalRunner
from agenteval_bench.models import EvalCase, EvalSuite, ExpectedOutput
from agenteval_bench.validity import (
    MetricValidity,
    ValidityGrade,
    ValidityReport,
    assess_decision,
    assess_scorer,
    validate_proxy,
)


def _case(case_id: str, dimension: str, answer: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        input=f"prompt-{case_id}",
        expected=ExpectedOutput(contains=[answer]),
        dimension=dimension,
        cost=1.0,
    )


def _suite(name: str, n_quality: int, n_safety: int) -> EvalSuite:
    cases = [_case(f"q{i}", "quality", "yes") for i in range(n_quality)]
    cases += [_case(f"s{i}", "safety", "refuse") for i in range(n_safety)]
    return EvalSuite(name=name, cases=cases)


def _seeded_stub(correct: dict[str, str]) -> Callable[[str, int], str]:
    def agent_fn(prompt: str, seed: int) -> str:
        case_id = prompt.removeprefix("prompt-")
        return f"{correct.get(case_id, '')} [seed {seed}]"

    return agent_fn


def main() -> int:
    # 1. Scorer built-ins: exact_match trusted, the proxies are not.
    scorer = assess_scorer()
    print(scorer.summary())
    assert scorer.trusted("exact_match"), scorer.summary()
    assert scorer.assessments["exact_match"].grade is ValidityGrade.DIRECT
    assert scorer.untrusted_metrics() == ["contains", "json_schema", "regex"]
    assert not scorer.all_trusted
    assert not scorer.trusted("contains")

    # 2. IVR example: the proxy measures silence after the word; ground
    # truth is whether the consonant clipped. They coincide 12/20 times.
    ivr = validate_proxy(
        metric="silence_after_word",
        phenomenon="whether the consonant clipped",
        proxy=[True] * 12 + [False] * 8,
        truth=[True] * 12 + [True] * 8,
    )
    print(f"IVR proxy: grade={ivr.grade.value} trusted={ivr.trusted}")
    assert ivr.grade is ValidityGrade.UNVALIDATED_PROXY
    assert not ivr.trusted
    assert ivr.agreement is not None and ivr.agreement[1] < 0.80

    # 3. A good proxy: 38/40 agreement clears the bar.
    good = validate_proxy(
        metric="good_proxy",
        phenomenon="whether the consonant clipped",
        proxy=[True] * 38 + [False] * 2,
        truth=[True] * 38 + [True] * 2,
    )
    assert good.grade is ValidityGrade.VALIDATED_PROXY
    assert good.trusted
    assert good.agreement is not None and good.agreement[0] >= 0.80

    # 4. Small-n trap: 5/5 agreement is a point estimate of 1.0 with a
    # Wilson lower bound of 0.565 — it does not clear the 0.80 bar.
    small_n = validate_proxy(
        metric="small_n_proxy",
        phenomenon="whether the consonant clipped",
        proxy=[True] * 5,
        truth=[True] * 5,
    )
    assert small_n.grade is ValidityGrade.UNVALIDATED_PROXY
    assert not small_n.trusted
    assert small_n.agreement is not None and small_n.agreement[0] < 0.80

    # 5. A live decision trusts all four dimensions.
    runner = EvalRunner()
    suite = _suite("demo-validity-live", 40, 20)
    correct = {c.id: (c.expected.contains[0] if c.expected.contains else "") for c in suite.cases}
    decision = runner.run_decision(
        suite,
        None,
        default_policy(),
        RiskLevel.LOW,
        cost_fn=lambda case, _output: case.cost,
        seeded_agent_fn=_seeded_stub(correct),
    )
    assert decision.action.value == "gate", decision.summary()
    live = assess_decision(decision, reliability_kind="live")
    print(live.summary())
    assert live.all_trusted, live.summary()
    assert live.assessments[Dimension.RELIABILITY.value].grade is ValidityGrade.VALIDATED_PROXY
    assert live.assessments[Dimension.COST.value].grade is ValidityGrade.DIRECT

    # 6. The same decision as replay does not trust reliability.
    replay = assess_decision(decision, reliability_kind="replay")
    print(replay.summary())
    assert not replay.all_trusted
    assert replay.untrusted_metrics() == [Dimension.RELIABILITY.value]
    assert replay.assessments[Dimension.RELIABILITY.value].grade is (
        ValidityGrade.UNVALIDATED_PROXY
    )

    # 7. An unassessed metric is never trusted, and a report that omits
    # a metric refuses to be built.
    unassessed = MetricValidity(
        metric="quality",
        grade=ValidityGrade.UNASSESSED,
        phenomenon="share of this suite's quality cases passed",
        basis="no validity note supplied",
    )
    assert not unassessed.trusted
    try:
        ValidityReport(
            assessments={"quality": unassessed},
            expected_metrics=frozenset({"quality", "safety"}),
        )
    except ValueError as exc:
        assert "missing metrics" in str(exc), exc
    else:  # pragma: no cover - the constructor must raise
        raise AssertionError("ValidityReport built with a missing metric")

    print("validity demo: all 7 expectations hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
