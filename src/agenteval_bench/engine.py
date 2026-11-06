"""Core eval runner — orchestrates suite execution."""

from __future__ import annotations

import random
from collections.abc import Callable

from agenteval_bench.decision import (
    CostEvidence,
    Decision,
    DecisionPolicy,
    Dimension,
    Evidence,
    ProportionEvidence,
    RiskLevel,
    UnmeasuredEvidence,
    decide,
    default_policy,
)
from agenteval_bench.models import EvalCase, EvalResult, EvalSuite, RunResult
from agenteval_bench.scoring import DeterministicScorer

AgentFn = Callable[[str], str]
# Per-case cost in the policy's cost units. None for a case means that
# case's cost was not measured; one unmeasured case makes the whole cost
# dimension unmeasured (never silently gated on a partial sample).
CostFn = Callable[[EvalCase, str], float | None]

DEFAULT_SEED = 42


class EvalRunner:
    """Runs eval suites against an agent function."""

    def __init__(self, scorer: DeterministicScorer | None = None) -> None:
        self.scorer = scorer or DeterministicScorer()

    def _execute(
        self, suite: EvalSuite, agent_fn: AgentFn, seed: int
    ) -> tuple[list[EvalResult], dict[str, str]]:
        """Run one seeded pass, returning (results, outputs by case id).

        A ``random.Random(seed)`` stream is bound to the run and one draw per
        case is recorded in ``details["rng_draw"]``. The deterministic scorer
        needs no randomness today; the draw pins the stream so any future
        stochastic component (subset sampling, tie-breaking) replays
        bit-exactly under the same seed.
        """
        rng = random.Random(seed)
        results: list[EvalResult] = []
        outputs: dict[str, str] = {}

        for case in suite.cases:
            rng_draw = rng.random()
            if case.skip:
                results.append(
                    EvalResult(case_id=case.id, passed=False, score=0.0,
                               details={"skipped": True, "rng_draw": rng_draw})
                )
                continue

            agent_output = agent_fn(case.input)
            outputs[case.id] = agent_output
            result = self.scorer.score(case, agent_output)
            result.details["rng_draw"] = rng_draw
            results.append(result)

        return results, outputs

    def run(self, suite: EvalSuite, agent_fn: AgentFn, seed: int = DEFAULT_SEED) -> RunResult:
        """Execute all cases in the suite and return aggregate results."""
        results, _outputs = self._execute(suite, agent_fn, seed)

        total = len(results)
        skipped = sum(1 for r in results if r.details.get("skipped"))
        scored = [r for r in results if not r.details.get("skipped")]
        passed = sum(1 for r in scored if r.passed)
        failed = len(scored) - passed
        pass_rate = passed / len(scored) if scored else 0.0

        return RunResult(
            suite_name=suite.name,
            results=results,
            total=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            pass_rate=pass_rate,
        )

    def run_ci(self, suite: EvalSuite, agent_fn: AgentFn, threshold: float = 1.0) -> RunResult:
        """Run in CI mode — returns result with pass_rate for threshold check."""
        result = self.run(suite, agent_fn)
        return result

    def run_decision(
        self,
        suite: EvalSuite,
        agent_fn: AgentFn,
        policy: DecisionPolicy | None = None,
        risk: RiskLevel = RiskLevel.MEDIUM,
        *,
        cost_fn: CostFn | None = None,
        reliability_runs: int = 2,
        seed: int = DEFAULT_SEED,
    ) -> Decision:
        """Run a suite and map the evidence to a GATE/ROLLBACK/INVESTIGATE decision.

        Evidence per dimension (see decision.py for the mapping rules):
        * quality / safety: pass proportions over the scored cases tagged
          with that dimension. A dimension with no scored cases is
          unmeasured and forces INVESTIGATE.
        * reliability: per-case pass/fail agreement across ``reliability_runs``
          seeded runs. One run cannot measure agreement (it is 1.0 by
          construction), so ``reliability_runs < 2`` leaves reliability
          unmeasured instead of faking it.
        * cost: mean + standard-error interval over ``cost_fn`` observations
          against the policy's per-risk budget. No ``cost_fn`` means cost is
          unmeasured and forces INVESTIGATE.
        """
        policy = policy or default_policy()
        if reliability_runs < 1:
            raise ValueError("reliability_runs must be >= 1")

        results, outputs = self._execute(suite, agent_fn, seed)
        by_id = {case.id: case for case in suite.cases}
        scored = [r for r in results if not r.details.get("skipped")]

        evidence: dict[Dimension, Evidence] = {}
        for dimension in (Dimension.QUALITY, Dimension.SAFETY):
            cases = [r for r in scored if _case_dimension(by_id[r.case_id]) is dimension]
            if not cases:
                evidence[dimension] = UnmeasuredEvidence(
                    dimension=dimension,
                    reason=f"no scored {dimension.value} cases in suite {suite.name!r}",
                )
            else:
                evidence[dimension] = ProportionEvidence(
                    dimension=dimension,
                    successes=sum(1 for r in cases if r.passed),
                    trials=len(cases),
                )

        evidence[Dimension.RELIABILITY] = self._reliability_evidence(
            suite, agent_fn, results, seed, reliability_runs
        )
        evidence[Dimension.COST] = self._cost_evidence(
            suite, scored, outputs, cost_fn, policy.for_risk(risk).cost_budget
        )
        return decide(evidence, policy, risk)

    def _reliability_evidence(
        self,
        suite: EvalSuite,
        agent_fn: AgentFn,
        first_results: list[EvalResult],
        seed: int,
        reliability_runs: int,
    ) -> Evidence:
        """Per-case pass/fail agreement across seeded runs.

        A case counts as agreeing when every run gives it the same
        pass/fail verdict. Cases skipped in any run are excluded, not
        counted as agreeing: a skipped case carries no reliability signal.
        """
        if reliability_runs < 2:
            return UnmeasuredEvidence(
                dimension=Dimension.RELIABILITY,
                reason=f"reliability needs >= 2 runs, got {reliability_runs}",
            )
        first = {r.case_id: r for r in first_results if not r.details.get("skipped")}
        if not first:
            return UnmeasuredEvidence(
                dimension=Dimension.RELIABILITY,
                reason="no scored cases to compare across runs",
            )
        verdicts: dict[str, list[bool]] = {cid: [r.passed] for cid, r in first.items()}
        for run_index in range(1, reliability_runs):
            run_results, _outputs = self._execute(suite, agent_fn, seed + run_index)
            seen = {r.case_id for r in run_results if not r.details.get("skipped")}
            for cid in list(verdicts):
                if cid not in seen:
                    del verdicts[cid]
                    continue
                verdicts[cid].append(
                    next(r.passed for r in run_results if r.case_id == cid)
                )
        if not verdicts:
            return UnmeasuredEvidence(
                dimension=Dimension.RELIABILITY,
                reason="no case was scored in every run",
            )
        agreeing = sum(1 for passes in verdicts.values() if len(set(passes)) == 1)
        return ProportionEvidence(
            dimension=Dimension.RELIABILITY, successes=agreeing, trials=len(verdicts)
        )

    def _cost_evidence(
        self,
        suite: EvalSuite,
        scored: list[EvalResult],
        outputs: dict[str, str],
        cost_fn: CostFn | None,
        budget: float,
    ) -> Evidence:
        """Mean-cost interval against the budget, or unmeasured."""
        if cost_fn is None:
            return UnmeasuredEvidence(
                dimension=Dimension.COST,
                reason="no cost function supplied",
            )
        costs: list[float] = []
        for result in scored:
            case = next(c for c in suite.cases if c.id == result.case_id)
            cost = cost_fn(case, outputs[result.case_id])
            if cost is None:
                return UnmeasuredEvidence(
                    dimension=Dimension.COST,
                    reason=f"cost not measured for case {result.case_id!r}",
                )
            if cost < 0:
                raise ValueError(f"cost_fn returned negative cost {cost} for case {case.id!r}")
            costs.append(cost)
        if not costs:
            return UnmeasuredEvidence(
                dimension=Dimension.COST,
                reason="no scored cases to price",
            )
        return CostEvidence.from_costs(costs, budget)


def _case_dimension(case: EvalCase) -> Dimension:
    """Map a case's dimension tag to the Dimension enum.

    Validated at the YAML boundary already; this is the second enforcement
    point for programmatically built suites.
    """
    try:
        dimension = Dimension(case.dimension)
    except ValueError:
        raise ValueError(
            f"Invalid case dimension {case.dimension!r} for case {case.id!r}: "
            f"expected one of {sorted(d.value for d in Dimension)}"
        ) from None
    if dimension not in (Dimension.QUALITY, Dimension.SAFETY):
        raise ValueError(
            f"Invalid case dimension {case.dimension!r} for case {case.id!r}: "
            "cases measure quality or safety only"
        )
    return dimension
