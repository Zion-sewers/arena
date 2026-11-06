"""Core eval runner — orchestrates suite execution."""

from __future__ import annotations

import random
import warnings
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
# An agent that consumes the run seed: seeded_agent_fn(prompt, seed).
# Reliability is agreement across repeated runs; a plain AgentFn is
# re-invoked bare, so it can only vary if the agent itself varies across
# calls (e.g. temperature sampling). A seeded agent lets the run seed
# genuinely perturb the agent, which is what makes the reliability runs
# independent samples instead of replays of one execution.
SeededAgentFn = Callable[[str, int], str]
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
        self,
        suite: EvalSuite,
        agent_fn: AgentFn,
        seed: int,
        seeded_agent_fn: SeededAgentFn | None = None,
    ) -> tuple[list[EvalResult], dict[str, str]]:
        """Run one seeded pass, returning (results, outputs by case id).

        A ``random.Random(seed)`` stream is bound to the run and one draw per
        case is recorded in ``details["rng_draw"]``. The deterministic scorer
        needs no randomness today; the draw pins the stream so any future
        stochastic component (subset sampling, tie-breaking) replays
        bit-exactly under the same seed. When ``seeded_agent_fn`` is given,
        the agent itself is called with the run seed, so the seed perturbs
        the agent instead of only the harness.
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

            if seeded_agent_fn is not None:
                agent_output = seeded_agent_fn(case.input, seed)
            else:
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
        """Run in CI mode — returns result with pass_rate for threshold check.

        .. deprecated::
            A point-estimate pass-rate gate with no dimensions, no
            uncertainty, and no unmeasured rule. Superseded by
            ``run_decision``; new gates should use that. Kept so existing
            callers do not change behavior silently.
        """
        warnings.warn(
            "run_ci is deprecated: it gates on a point-estimate pass_rate "
            "with no dimensions or uncertainty. Use run_decision instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.run(suite, agent_fn)
        return result

    def run_decision(
        self,
        suite: EvalSuite,
        agent_fn: AgentFn | None = None,
        policy: DecisionPolicy | None = None,
        risk: RiskLevel = RiskLevel.MEDIUM,
        *,
        cost_fn: CostFn | None = None,
        reliability_runs: int = 2,
        seed: int = DEFAULT_SEED,
        seeded_agent_fn: SeededAgentFn | None = None,
    ) -> Decision:
        """Run a suite and map the evidence to a GATE/ROLLBACK/INVESTIGATE decision.

        Evidence per dimension (see decision.py for the mapping rules):
        * quality / safety: pass proportions over the scored cases tagged
          with that dimension. A dimension with no scored cases is
          unmeasured and forces INVESTIGATE.
        * reliability: per-case pass/fail agreement across ``reliability_runs``
          repeated runs. One run cannot measure agreement (it is 1.0 by
          construction), so ``reliability_runs < 2`` leaves reliability
          unmeasured instead of faking it. Pass ``seeded_agent_fn`` to let
          the run seed perturb the agent; otherwise the runs are bare
          re-invocations and agreement measures stability across calls,
          which is exact for a deterministic agent and samples real
          variation for a stochastic one. A replay agent that re-reads
          one recording cannot vary across runs: callers that replay
          must say so (the CLI labels that reliability replay-agreement)
          rather than present it as live-agent reliability.
        * cost: mean + standard-error interval over ``cost_fn`` observations
          against the policy's per-risk budget. No ``cost_fn`` means cost is
          unmeasured and forces INVESTIGATE.

        Case ids must be unique: the reliability and cost joins are keyed
        by id, and duplicates would silently corrupt them.
        """
        policy = policy or default_policy()
        if reliability_runs < 1:
            raise ValueError("reliability_runs must be >= 1")
        if agent_fn is None and seeded_agent_fn is None:
            raise ValueError("run_decision needs agent_fn or seeded_agent_fn")
        if agent_fn is None:
            if seeded_agent_fn is None:  # pragma: no cover - narrowed above
                raise ValueError("run_decision needs agent_fn or seeded_agent_fn")
            seeded = seeded_agent_fn
            agent_fn = lambda prompt: seeded(prompt, seed)
        ids = [case.id for case in suite.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("run_decision needs unique case ids")

        results, outputs = self._execute(suite, agent_fn, seed, seeded_agent_fn)
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
            suite, agent_fn, results, seed, reliability_runs, seeded_agent_fn
        )
        evidence[Dimension.COST] = self._cost_evidence(
            suite, scored, outputs, cost_fn, by_id
        )
        return decide(evidence, policy, risk)

    def _reliability_evidence(
        self,
        suite: EvalSuite,
        agent_fn: AgentFn,
        first_results: list[EvalResult],
        seed: int,
        reliability_runs: int,
        seeded_agent_fn: SeededAgentFn | None = None,
    ) -> Evidence:
        """Per-case pass/fail agreement across repeated runs.

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
            run_results, _outputs = self._execute(
                suite, agent_fn, seed + run_index, seeded_agent_fn
            )
            seen = {r.case_id: r for r in run_results if not r.details.get("skipped")}
            for cid in list(verdicts):
                if cid not in seen:
                    del verdicts[cid]
                    continue
                verdicts[cid].append(seen[cid].passed)
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
        by_id: dict[str, EvalCase],
    ) -> Evidence:
        """Mean-cost interval against the budget, or unmeasured."""
        if cost_fn is None:
            return UnmeasuredEvidence(
                dimension=Dimension.COST,
                reason="no cost function supplied",
            )
        costs: list[float] = []
        for result in scored:
            case = by_id[result.case_id]
            cost = cost_fn(case, outputs[result.case_id])
            if cost is None:
                return UnmeasuredEvidence(
                    dimension=Dimension.COST,
                    reason=f"cost not measured for case {result.case_id!r}",
                )
            if not isinstance(cost, (int, float)) or isinstance(cost, bool):
                raise TypeError(f"cost_fn returned non-numeric cost {cost!r} for case {case.id!r}")
            if cost < 0:
                raise ValueError(f"cost_fn returned negative cost {cost} for case {case.id!r}")
            costs.append(float(cost))
        if not costs:
            return UnmeasuredEvidence(
                dimension=Dimension.COST,
                reason="no scored cases to price",
            )
        return CostEvidence.from_costs(costs)


def _case_dimension(case: EvalCase) -> Dimension:
    """Map a case's dimension tag to the Dimension enum.

    Validated at construction (EvalCase.__post_init__) and at the YAML
    boundary already; this is the third enforcement point for
    programmatically built suites.
    """
    try:
        dimension = Dimension(case.dimension)
    except ValueError:
        raise ValueError(
            f"Invalid case dimension {case.dimension!r} for case {case.id!r}: "
            "expected one of ['quality', 'safety']"
        ) from None
    if dimension not in (Dimension.QUALITY, Dimension.SAFETY):
        raise ValueError(
            f"Invalid case dimension {case.dimension!r} for case {case.id!r}: "
            "cases measure quality or safety only"
        )
    return dimension
