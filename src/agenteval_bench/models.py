"""Data models for eval suites and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agenteval_bench.decision import CASE_DIMENSIONS


@dataclass
class ExpectedOutput:
    """What the agent output should look like."""

    exact: str | None = None
    contains: list[str] = field(default_factory=list)
    regex: str | None = None
    json_schema: dict[str, Any] | None = None


@dataclass
class RubricCriterion:
    """A single grading criterion."""

    criterion: str
    weight: float = 1.0
    description: str = ""


@dataclass
class CostBound:
    """Token cost limits for an eval run."""

    max_input_tokens: int = 1500
    max_output_tokens: int = 120


@dataclass
class EvalCase:
    """A single test case within an eval suite."""

    id: str
    input: str
    expected: ExpectedOutput = field(default_factory=ExpectedOutput)
    rubric: list[RubricCriterion] = field(default_factory=list)
    skip: bool = False
    # Recorded agent output, for replay suites scored in CI without a live agent.
    output: str | None = None
    # Which decision dimension this case measures ("quality" or "safety").
    # Reliability is a cross-run property and cost comes from a per-case
    # cost function, so neither is a case tag (see decision.CASE_DIMENSIONS).
    dimension: str = "quality"
    # Recorded per-case cost in the policy's cost units (e.g. dollars).
    # None means the cost of this case was not measured.
    cost: float | None = None


def _parse_dimension(case_data: dict[str, Any]) -> str:
    """Validate a case's decision dimension at the YAML boundary.

    Only "quality" and "safety" are case tags. A typo must fail loudly
    here, not silently score as quality downstream.
    """
    dimension = case_data.get("dimension", "quality")
    if not isinstance(dimension, str) or dimension not in CASE_DIMENSIONS:
        raise ValueError(
            f"Invalid case dimension {dimension!r} for case "
            f"{case_data.get('id')!r}: expected one of {sorted(CASE_DIMENSIONS)}"
        )
    return dimension


def _parse_cost(case_data: dict[str, Any]) -> float | None:
    """Validate a case's recorded cost at the YAML boundary."""
    cost = case_data.get("cost")
    if cost is None:
        return None
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        raise TypeError(
            f"Invalid cost {cost!r} for case {case_data.get('id')!r}: expected a number"
        )
    if cost < 0:
        raise ValueError(
            f"Invalid cost {cost!r} for case {case_data.get('id')!r}: expected >= 0"
        )
    return float(cost)


@dataclass
class EvalSuite:
    """A collection of eval cases loaded from YAML."""

    name: str
    version: int = 1
    cases: list[EvalCase] = field(default_factory=list)
    cost_bound: CostBound = field(default_factory=CostBound)
    @classmethod
    def from_yaml(cls, path: str) -> EvalSuite:
        """Load an eval suite from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise TypeError(f"Invalid eval suite: expected mapping, got {type(data).__name__}")

        cases = []
        for case_data in data.get("cases", []):
            expected_data = case_data.get("expected", {})
            expected = ExpectedOutput(
                exact=expected_data.get("exact"),
                contains=expected_data.get("contains", []),
                regex=expected_data.get("regex"),
                json_schema=expected_data.get("json_schema"),
            )
            rubric = [
                RubricCriterion(
                    criterion=r["criterion"],
                    weight=r.get("weight", 1.0),
                    description=r.get("description", ""),
                )
                for r in case_data.get("rubric", [])
            ]
            cases.append(
                EvalCase(
                    id=case_data["id"],
                    input=case_data["input"],
                    expected=expected,
                    rubric=rubric,
                    skip=case_data.get("skip", False),
                    output=case_data.get("output"),
                    dimension=_parse_dimension(case_data),
                    cost=_parse_cost(case_data),
                )
            )

        cost_data = data.get("cost_bound", {})
        cost_bound = CostBound(
            max_input_tokens=cost_data.get("max_input_tokens", 1500),
            max_output_tokens=cost_data.get("max_output_tokens", 120),
        )

        return cls(
            name=data["name"],
            version=data.get("version", 1),
            cases=cases,
            cost_bound=cost_bound,
        )


@dataclass
class EvalResult:
    """Score for a single eval case."""

    case_id: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Aggregate result for an entire eval run."""

    suite_name: str
    results: list[EvalResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    pass_rate: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        return (
            f"Suite: {self.suite_name}\n"
            f"Total: {self.total} | Passed: {self.passed} | "
            f"Failed: {self.failed} | Skipped: {self.skipped}\n"
            f"Pass rate: {self.pass_rate:.1%}"
        )
