"""Trace-mined benchmarks: production traces -> living benchmark + scheduled regression.

Arena issue #41 (AIGUIDE book ch.9, catalog G.40): build benchmarks from
production traces — repeated tool calls, misrouting, step-limit loops,
schema violations, latency/cost spikes, frequent escalations, high
variance — with an owner and a cadence per benchmark, feeding regression.

What this module is
-------------------
A **generator** plus a **scheduled regression runner**:

1. :func:`mine_traces` consumes a trace corpus (JSONL of
   :class:`TraceRecord`, loaded by :func:`load_corpus`) and emits
   :class:`MinedCase` values. Every mined case is traceable to the incident
   it came from: its id is ``mined:<pattern>:<trace_id>`` and it carries the
   incident id, the detected pattern, and the evidence excerpt. A case
   without provenance cannot be constructed — :class:`MinedCase` refuses
   empty provenance fields, and :func:`build_manifest` refuses anything that
   is not a :class:`MinedCase`. A hand-written test set is therefore not a
   valid input to the benchmark; the benchmark is whatever the miner last
   produced from the corpus, and re-mining a changed corpus changes the
   benchmark (the living property, pinned by test).
2. :func:`run_scheduled` mines a corpus, publishes the newly mined cases as
   a new frozen snapshot version (mined cases enter with a required score of
   1.0 — the production failure must be fixed), replays a candidate against
   it through the existing :func:`compare_against_frozen` gate, and
   dispatches :class:`RegressionAction` values (gate / rollback /
   investigation) written to a machine-readable action ledger. The runner
   exits non-zero on gate failure, so a scheduler consumes the verdict.

Real-incidents boundary (read this before claiming otherwise)
-------------------------------------------------------------
This repo owns no production system and no trace corpus, so no code here can
conjure real production incidents. The generator takes the corpus as input:
an operator points it at their production trace export and every case it
emits is traceable to a real incident *in that corpus* by construction. The
checked-in corpus under ``examples/traces/`` is a recorded-format SAMPLE
used by the tests and the demo — it is labeled as such and is not a real
incident. The live-corpus seam is the corpus path itself (same shape as the
flywheel's injected ``GhCliTransport``: pure parsing here, transport
outside). The demo never claims its fixtures are real incidents.

Spec NG4 (no trace replay / observability) is respected: traces are INPUT
data only. Mining emits ordinary :class:`EvalCase` values scored by the
existing deterministic scorer; this module builds no trajectory replay and
no observability surface.

Cadence is a closed set (``daily`` / ``weekly`` / ``monthly``). A benchmark
whose last mine is older than its cadence is stale
(:meth:`BenchmarkManifest.is_stale`); staleness is a report field, not a
silent default.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from agenteval_bench.engine import DEFAULT_SEED
from agenteval_bench.models import CostBound, EvalCase, EvalSuite, ExpectedOutput, RubricCriterion
from agenteval_bench.regression import (
    ComparisonReport,
    FailureKind,
    PromotionRecord,
    compare_against_frozen,
)
from agenteval_bench.snapshots import (
    BaselineRecord,
    BenchmarkVersion,
    SnapshotError,
    SnapshotStore,
    VersionMeta,
)

__all__ = [
    "ActionRecord",
    "BenchmarkManifest",
    "Cadence",
    "CorpusLoad",
    "MineReport",
    "MineThresholds",
    "MinedCase",
    "RegressionAction",
    "ScheduledRunResult",
    "ToolCall",
    "TraceOutcome",
    "TracePattern",
    "TraceRecord",
    "TracebenchError",
    "build_manifest",
    "dispatch_actions",
    "load_corpus",
    "mine_traces",
    "parse_cadence",
    "pattern_to_failure_kind",
    "publish_mined_version",
    "required_baseline",
    "run_scheduled",
]

TraceOutcome = Literal["success", "failure", "escalated"]

TracePattern = Literal[
    "repeated_tool_calls",
    "misrouting",
    "step_limit_loop",
    "schema_violation",
    "latency_spike",
    "cost_spike",
    "frequent_escalation",
    "high_variance",
]

_PATTERNS: tuple[str, ...] = (
    "repeated_tool_calls",
    "misrouting",
    "step_limit_loop",
    "schema_violation",
    "latency_spike",
    "cost_spike",
    "frequent_escalation",
    "high_variance",
)

Cadence = Literal["daily", "weekly", "monthly"]

CADENCE_DAYS: Mapping[str, int] = MappingProxyType(
    {"daily": 1, "weekly": 7, "monthly": 30}
)

RegressionAction = Literal["gate", "rollback", "investigation"]

#: Consecutive gate failures at which a failed run also triggers rollback.
#: 2 prior failures + this failure = the 3rd consecutive failure.
ROLLBACK_AFTER_FAILURES = 2

#: Minimum traces per agent for the high-variance detector: a CV over fewer
#: than 3 points is noise, not a signal.
_VARIANCE_MIN_GROUP = 3


class TracebenchError(ValueError):
    """Fail-closed error for trace-mining / scheduled-regression misuse."""


# ---------------------------------------------------------------------------
# Strict field readers (boundaries narrow Mapping[str, object] per field;
# no unvalidated data flows into the logic).
# ---------------------------------------------------------------------------


def _require_str(d: Mapping[str, object], key: str, where: str) -> str:
    value = d.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: {key!r} must be a non-empty string")
    return value


def _optional_str(d: Mapping[str, object], key: str, where: str) -> str | None:
    value = d.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{where}: {key!r} must be a string or null")
    return value


def _require_int(d: Mapping[str, object], key: str, where: str) -> int:
    value = d.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}: {key!r} must be an integer")
    return value


def _require_number(d: Mapping[str, object], key: str, where: str) -> float:
    value = d.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where}: {key!r} must be a number")
    return float(value)


def _parse_instant(value: str, where: str) -> datetime:
    """Parse an ISO-8601 timestamp; naive timestamps are refused."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as e:
        raise ValueError(f"{where}: {value!r} is not ISO-8601: {e}") from e
    if parsed.tzinfo is None:
        raise ValueError(f"{where}: timestamp {value!r} must carry a timezone")
    return parsed


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(doc: dict[str, Any]) -> str:
    return json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Trace records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation inside a trace.

    ``expected`` is the tool the planner intended, when the trace records
    it. Misrouting is detectable only where the trace records an intent:
    a call whose name differs from its recorded expectation. Where no
    expectation is recorded the miner refuses to guess.
    """

    name: str
    expected: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolCall name must be non-empty")
        if self.expected is not None and not self.expected:
            raise ValueError("ToolCall expected must be non-empty when present")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "expected": self.expected}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> ToolCall:
        if not isinstance(d, Mapping):
            raise TypeError(f"tool call must be a mapping, got {type(d).__name__}")
        return cls(
            name=_require_str(d, "name", "tool call"),
            expected=_optional_str(d, "expected", "tool call"),
        )


@dataclass(frozen=True)
class TraceRecord:
    """One production trace, as exported by the operator's tracing system.

    ``expected_output`` is the oracle: what the agent should have produced
    for this trace's situation. A benchmark case without an expected output
    cannot be scored by the deterministic scorer (no matcher set means
    failure), so a trace without an oracle is evidence, not a benchmark
    case — the miner refuses it at the boundary instead of inventing one.
    """

    trace_id: str
    incident_id: str
    agent: str
    tool_calls: tuple[ToolCall, ...]
    steps: int
    step_limit: int
    duration_ms: float
    cost_usd: float
    outcome: TraceOutcome
    schema_violations: int
    expected_output: str
    error: str = ""

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id must be non-empty")
        if not self.incident_id:
            raise ValueError("incident_id must be non-empty")
        if not self.agent:
            raise ValueError("agent must be non-empty")
        if self.steps < 0:
            raise ValueError(f"steps must be >= 0, got {self.steps}")
        if self.step_limit < 1:
            raise ValueError(f"step_limit must be >= 1, got {self.step_limit}")
        if self.duration_ms < 0:
            raise ValueError(f"duration_ms must be >= 0, got {self.duration_ms}")
        if self.cost_usd < 0:
            raise ValueError(f"cost_usd must be >= 0, got {self.cost_usd}")
        if self.outcome not in ("success", "failure", "escalated"):
            raise ValueError(f"outcome must be success/failure/escalated, got {self.outcome!r}")
        if self.schema_violations < 0:
            raise ValueError(
                f"schema_violations must be >= 0, got {self.schema_violations}"
            )
        if not self.expected_output:
            raise ValueError("expected_output must be non-empty (the case oracle)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "incident_id": self.incident_id,
            "agent": self.agent,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "steps": self.steps,
            "step_limit": self.step_limit,
            "duration_ms": self.duration_ms,
            "cost_usd": self.cost_usd,
            "outcome": self.outcome,
            "schema_violations": self.schema_violations,
            "expected_output": self.expected_output,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> TraceRecord:
        if not isinstance(d, Mapping):
            raise TypeError(f"trace record must be a mapping, got {type(d).__name__}")
        where = "trace record"
        calls_raw = d.get("tool_calls", [])
        if not isinstance(calls_raw, list):
            raise TypeError(f"{where}: tool_calls must be a list")
        calls: list[ToolCall] = []
        for raw in calls_raw:
            if not isinstance(raw, Mapping):
                raise TypeError(f"{where}: tool call entries must be mappings")
            calls.append(ToolCall.from_dict(raw))
        outcome = _require_str(d, "outcome", where)
        if outcome not in ("success", "failure", "escalated"):
            raise ValueError(
                f"{where}: outcome must be success/failure/escalated, got {outcome!r}"
            )
        return cls(
            trace_id=_require_str(d, "trace_id", where),
            incident_id=_require_str(d, "incident_id", where),
            agent=_require_str(d, "agent", where),
            tool_calls=tuple(calls),
            steps=_require_int(d, "steps", where),
            step_limit=_require_int(d, "step_limit", where),
            duration_ms=_require_number(d, "duration_ms", where),
            cost_usd=_require_number(d, "cost_usd", where),
            outcome=outcome,  # type: ignore[arg-type]
            schema_violations=_require_int(d, "schema_violations", where),
            expected_output=_require_str(d, "expected_output", where),
            error=_optional_str(d, "error", where) or "",
        )


@dataclass(frozen=True)
class CorpusLoad:
    """Result of loading a trace corpus file (JSONL, one record per line)."""

    records: tuple[TraceRecord, ...]
    skipped: int
    total_lines: int
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": len(self.records),
            "skipped": self.skipped,
            "total_lines": self.total_lines,
            "digest": self.digest,
        }


def load_corpus(path: str | Path) -> CorpusLoad:
    """Load a JSONL trace corpus.

    Malformed lines (not JSON, not a mapping, failing
    :meth:`TraceRecord.from_dict`) are SKIPPED and counted, never guessed
    at — the count rides in the mine report, so a silently shrinking corpus
    is visible. A missing/unreadable file fails closed.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise TracebenchError(f"cannot read corpus {path}: {e}") from e
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise TracebenchError(f"corpus {path} is not valid UTF-8: {e}") from e
    records: list[TraceRecord] = []
    skipped = 0
    total = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(parsed, dict):
            skipped += 1
            continue
        try:
            records.append(TraceRecord.from_dict(parsed))
        except (ValueError, TypeError):
            skipped += 1
    return CorpusLoad(
        records=tuple(records), skipped=skipped, total_lines=total, digest=digest
    )


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------


def parse_cadence(value: str) -> Cadence:
    """Parse a cadence name; anything outside the closed set is refused."""
    if value not in CADENCE_DAYS:
        raise ValueError(
            f"cadence must be one of {sorted(CADENCE_DAYS)}, got {value!r}"
        )
    return value  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Mining thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MineThresholds:
    """Named detector thresholds. Defaults are documented starting points,
    recorded in every mine report — never hidden magic numbers."""

    repeat_tool_calls: int = 3
    latency_ms: float = 30_000.0
    cost_usd: float = 1.0
    escalations_per_agent: int = 3
    variance_cv: float = 0.5

    def __post_init__(self) -> None:
        if self.repeat_tool_calls < 2:
            raise ValueError("repeat_tool_calls must be >= 2")
        if self.latency_ms <= 0:
            raise ValueError("latency_ms must be > 0")
        if self.cost_usd <= 0:
            raise ValueError("cost_usd must be > 0")
        if self.escalations_per_agent < 2:
            raise ValueError("escalations_per_agent must be >= 2")
        if self.variance_cv <= 0:
            raise ValueError("variance_cv must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "repeat_tool_calls": self.repeat_tool_calls,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "escalations_per_agent": self.escalations_per_agent,
            "variance_cv": self.variance_cv,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> MineThresholds:
        if not isinstance(d, Mapping):
            raise TypeError(f"thresholds must be a mapping, got {type(d).__name__}")
        return cls(
            repeat_tool_calls=_require_int(d, "repeat_tool_calls", "thresholds"),
            latency_ms=_require_number(d, "latency_ms", "thresholds"),
            cost_usd=_require_number(d, "cost_usd", "thresholds"),
            escalations_per_agent=_require_int(
                d, "escalations_per_agent", "thresholds"
            ),
            variance_cv=_require_number(d, "variance_cv", "thresholds"),
        )


# ---------------------------------------------------------------------------
# Pattern detectors (pure functions; each returns the evidence excerpt or None)
# ---------------------------------------------------------------------------


def _most_repeated_tool(record: TraceRecord) -> tuple[str, int]:
    counts = Counter(c.name for c in record.tool_calls)
    if not counts:
        return "", 0
    name, count = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return name, count


def detect_repeated_tool_calls(
    record: TraceRecord, thresholds: MineThresholds
) -> str | None:
    name, count = _most_repeated_tool(record)
    if count >= thresholds.repeat_tool_calls:
        return (
            f"tool '{name}' called {count} times in one trace "
            f"(threshold {thresholds.repeat_tool_calls})"
        )
    return None


def detect_misrouting(record: TraceRecord) -> str | None:
    for i, call in enumerate(record.tool_calls):
        if call.expected is not None and call.name != call.expected:
            return (
                f"call {i + 1}: routed to '{call.name}', "
                f"expected '{call.expected}'"
            )
    return None


def detect_step_limit_loop(
    record: TraceRecord, thresholds: MineThresholds
) -> str | None:
    """Hit the step limit while repeating a tool: the agent looped until
    cut off. Hitting the limit with all-distinct tools is a long task, not
    a loop — no case."""
    if record.steps < record.step_limit:
        return None
    name, count = _most_repeated_tool(record)
    if count < 2:
        return None
    return (
        f"steps {record.steps} reached limit {record.step_limit} with "
        f"repeated tool '{name}' ({count}x)"
    )


def detect_schema_violation(record: TraceRecord) -> str | None:
    if record.schema_violations >= 1:
        return f"{record.schema_violations} schema violation(s)"
    return None


def detect_latency_spike(
    record: TraceRecord, thresholds: MineThresholds
) -> str | None:
    if record.duration_ms >= thresholds.latency_ms:
        return (
            f"duration {record.duration_ms:g}ms >= "
            f"threshold {thresholds.latency_ms:g}ms"
        )
    return None


def detect_cost_spike(record: TraceRecord, thresholds: MineThresholds) -> str | None:
    if record.cost_usd >= thresholds.cost_usd:
        return f"cost ${record.cost_usd:g} >= threshold ${thresholds.cost_usd:g}"
    return None


def detect_frequent_escalation(
    records: Sequence[TraceRecord], thresholds: MineThresholds
) -> list[tuple[TraceRecord, str]]:
    """Corpus-level: an agent that escalated at least N times. One case per
    escalated trace in the group, so each case stays anchored on its own
    incident."""
    by_agent: dict[str, list[TraceRecord]] = {}
    for record in records:
        by_agent.setdefault(record.agent, []).append(record)
    found: list[tuple[TraceRecord, str]] = []
    for agent in sorted(by_agent):
        escalated = [r for r in by_agent[agent] if r.outcome == "escalated"]
        if len(escalated) < thresholds.escalations_per_agent:
            continue
        evidence = (
            f"agent '{agent}' escalated {len(escalated)} times in this corpus "
            f"(threshold {thresholds.escalations_per_agent})"
        )
        for record in sorted(escalated, key=lambda r: r.trace_id):
            found.append((record, evidence))
    return found


def detect_high_variance(
    records: Sequence[TraceRecord], thresholds: MineThresholds
) -> list[tuple[TraceRecord, str]]:
    """Corpus-level: an agent whose trace durations vary widely (coefficient
    of variation >= threshold over >= 3 traces). One case per agent group,
    anchored on the slowest trace; the evidence names every trace in the
    group so the case is auditable back to the corpus."""
    by_agent: dict[str, list[TraceRecord]] = {}
    for record in records:
        by_agent.setdefault(record.agent, []).append(record)
    found: list[tuple[TraceRecord, str]] = []
    for agent in sorted(by_agent):
        group = by_agent[agent]
        if len(group) < _VARIANCE_MIN_GROUP:
            continue
        durations = [r.duration_ms for r in group]
        mean = statistics.fmean(durations)
        if mean <= 0:
            continue
        cv = statistics.stdev(durations) / mean
        if cv < thresholds.variance_cv:
            continue
        anchor = min(group, key=lambda r: (-r.duration_ms, r.trace_id))
        ids = ", ".join(sorted(r.trace_id for r in group))
        evidence = (
            f"agent '{agent}' duration CV {cv:.2f} over {len(group)} traces "
            f"(threshold {thresholds.variance_cv:.2f}); slowest trace "
            f"{anchor.trace_id} at {anchor.duration_ms:g}ms; traces: {ids}"
        )
        found.append((anchor, evidence))
    return found


# ---------------------------------------------------------------------------
# Mined cases
# ---------------------------------------------------------------------------


def pattern_to_failure_kind(pattern: TracePattern) -> FailureKind:
    """Coarse legacy bucket for the promotion record. The full pattern
    survives in the record's provenance mapping — the kind is a summary,
    never the whole story."""
    mapping: dict[TracePattern, FailureKind] = {
        "repeated_tool_calls": "retry_trace",
        "step_limit_loop": "retry_trace",
        "misrouting": "failed_tool_call",
        "schema_violation": "failed_tool_call",
        "latency_spike": "correction_pattern",
        "cost_spike": "correction_pattern",
        "frequent_escalation": "escalation",
        "high_variance": "correction_pattern",
    }
    return mapping[pattern]


def _render_trace_input(record: TraceRecord) -> str:
    tools = " -> ".join(c.name for c in record.tool_calls) or "(none)"
    lines = [
        f"agent: {record.agent}",
        f"outcome: {record.outcome}",
        f"tools: {tools}",
        f"steps: {record.steps}/{record.step_limit}",
        f"duration_ms: {record.duration_ms:g}",
        f"cost_usd: {record.cost_usd:g}",
    ]
    if record.error:
        lines.append(f"error: {record.error}")
    return "\n".join(lines)


@dataclass(frozen=True)
class MinedCase:
    """A benchmark case mined from one trace, with its provenance attached.

    The case is an ordinary :class:`EvalCase` (scored by the existing
    deterministic scorer); the provenance fields are what make it a *living*
    benchmark case instead of a hand-written one. Construction refuses
    empty provenance, owner, or cadence.
    """

    case: EvalCase
    trace_id: str
    incident_id: str
    pattern: TracePattern
    evidence: str
    outcome: TraceOutcome
    expected_output: str
    owner: str
    cadence: Cadence
    mined_at: str

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("MinedCase trace_id must be non-empty")
        if not self.incident_id:
            raise ValueError("MinedCase incident_id must be non-empty")
        if self.pattern not in _PATTERNS:
            raise ValueError(f"unknown trace pattern {self.pattern!r}")
        if not self.evidence:
            raise ValueError("MinedCase evidence must be non-empty")
        if not self.expected_output:
            raise ValueError("MinedCase expected_output must be non-empty")
        if not self.owner:
            raise ValueError("MinedCase owner must be non-empty")
        _parse_instant(self.mined_at, "MinedCase.mined_at")

    @property
    def prior_score(self) -> float:
        """Score the production behavior earned: a failed or escalated trace
        scored 0.0; a successful trace (spike / variance patterns) scored
        1.0 but still produced a case, because the pattern is the defect."""
        return 0.0 if self.outcome in ("failure", "escalated") else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": {
                "id": self.case.id,
                "input": self.case.input,
                "expected": {
                    "exact": self.case.expected.exact,
                    "contains": list(self.case.expected.contains),
                    "regex": self.case.expected.regex,
                    "json_schema": self.case.expected.json_schema,
                },
                "rubric": [
                    {
                        "criterion": r.criterion,
                        "weight": r.weight,
                        "description": r.description,
                    }
                    for r in self.case.rubric
                ],
                "skip": self.case.skip,
                "output": self.case.output,
            },
            "trace_id": self.trace_id,
            "incident_id": self.incident_id,
            "pattern": self.pattern,
            "evidence": self.evidence,
            "outcome": self.outcome,
            "expected_output": self.expected_output,
            "owner": self.owner,
            "cadence": self.cadence,
            "mined_at": self.mined_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> MinedCase:
        if not isinstance(d, Mapping):
            raise TypeError(f"mined case must be a mapping, got {type(d).__name__}")
        case_raw = d.get("case")
        if not isinstance(case_raw, Mapping):
            raise TypeError("mined case: 'case' must be a mapping")
        expected_raw = case_raw.get("expected")
        if not isinstance(expected_raw, Mapping):
            raise TypeError("mined case: 'case.expected' must be a mapping")
        rubric: list[RubricCriterion] = []
        rubric_raw = case_raw.get("rubric", [])
        if not isinstance(rubric_raw, list):
            raise TypeError("mined case: 'case.rubric' must be a list")
        for entry in rubric_raw:
            if not isinstance(entry, Mapping):
                raise TypeError("mined case: rubric entries must be mappings")
            rubric.append(
                RubricCriterion(
                    criterion=_require_str(entry, "criterion", "mined case rubric"),
                    weight=_require_number(entry, "weight", "mined case rubric")
                    if "weight" in entry
                    else 1.0,
                    description=_optional_str(entry, "description", "mined case rubric")
                    or "",
                )
            )
        pattern = _require_str(d, "pattern", "mined case")
        if pattern not in _PATTERNS:
            raise ValueError(f"mined case: unknown trace pattern {pattern!r}")
        outcome = _require_str(d, "outcome", "mined case")
        if outcome not in ("success", "failure", "escalated"):
            raise ValueError(f"mined case: bad outcome {outcome!r}")
        contains_raw = expected_raw.get("contains", [])
        if not isinstance(contains_raw, list) or not all(
            isinstance(x, str) for x in contains_raw
        ):
            raise TypeError(
                "mined case: 'case.expected.contains' must be a list of strings"
            )
        return cls(
            case=EvalCase(
                id=_require_str(case_raw, "id", "mined case"),
                input=_require_str(case_raw, "input", "mined case"),
                expected=ExpectedOutput(
                    exact=_optional_str(expected_raw, "exact", "mined case expected"),
                    contains=list(contains_raw),
                    regex=_optional_str(expected_raw, "regex", "mined case expected"),
                    json_schema=(
                        dict(expected_raw["json_schema"])
                        if isinstance(expected_raw.get("json_schema"), dict)
                        else None
                    ),
                ),
                rubric=rubric,
                skip=bool(case_raw.get("skip", False)),
                output=_optional_str(case_raw, "output", "mined case"),
            ),
            trace_id=_require_str(d, "trace_id", "mined case"),
            incident_id=_require_str(d, "incident_id", "mined case"),
            pattern=pattern,  # type: ignore[arg-type]
            evidence=_require_str(d, "evidence", "mined case"),
            outcome=outcome,  # type: ignore[arg-type]
            expected_output=_require_str(d, "expected_output", "mined case"),
            owner=_require_str(d, "owner", "mined case"),
            cadence=parse_cadence(_require_str(d, "cadence", "mined case")),
            mined_at=_require_str(d, "mined_at", "mined case"),
        )


def _build_mined_case(
    record: TraceRecord,
    pattern: TracePattern,
    evidence: str,
    *,
    owner: str,
    cadence: Cadence,
    mined_at: str,
) -> MinedCase:
    case_id = f"mined:{pattern}:{record.trace_id}"
    case = EvalCase(
        id=case_id,
        input=_render_trace_input(record),
        expected=ExpectedOutput(exact=record.expected_output),
        rubric=[
            RubricCriterion(
                criterion="trace-provenance",
                description=(
                    f"{pattern} in incident {record.incident_id} "
                    f"(trace {record.trace_id}): {evidence}"
                ),
            )
        ],
    )
    return MinedCase(
        case=case,
        trace_id=record.trace_id,
        incident_id=record.incident_id,
        pattern=pattern,
        evidence=evidence,
        outcome=record.outcome,
        expected_output=record.expected_output,
        owner=owner,
        cadence=cadence,
        mined_at=mined_at,
    )


@dataclass(frozen=True)
class MineReport:
    """What one mining run produced. Canonical JSON (sorted keys) so a
    re-mine of an unchanged corpus is byte-identical — the determinism the
    living-benchmark claim rests on."""

    cases: tuple[MinedCase, ...]
    thresholds: MineThresholds
    corpus_digest: str
    mined_at: str
    owner: str
    cadence: Cadence
    skipped_records: int
    source_records: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "agenteval-bench",
            "kind": "tracebench-mine",
            "cases": [c.to_dict() for c in self.cases],
            "thresholds": self.thresholds.to_dict(),
            "corpus_digest": self.corpus_digest,
            "mined_at": self.mined_at,
            "owner": self.owner,
            "cadence": self.cadence,
            "skipped_records": self.skipped_records,
            "source_records": self.source_records,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> MineReport:
        if not isinstance(d, Mapping):
            raise TypeError(f"mine report must be a mapping, got {type(d).__name__}")
        cases_raw = d.get("cases", [])
        if not isinstance(cases_raw, list):
            raise TypeError("mine report: cases must be a list")
        cases = [MinedCase.from_dict(c) for c in cases_raw]
        thresholds_raw = d.get("thresholds")
        if not isinstance(thresholds_raw, Mapping):
            raise TypeError("mine report: thresholds must be a mapping")
        return cls(
            cases=tuple(cases),
            thresholds=MineThresholds.from_dict(thresholds_raw),
            corpus_digest=_require_str(d, "corpus_digest", "mine report"),
            mined_at=_require_str(d, "mined_at", "mine report"),
            owner=_require_str(d, "owner", "mine report"),
            cadence=parse_cadence(_require_str(d, "cadence", "mine report")),
            skipped_records=_require_int(d, "skipped_records", "mine report"),
            source_records=_require_int(d, "source_records", "mine report"),
        )


def mine_traces(
    records: Sequence[TraceRecord],
    *,
    owner: str,
    cadence: str,
    mined_at: str | None = None,
    thresholds: MineThresholds | None = None,
    corpus_digest: str = "",
    skipped_records: int = 0,
) -> MineReport:
    """Mine benchmark cases from production traces.

    Pure and deterministic: same records + thresholds -> same cases in the
    same order. Duplicate trace ids fail closed (the corpus is corrupt, and
    silently dropping a trace would lose an incident). Per-trace patterns
    emit one case per (trace, pattern); corpus-level patterns
    (frequent escalation, high variance) emit one case per group or per
    escalated trace as documented on the detectors.
    """
    if not owner:
        raise ValueError("mine_traces: owner must be non-empty")
    parsed_cadence = parse_cadence(cadence)
    instant = _parse_instant(mined_at or _utcnow_iso(), "mine_traces.mined_at")
    mined_at = instant.isoformat()
    thresholds = thresholds or MineThresholds()
    seen: set[str] = set()
    for record in records:
        if record.trace_id in seen:
            raise ValueError(
                f"duplicate trace_id {record.trace_id!r} in corpus — refusing to mine"
            )
        seen.add(record.trace_id)

    mined: list[MinedCase] = []
    ordered = sorted(records, key=lambda r: r.trace_id)
    for record in ordered:
        for pattern, evidence in (
            ("repeated_tool_calls", detect_repeated_tool_calls(record, thresholds)),
            ("misrouting", detect_misrouting(record)),
            ("step_limit_loop", detect_step_limit_loop(record, thresholds)),
            ("schema_violation", detect_schema_violation(record)),
            ("latency_spike", detect_latency_spike(record, thresholds)),
            ("cost_spike", detect_cost_spike(record, thresholds)),
        ):
            if evidence is not None:
                mined.append(
                    _build_mined_case(
                        record,
                        pattern,  # type: ignore[arg-type]
                        evidence,
                        owner=owner,
                        cadence=parsed_cadence,
                        mined_at=mined_at,
                    )
                )
    for record, evidence in detect_frequent_escalation(records, thresholds):
        mined.append(
            _build_mined_case(
                record,
                "frequent_escalation",
                evidence,
                owner=owner,
                cadence=parsed_cadence,
                mined_at=mined_at,
            )
        )
    for record, evidence in detect_high_variance(records, thresholds):
        mined.append(
            _build_mined_case(
                record,
                "high_variance",
                evidence,
                owner=owner,
                cadence=parsed_cadence,
                mined_at=mined_at,
            )
        )
    mined.sort(key=lambda m: m.case.id)
    return MineReport(
        cases=tuple(mined),
        thresholds=thresholds,
        corpus_digest=corpus_digest,
        mined_at=mined_at,
        owner=owner,
        cadence=parsed_cadence,
        skipped_records=skipped_records,
        source_records=len(records),
    )


# ---------------------------------------------------------------------------
# Benchmark manifest (owner + cadence per benchmark, staleness)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkManifest:
    """One benchmark's identity: which cases it holds, where they came
    from, who owns it, and how often it must be re-mined."""

    benchmark_id: str
    owner: str
    cadence: Cadence
    case_ids: tuple[str, ...]
    source_incidents: tuple[str, ...]
    created_at: str
    last_mined_at: str

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be non-empty")
        if not self.owner:
            raise ValueError("owner must be non-empty")
        _parse_instant(self.created_at, "BenchmarkManifest.created_at")
        _parse_instant(self.last_mined_at, "BenchmarkManifest.last_mined_at")

    def is_stale(self, now: datetime) -> bool:
        """True when the last mine is older than the cadence. A naive
        ``now`` is refused — staleness must never be computed on an
        ambiguous clock."""
        if now.tzinfo is None:
            raise ValueError("is_stale: 'now' must carry a timezone")
        last = _parse_instant(self.last_mined_at, "BenchmarkManifest.last_mined_at")
        return (now - last) > timedelta(days=CADENCE_DAYS[self.cadence])

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "agenteval-bench",
            "kind": "tracebench-manifest",
            "benchmark_id": self.benchmark_id,
            "owner": self.owner,
            "cadence": self.cadence,
            "case_ids": list(self.case_ids),
            "source_incidents": list(self.source_incidents),
            "created_at": self.created_at,
            "last_mined_at": self.last_mined_at,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> BenchmarkManifest:
        if not isinstance(d, Mapping):
            raise TypeError(f"manifest must be a mapping, got {type(d).__name__}")
        case_ids = d.get("case_ids", [])
        incidents = d.get("source_incidents", [])
        if not isinstance(case_ids, list) or not all(
            isinstance(x, str) for x in case_ids
        ):
            raise TypeError("manifest: case_ids must be a list of strings")
        if not isinstance(incidents, list) or not all(
            isinstance(x, str) for x in incidents
        ):
            raise TypeError("manifest: source_incidents must be a list of strings")
        return cls(
            benchmark_id=_require_str(d, "benchmark_id", "manifest"),
            owner=_require_str(d, "owner", "manifest"),
            cadence=parse_cadence(_require_str(d, "cadence", "manifest")),
            case_ids=tuple(case_ids),
            source_incidents=tuple(incidents),
            created_at=_require_str(d, "created_at", "manifest"),
            last_mined_at=_require_str(d, "last_mined_at", "manifest"),
        )


def build_manifest(
    cases: Sequence[MinedCase],
    *,
    benchmark_id: str,
    owner: str,
    cadence: str,
    created_at: str | None = None,
    last_mined_at: str | None = None,
) -> BenchmarkManifest:
    """Build a benchmark manifest from mined cases.

    The runtime guard is the point: only :class:`MinedCase` values — which
    carry provenance by construction — may enter a manifest. A hand-written
    :class:`EvalCase` (or anything else) is refused with ``TypeError``.
    This is what makes the benchmark living rather than a static set.
    """
    parsed_cadence = parse_cadence(cadence)
    if not owner:
        raise ValueError("build_manifest: owner must be non-empty")
    for case in cases:
        if not isinstance(case, MinedCase):
            raise TypeError(
                "refusing to build a manifest from a non-mined case "
                f"({type(case).__name__}): only trace-mined cases with "
                "provenance may enter a benchmark"
            )
    now = _utcnow_iso()
    return BenchmarkManifest(
        benchmark_id=benchmark_id,
        owner=owner,
        cadence=parsed_cadence,
        case_ids=tuple(sorted(c.case.id for c in cases)),
        source_incidents=tuple(sorted({c.incident_id for c in cases})),
        created_at=created_at or now,
        last_mined_at=last_mined_at or now,
    )


# ---------------------------------------------------------------------------
# Regression actions (gate / rollback / investigation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionRecord:
    """One dispatched regression action: machine-readable, so the runner,
    the scheduler, and the workflow all consume the same artifact. An
    action is a record in the ledger, not a print statement."""

    action: RegressionAction
    benchmark_id: str
    run_id: str
    case_ids: tuple[str, ...]
    reason: str
    recorded_at: str

    def __post_init__(self) -> None:
        if self.action not in ("gate", "rollback", "investigation"):
            raise ValueError(f"unknown regression action {self.action!r}")
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be non-empty")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.reason:
            raise ValueError("reason must be non-empty")
        _parse_instant(self.recorded_at, "ActionRecord.recorded_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "benchmark_id": self.benchmark_id,
            "run_id": self.run_id,
            "case_ids": list(self.case_ids),
            "reason": self.reason,
            "recorded_at": self.recorded_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)


def dispatch_actions(
    report: ComparisonReport,
    *,
    benchmark_id: str,
    run_id: str,
    prior_consecutive_failures: int = 0,
    recorded_at: str | None = None,
) -> tuple[ActionRecord, ...]:
    """Map a gate verdict onto the actions it must trigger.

    - GATE_PASS: no actions.
    - GATE_FAIL: ``gate`` (block the release) + ``investigation`` (name the
      regressed cases, or the aggregate reason when no single case
      regressed).
    - GATE_FAIL with at least :data:`ROLLBACK_AFTER_FAILURES` prior
      consecutive failures: also ``rollback`` — this is the 3rd consecutive
      failure, so the benchmark itself demands the release be rolled back,
      not merely blocked.
    """
    if prior_consecutive_failures < 0:
        raise ValueError("prior_consecutive_failures must be >= 0")
    if report.verdict == "GATE_PASS":
        return ()
    stamp = _parse_instant(
        recorded_at or _utcnow_iso(), "dispatch_actions.recorded_at"
    ).isoformat()
    regressed = tuple(report.regressed_cases)
    actions: list[ActionRecord] = [
        ActionRecord(
            action="gate",
            benchmark_id=benchmark_id,
            run_id=run_id,
            case_ids=regressed,
            reason="; ".join(report.reasons) or "gate failed",
            recorded_at=stamp,
        ),
        ActionRecord(
            action="investigation",
            benchmark_id=benchmark_id,
            run_id=run_id,
            case_ids=regressed,
            reason=(
                f"investigate regressed cases: {', '.join(regressed)}"
                if regressed
                else f"investigate aggregate regression: {'; '.join(report.reasons)}"
            ),
            recorded_at=stamp,
        ),
    ]
    if prior_consecutive_failures >= ROLLBACK_AFTER_FAILURES:
        actions.append(
            ActionRecord(
                action="rollback",
                benchmark_id=benchmark_id,
                run_id=run_id,
                case_ids=regressed,
                reason=(
                    f"{prior_consecutive_failures + 1} consecutive gate failures "
                    f"({report.suite_name}/{report.version_id}); roll back the "
                    "release, do not merely block it"
                ),
                recorded_at=stamp,
            )
        )
    return tuple(actions)


# ---------------------------------------------------------------------------
# Publishing mined versions + the scheduled runner
# ---------------------------------------------------------------------------


def mined_promotion_record(
    mined: MinedCase, *, promoted_at: str | None = None
) -> PromotionRecord:
    """Promotion record for a mined case, carrying the full provenance the
    legacy kind bucket cannot express."""
    return PromotionRecord(
        failure_id=mined.trace_id,
        kind=pattern_to_failure_kind(mined.pattern),
        promoted_case_id=mined.case.id,
        prior_score=mined.prior_score,
        evaluator_notes=mined.evidence,
        promoted_at=promoted_at or _utcnow_iso(),
        provenance={
            "trace_id": mined.trace_id,
            "incident_id": mined.incident_id,
            "pattern": mined.pattern,
            "evidence": mined.evidence,
            "owner": mined.owner,
            "cadence": mined.cadence,
        },
    )


def required_baseline(
    base: BaselineRecord | None,
    *,
    suite_cases: Sequence[EvalCase],
    mined_cases: Sequence[MinedCase],
    seed: int,
) -> BaselineRecord:
    """The required baseline for a version containing mined cases.

    Base cases keep their pinned scores. Mined cases require 1.0: the
    production failure they encode must be fixed, so any score below 1.0
    is a per-case regression under :func:`compare_against_frozen`. The
    aggregate threshold is the mean of the per-case requirements. A base
    version that has no pinned score for a non-skipped case fails closed —
    a frozen version with an incomplete baseline cannot gate.
    """
    per_case: dict[str, float] = dict(base.per_case) if base is not None else {}
    for mined in mined_cases:
        per_case[mined.case.id] = 1.0
    scored = [c for c in suite_cases if not c.skip]
    for case in scored:
        if case.id not in per_case:
            raise TracebenchError(
                f"base baseline has no pinned score for case {case.id!r} — "
                "cannot gate an incompletely pinned version"
            )
    total = len(suite_cases)
    skipped = sum(1 for c in suite_cases if c.skip)
    passed = sum(1 for c in scored if per_case[c.id] >= 1.0)
    pass_rate = (
        sum(per_case[c.id] for c in scored) / len(scored) if scored else 0.0
    )
    return BaselineRecord(
        agent="tracebench-required",
        seed=seed,
        pass_rate=pass_rate,
        total=total,
        passed=passed,
        failed=total - passed - skipped,
        skipped=skipped,
        per_case=per_case,
    )


def publish_mined_version(
    store: SnapshotStore,
    base_suite: EvalSuite,
    version_id: str,
    meta: VersionMeta,
    *,
    mined_cases: Sequence[MinedCase],
    baseline: BaselineRecord,
    prev_version_id: str | None = None,
    promoted_at: str | None = None,
) -> tuple[BenchmarkVersion, EvalSuite]:
    """Publish a new frozen version holding the mined cases.

    The mined cases are appended to a copy of the base suite (the caller's
    suite is not mutated); promotion records carry full provenance. The
    store's own validation applies unchanged: duplicate case ids and
    non-canonical suites are refused at publish.
    """
    new_suite = EvalSuite(
        name=base_suite.name,
        version=base_suite.version,
        cases=[*base_suite.cases, *[m.case for m in mined_cases]],
        cost_bound=base_suite.cost_bound,
    )
    records = [
        mined_promotion_record(m, promoted_at=promoted_at).to_dict()
        for m in mined_cases
    ]
    version = store.publish(
        new_suite,
        version_id,
        meta,
        baseline,
        prev_version_id=prev_version_id,
        promotions=records,
    )
    return version, new_suite


@dataclass(frozen=True)
class ScheduledRunResult:
    """Outcome of one scheduled regression run."""

    version_id: str
    verdict: str
    published_new_version: bool
    mined_cases: int
    new_cases: int
    manifest: BenchmarkManifest
    actions: tuple[ActionRecord, ...]
    consecutive_failures: int
    report_path: str
    actions_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "agenteval-bench",
            "kind": "tracebench-regress",
            "version_id": self.version_id,
            "verdict": self.verdict,
            "published_new_version": self.published_new_version,
            "mined_cases": self.mined_cases,
            "new_cases": self.new_cases,
            "manifest": self.manifest.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "consecutive_failures": self.consecutive_failures,
            "report_path": self.report_path,
            "actions_path": self.actions_path,
            "manifest_path": self.manifest_path,
        }


def run_scheduled(
    *,
    corpus: str | Path,
    store: str | Path,
    suite_name: str,
    version_id: str,
    prev_version_id: str | None,
    candidate_outputs: Mapping[str, str],
    owner: str,
    cadence: str,
    out_dir: str | Path,
    seed: int = DEFAULT_SEED,
    candidate_name: str = "candidate",
    prior_consecutive_failures: int = 0,
    mined_at: str | None = None,
) -> ScheduledRunResult:
    """One scheduled regression run: mine -> publish (when new cases exist)
    -> gate against the frozen version -> dispatch actions -> write the
    run's artifacts (report, action ledger, manifest) to ``out_dir``.

    Steady state: when the corpus yields no cases the store does not yet
    hold, no new version is published — the candidate is gated against the
    previous version instead. A run with no mined cases at all and no
    previous version has nothing to regress and fails closed.
    """
    if prior_consecutive_failures < 0:
        raise ValueError("prior_consecutive_failures must be >= 0")
    parsed_cadence = parse_cadence(cadence)
    stamp = _parse_instant(mined_at or _utcnow_iso(), "run_scheduled.mined_at")

    load = load_corpus(corpus)
    report = mine_traces(
        load.records,
        owner=owner,
        cadence=parsed_cadence,
        mined_at=stamp.isoformat(),
        corpus_digest=load.digest,
        skipped_records=load.skipped,
    )
    manifest = build_manifest(
        report.cases,
        benchmark_id=suite_name,
        owner=owner,
        cadence=parsed_cadence,
        created_at=stamp.isoformat(),
        last_mined_at=stamp.isoformat(),
    )

    store_obj = SnapshotStore(store)
    if prev_version_id is not None:
        try:
            _prev_version, prev_suite = store_obj.load(suite_name, prev_version_id)
        except SnapshotError as e:
            raise TracebenchError(
                f"cannot load previous version {suite_name}/{prev_version_id}: {e}"
            ) from e
        base_baseline = _prev_version.baseline
        present = {c.id for c in prev_suite.cases}
        base_suite = prev_suite
    else:
        _prev_version = None
        base_baseline = None
        present = set()
        base_suite = EvalSuite(
            name=suite_name, version=1, cases=[], cost_bound=CostBound()
        )
    new_mined = [m for m in report.cases if m.case.id not in present]

    if new_mined:
        baseline = required_baseline(
            base_baseline,
            suite_cases=[*base_suite.cases, *[m.case for m in new_mined]],
            mined_cases=new_mined,
            seed=seed,
        )
        meta = VersionMeta(
            owner=owner,
            why=(
                f"tracebench: mined {len(new_mined)} case(s) from "
                f"{len(manifest.source_incidents)} incident(s), "
                f"corpus {report.corpus_digest[:19]}"
            ),
            changed_from_prev=(
                f"+{len(new_mined)} mined case(s)" if prev_version_id else ""
            ),
            cadence=parsed_cadence,
            regression_set=suite_name,
            feedback_loop="tracebench-regression",
        )
        try:
            _version, effective_suite = publish_mined_version(
                store_obj,
                base_suite,
                version_id,
                meta,
                mined_cases=new_mined,
                baseline=baseline,
                prev_version_id=prev_version_id,
                promoted_at=stamp.isoformat(),
            )
        except SnapshotError as e:
            raise TracebenchError(
                f"cannot publish mined version {suite_name}/{version_id}: {e}"
            ) from e
        effective_version = version_id
        published = True
    else:
        if prev_version_id is None:
            raise TracebenchError(
                "nothing to regress: the corpus mined no cases and no "
                "previous version was given"
            )
        effective_suite = base_suite
        effective_version = prev_version_id
        published = False

    # Candidate agent with call-order attribution (same contract as the
    # snapshot-compare CLI): any divergence between the engine's call order
    # and the frozen case order fails loudly, never silently misattributed.
    ordered_ids = [c.id for c in effective_suite.cases if not c.skip]
    replay_ids = iter(ordered_ids)
    called_ids: list[str] = []
    missing_ids: list[str] = []

    def candidate_fn(_input: str) -> str:
        try:
            cid = next(replay_ids)
        except StopIteration:
            raise TracebenchError(
                "candidate agent_fn called more times than the frozen suite "
                "has cases — attribution would be silent misattribution"
            ) from None
        called_ids.append(cid)
        if cid not in candidate_outputs:
            missing_ids.append(cid)
            return ""
        return candidate_outputs[cid]

    comparison = compare_against_frozen(
        store_obj,
        suite_name,
        effective_version,
        candidate_fn,
        seed=seed,
        candidate_agent=candidate_name,
    )
    if called_ids != ordered_ids:
        raise TracebenchError(
            "candidate output attribution mismatch — the engine's agent_fn "
            "call order diverged from frozen case order; refusing to score"
        )

    benchmark_id = f"{suite_name}/{effective_version}"
    actions = dispatch_actions(
        comparison,
        benchmark_id=benchmark_id,
        run_id=effective_version,
        prior_consecutive_failures=prior_consecutive_failures,
        recorded_at=stamp.isoformat(),
    )
    consecutive = (
        prior_consecutive_failures + 1 if comparison.verdict == "GATE_FAIL" else 0
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "report.json"
    actions_path = out_path / "actions.jsonl"
    manifest_path = out_path / "manifest.json"
    result = ScheduledRunResult(
        version_id=effective_version,
        verdict=comparison.verdict,
        published_new_version=published,
        mined_cases=len(report.cases),
        new_cases=len(new_mined),
        manifest=manifest,
        actions=actions,
        consecutive_failures=consecutive,
        report_path=str(report_path),
        actions_path=str(actions_path),
        manifest_path=str(manifest_path),
    )
    report_doc = result.to_dict()
    report_doc["comparison"] = {
        "suite_name": comparison.suite_name,
        "version_id": comparison.version_id,
        "digest": comparison.digest,
        "candidate_agent": comparison.candidate_agent,
        "seed": comparison.seed,
        "baseline_pass_rate": comparison.baseline_pass_rate,
        "candidate_pass_rate": comparison.candidate_pass_rate,
        "regressed_cases": list(comparison.regressed_cases),
        "reasons": list(comparison.reasons),
        "missing_candidate_outputs": missing_ids,
    }
    report_doc["mine"] = report.to_dict()
    report_path.write_text(_canonical_json(report_doc), encoding="utf-8")
    actions_path.write_text(
        "".join(a.to_json() + "\n" for a in actions), encoding="utf-8"
    )
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    return result
