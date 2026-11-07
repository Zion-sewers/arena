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
   The loop is closed across runs, not per run: with ``state_dir`` the
   runner reads the previous run's state (previous version id + consecutive
   failure streak) and writes the new one back, so versions chain and the
   rollback arm is reachable on the 3rd consecutive failure without any
   operator hand-feeding a streak. A mined case whose content changed is
   published as an update (same id, new content) and a mined case whose
   trace left the corpus is retired from the new version — the frozen
   benchmark is addition-only no longer; both are reported per run.
   Cases the mine never produced (a hand-seeded base suite) are carried
   forward untouched: only cases with mining provenance retire.
3. The drift gate: the scheduled run is a regression, not a smoke test.
   Before publishing, the newly mined oracle is scored against the
   previous frozen version — a case whose oracle changed since last week
   is a drifted case, and a drifted benchmark fails the run (gate +
   investigation, counting toward the rollback streak) even when the
   candidate passes the new version: the benchmark moved under the
   candidate, so last week's certification no longer applies. The new
   version still publishes, and an unchanged corpus clears the flag the
   next week. A schedule that mines a static corpus passes, correctly —
   nothing changed — but it can now fail on a corpus change with no
   operator in the loop.
4. Every artifact says what corpus it came from: ``corpus_kind``
   (``sample`` / ``production`` / ``unknown``) travels on the mine report
   and the manifest, so a consumer reading an artifact alone can tell
   invented fixture incidents from real ones. The runner never guesses
   the kind; the caller declares it (the shipped workflow declares
   ``sample`` for the checked-in fixture corpus).

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
(:meth:`BenchmarkManifest.is_stale`); :func:`run_scheduled` reports whether
the previous version was already stale when the run started
(``previous_stale`` in the run report), so staleness is observed by a
consumer on every run instead of living only in the manifest type.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
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

#: What corpus a mined artifact came from. The runner never guesses: the
#: caller declares it (``sample`` for the checked-in fixture corpus,
#: ``production`` for a real trace export, ``unknown`` when undeclared).
#: It travels on the mine report and the manifest so an artifact alone
#: tells a consumer whether its incident ids are real or invented.
CorpusKind = Literal["sample", "production", "unknown"]

CORPUS_KINDS: tuple[str, ...] = ("sample", "production", "unknown")

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


def parse_corpus_kind(value: str) -> CorpusKind:
    """Parse a corpus kind; anything outside the closed set is refused."""
    if value not in CORPUS_KINDS:
        raise ValueError(
            f"corpus_kind must be one of {sorted(CORPUS_KINDS)}, got {value!r}"
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
    corpus_kind: CorpusKind = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "agenteval-bench",
            "kind": "tracebench-mine",
            "cases": [c.to_dict() for c in self.cases],
            "thresholds": self.thresholds.to_dict(),
            "corpus_digest": self.corpus_digest,
            "corpus_kind": self.corpus_kind,
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
        corpus_kind_raw = d.get("corpus_kind", "unknown")
        if not isinstance(corpus_kind_raw, str):
            raise TypeError("mine report: corpus_kind must be a string")
        return cls(
            cases=tuple(cases),
            thresholds=MineThresholds.from_dict(thresholds_raw),
            corpus_digest=_require_str(d, "corpus_digest", "mine report"),
            mined_at=_require_str(d, "mined_at", "mine report"),
            owner=_require_str(d, "owner", "mine report"),
            cadence=parse_cadence(_require_str(d, "cadence", "mine report")),
            skipped_records=_require_int(d, "skipped_records", "mine report"),
            source_records=_require_int(d, "source_records", "mine report"),
            corpus_kind=parse_corpus_kind(corpus_kind_raw),
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
    corpus_kind: str = "unknown",
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
        corpus_kind=parse_corpus_kind(corpus_kind),
    )


# ---------------------------------------------------------------------------
# Benchmark manifest (owner + cadence per benchmark, staleness)
# ---------------------------------------------------------------------------


def _is_stale_since(last: datetime, cadence: Cadence, now: datetime) -> bool:
    """The one staleness predicate: stale when ``now`` is more than one
    cadence past ``last``. Both :meth:`BenchmarkManifest.is_stale` and
    :func:`run_scheduled` call this, so the manifest type and the
    production path can never drift into two definitions of stale."""
    return (now - last) > timedelta(days=CADENCE_DAYS[cadence])


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
    corpus_kind: CorpusKind = "unknown"

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
        ambiguous clock. Delegates to :func:`_is_stale_since`, the one
        predicate the scheduled runner also calls, so the manifest and
        the production path can never disagree about staleness."""
        if now.tzinfo is None:
            raise ValueError("is_stale: 'now' must carry a timezone")
        last = _parse_instant(self.last_mined_at, "BenchmarkManifest.last_mined_at")
        return _is_stale_since(last, self.cadence, now)

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
            "corpus_kind": self.corpus_kind,
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
        corpus_kind_raw = d.get("corpus_kind", "unknown")
        if not isinstance(corpus_kind_raw, str):
            raise TypeError("manifest: corpus_kind must be a string")
        return cls(
            benchmark_id=_require_str(d, "benchmark_id", "manifest"),
            owner=_require_str(d, "owner", "manifest"),
            cadence=parse_cadence(_require_str(d, "cadence", "manifest")),
            case_ids=tuple(case_ids),
            source_incidents=tuple(incidents),
            created_at=_require_str(d, "created_at", "manifest"),
            last_mined_at=_require_str(d, "last_mined_at", "manifest"),
            corpus_kind=parse_corpus_kind(corpus_kind_raw),
        )


def build_manifest(
    cases: Sequence[MinedCase],
    *,
    benchmark_id: str,
    owner: str,
    cadence: str,
    created_at: str | None = None,
    last_mined_at: str | None = None,
    corpus_kind: str = "unknown",
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
        corpus_kind=parse_corpus_kind(corpus_kind),
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
    # A retired case leaves no pinned score behind: scores are for suite
    # cases only, so a case the current mine dropped cannot linger in the
    # baseline of the version that drops it.
    suite_ids = {c.id for c in suite_cases}
    per_case = {cid: score for cid, score in per_case.items() if cid in suite_ids}
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
    suite_cases: Sequence[EvalCase] | None = None,
) -> tuple[BenchmarkVersion, EvalSuite]:
    """Publish a new frozen version holding the mined cases.

    The mined cases are appended to a copy of the base suite (the caller's
    suite is not mutated); promotion records carry full provenance. When
    ``suite_cases`` is given it IS the new suite — the update/retire path,
    where the caller already replaced updated cases and dropped retired
    ones — and ``mined_cases`` then only sources the promotion records
    (every mined case must appear in ``suite_cases``). The store's own
    validation applies unchanged: duplicate case ids and non-canonical
    suites are refused at publish.
    """
    if suite_cases is None:
        new_cases = [*base_suite.cases, *[m.case for m in mined_cases]]
    else:
        new_cases = list(suite_cases)
        suite_ids = {c.id for c in new_cases}
        missing = {m.case.id for m in mined_cases} - suite_ids
        if missing:
            raise TracebenchError(
                "publish_mined_version: mined case(s) "
                f"{sorted(missing)} missing from suite_cases — refusing to "
                "publish promotion records for cases the version does not hold"
            )
    new_suite = EvalSuite(
        name=base_suite.name,
        version=base_suite.version,
        cases=new_cases,
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
    updated_cases: int = 0
    retired_cases: int = 0
    updated_case_ids: tuple[str, ...] = ()
    retired_case_ids: tuple[str, ...] = ()
    previous_stale: bool | None = None
    version_ignored: bool = False
    drift_detected: bool = False
    drifted_case_ids: tuple[str, ...] = ()
    corpus_total_lines: int = 0
    corpus_skipped: int = 0
    previous_corpus_total_lines: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "agenteval-bench",
            "kind": "tracebench-regress",
            "version_id": self.version_id,
            "verdict": self.verdict,
            "published_new_version": self.published_new_version,
            "mined_cases": self.mined_cases,
            "new_cases": self.new_cases,
            "updated_cases": self.updated_cases,
            "retired_cases": self.retired_cases,
            "updated_case_ids": list(self.updated_case_ids),
            "retired_case_ids": list(self.retired_case_ids),
            "previous_stale": self.previous_stale,
            "version_ignored": self.version_ignored,
            "drift_detected": self.drift_detected,
            "drifted_case_ids": list(self.drifted_case_ids),
            "corpus_total_lines": self.corpus_total_lines,
            "corpus_skipped": self.corpus_skipped,
            "previous_corpus_total_lines": self.previous_corpus_total_lines,
            "manifest": self.manifest.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "consecutive_failures": self.consecutive_failures,
            "report_path": self.report_path,
            "actions_path": self.actions_path,
            "manifest_path": self.manifest_path,
        }


def _case_content(case: EvalCase) -> dict[str, Any]:
    """Canonical content of a case for update detection: two cases with the
    same id but different content are an update, not the same case."""
    return asdict(case)


_RUN_STATE_FILENAME = "state.json"


def _load_run_state(
    state_dir: str | Path, suite_name: str
) -> dict[str, Any] | None:
    """Read the previous run's state (previous version + failure streak).

    Returns None when no state file exists yet (first run). A present but
    malformed file, or state recorded for a different suite, fails closed —
    a scheduler must never silently restart the streak or the version chain
    because its state was corrupted.
    """
    path = Path(state_dir) / _RUN_STATE_FILENAME
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise TracebenchError(f"cannot read run state {path}: {e}") from e
    if not isinstance(raw, Mapping):
        raise TracebenchError(f"run state {path} must be a JSON object")
    if raw.get("suite_name") != suite_name:
        raise TracebenchError(
            f"run state {path} is for suite {raw.get('suite_name')!r}, "
            f"not {suite_name!r} — refusing to chain across suites"
        )
    if "prev_version_id" not in raw:
        raise TracebenchError(
            f"run state {path} has no prev_version_id — refusing to "
            "chain from a state file that does not name its version"
        )
    prev = raw.get("prev_version_id")
    if prev is not None and not isinstance(prev, str):
        raise TracebenchError(
            f"run state {path}: prev_version_id must be a string or null"
        )
    streak = raw.get("consecutive_failures")
    if isinstance(streak, bool) or not isinstance(streak, int) or streak < 0:
        raise TracebenchError(
            f"run state {path}: consecutive_failures must be an integer >= 0"
        )
    total = raw.get("corpus_total_lines")
    if total is not None and (
        isinstance(total, bool) or not isinstance(total, int) or total < 0
    ):
        raise TracebenchError(
            f"run state {path}: corpus_total_lines must be an integer >= 0"
        )
    return {
        "prev_version_id": prev,
        "consecutive_failures": streak,
        "corpus_total_lines": total,
    }


def _write_run_state(
    state_dir: str | Path,
    *,
    suite_name: str,
    prev_version_id: str,
    consecutive_failures: int,
    updated_at: str,
    corpus_total_lines: int | None = None,
) -> None:
    """Persist the run's state atomically (unique temp file + replace).
    Any I/O failure is a TracebenchError, never a raw traceback: the CLI
    reports it as a clean error line."""
    directory = Path(state_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        doc = {
            "tool": "agenteval-bench",
            "kind": "tracebench-run-state",
            "suite_name": suite_name,
            "prev_version_id": prev_version_id,
            "consecutive_failures": consecutive_failures,
            "corpus_total_lines": corpus_total_lines,
            "updated_at": updated_at,
        }
        fd, tmp_name = tempfile.mkstemp(
            dir=directory, prefix=_RUN_STATE_FILENAME + "."
        )
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write(_canonical_json(doc))
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        Path(tmp_name).replace(directory / _RUN_STATE_FILENAME)
    except TracebenchError:
        raise
    except OSError as e:
        raise TracebenchError(f"cannot write run state in {directory}: {e}") from e


def _check_state_dir_writable(state_dir: str | Path) -> None:
    """Preflight for ``--state-dir``: prove the directory can be written
    BEFORE the run publishes anything. A state write that fails after
    publishing leaves a published version with no state, and the next run
    then re-derives version "1" and dies on the duplicate — the exact
    hand-feeding ``--state-dir`` was built to eliminate. Fail here, with
    nothing half-done, instead."""
    directory = Path(state_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".writable-probe.")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write("ok")
        finally:
            Path(tmp_name).unlink(missing_ok=True)
    except OSError as e:
        raise TracebenchError(
            f"state dir {directory} is not writable: {e}"
        ) from e


def _check_out_dir_writable(out_dir: str | Path) -> None:
    """Preflight for ``--out``: prove the output directory can be
    created and written BEFORE the run publishes anything. The
    publish-then-write order means an out_dir failure after publishing
    leaves a published version with no artifacts and no state, and the
    next run re-derives the same version id and dies on the duplicate —
    the chain is bricked and needs manual recovery. Fail here, with
    nothing half-done, instead. Mirrors :func:`_check_state_dir_writable`."""
    directory = Path(out_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".writable-probe.")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write("ok")
        finally:
            Path(tmp_name).unlink(missing_ok=True)
    except OSError as e:
        raise TracebenchError(
            f"out dir {directory} is not writable: {e}"
        ) from e


def _mined_case_ids(
    store: SnapshotStore, suite_name: str, version_id: str
) -> set[str]:
    """Case ids this suite ever mined, walked down the version chain.

    Each version's promotions record only the cases promoted in THAT
    version, so a case mined in v1 and carried unchanged into v2 has no
    promotion record in v2 — the set must be collected from every version
    on the chain. This is what scopes retirement: only a case the mine
    once produced can retire when its trace leaves the corpus. A
    hand-seeded base case has no promotion record anywhere on the chain
    and is therefore never retired. A broken chain fails closed.
    """
    mined: set[str] = set()
    current: str | None = version_id
    seen: set[str] = set()
    while current is not None:
        if current in seen:
            raise TracebenchError(
                f"version chain for {suite_name} loops at {current!r} — "
                "refusing to scope retirement from a corrupt chain"
            )
        seen.add(current)
        try:
            version, _suite = store.load(suite_name, current)
        except SnapshotError as e:
            raise TracebenchError(
                f"cannot walk the version chain for {suite_name}: {e}"
            ) from e
        for promotion in version.promotions:
            case_id = promotion.get("promoted_case_id")
            if isinstance(case_id, str):
                mined.add(case_id)
        current = version.prev_version_id
    return mined


def _suite_source_incidents(
    store: SnapshotStore, suite_name: str, version_id: str, case_ids: set[str]
) -> tuple[str, ...]:
    """Incident ids behind the mined cases in a suite version, read from
    promotion provenance down the version chain. A case the mine once
    produced carries its incident id in its promotion record; a
    hand-seeded base case has no promotion and contributes no incident.
    This is what lets the run manifest describe the benchmark actually
    gated (the effective suite) instead of only the cases this run mined.
    A broken chain fails closed, like :func:`_mined_case_ids`."""
    incidents: set[str] = set()
    current: str | None = version_id
    seen: set[str] = set()
    while current is not None:
        if current in seen:
            raise TracebenchError(
                f"version chain for {suite_name} loops at {current!r} — "
                "refusing to describe the gated suite from a corrupt chain"
            )
        seen.add(current)
        try:
            version, _suite = store.load(suite_name, current)
        except SnapshotError as e:
            raise TracebenchError(
                f"cannot walk the version chain for {suite_name}: {e}"
            ) from e
        for promotion in version.promotions:
            if promotion.get("promoted_case_id") not in case_ids:
                continue
            provenance = promotion.get("provenance")
            if not isinstance(provenance, Mapping):
                continue
            incident = provenance.get("incident_id")
            if isinstance(incident, str) and incident:
                incidents.add(incident)
        current = version.prev_version_id
    return tuple(sorted(incidents))


def _derive_next_version_id(prev_version_id: str | None) -> str:
    """Next version id when the caller passes none: ``"1"`` for a first
    run, otherwise the previous numeric id + 1. A non-numeric previous id
    fails closed — the runner must never guess how a human versions.
    ``int()`` is attempted directly (``str.isdigit`` accepts unicode
    digits such as "²" that ``int()`` refuses, so it cannot be the test)."""
    if prev_version_id is None:
        return "1"
    try:
        number = int(prev_version_id)
    except ValueError:
        raise TracebenchError(
            "cannot auto-derive the next version id from non-numeric "
            f"previous version {prev_version_id!r} — pass --version"
        ) from None
    if number < 0:
        raise TracebenchError(
            "cannot auto-derive the next version id from negative "
            f"previous version {prev_version_id!r} — pass --version"
        )
    return str(number + 1)


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
    prior_consecutive_failures: int | None = None,
    mined_at: str | None = None,
    state_dir: str | Path | None = None,
    corpus_kind: str = "unknown",
) -> ScheduledRunResult:
    """One scheduled regression run: mine -> publish (when the benchmark
    changed) -> gate against the frozen version -> dispatch actions ->
    write the run's artifacts (report, action ledger, manifest) to
    ``out_dir``.

    Steady state: when the corpus yields no cases the store does not yet
    hold, no new version is published — the candidate is gated against the
    previous version instead. A run with no mined cases at all and no
    previous version has nothing to regress and fails closed.

    ``state_dir`` closes the loop across runs: the runner reads the
    previous run's state (previous version id + consecutive failure
    streak) and writes the new one back, so versions chain and the
    rollback arm is reachable on the 3rd consecutive failure without any
    operator hand-feeding. Explicit ``prev_version_id`` /
    ``prior_consecutive_failures`` override the state; with no state dir
    and no explicit values the run behaves exactly as before (previous
    version None, streak 0). A changed case (same id, new content) is
    published as an update and a mined case whose trace left the corpus
    is retired from the new version — except when the mine yields no
    cases at all, which never retires anything (an unreadable corpus
    must not wipe the benchmark). Cases the mine never produced (a
    hand-seeded base suite) are carried forward untouched: only cases
    with mining provenance retire. An empty ``version_id`` auto-derives
    the next id from the chained previous version (``"1"`` first run).

    The drift gate: before the candidate is scored, the newly mined
    oracle is compared with the previous frozen version. A case whose
    content changed since last week is drift, and drift fails the run
    (gate + investigation, counting toward the rollback streak) even
    when the candidate passes the new version — the benchmark moved
    under the candidate, so last week's certification no longer
    applies. The new version still publishes; an unchanged corpus
    clears the flag the next week. This is what makes the scheduled
    run a regression instead of a smoke test: with a static corpus it
    passes, correctly, but a corpus change fails it with no operator
    in the loop.

    The corpus input boundary fails closed: a corpus with malformed
    lines is refused before anything is mined or published, because a
    corrupt export is indistinguishable from traces legitimately
    leaving the corpus. Corpus totals ride in the report (current and
    previous run) so a shrink is visible at the consumption surface.
    Both output locations are preflighted writable before publishing:
    a run that publishes and then cannot write its artifacts or state
    bricks the version chain, so that failure must happen first, with
    nothing half-done. The written manifest describes the benchmark
    actually gated (the effective suite), and a previous version that
    was already past its cadence dispatches an investigation action —
    the run itself is the refresh, so staleness is flagged, not failed.
    """
    parsed_cadence = parse_cadence(cadence)
    stamp = _parse_instant(mined_at or _utcnow_iso(), "run_scheduled.mined_at")
    state = (
        _load_run_state(state_dir, suite_name) if state_dir is not None else None
    )
    if state_dir is not None:
        _check_state_dir_writable(state_dir)
    _check_out_dir_writable(out_dir)
    effective_prev: str | None = (
        prev_version_id
        if prev_version_id is not None
        else (state["prev_version_id"] if state is not None else None)
    )
    effective_prior = (
        prior_consecutive_failures
        if prior_consecutive_failures is not None
        else (state["consecutive_failures"] if state is not None else 0)
    )
    if effective_prior < 0:
        raise ValueError("prior_consecutive_failures must be >= 0")
    requested_version = version_id or _derive_next_version_id(effective_prev)

    load = load_corpus(corpus)
    if load.skipped > 0:
        raise TracebenchError(
            f"corpus {corpus} has {load.skipped} malformed line(s) out of "
            f"{load.total_lines} — refusing to mine a corrupt export: a "
            "truncated or corrupt export is indistinguishable from traces "
            "legitimately leaving the corpus, so mining it would silently "
            "shrink the benchmark. Fix the export instead."
        )
    report = mine_traces(
        load.records,
        owner=owner,
        cadence=parsed_cadence,
        mined_at=stamp.isoformat(),
        corpus_digest=load.digest,
        skipped_records=load.skipped,
        corpus_kind=corpus_kind,
    )

    store_obj = SnapshotStore(store)
    previous_stale: bool | None = None
    prev_created: datetime | None = None
    prev_cadence: Cadence | None = None
    if effective_prev is not None:
        try:
            _prev_version, prev_suite = store_obj.load(suite_name, effective_prev)
        except SnapshotError as e:
            raise TracebenchError(
                f"cannot load previous version {suite_name}/{effective_prev}: {e}"
            ) from e
        base_baseline = _prev_version.baseline
        present = {c.id for c in prev_suite.cases}
        base_suite = prev_suite
        prev_created = _parse_instant(
            _prev_version.created_at, "previous version created_at"
        )
        # Staleness is judged by the benchmark's own recorded cadence, not
        # the flag this run happened to pass: the same version must not
        # flip stale/not-stale with the caller's flag. The predicate is
        # the shared _is_stale_since, the same one
        # BenchmarkManifest.is_stale calls.
        prev_cadence = (
            parse_cadence(_prev_version.meta.cadence)
            if _prev_version.meta.cadence
            else parsed_cadence
        )
        previous_stale = _is_stale_since(prev_created, prev_cadence, stamp)
    else:
        _prev_version = None
        base_baseline = None
        present = set()
        base_suite = EvalSuite(
            name=suite_name, version=1, cases=[], cost_bound=CostBound()
        )
    new_mined = [m for m in report.cases if m.case.id not in present]
    updated: list[MinedCase] = []
    retired_ids: list[str] = []
    if _prev_version is not None and report.cases:
        prev_by_id = {c.id: c for c in base_suite.cases}
        mined_ids = {m.case.id for m in report.cases}
        updated = [
            m
            for m in report.cases
            if m.case.id in prev_by_id
            and _case_content(prev_by_id[m.case.id]) != _case_content(m.case)
        ]
        # Retire only cases with mining provenance (ever promoted on the
        # version chain): a hand-seeded base case has no trace to leave
        # the corpus, so "not in the current mine" must never delete it.
        # _prev_version is only set when effective_prev is set.
        assert effective_prev is not None
        ever_mined = _mined_case_ids(store_obj, suite_name, effective_prev)
        retired_ids = [
            c.id
            for c in base_suite.cases
            if c.id not in mined_ids and c.id in ever_mined
        ]

    if new_mined or updated or retired_ids:
        updated_by_id = {m.case.id: m for m in updated}
        effective_cases: list[EvalCase] = []
        for prev_case in base_suite.cases:
            if prev_case.id in retired_ids:
                continue
            if prev_case.id in updated_by_id:
                effective_cases.append(updated_by_id[prev_case.id].case)
            else:
                effective_cases.append(prev_case)
        effective_cases.extend(m.case for m in new_mined)
        baseline = required_baseline(
            base_baseline,
            suite_cases=effective_cases,
            mined_cases=[*new_mined, *updated],
            seed=seed,
        )
        change_parts = [
            f"+{len(new_mined)} new" if new_mined else "",
            f"~{len(updated)} updated" if updated else "",
            f"-{len(retired_ids)} retired" if retired_ids else "",
        ]
        meta = VersionMeta(
            owner=owner,
            why=(
                f"tracebench: mined {len(new_mined)} case(s) from "
                f"{len({m.incident_id for m in report.cases})} incident(s), "
                f"corpus {report.corpus_digest[:19]}"
            ),
            changed_from_prev=(
                ", ".join(p for p in change_parts if p)
                if effective_prev
                else ""
            ),
            cadence=parsed_cadence,
            regression_set=suite_name,
            feedback_loop="tracebench-regression",
        )
        try:
            _version, effective_suite = publish_mined_version(
                store_obj,
                base_suite,
                requested_version,
                meta,
                mined_cases=[*new_mined, *updated],
                baseline=baseline,
                prev_version_id=effective_prev,
                promoted_at=stamp.isoformat(),
                suite_cases=effective_cases,
            )
        except SnapshotError as e:
            raise TracebenchError(
                "cannot publish mined version "
                f"{suite_name}/{requested_version}: {e}"
            ) from e
        effective_version = requested_version
        published = True
    else:
        if effective_prev is None:
            raise TracebenchError(
                "nothing to regress: the corpus mined no cases and no "
                "previous version was given"
            )
        effective_suite = base_suite
        effective_version = effective_prev
        published = False
    version_ignored = not published and requested_version != effective_version

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

    # Drift gate: the newly mined oracle against the previous frozen
    # version. An updated case means the benchmark moved since last week,
    # so the previous certification no longer applies — the run fails even
    # when the candidate passes the new version. The drifted ids are the
    # updated ids: same detection, same meaning.
    drifted_ids = [m.case.id for m in updated]
    drift_reasons = tuple(
        f"benchmark drift: case {cid} changed since version "
        f"{effective_prev} — re-verify the candidate against the new version"
        for cid in drifted_ids
    )
    if drifted_ids:
        regressed = tuple(
            dict.fromkeys([*comparison.regressed_cases, *drifted_ids])
        )
        effective_comparison = replace(
            comparison,
            verdict="GATE_FAIL",
            regressed_cases=regressed,
            reasons=(*comparison.reasons, *drift_reasons),
        )
    else:
        effective_comparison = comparison

    benchmark_id = f"{suite_name}/{effective_version}"
    actions = dispatch_actions(
        effective_comparison,
        benchmark_id=benchmark_id,
        run_id=effective_version,
        prior_consecutive_failures=effective_prior,
        recorded_at=stamp.isoformat(),
    )
    if previous_stale:
        # A past-cadence benchmark is flagged, not just recorded: the
        # run itself is the refresh, so staleness is an investigation
        # action, not a failure — but it is a dispatched action in the
        # ledger and the workflow summary, never an inert boolean in an
        # artifact nobody is woken by.
        assert effective_prev is not None and prev_cadence is not None
        assert prev_created is not None
        actions = (
            *actions,
            ActionRecord(
                action="investigation",
                benchmark_id=benchmark_id,
                run_id=effective_version,
                case_ids=(),
                reason=(
                    f"previous version {suite_name}/{effective_prev} was "
                    f"past its {prev_cadence} cadence when this run "
                    f"started (created {prev_created.isoformat()}); this "
                    "run re-mined it — verify the refresh"
                ),
                recorded_at=stamp.isoformat(),
            ),
        )
    consecutive = (
        effective_prior + 1 if effective_comparison.verdict == "GATE_FAIL" else 0
    )

    # The manifest describes the benchmark actually gated — the effective
    # suite — not only the cases this run mined. A steady-state run that
    # mined nothing gates the previous version, so its manifest must list
    # that version's cases; a publishing run's manifest lists the new
    # version's cases. Incident ids come from promotion provenance down
    # the gated version's chain; hand-seeded base cases carry none.
    gated_case_ids = {c.id for c in effective_suite.cases}
    manifest = BenchmarkManifest(
        benchmark_id=suite_name,
        owner=owner,
        cadence=parsed_cadence,
        case_ids=tuple(sorted(gated_case_ids)),
        source_incidents=_suite_source_incidents(
            store_obj, suite_name, effective_version, gated_case_ids
        ),
        created_at=(
            prev_created.isoformat()
            if not published and prev_created is not None
            else stamp.isoformat()
        ),
        last_mined_at=stamp.isoformat(),
        corpus_kind=parse_corpus_kind(corpus_kind),
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "report.json"
    actions_path = out_path / "actions.jsonl"
    manifest_path = out_path / "manifest.json"
    result = ScheduledRunResult(
        version_id=effective_version,
        verdict=effective_comparison.verdict,
        published_new_version=published,
        mined_cases=len(report.cases),
        new_cases=len(new_mined),
        updated_cases=len(updated),
        retired_cases=len(retired_ids),
        updated_case_ids=tuple(m.case.id for m in updated),
        retired_case_ids=tuple(retired_ids),
        previous_stale=previous_stale,
        version_ignored=version_ignored,
        drift_detected=bool(drifted_ids),
        drifted_case_ids=tuple(drifted_ids),
        corpus_total_lines=load.total_lines,
        corpus_skipped=load.skipped,
        previous_corpus_total_lines=(
            state["corpus_total_lines"] if state is not None else None
        ),
        manifest=manifest,
        actions=actions,
        consecutive_failures=consecutive,
        report_path=str(report_path),
        actions_path=str(actions_path),
        manifest_path=str(manifest_path),
    )
    report_doc = result.to_dict()
    report_doc["comparison"] = {
        "suite_name": effective_comparison.suite_name,
        "version_id": effective_comparison.version_id,
        "digest": effective_comparison.digest,
        "candidate_agent": effective_comparison.candidate_agent,
        "seed": effective_comparison.seed,
        "baseline_pass_rate": effective_comparison.baseline_pass_rate,
        "candidate_pass_rate": effective_comparison.candidate_pass_rate,
        "regressed_cases": list(effective_comparison.regressed_cases),
        "reasons": list(effective_comparison.reasons),
        "missing_candidate_outputs": missing_ids,
    }
    report_doc["drift"] = {
        "detected": bool(drifted_ids),
        "drifted_case_ids": list(drifted_ids),
        "reasons": list(drift_reasons),
    }
    report_doc["mine"] = report.to_dict()
    report_path.write_text(_canonical_json(report_doc), encoding="utf-8")
    actions_path.write_text(
        "".join(a.to_json() + "\n" for a in actions), encoding="utf-8"
    )
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    if state_dir is not None:
        _write_run_state(
            state_dir,
            suite_name=suite_name,
            prev_version_id=effective_version,
            consecutive_failures=consecutive,
            updated_at=stamp.isoformat(),
            corpus_total_lines=load.total_lines,
        )
    return result
