"""Trace-mined benchmarks (arena#41): adversarial tests.

Every test here attacks the acceptance criteria, not the happy path:
provenance is structural (a case without it cannot exist), the benchmark
is living (re-mining a changed corpus changes it; re-mining an unchanged
corpus reproduces it byte-identically), every pattern has a positive and
a negative control, and every gate failure dispatches a real action
record — gate, investigation, and rollback on the third consecutive
failure — into a machine-readable ledger.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agenteval_bench.models import EvalCase
from agenteval_bench.regression import ComparisonReport, PromotionRecord
from agenteval_bench.snapshots import SnapshotStore
from agenteval_bench.tracebench import (
    BenchmarkManifest,
    MineReport,
    MineThresholds,
    ToolCall,
    TracebenchError,
    TraceRecord,
    build_manifest,
    detect_frequent_escalation,
    detect_high_variance,
    detect_latency_spike,
    detect_misrouting,
    detect_repeated_tool_calls,
    detect_schema_violation,
    detect_step_limit_loop,
    dispatch_actions,
    load_corpus,
    mine_traces,
    parse_cadence,
    publish_mined_version,
    required_baseline,
    run_scheduled,
)

MINED_AT = "2026-11-06T19:00:00+00:00"
OWNER = "arena-maintainers"


def _trace(
    trace_id: str = "t-1",
    incident_id: str = "inc-1",
    agent: str = "agent-a",
    tool_calls: list[dict[str, str]] | None = None,
    steps: int = 2,
    step_limit: int = 10,
    duration_ms: float = 1000,
    cost_usd: float = 0.05,
    outcome: str = "success",
    schema_violations: int = 0,
    expected_output: str = "the right answer",
    error: str = "",
) -> TraceRecord:
    return TraceRecord(
        trace_id=trace_id,
        incident_id=incident_id,
        agent=agent,
        tool_calls=tuple(
            ToolCall(name=c["name"], expected=c.get("expected"))
            for c in (tool_calls if tool_calls is not None else [{"name": "search"}])
        ),
        steps=steps,
        step_limit=step_limit,
        duration_ms=duration_ms,
        cost_usd=cost_usd,
        outcome=outcome,  # type: ignore[arg-type]
        schema_violations=schema_violations,
        expected_output=expected_output,
        error=error,
    )


def _sample_corpus_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "traces" / (
        "production-traces.sample.jsonl"
    )


# ---------------------------------------------------------------------------
# TraceRecord boundaries
# ---------------------------------------------------------------------------


def test_trace_record_roundtrip() -> None:
    record = _trace()
    assert TraceRecord.from_dict(record.to_dict()) == record


def test_trace_record_refuses_missing_provenance() -> None:
    for key in ("trace_id", "incident_id"):
        doc = _trace().to_dict()
        doc[key] = ""
        with pytest.raises(ValueError, match=key):
            TraceRecord.from_dict(doc)


def test_trace_record_refuses_missing_oracle() -> None:
    doc = _trace().to_dict()
    doc["expected_output"] = ""
    with pytest.raises(ValueError, match="expected_output"):
        TraceRecord.from_dict(doc)


def test_trace_record_refuses_bad_values() -> None:
    doc = _trace().to_dict()
    doc["outcome"] = "shrugged"
    with pytest.raises(ValueError, match="outcome"):
        TraceRecord.from_dict(doc)
    doc = _trace().to_dict()
    doc["duration_ms"] = -5
    with pytest.raises(ValueError, match="duration_ms"):
        TraceRecord.from_dict(doc)
    doc = _trace().to_dict()
    doc["duration_ms"] = True  # bool is an int subclass; must not slip through
    with pytest.raises(TypeError, match="duration_ms"):
        TraceRecord.from_dict(doc)
    doc = _trace().to_dict()
    doc["tool_calls"] = ["search"]  # entries must be mappings, not bare strings
    with pytest.raises(TypeError, match="tool call"):
        TraceRecord.from_dict(doc)


# ---------------------------------------------------------------------------
# Detectors: positive + negative controls per pattern
# ---------------------------------------------------------------------------


def test_repeated_tool_calls_positive_and_negative() -> None:
    thresholds = MineThresholds()
    repeated = _trace(
        tool_calls=[{"name": "search"}, {"name": "search"}, {"name": "search"}]
    )
    assert detect_repeated_tool_calls(repeated, thresholds) is not None
    assert "search" in (detect_repeated_tool_calls(repeated, thresholds) or "")
    below = _trace(tool_calls=[{"name": "search"}, {"name": "search"}])
    assert detect_repeated_tool_calls(below, thresholds) is None
    distinct = _trace(
        tool_calls=[{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}]
    )
    assert detect_repeated_tool_calls(distinct, thresholds) is None


def test_misrouting_positive_and_negative() -> None:
    misrouted = _trace(
        tool_calls=[
            {"name": "lookup", "expected": "lookup"},
            {"name": "cancel", "expected": "lookup"},
        ]
    )
    evidence = detect_misrouting(misrouted)
    assert evidence is not None and "cancel" in evidence and "lookup" in evidence
    clean = _trace(
        tool_calls=[
            {"name": "lookup", "expected": "lookup"},
            {"name": "cancel", "expected": "cancel"},
        ]
    )
    assert detect_misrouting(clean) is None
    # No recorded intent anywhere: the miner refuses to guess.
    unknown = _trace(tool_calls=[{"name": "lookup"}, {"name": "cancel"}])
    assert detect_misrouting(unknown) is None


def test_step_limit_loop_needs_limit_and_repetition() -> None:
    loop = _trace(
        steps=10,
        step_limit=10,
        tool_calls=[{"name": "search"}, {"name": "search"}],
    )
    assert detect_step_limit_loop(loop, MineThresholds()) is not None
    under_limit = _trace(
        steps=9,
        step_limit=10,
        tool_calls=[{"name": "search"}, {"name": "search"}],
    )
    assert detect_step_limit_loop(under_limit, MineThresholds()) is None
    # Hit the limit with all-distinct tools: a long task, not a loop.
    distinct = _trace(
        steps=10,
        step_limit=10,
        tool_calls=[{"name": "a"}, {"name": "b"}, {"name": "c"}],
    )
    assert detect_step_limit_loop(distinct, MineThresholds()) is None


def test_schema_violation_positive_and_negative() -> None:
    assert detect_schema_violation(_trace(schema_violations=2)) is not None
    assert detect_schema_violation(_trace(schema_violations=0)) is None


def test_latency_spike_boundary() -> None:
    thresholds = MineThresholds(latency_ms=1000.0)
    assert detect_latency_spike(_trace(duration_ms=1000), thresholds) is not None
    assert detect_latency_spike(_trace(duration_ms=999), thresholds) is None


def test_frequent_escalation_needs_the_count() -> None:
    thresholds = MineThresholds(escalations_per_agent=3)
    records = [
        _trace(trace_id=f"t-{i}", incident_id=f"inc-{i}", outcome="escalated")
        for i in range(2)
    ]
    assert detect_frequent_escalation(records, thresholds) == []
    records.append(_trace(trace_id="t-2", incident_id="inc-2", outcome="escalated"))
    found = detect_frequent_escalation(records, thresholds)
    assert [r.trace_id for r, _ in found] == ["t-0", "t-1", "t-2"]
    assert "3 times" in found[0][1]
    # A successful trace from the same agent is not an escalation case.
    records.append(_trace(trace_id="t-3", incident_id="inc-3", outcome="success"))
    assert len(detect_frequent_escalation(records, thresholds)) == 3


def test_high_variance_needs_spread_and_group_size() -> None:
    thresholds = MineThresholds(variance_cv=0.5)
    flat = [_trace(trace_id=f"t-{i}", duration_ms=1000) for i in range(4)]
    assert detect_high_variance(flat, thresholds) == []
    spread = [
        _trace(trace_id="t-1", duration_ms=200),
        _trace(trace_id="t-2", duration_ms=800),
        _trace(trace_id="t-3", duration_ms=5000),
        _trace(trace_id="t-4", duration_ms=12000),
    ]
    found = detect_high_variance(spread, thresholds)
    assert len(found) == 1
    anchor, evidence = found[0]
    assert anchor.trace_id == "t-4"  # slowest trace anchors the case
    assert "t-1" in evidence and "t-4" in evidence
    # Two traces cannot establish variance, however far apart.
    assert detect_high_variance(spread[:2], thresholds) == []


# ---------------------------------------------------------------------------
# Mining: living property, determinism, provenance
# ---------------------------------------------------------------------------


def test_mine_sample_corpus_covers_every_pattern() -> None:
    load = load_corpus(_sample_corpus_path())
    assert len(load.records) == 14 and load.skipped == 0
    report = mine_traces(
        load.records,
        owner=OWNER,
        cadence="weekly",
        mined_at=MINED_AT,
        corpus_digest=load.digest,
    )
    assert len(report.cases) == 12
    patterns = {c.pattern for c in report.cases}
    assert patterns == {
        "repeated_tool_calls",
        "misrouting",
        "step_limit_loop",
        "schema_violation",
        "latency_spike",
        "cost_spike",
        "frequent_escalation",
        "high_variance",
    }
    # The clean trace mines nothing.
    assert not any(c.trace_id == "t-014" for c in report.cases)
    # Every case carries its provenance and the owner/cadence.
    for case in report.cases:
        assert case.case.id == f"mined:{case.pattern}:{case.trace_id}"
        assert case.incident_id.startswith("inc-")
        assert case.evidence
        assert case.owner == OWNER and case.cadence == "weekly"
        assert case.case.expected.exact == case.expected_output


def test_mine_is_deterministic_and_roundtrips() -> None:
    load = load_corpus(_sample_corpus_path())
    kwargs = {
        "owner": OWNER,
        "cadence": "weekly",
        "mined_at": MINED_AT,
        "corpus_digest": load.digest,
    }
    first = mine_traces(load.records, **kwargs)
    second = mine_traces(load.records, **kwargs)
    assert first.to_json() == second.to_json()
    assert MineReport.from_dict(json.loads(first.to_json())) == first


def test_mine_is_living_not_static() -> None:
    load = load_corpus(_sample_corpus_path())
    base = mine_traces(load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT)
    extended = list(load.records) + [
        _trace(
            trace_id="t-999",
            incident_id="inc-999",
            tool_calls=[{"name": "x"}, {"name": "x"}, {"name": "x"}, {"name": "x"}],
            outcome="failure",
            error="looped on x",
            expected_output="resolved",
        )
    ]
    grown = mine_traces(extended, owner=OWNER, cadence="weekly", mined_at=MINED_AT)
    assert len(grown.cases) == len(base.cases) + 1
    assert any(c.trace_id == "t-999" for c in grown.cases)
    assert {c.case.id for c in base.cases} < {c.case.id for c in grown.cases}


def test_mine_refuses_duplicate_trace_ids() -> None:
    records = [_trace(trace_id="t-1"), _trace(trace_id="t-1", incident_id="inc-2")]
    with pytest.raises(ValueError, match="duplicate trace_id"):
        mine_traces(records, owner=OWNER, cadence="weekly", mined_at=MINED_AT)


def test_mine_refuses_bad_cadence_and_naive_time() -> None:
    records = [_trace()]
    with pytest.raises(ValueError, match="cadence"):
        mine_traces(records, owner=OWNER, cadence="fortnightly", mined_at=MINED_AT)
    with pytest.raises(ValueError, match="timezone"):
        mine_traces(records, owner=OWNER, cadence="weekly", mined_at="2026-11-06T19:00:00")
    with pytest.raises(ValueError, match="owner"):
        mine_traces(records, owner="", cadence="weekly", mined_at=MINED_AT)


def test_load_corpus_skips_malformed_lines_and_counts_them(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"trace_id": "t-1", "incident_id": "inc-1", "agent": "a", '
        '"tool_calls": [], "steps": 1, "step_limit": 5, "duration_ms": 10, '
        '"cost_usd": 0.01, "outcome": "success", "schema_violations": 0, '
        '"expected_output": "ok"}\n'
        "not json at all\n"
        '{"trace_id": "", "incident_id": "inc-2"}\n'
        "\n",
        encoding="utf-8",
    )
    load = load_corpus(corpus)
    assert len(load.records) == 1
    assert load.skipped == 2
    assert load.total_lines == 3
    assert load.digest.startswith("sha256:")


def test_load_corpus_missing_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(TracebenchError, match="cannot read corpus"):
        load_corpus(tmp_path / "nope.jsonl")


def test_parse_cadence() -> None:
    assert parse_cadence("daily") == "daily"
    with pytest.raises(ValueError, match="cadence"):
        parse_cadence("Daily")


# ---------------------------------------------------------------------------
# Manifest: provenance gate + staleness
# ---------------------------------------------------------------------------


def test_manifest_build_and_staleness() -> None:
    load = load_corpus(_sample_corpus_path())
    report = mine_traces(load.records, owner=OWNER, cadence="daily", mined_at=MINED_AT)
    manifest = build_manifest(
        report.cases,
        benchmark_id="bench",
        owner=OWNER,
        cadence="daily",
        created_at=MINED_AT,
        last_mined_at=MINED_AT,
    )
    assert len(manifest.case_ids) == 12
    assert manifest.source_incidents == tuple(sorted({c.incident_id for c in report.cases}))
    assert BenchmarkManifest.from_dict(json.loads(manifest.to_json())) == manifest
    mined_at = datetime.fromisoformat(MINED_AT)
    assert not manifest.is_stale(mined_at + timedelta(hours=12))
    assert manifest.is_stale(mined_at + timedelta(days=2))
    with pytest.raises(ValueError, match="timezone"):
        manifest.is_stale(datetime(2026, 11, 8))


def test_manifest_refuses_hand_written_cases() -> None:
    from agenteval_bench.models import ExpectedOutput

    hand_written = EvalCase(
        id="hand-made", input="q", expected=ExpectedOutput(exact="a")
    )
    with pytest.raises(TypeError, match="non-mined case"):
        build_manifest(
            [hand_written],  # type: ignore[list-item]
            benchmark_id="bench",
            owner=OWNER,
            cadence="weekly",
        )


# ---------------------------------------------------------------------------
# Promotion records carry provenance (additive, existing shape unchanged)
# ---------------------------------------------------------------------------


def test_promotion_record_provenance_roundtrip() -> None:
    record = PromotionRecord(
        failure_id="t-1",
        kind="retry_trace",
        promoted_case_id="mined:repeated_tool_calls:t-1",
        prior_score=0.0,
        evaluator_notes="tool 'x' called 3 times",
        promoted_at=MINED_AT,
        provenance={"trace_id": "t-1", "incident_id": "inc-1", "pattern": "repeated_tool_calls"},
    )
    assert PromotionRecord.from_dict(record.to_dict()) == record
    legacy = PromotionRecord(
        failure_id="f-1",
        kind="escalation",
        promoted_case_id="prod:escalation:f-1",
        prior_score=0.0,
        evaluator_notes="",
        promoted_at=MINED_AT,
    )
    assert "provenance" not in legacy.to_dict()
    assert PromotionRecord.from_dict(legacy.to_dict()) == legacy
    bad = legacy.to_dict()
    bad["provenance"] = {"trace_id": 7}
    with pytest.raises(ValueError, match="provenance"):
        PromotionRecord.from_dict(bad)


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------


def _report(verdict: str, regressed: list[str], reasons: list[str]) -> ComparisonReport:
    return ComparisonReport(
        suite_name="bench",
        version_id="1",
        digest="sha256:x",
        candidate_agent="candidate",
        seed=42,
        baseline_pass_rate=1.0,
        candidate_pass_rate=0.0 if regressed else 1.0,
        case_deltas=(),
        regressed_cases=tuple(regressed),
        verdict=verdict,  # type: ignore[arg-type]
        reasons=tuple(reasons),
    )


def test_dispatch_passes_silently() -> None:
    assert dispatch_actions(
        _report("GATE_PASS", [], []), benchmark_id="bench/1", run_id="1"
    ) == ()


def test_dispatch_gate_fail_triggers_gate_and_investigation() -> None:
    actions = dispatch_actions(
        _report("GATE_FAIL", ["mined:repeated_tool_calls:t-001"], ["per-case regressions"]),
        benchmark_id="bench/1",
        run_id="1",
    )
    assert [a.action for a in actions] == ["gate", "investigation"]
    assert actions[1].case_ids == ("mined:repeated_tool_calls:t-001",)
    assert "t-001" in actions[1].reason


def test_dispatch_rollback_on_third_consecutive_failure() -> None:
    report = _report("GATE_FAIL", ["mined:misrouting:t-002"], ["per-case regressions"])
    assert [a.action for a in dispatch_actions(
        report, benchmark_id="bench/1", run_id="1", prior_consecutive_failures=1
    )] == ["gate", "investigation"]
    actions = dispatch_actions(
        report, benchmark_id="bench/1", run_id="1", prior_consecutive_failures=2
    )
    assert [a.action for a in actions] == ["gate", "investigation", "rollback"]
    assert "3 consecutive" in actions[2].reason
    with pytest.raises(ValueError, match="prior_consecutive_failures"):
        dispatch_actions(report, benchmark_id="bench/1", run_id="1",
                         prior_consecutive_failures=-1)


def test_dispatch_aggregate_only_failure_still_investigates() -> None:
    actions = dispatch_actions(
        _report("GATE_FAIL", [], ["aggregate pass_rate 50.0% below threshold 100.0%"]),
        benchmark_id="bench/1",
        run_id="1",
    )
    assert [a.action for a in actions] == ["gate", "investigation"]
    assert actions[1].case_ids == ()
    assert "aggregate" in actions[1].reason


# ---------------------------------------------------------------------------
# End to end: mine -> publish -> gate -> dispatch, in a real store
# ---------------------------------------------------------------------------


def _oracle_outputs(report: MineReport) -> dict[str, str]:
    return {c.case.id: c.expected_output for c in report.cases}


def test_required_baseline_pins_mined_cases_at_one() -> None:
    load = load_corpus(_sample_corpus_path())
    report = mine_traces(load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT)
    baseline = required_baseline(
        None,
        suite_cases=[c.case for c in report.cases],
        mined_cases=list(report.cases),
        seed=42,
    )
    assert all(v == 1.0 for v in baseline.per_case.values())
    assert baseline.pass_rate == 1.0
    assert baseline.agent == "tracebench-required"


def test_scheduled_run_pass_then_fail_then_rollback(tmp_path: Path) -> None:
    corpus = _sample_corpus_path()
    store = tmp_path / "store"
    out1 = tmp_path / "run1"
    load = load_corpus(corpus)
    report = mine_traces(
        load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    oracle = _oracle_outputs(report)

    # Run 1: oracle candidate passes every mined case -> GATE_PASS, v1 published.
    first = run_scheduled(
        corpus=corpus,
        store=store,
        suite_name="bench",
        version_id="1",
        prev_version_id=None,
        candidate_outputs=oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=out1,
        mined_at=MINED_AT,
    )
    assert first.verdict == "GATE_PASS"
    assert first.published_new_version is True
    assert first.new_cases == 12 and first.mined_cases == 12
    assert first.actions == () and first.consecutive_failures == 0
    assert (out1 / "actions.jsonl").read_text(encoding="utf-8") == ""
    assert (out1 / "manifest.json").exists()
    version, suite = SnapshotStore(store).load("bench", "1")
    assert len(suite.cases) == 12
    promotions = version.promotions
    assert len(promotions) == 12
    assert all("provenance" in p for p in promotions)
    assert promotions[0]["provenance"]["incident_id"].startswith("inc-")

    # Run 2: same corpus, degraded candidate -> no new cases, gate against
    # v1 -> GATE_FAIL with gate + investigation, streak 1.
    degraded = {case_id: "wrong answer" for case_id in oracle}
    out2 = tmp_path / "run2"
    second = run_scheduled(
        corpus=corpus,
        store=store,
        suite_name="bench",
        version_id="2",
        prev_version_id="1",
        candidate_outputs=degraded,
        owner=OWNER,
        cadence="weekly",
        out_dir=out2,
        mined_at=MINED_AT,
    )
    assert second.verdict == "GATE_FAIL"
    assert second.published_new_version is False  # steady state: nothing new
    assert second.version_id == "1"
    assert second.new_cases == 0
    assert [a.action for a in second.actions] == ["gate", "investigation"]
    assert second.consecutive_failures == 1
    ledger = (out2 / "actions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger) == 2
    assert json.loads(ledger[0])["action"] == "gate"

    # Run 3: third consecutive failure (prior streak 2) -> rollback appears.
    out3 = tmp_path / "run3"
    third = run_scheduled(
        corpus=corpus,
        store=store,
        suite_name="bench",
        version_id="3",
        prev_version_id="1",
        candidate_outputs=degraded,
        owner=OWNER,
        cadence="weekly",
        out_dir=out3,
        mined_at=MINED_AT,
        prior_consecutive_failures=2,
    )
    assert third.verdict == "GATE_FAIL"
    assert [a.action for a in third.actions] == ["gate", "investigation", "rollback"]
    assert third.consecutive_failures == 3


def test_scheduled_run_with_nothing_to_regress(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    record = _trace()
    clean.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
    with pytest.raises(TracebenchError, match="nothing to regress"):
        run_scheduled(
            corpus=clean,
            store=tmp_path / "store",
            suite_name="bench",
            version_id="1",
            prev_version_id=None,
            candidate_outputs={},
            owner=OWNER,
            cadence="weekly",
            out_dir=tmp_path / "out",
            mined_at=MINED_AT,
        )


def test_scheduled_run_refuses_duplicate_version(tmp_path: Path) -> None:
    corpus = _sample_corpus_path()
    store = tmp_path / "store"
    load = load_corpus(corpus)
    report = mine_traces(load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT)
    kwargs = {
        "corpus": corpus,
        "store": store,
        "suite_name": "bench",
        "version_id": "1",
        "prev_version_id": None,
        "candidate_outputs": _oracle_outputs(report),
        "owner": OWNER,
        "cadence": "weekly",
        "mined_at": MINED_AT,
    }
    run_scheduled(out_dir=tmp_path / "run1", **kwargs)
    with pytest.raises(TracebenchError, match="cannot publish mined version"):
        run_scheduled(out_dir=tmp_path / "run2", **kwargs)


def test_publish_mined_version_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    load = load_corpus(_sample_corpus_path())
    report = mine_traces(load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT)
    from agenteval_bench.models import EvalSuite
    from agenteval_bench.snapshots import SnapshotError, VersionMeta

    store = SnapshotStore(tmp_path / "store")
    base = EvalSuite(name="bench", version=1, cases=[report.cases[0].case])
    baseline = required_baseline(
        None,
        suite_cases=[c.case for c in report.cases],
        mined_cases=list(report.cases),
        seed=42,
    )
    with pytest.raises(SnapshotError, match="duplicate case id"):
        publish_mined_version(
            store,
            base,
            "1",
            VersionMeta(owner=OWNER, why="x", cadence="weekly"),
            mined_cases=list(report.cases),
            baseline=baseline,
        )

# ---------------------------------------------------------------------------
# CLI: tracebench mine | regress (same _run contract as test_cli.py)
# ---------------------------------------------------------------------------


def _run_cli(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> tuple[int, str, str]:
    from agenteval_bench import cli

    monkeypatch.setattr("sys.argv", ["agenteval-bench", *argv])
    try:
        cli.main()
        captured = capsys.readouterr()
        return 0, captured.out, captured.err
    except SystemExit as e:
        captured = capsys.readouterr()
        return e.code or 0, captured.out, captured.err


def test_cli_mine_writes_manifest_cases_and_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "mine-out"
    code, stdout, _ = _run_cli(
        [
            "tracebench", "mine",
            "--corpus", str(_sample_corpus_path()),
            "--owner", OWNER,
            "--cadence", "weekly",
            "--out", str(out),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert "Mined 12 case(s)" in stdout
    manifest = BenchmarkManifest.from_dict(
        json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    )
    assert manifest.owner == OWNER and manifest.cadence == "weekly"
    assert len(manifest.case_ids) == 12
    mined = MineReport.from_dict(
        json.loads((out / "mined-cases.json").read_text(encoding="utf-8"))
    )
    assert len(mined.cases) == 12
    oracle = json.loads((out / "oracle-outputs.json").read_text(encoding="utf-8"))
    assert len(oracle) == 12


def test_cli_mine_rejects_unknown_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    code, _, stderr = _run_cli(
        [
            "tracebench", "mine",
            "--corpus", str(_sample_corpus_path()),
            "--owner", OWNER,
            "--cadence", "fortnightly",
            "--out", str(tmp_path / "out"),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 1
    assert "cadence" in stderr


def test_cli_regress_pass_and_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    mine_out = tmp_path / "mine-out"
    code, _, _ = _run_cli(
        [
            "tracebench", "mine",
            "--corpus", str(_sample_corpus_path()),
            "--owner", OWNER,
            "--cadence", "weekly",
            "--out", str(mine_out),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0
    oracle_path = str(mine_out / "oracle-outputs.json")
    store = tmp_path / "store"
    code, stdout, _ = _run_cli(
        [
            "tracebench", "regress",
            "--corpus", str(_sample_corpus_path()),
            "--store", str(store),
            "--suite", "bench",
            "--version", "1",
            "--candidate-outputs", oracle_path,
            "--owner", OWNER,
            "--cadence", "weekly",
            "--out", str(tmp_path / "run1"),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert "verdict: GATE_PASS" in stdout
    degraded = {
        case_id: "wrong answer"
        for case_id in json.loads(
            (mine_out / "oracle-outputs.json").read_text(encoding="utf-8")
        )
    }
    degraded_path = tmp_path / "degraded.json"
    degraded_path.write_text(json.dumps(degraded), encoding="utf-8")
    code, stdout, _ = _run_cli(
        [
            "tracebench", "regress",
            "--corpus", str(_sample_corpus_path()),
            "--store", str(store),
            "--suite", "bench",
            "--version", "2",
            "--prev", "1",
            "--candidate-outputs", str(degraded_path),
            "--owner", OWNER,
            "--cadence", "weekly",
            "--out", str(tmp_path / "run2"),
            "--prior-failures", "2",
        ],
        monkeypatch,
        capsys,
    )
    assert code == 2
    assert "verdict: GATE_FAIL" in stdout
    assert "action: rollback" in stdout


def test_cli_unknown_subcommand(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    code, _, stderr = _run_cli(["tracebench", "explode"], monkeypatch, capsys)
    assert code == 1
    assert "unknown tracebench subcommand" in stderr
