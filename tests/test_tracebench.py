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

from agenteval_bench.models import EvalCase, EvalSuite
from agenteval_bench.regression import ComparisonReport, PromotionRecord
from agenteval_bench.snapshots import SnapshotStore, VersionMeta
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


# ---------------------------------------------------------------------------
# Closed loop (fix-forward on the rebased head): state-chained runs,
# update/retire propagation, staleness observed, malformed state refused.
# These are the adversarial tests for the architecture/product FAILs:
# the scheduled deployment must chain versions, carry the failure
# streak, and publish benchmark changes — with no operator hand-feeding.
# ---------------------------------------------------------------------------


def _corpus_with_modified_t001(tmp_path: Path, *, change_oracle: bool) -> Path:
    """Sample corpus with t-001's oracle changed (update path) or its
    repeated-tool-calls pattern removed (retire path)."""
    lines = [
        json.loads(line)
        for line in _sample_corpus_path().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in lines:
        if record["trace_id"] == "t-001":
            if change_oracle:
                record["expected_output"] = "escalated with the docs link (fixed)"
            else:
                record["tool_calls"] = [
                    {"name": "search_docs"},
                    {"name": "fetch_page"},
                    {"name": "summarize"},
                    {"name": "reply"},
                ]
    path = tmp_path / "modified.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in lines), encoding="utf-8"
    )
    return path


def _first_run(tmp_path: Path, state_dir: Path | None = None) -> tuple[Path, dict[str, str]]:
    """Publish v1 from the sample corpus with the oracle candidate."""
    corpus = _sample_corpus_path()
    store = tmp_path / "store"
    load = load_corpus(corpus)
    report = mine_traces(
        load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    oracle = _oracle_outputs(report)
    first = run_scheduled(
        corpus=corpus,
        store=store,
        suite_name="bench",
        version_id="1",
        prev_version_id=None,
        candidate_outputs=oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "run1",
        mined_at=MINED_AT,
        state_dir=state_dir,
    )
    assert first.verdict == "GATE_PASS"
    return store, oracle


def test_state_dir_chains_three_weeks_to_rollback(tmp_path: Path) -> None:
    """No --prev, no --prior-failures, no --version: the state file alone
    chains the version and the streak, and rollback fires on week 3."""
    corpus = _sample_corpus_path()
    store = tmp_path / "store"
    state_dir = tmp_path / "state"
    load = load_corpus(corpus)
    report = mine_traces(
        load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    degraded = {c.case.id: "wrong answer" for c in report.cases}
    actions_seen: list[list[str]] = []
    for week in (1, 2, 3):
        result = run_scheduled(
            corpus=corpus,
            store=store,
            suite_name="bench",
            version_id="",
            prev_version_id=None,
            candidate_outputs=degraded,
            owner=OWNER,
            cadence="weekly",
            out_dir=tmp_path / f"week{week}",
            mined_at=MINED_AT,
            state_dir=state_dir,
        )
        assert result.verdict == "GATE_FAIL"
        assert result.version_id == "1"  # steady state after week 1
        # Week 1 publishes v1 (first run); weeks 2-3 are steady state.
        assert result.published_new_version is (week == 1)
        assert result.consecutive_failures == week
        actions_seen.append([a.action for a in result.actions])
        state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
        assert state["suite_name"] == "bench"
        assert state["prev_version_id"] == "1"
        assert state["consecutive_failures"] == week
    assert actions_seen[0] == ["gate", "investigation"]
    assert actions_seen[1] == ["gate", "investigation"]
    assert actions_seen[2] == ["gate", "investigation", "rollback"]
    # A passing week resets the streak in the state file.
    oracle = _oracle_outputs(report)
    reset = run_scheduled(
        corpus=corpus,
        store=store,
        suite_name="bench",
        version_id="",
        prev_version_id=None,
        candidate_outputs=oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "week4",
        mined_at=MINED_AT,
        state_dir=state_dir,
    )
    assert reset.verdict == "GATE_PASS"
    assert reset.consecutive_failures == 0
    state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    assert state["consecutive_failures"] == 0


def test_state_dir_auto_derives_next_version(tmp_path: Path) -> None:
    """A new case in week 2 publishes version 2 without --version."""
    corpus = _sample_corpus_path()
    store = tmp_path / "store"
    state_dir = tmp_path / "state"
    load = load_corpus(corpus)
    report = mine_traces(
        load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    oracle = _oracle_outputs(report)
    first = run_scheduled(
        corpus=corpus,
        store=store,
        suite_name="bench",
        version_id="",
        prev_version_id=None,
        candidate_outputs=oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "run1",
        mined_at=MINED_AT,
        state_dir=state_dir,
    )
    assert first.version_id == "1"
    # Append a new failing trace: a 13th case must be mined next run.
    extra = _trace(
        trace_id="t-099",
        incident_id="inc-199",
        tool_calls=[{"name": "search_docs"}] * 4,
        outcome="failure",
        expected_output="escalated to a human",
    )
    lines = _sample_corpus_path().read_text(encoding="utf-8")
    corpus2 = tmp_path / "corpus2.jsonl"
    corpus2.write_text(lines + json.dumps(extra.to_dict()) + "\n", encoding="utf-8")
    load2 = load_corpus(corpus2)
    report2 = mine_traces(
        load2.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    oracle2 = _oracle_outputs(report2)
    assert len(report2.cases) == 13
    second = run_scheduled(
        corpus=corpus2,
        store=store,
        suite_name="bench",
        version_id="",
        prev_version_id=None,
        candidate_outputs=oracle2,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "run2",
        mined_at=MINED_AT,
        state_dir=state_dir,
    )
    assert second.version_id == "2"
    assert second.published_new_version is True
    assert second.new_cases == 1
    _, suite = SnapshotStore(store).load("bench", "2")
    assert len(suite.cases) == 13


def test_changed_oracle_publishes_update_and_catches_stale_agent(
    tmp_path: Path,
) -> None:
    """A changed expected_output for an existing case id publishes an
    update — and the run FAILS on drift even though the candidate passes
    the new version (the benchmark moved, so last week's certification
    no longer applies). The previous candidate's outputs then also FAIL
    the new version, so the living benchmark catches a stale agent."""
    store, oracle = _first_run(tmp_path)
    changed = _corpus_with_modified_t001(tmp_path, change_oracle=True)
    load = load_corpus(changed)
    report = mine_traces(
        load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    assert len(report.cases) == 12
    new_oracle = _oracle_outputs(report)
    updated_id = "mined:repeated_tool_calls:t-001"
    assert new_oracle[updated_id] != oracle[updated_id]
    second = run_scheduled(
        corpus=changed,
        store=store,
        suite_name="bench",
        version_id="2",
        prev_version_id="1",
        candidate_outputs=new_oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "run2",
        mined_at=MINED_AT,
    )
    assert second.verdict == "GATE_FAIL"
    assert second.drift_detected is True
    assert second.drifted_case_ids == (updated_id,)
    assert second.published_new_version is True
    assert second.version_id == "2"
    assert second.updated_cases == 1
    assert second.updated_case_ids == (updated_id,)
    assert second.retired_cases == 0
    assert [a.action for a in second.actions] == ["gate", "investigation"]
    assert second.consecutive_failures == 1
    # The OLD oracle outputs (stale agent) against the new version fail.
    third = run_scheduled(
        corpus=changed,
        store=store,
        suite_name="bench",
        version_id="3",
        prev_version_id="2",
        candidate_outputs=oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "run3",
        mined_at=MINED_AT,
    )
    assert third.verdict == "GATE_FAIL"
    assert third.published_new_version is False  # steady state on v2
    report_doc = json.loads(
        (tmp_path / "run3" / "report.json").read_text(encoding="utf-8")
    )
    assert report_doc["comparison"]["regressed_cases"] == [updated_id]


def test_vanished_pattern_retires_case(tmp_path: Path) -> None:
    """A trace whose pattern left the corpus retires its case from the
    new version, and the pinned baseline score goes with it."""
    store, _oracle = _first_run(tmp_path)
    changed = _corpus_with_modified_t001(tmp_path, change_oracle=False)
    load = load_corpus(changed)
    report = mine_traces(
        load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    assert len(report.cases) == 11
    new_oracle = _oracle_outputs(report)
    retired_id = "mined:repeated_tool_calls:t-001"
    assert retired_id not in new_oracle
    second = run_scheduled(
        corpus=changed,
        store=store,
        suite_name="bench",
        version_id="2",
        prev_version_id="1",
        candidate_outputs=new_oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "run2",
        mined_at=MINED_AT,
    )
    assert second.verdict == "GATE_PASS"
    assert second.published_new_version is True
    assert second.retired_cases == 1
    assert second.retired_case_ids == (retired_id,)
    version, suite = SnapshotStore(store).load("bench", "2")
    assert len(suite.cases) == 11
    assert retired_id not in version.baseline.per_case
    assert retired_id not in {c.id for c in suite.cases}


def test_empty_mine_never_retires(tmp_path: Path) -> None:
    """A corpus that mines nothing must not wipe the benchmark: the run
    gates the previous version and publishes nothing."""
    store, oracle = _first_run(tmp_path)
    clean = tmp_path / "clean.jsonl"
    clean.write_text(json.dumps(_trace().to_dict()) + "\n", encoding="utf-8")
    result = run_scheduled(
        corpus=clean,
        store=store,
        suite_name="bench",
        version_id="2",
        prev_version_id="1",
        candidate_outputs=oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "run2",
        mined_at=MINED_AT,
    )
    assert result.verdict == "GATE_PASS"
    assert result.published_new_version is False
    assert result.retired_cases == 0
    assert result.version_id == "1"


def test_previous_stale_is_observed(tmp_path: Path) -> None:
    """A previous version older than its cadence is reported stale on
    the next run instead of living only in the manifest type."""
    store, oracle = _first_run(tmp_path)
    eight_days_later = "2026-11-14T19:00:00+00:00"
    result = run_scheduled(
        corpus=_sample_corpus_path(),
        store=store,
        suite_name="bench",
        version_id="2",
        prev_version_id="1",
        candidate_outputs=oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "run2",
        mined_at=eight_days_later,
    )
    assert result.previous_stale is True
    report_doc = json.loads(
        (tmp_path / "run2" / "report.json").read_text(encoding="utf-8")
    )
    assert report_doc["previous_stale"] is True


def test_malformed_run_state_fails_closed(tmp_path: Path) -> None:
    """A corrupt, cross-suite, or negative-streak state file must never
    silently restart the version chain or the failure streak."""
    corpus = _sample_corpus_path()
    store = tmp_path / "store"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    base_kwargs = {
        "corpus": corpus,
        "store": store,
        "suite_name": "bench",
        "version_id": "",
        "prev_version_id": None,
        "candidate_outputs": {},
        "owner": OWNER,
        "cadence": "weekly",
        "out_dir": tmp_path / "out",
        "mined_at": MINED_AT,
        "state_dir": state_dir,
    }
    (state_dir / "state.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(TracebenchError, match="cannot read run state"):
        run_scheduled(**base_kwargs)
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "suite_name": "other-suite",
                "prev_version_id": "1",
                "consecutive_failures": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TracebenchError, match="not .bench."):
        run_scheduled(**base_kwargs)
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "suite_name": "bench",
                "prev_version_id": "1",
                "consecutive_failures": -1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TracebenchError, match="consecutive_failures"):
        run_scheduled(**base_kwargs)


def test_non_numeric_prev_refuses_auto_version(tmp_path: Path) -> None:
    """Auto-deriving the next version from a non-numeric previous id
    fails closed instead of guessing a version scheme."""
    corpus = _sample_corpus_path()
    store = tmp_path / "store"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "suite_name": "bench",
                "prev_version_id": "v1.2",
                "consecutive_failures": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TracebenchError, match="cannot auto-derive"):
        run_scheduled(
            corpus=corpus,
            store=store,
            suite_name="bench",
            version_id="",
            prev_version_id=None,
            candidate_outputs={},
            owner=OWNER,
            cadence="weekly",
            out_dir=tmp_path / "out",
            mined_at=MINED_AT,
            state_dir=state_dir,
        )


def test_cli_regress_state_dir_closes_loop_without_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The CLI wires --state-dir through: three weekly regress calls with
    no --version and no --prior-failures end in rollback on week 3."""
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
    oracle = json.loads(
        (mine_out / "oracle-outputs.json").read_text(encoding="utf-8")
    )
    degraded_path = tmp_path / "degraded.json"
    degraded_path.write_text(
        json.dumps({k: "wrong answer" for k in oracle}), encoding="utf-8"
    )
    store = tmp_path / "store"
    state_dir = tmp_path / "state"
    for week in (1, 2, 3):
        code, stdout, _ = _run_cli(
            [
                "tracebench", "regress",
                "--corpus", str(_sample_corpus_path()),
                "--store", str(store),
                "--suite", "bench",
                "--candidate-outputs", str(degraded_path),
                "--owner", OWNER,
                "--cadence", "weekly",
                "--state-dir", str(state_dir),
                "--out", str(tmp_path / f"run{week}"),
            ],
            monkeypatch,
            capsys,
        )
        assert code == 2
        assert "verdict: GATE_FAIL" in stdout
        assert f"consecutive failures: {week}" in stdout
    assert "action: rollback" in stdout
    state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    assert state["consecutive_failures"] == 3


def test_cli_regress_needs_version_without_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    code, _, stderr = _run_cli(
        [
            "tracebench", "regress",
            "--corpus", str(_sample_corpus_path()),
            "--store", str(tmp_path / "store"),
            "--suite", "bench",
            "--candidate-outputs", str(tmp_path / "none.json"),
            "--owner", OWNER,
            "--out", str(tmp_path / "out"),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 1
    assert "--version" in stderr


# ---------------------------------------------------------------------------
# Fix-forward round 2 (on head bf258190): the drift gate, provenance-
# scoped retirement, state-write hardening, corpus_kind labeling, and
# the workflow effector/alarm. The scheduled run must be a regression
# that can fail on its own — not a smoke test that reports green forever.
# ---------------------------------------------------------------------------


def _seeded_store(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A store whose v1 holds the 12 mined cases PLUS one hand-seeded
    base case (``seed:hand-written``) with a pinned baseline of 1.0."""
    from agenteval_bench.models import CostBound
    from agenteval_bench.snapshots import BaselineRecord

    corpus = _sample_corpus_path()
    store = tmp_path / "store"
    load = load_corpus(corpus)
    report = mine_traces(
        load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    oracle = _oracle_outputs(report)
    seed_case = EvalCase(
        id="seed:hand-written",
        input="seeded base case",
        expected=report.cases[0].case.expected,
    )
    # The seed case's expected output is the first mined case's oracle
    # string (its expected block is copied from that case), so the
    # oracle candidate must answer it too.
    oracle["seed:hand-written"] = report.cases[0].expected_output
    base_suite = EvalSuite(
        name="bench",
        version=1,
        cases=[seed_case],
        cost_bound=CostBound(),
    )
    # The seed case is a base case, so its pinned score comes from the
    # base baseline, not from the mine.
    seed_baseline = BaselineRecord(
        agent="seed",
        seed=42,
        pass_rate=1.0,
        total=1,
        passed=1,
        failed=0,
        skipped=0,
        per_case={"seed:hand-written": 1.0},
    )
    baseline = required_baseline(
        seed_baseline,
        suite_cases=[seed_case, *[c.case for c in report.cases]],
        mined_cases=list(report.cases),
        seed=42,
    )
    publish_mined_version(
        SnapshotStore(store),
        base_suite,
        "1",
        VersionMeta(owner=OWNER, why="seeded v1", cadence="weekly"),
        mined_cases=list(report.cases),
        baseline=baseline,
        suite_cases=[seed_case, *[c.case for c in report.cases]],
    )
    return store, oracle


def test_base_case_survives_unchanged_run(tmp_path: Path) -> None:
    """Retirement is scoped to mining provenance: a hand-seeded base
    case is carried forward untouched by an unchanged-corpus run."""
    store, oracle = _seeded_store(tmp_path)
    result = run_scheduled(
        corpus=_sample_corpus_path(),
        store=store,
        suite_name="bench",
        version_id="2",
        prev_version_id="1",
        candidate_outputs=oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "run2",
        mined_at=MINED_AT,
    )
    assert result.verdict == "GATE_PASS"
    assert result.published_new_version is False  # steady state
    assert result.retired_cases == 0
    _, suite = SnapshotStore(store).load("bench", "1")
    assert "seed:hand-written" in {c.id for c in suite.cases}


def test_base_case_survives_retire_run(tmp_path: Path) -> None:
    """When a mined pattern genuinely vanishes, only that mined case
    retires — the hand-seeded base case survives into the new version."""
    store, oracle = _seeded_store(tmp_path)
    changed = _corpus_with_modified_t001(tmp_path, change_oracle=False)
    load = load_corpus(changed)
    report = mine_traces(
        load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    new_oracle = _oracle_outputs(report)
    new_oracle["seed:hand-written"] = oracle["seed:hand-written"]
    retired_id = "mined:repeated_tool_calls:t-001"
    result = run_scheduled(
        corpus=changed,
        store=store,
        suite_name="bench",
        version_id="2",
        prev_version_id="1",
        candidate_outputs=new_oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "run2",
        mined_at=MINED_AT,
    )
    assert result.verdict == "GATE_PASS"
    assert result.published_new_version is True
    assert result.retired_cases == 1
    assert result.retired_case_ids == (retired_id,)
    _, suite = SnapshotStore(store).load("bench", "2")
    case_ids = {c.id for c in suite.cases}
    assert retired_id not in case_ids
    assert "seed:hand-written" in case_ids
    assert len(suite.cases) == 12  # 11 mined + 1 base


def test_drift_gate_fails_run_on_oracle_change(tmp_path: Path) -> None:
    """The drift gate: a changed oracle fails the run even though the
    candidate passes the new version, and the report names the drift."""
    store, _oracle = _first_run(tmp_path)
    changed = _corpus_with_modified_t001(tmp_path, change_oracle=True)
    load = load_corpus(changed)
    report = mine_traces(
        load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    new_oracle = _oracle_outputs(report)
    result = run_scheduled(
        corpus=changed,
        store=store,
        suite_name="bench",
        version_id="2",
        prev_version_id="1",
        candidate_outputs=new_oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "run2",
        mined_at=MINED_AT,
    )
    assert result.verdict == "GATE_FAIL"
    assert result.drift_detected is True
    assert result.drifted_case_ids == ("mined:repeated_tool_calls:t-001",)
    assert result.published_new_version is True  # benchmark still moves
    assert result.version_id == "2"
    assert [a.action for a in result.actions] == ["gate", "investigation"]
    assert result.consecutive_failures == 1
    report_doc = json.loads(
        (tmp_path / "run2" / "report.json").read_text(encoding="utf-8")
    )
    assert report_doc["drift"]["detected"] is True
    assert report_doc["drift"]["drifted_case_ids"] == [
        "mined:repeated_tool_calls:t-001"
    ]
    assert "drift" in report_doc["comparison"]["reasons"][0]


def test_drift_flag_self_clears_next_unchanged_week(tmp_path: Path) -> None:
    """A drift failure is a one-week flag: the next run over the same
    changed corpus passes and the streak resets."""
    store, _oracle = _first_run(tmp_path)
    changed = _corpus_with_modified_t001(tmp_path, change_oracle=True)
    load = load_corpus(changed)
    report = mine_traces(
        load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    new_oracle = _oracle_outputs(report)
    failed = run_scheduled(
        corpus=changed,
        store=store,
        suite_name="bench",
        version_id="2",
        prev_version_id="1",
        candidate_outputs=new_oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "run2",
        mined_at=MINED_AT,
    )
    assert failed.verdict == "GATE_FAIL"
    cleared = run_scheduled(
        corpus=changed,
        store=store,
        suite_name="bench",
        version_id="3",
        prev_version_id="2",
        candidate_outputs=new_oracle,
        owner=OWNER,
        cadence="weekly",
        out_dir=tmp_path / "run3",
        mined_at=MINED_AT,
    )
    assert cleared.verdict == "GATE_PASS"
    assert cleared.drift_detected is False
    assert cleared.consecutive_failures == 0
    assert cleared.published_new_version is False


def test_three_consecutive_drift_weeks_reach_rollback(tmp_path: Path) -> None:
    """Three weeks of continuous benchmark movement reach rollback
    through the drift gate alone, with no degraded candidate anywhere."""
    state_dir = tmp_path / "state"
    store, oracle = _first_run(tmp_path, state_dir=state_dir)
    current_corpus = _sample_corpus_path()
    current_oracle = oracle
    for week in (1, 2, 3):
        lines = [
            json.loads(line)
            for line in current_corpus.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for record in lines:
            if record["trace_id"] == "t-001":
                record["expected_output"] = (
                    f"escalated with the docs link (week {week})"
                )
        week_corpus = tmp_path / f"corpus-week{week}.jsonl"
        week_corpus.write_text(
            "".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8"
        )
        load = load_corpus(week_corpus)
        report = mine_traces(
            load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
        )
        current_oracle = _oracle_outputs(report)
        result = run_scheduled(
            corpus=week_corpus,
            store=store,
            suite_name="bench",
            version_id="",
            prev_version_id=None,
            candidate_outputs=current_oracle,
            owner=OWNER,
            cadence="weekly",
            out_dir=tmp_path / f"week{week}",
            mined_at=MINED_AT,
            state_dir=state_dir,
        )
        assert result.verdict == "GATE_FAIL"
        assert result.drift_detected is True
        assert result.consecutive_failures == week
        current_corpus = week_corpus
    assert [a.action for a in result.actions] == [
        "gate",
        "investigation",
        "rollback",
    ]


def test_drift_via_exact_workflow_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The deployment configuration itself can fail: the exact workflow
    mine + regress commands, oracle candidate, state-dir chained. Week 1
    passes; week 2 with a changed oracle fails on drift (exit 2)."""
    mine_out = tmp_path / "mine-out"
    code, _, _ = _run_cli(
        [
            "tracebench", "mine",
            "--corpus", str(_sample_corpus_path()),
            "--owner", OWNER,
            "--cadence", "weekly",
            "--corpus-kind", "sample",
            "--out", str(mine_out),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0
    store = tmp_path / "store"
    state_dir = tmp_path / "state"
    code, stdout, _ = _run_cli(
        [
            "tracebench", "regress",
            "--corpus", str(_sample_corpus_path()),
            "--store", str(store),
            "--suite", "bench",
            "--candidate-outputs", str(mine_out / "oracle-outputs.json"),
            "--owner", OWNER,
            "--cadence", "weekly",
            "--corpus-kind", "sample",
            "--state-dir", str(state_dir),
            "--out", str(tmp_path / "run1"),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert "verdict: GATE_PASS" in stdout
    changed = _corpus_with_modified_t001(tmp_path, change_oracle=True)
    mine_out2 = tmp_path / "mine-out2"
    code, _, _ = _run_cli(
        [
            "tracebench", "mine",
            "--corpus", str(changed),
            "--owner", OWNER,
            "--cadence", "weekly",
            "--corpus-kind", "sample",
            "--out", str(mine_out2),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0
    code, stdout, _ = _run_cli(
        [
            "tracebench", "regress",
            "--corpus", str(changed),
            "--store", str(store),
            "--suite", "bench",
            "--candidate-outputs", str(mine_out2 / "oracle-outputs.json"),
            "--owner", OWNER,
            "--cadence", "weekly",
            "--corpus-kind", "sample",
            "--state-dir", str(state_dir),
            "--out", str(tmp_path / "run2"),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 2
    assert "verdict: GATE_FAIL" in stdout
    assert "benchmark drift:" in stdout
    assert "action: investigation" in stdout


def test_state_dir_pointing_at_file_fails_before_publish(
    tmp_path: Path,
) -> None:
    """A state-dir that cannot be written fails closed BEFORE anything
    is published — no half-done run, no recovery trap on the next run."""
    corpus = _sample_corpus_path()
    store = tmp_path / "store"
    state_file = tmp_path / "state-file"
    state_file.write_text("not a directory", encoding="utf-8")
    load = load_corpus(corpus)
    report = mine_traces(
        load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    with pytest.raises(TracebenchError, match="not writable"):
        run_scheduled(
            corpus=corpus,
            store=store,
            suite_name="bench",
            version_id="1",
            prev_version_id=None,
            candidate_outputs=_oracle_outputs(report),
            owner=OWNER,
            cadence="weekly",
            out_dir=tmp_path / "out",
            mined_at=MINED_AT,
            state_dir=state_file,
        )
    # Nothing was published: the store directory was never created.
    assert not (store).exists()


def test_cli_state_dir_file_is_clean_error_not_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Same failure through the real CLI: exit 1, one clean Error line,
    no traceback on stderr."""
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
    state_file = tmp_path / "state-file"
    state_file.write_text("not a directory", encoding="utf-8")
    code, _, stderr = _run_cli(
        [
            "tracebench", "regress",
            "--corpus", str(_sample_corpus_path()),
            "--store", str(tmp_path / "store"),
            "--suite", "bench",
            "--candidate-outputs", str(mine_out / "oracle-outputs.json"),
            "--owner", OWNER,
            "--cadence", "weekly",
            "--state-dir", str(state_file),
            "--out", str(tmp_path / "out"),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 1
    assert stderr.startswith("Error: ")
    assert "Traceback" not in stderr


def test_derive_next_version_unicode_digit(tmp_path: Path) -> None:
    """A stored version id of "²" (isdigit True, int() refuses) fails
    closed as a TracebenchError, not a raw ValueError."""
    from agenteval_bench.tracebench import _derive_next_version_id

    assert _derive_next_version_id("007") == "8"
    with pytest.raises(TracebenchError, match="cannot auto-derive"):
        _derive_next_version_id("²")
    with pytest.raises(TracebenchError, match="cannot auto-derive"):
        _derive_next_version_id("v1.2")


def test_load_run_state_missing_prev_key_fails(tmp_path: Path) -> None:
    """A state file with no prev_version_id key fails closed instead of
    silently defaulting the version chain to None."""
    from agenteval_bench.tracebench import _load_run_state

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        json.dumps({"suite_name": "bench", "consecutive_failures": 0}),
        encoding="utf-8",
    )
    with pytest.raises(TracebenchError, match="no prev_version_id"):
        _load_run_state(state_dir, "bench")


def test_previous_stale_uses_recorded_cadence(tmp_path: Path) -> None:
    """Staleness is judged by the previous version's recorded weekly
    cadence, not the flag this run passes: 2 days later with --cadence
    daily the weekly benchmark is NOT stale."""
    store, oracle = _first_run(tmp_path)
    two_days_later = "2026-11-08T19:00:00+00:00"
    result = run_scheduled(
        corpus=_sample_corpus_path(),
        store=store,
        suite_name="bench",
        version_id="2",
        prev_version_id="1",
        candidate_outputs=oracle,
        owner=OWNER,
        cadence="daily",
        out_dir=tmp_path / "run2",
        mined_at=two_days_later,
    )
    assert result.previous_stale is False


def test_corpus_kind_flows_to_artifacts(tmp_path: Path) -> None:
    """corpus_kind travels from the mine call onto the mine report and
    the manifest, and defaults to unknown when undeclared."""
    load = load_corpus(_sample_corpus_path())
    report = mine_traces(
        load.records,
        owner=OWNER,
        cadence="weekly",
        mined_at=MINED_AT,
        corpus_kind="sample",
    )
    assert report.corpus_kind == "sample"
    assert report.to_dict()["corpus_kind"] == "sample"
    manifest = build_manifest(
        report.cases,
        benchmark_id="bench",
        owner=OWNER,
        cadence="weekly",
        corpus_kind="sample",
    )
    assert manifest.corpus_kind == "sample"
    assert manifest.to_dict()["corpus_kind"] == "sample"
    defaulted = mine_traces(
        load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    assert defaulted.corpus_kind == "unknown"
    with pytest.raises(ValueError, match="corpus_kind"):
        mine_traces(
            load.records,
            owner=OWNER,
            cadence="weekly",
            mined_at=MINED_AT,
            corpus_kind="real-ish",
        )


def test_cli_mine_corpus_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The CLI --corpus-kind flag lands in mined-cases.json and
    manifest.json, so the sample fixture is labeled in the artifact."""
    out = tmp_path / "mine-out"
    code, _, _ = _run_cli(
        [
            "tracebench", "mine",
            "--corpus", str(_sample_corpus_path()),
            "--owner", OWNER,
            "--cadence", "weekly",
            "--corpus-kind", "sample",
            "--out", str(out),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0
    mined = json.loads((out / "mined-cases.json").read_text(encoding="utf-8"))
    assert mined["corpus_kind"] == "sample"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus_kind"] == "sample"


def test_workflow_wiring() -> None:
    """The deployment configuration itself is pinned: the workflow must
    restore store + state from cache, chain via --state-dir with no
    hardcoded --version 1, offer real corpus/candidate inputs, surface
    the action ledger, and alarm on a lost run state."""
    import yaml

    workflow_path = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "tracebench-regression.yml"
    )
    doc = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    text = workflow_path.read_text(encoding="utf-8")
    assert "actions/cache" in text
    assert "--state-dir" in text
    assert "--version 1" not in text
    assert "candidate-artifact" in text
    assert "corpus-artifact" in text
    assert "actions.jsonl" in text
    assert "GITHUB_STEP_SUMMARY" in text
    assert "tracebench-rollback" in text
    assert "previous completed run" in text
    # PyYAML parses the `on` key as boolean True (YAML 1.1).
    on_block = doc.get("on", doc.get(True))
    inputs = on_block["workflow_dispatch"]["inputs"]
    assert "candidate-artifact" in inputs
    assert "corpus-artifact" in inputs
    steps = doc["jobs"]["regress"]["steps"]
    step_text = "\n".join(str(s.get("run", "")) + str(s.get("uses", "")) for s in steps)
    assert "actions/cache" in step_text
    # Guards use truthiness, not != '' (unverifiable in this VM otherwise).
    assert "inputs.candidate-artifact != ''" not in text
    assert "inputs.corpus-artifact != ''" not in text

# ---------------------------------------------------------------------------
# Fix-forward round 3: coverage floor (tiered expanded requires >= 80%).
# The round-2 growth (drift gate + demo drift section) pushed total
# coverage to 79.79%. These tests pin the round-2 hardening paths and run
# the demo under coverage, instead of weakening the floor.
# ---------------------------------------------------------------------------


def test_demo_main_runs_end_to_end() -> None:
    """The demo (incl. the drift section) runs green and returns 0.

    Running it under pytest also covers the demo module, which the
    tiered coverage floor counts but no test previously executed.
    """
    from agenteval_bench.tracebench_demo import main

    assert main() == 0


def test_mine_report_rejects_non_string_corpus_kind() -> None:
    load = load_corpus(_sample_corpus_path())
    report = mine_traces(load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT)
    doc = json.loads(report.to_json())
    doc["corpus_kind"] = 123
    with pytest.raises(TypeError, match="corpus_kind must be a string"):
        MineReport.from_dict(doc)


def test_manifest_rejects_non_string_corpus_kind() -> None:
    load = load_corpus(_sample_corpus_path())
    report = mine_traces(load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT)
    manifest = build_manifest(
        report.cases,
        benchmark_id="bench",
        owner=OWNER,
        cadence="weekly",
        created_at=MINED_AT,
        last_mined_at=MINED_AT,
    )
    doc = json.loads(manifest.to_json())
    doc["corpus_kind"] = ["sample"]
    with pytest.raises(TypeError, match="corpus_kind must be a string"):
        BenchmarkManifest.from_dict(doc)


def test_write_run_state_wraps_os_error(tmp_path: Path) -> None:
    """A state dir that is an existing file fails as a clean
    TracebenchError, never a raw traceback."""
    from agenteval_bench.tracebench import _write_run_state

    state_file = tmp_path / "state-file"
    state_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(TracebenchError, match="cannot write run state"):
        _write_run_state(
            state_file,
            suite_name="bench",
            prev_version_id="1",
            consecutive_failures=0,
            updated_at=MINED_AT,
        )


def test_mined_case_ids_fails_closed_on_missing_version(tmp_path: Path) -> None:
    """Walking a version chain through a version the store never
    published fails closed instead of silently scoping retirement to
    nothing."""
    from agenteval_bench.tracebench import _mined_case_ids

    store = SnapshotStore(tmp_path / "store")
    with pytest.raises(TracebenchError, match="cannot walk the version chain"):
        _mined_case_ids(store, "bench", "99")


def test_derive_next_version_id_rejects_negative() -> None:
    """A negative previous version id cannot auto-derive a next id."""
    from agenteval_bench.tracebench import _derive_next_version_id

    with pytest.raises(TracebenchError, match="cannot auto-derive"):
        _derive_next_version_id("-1")
