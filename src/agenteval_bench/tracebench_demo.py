"""Demo: trace-mined benchmarks end to end (arena#41).

Mines the SAMPLE corpus (recorded-format fixtures — not real incidents),
then runs the scheduled regression three times against a temporary store:

1. oracle candidate -> GATE_PASS, no actions, version 1 published;
2. degraded candidate -> GATE_FAIL, gate + investigation actions;
3. degraded candidate again, after 2 prior failures -> rollback appears.

Then the closed loop: three consecutive weeks against a fresh store,
all through ``--state-dir`` with no hand-fed --prev / --prior-failures
and no --version (auto-derived from the chained state):

4. week 1 degraded -> GATE_FAIL, streak 1, gate + investigation;
5. week 2 degraded -> GATE_FAIL, streak 2, gate + investigation;
6. week 3 degraded -> GATE_FAIL, streak 3, rollback appears.

Run: ``PYTHONPATH=src python -m agenteval_bench.tracebench_demo``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agenteval_bench.tracebench import (
    load_corpus,
    mine_traces,
    run_scheduled,
)

MINED_AT = "2026-11-06T19:00:00+00:00"
OWNER = "arena-maintainers"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    corpus = repo_root / "examples" / "traces" / "production-traces.sample.jsonl"
    print("tracebench demo — SAMPLE corpus (recorded-format fixtures,")
    print("  not real production incidents):", corpus)
    load = load_corpus(corpus)
    report = mine_traces(
        load.records, owner=OWNER, cadence="weekly", mined_at=MINED_AT
    )
    print(f"mined {len(report.cases)} case(s) from {len(load.records)} trace(s)")
    oracle = {c.case.id: c.expected_output for c in report.cases}
    degraded = {case_id: "wrong answer" for case_id in oracle}

    with tempfile.TemporaryDirectory(prefix="tracebench-demo-") as tmp:
        store = Path(tmp) / "store"

        first = run_scheduled(
            corpus=corpus,
            store=store,
            suite_name="bench",
            version_id="1",
            prev_version_id=None,
            candidate_outputs=oracle,
            owner=OWNER,
            cadence="weekly",
            out_dir=Path(tmp) / "run1",
            mined_at=MINED_AT,
        )
        assert first.verdict == "GATE_PASS", first.verdict
        assert first.actions == ()
        print(f"run 1 (oracle candidate): {first.verdict}, "
              f"published v{first.version_id}, {first.new_cases} new case(s)")

        second = run_scheduled(
            corpus=corpus,
            store=store,
            suite_name="bench",
            version_id="2",
            prev_version_id="1",
            candidate_outputs=degraded,
            owner=OWNER,
            cadence="weekly",
            out_dir=Path(tmp) / "run2",
            mined_at=MINED_AT,
        )
        assert second.verdict == "GATE_FAIL", second.verdict
        assert [a.action for a in second.actions] == ["gate", "investigation"]
        assert second.published_new_version is False  # steady state: nothing new
        print(f"run 2 (degraded candidate): {second.verdict}, "
              f"actions: {', '.join(a.action for a in second.actions)}")

        third = run_scheduled(
            corpus=corpus,
            store=store,
            suite_name="bench",
            version_id="3",
            prev_version_id="1",
            candidate_outputs=degraded,
            owner=OWNER,
            cadence="weekly",
            out_dir=Path(tmp) / "run3",
            mined_at=MINED_AT,
            prior_consecutive_failures=2,
        )
        assert third.verdict == "GATE_FAIL", third.verdict
        assert [a.action for a in third.actions] == [
            "gate",
            "investigation",
            "rollback",
        ]
        print(f"run 3 (3rd consecutive failure): {third.verdict}, "
              f"actions: {', '.join(a.action for a in third.actions)}")

    # Closed loop: the same three weeks, but the runner chains itself —
    # state_dir supplies --prev, the streak, and the next --version.
    # Nothing is hand-fed; rollback must still appear on week 3.
    with tempfile.TemporaryDirectory(prefix="tracebench-demo-loop-") as tmp:
        store = Path(tmp) / "store"
        state_dir = Path(tmp) / "state"
        weeks: list[str] = []
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
                out_dir=Path(tmp) / f"week{week}",
                mined_at=MINED_AT,
                state_dir=state_dir,
            )
            assert result.verdict == "GATE_FAIL", result.verdict
            assert result.consecutive_failures == week, result.to_dict()
            assert result.version_id == "1", result.version_id
            weeks.append(", ".join(a.action for a in result.actions))
            print(f"week {week} (closed loop, no hand-feeding): "
                  f"{result.verdict}, streak {result.consecutive_failures}, "
                  f"actions: {weeks[-1]}")
        assert weeks[0] == "gate, investigation"
        assert weeks[1] == "gate, investigation"
        assert weeks[2] == "gate, investigation, rollback"

    print("demo: mine -> publish -> gate -> rollback, all contracts held")
    print("demo: closed loop — three weeks, state-chained, rollback on week 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
