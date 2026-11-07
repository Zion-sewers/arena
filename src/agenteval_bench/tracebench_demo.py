"""Demo: trace-mined benchmarks end to end (arena#41).

Mines the SAMPLE corpus (recorded-format fixtures — not real incidents),
then runs the scheduled regression three times against a temporary store:

1. oracle candidate -> GATE_PASS, no actions, version 1 published;
2. degraded candidate -> GATE_FAIL, gate + investigation actions;
3. degraded candidate again, after 2 prior failures -> rollback appears.

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

    print("demo: mine -> publish -> gate -> rollback, all contracts held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
