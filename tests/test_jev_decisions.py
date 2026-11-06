"""Decision-log rounding pins (arena#39 nightly).

jev/test_hardening.py covers the append-only shape, but it lives outside
testpaths=["tests"] so pytest never runs it — and it never pins the exact
rounded values anyway. A negated rounding precision (round(x, 4) ->
round(x, -4), i.e. 0.0) survived the suite before this test: the recorded
complexity/confidence silently collapsed to zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev.decisions import log_decision


def test_log_decision_rounds_to_four_places(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "decisions.jsonl"
    monkeypatch.setenv("JEV_DECISIONS_LOG", str(log))
    record = log_decision(
        task="round me",
        route_name="ollama-local",
        route_kind="ollama",
        complexity=0.123456,
        confidence=0.987654,
        kept_local=True,
        reason="below threshold",
        escalated=False,
        budget="free",
    )
    assert record["complexity"] == 0.1235
    assert record["confidence"] == 0.9877
    rows = [json.loads(line) for line in log.read_text().strip().split("\n")]
    assert len(rows) == 1
    assert rows[0]["complexity"] == 0.1235
    assert rows[0]["confidence"] == 0.9877
    assert "round me" not in log.read_text()  # raw task text never stored (PII)
