"""CLI CI-gate tests: a replay suite must exit non-zero below threshold."""

import os
import tempfile

from agenteval_bench import cli

REPLAY_SUITE = """
name: replay
cases:
  - id: good
    input: "q1"
    output: "the answer contains hello"
    expected:
      contains: ["hello"]
  - id: bad
    input: "q2"
    output: "totally unrelated"
    expected:
      contains: ["goodbye"]
"""


def _write(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def _run(argv, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["agenteval-bench", *argv])
    try:
        cli.main()
        captured = capsys.readouterr()
        return 0, captured.out, captured.err
    except SystemExit as e:
        captured = capsys.readouterr()
        return e.code or 0, captured.out, captured.err


def test_ci_gate_fails_below_threshold(monkeypatch, capsys):
    # 1 of 2 pass -> 50%. Threshold 0.9 must FAIL (exit 2).
    path = _write(REPLAY_SUITE)
    try:
        code, _, _ = _run(["run", "--suite", path, "--ci", "--threshold", "0.9"], monkeypatch, capsys)
        assert code == 2
    finally:
        os.unlink(path)


def test_ci_gate_passes_at_or_below_actual(monkeypatch, capsys):
    # 50% pass rate, threshold 0.5 -> PASS (exit 0).
    path = _write(REPLAY_SUITE)
    try:
        code, _, _ = _run(["run", "--suite", path, "--ci", "--threshold", "0.5"], monkeypatch, capsys)
        assert code == 0
    finally:
        os.unlink(path)


def test_non_ci_run_never_exits_nonzero(monkeypatch, capsys):
    # Without --ci, a low pass rate still exits 0 (reporting only).
    path = _write(REPLAY_SUITE)
    try:
        code, _, _ = _run(["run", "--suite", path], monkeypatch, capsys)
        assert code == 0
    finally:
        os.unlink(path)


def test_ci_failure_prints_summary(monkeypatch, capsys):
    # When the CI gate fails, the summary and gate status must still be printed.
    path = _write(REPLAY_SUITE)
    try:
        code, out, _ = _run(["run", "--suite", path, "--ci", "--threshold", "0.9"], monkeypatch, capsys)
        assert code == 2
        assert "Suite: replay" in out
        assert "Pass rate:" in out
        assert "CI gate" in out
        assert "FAIL" in out
    finally:
        os.unlink(path)


def test_out_of_range_threshold_high(monkeypatch, capsys):
    # Threshold above 1.0 is invalid and must produce a user-facing error.
    path = _write(REPLAY_SUITE)
    try:
        code, _, err = _run(["run", "--suite", path, "--ci", "--threshold", "1.5"], monkeypatch, capsys)
        assert code == 1
        assert "threshold" in err.lower()
    finally:
        os.unlink(path)


def test_out_of_range_threshold_low(monkeypatch, capsys):
    # Threshold below 0.0 is invalid and must produce a user-facing error.
    path = _write(REPLAY_SUITE)
    try:
        code, _, err = _run(["run", "--suite", path, "--ci", "--threshold", "-0.5"], monkeypatch, capsys)
        assert code == 1
        assert "threshold" in err.lower()
    finally:
        os.unlink(path)


def test_bugsmith_requires_repo_and_out(monkeypatch, capsys):
    code, _, err = _run(["bugsmith", "--repo", "x"], monkeypatch, capsys)
    assert code == 1
    assert "--repo" in err or "repo" in err.lower()


def test_bugsmith_rejects_missing_repo(monkeypatch, capsys):
    code, _, err = _run(
        ["bugsmith", "--repo", "/nonexistent", "--out", "/tmp/x"],
        monkeypatch, capsys,
    )
    assert code == 1
    assert "not found" in err


def test_bugsmith_end_to_end_local(monkeypatch, capsys, tmp_path):
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1] / "src" / "bugsmith" / "fixtures" / "target"
    out = tmp_path / "bench"
    code, stdout, _ = _run(
        ["bugsmith", "--repo", str(repo), "--out", str(out),
         "--procedural", "4", "--seed", "5", "--local"],
        monkeypatch, capsys,
    )
    assert code == 0, stdout
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["baseline_tests"] == 16
    assert manifest["valid_instances"] >= 1
    assert 1 <= len(manifest["curated"]) <= manifest["valid_instances"]
    for name in manifest["curated"]:
        inst = json.loads((out / f"{name}.json").read_text())
        assert inst["fail_to_pass"], "curated instance must break >= 1 test"
        assert inst["seed"] == 5


def test_bugsmith_unknown_command(monkeypatch, capsys):
    code, _, err = _run(["frobnicate"], monkeypatch, capsys)
    assert code == 1
    assert "Unknown command" in err


# --- prespec review gate (arena#48) ---

PRESPEC_PLAN = {
    "name": "reranker-v2-failure-modes",
    "mode": "confirmatory",
    "question": "Does reranker v2 reduce the hallucinated-citation failure mode?",
    "hypotheses": [
        "reranker v2 cuts hallucinated citations by at least 10pp vs v1"
    ],
    "assumptions": [
        {
            "text": "the eval suite's citation cases represent production traffic",
            "falsifier": "suite citation cases differ from production by >20pp",
        }
    ],
    "n_planned": 200,
    "decision_rule": "confirm iff the 95% interval on the delta excludes 0",
    "data_source": "golden replay suite, seed 42",
}


def _prespec_files(tmp_path, hypothesis, accessed="2026-11-08T00:00:00+00:00"):
    import json

    from experiment import plan_from_dict, register_analysis_plan

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(PRESPEC_PLAN), encoding="utf-8")
    registry = str(tmp_path / "analyses.jsonl")
    digest = register_analysis_plan(plan_from_dict(PRESPEC_PLAN), registry)
    finding = {
        "plan_digest": digest,
        "mode": "confirmatory",
        "hypothesis": hypothesis,
        "claim": "v2 cuts hallucinated citations by 14pp",
        "n_observed": 200,
        "interval": [0.08, 0.20],
        "data_digest": "sha256:golden-suite-seed42",
        "data_accessed_at": accessed,
        "code_version": "arena@dbf6909",
    }
    finding_path = tmp_path / "finding.json"
    finding_path.write_text(json.dumps(finding), encoding="utf-8")
    return str(plan_path), str(finding_path), registry


def test_prespec_cli_confirms(monkeypatch, capsys, tmp_path):
    plan, finding, registry = _prespec_files(
        tmp_path,
        "reranker v2 cuts hallucinated citations by at least 10pp vs v1",
    )
    code, out, _ = _run(
        ["prespec", "--finding", finding, "--plan", plan, "--registry", registry],
        monkeypatch, capsys,
    )
    assert code == 0, out
    assert "verdict: CONFIRMED" in out
    assert "[PASS] Q5" in out


def test_prespec_cli_refuses_post_hoc(monkeypatch, capsys, tmp_path):
    _, finding, registry = _prespec_files(
        tmp_path, "v2 is better somehow",
    )
    code, out, _ = _run(
        ["prespec", "--finding", finding, "--registry", registry],
        monkeypatch, capsys,
    )
    assert code == 1, out
    assert "verdict: REFUSED" in out
    assert "POST_HOC_CONFIRMATION" in out


def test_prespec_cli_bad_input(monkeypatch, capsys, tmp_path):
    code, _, err = _run(
        ["prespec", "--finding", str(tmp_path / "missing.json")],
        monkeypatch, capsys,
    )
    assert code == 2
    assert "invalid finding doc" in err


def test_prespec_cli_bad_interval_exit_2(monkeypatch, capsys, tmp_path):
    import json
    from pathlib import Path

    _, finding, registry = _prespec_files(
        tmp_path,
        "reranker v2 cuts hallucinated citations by at least 10pp vs v1",
    )
    doc = json.loads(Path(finding).read_text(encoding="utf-8"))
    doc["interval"] = [0.08]
    bad = tmp_path / "finding-bad-interval.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    code, _, err = _run(
        ["prespec", "--finding", str(bad), "--registry", registry],
        monkeypatch, capsys,
    )
    assert code == 2
    assert "invalid finding doc" in err
