"""CLI-level tests for the metric-assumption ledger (arena#56, round 3).

The growth check the static table could never provide: each test runs
a real CLI command and asserts the metrics it prints are covered by
the audit it prints — a command that grows a printed metric without a
ledger entry fails here, because the audit would flag it (or the
headline would not match). Live derivation is asserted, not the
registry against itself: run without labels audits pass_rate only,
with labels it audits the measured matcher and its backup; decide
audits only measured dimensions; calibrate/perturb flag their
honestly-unbacked soft metrics.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

from agenteval_bench import cli

LABELS = ["a", "b", "c"]

STABLE_SLICES = {
    "pooled_n_treated": 10000,
    "pooled_n_control": 10000,
    "strata": [
        {"name": "a", "n_treated": 5000, "n_control": 5000,
         "treated_successes": 115, "control_successes": 101},
        {"name": "b", "n_treated": 5000, "n_control": 5000,
         "treated_successes": 60, "control_successes": 50},
    ],
}


def _examples(name: str) -> str:
    return str(Path(__file__).resolve().parents[1] / "examples" / name)


def _run(argv, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["agenteval-bench", *argv])
    try:
        cli.main()
        captured = capsys.readouterr()
        return 0, captured.out, captured.err
    except SystemExit as e:
        captured = capsys.readouterr()
        return e.code or 0, captured.out, captured.err


def _write_json(doc: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f)
    return path


def _write_jsonl(rows: list[dict]) -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def test_run_audits_only_pass_rate(monkeypatch, capsys):
    # Round-2 B3 (product): the headline must not certify metrics the
    # report never shows — no scorer_validity number, no
    # label_agreement number without --validity-labels.
    code, out, _ = _run(
        ["run", "--suite", _examples("support-agent.yaml")], monkeypatch, capsys
    )
    assert code == 0
    assert "Metric ledger (run): 1/1 metrics clean" in out
    assert "scorer_validity" not in out
    assert "label_agreement" not in out


def test_run_with_validity_labels_audits_measured_matcher(monkeypatch, capsys):
    # The labels file validates `contains`: the audit grows to the
    # measured matcher and its backup, and only those.
    code, out, _ = _run(
        [
            "run",
            "--suite",
            _examples("support-agent.yaml"),
            "--validity-labels",
            _examples("contains-truth-labels.yaml"),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert "Metric ledger (run): 3/3 metrics clean" in out
    assert "label_agreement" in out
    assert "contains" in out
    assert "scorer_validity" not in out


def test_decide_audits_only_measured_dimensions(monkeypatch, capsys, tmp_path):
    out_path = str(tmp_path / "decision.json")
    code, out, _ = _run(
        [
            "decide",
            "--suite",
            _examples("decision-suite.yaml"),
            "--policy",
            _examples("decision-policy.yaml"),
            "--risk",
            "medium",
            "--out",
            out_path,
        ],
        monkeypatch,
        capsys,
    )
    assert code == 2  # INVESTIGATE: small-n suite, by design
    assert "Metric ledger (decide):" in out
    assert "scorer_validity" not in out
    assert "label_agreement" not in out
    with open(out_path, encoding="utf-8") as f:
        report = json.load(f)
    ledger = report["metric_ledger"]
    # Every audited metric is a dimension the decision measured.
    assert set(ledger["metrics"]) == {"quality", "safety", "reliability", "cost"}
    assert ledger["all_ok"] is True


def test_calibrate_flags_unbacked_gap(monkeypatch, capsys):
    code, out, _ = _run(
        ["calibrate", "--observations", _examples("calibration-observations.json")],
        monkeypatch,
        capsys,
    )
    assert code in (0, 2)
    assert "Metric ledger (calibrate): 1/2 metrics clean" in out
    assert "calibration_gap" in out
    assert "FLAGGED" in out


def test_calibrate_strict_ledger_refuses(monkeypatch, capsys):
    # The gap is flagged by design, so strict mode refuses the report.
    code, _out, err = _run(
        [
            "calibrate",
            "--observations",
            _examples("calibration-observations.json"),
            "--strict-ledger",
        ],
        monkeypatch,
        capsys,
    )
    assert code == 2
    assert "Ledger gate (strict)" in err


def test_perturb_flags_unbacked_stability(monkeypatch, capsys):
    code, out, _ = _run(
        ["perturb", "--suite", _examples("perturbation-triple.yaml")],
        monkeypatch,
        capsys,
    )
    assert code in (0, 1, 2)
    assert "Metric ledger (perturb):" in out
    assert "perturbation_stability" in out
    assert "FLAGGED" in out
    assert "pass_rate" in out


def test_simpson_json_carries_ledger(monkeypatch, capsys, tmp_path):
    path = _write_json(STABLE_SLICES)
    report = str(tmp_path / "simpson.json")
    try:
        code, out, _ = _run(
            ["simpson", "--slices", path, "--out", report], monkeypatch, capsys
        )
        assert code == 0
        assert "Metric ledger (simpson): 2/2 metrics clean" in out
        with open(report, encoding="utf-8") as f:
            payload = json.load(f)
        ledger = payload["metric_ledger"]
        assert ledger["all_ok"] is True
        assert set(ledger["metrics"]) == {
            "simpson_stratum_rate",
            "simpson_pooled_delta",
        }
    finally:
        os.unlink(path)


def test_diverge_json_carries_ledger(monkeypatch, capsys, tmp_path):
    report = str(tmp_path / "diverge.json")
    code, out, _ = _run(
        [
            "diverge",
            "--candidates",
            _examples("diverge-candidates.json"),
            "--out",
            report,
        ],
        monkeypatch,
        capsys,
    )
    assert code in (0, 2)
    assert "Metric ledger (diverge): 3/3 metrics clean" in out
    with open(report, encoding="utf-8") as f:
        payload = json.load(f)
    ledger = payload["metric_ledger"]
    assert ledger["all_ok"] is True
    assert set(ledger["metrics"]) == {
        "pairwise_diversity",
        "conventional_share",
        "p_conventional",
    }


def test_ship_json_carries_ledger(monkeypatch, capsys, tmp_path):
    report = str(tmp_path / "ship.json")
    code, out, _ = _run(
        [
            "ship",
            "--design",
            _examples("ship-design.json"),
            "--evidence",
            _examples("ship-evidence.json"),
            "--registry",
            _examples("ship-registry.jsonl"),
            "--out",
            report,
        ],
        monkeypatch,
        capsys,
    )
    assert code in (0, 1)
    assert "Metric ledger (ship):" in out
    with open(report, encoding="utf-8") as f:
        payload = json.load(f)
    ledger = payload["metric_ledger"]
    assert ledger["all_ok"] is True
    assert {"oec_z", "srm_chi2", "aa_z", "guardrail_delta"} <= set(
        ledger["metrics"]
    )


def test_alt_test_json_carries_ledger(monkeypatch, capsys):
    rng = random.Random(121)
    rows = []
    for i in range(60):
        truth = rng.choice(LABELS)
        rows.append(
            {
                "id": f"i{i}",
                "task": "discrete",
                "annotators": {
                    f"ann{a}": truth if rng.random() > 0.15 else rng.choice(LABELS)
                    for a in range(4)
                },
                "judges": {"j": truth if rng.random() > 0.1 else rng.choice(LABELS)},
            }
        )
    data = _write_jsonl(rows)
    out = data + ".report.json"
    try:
        code, stdout, _ = _run(
            ["alt-test", "--data", data, "--epsilon", "0.1", "--out", out],
            monkeypatch,
            capsys,
        )
        assert code == 0, stdout
        assert "Metric ledger (alt-test): 2/2 metrics clean" in stdout
        with open(out, encoding="utf-8") as f:
            report = json.load(f)
        ledger = report["metric_ledger"]
        assert ledger["all_ok"] is True
        assert set(ledger["metrics"]) == {"alt_win_rate", "alt_coverage"}
    finally:
        os.unlink(data)
        if os.path.exists(out):
            os.unlink(out)


def test_tiered_gate_json_carries_ledger(monkeypatch, capsys, tmp_path):
    # Gate phase measures neither coverage nor mutation: the JSON
    # still carries the (empty) audit, so the artifact never implies
    # a ledger it does not have.
    report = str(tmp_path / "tiered.json")
    code, _out, _ = _run(
        ["tiered", "--phase", "gate", "--root", ".", "--out", report],
        monkeypatch,
        capsys,
    )
    assert code in (0, 2)
    with open(report, encoding="utf-8") as f:
        payload = json.load(f)
    assert "metric_ledger" in payload
    assert payload["metric_ledger"]["findings"] == []


def test_snapshot_compare_audits_pass_rate(monkeypatch, capsys, tmp_path):
    store = str(tmp_path / "store")
    code, out, _ = _run(
        [
            "snapshot",
            "publish",
            "--suite",
            _examples("support-agent.yaml"),
            "--store",
            store,
            "--version",
            "v1",
            "--owner",
            "test",
            "--why",
            "ledger check",
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert "Metric ledger (snapshot publish): 1/1 metrics clean" in out
    # Candidate outputs identical to the recorded baseline: GATE_PASS.
    import yaml

    with open(_examples("support-agent.yaml"), encoding="utf-8") as f:
        suite = yaml.safe_load(f)
    candidate = {
        case["id"]: case["output"] for case in suite["cases"] if "output" in case
    }
    cand_path = _write_json(candidate)
    try:
        code, out, _ = _run(
            [
                "snapshot",
                "compare",
                "--store",
                store,
                "--suite",
                "support-agent-golden",
                "--version",
                "v1",
                "--candidate-outputs",
                cand_path,
            ],
            monkeypatch,
            capsys,
        )
        assert code == 0, out
        assert "Metric ledger (snapshot compare): 1/1 metrics clean" in out
    finally:
        os.unlink(cand_path)


def test_snapshot_show_audits_pass_rate(monkeypatch, capsys, tmp_path):
    store = str(tmp_path / "store")
    code, _, _ = _run(
        [
            "snapshot",
            "publish",
            "--suite",
            _examples("support-agent.yaml"),
            "--store",
            store,
            "--version",
            "v1",
            "--owner",
            "test",
            "--why",
            "ledger check",
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0
    code, out, _ = _run(
        [
            "snapshot",
            "show",
            "--store",
            store,
            "--suite",
            "support-agent-golden",
            "--version",
            "v1",
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert "Metric ledger (snapshot show): 1/1 metrics clean" in out


def test_baseline_prints_frozen_scores_notice(monkeypatch, capsys, tmp_path):
    out_path = str(tmp_path / "baseline.json")
    code, out, _ = _run(
        [
            "baseline",
            "--suite",
            _examples("support-agent.yaml"),
            "--out",
            out_path,
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert "frozen recordings, not a reported aggregate metric" in out


def test_tracebench_mine_prints_not_covered_notice(monkeypatch, capsys, tmp_path):
    out_dir = str(tmp_path / "mined")
    code, out, _ = _run(
        [
            "tracebench",
            "mine",
            "--corpus",
            _examples("traces/production-traces.sample.jsonl"),
            "--owner",
            "test",
            "--cadence",
            "weekly",
            "--out",
            out_dir,
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0, out
    assert "the metric ledger does not apply to this output" in out
