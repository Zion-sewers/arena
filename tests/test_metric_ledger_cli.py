"""CLI-level tests for the metric-assumption ledger (arena#56, round 4).

Two checks per command. First, the audit covers the metrics the
report printed: run without labels audits pass_rate only, with
labels it audits the measured matcher and its backup; decide audits
only measured dimensions; calibrate/perturb flag their
honestly-unbacked soft metrics; an empty audit says "no metrics
measured", never "0/0 metrics clean", and passes --strict-ledger
through. Second, the growth check (arch B3): ``_assert_stdout_covered``
is called on the stdout of every covered command — every line with
a rendered number must match a declared rendering pattern tied to
an audited metric or a declared component of one. A command that
grows a printed metric without a ledger entry fails here, because
the new line matches no pattern: the audit can only flag metrics
it is told about, so the test, not the audit, is the backstop. The
arch reviewer's exact mutation line is pinned as a failing case.
"""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
from pathlib import Path

import pytest

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
    # Round-4 B1 (product): the report states the headline's scope
    # in its own text — the validity table's grades are not ledger
    # metrics, and the headline never certifies them.
    assert "Ledger headline covers the printed aggregate only" in out
    _assert_stdout_covered(
        out,
        "run",
        [(r"^Pass rate: \d+(?:\.\d+)?%", "pass_rate", "run pass-rate line")],
    )


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
    # Growth scan: the labels line renders label_agreement, and the
    # table row for the validated matcher renders its agreement CI.
    _assert_stdout_covered(
        out,
        "run --validity-labels",
        [
            (r"^Pass rate: \d+(?:\.\d+)?%", "pass_rate", "run pass-rate line"),
            (r"^Validity labels: ", "label_agreement", "labels agreement line"),
            (
                r"^  contains +\S+ +agreement 95% CI=\[",
                "label_agreement",
                "validated matcher agreement row",
            ),
        ],
    )


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
    assert "Ledger headline covers the printed dimensions only" in out
    _assert_stdout_covered(
        out,
        "decide",
        [
            (r"^  quality\s+\w+\s+point=\d", "quality", "decide quality line"),
            (r"^  safety\s+\w+\s+point=\d", "safety", "decide safety line"),
            (r"^  cost\s+\w+\s+point=\d", "cost", "decide cost line"),
            (
                r"^  reliability\s+\w+\s+point=\d",
                "reliability",
                "decide reliability line",
            ),
        ],
    )
    with open(out_path, encoding="utf-8") as f:
        report = json.load(f)
    ledger = report["metric_ledger"]
    # Every audited metric is a dimension the decision measured.
    assert set(ledger["metrics"]) == {"quality", "safety", "reliability", "cost"}
    assert ledger["all_ok"] is True
    # Round-4 B1 (product): the JSON states the headline's scope
    # next to the scorer_validity object it does not cover.
    assert "metric_ledger_scope" in report
    assert "printed dimensions only" in report["metric_ledger_scope"]


def test_calibrate_flags_unbacked_gap(monkeypatch, capsys):
    code, out, _ = _run(
        ["calibrate", "--observations", _examples("calibration-observations.json")],
        monkeypatch,
        capsys,
    )
    assert code in (0, 2)
    # Round 4: ECE is its own audited entry now, so the headline is
    # 2/3 (agreement_rate + ece clean, calibration_gap flagged).
    assert "Metric ledger (calibrate): 2/3 metrics clean" in out
    assert "calibration_gap" in out
    assert "ece" in out
    assert "FLAGGED" in out
    # Round-4 B2 (product): the headline carries the flag next to the
    # number, and the gating sentence scopes gating to the bin-check
    # grade, never to the ECE number.
    assert "ECE unbacked (see ledger): do not quote it as verified" in out
    assert "not on the ECE number" in out
    _assert_stdout_covered(
        out,
        "calibrate",
        [
            (
                r"^judge \S+: \w+ \(\d+ observations, ECE \d",
                "ece",
                "calibrate headline ECE",
            ),
            (
                r"^  bin \[",
                "agreement_rate",
                "calibrate bin line (agreement + gap components)",
            ),
        ],
    )


def test_calibrate_too_few_observations_audits_nothing(monkeypatch, capsys, tmp_path):
    # Round-4 B2 (impl): with too few observations no bin and no ECE
    # is printed, so the audit must not certify an agreement rate
    # the report never showed.
    path = tmp_path / "tiny.json"
    path.write_text(
        json.dumps(
            {
                "judge": "j",
                "observations": [
                    {"item": "a", "confidence": 0.9, "agreed": True},
                    {"item": "b", "confidence": 0.8, "agreed": True},
                    {"item": "c", "confidence": 0.7, "agreed": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    code, out, _ = _run(
        ["calibrate", "--observations", str(path)], monkeypatch, capsys
    )
    assert code == 2  # uncalibrated: too few observations
    assert "ECE n/a" in out
    assert "Metric ledger (calibrate): no metrics measured" in out
    assert "0/0 metrics clean" not in out
    assert "agreement_rate" not in out
    # Growth scan with no patterns: with too few observations the
    # report prints no numeric metric line at all.
    _assert_stdout_covered(out, "calibrate (too few)", [])


def test_calibrate_json_carries_status_note_when_flagged(monkeypatch, capsys, tmp_path):
    # Round-4 B2 (product): the JSON must not certify the flagged
    # number by itself — the status note scopes the grade to the
    # bin checks next to the ledger flag.
    out_path = str(tmp_path / "cal.json")
    code, stdout, _ = _run(
        [
            "calibrate",
            "--observations",
            _examples("calibration-observations.json"),
            "--json",
            "--out",
            out_path,
        ],
        monkeypatch,
        capsys,
    )
    assert code in (0, 2)
    payload = json.loads(stdout)
    assert payload["status"] == "calibrated"
    assert payload["metric_ledger"]["all_ok"] is False
    assert "status_note" in payload
    assert "unbacked" in payload["status_note"]
    with open(out_path, encoding="utf-8") as f:
        assert json.load(f)["status_note"] == payload["status_note"]


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
    # Round-4 B3 (product): the verdict line carries the flag next
    # to STABLE, so the label is never quoted without its
    # qualification.
    assert "verdict: STABLE (stability label unbacked — see ledger)" in out
    _assert_stdout_covered(
        out,
        "perturb",
        [
            (
                r"^  (clean|light|adversarial): pass rate \d",
                "pass_rate",
                "perturb regime pass rate",
            ),
            (
                r"^  drops: clean->light ",
                "perturbation_stability",
                "perturb drops",
            ),
        ],
    )


def test_perturb_strict_ledger_downgrades_the_artifact(monkeypatch, capsys, tmp_path):
    # Round-4 B3 (product): under --strict-ledger the artifact
    # itself is downgraded — stdout, JSON, and exit code — not just
    # the exit code while the artifact still says STABLE.
    out_path = str(tmp_path / "pert.json")
    code, out, err = _run(
        [
            "perturb",
            "--suite",
            _examples("perturbation-triple.yaml"),
            "--strict-ledger",
            "--out",
            out_path,
        ],
        monkeypatch,
        capsys,
    )
    assert code == 1
    assert "verdict: UNSTABLE" in out
    assert "verdict: STABLE" not in out
    assert "Ledger gate (strict): STABLE downgraded to UNSTABLE" in err
    # The downgraded artifact still renders the same rates; the
    # scan pins them, so a downgrade can never smuggle in a new one.
    _assert_stdout_covered(
        out,
        "perturb --strict-ledger",
        [
            (
                r"^  (clean|light|adversarial): pass rate \d",
                "pass_rate",
                "perturb regime pass rate",
            ),
            (
                r"^  drops: clean->light ",
                "perturbation_stability",
                "perturb drops",
            ),
        ],
    )
    with open(out_path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["verdict"]["outcome"] == "unstable"
    assert payload["metric_ledger"]["all_ok"] is False


def test_simpson_json_carries_ledger(monkeypatch, capsys, tmp_path):
    path = _write_json(STABLE_SLICES)
    report = str(tmp_path / "simpson.json")
    try:
        code, out, _ = _run(
            ["simpson", "--slices", path, "--out", report], monkeypatch, capsys
        )
        assert code == 0
        # Round 4: the treated share printed on each stratum line
        # is audited too (3/3).
        assert "Metric ledger (simpson): 3/3 metrics clean" in out
        _assert_stdout_covered(
            out,
            "simpson",
            [
                (
                    r"^  \S+: treated \d",
                    "simpson_stratum_rate",
                    "simpson stratum rate",
                ),
                (
                    r"^  \S+: treated \d",
                    "treated_share",
                    "simpson stratum treated share",
                ),
                (
                    r"^pooled \(descriptive only",
                    "simpson_pooled_delta",
                    "simpson pooled line",
                ),
            ],
        )
        with open(report, encoding="utf-8") as f:
            payload = json.load(f)
        ledger = payload["metric_ledger"]
        assert ledger["all_ok"] is True
        assert set(ledger["metrics"]) == {
            "simpson_stratum_rate",
            "treated_share",
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
    _assert_stdout_covered(
        out,
        "diverge",
        [
            (
                r"^diverge: \w+ \(min pairwise diversity ",
                "pairwise_diversity",
                "diverge headline diversity",
            ),
            (
                r"^  p_conventional \d",
                "p_conventional",
                "diverge p check line",
            ),
            (
                r"^  -> the candidates converge",
                "conventional_share",
                "diverge convergence line",
            ),
            (
                r"^  reason: ",
                "pairwise_diversity",
                "diverge selection reason (candidate scores)",
            ),
        ],
    )
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
    _assert_stdout_covered(
        out,
        "ship",
        [
            (r"^  OEC final z=", "oec_z", "ship OEC line"),
            (r"^  SRM: .*chi2=", "srm_chi2", "ship SRM line"),
            (r"^  A/A: .*z=", "aa_z", "ship A/A line"),
            (
                r"^  guardrail \S+: .*delta ",
                "guardrail_delta",
                "ship guardrail line",
            ),
            (r"^  ~ SRM clean: chi2=", "srm_chi2", "ship SRM note"),
            (r"^  ~ A/A clean:.*z=", "aa_z", "ship A/A note"),
            (r"^  ~ OEC .*final z=", "oec_z", "ship OEC note"),
        ],
    )
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
        # Round 4: the conservative score that orders the ranking
        # and the per-annotator rho_f values are audited too (4/4).
        assert "Metric ledger (alt-test): 4/4 metrics clean" in stdout
        _assert_stdout_covered(
            stdout,
            "alt-test",
            [
                (r"^epsilon=\d", None, "alt-test config parameters"),
                (
                    r"^\S+\s+\d\.\d{3}\s+\d\.\d{3}\s+\w+\s+\d+/\d+",
                    "alt_win_rate",
                    "alt-test rho/omega row",
                ),
                (
                    r"\d\.\d\d,\d\.\d\d",
                    "alt_annotator_rho",
                    "alt-test per-annotator rho_f",
                ),
                (
                    r"^\S+\s+\d\.\d{3}\s+\d+/\d+\s+RANKED",
                    "alt_conservative_score",
                    "alt-test conservative column",
                ),
            ],
        )
        with open(out, encoding="utf-8") as f:
            report = json.load(f)
        ledger = report["metric_ledger"]
        assert ledger["all_ok"] is True
        assert set(ledger["metrics"]) == {
            "alt_win_rate",
            "alt_annotator_rho",
            "alt_conservative_score",
            "alt_coverage",
        }
    finally:
        os.unlink(data)
        if os.path.exists(out):
            os.unlink(out)


def test_alt_test_refusal_audits_only_the_printed_board(monkeypatch, capsys):
    # Round-4 A1 (product): a refused ranking prints only the
    # coverage-gated board — conservative scores + coverage — never
    # rho/omega or per-annotator rho_f, so the audit covers only
    # those two and never certifies a ranking that was refused.
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
    try:
        code, stdout, _ = _run(
            ["alt-test", "--data", data, "--epsilon", "0.1", "--min-cell-n", "100"],
            monkeypatch,
            capsys,
        )
        assert code == 1, stdout
        assert "Metric ledger (alt-test): 2/2 metrics clean" in stdout
        assert "alt_conservative_score" in stdout
        assert "alt_win_rate" not in stdout
        assert "alt_annotator_rho" not in stdout
        # Growth scan: the refused board prints only the masked
        # diagnostic row — no rho/omega, no per-annotator rho_f, and
        # no epsilon header line (the refusal prints its own header).
        _assert_stdout_covered(
            stdout,
            "alt-test (refusal)",
            [
                (
                    r"^\S+\s+\d\.\d{3}\s+\d+/\d+\s+MASKED",
                    "alt_conservative_score",
                    "alt-test masked diagnostic row",
                ),
            ],
        )
    finally:
        os.unlink(data)


def test_tiered_gate_json_carries_ledger(monkeypatch, capsys, tmp_path):
    # Gate phase measures neither coverage nor mutation: the JSON
    # still carries the (empty) audit, so the artifact never implies
    # a ledger it does not have. Round 4: the empty audit says "no
    # metrics measured" (never "0/0 clean") and its JSON never pairs
    # the PASS verdict with all_ok false.
    report = str(tmp_path / "tiered.json")
    code, out, _ = _run(
        ["tiered", "--phase", "gate", "--root", ".", "--out", report],
        monkeypatch,
        capsys,
    )
    assert code in (0, 2)
    assert "Metric ledger (tiered): no metrics measured" in out
    assert "0/0 metrics clean" not in out
    # Growth scan: the gate prints the inner run reports twice, so
    # the audit-block splitter must exclude every block, not just
    # the first. The phase duration is a timing component, not a
    # metric.
    _assert_stdout_covered(
        out,
        "tiered gate",
        [
            (
                r"^Pass rate: \d+(?:\.\d+)?%",
                "pass_rate",
                "tiered inner run pass rate",
            ),
            (
                r"^CI gate: pass_rate ",
                "pass_rate",
                "tiered inner CI gate line",
            ),
            (
                r"^phase \w+: \w+ \(\d+\.\d+s\)",
                None,
                "tiered phase duration (timing)",
            ),
        ],
    )
    with open(report, encoding="utf-8") as f:
        payload = json.load(f)
    assert "metric_ledger" in payload
    assert payload["metric_ledger"]["findings"] == []
    assert payload["metric_ledger"]["all_ok"] is True


def test_tiered_gate_strict_ledger_passes_through(monkeypatch, capsys, tmp_path):
    # Round-4 B4 (impl/arch): --strict-ledger on a legitimately
    # empty audit passes through — a gate that refuses while naming
    # zero flagged metrics is a false positive.
    report = str(tmp_path / "tiered.json")
    code, _out, err = _run(
        ["tiered", "--phase", "gate", "--root", ".", "--strict-ledger", "--out", report],
        monkeypatch,
        capsys,
    )
    assert code in (0, 2)
    assert "Ledger gate (strict)" not in err
    with open(report, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["metric_ledger"]["all_ok"] is True


def test_ship_refusal_audits_nothing(monkeypatch, capsys, tmp_path):
    # Round-4 B3 (impl): a trust refusal carries no metric, so the
    # audit is empty — it must say "no metrics measured", never
    # "0/0 metrics clean", while the JSON agrees.
    with open(_examples("ship-evidence.json"), encoding="utf-8") as f:
        evidence = json.load(f)
    evidence["plan_digest"] = "0" * 64
    bad_path = str(tmp_path / "bad-evidence.json")
    with open(bad_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f)
    report = str(tmp_path / "ship.json")
    code, out, _ = _run(
        [
            "ship",
            "--design",
            _examples("ship-design.json"),
            "--evidence",
            bad_path,
            "--registry",
            _examples("ship-registry.jsonl"),
            "--out",
            report,
        ],
        monkeypatch,
        capsys,
    )
    assert code == 1
    assert "Metric ledger (ship): no metrics measured" in out
    assert "0/0 metrics clean" not in out
    # Growth scan with no patterns: a trust refusal prints no
    # numeric metric line at all.
    _assert_stdout_covered(out, "ship (refusal)", [])
    with open(report, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["metric_ledger"]["findings"] == []
    assert payload["metric_ledger"]["all_ok"] is True


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
    _assert_stdout_covered(
        out,
        "snapshot publish",
        [(r"^Baseline: pass_rate \d", "pass_rate", "snapshot publish baseline")],
    )
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
        _assert_stdout_covered(
            out,
            "snapshot compare",
            [
                (
                    r"^baseline pass_rate \d",
                    "pass_rate",
                    "snapshot compare pass rates",
                )
            ],
        )
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
    _assert_stdout_covered(
        out,
        "snapshot show",
        [
            (
                r"^created: \d{4}-\d\d-\d\dT",
                None,
                "snapshot show created timestamp",
            ),
            (
                r"^cases: \d+ +baseline: pass_rate \d",
                "pass_rate",
                "snapshot show baseline",
            ),
        ],
    )


def test_snapshot_show_audits_prior_score(monkeypatch, capsys, tmp_path):
    # Round-4 A2 (impl): a promotion line prints a prior_score
    # number, so the show audit covers it (2/2), never silently.
    store = str(tmp_path / "store")
    promotions = [
        {
            "failure_id": "t-1",
            "kind": "retry_trace",
            "promoted_case_id": "mined:repeated_tool_calls:t-1",
            "prior_score": 0.0,
            "evaluator_notes": "tool 'x' called 3 times",
            "promoted_at": "2026-11-08T00:00:00+00:00",
        }
    ]
    promo_path = _write_json(promotions)
    try:
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
                "--promotions",
                promo_path,
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
        assert "prior_score=0.0" in out
        assert "Metric ledger (snapshot show): 2/2 metrics clean" in out
        assert "prior_score" in out
        _assert_stdout_covered(
            out,
            "snapshot show (promotions)",
            [
                (
                    r"^created: \d{4}-\d\d-\d\dT",
                    None,
                    "snapshot show created timestamp",
                ),
                (
                    r"^cases: \d+ +baseline: pass_rate \d",
                    "pass_rate",
                    "snapshot show baseline",
                ),
                (
                    r"^  - \S+ <- \S+ \(prior_score=",
                    "prior_score",
                    "snapshot show promotion prior_score",
                ),
            ],
        )
    finally:
        os.unlink(promo_path)
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


# --- Growth safety (round-4 arch B3) ---------------------------------------
# The audit can only flag metrics it is told about, so a command that
# grows a printed metric without a ledger entry stays green under the
# audit alone — the arch reviewer's mutation proof (a new printed
# metric in cmd_simpson, 70/70 tests green, audit 2/2 clean). The
# backstop is this scan: every stdout line that renders a number must
# match a declared rendering pattern tied to an audited metric (or a
# declared component of one). A new metric line matches no pattern
# and fails here. Integer counts are the documented boundary: the
# scan gates on decimal/percentage/z/chi2 renderings plus the three
# regress integer shapes (mined/corpus/streak); bare integer pipeline
# counts (case totals, n=, seeds) are components under the docs rule
# and are asserted by the per-command audit tests, not this scan.


_METRIC_VALUE = re.compile(
    r"\d+\.\d+\s*%|\d+\.\d+|z\s*=\s*[-+]?\d|chi2\s*=\s*[-+]?\d"
    r"|mined \d+|corpus: \d+ trace|consecutive failures: \d+"
)

_AUDIT_LINE = re.compile(
    r"^  [a-z_0-9]+ +(?:hard|soft) backup=|^    - |^  FLAGGED: "
)


def _split_audit_blocks(out: str) -> tuple[list[str], list[str]]:
    """Split stdout into audit-block lines and everything else.

    A report can carry several audit blocks (tiered gate prints the
    inner run reports), so every block is excluded, not just the
    first. A block is its title line plus the metric rows,
    assumption lines, and FLAGGED line that follow it.
    """
    lines = out.splitlines()
    in_audit = [False] * len(lines)
    i = 0
    while i < len(lines):
        if lines[i].startswith("Metric ledger ("):
            in_audit[i] = True
            i += 1
            while i < len(lines) and _AUDIT_LINE.match(lines[i]):
                in_audit[i] = True
                i += 1
        else:
            i += 1
    audit_lines = [l for l, flag in zip(lines, in_audit) if flag]
    rest = [l for l, flag in zip(lines, in_audit) if not flag]
    return audit_lines, rest


def _assert_stdout_covered(
    out: str, command: str, patterns: list[tuple[str, str | None, str]]
) -> None:
    """Every numeric metric line must match a declared rendering.

    ``patterns`` are (regex, audited metric or None, note) triples:
    the regex matches one rendering, the metric is the audited entry
    it renders (None for a declared component, e.g. a phase
    duration). Each pattern must match at least one line (no dead
    declarations), and each named metric must appear in the audit
    blocks the output carries.
    """
    compiled = [(re.compile(rx), metric, note) for rx, metric, note in patterns]
    audit_lines, rest = _split_audit_blocks(out)
    audited: set[str] = set()
    for line in audit_lines:
        m = re.match(r"^  ([a-z_0-9]+) +(?:hard|soft) backup=", line)
        if m:
            audited.add(m.group(1))
    uncovered: list[str] = []
    used: set[int] = set()
    for line in rest:
        if not _METRIC_VALUE.search(line):
            continue
        hit = False
        for idx, (rx, _metric, _note) in enumerate(compiled):
            if rx.search(line):
                hit = True
                used.add(idx)
        if not hit:
            uncovered.append(line)
    assert not uncovered, (
        f"{command}: numeric metric lines the audit does not cover:\n"
        + "\n".join(uncovered)
    )
    for idx, (_rx, _metric, note) in enumerate(compiled):
        assert idx in used, (
            f"{command}: declared rendering never printed: {note}"
        )
    for _rx, metric, note in compiled:
        if metric is not None:
            assert metric in audited, (
                f"{command}: rendering {note!r} names {metric}, which "
                "the audit does not cover"
            )


def test_tracebench_regress_carries_ledger(monkeypatch, capsys, tmp_path):
    # Round-4 B2 (arch) + B5 (impl): the gate verdict rests on the
    # candidate pass rate, so regress audits it — read back from the
    # comparison object actually written to report.json — plus the
    # mined/new/updated/retired counts and the failure streak, and
    # the ledger travels in the report file itself.
    mine_dir = str(tmp_path / "mine")
    store = str(tmp_path / "store")
    out_dir = str(tmp_path / "out")
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
            mine_dir,
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0, out
    code, out, _ = _run(
        [
            "tracebench",
            "regress",
            "--corpus",
            _examples("traces/production-traces.sample.jsonl"),
            "--store",
            store,
            "--suite",
            "support-agent-golden",
            "--version",
            "v2",
            "--candidate-outputs",
            str(Path(mine_dir) / "oracle-outputs.json"),
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
    assert "Metric ledger (tracebench regress): 3/3 metrics clean" in out
    assert "not yet covered" not in out
    _assert_stdout_covered(
        out,
        "tracebench regress",
        [
            (
                (
                    r"^Tracebench regression: \S+ \(mined \d+, new \d+, "
                    r"updated \d+, retired \d+\)"
                ),
                "regression_counts",
                "regress mined/new/updated/retired counts",
            ),
            (
                r"^corpus: \d+ trace\(s\)",
                None,
                "regress corpus trace count (pipeline input)",
            ),
            (
                r"^consecutive failures: \d+",
                "regression_streak",
                "regress failure streak",
            ),
        ],
    )
    with open(Path(out_dir) / "report.json", encoding="utf-8") as f:
        payload = json.load(f)
    ledger = payload["metric_ledger"]
    assert ledger["all_ok"] is True
    assert set(ledger["metrics"]) == {
        "pass_rate",
        "regression_counts",
        "regression_streak",
    }
    assert payload["comparison"]["candidate_pass_rate"] == 1.0


def test_growth_scan_catches_the_arch_mutation(monkeypatch, capsys):
    # Round-4 arch B3, pinned verbatim: the mutation that proved the
    # hole — a new printed metric in cmd_simpson, no ledger change —
    # must fail the scan. The check runs on real simpson stdout plus
    # the mutation line, so it pins the checker itself.
    path = _write_json(STABLE_SLICES)
    try:
        code, out, _ = _run(
            ["simpson", "--slices", path], monkeypatch, capsys
        )
    finally:
        os.unlink(path)
    assert code == 0, out
    mutated = out + (
        "  win rate 87.0% (new printed metric, never added to the ledger)\n"
    )
    with pytest.raises(AssertionError, match="does not cover"):
        _assert_stdout_covered(
            mutated,
            "simpson",
            [
                (r"^  \S+: treated \d", "simpson_stratum_rate", "stratum line"),
                (r"^  \S+: treated \d", "treated_share", "stratum share"),
                (r"^pooled \(descriptive only", "simpson_pooled_delta", "pooled"),
            ],
        )
