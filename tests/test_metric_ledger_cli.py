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
        [
            (r"^Pass rate: \d+(?:\.\d+)?%", "pass_rate", "run pass-rate line"),
            (
                r"^Validity: \d+/\d+ metrics trusted",
                None,
                (
                    "validity table trusted count (grade count, not a "
                    "ledger metric: the table grades matcher proxies)"
                ),
            ),
        ],
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
            (
                r"^Validity labels: ",
                "label_agreement",
                "labels agreement line",
            ),
            (
                r"^  contains +\S+ +agreement 95% CI=\[",
                "label_agreement",
                "validated matcher agreement row",
            ),
            (
                r"^Validity: \d+/\d+ metrics trusted",
                None,
                (
                    "validity table trusted count (grade count, not a "
                    "ledger metric: the table grades matcher proxies)"
                ),
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
            (
                r"^Validity: \d+/\d+ metrics trusted",
                None,
                (
                    "validity table trusted count (grade count, not a "
                    "ledger metric: the table grades matcher proxies)"
                ),
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
    # Round-5 B1 (all three lenses): the headline names the
    # calibration gap, never ECE, as unbacked — the ledger audits ECE
    # clean and flags the gap, and the artifact must agree with its
    # own ledger. The flagged gap number is printed on the bin line,
    # so the flagged metric has a visible value.
    assert (
        "calibration gap unbacked (see ledger): "
        "do not quote it as verified" in out
    )
    assert "ECE unbacked" not in out
    assert "not on the calibration gap" in out
    assert re.search(r"^  bin \[.*gap -?\d+\.\d\d", out, re.MULTILINE)
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
    # Growth scan with patterns: the too-few headline carries the
    # observation count (an integer+unit rendering) and the reason
    # line carries the policy numbers — both are components, not
    # metrics: no metric was measured, so there is nothing to audit.
    _assert_stdout_covered(
        out,
        "calibrate (too few)",
        [
            (
                r"^judge \S+: \w+ \(\d+ observations, ECE n/a\)",
                None,
                "calibrate too-few headline (no metric measured)",
            ),
            (
                r"^  ! only \d+ observations, policy needs \d+",
                None,
                "calibrate too-few reason (policy config)",
            ),
        ],
    )


def test_calibrate_json_carries_status_note_when_flagged(monkeypatch, capsys, tmp_path):
    # Round-5 B2 (product): the JSON must not certify the flagged
    # number by itself — status is qualified at the point of the
    # verdict, the bin-check grade is preserved in "grade", and the
    # status note scopes the grade to the bin checks next to the
    # ledger flag. The note names the calibration gap, never ECE.
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
    assert payload["status"] == "calibrated (flagged: calibration_gap unbacked)"
    assert payload["grade"] == "calibrated"
    assert payload["metric_ledger"]["all_ok"] is False
    assert "status_note" in payload
    assert "unbacked" in payload["status_note"]
    assert "calibration gap" in payload["status_note"]
    assert "(ECE)" not in payload["status_note"]
    with open(out_path, encoding="utf-8") as f:
        written = json.load(f)
    assert written["status_note"] == payload["status_note"]
    assert written["status"] == payload["status"]
    assert written["grade"] == "calibrated"


def test_calibrate_strict_ledger_refuses(monkeypatch, capsys, tmp_path):
    # The gap is flagged by design, so strict mode refuses the report.
    # Round-5 B2 (product): the refused file on disk must say
    # refused, not calibrated, with the grade preserved.
    out_path = str(tmp_path / "cal_strict.json")
    code, _out, err = _run(
        [
            "calibrate",
            "--observations",
            _examples("calibration-observations.json"),
            "--strict-ledger",
            "--json",
            "--out",
            out_path,
        ],
        monkeypatch,
        capsys,
    )
    assert code == 2
    assert "Ledger gate (strict)" in err
    with open(out_path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["status"] == "refused (ledger gate)"
    assert payload["grade"] == "calibrated"
    assert payload["metric_ledger"]["all_ok"] is False


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
            (
                r"^Validity: \d+/\d+ metrics trusted",
                None,
                (
                    "validity table trusted count (grade count, not a "
                    "ledger metric: the table grades matcher proxies)"
                ),
            ),
            (
                r"^Replay bit-exact: \d+ cases, seed=\d+",
                None,
                "replay bit-exact case count (pipeline bookkeeping)",
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
# The backstop is this scan: every stdout line that renders a number must
# match a declared rendering pattern tied to an audited metric (or a
# declared component of one). A new metric line matches no pattern
# and fails here. The detector sees numbers, not formats: decimals,
# integer percents, fractions, integer counts with a unit noun
# ("12 tests", "42 cases", "7 instances"), and labeled bare integers
# ("score 87"). The remaining boundary, documented so it cannot drift
# silently: bare integers with no unit, label, percent, or fraction
# shape (seeds, n=, digests, version ids, timestamps) are components
# under the docs rule and are asserted by the per-command audit tests,
# not this scan — the dispatch test is the backstop for a new command,
# and a new metric printed in one of those shapes must grow a pattern
# here, not hide behind the boundary.


_METRIC_VALUE = re.compile(
    r"\d+\.\d+\s*%|\d+\.\d+|\d+\s*%|\d+/\d+"
    r"|z\s*=\s*[-+]?\d|chi2\s*=\s*[-+]?\d"
    r"|mined \d+|corpus: \d+ trace|consecutive failures: \d+"
    r"|\d+ (?:tests?|cases?|instances?|runs?|bins?|observations?|scores?|points?|versions?|traces?|prompts?|payloads?|candidates?)\b"
    r"|\b(?:score|rate|total|count|recorded)\s*:?\s*\d+\b"
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


def test_run_ci_scans_the_gate_line(monkeypatch, capsys):
    # Round-5 (arch B2): run --ci prints the CI gate line, whose
    # threshold renders as an integer percent the old detector could
    # not see. The gate line is the same pass_rate the audit covers.
    code, out, _ = _run(
        ["run", "--suite", _examples("support-agent.yaml"), "--ci"],
        monkeypatch,
        capsys,
    )
    assert code in (0, 1, 2)
    assert "CI gate: pass_rate" in out
    _assert_stdout_covered(
        out,
        "run --ci",
        [
            (r"^Pass rate: \d+(?:\.\d+)?%", "pass_rate", "run pass-rate line"),
            (
                r"^CI gate: pass_rate ",
                "pass_rate",
                "run CI gate line (same pass_rate, integer threshold)",
            ),
            (
                r"^Validity: \d+/\d+ metrics trusted",
                None,
                (
                    "validity table trusted count (grade count, not a "
                    "ledger metric: the table grades matcher proxies)"
                ),
            ),
        ],
    )


def test_perturb_unstable_scans_drop_reasons(monkeypatch, capsys, tmp_path):
    # Round-5 N3 (impl): the UNSTABLE branch prints the drop reason
    # ("clean-to-adversarial drop 100.0% exceeds max 50%") — the
    # numbers are the audited drop and the policy threshold, so the
    # reason line is a rendering of perturbation_stability.
    import yaml

    triple = yaml.safe_load(
        Path(_examples("perturbation-triple.yaml")).read_text(encoding="utf-8")
    )
    for case in triple["cases"]:
        case["output_adversarial"] = "PERTURBATION-PROBE-REFUSAL"
    suite_path = tmp_path / "unstable.yaml"
    suite_path.write_text(yaml.safe_dump(triple), encoding="utf-8")
    code, out, _ = _run(
        ["perturb", "--suite", str(suite_path)], monkeypatch, capsys
    )
    assert code == 1
    assert "verdict: UNSTABLE" in out
    assert "Metric ledger (perturb):" in out
    _assert_stdout_covered(
        out,
        "perturb UNSTABLE",
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
            (
                r"^  ! clean-to-adversarial drop ",
                "perturbation_stability",
                "perturb UNSTABLE drop reason",
            ),
        ],
    )


def test_calibrate_non_default_policy_scans_policy_line(monkeypatch, capsys):
    # Round-5 N3 (impl): a non-default policy prints the policy line,
    # whose tolerance is a config decimal — a declared component, the
    # same class as alt-test's epsilon.
    code, out, _ = _run(
        [
            "calibrate",
            "--observations",
            _examples("calibration-observations.json"),
            "--bins",
            "5",
            "--tolerance",
            "0.2",
        ],
        monkeypatch,
        capsys,
    )
    assert code in (0, 2)
    assert "policy (non-default)" in out
    _assert_stdout_covered(
        out,
        "calibrate (non-default policy)",
        [
            (
                r"^judge \S+: \w+ \(\d+ observations, ECE \d",
                "ece",
                "calibrate headline ECE",
            ),
            (
                r"^  policy \(non-default\): ",
                None,
                "calibrate policy line (config, not a metric)",
            ),
            (
                r"^  bin \[",
                "agreement_rate",
                "calibrate bin line (agreement + gap components)",
            ),
        ],
    )


def _patch_tiered_fakes(monkeypatch):
    """Run the real tiered pipeline against the fake executor + fake
    mutation scorer test_tiered.py uses, so the CLI scan tests stay
    fast: no nested full-suite run. Monkeypatches
    ``agenteval_bench.tiered.run_pipeline`` in place.
    """
    import json as _json

    from agenteval_bench import tiered
    from agenteval_bench.tiered import CommandResult, MutationScore
    from agenteval_bench.tiered import run_pipeline as _real_run_pipeline

    class _FakeExecutor:
        """Records argv; fakes pytest's coverage side effect."""

        def __init__(self) -> None:
            self.coverage_percent = 85.0

        def __call__(self, argv, cwd, env=None):  # type: ignore[no-untyped-def]
            for arg in argv:
                if arg.startswith("--cov-report=json:"):
                    path = Path(arg.split(":", 1)[1])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        _json.dumps(
                            {"totals": {"percent_covered": self.coverage_percent}}
                        ),
                        encoding="utf-8",
                    )
            return CommandResult(argv=tuple(argv), returncode=0, stdout="")

    def _fake_mutation(root, *, mutants, seed, work_dir):  # type: ignore[no-untyped-def]
        return MutationScore(killed=6, validated=8, skipped=0, score=0.75)

    def _patched(root, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["executor"] = _FakeExecutor()
        kwargs["mutation_fn"] = _fake_mutation
        return _real_run_pipeline(root, *args, **kwargs)

    monkeypatch.setattr(tiered, "run_pipeline", _patched)


def test_tiered_expanded_scans_coverage_and_mutation(monkeypatch, capsys):
    # Round-5 N3 (impl): the expanded phase prints coverage — the
    # mutation score is measured only by the nightly phase (see the
    # nightly test below). The real pipeline runs against the fake
    # executor + fake mutation scorer, so this stays fast.
    _patch_tiered_fakes(monkeypatch)
    code, out, _ = _run(
        ["tiered", "--phase", "expanded", "--root", "."], monkeypatch, capsys
    )
    assert code == 0, out
    assert "Metric ledger (tiered): 1/1 metrics clean" in out
    _assert_stdout_covered(
        out,
        "tiered expanded",
        [
            (
                r"^phase \w+: \w+ \(\d+\.\d+s\)",
                None,
                "tiered phase duration (timing)",
            ),
            (
                r"^coverage: \d+\.\d+%",
                "coverage_pct",
                "tiered expanded coverage line",
            ),
        ],
    )


def test_tiered_nightly_scans_mutation_score(monkeypatch, capsys):
    # Round-5 N3 (impl): the nightly phase prints the mutation score,
    # which renders as an integer percent — the shape the old
    # decimal-only detector could not see.
    _patch_tiered_fakes(monkeypatch)
    code, out, _ = _run(
        ["tiered", "--phase", "nightly", "--root", "."], monkeypatch, capsys
    )
    assert code == 0, out
    assert "Metric ledger (tiered): 2/2 metrics clean" in out
    _assert_stdout_covered(
        out,
        "tiered nightly",
        [
            (
                r"^phase \w+: \w+ \(\d+\.\d+s\)",
                None,
                "tiered phase duration (timing)",
            ),
            (
                r"^coverage: \d+\.\d+%",
                "coverage_pct",
                "tiered nightly coverage line",
            ),
            (
                r"^mutation score: \d+%",
                "mutation_score",
                "tiered nightly mutation line (integer percent)",
            ),
            (
                r"^  note: mutation score \d+/\d+ = \d+%",
                "mutation_score",
                "tiered nightly mutation note (fraction + integer percent)",
            ),
        ],
    )


def test_flywheel_scans_gate_catch_rate(monkeypatch, capsys, tmp_path):
    # Round-5 N3 (impl): the flywheel rate renders as an integer
    # percent ("leading gate-catch rate: 100%"), which the old
    # decimal-only detector could not see. Runs the real loop with
    # the recorded-transport stub test_flywheel.py uses.
    from flywheel import FixCommit, is_fix_commit

    class _RecordedTransport:
        """Hand-written stub transport carrying recorded commits."""

        def __init__(self, commits):  # type: ignore[no-untyped-def]
            self._commits = commits

        def list_fix_commits(self, repo, since=None):  # type: ignore[no-untyped-def]
            del since
            return [
                c
                for c in self._commits
                if c.repo == repo and is_fix_commit(c.message)
            ]

    commits = [
        FixCommit(sha=s, message=m, repo="Zion-sewers/arena")
        for s, m in [
            ("a1b2c3d4", "fix: replace bare except in engine"),
            ("b2c3d4e5", "fix: blind except swallowed input"),
            ("c3d4e5f6", "fix: mock leaked across tests"),
            ("d4e5f6a7", "fix: remove mock of the boundary"),
            ("e5f6a7b8", "feat: add scoreboard"),
        ]
    ]
    monkeypatch.setattr(
        "flywheel.detect.GhCliTransport",
        lambda: _RecordedTransport(commits),
    )
    target = tmp_path / "target"
    (target / "src").mkdir(parents=True)
    (target / "src" / "a.py").write_text(
        "def f() -> int:\n    return 1\n", encoding="utf-8"
    )
    (target / "pyproject.toml").write_text(
        '[project]\nname = "t"\n', encoding="utf-8"
    )
    corpus = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "bugsmith"
        / "fixtures"
        / "target"
    )
    code, out, _ = _run(
        [
            "flywheel",
            "--repo",
            "Zion-sewers/arena",
            "--target-root",
            str(target),
            "--corpus-root",
            str(corpus),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0, out
    assert "Metric ledger (flywheel): 1/1 metrics clean" in out
    _assert_stdout_covered(
        out,
        "flywheel",
        [
            (
                r"^detected=\d+ sampled=\d+ payloads=\d+",
                None,
                "flywheel loop counts (pipeline bookkeeping)",
            ),
            (
                r"^ledger: .* \(leading gate-catch rate: \d+%\)",
                "gate_catch_rate",
                "flywheel gate-catch rate (integer percent)",
            ),
        ],
    )


def test_tracebench_regress_gate_fail_scans_reasons(monkeypatch, capsys, tmp_path):
    # Round-5 N3 (impl): a GATE_FAIL regress prints the gate reasons,
    # which carry decimals ("aggregate pass_rate 50.0% below threshold
    # 100.0%") — the failing rate is the audited pass_rate.
    mine_dir = str(tmp_path / "mine")
    store = str(tmp_path / "store")
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
    oracle = Path(mine_dir) / "oracle-outputs.json"
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
            "v1",
            "--candidate-outputs",
            str(oracle),
            "--owner",
            "test",
            "--cadence",
            "weekly",
            "--out",
            str(tmp_path / "run1"),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0, out
    assert "verdict: GATE_PASS" in out
    degraded = {
        case_id: "wrong answer"
        for case_id in json.loads(oracle.read_text(encoding="utf-8"))
    }
    degraded_path = tmp_path / "degraded.json"
    degraded_path.write_text(json.dumps(degraded), encoding="utf-8")
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
            "--prev",
            "v1",
            "--candidate-outputs",
            str(degraded_path),
            "--owner",
            "test",
            "--cadence",
            "weekly",
            "--out",
            str(tmp_path / "run2"),
            "--prior-failures",
            "2",
        ],
        monkeypatch,
        capsys,
    )
    assert code == 2, out
    assert "verdict: GATE_FAIL" in out
    assert "action: rollback" in out
    _assert_stdout_covered(
        out,
        "tracebench regress GATE_FAIL",
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
            (
                r"^  action: ",
                "pass_rate",
                "regress GATE_FAIL action (the failing rate)",
            ),
        ],
    )


def test_bugsmith_scans_validated_share(monkeypatch, capsys, tmp_path):
    # Round-5 (arch B2): bugsmith's "validated: A/B" line is a
    # reported proportion — the validated share, a HARD ledger entry.
    # The baseline/curated/wrote lines are pipeline counts,
    # components of it.
    repo = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "bugsmith"
        / "fixtures"
        / "target"
    )
    out = tmp_path / "bench"
    code, stdout, _ = _run(
        [
            "bugsmith",
            "--repo",
            str(repo),
            "--out",
            str(out),
            "--procedural",
            "4",
            "--seed",
            "5",
            "--local",
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0, stdout
    assert "Metric ledger (bugsmith): 1/1 metrics clean" in stdout
    _assert_stdout_covered(
        stdout,
        "bugsmith",
        [
            (
                r"^baseline: \d+ tests passing",
                None,
                "bugsmith baseline count (pipeline input)",
            ),
            (
                r"^validated: \d+/\d+ instances break",
                "validated_share",
                "bugsmith validated share",
            ),
            (
                r"^curated: \d+ instances",
                None,
                "bugsmith curated count (pipeline bookkeeping)",
            ),
            (
                r"^wrote \d+ instances \+ manifest",
                None,
                "bugsmith wrote count (pipeline bookkeeping)",
            ),
        ],
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metric_ledger"]["all_ok"] is True
    assert set(manifest["metric_ledger"]["metrics"]) == {"validated_share"}


_REPLAY_SUITE = """
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


def _write_replay_suite(tmp_path: Path) -> str:
    path = tmp_path / "suite.yaml"
    path.write_text(_REPLAY_SUITE, encoding="utf-8")
    return str(path)


def test_replay_scans_bit_exact_counts(monkeypatch, capsys, tmp_path):
    # Round-5 (arch B2): replay prints case counts — the `replay`
    # command prints "Replayed N cases" when it rewrites the log and
    # "Replay bit-exact: N cases" under --check — pipeline
    # bookkeeping, components, pinned by a scan test so a future
    # metric printed here cannot hide.
    suite = _write_replay_suite(tmp_path)
    log = str(tmp_path / "replay.json")
    code, run_out, _ = _run(
        ["run", "--suite", suite, "--replay-log", log], monkeypatch, capsys
    )
    assert code == 0
    _assert_stdout_covered(
        run_out,
        "run --replay-log",
        [
            (
                r"^Pass rate: \d+(?:\.\d+)?%",
                "pass_rate",
                "run --replay-log pass-rate line",
            ),
            (
                r"^Validity: \d+/\d+ metrics trusted",
                None,
                (
                    "validity table trusted count (grade count, not a "
                    "ledger metric: the table grades matcher proxies)"
                ),
            ),
        ],
    )
    code, out, _ = _run(
        ["replay", "--log", log, "--suite", suite], monkeypatch, capsys
    )
    assert code == 0, out
    assert "Replayed 2 cases" in out
    _assert_stdout_covered(
        out,
        "replay (rewrite)",
        [
            (
                r"^Replayed \d+ cases \(seed=\d+\)",
                None,
                "replay log case count (pipeline bookkeeping)",
            ),
        ],
    )
    code, out, _ = _run(
        ["replay", "--log", log, "--suite", suite, "--check"], monkeypatch, capsys
    )
    assert code == 0, out
    assert "Replay bit-exact" in out
    _assert_stdout_covered(
        out,
        "replay",
        [
            (
                r"^Replay bit-exact: \d+ cases, seed=\d+",
                None,
                "replay bit-exact case count (pipeline bookkeeping)",
            ),
        ],
    )


def test_judges_scans_picked_count(monkeypatch, capsys, tmp_path):
    # Round-5 (arch B2): judges prints an allocation fraction
    # ("picked (2/2)") — an allocation count, not a scored metric.
    registry = {
        "judges": [
            {"id": "a", "host": "h1", "model_family": "fam-a", "reachable": "measured"},
            {"id": "b", "host": "h2", "model_family": "fam-b", "reachable": "measured"},
            {"id": "dead", "host": "h3", "model_family": "fam-c", "reachable": "unreachable"},
        ]
    }
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(registry), encoding="utf-8")
    out_path = tmp_path / "allocation.json"
    code, out, _ = _run(
        ["judges", "--registry", str(reg_path), "--want", "2", "--out", str(out_path)],
        monkeypatch,
        capsys,
    )
    assert code == 0, out
    _assert_stdout_covered(
        out,
        "judges",
        [
            (
                r"^picked \(\d+/\d+\)",
                None,
                "judges picked allocation count (not a scored metric)",
            ),
        ],
    )


def test_baseline_scans_snapshot_count(monkeypatch, capsys, tmp_path):
    # Round-5 (arch B2): baseline prints a score count ("Baseline
    # snapshot: 2 scores") — a pipeline count, and a notice saying
    # the frozen scores are not an aggregate metric.
    suite = _write_replay_suite(tmp_path)
    out_path = str(tmp_path / "baseline.json")
    code, out, _ = _run(
        ["baseline", "--suite", suite, "--out", out_path, "--seed", "42"],
        monkeypatch,
        capsys,
    )
    assert code == 0, out
    assert "not a reported aggregate metric" in out
    _assert_stdout_covered(
        out,
        "baseline",
        [
            (
                r"^Baseline snapshot: \d+ scores \(seed=\d+\)",
                None,
                "baseline score count (pipeline bookkeeping)",
            ),
        ],
    )


def test_register_design_scans_nothing_numeric(monkeypatch, capsys, tmp_path):
    # Round-5 (arch B2): register-design prints only a registration
    # digest — no numeric metric line. The zero-pattern scan proves
    # it: any future number on this output fails here.
    from experiment.oec import ExperimentDesign, MetricDeclaration
    from experiment.registry import register_plan
    from experiment.types import ExperimentPlan

    registry = str(tmp_path / "experiments.jsonl")
    plan_digest = register_plan(
        ExperimentPlan(name="reranker-v2-vs-v1", n_planned=800, alpha=0.05, looks=(0.5, 1.0)),
        registry,
    )
    design = ExperimentDesign(
        plan_digest=plan_digest,
        metrics=(
            MetricDeclaration("task_success", "oec", "higher_better"),
            MetricDeclaration("safety_violations", "guardrail", "lower_better", 0.02),
        ),
        expected_allocation=(0.5, 0.5),
    )
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design.to_dict()), encoding="utf-8")
    code, out, _ = _run(
        ["register-design", "--design", str(design_path), "--registry", registry],
        monkeypatch,
        capsys,
    )
    assert code == 0, out
    assert "registered design" in out
    _assert_stdout_covered(out, "register-design", [])


_PRESPEC_PLAN = {
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


def _prespec_files(tmp_path, hypothesis, accessed=None):
    """Build plan + finding + registry files for the prespec CLI tests.

    ``accessed`` defaults to one hour after the plan's registration
    time: the finding must claim data accessed after the plan was
    written (Q2), and a hardcoded absolute timestamp is a time-bomb.
    """
    from datetime import UTC, datetime, timedelta

    from experiment import load_analysis_plan, plan_from_dict, register_analysis_plan

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_PRESPEC_PLAN), encoding="utf-8")
    registry = str(tmp_path / "analyses.jsonl")
    digest = register_analysis_plan(plan_from_dict(_PRESPEC_PLAN), registry)
    _, registered_at = load_analysis_plan(digest, registry)
    if accessed is None:
        parsed = datetime.fromisoformat(registered_at)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        accessed = (parsed + timedelta(hours=1)).isoformat()
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


def test_register_analysis_plan_scans_nothing_numeric(monkeypatch, capsys, tmp_path):
    # Round-5 (arch B2): register-analysis-plan prints only a
    # registration digest — no numeric metric line.
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_PRESPEC_PLAN), encoding="utf-8")
    code, out, _ = _run(
        [
            "register-analysis-plan",
            "--plan",
            str(plan_path),
            "--registry",
            str(tmp_path / "analyses.jsonl"),
        ],
        monkeypatch,
        capsys,
    )
    assert code == 0, out
    assert "registered analysis plan" in out
    _assert_stdout_covered(out, "register-analysis-plan", [])


def test_prespec_scans_review_gate(monkeypatch, capsys, tmp_path):
    # Round-5 (arch B2): prespec prints a 10-question review gate —
    # a verdict label plus question lines, no numeric metric. The
    # scan pins the gate output so a future metric printed here
    # cannot hide.
    plan, finding, registry = _prespec_files(
        tmp_path,
        "reranker v2 cuts hallucinated citations by at least 10pp vs v1",
    )
    code, out, _ = _run(
        ["prespec", "--finding", finding, "--plan", plan, "--registry", registry],
        monkeypatch,
        capsys,
    )
    assert code == 0, out
    assert "verdict: CONFIRMED" in out
    # The question lines quote evidence text (timestamps, the stated
    # interval, the decision rule's 95%) — quoted from the plan and
    # finding docs, not numbers the CLI computes or certifies. The
    # gate's verdict is the label; the evidence lines are a declared
    # component class, not metrics.
    _assert_stdout_covered(
        out,
        "prespec",
        [
            (
                r"^  \[(PASS|FAIL|SKIP)\] Q\d+ ",
                None,
                "prespec question line (quoted evidence, not a metric)",
            ),
        ],
    )


_CLI_SNAPSHOT_SUITE = """
name: cli-smoke
cases:
  - id: a
    input: "ping"
    output: "pong ok"
    expected:
      contains: ["pong"]
  - id: b
    input: "ding"
    output: "dong"
    expected:
      exact: "dong"
"""


def _snapshot_publish(monkeypatch, capsys, tmp_path, promotions=None):
    """Publish v1 of the CLI smoke suite; return (store, suite_path)."""
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(_CLI_SNAPSHOT_SUITE, encoding="utf-8")
    store = str(tmp_path / "store")
    argv = [
        "snapshot",
        "publish",
        "--suite",
        str(suite_path),
        "--store",
        store,
        "--version",
        "v1",
        "--owner",
        "eval-team",
        "--why",
        "smoke baseline",
        "--changed",
        "initial freeze",
        "--cadence",
        "weekly",
    ]
    if promotions is not None:
        prom_path = tmp_path / "promotions.json"
        prom_path.write_text(json.dumps(promotions), encoding="utf-8")
        argv += ["--promotions", str(prom_path)]
    code, out, err = _run(argv, monkeypatch, capsys)
    assert code == 0, err
    return store, out


def test_snapshot_publish_scans_baseline_line(monkeypatch, capsys, tmp_path):
    # Round-5 (arch B2): publish prints the baseline pass rate —
    # the audited pass_rate — plus a promotions count component.
    _store, out = _snapshot_publish(
        monkeypatch,
        capsys,
        tmp_path,
        promotions=[
            {
                "failure_id": "esc-1042",
                "kind": "escalation",
                "promoted_case_id": "a",
                "prior_score": 1.0,
                "evaluator_notes": "smoke promotion",
                "promoted_at": "2026-11-08T00:00:00+00:00",
            }
        ],
    )
    assert "Metric ledger (snapshot publish): 1/1 metrics clean" in out
    _assert_stdout_covered(
        out,
        "snapshot publish",
        [
            (
                r"^Baseline: pass_rate ",
                "pass_rate",
                "snapshot publish baseline pass rate",
            ),
            (
                r"^Promotions recorded: \d+",
                None,
                "snapshot publish promotions count (pipeline bookkeeping)",
            ),
        ],
    )


def test_snapshot_verify_list_check_repair_scan(monkeypatch, capsys, tmp_path):
    # Round-5 (arch B2): the read/repair subcommands print no
    # metric — only digests, version ids, and counts of corrupted or
    # repaired versions, pinned as components.
    store, _ = _snapshot_publish(monkeypatch, capsys, tmp_path)
    code, out, _ = _run(
        ["snapshot", "verify", "--store", store, "--suite", "cli-smoke", "--version", "v1"],
        monkeypatch,
        capsys,
    )
    assert code == 0, out
    assert "OK: cli-smoke/v1 hash-verified" in out
    _assert_stdout_covered(out, "snapshot verify", [])
    code, out, _ = _run(
        ["snapshot", "list", "--store", store], monkeypatch, capsys
    )
    assert code == 0, out
    assert "cli-smoke/v1" in out
    _assert_stdout_covered(out, "snapshot list", [])
    code, out, _ = _run(
        ["snapshot", "check", "--store", store], monkeypatch, capsys
    )
    assert code == 0, out
    assert "OK: all published versions hash-verified" in out
    _assert_stdout_covered(out, "snapshot check", [])
    (tmp_path / "store" / "journal.jsonl").unlink()
    code, out, _ = _run(
        ["snapshot", "repair", "--store", store, "--reason", "crash recovery"],
        monkeypatch,
        capsys,
    )
    assert code == 0, out
    assert "snapshot-repair" in out
    _assert_stdout_covered(
        out,
        "snapshot repair",
        [
            (
                r"^Repaired \d+ version\(s\)",
                None,
                "snapshot repair count (pipeline bookkeeping)",
            ),
        ],
    )


def test_growth_scan_catches_integer_format_mutations(monkeypatch, capsys):
    # Round-5 (arch B2): the pinned mutation in its integer shapes —
    # an integer percent, a fraction, an integer count, and a labeled
    # bare integer — must fail the scan the same way the decimal form
    # does.
    path = _write_json(STABLE_SLICES)
    try:
        code, out, _ = _run(["simpson", "--slices", path], monkeypatch, capsys)
    finally:
        os.unlink(path)
    assert code == 0, out
    patterns = [
        (r"^  \S+: treated \d", "simpson_stratum_rate", "stratum line"),
        (r"^  \S+: treated \d", "treated_share", "stratum share"),
        (r"^pooled \(descriptive only", "simpson_pooled_delta", "pooled"),
    ]
    for mutated_line in (
        "  win rate 87% (new printed metric, never added to the ledger)\n",
        "  score: 4/5 (new printed metric, never added to the ledger)\n",
        "  baseline: 12 tests passing (new printed metric, never added to the ledger)\n",
        "  score 87 (new printed metric, never added to the ledger)\n",
    ):
        with pytest.raises(AssertionError, match="does not cover"):
            _assert_stdout_covered(out + mutated_line, "simpson", patterns)


# Round-5 dispatch coverage (arch B2, hole 2): every command main()
# can reach must be classified here — a scan test in this file proves
# its stdout has no undeclared numeric metric line, or a named test
# proves it prints the out-of-scope notice. The test below derives
# the reachable set from the CLI source, so a new command (or
# subcommand) fails the test until it is classified. The map value is
# the test function name that proves the classification.
_DISPATCH_COVERAGE: dict[tuple[str, str | None], str] = {
    ("run", None): "test_run_audits_only_pass_rate",
    ("decide", None): "test_decide_audits_only_measured_dimensions",
    ("tiered", None): "test_tiered_nightly_scans_mutation_score",
    ("flywheel", None): "test_flywheel_scans_gate_catch_rate",
    ("tracebench", "mine"): "test_tracebench_mine_prints_not_covered_notice",
    ("tracebench", "regress"): "test_tracebench_regress_carries_ledger",
    ("baseline", None): "test_baseline_scans_snapshot_count",
    ("replay", None): "test_replay_scans_bit_exact_counts",
    ("alt-test", None): "test_alt_test_json_carries_ledger",
    ("simpson", None): "test_simpson_json_carries_ledger",
    ("judges", None): "test_judges_scans_picked_count",
    ("calibrate", None): "test_calibrate_flags_unbacked_gap",
    ("diverge", None): "test_diverge_json_carries_ledger",
    ("perturb", None): "test_perturb_flags_unbacked_stability",
    ("ship", None): "test_ship_json_carries_ledger",
    ("register-design", None): "test_register_design_scans_nothing_numeric",
    ("prespec", None): "test_prespec_scans_review_gate",
    ("register-analysis-plan", None): "test_register_analysis_plan_scans_nothing_numeric",
    ("bugsmith", None): "test_bugsmith_scans_validated_share",
    ("snapshot", "publish"): "test_snapshot_publish_scans_baseline_line",
    ("snapshot", "show"): "test_snapshot_show_audits_pass_rate",
    ("snapshot", "compare"): "test_snapshot_compare_audits_pass_rate",
    ("snapshot", "verify"): "test_snapshot_verify_list_check_repair_scan",
    ("snapshot", "list"): "test_snapshot_verify_list_check_repair_scan",
    ("snapshot", "check"): "test_snapshot_verify_list_check_repair_scan",
    ("snapshot", "repair"): "test_snapshot_verify_list_check_repair_scan",
}


def test_dispatch_every_command_is_classified():
    # Round-5 (arch B2): enumerate every command main() can reach —
    # top-level dispatch plus the tracebench/snapshot subcommand
    # dispatch — and require a classification for each, backed by a
    # scan (or notice) test defined in this file.
    cli_path = Path(__file__).resolve().parents[1] / "src" / "agenteval_bench" / "cli.py"
    source = cli_path.read_text(encoding="utf-8")

    def _fn_slice(start_marker: str, end_marker: str) -> str:
        start = source.index(start_marker)
        end = source.index(end_marker, start)
        return source[start:end]

    main_src = _fn_slice("def main()", "if __name__ ==")
    commands = set(re.findall(r'(?:if|elif) cmd == "([^"]+)"', main_src))
    reachable: set[tuple[str, str | None]] = {(cmd, None) for cmd in commands}
    for parent, sub_src in (
        ("tracebench", _fn_slice("def cmd_tracebench", "def cmd_flywheel")),
        ("snapshot", _fn_slice("def cmd_snapshot", "def cmd_tiered")),
    ):
        subs = set(re.findall(r'if sub == "([^"]+)"', sub_src))
        reachable.discard((parent, None))
        reachable.update((parent, sub) for sub in subs)
    missing = reachable - set(_DISPATCH_COVERAGE)
    assert not missing, (
        "commands with no ledger classification (add a scan test or an "
        f"out-of-scope notice): {sorted(missing)}"
    )
    stale = set(_DISPATCH_COVERAGE) - reachable
    assert not stale, f"classifications for commands that no longer exist: {sorted(stale)}"
    for key, test_name in sorted(_DISPATCH_COVERAGE.items()):
        assert test_name in globals(), (
            f"{key}: proving test {test_name!r} is not defined in this file"
        )
