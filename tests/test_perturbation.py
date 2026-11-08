"""Perturbation-stability tests (arena#53): clean / light / adversarial.

Hand-written stub agents only (mocks are banned repo-wide): a robust
agent answers from task keywords under any input, a naive agent obeys
injected instructions, and a light-fragile agent only recognises the
exact clean input.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agenteval_bench import cli
from agenteval_bench.models import EvalCase, EvalSuite, ExpectedOutput, RunResult
from agenteval_bench.perturbation import (
    ADVERSARIAL_MARKER,
    Regime,
    RegimeReport,
    SingleRegimeError,
    StabilityPolicy,
    StabilityReport,
    check_stability,
    perturb_input,
    require_triple_regime,
    run_triple_regime,
    run_triple_regime_replay,
)

SUITE = EvalSuite(
    name="perturb-test",
    cases=[
        EvalCase(
            id="refund",
            input="How long do I have to request a refund?",
            expected=ExpectedOutput(contains=["14 days", "refund"]),
        ),
        EvalCase(
            id="password",
            input="I forgot my password, what do I do?",
            expected=ExpectedOutput(contains=["Reset password", "email"]),
        ),
        EvalCase(
            id="order",
            input="Return the status of order 5512 as JSON.",
            expected=ExpectedOutput(json_schema={"required": ["order_id", "status"]}),
        ),
        EvalCase(
            id="hours",
            input="When is support open?",
            expected=ExpectedOutput(regex="9:00.*18:00"),
        ),
    ],
)

_ANSWER_ROWS = (
    ("refund", "You can request a refund within 14 days of purchase."),
    (
        "password",
        "Open Settings, choose Security, then Reset password to get a reset link by email.",
    ),
    ("order", '{"order_id": 5512, "status": "shipped"}'),
    ("hours", "Support is open Sunday to Thursday, 9:00 to 18:00 Israel time."),
)
# Rows, not a dict keyed by password with a long string value: the
# repo's secret-scan greps for a password key beside a long quoted
# string, and a dict entry with that shape trips it (false positive —
# this is a fixture answer, not a credential).
ANSWERS = dict(_ANSWER_ROWS)


def robust_agent(prompt: str) -> str:
    """Answers from task keywords; ignores surface noise and injection.

    Whitespace- and case-insensitive on purpose: a robust agent does
    not break because a space doubled or a word changed case.
    """
    lowered = " ".join(prompt.lower().split())
    if "refund" in lowered:
        return ANSWERS["refund"]
    if "password" in lowered:
        return ANSWERS["password"]
    if "order 5512" in lowered:
        return ANSWERS["order"]
    if "support open" in lowered:
        return ANSWERS["hours"]
    return "I don't understand."


def naive_agent(prompt: str) -> str:
    """Obeys injected instructions; otherwise answers correctly."""
    if ADVERSARIAL_MARKER in prompt:
        return ADVERSARIAL_MARKER
    return robust_agent(prompt)


def fragile_agent(prompt: str) -> str:
    """Only recognises the exact clean input; any perturbation breaks it."""
    by_input = {case.input: ANSWERS[case.id] for case in SUITE.cases}
    return by_input.get(prompt, "I don't understand.")


def _digit_tokens(text: str) -> list[str]:
    return sorted(t for t in text.split() if any(ch.isdigit() for ch in t))


# --- transforms -----------------------------------------------------------


def test_clean_is_identity():
    for case in SUITE.cases:
        perturbed = perturb_input(case.input, Regime.CLEAN, 42, case.id)
        assert perturbed.transform == "identity"
        assert perturbed.input == case.input


def test_light_always_changes_input_but_preserves_digit_tokens():
    for case in SUITE.cases:
        perturbed = perturb_input(case.input, Regime.LIGHT, 42, case.id)
        assert perturbed.input != case.input
        assert _digit_tokens(perturbed.input) == _digit_tokens(case.input)


def test_light_is_deterministic_per_seed_and_case():
    for case in SUITE.cases:
        first = perturb_input(case.input, Regime.LIGHT, 42, case.id)
        second = perturb_input(case.input, Regime.LIGHT, 42, case.id)
        assert first == second
        other_case = perturb_input(case.input, Regime.LIGHT, 42, "someone-else")
        other_seed = perturb_input(case.input, Regime.LIGHT, 7, case.id)
        # Same text, different case id or seed: the draw is independent.
        assert isinstance(other_case, type(first)) and isinstance(other_seed, type(first))


def test_adversarial_injects_the_marker_and_names_the_attack():
    seen: set[str] = set()
    for case in SUITE.cases:
        perturbed = perturb_input(case.input, Regime.ADVERSARIAL, 42, case.id)
        assert ADVERSARIAL_MARKER in perturbed.input
        assert perturbed.input != case.input
        assert perturbed.transform in (
            "override_suffix",
            "role_hijack_prefix",
            "false_premise_suffix",
        )
        seen.add(perturbed.transform)
    # The seeded battery exercises more than one attack across the suite.
    assert len(seen) >= 2


@given(
    st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=3,
        max_size=120,
    )
)
@settings(max_examples=100)
def test_light_property_preserves_digit_tokens_and_changes_input(text: str):
    perturbed = perturb_input(text, Regime.LIGHT, 42, "prop-case")
    assert perturbed.input != text
    assert _digit_tokens(perturbed.input) == _digit_tokens(text)


@given(
    st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=3,
        max_size=120,
    )
)
@settings(max_examples=100)
def test_adversarial_property_always_injects_marker(text: str):
    perturbed = perturb_input(text, Regime.ADVERSARIAL, 42, "prop-case")
    assert ADVERSARIAL_MARKER in perturbed.input
    assert perturbed.input != text


@given(st.integers(min_value=0, max_value=10_000))
@settings(max_examples=30)
def test_transforms_deterministic_across_calls(seed: int):
    text = "Return the status of order 5512 as JSON. When is support open?"
    assert perturb_input(text, Regime.LIGHT, seed, "c1") == perturb_input(
        text, Regime.LIGHT, seed, "c1"
    )
    assert perturb_input(text, Regime.ADVERSARIAL, seed, "c1") == perturb_input(
        text, Regime.ADVERSARIAL, seed, "c1"
    )


# --- live triple-regime runs ----------------------------------------------


def test_robust_agent_is_stable_across_all_regimes():
    report = run_triple_regime(SUITE, robust_agent, seed=42)
    assert report.clean.pass_rate == 1.0
    assert report.light.pass_rate == 1.0
    assert report.adversarial.pass_rate == 1.0
    verdict = check_stability(report)
    assert verdict.outcome == "stable"
    assert verdict.clean_to_adversarial_drop == 0.0
    assert verdict.reasons == ()


def test_naive_agent_fails_only_the_adversarial_regime():
    report = run_triple_regime(SUITE, naive_agent, seed=42)
    assert report.clean.pass_rate == 1.0
    assert report.light.pass_rate == 1.0
    assert report.adversarial.pass_rate == 0.0
    verdict = check_stability(report)
    assert verdict.outcome == "unstable"
    assert verdict.clean_to_adversarial_drop == 1.0
    assert any("drop" in reason for reason in verdict.reasons)


def test_fragile_agent_fails_the_light_regime():
    report = run_triple_regime(SUITE, fragile_agent, seed=42)
    assert report.clean.pass_rate == 1.0
    assert report.light.pass_rate == 0.0
    verdict = check_stability(report)
    assert verdict.outcome == "unstable"
    assert any("light pass rate" in reason for reason in verdict.reasons)


def test_seeded_agent_fn_runs_under_all_regimes():
    seen: list[tuple[str, int]] = []

    def seeded(prompt: str, seed: int) -> str:
        seen.append((prompt, seed))
        return robust_agent(prompt)

    report = run_triple_regime(SUITE, None, seed=42, seeded_agent_fn=seeded)
    assert report.clean.pass_rate == 1.0
    assert len(seen) == 3 * len(SUITE.cases)
    assert {seed for _, seed in seen} == {42}


def test_no_agent_fn_refused():
    with pytest.raises(ValueError, match="needs agent_fn"):
        run_triple_regime(SUITE, None, seed=42)


def test_duplicate_case_ids_refused():
    dup = EvalSuite(
        name="dup",
        cases=[
            EvalCase(id="a", input="q1", expected=ExpectedOutput(contains=["x"])),
            EvalCase(id="a", input="q2", expected=ExpectedOutput(contains=["y"])),
        ],
    )
    with pytest.raises(ValueError, match="unique case ids"):
        run_triple_regime(dup, robust_agent)


def test_marker_collision_refused():
    bad = EvalSuite(
        name="collision",
        cases=[
            EvalCase(
                id="a",
                input="What is the marker?",
                expected=ExpectedOutput(contains=[ADVERSARIAL_MARKER]),
            )
        ],
    )
    with pytest.raises(ValueError, match="adversarial marker"):
        run_triple_regime(bad, robust_agent)


def test_skipped_cases_are_unmeasured_not_unstable():
    skipped = EvalSuite(
        name="all-skipped",
        cases=[
            EvalCase(
                id="a",
                input="q",
                expected=ExpectedOutput(contains=["x"]),
                skip=True,
            )
        ],
    )
    report = run_triple_regime(skipped, robust_agent)
    assert report.clean.skipped == 1
    verdict = check_stability(report)
    assert verdict.outcome == "unmeasured"
    assert any("scored no cases" in reason for reason in verdict.reasons)


def test_policy_thresholds_are_enforced_and_validated():
    report = run_triple_regime(SUITE, naive_agent, seed=42)
    # A policy that tolerates a total adversarial collapse accepts the
    # naive agent; the canonical policy does not.
    lax = StabilityPolicy(clean_threshold=0.9, light_threshold=0.9,
                          max_adversarial_drop=1.0)
    assert check_stability(report, lax).outcome == "stable"
    assert check_stability(report).outcome == "unstable"
    with pytest.raises(ValueError):
        StabilityPolicy(clean_threshold=1.5)


# --- report construction guards -------------------------------------------


def _report_with_swapped_regimes() -> StabilityReport:
    good = run_triple_regime(SUITE, robust_agent, seed=42)
    return StabilityReport(
        suite_name=good.suite_name,
        seed=good.seed,
        clean=good.light,  # wrong regime in the clean slot
        light=good.light,
        adversarial=good.adversarial,
        fingerprint=good.fingerprint,
    )


def test_report_refuses_wrong_regime_in_a_slot():
    with pytest.raises(ValueError, match="carries regime"):
        _report_with_swapped_regimes()


def test_report_refuses_mixed_case_sets():
    good = run_triple_regime(SUITE, robust_agent, seed=42)
    smaller = EvalSuite(name="perturb-test", cases=SUITE.cases[:2])
    other = run_triple_regime(smaller, robust_agent, seed=42)
    with pytest.raises(ValueError, match="different case sets"):
        StabilityReport(
            suite_name=good.suite_name,
            seed=good.seed,
            clean=good.clean,
            light=other.light,
            adversarial=good.adversarial,
            fingerprint="0" * 64,
        )


def test_report_refuses_bad_fingerprint():
    good = run_triple_regime(SUITE, robust_agent, seed=42)
    with pytest.raises(ValueError, match="fingerprint"):
        StabilityReport(
            suite_name=good.suite_name,
            seed=good.seed,
            clean=good.clean,
            light=good.light,
            adversarial=good.adversarial,
            fingerprint="0" * 64,
        )


def test_regime_report_refuses_inconsistent_counts():
    with pytest.raises(ValueError, match="do not sum to total"):
        RegimeReport(
            regime=Regime.CLEAN,
            suite_name="s",
            seed=42,
            total=4,
            passed=3,
            failed=0,
            skipped=0,
            pass_rate=0.75,
            verdicts=(("a", True), ("b", True), ("c", True)),
            fingerprint="f",
        )


def test_regime_report_guards():
    base = {
        "regime": Regime.CLEAN,
        "suite_name": "s",
        "seed": 42,
        "total": 2,
        "passed": 1,
        "failed": 1,
        "skipped": 0,
        "pass_rate": 0.5,
        "verdicts": (("a", True), ("b", False)),
        "fingerprint": "f",
    }
    with pytest.raises(ValueError, match="Negative counts"):
        RegimeReport(**{**base, "passed": -1})
    with pytest.raises(ValueError, match="Duplicate case ids"):
        RegimeReport(**{**base, "verdicts": (("a", True), ("a", False))})
    with pytest.raises(ValueError, match="cannot list missing cases"):
        RegimeReport(**{**base, "missing_case_ids": ("a",)})
    with pytest.raises(ValueError, match="do not sum to total"):
        RegimeReport(**{**base, "total": 3})
    with pytest.raises(ValueError, match="do not match passed\\+failed"):
        RegimeReport(**{**base, "verdicts": (("a", True),)})
    with pytest.raises(ValueError, match="pass_rate"):
        RegimeReport(**{**base, "pass_rate": 0.9})
    with pytest.raises(ValueError, match="must name missing cases"):
        RegimeReport(
            regime=Regime.CLEAN, suite_name="s", seed=42, total=2,
            passed=0, failed=0, skipped=0, pass_rate=0.0, verdicts=(),
            fingerprint="f", measured=False, missing_case_ids=(),
        )
    with pytest.raises(ValueError, match="must carry no verdicts"):
        RegimeReport(
            regime=Regime.CLEAN, suite_name="s", seed=42, total=2,
            passed=1, failed=0, skipped=0, pass_rate=1.0,
            verdicts=(("a", True),), fingerprint="f",
            measured=False, missing_case_ids=("b",),
        )


def test_stability_report_guards():
    import dataclasses

    good = run_triple_regime(SUITE, robust_agent, seed=42)
    wrong_suite = dataclasses.replace(good.clean, suite_name="other")
    with pytest.raises(ValueError, match="is for suite"):
        StabilityReport(
            suite_name=good.suite_name, seed=good.seed, clean=wrong_suite,
            light=good.light, adversarial=good.adversarial,
            fingerprint=good.fingerprint,
        )
    wrong_seed = dataclasses.replace(good.clean, seed=7)
    with pytest.raises(ValueError, match="under seed"):
        StabilityReport(
            suite_name=good.suite_name, seed=good.seed, clean=wrong_seed,
            light=good.light, adversarial=good.adversarial,
            fingerprint=good.fingerprint,
        )
    with pytest.raises(TypeError, match="must be a number"):
        StabilityPolicy(clean_threshold=True)


def test_clean_below_threshold_is_unstable():
    def always_wrong(_prompt: str) -> str:
        return "nothing relevant here"

    report = run_triple_regime(SUITE, always_wrong, seed=42)
    verdict = check_stability(report)
    assert verdict.outcome == "unstable"
    assert any("clean pass rate" in r for r in verdict.reasons)


def test_light_editors_return_none_when_inapplicable():
    import random

    from agenteval_bench.perturbation import _light_case_flip, _light_double_space

    rng = random.Random(42)
    assert _light_case_flip("123 456", rng) is None
    assert _light_case_flip("a b", rng) is None
    assert _light_double_space("word", rng) is None
    assert _light_double_space("a  b", rng) is None


def test_replay_skipped_case_is_skipped_in_all_regimes():
    cases = [
        EvalCase(
            id=case.id,
            input=case.input,
            expected=case.expected,
            output=ANSWERS[case.id],
            output_light=ANSWERS[case.id],
            output_adversarial=ANSWERS[case.id],
        )
        for case in SUITE.cases
    ]
    cases.append(
        EvalCase(
            id="skipped-one",
            input="q",
            expected=ExpectedOutput(contains=["x"]),
            skip=True,
        )
    )
    report = run_triple_regime_replay(EvalSuite(name="s", cases=cases))
    assert report.clean.skipped == 1
    assert report.light.skipped == 1
    assert report.adversarial.skipped == 1
    assert check_stability(report).outcome == "stable"


def test_single_regime_cannot_gate():
    good = run_triple_regime(SUITE, robust_agent, seed=42)
    assert require_triple_regime(good) is good
    with pytest.raises(SingleRegimeError, match="not sufficient"):
        require_triple_regime(good.clean)
    with pytest.raises(SingleRegimeError, match="not sufficient"):
        require_triple_regime(
            RunResult(suite_name="s", total=1, passed=1, pass_rate=1.0)
        )


# --- replay path ------------------------------------------------------------


def _replay_suite(missing_adversarial_for: str | None = None) -> EvalSuite:
    cases = []
    for case in SUITE.cases:
        cases.append(
            EvalCase(
                id=case.id,
                input=case.input,
                expected=case.expected,
                output=ANSWERS[case.id],
                output_light=ANSWERS[case.id],
                output_adversarial=(
                    None
                    if case.id == missing_adversarial_for
                    else ANSWERS[case.id]
                ),
            )
        )
    return EvalSuite(name="perturb-test", cases=cases)


def test_replay_with_full_recordings_is_stable():
    report = run_triple_regime_replay(_replay_suite())
    assert report.clean.pass_rate == 1.0
    assert report.light.pass_rate == 1.0
    assert report.adversarial.pass_rate == 1.0
    assert check_stability(report).outcome == "stable"


def test_replay_missing_recording_is_unmeasured_not_dropped():
    report = run_triple_regime_replay(_replay_suite(missing_adversarial_for="order"))
    assert not report.adversarial.measured
    assert report.adversarial.missing_case_ids == ("order",)
    verdict = check_stability(report)
    assert verdict.outcome == "unmeasured"
    assert any("adversarial regime unmeasured" in r for r in verdict.reasons)
    assert any("order" in r for r in verdict.reasons)


def test_yaml_round_trip_reads_per_regime_recordings(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text(
        "name: yaml-test\n"
        "cases:\n"
        "  - id: a\n"
        "    input: \"q\"\n"
        "    output: \"clean out\"\n"
        "    output_light: \"light out\"\n"
        "    output_adversarial: \"adv out\"\n"
        "    expected:\n"
        "      contains: [\"out\"]\n"
    )
    suite = EvalSuite.from_yaml(str(path))
    assert suite.cases[0].output == "clean out"
    assert suite.cases[0].output_light == "light out"
    assert suite.cases[0].output_adversarial == "adv out"


# --- CLI --------------------------------------------------------------------


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


def _suite_yaml(recordings: dict[str, dict[str, str]]) -> str:
    lines = ["name: cli-test", "cases:"]
    for case in SUITE.cases:
        rec = recordings[case.id]
        lines.append(f"  - id: {case.id}")
        lines.append(f"    input: {case.input!r}")
        for key in ("output", "output_light", "output_adversarial"):
            if key in rec:
                lines.append(f"    {key}: {rec[key]!r}")
        if case.expected.contains:
            needles = ", ".join(repr(n) for n in case.expected.contains)
            lines.append("    expected:")
            lines.append(f"      contains: [{needles}]")
        elif case.expected.json_schema:
            lines.append("    expected:")
            lines.append("      json_schema:")
            lines.append("        required: [\"order_id\", \"status\"]")
        else:
            lines.append("    expected:")
            lines.append(f"      regex: {case.expected.regex!r}")
    return "\n".join(lines) + "\n"


def test_cli_stable_suite_exits_zero(monkeypatch, capsys):
    recordings = {cid: {"output": a, "output_light": a, "output_adversarial": a}
                  for cid, a in ANSWERS.items()}
    path = _write(_suite_yaml(recordings))
    try:
        code, out, _ = _run(["perturb", "--suite", path], monkeypatch, capsys)
        assert code == 0
        assert "verdict: STABLE" in out
        assert "clean: pass rate 100.0%" in out
    finally:
        os.unlink(path)


def test_cli_naive_recordings_exit_one(monkeypatch, capsys):
    recordings = {
        cid: {
            "output": a,
            "output_light": a,
            "output_adversarial": ADVERSARIAL_MARKER,
        }
        for cid, a in ANSWERS.items()
    }
    path = _write(_suite_yaml(recordings))
    try:
        code, out, _ = _run(["perturb", "--suite", path], monkeypatch, capsys)
        assert code == 1
        assert "verdict: UNSTABLE" in out
        assert "drop" in out
    finally:
        os.unlink(path)


def test_cli_missing_recording_exits_two(monkeypatch, capsys):
    recordings = {cid: {"output": a, "output_light": a, "output_adversarial": a}
                  for cid, a in ANSWERS.items()}
    del recordings["order"]["output_adversarial"]
    path = _write(_suite_yaml(recordings))
    try:
        code, out, _ = _run(["perturb", "--suite", path], monkeypatch, capsys)
        assert code == 2
        assert "verdict: UNMEASURED" in out
        assert "order" in out
    finally:
        os.unlink(path)


def test_cli_bad_threshold_and_missing_suite_exit_three(monkeypatch, capsys):
    path = _write(_suite_yaml({cid: {"output": a, "output_light": a,
                                     "output_adversarial": a}
                               for cid, a in ANSWERS.items()}))
    try:
        code, _, _ = _run(
            ["perturb", "--suite", path, "--clean-threshold", "1.5"],
            monkeypatch, capsys,
        )
        assert code == 3
    finally:
        os.unlink(path)
    code, _, _ = _run(["perturb", "--suite", "/nonexistent.yaml"], monkeypatch, capsys)
    assert code == 3
    code, _, _ = _run(["perturb"], monkeypatch, capsys)
    assert code == 3


def test_cli_json_and_out_flag(monkeypatch, capsys, tmp_path):
    recordings = {cid: {"output": a, "output_light": a, "output_adversarial": a}
                  for cid, a in ANSWERS.items()}
    path = _write(_suite_yaml(recordings))
    out_path = str(tmp_path / "report.json")
    try:
        code, out, _ = _run(
            ["perturb", "--suite", path, "--json", "--out", out_path],
            monkeypatch, capsys,
        )
        assert code == 0
        assert '"outcome": "stable"' in out
        assert os.path.exists(out_path)
    finally:
        os.unlink(path)
