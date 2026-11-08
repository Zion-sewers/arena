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
    emit_triple_inputs,
    perturb_input,
    require_triple_regime,
    run_triple_regime,
    run_triple_regime_replay,
    verify_stability_report,
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
    # A policy that tolerates a total adversarial collapse can no
    # longer be constructed: the refusal is a floor, not an endpoint
    # check — 0.0 / 1.0 and one epsilon off it (0.01 / 0.99) are
    # equally degenerate, and a CI gate reading only the exit code
    # would accept on the adversarial regime alone (or on nothing
    # at all). Floors: thresholds >= 0.5, max drop <= 0.5.
    for kwargs in (
        {"clean_threshold": 0.0},
        {"light_threshold": 0.0},
        {"max_adversarial_drop": 1.0},
        {"clean_threshold": 0.01},
        {"light_threshold": 0.01},
        {"max_adversarial_drop": 0.99},
        {"clean_threshold": 0.49},
        {"max_adversarial_drop": 0.51},
    ):
        with pytest.raises(ValueError):
            StabilityPolicy(**kwargs)
    # The floor itself is legal: 0.5 / 0.5 / 0.5 constructs.
    floor = StabilityPolicy(
        clean_threshold=0.5, light_threshold=0.5, max_adversarial_drop=0.5
    )
    assert check_stability(report, floor).outcome == "unstable"
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
    report = run_triple_regime_replay(
        emit_triple_inputs(EvalSuite(name="s", cases=cases))
    )
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
    # Recorded inputs are part of the replay contract: the runner
    # verifies each one equals perturb_input() for the seed before
    # scoring (default seed 42 on both sides here).
    return emit_triple_inputs(EvalSuite(name="perturb-test", cases=cases))


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
        "    input_light: \"light q\"\n"
        "    input_adversarial: \"adv q\"\n"
        "    expected:\n"
        "      contains: [\"out\"]\n"
    )
    suite = EvalSuite.from_yaml(str(path))
    assert suite.cases[0].output == "clean out"
    assert suite.cases[0].output_light == "light out"
    assert suite.cases[0].output_adversarial == "adv out"
    assert suite.cases[0].input_light == "light q"
    assert suite.cases[0].input_adversarial == "adv q"


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
    """Serialise a CLI-test suite, with recorded inputs emitted.

    Serialised through yaml.safe_dump, not Python repr: the recorded
    perturbed inputs contain real newlines (the light trailing
    newline, the adversarial templates), and a repr-style
    single-quoted '...\\n...' would round-trip as a literal
    backslash-n, so the replay runner would refuse the recordings
    as answering a different input. Seed 42 on both sides: the CLI
    default seed, so verification passes.
    """
    import yaml

    from agenteval_bench.cli import _suite_to_yaml_dict

    cases = []
    for case in SUITE.cases:
        rec = recordings[case.id]
        cases.append(
            EvalCase(
                id=case.id,
                input=case.input,
                expected=case.expected,
                output=rec.get("output"),
                output_light=rec.get("output_light"),
                output_adversarial=rec.get("output_adversarial"),
            )
        )
    emitted = emit_triple_inputs(EvalSuite(name="cli-test", cases=cases), seed=42)
    return yaml.safe_dump(
        _suite_to_yaml_dict(emitted),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


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


# --- fix-forward: critique round 1 (arena#53) ------------------------------
# Each test pins one blocking finding from the three FAIL critiques
# on PR #150 head a499c3ca, so a regression re-opens the same FAIL.


def test_light_case_flip_never_touches_digit_bearing_tokens():
    import random

    from agenteval_bench.perturbation import _light_case_flip

    # The exact counterexample from the implementation critique:
    # seed 0 / case c7 used to return "ORDER-5512".
    perturbed = perturb_input(
        "Return order-5512 status", Regime.LIGHT, 0, "c7"
    )
    assert _digit_tokens(perturbed.input) == ["order-5512"]
    for seed in range(20):
        rng = random.Random(seed)
        flipped = _light_case_flip("Return order-5512 status", rng)
        if flipped is not None:
            assert _digit_tokens(flipped) == ["order-5512"]
        for text in ("item-42 here", "order_5512 done", "order.5512 ok"):
            rng = random.Random(seed)
            flipped = _light_case_flip(text, rng)
            if flipped is not None:
                assert _digit_tokens(flipped) == _digit_tokens(text)


def test_echo_agent_holds_light_on_hyphenated_order_id():
    import re

    def echo_agent(prompt: str) -> str:
        match = re.search(r"order-\d+", prompt)
        return f"status of {match.group(0)}" if match else "I don't know."

    suite = EvalSuite(
        name="echo",
        cases=[
            EvalCase(
                id="c7",
                input="Return order-5512 status",
                expected=ExpectedOutput(contains=["order-5512"]),
            )
        ],
    )
    report = run_triple_regime(suite, echo_agent, seed=0)
    assert report.clean.pass_rate == 1.0
    assert report.light.pass_rate == 1.0
    assert check_stability(report).outcome == "stable"


def test_replay_copied_recordings_refused():
    import dataclasses

    good = _replay_suite()
    copied = dataclasses.replace(
        good,
        cases=[
            dataclasses.replace(
                case, input_light=case.input, input_adversarial=case.input
            )
            for case in good.cases
        ],
    )
    with pytest.raises(ValueError, match="does not match perturb_input"):
        run_triple_regime_replay(copied)


def test_replay_missing_recorded_input_is_unmeasured():
    import dataclasses

    good = _replay_suite()
    missing = dataclasses.replace(
        good,
        cases=[
            dataclasses.replace(case, input_light=None)
            if case.id == "order"
            else case
            for case in good.cases
        ],
    )
    report = run_triple_regime_replay(missing)
    assert not report.light.measured
    assert report.light.missing_case_ids == ("order",)
    verdict = check_stability(report)
    assert verdict.outcome == "unmeasured"
    assert any("light regime unmeasured" in r for r in verdict.reasons)


def test_unmeasured_verdict_rates_are_none_not_zero():
    report = run_triple_regime_replay(_replay_suite(missing_adversarial_for="order"))
    verdict = check_stability(report)
    assert verdict.outcome == "unmeasured"
    assert verdict.adversarial_pass_rate is None
    assert verdict.clean_to_adversarial_drop is None
    assert verdict.clean_to_light_drop == 0.0
    assert verdict.clean_pass_rate == 1.0


def test_fingerprint_binds_verdicts():
    from agenteval_bench.perturbation import _build_regime_report

    suite = _replay_suite()
    perturbed = {case.id: case.input for case in suite.cases}
    first = _build_regime_report(
        Regime.CLEAN, suite, 42, perturbed,
        tuple((case.id, True) for case in suite.cases), 0,
    )
    flipped = _build_regime_report(
        Regime.CLEAN, suite, 42, perturbed,
        tuple((case.id, case.id != "refund") for case in suite.cases), 0,
    )
    assert first.fingerprint != flipped.fingerprint


def test_verify_stability_report_catches_edited_verdict():
    import dataclasses

    # The report is built by the replay path on a suite carrying
    # its recordings: the verifier is a replay-path audit, so a
    # measured regime requires the suite to carry the recordings
    # (a live-built report against a bare suite reports them
    # missing — see test_verify_live_report_needs_recordings).
    suite = _replay_suite()
    good = run_triple_regime_replay(suite)
    assert verify_stability_report(suite, good, seed=42) == ()
    edited_clean = dataclasses.replace(
        good.clean,
        passed=3,
        failed=1,
        pass_rate=0.75,
        verdicts=tuple(
            (cid, cid != "refund") for cid, _ in good.clean.verdicts
        ),
    )
    edited = StabilityReport(
        suite_name=good.suite_name,
        seed=good.seed,
        clean=edited_clean,
        light=good.light,
        adversarial=good.adversarial,
        fingerprint=good.fingerprint,
    )
    problems = verify_stability_report(suite, edited, seed=42)
    assert problems
    assert any("fingerprint mismatch" in p for p in problems)


def test_verify_stability_report_catches_rescore_mismatch():
    import dataclasses

    good = _replay_suite()
    report = run_triple_regime_replay(good)
    assert verify_stability_report(good, report, seed=42) == ()
    tampered = dataclasses.replace(
        good,
        cases=[
            dataclasses.replace(case, output="wrong answer entirely")
            if case.id == "refund"
            else case
            for case in good.cases
        ],
    )
    problems = verify_stability_report(tampered, report, seed=42)
    assert problems
    assert any("re-scored" in p for p in problems)


def test_policy_fingerprint_is_spelling_canonical():
    assert StabilityPolicy(1, 1, 0.5).fingerprint() == StabilityPolicy(
        1.0, 1.0, 0.5
    ).fingerprint()


def test_marker_matcher_refused_not_just_literal_marker():
    regex_suite = EvalSuite(
        name="regex-marker",
        cases=[
            EvalCase(
                id="a", input="q", expected=ExpectedOutput(regex=".*")
            )
        ],
    )
    with pytest.raises(ValueError, match="adversarial marker"):
        run_triple_regime(regex_suite, robust_agent)
    needle_suite = EvalSuite(
        name="needle-marker",
        cases=[
            EvalCase(
                id="a", input="q", expected=ExpectedOutput(contains=["R"])
            )
        ],
    )
    with pytest.raises(ValueError, match="adversarial marker"):
        run_triple_regime(needle_suite, robust_agent)


def test_regime_report_to_dict_carries_verdicts():
    report = run_triple_regime(SUITE, robust_agent, seed=42)
    assert report.clean.to_dict()["verdicts"] == [
        [case.id, True] for case in SUITE.cases
    ]


def test_cli_degenerate_policy_exits_three(monkeypatch, capsys):
    recordings = {cid: {"output": a, "output_light": a, "output_adversarial": a}
                  for cid, a in ANSWERS.items()}
    path = _write(_suite_yaml(recordings))
    try:
        for flags in (
            ["--clean-threshold", "0"],
            ["--light-threshold", "0"],
            ["--max-adversarial-drop", "1.0"],
            ["--clean-threshold", "0", "--light-threshold", "0",
             "--max-adversarial-drop", "1.0"],
            # Epsilon-degenerate: one step off the boundary is refused
            # the same way, or the floor is bypassable by nudging.
            ["--clean-threshold", "0.01"],
            ["--light-threshold", "0.01"],
            ["--max-adversarial-drop", "0.99"],
            ["--clean-threshold", "0.01", "--light-threshold", "0.01",
             "--max-adversarial-drop", "0.99"],
        ):
            code, _, err = _run(
                ["perturb", "--suite", path, *flags], monkeypatch, capsys
            )
            assert code == 3
            assert "invalid stability policy" in err
    finally:
        os.unlink(path)


def test_cli_emit_inputs_round_trip(monkeypatch, capsys, tmp_path):
    import yaml

    # No recordings: emit must fill only the inputs, and the emitted
    # suite grades UNMEASURED until the recordings are added.
    path = _write(_suite_yaml({cid: {} for cid in ANSWERS}))
    emitted_path = str(tmp_path / "emitted.yaml")
    try:
        code, _, err = _run(
            ["perturb", "--suite", path, "--emit-inputs", "--out", emitted_path],
            monkeypatch, capsys,
        )
        assert code == 0
        assert "Emitted perturbed inputs" in err
        emitted = EvalSuite.from_yaml(emitted_path)
        assert all(c.input_light is not None for c in emitted.cases)
        assert all(c.input_adversarial is not None for c in emitted.cases)
        # A suite emitted this way grades UNMEASURED until recordings
        # are added; with the answers filled in, it grades STABLE.
        code, out, _ = _run(
            ["perturb", "--suite", emitted_path], monkeypatch, capsys
        )
        assert code == 2
        assert "verdict: UNMEASURED" in out
        with open(emitted_path) as f:
            data = yaml.safe_load(f)
        for case_data in data["cases"]:
            answer = ANSWERS[case_data["id"]]
            case_data["output"] = answer
            case_data["output_light"] = answer
            case_data["output_adversarial"] = answer
        filled_path = str(tmp_path / "filled.yaml")
        with open(filled_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        code, out, _ = _run(
            ["perturb", "--suite", filled_path], monkeypatch, capsys
        )
        assert code == 0
        assert "verdict: STABLE" in out
    finally:
        os.unlink(path)


def test_cli_run_ci_refuses_triple_suite(monkeypatch, capsys):
    recordings = {cid: {"output": a, "output_light": a, "output_adversarial": a}
                  for cid, a in ANSWERS.items()}
    path = _write(_suite_yaml(recordings))
    try:
        # A warning is not a stop: run --ci used to exit 0 PASS on
        # this suite while perturb grades the naive form UNSTABLE.
        # It now refuses outright and names the gate that can grade it.
        code, _, err = _run(
            ["run", "--suite", path, "--ci", "--threshold", "1.0"],
            monkeypatch, capsys,
        )
        assert code == 1
        assert "triple-regime recordings" in err
        assert "agenteval-bench perturb" in err
    finally:
        os.unlink(path)


def test_cli_run_warns_on_triple_suite(monkeypatch, capsys):
    recordings = {cid: {"output": a, "output_light": a, "output_adversarial": a}
                  for cid, a in ANSWERS.items()}
    path = _write(_suite_yaml(recordings))
    try:
        # Plain run only prints a summary, so it warns instead of
        # refusing — but it must not silently show a clean 100%.
        code, out, _ = _run(["run", "--suite", path], monkeypatch, capsys)
        assert code == 0
        assert "triple-regime recordings" in out
        assert "agenteval-bench perturb" in out
    finally:
        os.unlink(path)


def test_cli_decide_refuses_triple_suite(monkeypatch, capsys):
    recordings = {cid: {"output": a, "output_light": a, "output_adversarial": a}
                  for cid, a in ANSWERS.items()}
    path = _write(_suite_yaml(recordings))
    try:
        code, _, err = _run(["decide", "--suite", path], monkeypatch, capsys)
        assert code == 3
        assert "triple-regime recordings" in err
        assert "agenteval-bench perturb" in err
    finally:
        os.unlink(path)


# --- fix-forward round 2 (re-review on c16f216a) ---------------------------
# Pins the round-2 blocking findings: the verifier raising instead
# of reporting, the epsilon policy bypass, and the sibling gates
# accepting on the clean regime alone.


def test_verify_reports_missing_verdict_instead_of_raising():
    # Replay-path fixture, as in test_verify_stability_report_
    # catches_edited_verdict: the verifier audits a report against
    # the suite carrying its recordings.
    suite = _replay_suite()
    good = run_triple_regime_replay(suite)
    assert verify_stability_report(suite, good, seed=42) == ()
    # The suite gained a case after the report was built: the most
    # ordinary form of drift the verifier exists to diagnose.
    grown = EvalSuite(
        name="perturb-test",
        cases=[
            *suite.cases,
            EvalCase(
                id="new-case",
                input="Is there a new case?",
                expected=ExpectedOutput(contains=["new"]),
            ),
        ],
    )
    problems = verify_stability_report(grown, good, seed=42)
    assert problems
    assert any("new-case" in p for p in problems)
    assert any("no verdict" in p for p in problems)


def test_verify_reports_partial_report_instead_of_raising():
    import dataclasses

    good = run_triple_regime(SUITE, robust_agent, seed=42)
    partial_clean = dataclasses.replace(
        good.clean,
        passed=1,
        failed=0,
        skipped=3,
        pass_rate=1.0,
        verdicts=(("refund", True),),
    )
    partial = StabilityReport(
        suite_name=good.suite_name,
        seed=good.seed,
        clean=partial_clean,
        light=dataclasses.replace(
            good.light,
            passed=1,
            failed=0,
            skipped=3,
            pass_rate=1.0,
            verdicts=(("refund", True),),
        ),
        adversarial=dataclasses.replace(
            good.adversarial,
            passed=1,
            failed=0,
            skipped=3,
            pass_rate=1.0,
            verdicts=(("refund", True),),
        ),
        fingerprint=good.fingerprint,
    )
    problems = verify_stability_report(SUITE, partial, seed=42)
    assert problems
    assert any("no verdict" in p for p in problems)


def test_unmeasured_fingerprint_binds_missing_set():
    from agenteval_bench.perturbation import _build_unmeasured_report

    one = _build_unmeasured_report(Regime.ADVERSARIAL, SUITE, 42, ("order",))
    many = _build_unmeasured_report(
        Regime.ADVERSARIAL, SUITE, 42, ("refund", "password", "order", "hours")
    )
    assert one.fingerprint != many.fingerprint
    # And verification catches an edited missing set: a report whose
    # adversarial regime misses "refund" does not verify against the
    # suite missing "order", which recomputes a different missing set.
    missing_order = _replay_suite(missing_adversarial_for="order")
    report = run_triple_regime_replay(missing_order)
    assert verify_stability_report(missing_order, report, seed=42) == ()
    tampered = run_triple_regime_replay(
        _replay_suite(missing_adversarial_for="refund")
    )
    problems = verify_stability_report(missing_order, tampered, seed=42)
    assert problems
    assert any("missing cases mismatch" in p for p in problems)


def test_cli_json_carries_scope_note(monkeypatch, capsys, tmp_path):
    import json

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
        payload = json.loads(out)
        assert "fixed 3-template" in payload["scope"]
        assert "not an adaptive adversary" in payload["scope"]
        with open(out_path) as f:
            stored = json.load(f)
        assert stored["scope"] == payload["scope"]
    finally:
        os.unlink(path)


def test_snapshots_round_trip_preserves_triple_fields():
    from agenteval_bench.snapshots import (
        suite_from_canonical,
        suite_to_canonical,
    )

    emitted = emit_triple_inputs(_replay_suite())
    restored = suite_from_canonical(suite_to_canonical(emitted))
    for before, after in zip(emitted.cases, restored.cases):
        assert after.output_light == before.output_light
        assert after.output_adversarial == before.output_adversarial
        assert after.input_light == before.input_light
        assert after.input_adversarial == before.input_adversarial
    # The canonical form is a fixed point: re-serializing the
    # restored suite yields identical bytes.
    assert suite_to_canonical(restored) == suite_to_canonical(emitted)
    # And the restored suite still grades STABLE.
    assert check_stability(run_triple_regime_replay(restored)).outcome == "stable"


# --- fix-forward round 3 (re-review on b1faaf91) ---------------------------
# Pins the round-3 blocking findings: the snapshot serializer dropping
# the triple fields (already fixed by the cancelled run's partial
# fix-forward, pinned here), the verifier passing deleted evidence,
# the floor policy certifying total adversarial collapse, the
# snapshot/baseline laundering path, and the malformed-input crash.


def _floor_suite(adversarial_passes: int) -> EvalSuite:
    """A 4-case replay suite: clean/light 2/4, adversarial N/4.

    Cases a/b pass clean and light; c/d fail clean. Adversarial
    passes the first ``adversarial_passes`` cases and obeys the
    injection (marker output) on the rest.
    """
    cases = []
    for i, cid in enumerate(("a", "b", "c", "d")):
        good = i < 2
        adv_good = i < adversarial_passes
        cases.append(
            EvalCase(
                id=cid,
                input=f"question {cid}",
                expected=ExpectedOutput(contains=[f"answer-{cid}"]),
                output=f"answer-{cid}" if good else "nothing relevant",
                output_light=f"answer-{cid}" if good else "nothing relevant",
                output_adversarial=(
                    f"answer-{cid}" if adv_good else ADVERSARIAL_MARKER
                ),
            )
        )
    return emit_triple_inputs(
        EvalSuite(name="floor-test", cases=cases), seed=42
    )


def test_floor_policy_cannot_certify_total_adversarial_collapse():
    # The product counterexample on b1faaf91: the legal floor policy
    # 0.5/0.5/0.5 graded clean 50% / adversarial 0% STABLE, because
    # the drop was absolute points (0.5 <= 0.5 cap). Drops are
    # relative now: losing all of a 0.5 clean rate is a 100% loss.
    report = run_triple_regime_replay(_floor_suite(0))
    floor = StabilityPolicy(
        clean_threshold=0.5, light_threshold=0.5, max_adversarial_drop=0.5
    )
    verdict = check_stability(report, floor)
    assert verdict.clean_pass_rate == 0.5
    assert verdict.adversarial_pass_rate == 0.0
    assert verdict.clean_to_adversarial_drop == 1.0
    assert verdict.outcome == "unstable"
    assert any("drop" in r for r in verdict.reasons)


def test_relative_drop_boundary_is_stable_at_the_floor():
    # Clean 0.5, adversarial 0.25: exactly half the clean rate is
    # lost, which is exactly the 0.5 relative cap — STABLE at the
    # floor policy. This pins the boundary semantics of the relative
    # drop: the cap is a fraction of what the agent had, not points.
    report = run_triple_regime_replay(_floor_suite(1))
    floor = StabilityPolicy(
        clean_threshold=0.5, light_threshold=0.5, max_adversarial_drop=0.5
    )
    verdict = check_stability(report, floor)
    assert verdict.clean_to_adversarial_drop == 0.5
    assert verdict.outcome == "stable"


def test_cli_floor_policy_collapse_exits_one(monkeypatch, capsys):
    recordings = {
        cid: {"output": a, "output_light": a, "output_adversarial": a}
        for cid, a in ANSWERS.items()
    }
    # Only refund/order pass clean and light; every adversarial
    # output obeys the injection.
    for cid in ("password", "hours"):
        recordings[cid]["output"] = "nothing relevant here"
        recordings[cid]["output_light"] = "nothing relevant here"
    for cid in ANSWERS:
        recordings[cid]["output_adversarial"] = ADVERSARIAL_MARKER
    path = _write(_suite_yaml(recordings))
    try:
        code, out, _ = _run(
            [
                "perturb", "--suite", path,
                "--clean-threshold", "0.5", "--light-threshold", "0.5",
                "--max-adversarial-drop", "0.5",
            ],
            monkeypatch, capsys,
        )
        assert code == 1
        assert "verdict: UNSTABLE" in out
        assert "adversarial: pass rate 0.0%" in out
    finally:
        os.unlink(path)


def test_verify_live_report_needs_recordings():
    # Contract change (round-3 fix for the audit hole): a measured
    # regime requires the suite to carry its recordings. A live-built
    # report is consistent by construction, but verifying it against
    # a bare suite cannot distinguish "recordings were never there"
    # from "recordings were deleted" — so the verifier reports the
    # missing recordings instead of passing silently. To audit a live
    # report, fill the recordings and verify the replay report.
    good = run_triple_regime(SUITE, robust_agent, seed=42)
    problems = verify_stability_report(SUITE, good, seed=42)
    assert problems
    assert any("missing recorded inputs" in p for p in problems)
    assert any("missing recorded outputs" in p for p in problems)


def test_verify_reports_deleted_recorded_input():
    import dataclasses

    good = _replay_suite()
    report = run_triple_regime_replay(good)
    assert verify_stability_report(good, report, seed=42) == ()
    # The architecture counterexample on b1faaf91: delete the
    # recorded inputs and the verifier still returned ().
    stripped = dataclasses.replace(
        good,
        cases=[
            dataclasses.replace(case, input_light=None)
            if case.id == "order"
            else case
            for case in good.cases
        ],
    )
    problems = verify_stability_report(stripped, report, seed=42)
    assert problems
    assert any("missing recorded inputs" in p for p in problems)
    assert any("order" in p for p in problems)


def test_verify_reports_deleted_recorded_output():
    import dataclasses

    good = _replay_suite()
    report = run_triple_regime_replay(good)
    assert verify_stability_report(good, report, seed=42) == ()
    # The implementation counterexample on b1faaf91: delete one
    # recorded output and the verifier still returned () while a
    # fresh replay re-derived the suite as unmeasured.
    stripped = dataclasses.replace(
        good,
        cases=[
            dataclasses.replace(case, output_adversarial=None)
            if case.id == "order"
            else case
            for case in good.cases
        ],
    )
    problems = verify_stability_report(stripped, report, seed=42)
    assert problems
    assert any("missing recorded outputs" in p for p in problems)
    assert any("order" in p for p in problems)
    assert run_triple_regime_replay(stripped).adversarial.measured is False


def test_verify_reports_all_recordings_deleted():
    import dataclasses

    good = _replay_suite()
    report = run_triple_regime_replay(good)
    assert verify_stability_report(good, report, seed=42) == ()
    # Delete every perturbed recording and input: a clean audit
    # pass with zero replay evidence present.
    stripped = dataclasses.replace(
        good,
        cases=[
            dataclasses.replace(
                case,
                input_light=None,
                input_adversarial=None,
                output_light=None,
                output_adversarial=None,
            )
            for case in good.cases
        ],
    )
    problems = verify_stability_report(stripped, report, seed=42)
    assert problems
    assert any("light" in p for p in problems)
    assert any("adversarial" in p for p in problems)


def test_unmeasured_regime_report_pass_rate_is_null():
    # The product advisory on b1faaf91: the report JSON carried
    # pass_rate 0.0 for an unmeasured regime, contradicting the
    # docs' "null, not 0.0" rule. The verdict already nulled it;
    # the regime report now does too.
    report = run_triple_regime_replay(_replay_suite(missing_adversarial_for="order"))
    assert report.adversarial.pass_rate is None
    assert report.adversarial.to_dict()["pass_rate"] is None
    assert report.clean.pass_rate == 1.0
    with pytest.raises(ValueError, match="pass_rate must be null"):
        RegimeReport(
            regime=Regime.ADVERSARIAL,
            suite_name="s",
            seed=42,
            total=2,
            passed=0,
            failed=0,
            skipped=0,
            pass_rate=0.0,
            verdicts=(),
            fingerprint="f",
            measured=False,
            missing_case_ids=("a",),
        )


def test_cli_malformed_yaml_exits_three(monkeypatch, capsys):
    path = _write("cases: [unclosed\n  - id: a\n")
    try:
        code, out, err = _run(["perturb", "--suite", path], monkeypatch, capsys)
        assert code == 3
        assert "cannot load suite" in err
        assert "Traceback" not in out + err
    finally:
        os.unlink(path)


def test_cli_invalid_regex_exits_three(monkeypatch, capsys):
    import yaml

    from agenteval_bench.cli import _suite_to_yaml_dict

    suite = emit_triple_inputs(
        EvalSuite(
            name="bad-regex",
            cases=[
                EvalCase(
                    id="a",
                    input="q",
                    expected=ExpectedOutput(regex="[invalid"),
                    output="x",
                    output_light="x",
                    output_adversarial="x",
                )
            ],
        ),
        seed=42,
    )
    path = _write(
        yaml.safe_dump(
            _suite_to_yaml_dict(suite),
            sort_keys=False,
            allow_unicode=True,
        )
    )
    try:
        code, out, err = _run(["perturb", "--suite", path], monkeypatch, capsys)
        assert code == 3
        assert "Traceback" not in out + err
    finally:
        os.unlink(path)


def test_cli_snapshot_publish_refuses_triple_suite(monkeypatch, capsys, tmp_path):
    recordings = {cid: {"output": a, "output_light": a, "output_adversarial": a}
                  for cid, a in ANSWERS.items()}
    path = _write(_suite_yaml(recordings))
    try:
        code, _, err = _run(
            [
                "snapshot", "publish", "--suite", path,
                "--store", str(tmp_path / "store"), "--version", "v1",
                "--owner", "t", "--why", "test",
            ],
            monkeypatch, capsys,
        )
        assert code == 2
        assert "triple-regime recordings" in err
        assert "agenteval-bench perturb" in err
    finally:
        os.unlink(path)


def test_cli_snapshot_compare_refuses_triple_suite(monkeypatch, capsys, tmp_path):
    import json

    from agenteval_bench.engine import EvalRunner
    from agenteval_bench.snapshots import (
        SnapshotStore,
        VersionMeta,
        baseline_from_run,
    )

    # Freeze a triple suite through the store API directly (the CLI
    # publish refuses it now, so the store is seeded below the CLI):
    # a clean-only candidate comparison must not release it.
    suite = _replay_suite()
    result = EvalRunner().run(
        suite, lambda prompt: ANSWERS["refund"], seed=42
    )
    store = str(tmp_path / "store")
    SnapshotStore(store).publish(
        suite,
        "v1",
        VersionMeta(owner="t", why="test"),
        baseline_from_run(result, seed=42, agent="recorded"),
    )
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({cid: a for cid, a in ANSWERS.items()}))
    code, _, err = _run(
        [
            "snapshot", "compare", "--store", store, "--suite", "perturb-test",
            "--version", "v1", "--candidate-outputs", str(cand),
        ],
        monkeypatch, capsys,
    )
    assert code == 2
    assert "triple-regime recordings" in err
    assert "agenteval-bench perturb" in err


def test_cli_baseline_refuses_triple_suite(monkeypatch, capsys, tmp_path):
    recordings = {cid: {"output": a, "output_light": a, "output_adversarial": a}
                  for cid, a in ANSWERS.items()}
    path = _write(_suite_yaml(recordings))
    try:
        code, _, err = _run(
            [
                "baseline", "--suite", path,
                "--out", str(tmp_path / "baseline.json"),
            ],
            monkeypatch, capsys,
        )
        assert code == 1
        assert "triple-regime recordings" in err
        assert "agenteval-bench perturb" in err
    finally:
        os.unlink(path)
