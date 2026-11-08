"""Tests for the persona symmetry gate (arena#52).

Every acceptance criterion has a named test, and the property tests at
the bottom pin the symmetries for every generated input: permuting
irrelevant inputs gives identical outputs, and the gate FAILs on
deliberately asymmetric stand-in builders (the negative controls prove
the gate bites — a gate that cannot fail is theater).
"""

from __future__ import annotations

import ast
import random
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agenteval_bench.symmetry import (
    ALL_SYMMETRIES,
    PERSONA_CONTRACTS,
    SAMPLE_ALTERNATE_MODEL,
    SAMPLE_ALTERNATE_TOPIC,
    SAMPLE_INPUT,
    DeclaredSymmetry,
    PersonaCall,
    PersonaContract,
    PersonaInput,
    PersonaName,
    SymmetryKind,
    SymmetryReport,
    SymmetryVerdict,
    TransformationKind,
    apply_transformation,
    build_persona_call,
    check_all_personas,
    check_symmetry,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
RUN_COFFEE: Path = REPO_ROOT / ".github" / "agent-coffee" / "run_coffee.py"

BUILDER_CONTRACT = PERSONA_CONTRACTS[PersonaName.BUILDER]


def _check_kwargs() -> dict[str, str]:
    return {
        "alternate_model": SAMPLE_ALTERNATE_MODEL,
        "alternate_topic": SAMPLE_ALTERNATE_TOPIC,
    }


# --- Registry: every persona declares its symmetry contract. ---


def test_registry_covers_exactly_the_four_coffee_personas() -> None:
    assert set(PERSONA_CONTRACTS) == {
        PersonaName.BUILDER,
        PersonaName.CRITIC,
        PersonaName.GUARDIAN,
        PersonaName.DREAMER,
    }


def test_each_contract_declares_all_four_symmetries() -> None:
    for name, contract in PERSONA_CONTRACTS.items():
        assert contract.persona is name
        assert contract.symmetries == ALL_SYMMETRIES
        assert len(contract.symmetries) == 4
        assert {d.transformation for d in contract.symmetries} == {
            TransformationKind.MODEL_SUBSTITUTION,
            TransformationKind.WHITESPACE,
            TransformationKind.CONTEXT_ORDER,
            TransformationKind.TOPIC_SUBSTITUTION,
        }


def test_contracts_differ_only_in_prompt() -> None:
    # The symmetries are properties of the shared call construction, so
    # the contracts share them; the persona identity is the prompt.
    prompts = {c.system_prompt for c in PERSONA_CONTRACTS.values()}
    assert len(prompts) == 4
    assert {c.max_tokens for c in PERSONA_CONTRACTS.values()} == {400}


# --- Sync: the registry is the single source of truth for the prompts. ---


def _coffee_prompts() -> dict[str, str]:
    """PERSONA_PROMPTS from run_coffee.py, parsed (never executed)."""
    tree = ast.parse(RUN_COFFEE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "PERSONA_PROMPTS":
                assert isinstance(node.value, ast.Dict)
                prompts: dict[str, str] = {}
                for key, value in zip(node.value.keys, node.value.values):
                    assert isinstance(key, ast.Constant) and isinstance(key.value, str)
                    assert isinstance(value, ast.Constant) and isinstance(value.value, str), (
                        f"persona prompt for {key.value!r} is not a plain string"
                    )
                    prompts[key.value] = value.value
                return prompts
    raise AssertionError("PERSONA_PROMPTS not found in run_coffee.py")


def _coffee_max_tokens() -> int:
    """The max_tokens literal in run_coffee.py's chat() call."""
    tree = ast.parse(RUN_COFFEE.read_text(encoding="utf-8"))
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "max_tokens"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, int)
            ):
                found.add(value.value)
    assert found, "no max_tokens literal found in run_coffee.py"
    assert len(found) == 1, f"run_coffee.py uses several max_tokens: {found}"
    return found.pop()


def test_registry_prompts_match_run_coffee() -> None:
    prompts = _coffee_prompts()
    assert set(prompts) == {name.value for name in PersonaName}
    for name, contract in PERSONA_CONTRACTS.items():
        assert contract.system_prompt == prompts[name.value], (
            f"registry prompt for {name.value} drifted from run_coffee.py: "
            "update the registry (single source of truth), never the test"
        )


def test_registry_max_tokens_match_run_coffee() -> None:
    assert _coffee_max_tokens() == 400
    for contract in PERSONA_CONTRACTS.values():
        assert contract.max_tokens == _coffee_max_tokens()


# --- The gate passes on the real contracts. ---


def test_gate_passes_on_all_personas_for_sample_input() -> None:
    reports = check_all_personas(SAMPLE_INPUT)
    assert len(reports) == 4
    for report in reports:
        assert report.verdict is SymmetryVerdict.PASS, report.summary()
        assert not report.violations
        assert len(report.checks) == 4
        assert all(check.passed for check in report.checks)


def test_build_is_deterministic() -> None:
    # The identity property: the same input twice gives the same call.
    # A non-deterministic builder fails this before any symmetry is tried.
    first = build_persona_call(BUILDER_CONTRACT, SAMPLE_INPUT)
    second = build_persona_call(BUILDER_CONTRACT, SAMPLE_INPUT)
    assert first == second


def test_canonicalization_is_idempotent() -> None:
    from agenteval_bench.symmetry import (
        _canonical_context_items,
        _canonical_topic,
    )

    topic = "  spaced topic  \n\n\nsecond line   \n\n"
    assert _canonical_topic(_canonical_topic(topic)) == _canonical_topic(topic)
    assert _canonical_topic(topic) == "spaced topic\n\nsecond line"
    items = ("  b  ", "", "a", "  b", "A")
    assert _canonical_context_items(_canonical_context_items(items)) == _canonical_context_items(
        items
    )
    assert _canonical_context_items(items) == ("A", "a", "b", "b")


def test_build_canonicalizes_whitespace_and_order() -> None:
    messy = PersonaInput(
        topic="\n\n  Should the eval suite add a symmetry gate for personas?  \n\n",
        context_items=(
            "  CI must fail on violation   ",
            "",
            "personas deliberate in rounds",
            "Geometric Deep Learning links symmetry to stability",
        ),
        model="gpt-4o-mini",
    )
    assert build_persona_call(BUILDER_CONTRACT, messy) == build_persona_call(
        BUILDER_CONTRACT, SAMPLE_INPUT
    )


def test_check_all_personas_summary_lists_every_check() -> None:
    (report,) = [r for r in check_all_personas(SAMPLE_INPUT) if r.persona is PersonaName.BUILDER]
    summary = report.summary()
    assert "persona builder: PASS" in summary
    for transformation in (
        "model_substitution",
        "whitespace",
        "context_order",
        "topic_substitution",
    ):
        assert transformation in summary


# --- Negative controls: the gate must FAIL on asymmetric stand-ins. ---


def _model_naming_builder(contract: PersonaContract, base: PersonaInput) -> PersonaCall:
    """Stand-in that bakes the model name into the system prompt."""
    call = build_persona_call(contract, base)
    return replace(call, system_prompt=f"[{base.model}] {call.system_prompt}")


def _unsorted_builder(contract: PersonaContract, base: PersonaInput) -> PersonaCall:
    """Stand-in that joins context items in given order (no sorting)."""
    from agenteval_bench.symmetry import _canonical_topic

    message = f"Topic: {_canonical_topic(base.topic)}\nContext: " + "\n".join(
        item.strip() for item in base.context_items if item.strip()
    )
    return PersonaCall(
        contract.persona,
        contract.system_prompt,
        message,
        contract.max_tokens,
        base.model,
    )


def _raw_builder(contract: PersonaContract, base: PersonaInput) -> PersonaCall:
    """Stand-in that skips canonicalization entirely."""
    joined = "\n".join(base.context_items)
    message = f"Topic: {base.topic}\nContext: {joined}"
    return PersonaCall(
        contract.persona,
        contract.system_prompt,
        message,
        contract.max_tokens,
        base.model,
    )


def _topic_dropping_builder(contract: PersonaContract, base: PersonaInput) -> PersonaCall:
    """Stand-in that drops the topic (context sorted, so only the
    equivariance check can fail)."""
    from agenteval_bench.symmetry import _canonical_context_items

    message = "Context: " + "\n".join(_canonical_context_items(base.context_items))
    return PersonaCall(
        contract.persona,
        contract.system_prompt,
        message,
        contract.max_tokens,
        base.model,
    )


def _topic_echo_builder(contract: PersonaContract, base: PersonaInput) -> PersonaCall:
    """Stand-in that echoes the topic into the system prompt too."""
    from agenteval_bench.symmetry import _canonical_topic

    call = build_persona_call(contract, base)
    return replace(
        call,
        system_prompt=call.system_prompt + " Topic: " + _canonical_topic(base.topic),
    )


def _violated(report: SymmetryReport, transformation: TransformationKind) -> bool:
    return any(v.transformation is transformation for v in report.violations)


def test_gate_fails_model_naming_builder() -> None:
    report = check_symmetry(
        BUILDER_CONTRACT, SAMPLE_INPUT, build_fn=_model_naming_builder, **_check_kwargs()
    )
    assert report.verdict is SymmetryVerdict.FAIL
    assert {v.transformation for v in report.violations} == {TransformationKind.MODEL_SUBSTITUTION}


def test_gate_fails_unsorted_builder() -> None:
    report = check_symmetry(
        BUILDER_CONTRACT, SAMPLE_INPUT, build_fn=_unsorted_builder, **_check_kwargs()
    )
    assert report.verdict is SymmetryVerdict.FAIL
    # CONTEXT_ORDER is the violation this stand-in exists to prove. It
    # also fails TOPIC_SUBSTITUTION, genuinely: the equivariance
    # expectation uses the canonical (sorted) context, so a builder whose
    # context part is unsorted cannot move exactly under topic
    # substitution either. MODEL_SUBSTITUTION and WHITESPACE still pass.
    assert _violated(report, TransformationKind.CONTEXT_ORDER)
    assert not _violated(report, TransformationKind.MODEL_SUBSTITUTION)
    assert not _violated(report, TransformationKind.WHITESPACE)


def test_gate_fails_raw_builder_on_whitespace() -> None:
    report = check_symmetry(
        BUILDER_CONTRACT, SAMPLE_INPUT, build_fn=_raw_builder, **_check_kwargs()
    )
    assert report.verdict is SymmetryVerdict.FAIL
    assert _violated(report, TransformationKind.WHITESPACE)


def test_gate_fails_topic_dropping_builder() -> None:
    report = check_symmetry(
        BUILDER_CONTRACT,
        SAMPLE_INPUT,
        build_fn=_topic_dropping_builder,
        **_check_kwargs(),
    )
    assert report.verdict is SymmetryVerdict.FAIL
    assert {v.transformation for v in report.violations} == {TransformationKind.TOPIC_SUBSTITUTION}


def test_gate_fails_topic_echo_builder() -> None:
    report = check_symmetry(
        BUILDER_CONTRACT, SAMPLE_INPUT, build_fn=_topic_echo_builder, **_check_kwargs()
    )
    assert report.verdict is SymmetryVerdict.FAIL
    assert {v.transformation for v in report.violations} == {TransformationKind.TOPIC_SUBSTITUTION}


# --- Fail-closed validation. ---


def test_empty_symmetries_refused() -> None:
    with pytest.raises(ValueError, match="declares no symmetries"):
        PersonaContract(
            persona=PersonaName.BUILDER,
            system_prompt="x",
            max_tokens=400,
            symmetries=(),
        )


def test_duplicate_transformation_refused() -> None:
    with pytest.raises(ValueError, match="declared twice"):
        PersonaContract(
            persona=PersonaName.BUILDER,
            system_prompt="x",
            max_tokens=400,
            symmetries=(
                DeclaredSymmetry(TransformationKind.WHITESPACE, SymmetryKind.INVARIANT),
                DeclaredSymmetry(TransformationKind.WHITESPACE, SymmetryKind.INVARIANT),
            ),
        )


def test_uncheckable_pair_refused() -> None:
    # WHITESPACE is invariant in this gate; declaring it equivariant is
    # a promise the gate cannot check, so it is refused at construction.
    with pytest.raises(ValueError, match="unsupported symmetry pair"):
        DeclaredSymmetry(TransformationKind.WHITESPACE, SymmetryKind.EQUIVARIANT)
    with pytest.raises(ValueError, match="unsupported symmetry pair"):
        DeclaredSymmetry(TransformationKind.TOPIC_SUBSTITUTION, SymmetryKind.INVARIANT)


def test_empty_topic_refused() -> None:
    with pytest.raises(ValueError, match="topic must not be empty"):
        PersonaInput(topic="   \n\n  ", context_items=(), model="gpt-4o-mini")


def test_empty_model_refused() -> None:
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        PersonaInput(topic="t", context_items=(), model="  ")


def test_multiline_context_item_refused() -> None:
    with pytest.raises(ValueError, match="single line"):
        PersonaInput(topic="t", context_items=("one\ntwo",), model="m")


def test_non_tuple_context_items_refused() -> None:
    with pytest.raises(TypeError, match="tuple of str"):
        PersonaInput(topic="t", context_items=["a", "b"], model="m")  # type: ignore[arg-type]


def test_non_str_context_item_refused() -> None:
    with pytest.raises(TypeError, match="must be a str"):
        PersonaInput(topic="t", context_items=(1,), model="m")  # type: ignore[arg-type]


def test_bool_max_tokens_refused() -> None:
    with pytest.raises(TypeError, match="must be an int"):
        PersonaContract(
            persona=PersonaName.BUILDER,
            system_prompt="x",
            max_tokens=True,
            symmetries=ALL_SYMMETRIES,
        )


def test_vacuous_alternates_refused() -> None:
    with pytest.raises(ValueError, match="must differ from the base model"):
        apply_transformation(
            TransformationKind.MODEL_SUBSTITUTION,
            SAMPLE_INPUT,
            alternate_model="gpt-4o-mini",
        )
    with pytest.raises(ValueError, match="must differ from the base topic"):
        apply_transformation(
            TransformationKind.TOPIC_SUBSTITUTION,
            SAMPLE_INPUT,
            alternate_topic="  Should the eval suite add a symmetry gate for personas?  ",
        )
    with pytest.raises(ValueError, match="needs alternate_model"):
        apply_transformation(TransformationKind.MODEL_SUBSTITUTION, SAMPLE_INPUT)
    with pytest.raises(ValueError, match="needs alternate_topic"):
        apply_transformation(TransformationKind.TOPIC_SUBSTITUTION, SAMPLE_INPUT)


# --- Property tests: the symmetries hold for every generated input. ---

WORDS = st.sampled_from(
    ["symmetry", "gate", "persona", "contract", "coffee", "rounds", "CI", "GDL"]
)
ITEMS = st.lists(st.text(alphabet="abcXYZ 012", min_size=1, max_size=12), min_size=0, max_size=5)
TOPICS = st.lists(WORDS, min_size=1, max_size=6).map(" ".join)
MODELS = st.sampled_from(["gpt-4o-mini", "claude-haiku-4-5", "llama-4-scout", "gpt-oss-20b"])


@settings(max_examples=30, deadline=None)
@given(
    topic=TOPICS,
    items=ITEMS,
    model=MODELS,
)
def test_context_order_property(topic: str, items: list[str], model: str) -> None:
    base = PersonaInput(topic, tuple(items), model)
    shuffled = list(items)
    random.Random(1234).shuffle(shuffled)
    assert build_persona_call(BUILDER_CONTRACT, base) == build_persona_call(
        BUILDER_CONTRACT, PersonaInput(topic, tuple(shuffled), model)
    )


@settings(max_examples=30, deadline=None)
@given(topic=TOPICS, items=ITEMS, model_a=MODELS, model_b=MODELS)
def test_model_substitution_property(
    topic: str, items: list[str], model_a: str, model_b: str
) -> None:
    from agenteval_bench.symmetry import _payload

    call_a = build_persona_call(BUILDER_CONTRACT, PersonaInput(topic, tuple(items), model_a))
    call_b = build_persona_call(BUILDER_CONTRACT, PersonaInput(topic, tuple(items), model_b))
    assert _payload(call_a) == _payload(call_b)
    assert call_a.model == model_a and call_b.model == model_b


@settings(max_examples=30, deadline=None)
@given(topic_a=TOPICS, topic_b=TOPICS, items=ITEMS, model=MODELS)
def test_topic_substitution_property(
    topic_a: str, topic_b: str, items: list[str], model: str
) -> None:
    if topic_a == topic_b:
        return
    report = check_symmetry(
        BUILDER_CONTRACT,
        PersonaInput(topic_a, tuple(items), model),
        alternate_model=SAMPLE_ALTERNATE_MODEL
        if model != SAMPLE_ALTERNATE_MODEL
        else "gpt-4o-mini",
        alternate_topic=topic_b,
    )
    assert report.verdict is SymmetryVerdict.PASS, report.summary()


@settings(max_examples=30, deadline=None)
@given(
    topic=st.text(alphabet="abc \n", min_size=1, max_size=40),
    items=ITEMS,
    model=MODELS,
)
def test_whitespace_property(topic: str, items: list[str], model: str) -> None:
    from agenteval_bench.symmetry import _canonical_topic

    if not _canonical_topic(topic):
        return
    base = PersonaInput(topic, tuple(items), model)
    perturbed = apply_transformation(TransformationKind.WHITESPACE, base)
    assert build_persona_call(BUILDER_CONTRACT, base) == build_persona_call(
        BUILDER_CONTRACT, perturbed
    )
