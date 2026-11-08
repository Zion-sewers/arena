"""Symmetry tests for personas (arena#52).

Source: Bronstein et al., "Geometric Deep Learning" (catalog G.25).
Symmetry -> equivariance -> stability. Each persona contract states which
transformations it is equivariant/invariant to. Test: permuting irrelevant
inputs must give identical outputs.

What this module is. A typed symmetry gate over the deterministic part of
a persona: the call construction. A persona here is one of the four
coffee-break personas (builder, critic, guardian, dreamer — see
``.github/agent-coffee/COFFEE.md``), and its call is the pure triple
(system prompt, user message, max_tokens) plus the model the jev router
picked. The gate checks, for each persona's DECLARED symmetries, that the
built call obeys them exactly:

* INVARIANT: the call is byte-identical after the transformation.
* EQUIVARIANT: the call transforms in exactly the declared way — the
  transformed input's topic replaces exactly its occurrence in the user
  message, and nothing else changes.

The four declared transformations (a closed enum — a transformation the
gate cannot check cannot be declared):

* MODEL_SUBSTITUTION (invariant): the payload is identical whichever
  model the router picks. The repo's own stated property: "No persona
  names a model; the jev router picks the cheapest capable one."
* WHITESPACE (invariant): trailing spaces, leading/trailing blank lines
  and doubled blank lines in the topic or context items do not change
  the call. The build path canonicalizes them away.
* CONTEXT_ORDER (invariant): any permutation of the context items does
  not change the call. Context items are an explicit sequence of
  INDEPENDENT items — the caller asserts independence by using the type;
  the build path sorts them into a canonical order.
* TOPIC_SUBSTITUTION (equivariant): replacing the topic replaces exactly
  its occurrence in the user message; system prompt, max_tokens and model
  are unchanged.

What this module is NOT. It cannot prove the live model's text output is
symmetric: model outputs are non-deterministic, and the repo's first
principle forbids the core from requiring a model to produce stable
results. The gate proves the symmetry of everything the repo controls —
the call construction and the contract machinery — and says so. A test
that called a live model and asserted byte-equality would be flaky
theater, worse than the boundary. The negative controls in the test suite
prove the gate bites: stand-in builders that embed the model name, join
context unsorted, skip canonicalization, or drop the topic each FAIL with
the violated transformation named.

One deliberate omission: pairwise-composition checks. For exact
invariance, invariance under t1 and t2 implies invariance under t1∘t2,
so a composition check cannot fail when the singles pass — including it
would be a check that cannot fail, i.e. theater. The property tests
quantify over generated inputs instead, which is the real "permuting
irrelevant inputs" claim.

Sync obligation: the persona prompts in PERSONA_CONTRACTS are the single
source of truth and must stay byte-identical to PERSONA_PROMPTS in
``.github/agent-coffee/run_coffee.py``. That script runs standalone in
GitHub Actions without the package installed, so it cannot import this
module; tests/test_symmetry.py parses its prompts and fails CI on drift.
The same test pins max_tokens against the literal in run_coffee.py's
chat() call.

Everything is stdlib-only and deterministic; there is no model call here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

# The four coffee-break personas. The order matches COFFEE.md's table.


class PersonaName(str, Enum):
    BUILDER = "builder"
    CRITIC = "critic"
    GUARDIAN = "guardian"
    DREAMER = "dreamer"


class SymmetryKind(str, Enum):
    INVARIANT = "invariant"
    EQUIVARIANT = "equivariant"


class TransformationKind(str, Enum):
    MODEL_SUBSTITUTION = "model_substitution"
    WHITESPACE = "whitespace"
    CONTEXT_ORDER = "context_order"
    TOPIC_SUBSTITUTION = "topic_substitution"


class SymmetryVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


# Which (transformation, kind) pairs the gate can actually check. A
# contract may only declare pairs from this table: a declared symmetry the
# gate cannot check would be a promise with no enforcement behind it.
_CHECKABLE: MappingProxyType[TransformationKind, SymmetryKind] = MappingProxyType(
    {
        TransformationKind.MODEL_SUBSTITUTION: SymmetryKind.INVARIANT,
        TransformationKind.WHITESPACE: SymmetryKind.INVARIANT,
        TransformationKind.CONTEXT_ORDER: SymmetryKind.INVARIANT,
        TransformationKind.TOPIC_SUBSTITUTION: SymmetryKind.EQUIVARIANT,
    }
)


def _check_str(value: str, field_name: str) -> str:
    """One string field: a real str, never a bool or another type."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str, got {value!r}")
    return value


def _check_int(value: int, field_name: str) -> int:
    """One integer field: a real int, never a bool, never a float."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {value!r}")
    return value


@dataclass(frozen=True)
class DeclaredSymmetry:
    """One declared symmetry: a transformation and the kind it is held to."""

    transformation: TransformationKind
    kind: SymmetryKind

    def __post_init__(self) -> None:
        if not isinstance(self.transformation, TransformationKind):
            raise TypeError(
                f"transformation must be a TransformationKind, got {self.transformation!r}"
            )
        if not isinstance(self.kind, SymmetryKind):
            raise TypeError(f"kind must be a SymmetryKind, got {self.kind!r}")
        checkable = _CHECKABLE.get(self.transformation)
        if checkable is not self.kind:
            raise ValueError(
                f"unsupported symmetry pair: {self.transformation.value} "
                f"is {checkable.value if checkable is not None else 'unknown'} "
                f"in this gate, not {self.kind.value}"
            )


# Every persona declares the same four symmetries: they are properties of
# the shared call construction, and the personas differ in their prompts.
# Built here, after DeclaredSymmetry, so the pairs are validated.
ALL_SYMMETRIES: tuple[DeclaredSymmetry, ...] = tuple(
    DeclaredSymmetry(transformation, kind) for transformation, kind in _CHECKABLE.items()
)


@dataclass(frozen=True)
class PersonaInput:
    """Input to one persona call: topic, independent context items, model.

    ``context_items`` are an explicit sequence of INDEPENDENT items (one
    fact each, single line). The caller asserts independence by using this
    type — that is what makes CONTEXT_ORDER a legitimate symmetry claim.
    Free-text context with narrative order does not belong here; the
    coffee ritual's TOPIC_BODY path stays free text and claims no order
    symmetry. An item containing a newline is two items, and is refused.
    """

    topic: str
    context_items: tuple[str, ...]
    model: str

    def __post_init__(self) -> None:
        _check_str(self.topic, "topic")
        if not isinstance(self.context_items, tuple):
            raise TypeError(f"context_items must be a tuple of str, got {self.context_items!r}")
        for item in self.context_items:
            _check_str(item, "context item")
            if "\n" in item or "\r" in item:
                raise ValueError(f"context item must be a single line, got {item!r}")
        _check_str(self.model, "model")
        if not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not _canonical_topic(self.topic):
            raise ValueError("topic must not be empty or whitespace-only")


@dataclass(frozen=True)
class PersonaCall:
    """One built persona call: the deterministic payload plus the model."""

    persona: PersonaName
    system_prompt: str
    user_message: str
    max_tokens: int
    model: str

    def __post_init__(self) -> None:
        if not isinstance(self.persona, PersonaName):
            raise TypeError(f"persona must be a PersonaName, got {self.persona!r}")
        _check_str(self.system_prompt, "system_prompt")
        _check_str(self.user_message, "user_message")
        _check_int(self.max_tokens, "max_tokens")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        _check_str(self.model, "model")
        if not self.model.strip():
            raise ValueError("model must be a non-empty string")


@dataclass(frozen=True)
class PersonaContract:
    """One persona's symmetry contract: its prompt, token cap, symmetries.

    A contract with no declared symmetries is refused: a persona that
    claims nothing is exactly "it seemed robust", and this gate does not
    accept it. A transformation may appear at most once, and only in a
    pair the gate can check (see _CHECKABLE).
    """

    persona: PersonaName
    system_prompt: str
    max_tokens: int
    symmetries: tuple[DeclaredSymmetry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.persona, PersonaName):
            raise TypeError(f"persona must be a PersonaName, got {self.persona!r}")
        _check_str(self.system_prompt, "system_prompt")
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        _check_int(self.max_tokens, "max_tokens")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if not isinstance(self.symmetries, tuple):
            raise TypeError(
                f"symmetries must be a tuple of DeclaredSymmetry, got {self.symmetries!r}"
            )
        if not self.symmetries:
            raise ValueError(
                f"persona {self.persona.value} declares no symmetries: "
                "a contract that claims nothing is not a contract"
            )
        seen: set[TransformationKind] = set()
        for declared in self.symmetries:
            if not isinstance(declared, DeclaredSymmetry):
                raise TypeError(f"symmetries must hold DeclaredSymmetry, got {declared!r}")
            if declared.transformation in seen:
                raise ValueError(
                    "transformation "
                    f"{declared.transformation.value} declared twice for "
                    f"persona {self.persona.value}"
                )
            seen.add(declared.transformation)


def _canonical_topic(topic: str) -> str:
    """Canonical topic: strip ends, strip each line's trailing space,
    collapse runs of blank lines to one. Idempotent by construction."""
    lines = [line.rstrip() for line in topic.strip().splitlines()]
    out: list[str] = []
    prev_blank = True  # skip leading blank lines
    for line in lines:
        if line == "" and prev_blank:
            continue
        out.append(line)
        prev_blank = line == ""
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def _canonical_context_items(items: tuple[str, ...]) -> tuple[str, ...]:
    """Canonical context: strip each item, drop empties, sort.

    Sorting is what makes CONTEXT_ORDER true: the items are a set of
    independent facts, so their canonical form is order-free.
    """
    return tuple(sorted(item.strip() for item in items if item.strip()))


def _format_user_message(topic: str, context_items: tuple[str, ...]) -> str:
    """The user-message format, shared by the builder and the
    equivariance expectation: "Topic: {topic}\\nContext: {items}"."""
    joined = "\n".join(_canonical_context_items(context_items))
    return f"Topic: {_canonical_topic(topic)}\nContext: {joined}"


def build_persona_call(contract: PersonaContract, base: PersonaInput) -> PersonaCall:
    """Build one persona call from its contract and input (pure).

    Canonicalizes the topic and context items, so whitespace-only and
    order-only differences in the input cannot reach the call.
    """
    if not isinstance(contract, PersonaContract):
        raise TypeError(f"contract must be a PersonaContract, got {contract!r}")
    if not isinstance(base, PersonaInput):
        raise TypeError(f"base must be a PersonaInput, got {base!r}")
    return PersonaCall(
        persona=contract.persona,
        system_prompt=contract.system_prompt,
        user_message=_format_user_message(base.topic, base.context_items),
        max_tokens=contract.max_tokens,
        model=base.model,
    )


BuildFn = Callable[[PersonaContract, PersonaInput], PersonaCall]


def _whitespace_perturb_topic(topic: str) -> str:
    """Add trailing spaces to every line and blank lines at both ends."""
    return "\n\n" + "\n".join(line + "   " for line in topic.split("\n")) + "\n\n"


def apply_transformation(
    kind: TransformationKind,
    base: PersonaInput,
    *,
    alternate_model: str | None = None,
    alternate_topic: str | None = None,
) -> PersonaInput:
    """Apply one declared transformation to an input (pure).

    MODEL_SUBSTITUTION needs ``alternate_model`` (a different, non-empty
    model — the same model would make the check vacuous, so it is
    refused). TOPIC_SUBSTITUTION needs ``alternate_topic`` (whose canonical
    form must differ from the base topic's, else refused). WHITESPACE and
    CONTEXT_ORDER take no parameters.
    """
    if not isinstance(kind, TransformationKind):
        raise TypeError(f"kind must be a TransformationKind, got {kind!r}")
    if not isinstance(base, PersonaInput):
        raise TypeError(f"base must be a PersonaInput, got {base!r}")
    if kind is TransformationKind.MODEL_SUBSTITUTION:
        if alternate_model is None:
            raise ValueError("MODEL_SUBSTITUTION needs alternate_model")
        _check_str(alternate_model, "alternate_model")
        if not alternate_model.strip():
            raise ValueError("alternate_model must be a non-empty string")
        if alternate_model == base.model:
            raise ValueError(
                "alternate_model must differ from the base model, "
                "else the invariance check is vacuous"
            )
        return PersonaInput(base.topic, base.context_items, alternate_model)
    if kind is TransformationKind.TOPIC_SUBSTITUTION:
        if alternate_topic is None:
            raise ValueError("TOPIC_SUBSTITUTION needs alternate_topic")
        _check_str(alternate_topic, "alternate_topic")
        if _canonical_topic(alternate_topic) == _canonical_topic(base.topic):
            raise ValueError(
                "alternate_topic must differ from the base topic "
                "(after canonicalization), else the equivariance "
                "check is vacuous"
            )
        return PersonaInput(alternate_topic, base.context_items, base.model)
    if kind is TransformationKind.WHITESPACE:
        return PersonaInput(
            _whitespace_perturb_topic(base.topic),
            tuple(item + "   " for item in base.context_items),
            base.model,
        )
    # CONTEXT_ORDER: reversal is a deterministic permutation (and its own
    # inverse). The property tests cover arbitrary permutations.
    return PersonaInput(base.topic, tuple(reversed(base.context_items)), base.model)


@dataclass(frozen=True)
class SymmetryCheck:
    """One declared symmetry, checked: pass/fail plus the evidence."""

    persona: PersonaName
    transformation: TransformationKind
    kind: SymmetryKind
    passed: bool
    detail: str


@dataclass(frozen=True)
class SymmetryViolation:
    """One failed symmetry: what was expected, what the call did."""

    persona: PersonaName
    transformation: TransformationKind
    kind: SymmetryKind
    expected: str
    actual: str


@dataclass(frozen=True)
class SymmetryReport:
    """The gate's verdict for one persona over one base input."""

    persona: PersonaName
    verdict: SymmetryVerdict
    checks: tuple[SymmetryCheck, ...]
    violations: tuple[SymmetryViolation, ...]

    def summary(self) -> str:
        """One line per check plus the verdict, for CLI/demo output."""
        lines = [f"persona {self.persona.value}: {self.verdict.value}"]
        for check in self.checks:
            mark = "PASS" if check.passed else "FAIL"
            lines.append(
                f"  [{mark}] {check.transformation.value} ({check.kind.value}): {check.detail}"
            )
        for violation in self.violations:
            lines.append(
                f"  violation {violation.transformation.value}: "
                f"expected {violation.expected}; actual {violation.actual}"
            )
        return "\n".join(lines)


def _payload(call: PersonaCall) -> tuple[str, str, int]:
    """The model-independent payload: everything but the model field."""
    return (call.system_prompt, call.user_message, call.max_tokens)


def check_symmetry(
    contract: PersonaContract,
    base: PersonaInput,
    *,
    build_fn: BuildFn = build_persona_call,
    alternate_model: str,
    alternate_topic: str,
) -> SymmetryReport:
    """Check one persona's declared symmetries over one base input.

    ``build_fn`` is injectable so the gate can be tested against
    deliberately asymmetric stand-in builders (the negative controls):
    the gate must FAIL on them, with the violated transformation named.
    ``alternate_model`` and ``alternate_topic`` are required (no
    defaults) so no caller can run the gate with a vacuous alternate.
    """
    if not isinstance(contract, PersonaContract):
        raise TypeError(f"contract must be a PersonaContract, got {contract!r}")
    if not isinstance(base, PersonaInput):
        raise TypeError(f"base must be a PersonaInput, got {base!r}")
    base_call = build_fn(contract, base)
    checks: list[SymmetryCheck] = []
    violations: list[SymmetryViolation] = []
    for declared in contract.symmetries:
        transformation = declared.transformation
        kind = declared.kind
        if kind is SymmetryKind.INVARIANT:
            if transformation is TransformationKind.MODEL_SUBSTITUTION:
                alt = apply_transformation(transformation, base, alternate_model=alternate_model)
                alt_call = build_fn(contract, alt)
                passed = _payload(alt_call) == _payload(base_call)
                detail = (
                    "payload identical across models" if passed else "payload differs across models"
                )
                expected = f"payload {_payload(base_call)!r}"
                actual = f"payload {_payload(alt_call)!r}"
            else:
                alt = apply_transformation(transformation, base)
                alt_call = build_fn(contract, alt)
                passed = alt_call == base_call
                detail = (
                    f"call identical after {transformation.value}"
                    if passed
                    else f"call differs after {transformation.value}"
                )
                expected = f"call {base_call!r}"
                actual = f"call {alt_call!r}"
        else:  # EQUIVARIANT: TOPIC_SUBSTITUTION is the only checkable pair.
            alt = apply_transformation(transformation, base, alternate_topic=alternate_topic)
            alt_call = build_fn(contract, alt)
            expected_message = _format_user_message(alternate_topic, base.context_items)
            passed = (
                alt_call.user_message == expected_message
                and alt_call.system_prompt == base_call.system_prompt
                and alt_call.max_tokens == base_call.max_tokens
                and alt_call.model == base_call.model
            )
            detail = (
                "topic replaced exactly in the user message, nothing else changed"
                if passed
                else "topic substitution did not move exactly"
            )
            expected = (
                f"user_message {expected_message!r}, system_prompt "
                f"{base_call.system_prompt!r}, max_tokens "
                f"{base_call.max_tokens}, model {base_call.model!r}"
            )
            actual = (
                f"user_message {alt_call.user_message!r}, system_prompt "
                f"{alt_call.system_prompt!r}, max_tokens "
                f"{alt_call.max_tokens}, model {alt_call.model!r}"
            )
        checks.append(SymmetryCheck(contract.persona, transformation, kind, passed, detail))
        if not passed:
            violations.append(
                SymmetryViolation(contract.persona, transformation, kind, expected, actual)
            )
    verdict = SymmetryVerdict.PASS if not violations else SymmetryVerdict.FAIL
    return SymmetryReport(
        persona=contract.persona,
        verdict=verdict,
        checks=tuple(checks),
        violations=tuple(violations),
    )


# Prompts copied verbatim from .github/agent-coffee/run_coffee.py
# (PERSONA_PROMPTS). The sync test in tests/test_symmetry.py fails CI if
# either copy drifts; run_coffee.py cannot import this module (it runs
# standalone in GitHub Actions without the package installed).
PERSONA_CONTRACTS: MappingProxyType[PersonaName, PersonaContract] = MappingProxyType(
    {
        PersonaName.BUILDER: PersonaContract(
            persona=PersonaName.BUILDER,
            system_prompt=(
                "You are The Builder. Propose the most direct solution "
                "in <=10 lines. Bias to shipping."
            ),
            max_tokens=400,
            symmetries=ALL_SYMMETRIES,
        ),
        PersonaName.CRITIC: PersonaContract(
            persona=PersonaName.CRITIC,
            system_prompt=(
                "You are The Jealous Critic. You are adversarial and "
                "self-protective, envious of attention the proposal "
                "gets. Hunt every weakness: hidden assumptions, cost "
                "blowups, scope creep. Try to kill the proposal in "
                "<=10 lines. No politeness padding."
            ),
            max_tokens=400,
            symmetries=ALL_SYMMETRIES,
        ),
        PersonaName.GUARDIAN: PersonaContract(
            persona=PersonaName.GUARDIAN,
            system_prompt=(
                "You are The Guardian. Rule on safety, scope and cost "
                "in <=5 lines. Veto anything irreversible."
            ),
            max_tokens=400,
            symmetries=ALL_SYMMETRIES,
        ),
        PersonaName.DREAMER: PersonaContract(
            persona=PersonaName.DREAMER,
            system_prompt=(
                "You are The Dreamer. Offer one wild alternative in <=5 lines. Break the frame."
            ),
            max_tokens=400,
            symmetries=ALL_SYMMETRIES,
        ),
    }
)

# Canonical sample input for the demo, the CLI command, and the tests.
# The model is run_coffee.py's default (JEV_MODEL); the alternate is a
# different model family the jev router could pick instead.
SAMPLE_INPUT = PersonaInput(
    topic="Should the eval suite add a symmetry gate for personas?",
    context_items=(
        "Geometric Deep Learning links symmetry to stability",
        "personas deliberate in rounds",
        "CI must fail on violation",
    ),
    model="gpt-4o-mini",
)
SAMPLE_ALTERNATE_MODEL = "claude-haiku-4-5"
SAMPLE_ALTERNATE_TOPIC = "What breaks first when a persona's symmetry contract is violated?"


def check_all_personas(
    base: PersonaInput,
    *,
    build_fn: BuildFn = build_persona_call,
    alternate_model: str = SAMPLE_ALTERNATE_MODEL,
    alternate_topic: str = SAMPLE_ALTERNATE_TOPIC,
) -> tuple[SymmetryReport, ...]:
    """Check every registered persona's contract over one base input."""
    return tuple(
        check_symmetry(
            contract,
            base,
            build_fn=build_fn,
            alternate_model=alternate_model,
            alternate_topic=alternate_topic,
        )
        for contract in PERSONA_CONTRACTS.values()
    )
