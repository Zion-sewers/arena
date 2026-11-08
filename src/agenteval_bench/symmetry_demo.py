"""Seeded end-to-end demo of the persona symmetry gate (arena#52).

Runs the symmetry gate offline over the four coffee-break personas. Exit
0 only if every acceptance expectation holds:
- every registered persona declares a non-empty symmetry contract, and
  the gate PASSES all four personas on the canonical sample input;
- a stand-in builder that bakes the model name into the prompt FAILs
  with MODEL_SUBSTITUTION named;
- a stand-in builder that joins context unsorted FAILs with
  CONTEXT_ORDER named;
- a stand-in builder that skips canonicalization FAILs with WHITESPACE
  named;
- a stand-in builder that drops the topic FAILs with TOPIC_SUBSTITUTION
  named.

Any failure raises, so CI can gate on this demo directly.
"""

from __future__ import annotations

import sys
from dataclasses import replace

from agenteval_bench.symmetry import (
    PERSONA_CONTRACTS,
    SAMPLE_ALTERNATE_MODEL,
    SAMPLE_ALTERNATE_TOPIC,
    SAMPLE_INPUT,
    PersonaCall,
    PersonaContract,
    PersonaInput,
    SymmetryVerdict,
    TransformationKind,
    _canonical_context_items,
    _canonical_topic,
    build_persona_call,
    check_all_personas,
    check_symmetry,
)


def _model_naming_builder(contract: PersonaContract, base: PersonaInput) -> PersonaCall:
    call = build_persona_call(contract, base)
    return replace(call, system_prompt=f"[{base.model}] {call.system_prompt}")


def _unsorted_builder(contract: PersonaContract, base: PersonaInput) -> PersonaCall:
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
    message = "Context: " + "\n".join(_canonical_context_items(base.context_items))
    return PersonaCall(
        contract.persona,
        contract.system_prompt,
        message,
        contract.max_tokens,
        base.model,
    )


def main() -> int:
    # 1. Every persona declares a contract, and the gate passes all four.
    assert len(PERSONA_CONTRACTS) == 4
    reports = check_all_personas(SAMPLE_INPUT)
    for report in reports:
        print(report.summary())
        assert report.verdict is SymmetryVerdict.PASS, report.summary()
        assert report.checks and not report.violations

    # 2-5. Each violation class is demonstrated on a stand-in builder:
    # the gate must FAIL, naming the violated transformation.
    builder = PERSONA_CONTRACTS[reports[0].persona]
    kwargs = {
        "alternate_model": SAMPLE_ALTERNATE_MODEL,
        "alternate_topic": SAMPLE_ALTERNATE_TOPIC,
    }
    cases = [
        (_model_naming_builder, TransformationKind.MODEL_SUBSTITUTION),
        (_unsorted_builder, TransformationKind.CONTEXT_ORDER),
        (_raw_builder, TransformationKind.WHITESPACE),
        (_topic_dropping_builder, TransformationKind.TOPIC_SUBSTITUTION),
    ]
    for build_fn, transformation in cases:
        report = check_symmetry(builder, SAMPLE_INPUT, build_fn=build_fn, **kwargs)
        print(report.summary())
        assert report.verdict is SymmetryVerdict.FAIL, report.summary()
        assert any(v.transformation is transformation for v in report.violations), report.summary()

    print("symmetry demo: all 5 expectations hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
