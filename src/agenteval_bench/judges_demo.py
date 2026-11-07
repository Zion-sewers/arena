"""Seeded end-to-end demo of judge decorrelation (arena#44).

Runs the full HOST != FAMILY allocation offline on in-memory registries
plus one real CLI invocation on a temporary registry file. Exit 0 only if
every acceptance expectation holds:
- the documented trap: nvidia-nim + openrouter both serving deepseek-r1
  (different hosts, same family) allocate ONE judge; the second is named
  as rejected, not silently dropped;
- the same host serving two families still pairs (host is irrelevant);
- a VARIES-BY-MODEL reseller alone is picked as a single opinion with a
  note; with a known-family partner available it is excluded from the
  pair, in either slot order;
- unreachable judges (declared, and probe-reported) appear as explicit
  holes, never picked, never dropped; a probe that raises becomes a hole
  with the exception named, not a crash;
- a thin registry (want 2, one family) returns one pick and says so;
- allocation is deterministic: the same registry allocates identically twice;
- the registry loader fails closed on duplicate ids, unknown keys and a
  missing required field;
- the CLI `judges` command on a registry file exits 0 when satisfied and
  writes the report JSON with the holes in it.

The fixture judges mirror the issue's named source (claude-setup
tools/review/actors.json): nvidia-nim and openrouter as multi-vendor
resellers, the VARIES-BY-MODEL reseller, and the measured / declared /
unreachable reachability legend. The reachability values in the demo
registry below are fixture labels, not measurements: no probe runs in
this demo, and the CLI runs no probe either. Any failure raises, so CI
can gate on this demo directly.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from agenteval_bench.cli import cmd_judges
from agenteval_bench.judges import (
    UNKNOWN_FAMILY,
    Judge,
    allocate,
    load_registry,
    registry_from_json,
)

# The documented trap, as fixture data: two hosts, one family.
TRAP = [
    Judge("deepseek-nim", "nvidia-nim", "deepseek-r1", "deepseek-ai/DeepSeek-R1"),
    Judge("deepseek-or", "openrouter", "deepseek-r1", "deepseek/deepseek-r1"),
    Judge("qwen-ds", "alibaba-dashscope", "alibaba-qwen", "qwen3-flash"),
]


def _expect(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"  ok    {label}")


def main() -> int:
    print("Judge decorrelation demo (arena#44)")

    # 1. The trap: different hosts, same family -> one pick, named rejection.
    report = allocate(TRAP, want=2)
    _expect(len(report.picked) == 2, "two decorrelated judges picked from the trap")
    _expect(
        {j.model_family for j in report.picked} == {"deepseek-r1", "alibaba-qwen"},
        "the two picks carry distinct model families",
    )
    _expect(
        len(report.rejected) == 1 and report.rejected[0].judge == "deepseek-or",
        "the second deepseek host is rejected by name, not silently dropped",
    )
    _expect(
        "host diversity is not decorrelation" in report.rejected[0].reason,
        "the rejection says why host diversity is not decorrelation",
    )

    # 2. Same host, different families still pairs: host is irrelevant.
    same_host = [
        Judge("qwen-nim", "nvidia-nim", "alibaba-qwen", "qwen3-flash"),
        Judge("llama-nim", "nvidia-nim", "meta-llama", "llama-4-scout"),
    ]
    report = allocate(same_host, want=2)
    _expect(len(report.picked) == 2, "one host serving two families still pairs")
    _expect(report.satisfied, "same-host pair satisfies want=2")

    # 3. VARIES-BY-MODEL: single only, never half of a pair, either order.
    reseller = Judge("reseller", "nvidia-nim", UNKNOWN_FAMILY, "varies")
    partner = Judge("qwen-ds", "alibaba-dashscope", "alibaba-qwen", "qwen3-flash")
    solo = allocate([reseller], want=2)
    _expect(
        [j.id for j in solo.picked] == ["reseller"],
        "an unknown-family judge alone is picked as a single opinion",
    )
    _expect(
        any("single opinion" in n for n in solo.notes),
        "the single-opinion pick says why it cannot pair",
    )
    for ordered in ([reseller, partner], [partner, reseller]):
        paired = allocate(ordered, want=2)
        _expect(
            "reseller" not in [j.id for j in paired.picked] or len(paired.picked) == 1,
            "a VARIES-BY-MODEL reseller is never half of a pair",
        )
        _expect(
            any("cannot be shown to differ" in n for n in paired.notes),
            "the reseller exclusion is named in the notes",
        )

    # 4. Unreachable judges are holes: declared, probe-reported, probe-raising.
    dead = Judge("codex", "openai", "openai-gpt", "gpt-5", "unreachable")
    report = allocate([*TRAP, dead], want=2)
    _expect(
        len(report.holes) == 1 and report.holes[0].judge == "codex",
        "a declared-unreachable judge appears as an explicit hole",
    )
    _expect(
        "codex" not in [j.id for j in report.picked],
        "an unreachable judge is never picked",
    )

    def flaky_probe(judge: Judge) -> bool:
        if judge.id == "qwen-ds":
            return False
        if judge.id == "deepseek-or":
            raise ConnectionError("host timed out")
        return True

    report = allocate(TRAP, want=2, probe=flaky_probe)
    _expect(
        {h.judge for h in report.holes} == {"qwen-ds", "deepseek-or"},
        "probe-reported and probe-raising judges both become holes",
    )
    _expect(
        any("ConnectionError" in h.reason for h in report.holes),
        "the probe-raising hole names the exception",
    )
    _expect(
        [j.id for j in report.picked] == ["deepseek-nim"],
        "the probe run still picks the reachable judge",
    )
    _expect(
        report.picked[0].reachable == "measured",
        "a probe-True verdict stamps the pick measured, never unreachable",
    )
    _expect(
        any("probe measured reachable" in n for n in report.notes),
        "the probe override is noted in the report",
    )

    # 5. Thin registry says so.
    thin = allocate(TRAP[:2], want=2)
    _expect(len(thin.picked) == 1, "one family supplies one pick, not two")
    _expect(not thin.satisfied, "the thin allocation reports unsatisfied")
    _expect(
        any("wanted 2" in n for n in thin.notes),
        "the thin allocation says why it is thin",
    )

    # 6. Deterministic.
    _expect(
        allocate(TRAP, want=2).to_dict() == allocate(TRAP, want=2).to_dict(),
        "the same registry allocates identically twice",
    )

    # 7. The loader fails closed.
    for bad, label in (
        (
            [
                {"id": "a", "host": "h", "model_family": "f"},
                {"id": "a", "host": "h", "model_family": "g"},
            ],
            "duplicate judge ids are refused",
        ),
        (
            [{"id": "a", "host": "h", "model_family": "f", "model_familiy": "x"}],
            "an unknown registry key is refused",
        ),
        (
            [{"id": "a", "host": "h"}],
            "a missing model_family is refused",
        ),
    ):
        try:
            load_registry(bad)
        except ValueError:
            _expect(True, label)
        else:
            raise AssertionError(label)

    # 8. CLI end to end: satisfied run exits 0 and writes the report JSON.
    registry = {
        "judges": [
            {
                "id": "deepseek-nim",
                "host": "nvidia-nim",
                "model_family": "deepseek-r1",
                "model": "deepseek-ai/DeepSeek-R1",
                "reachable": "declared",
            },
            {
                "id": "qwen-ds",
                "host": "alibaba-dashscope",
                "model_family": "alibaba-qwen",
                "model": "qwen3-flash",
                "reachable": "declared",
            },
            {
                "id": "codex",
                "host": "openai",
                "model_family": "openai-gpt",
                "model": "gpt-5",
                "reachable": "unreachable",
            },
        ]
    }
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "registry.json"
        reg_path.write_text(json.dumps(registry), encoding="utf-8")
        out_path = Path(tmp) / "allocation.json"
        code = cmd_judges(["--registry", str(reg_path), "--want", "2", "--out", str(out_path)])
        _expect(code == 0, "the CLI judges command exits 0 when satisfied")
        written = json.loads(out_path.read_text(encoding="utf-8"))
        _expect(written["satisfied"] is True, "the written report is satisfied")
        _expect(
            [h["judge"] for h in written["holes"]] == ["codex"],
            "the written report carries the unreachable judge as a hole",
        )
        _expect(
            registry_from_json(reg_path.read_text(encoding="utf-8"))[0].id == "deepseek-nim",
            "the JSON registry loader round-trips the file",
        )

    print("VERDICT: the allocator decorrelates on family and reports every hole")
    return 0


if __name__ == "__main__":
    sys.exit(main())
