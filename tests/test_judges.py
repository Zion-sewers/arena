"""Tests for judge decorrelation (arena#44): HOST != FAMILY.

Every acceptance criterion has a named test, and the property test at the
bottom pins the invariants for every generated registry: picked families
are pairwise distinct, no picked judge is unreachable, every unreachable
judge appears as a hole, and a VARIES-BY-MODEL judge is never half of a pair.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agenteval_bench.cli import cmd_judges
from agenteval_bench.judges import (
    UNKNOWN_FAMILY,
    Judge,
    allocate,
    load_registry,
    registry_from_json,
)

# The documented trap: two hosts, one family (claude-setup actors.json A.5).
TRAP = [
    Judge("deepseek-nim", "nvidia-nim", "deepseek-r1", "deepseek-ai/DeepSeek-R1"),
    Judge("deepseek-or", "openrouter", "deepseek-r1", "deepseek/deepseek-r1"),
    Judge("qwen-ds", "alibaba-dashscope", "alibaba-qwen", "qwen3-flash"),
]


def test_allocation_is_keyed_on_family_not_host() -> None:
    report = allocate(TRAP, want=2)
    assert [j.id for j in report.picked] == ["deepseek-nim", "qwen-ds"]
    assert report.families == ("deepseek-r1", "alibaba-qwen")
    assert report.satisfied


def test_same_family_pair_is_rejected_by_name() -> None:
    report = allocate(TRAP, want=2)
    assert len(report.rejected) == 1
    rejected = report.rejected[0]
    assert rejected.judge == "deepseek-or"
    assert rejected.family == "deepseek-r1"
    assert "host diversity is not decorrelation" in rejected.reason
    assert "deepseek-or" not in [j.id for j in report.picked]


def test_provider_name_diversity_is_not_decorrelation() -> None:
    # Three resellers, one family: the pair is still one pick.
    resellers = [
        Judge("d-nim", "nvidia-nim", "deepseek-r1"),
        Judge("d-or", "openrouter", "deepseek-r1"),
        Judge("d-together", "together", "deepseek-r1"),
    ]
    report = allocate(resellers, want=2)
    assert len(report.picked) == 1
    assert len(report.rejected) == 2
    assert not report.satisfied
    assert any("wanted 2" in n for n in report.notes)


def test_same_host_different_families_still_pairs() -> None:
    same_host = [
        Judge("qwen-nim", "nvidia-nim", "alibaba-qwen"),
        Judge("llama-nim", "nvidia-nim", "meta-llama"),
    ]
    report = allocate(same_host, want=2)
    assert len(report.picked) == 2
    assert report.satisfied
    assert not report.rejected


def test_unknown_family_alone_is_a_single_opinion() -> None:
    reseller = Judge("reseller", "nvidia-nim", UNKNOWN_FAMILY)
    report = allocate([reseller], want=2)
    assert [j.id for j in report.picked] == ["reseller"]
    assert any("single opinion" in n for n in report.notes)


def test_unknown_family_never_half_of_a_pair_either_order() -> None:
    reseller = Judge("reseller", "nvidia-nim", UNKNOWN_FAMILY)
    partner = Judge("qwen-ds", "alibaba-dashscope", "alibaba-qwen")
    for ordered in ([reseller, partner], [partner, reseller]):
        report = allocate(ordered, want=2)
        assert [j.id for j in report.picked] == ["qwen-ds"]
        assert any("cannot be shown to differ" in n for n in report.notes)


def test_declared_unreachable_judge_is_a_hole_never_picked() -> None:
    dead = Judge("codex", "openai", "openai-gpt", "gpt-5", "unreachable")
    report = allocate([*TRAP, dead], want=2)
    assert len(report.holes) == 1
    hole = report.holes[0]
    assert hole.judge == "codex"
    assert hole.host == "openai"
    assert hole.model_family == "openai-gpt"
    assert hole.reason == "declared unreachable"
    assert "codex" not in [j.id for j in report.picked]
    assert len(report.picked) == 2


def test_probe_false_becomes_a_hole() -> None:
    report = allocate(TRAP, want=2, probe=lambda j: j.id != "qwen-ds")
    assert [h.judge for h in report.holes] == ["qwen-ds"]
    assert [j.id for j in report.picked] == ["deepseek-nim"]
    assert report.holes[0].reason == "probe reported unreachable"


def test_probe_raising_becomes_a_hole_not_a_crash() -> None:
    def probe(judge: Judge) -> bool:
        if judge.id == "deepseek-or":
            raise ConnectionError("host timed out")
        return True

    report = allocate(TRAP, want=2, probe=probe)
    assert [h.judge for h in report.holes] == ["deepseek-or"]
    assert "ConnectionError" in report.holes[0].reason
    assert "host timed out" in report.holes[0].reason
    assert [j.id for j in report.picked] == ["deepseek-nim", "qwen-ds"]


def test_probe_revives_a_declared_unreachable_judge() -> None:
    dead = Judge("codex", "openai", "openai-gpt", "gpt-5", "unreachable")
    report = allocate([dead], want=1, probe=lambda j: True)
    assert [j.id for j in report.picked] == ["codex"]
    assert not report.holes


def test_probe_is_called_once_per_judge() -> None:
    seen: list[str] = []
    allocate(TRAP, want=2, probe=lambda j: seen.append(j.id) or True)
    assert sorted(seen) == sorted(j.id for j in TRAP)


def test_reachability_orders_picks_measured_first() -> None:
    judges = [
        Judge("slow-declared", "h1", "fam-a", reachable="declared"),
        Judge("fast-measured", "h2", "fam-b", reachable="measured"),
        Judge("also-measured", "h3", "fam-a", reachable="measured"),
    ]
    report = allocate(judges, want=2)
    assert [j.id for j in report.picked] == ["also-measured", "fast-measured"]


def test_thin_allocation_says_so() -> None:
    report = allocate(TRAP[:2], want=2)
    assert len(report.picked) == 1
    assert not report.satisfied
    assert any("wanted 2" in n for n in report.notes)


def test_empty_registry_allocates_nothing_and_says_so() -> None:
    report = allocate([], want=2)
    assert report.picked == ()
    assert report.holes == ()
    assert not report.satisfied
    assert any("nothing to allocate" in n for n in report.notes)


def test_allocation_is_deterministic() -> None:
    assert allocate(TRAP, want=2).to_dict() == allocate(TRAP, want=2).to_dict()


def test_want_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="want must be >= 1"):
        allocate(TRAP, want=0)


def test_duplicate_judge_ids_are_refused() -> None:
    dupes = [Judge("a", "h1", "fam-a"), Judge("a", "h2", "fam-b")]
    with pytest.raises(ValueError, match="duplicate judge id"):
        allocate(dupes, want=2)


def test_judge_with_empty_family_is_refused() -> None:
    with pytest.raises(ValueError, match="model_family must be non-empty"):
        Judge("a", "h1", "")


def test_judge_with_bad_reachability_is_refused() -> None:
    with pytest.raises(ValueError, match="reachable must be one of"):
        Judge("a", "h1", "fam-a", reachable="sometimes")  # type: ignore[arg-type]


def test_report_round_trips_through_json() -> None:
    dead = Judge("codex", "openai", "openai-gpt", "gpt-5", "unreachable")
    report = allocate([*TRAP, dead], want=2)
    data = json.loads(json.dumps(report.to_dict()))
    assert data["satisfied"] is True
    assert [p["id"] for p in data["picked"]] == ["deepseek-nim", "qwen-ds"]
    assert data["families"] == ["deepseek-r1", "alibaba-qwen"]
    assert [h["judge"] for h in data["holes"]] == ["codex"]
    assert [r["judge"] for r in data["rejected"]] == ["deepseek-or"]


def test_registry_loader_accepts_mapping_and_bare_list() -> None:
    entry = {
        "id": "a",
        "host": "h",
        "model_family": "fam-a",
        "model": "m",
        "reachable": "measured",
    }
    from_mapping = load_registry({"judges": [entry]})
    from_list = load_registry([entry])
    assert from_mapping == from_list
    assert from_mapping[0].reachable == "measured"
    assert from_mapping[0].model == "m"


def test_registry_loader_defaults_reachability_and_model() -> None:
    (judge,) = load_registry([{"id": "a", "host": "h", "model_family": "fam-a"}])
    assert judge.reachable == "declared"
    assert judge.model == ""


def test_registry_loader_fails_closed() -> None:
    with pytest.raises(ValueError, match="duplicate judge id"):
        load_registry(
            [
                {"id": "a", "host": "h", "model_family": "f"},
                {"id": "a", "host": "h", "model_family": "g"},
            ]
        )
    with pytest.raises(ValueError, match="unknown key"):
        load_registry([{"id": "a", "host": "h", "model_family": "f", "model_familiy": "x"}])
    with pytest.raises(ValueError, match="missing required field"):
        load_registry([{"id": "a", "host": "h"}])
    with pytest.raises(ValueError, match="must carry a 'judges' list"):
        load_registry({"judges_missing": []})
    with pytest.raises(ValueError, match="not valid JSON"):
        registry_from_json("{not json")


def test_cli_judges_satisfied_exits_zero_and_writes_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
    code = cmd_judges(["--registry", str(reg_path), "--want", "2", "--out", str(out_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "fam-a" in out and "fam-b" in out
    assert "HOLES" in out and "dead" in out
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["satisfied"] is True
    assert [h["judge"] for h in written["holes"]] == ["dead"]


def test_cli_judges_thin_exits_two(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps({"judges": [{"id": "a", "host": "h1", "model_family": "fam-a"}]}),
        encoding="utf-8",
    )
    assert cmd_judges(["--registry", str(reg_path), "--want", "2"]) == 2


def test_cli_judges_bad_registry_exits_one(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text("{not json", encoding="utf-8")
    assert cmd_judges(["--registry", str(reg_path)]) == 1
    assert cmd_judges(["--registry", str(tmp_path / "missing.json")]) == 1


def test_cli_judges_json_flag_prints_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps({"judges": [{"id": "a", "host": "h1", "model_family": "fam-a"}]}),
        encoding="utf-8",
    )
    assert cmd_judges(["--registry", str(reg_path), "--want", "1", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["satisfied"] is True
    assert data["picked"][0]["id"] == "a"


_FAMILIES = st.sampled_from(["fam-a", "fam-b", "fam-c", UNKNOWN_FAMILY])
_HOSTS = st.sampled_from(["nvidia-nim", "openrouter", "h3"])
_REACH = st.sampled_from(["measured", "declared", "unreachable"])


@settings(max_examples=100, deadline=None)
@given(
    specs=st.lists(
        st.tuples(_FAMILIES, _HOSTS, _REACH),
        min_size=0,
        max_size=8,
    ),
    want=st.integers(min_value=1, max_value=3),
)
def test_allocate_invariants_hold_for_every_registry(
    specs: list[tuple[str, str, str]], want: int
) -> None:
    judges = [
        Judge(f"j{i}", host, family, reachable=reach)  # type: ignore[arg-type]
        for i, (family, host, reach) in enumerate(specs)
    ]
    report = allocate(judges, want=want)
    picked_families = [j.model_family for j in report.picked]
    # No two picks share a family: allocation is keyed on family.
    assert len(set(picked_families)) == len(picked_families)
    # No picked judge is unreachable.
    assert all(j.reachable != "unreachable" for j in report.picked)
    # Every unreachable judge appears as a hole.
    unreachable = {j.id for j in judges if j.reachable == "unreachable"}
    assert unreachable == {h.judge for h in report.holes}
    # An unknown-family judge is never half of a pair.
    if len(report.picked) > 1:
        assert UNKNOWN_FAMILY not in picked_families
    # Picks never exceed the request; a satisfied report meets it.
    assert len(report.picked) <= want
    assert report.satisfied == (len(report.picked) >= want)
    # Every rejection names a family already picked.
    for rejected in report.rejected:
        assert rejected.family in picked_families
