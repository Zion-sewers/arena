"""Adversarial tests for the bugsmith pipeline (agenteval-bench#37).

Every behavior change ships with a running test — prose is not
enforcement. The Docker path itself is exercised by the CI demo
(``python -m bugsmith.demo``); these tests pin the generation,
validation-math, curation, and patch-application contracts with the
local runner.
"""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
from pathlib import Path

import pytest

from bugsmith.buggen import (
    LLMGeneratedBugGenerator,
    PRMirrorGenerator,
    ProceduralBugGenerator,
    _collect_sites,
    _mutate_source,
)
from bugsmith.curate import instance_id, select_subset, to_instance
from bugsmith.harness import (
    LocalRunner,
    _parse_collected,
    baseline,
    parse_pytest_output,
    resolve_node_id,
    validate,
)
from bugsmith.harness import (
    TestOutcome as HarnessOutcome,
)
from bugsmith.images import build_context, dockerfile_text, repo_digest
from bugsmith.patch import PatchError, apply_patch, split_files
from bugsmith.types import (
    BugCandidate,
    BugsmithError,
    BugStrategy,
    CurationConfig,
    GenerationRecord,
    ValidationReport,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "src" / "bugsmith" / "fixtures" / "target"


def _record(**kw) -> GenerationRecord:
    base = {
        "strategy": BugStrategy.PROCEDURAL_AST,
        "seed": 1,
        "generator_version": "test",
        "target_file": "src/mcalc/arithmetic.py",
        "site_description": "test",
    }
    base.update(kw)
    return GenerationRecord(**base)  # type: ignore[arg-type]


def _report(fail=(), passed=(), base=()) -> ValidationReport:
    return ValidationReport(
        candidate=BugCandidate(record=_record(), patch="--- a/x\n+++ b/x\n"),
        baseline_passed=tuple(base),
        failed_after=tuple(fail),
        passed_after=tuple(passed),
    )


# --- generation -----------------------------------------------------------


def test_procedural_generation_is_seed_reproducible() -> None:
    a = ProceduralBugGenerator(7).generate(FIXTURE, 6)
    b = ProceduralBugGenerator(7).generate(FIXTURE, 6)
    assert [c.patch for c in a] == [c.patch for c in b]
    assert all(c.patch.strip() for c in a)
    c = ProceduralBugGenerator(8).generate(FIXTURE, 6)
    assert [x.patch for x in c] != [x.patch for x in a]


def test_every_generated_patch_applies_and_parses(tmp_path: Path) -> None:
    n = 0
    for seed in range(6):
        for candidate in ProceduralBugGenerator(seed).generate(FIXTURE, 4):
            n += 1
            work = tmp_path / f"s{seed}_{n}"
            shutil.copytree(
                FIXTURE, work,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            applied = apply_patch(work, candidate.patch)
            for rel, text in applied.items():
                ast.parse(text)  # the generator never emits unparseable code
                assert text != (FIXTURE / rel).read_text(encoding="utf-8")


def test_site_mutation_hits_exact_node() -> None:
    source = "def f(a, b):\n    return a + b + 1\n"
    tree = ast.parse(source)
    sites = _collect_sites(tree)
    binops = [s for s in sites if s.kind is ast.BinOp]
    assert len(binops) == 2  # (a+b) and ((a+b)+1) are distinct sites
    mutated = _mutate_source(source, binops[0])
    assert mutated != source
    ast.parse(mutated)


def test_procedural_never_mutates_test_files() -> None:
    for seed in range(10):
        for candidate in ProceduralBugGenerator(seed).generate(FIXTURE, 5):
            assert "/tests/" not in candidate.record.target_file.replace("\\", "/")
            assert "test" not in Path(candidate.record.target_file).parts
            assert candidate.record.target_file.startswith("src/mcalc/")


def test_procedural_exclude_never_mutates_excluded_paths() -> None:
    """The tiered pipeline excludes its own scoring path (src/bugsmith)
    so a mutant there cannot fabricate the score: the generator must
    never return a candidate from an excluded directory, and the
    exclusion must be a no-op for other callers (default: empty)."""
    for seed in range(10):
        for candidate in ProceduralBugGenerator(seed).generate(
            FIXTURE, 5, exclude=("src/mcalc/arithmetic.py",)
        ):
            assert candidate.record.target_file != "src/mcalc/arithmetic.py"
    # Excluding everything still in the pool raises, never silently empty.
    with pytest.raises(Exception, match="no source files"):
        ProceduralBugGenerator(7).generate(FIXTURE, 1, exclude=("src/",))


def test_llm_generator_refuses_without_backend() -> None:
    with pytest.raises(BugsmithError):
        LLMGeneratedBugGenerator()


def test_llm_generator_applies_injected_rewrite() -> None:
    def rewrite(source: str, target_file: str, hint: str) -> str:
        return source.replace("return a + b", "return a - b")

    gen = LLMGeneratedBugGenerator(rewrite_fn=rewrite, seed=3)
    (candidate,) = gen.generate(FIXTURE, ["src/mcalc/arithmetic.py"])
    assert candidate.record.strategy is BugStrategy.LLM_GENERATED
    assert candidate.record.seed == 3
    assert "-    return a + b" in candidate.patch
    assert "+    return a - b" in candidate.patch


def test_llm_generator_rejects_identical_rewrite() -> None:
    gen = LLMGeneratedBugGenerator(rewrite_fn=lambda s, f, h: s)
    with pytest.raises(BugsmithError):
        gen.generate(FIXTURE, ["src/mcalc/arithmetic.py"])


def test_pr_mirror_wraps_real_patch() -> None:
    patch = (REPO_ROOT / "src" / "bugsmith" / "fixtures"
             / "pr_mirror_median.patch").read_text(encoding="utf-8")
    gen = PRMirrorGenerator(pr_ref="mcalc#41-revert")
    (candidate,) = gen.generate(patch, ["src/mcalc/stats.py"])
    assert candidate.patch == patch
    assert candidate.record.strategy is BugStrategy.PR_MIRROR
    assert candidate.record.seed is None
    assert candidate.record.extra["pr_ref"] == "mcalc#41-revert"


def test_pr_mirror_requires_pr_ref() -> None:
    with pytest.raises(BugsmithError):
        PRMirrorGenerator(pr_ref="")


# --- patch application ----------------------------------------------------


def test_patch_applier_roundtrip_pr_mirror(tmp_path: Path) -> None:
    patch = (REPO_ROOT / "src" / "bugsmith" / "fixtures"
             / "pr_mirror_median.patch").read_text(encoding="utf-8")
    work = tmp_path / "t"
    shutil.copytree(FIXTURE, work, ignore=shutil.ignore_patterns("__pycache__"))
    applied = apply_patch(work, patch)
    assert applied == {"src/mcalc/stats.py": applied["src/mcalc/stats.py"]}
    assert "    return ordered[mid]\n" in applied["src/mcalc/stats.py"]
    assert "(ordered[mid - 1] + ordered[mid]) / 2" not in applied["src/mcalc/stats.py"]


def test_patch_applier_refuses_path_traversal(tmp_path: Path) -> None:
    evil = "--- a/../evil.py\n+++ b/../evil.py\n@@ -1,0 +1,1 @@\n+x = 1\n"
    with pytest.raises(PatchError):
        split_files(evil)
    inside = "--- a/sub/x.py\n+++ b/sub/x.py\n@@ -1,0 +1,1 @@\n+x = 1\n"
    assert list(split_files(inside)) == ["sub/x.py"]


def test_patch_applier_refuses_context_mismatch(tmp_path: Path) -> None:
    work = tmp_path / "t"
    shutil.copytree(FIXTURE, work, ignore=shutil.ignore_patterns("__pycache__"))
    bad = ("--- a/src/mcalc/stats.py\n+++ b/src/mcalc/stats.py\n"
           "@@ -15,3 +15,3 @@ def median(values: list[float]) -> float:\n"
           "     not the real context\n"
           "     n = len(ordered)\n"
           "     mid = n // 2\n")
    with pytest.raises(PatchError):
        apply_patch(work, bad)


# --- validation math ------------------------------------------------------


def test_fail_to_pass_and_pass_to_pass_math() -> None:
    base = ("t1", "t2", "t3")
    r = _report(fail=("t1", "tx"), passed=("t2",), base=base)
    assert r.fail_to_pass == ("t1",)  # tx never passed: not a target
    assert r.pass_to_pass == ("t2",)
    assert r.is_valid_instance


def test_zero_break_candidate_is_discarded() -> None:
    r = _report(fail=(), passed=("t1", "t2"), base=("t1", "t2"))
    assert not r.is_valid_instance


def test_parse_pytest_output_failure_lines() -> None:
    out = ("F..\n==== short test summary info ====\n"
           "FAILED tests/test_a.py::test_x - assert 1 == 2\n"
           "ERROR tests/test_b.py::test_y - boom\n"
           "==== 1 failed, 1 error ====\n")
    o = parse_pytest_output(out, 1)
    assert o.failed == frozenset({"tests/test_a.py::test_x", "tests/test_b.py::test_y"})
    assert not o.collection_error


def test_parse_pytest_output_keeps_spaced_node_ids() -> None:
    # Parametrized IDs with spaces must survive parsing intact: the stored
    # FAIL_TO_PASS lists are only useful if they address real tests.
    out = ("F.\n==== short test summary info ====\n"
           "FAILED tests/test_p.py::test_add_param[a - b] - assert 1 == 2\n"
           "FAILED tests/test_p.py::test_add_param[plain] - assert 3 == 4\n"
           "==== 2 failed ====\n")
    o = parse_pytest_output(out, 1)
    assert o.failed == frozenset({
        "tests/test_p.py::test_add_param[a - b]",
        "tests/test_p.py::test_add_param[plain]",
    })


def test_parse_collected_keeps_spaced_node_ids() -> None:
    out = ("tests/test_p.py::test_add_param[a - b]\n"
           "tests/test_p.py::test_add_param[plain]\n")
    ids = _parse_collected(out)
    assert ids == frozenset({
        "tests/test_p.py::test_add_param[a - b]",
        "tests/test_p.py::test_add_param[plain]",
    })


def test_resolve_node_id_disambiguates_reason_with_separator() -> None:
    # The exact ambiguity class: both the node ID and the failure reason
    # contain " - ". Resolution must use the collected ID set, not a blind
    # split.
    known = frozenset({
        "tests/test_sp.py::test_lab[a - b]",
        "tests/test_sp.py::test_lab[plain]",
    })
    tail = "tests/test_sp.py::test_lab[a - b] - AssertionError: assert 'a - b' == 'plain'"
    assert resolve_node_id(tail, known) == "tests/test_sp.py::test_lab[a - b]"
    # Exact match still wins when there is no reason at all.
    assert resolve_node_id("tests/test_sp.py::test_lab[plain]", known) == \
        "tests/test_sp.py::test_lab[plain]"
    # No known set: best-effort last-separator split (unchanged legacy path).
    assert resolve_node_id("tests/t.py::test_x - assert 1 == 2", frozenset()) == \
        "tests/t.py::test_x"


def test_spaced_node_ids_survive_live_run(tmp_path: Path) -> None:
    # Live regression: a real pytest run through LocalRunner must produce
    # the full spaced node IDs in both collect() and run().failed — the
    # validate() path passes the collected set as known_ids.
    target = tmp_path / "target"
    (target / "tests").mkdir(parents=True)
    (target / "tests" / "__init__.py").write_text("")
    (target / "tests" / "test_sp.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('label', ['a - b', 'plain'])\n"
        "def test_lab(label):\n"
        "    assert label == 'plain'\n"
    )
    runner = LocalRunner()
    collected = runner.collect(target)
    spaced = "tests/test_sp.py::test_lab[a - b]"
    plain = "tests/test_sp.py::test_lab[plain]"
    assert spaced in collected
    assert plain in collected
    outcome = runner.run(target, known_ids=collected)
    assert spaced in outcome.failed
    assert plain not in outcome.failed


def test_validate_survives_target_addopts(tmp_path: Path) -> None:
    # Regression (arena#39 nightly): a target whose own pyproject sets
    # addopts="-v" used to break collection — the target's -v plus the
    # harness's -q cancel to default verbosity, and --collect-only then
    # prints the tree format instead of one node ID per line, so the
    # baseline refuses with "collected zero tests". The harness pins
    # -o addopts=, so its invocation is hermetic to target addopts.
    target = tmp_path / "target"
    (target / "src" / "pkg").mkdir(parents=True)
    (target / "src" / "pkg" / "__init__.py").write_text(
        'def val(label):\n    return "ok"\n', encoding="utf-8"
    )
    (target / "tests").mkdir(parents=True)
    (target / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (target / "tests" / "test_sp.py").write_text(
        "import pytest\n"
        "from pkg import val\n\n"
        "@pytest.mark.parametrize('label', ['a - b', 'plain'])\n"
        "def test_lab(label):\n"
        "    assert val(label) == 'ok', label\n",
        encoding="utf-8",
    )
    (target / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = \"-v --tb=short\"\n",
        encoding="utf-8",
    )
    work = tmp_path / "w"
    runner = LocalRunner()
    passed = baseline(target, runner, work)
    spaced = "tests/test_sp.py::test_lab[a - b]"
    assert spaced in passed
    patch = ("--- a/src/pkg/__init__.py\n"
             "+++ b/src/pkg/__init__.py\n"
             "@@ -1,2 +1,2 @@\n"
             '-def val(label):\n'
             '-    return "ok"\n'
             '+def val(label):\n'
             '+    return "ok" if label == "plain" else "bad"\n')
    candidate = BugCandidate(
        record=_record(strategy=BugStrategy.PROCEDURAL_AST,
                       target_file="src/pkg/__init__.py",
                       site_description="target-addopts regression"),
        patch=patch,
    )
    report = validate(candidate, target, runner, work, passed)
    assert report.is_valid_instance
    assert spaced in report.fail_to_pass


def test_validate_survives_ancestor_config_file(tmp_path: Path) -> None:
    # Regression (arena#39 nightly): a config file in an ancestor of the
    # work tree (e.g. a stray /tmp/pyproject.toml, where pytest's tmp_path
    # lives) used to make pytest resolve rootdir to that ancestor, so node
    # IDs embedded the work-tree path — and the clean-tree and patched-tree
    # IDs for the SAME test differed, so fail_to_pass matching reported
    # every mutant as surviving. The harness pins --rootdir to the work
    # tree, so node IDs stay work-tree-relative whatever the ancestors hold.
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "not-this-repo"\nversion = "0.0.1"\n',
        encoding="utf-8",
    )
    target = tmp_path / "target"
    (target / "src" / "pkg").mkdir(parents=True)
    (target / "src" / "pkg" / "__init__.py").write_text(
        'def val(label):\n    return "ok"\n', encoding="utf-8"
    )
    (target / "tests").mkdir(parents=True)
    (target / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (target / "tests" / "test_sp.py").write_text(
        "import pytest\n"
        "from pkg import val\n\n"
        "@pytest.mark.parametrize('label', ['a - b', 'plain'])\n"
        "def test_lab(label):\n"
        "    assert val(label) == 'ok', label\n",
        encoding="utf-8",
    )
    work = tmp_path / "w"
    runner = LocalRunner()
    passed = baseline(target, runner, work)
    spaced = "tests/test_sp.py::test_lab[a - b]"
    assert spaced in passed
    patch = ("--- a/src/pkg/__init__.py\n"
             "+++ b/src/pkg/__init__.py\n"
             "@@ -1,2 +1,2 @@\n"
             '-def val(label):\n'
             '-    return "ok"\n'
             '+def val(label):\n'
             '+    return "ok" if label == "plain" else "bad"\n')
    candidate = BugCandidate(
        record=_record(strategy=BugStrategy.PROCEDURAL_AST,
                       target_file="src/pkg/__init__.py",
                       site_description="ancestor-config regression"),
        patch=patch,
    )
    report = validate(candidate, target, runner, work, passed)
    assert report.is_valid_instance
    assert spaced in report.fail_to_pass
    assert "tests/test_sp.py::test_lab[plain]" in report.pass_to_pass


def test_validate_threads_known_ids_for_spaced_tests(tmp_path: Path) -> None:
    # Pins the validate() -> run(known_ids=...) wiring that the B1 fix
    # depends on: a candidate breaking a spaced-ID test must land the FULL
    # re-runnable node ID in fail_to_pass. Deleting known_ids=collected from
    # validate() makes this fail (the reason contains " - ", so the
    # best-effort fallback mis-splits the tail).
    target = tmp_path / "target"
    (target / "src" / "pkg").mkdir(parents=True)
    (target / "src" / "pkg" / "__init__.py").write_text(
        'def val(label):\n    return "ok"\n'
    )
    (target / "tests").mkdir(parents=True)
    (target / "tests" / "__init__.py").write_text("")
    (target / "tests" / "test_sp.py").write_text(
        "import pytest\n"
        "from pkg import val\n\n"
        "@pytest.mark.parametrize('label', ['a - b', 'plain'])\n"
        "def test_lab(label):\n"
        "    assert val(label) == 'ok', label\n"
    )
    work = tmp_path / "w"
    runner = LocalRunner()
    passed = baseline(target, runner, work)
    spaced = "tests/test_sp.py::test_lab[a - b]"
    assert spaced in passed
    patch = ("--- a/src/pkg/__init__.py\n"
             "+++ b/src/pkg/__init__.py\n"
             "@@ -1,2 +1,2 @@\n"
             '-def val(label):\n'
             '-    return "ok"\n'
             '+def val(label):\n'
             '+    return "ok" if label == "plain" else "bad"\n')
    candidate = BugCandidate(
        record=_record(strategy=BugStrategy.PROCEDURAL_AST,
                       target_file="src/pkg/__init__.py",
                       site_description="spaced validate threading"),
        patch=patch,
    )
    report = validate(candidate, target, runner, work, passed)
    assert report.is_valid_instance
    assert spaced in report.fail_to_pass
    assert "tests/test_sp.py::test_lab[plain]" in report.pass_to_pass


def test_parse_pytest_output_collection_error() -> None:
    out = "==== short test summary info ====\nERROR tests/test_a.py\n" \
          "!!!! Interrupted: 1 error during collection !!!!\n"
    o = parse_pytest_output(out, 2)
    assert o.collection_error


def test_validate_end_to_end_local_runner(tmp_path: Path) -> None:
    work = tmp_path / "w"
    runner = LocalRunner()
    passed = baseline(FIXTURE, runner, work)
    assert len(passed) == 16
    candidate = ProceduralBugGenerator(7).generate(FIXTURE, 3)[0]
    report = validate(candidate, FIXTURE, runner, work, passed)
    # Whatever this candidate does, the accounting must be consistent.
    assert set(report.fail_to_pass) <= set(passed)
    assert set(report.pass_to_pass) <= set(passed)
    assert not (set(report.fail_to_pass) & set(report.pass_to_pass))


def test_validate_collection_error_breaks_everything(tmp_path: Path, monkeypatch) -> None:
    work = tmp_path / "w"
    runner = LocalRunner()
    passed = baseline(FIXTURE, runner, work)

    def fake_run(self, work_tree: Path, known_ids: frozenset = frozenset()) -> HarnessOutcome:
        return HarnessOutcome(failed=frozenset(), collection_error=True,
                              returncode=2)

    monkeypatch.setattr(LocalRunner, "run", fake_run)
    monkeypatch.setattr(LocalRunner, "collect", lambda self, w: frozenset())
    candidate = ProceduralBugGenerator(7).generate(FIXTURE, 3)[0]
    report = validate(candidate, FIXTURE, runner, work, passed)
    assert report.collection_error
    assert set(report.fail_to_pass) == set(passed)


def test_baseline_requires_green_tree(tmp_path: Path, monkeypatch) -> None:
    work = tmp_path / "w"

    def fake_run(self, work_tree: Path, known_ids: frozenset = frozenset()) -> HarnessOutcome:
        return HarnessOutcome(failed=frozenset({"t1"}), collection_error=False,
                              returncode=1)

    monkeypatch.setattr(LocalRunner, "run", fake_run)
    with pytest.raises(BugsmithError):
        baseline(FIXTURE, LocalRunner(), work)


def test_never_touches_the_fixture_tree(tmp_path: Path) -> None:
    before = repo_digest(FIXTURE)
    work = tmp_path / "w"
    runner = LocalRunner()
    passed = baseline(FIXTURE, runner, work)
    for c in ProceduralBugGenerator(11).generate(FIXTURE, 5):
        validate(c, FIXTURE, runner, work, passed)
    assert repo_digest(FIXTURE) == before


# --- curation -------------------------------------------------------------


def _valid_report(strategy: BugStrategy, n_fail: int, tag: str) -> ValidationReport:
    base = tuple(f"t{i}" for i in range(8))
    return ValidationReport(
        candidate=BugCandidate(
            record=_record(strategy=strategy, seed=5,
                           site_description=tag, target_file=f"{tag}.py"),
            patch=f"--- a/{tag}.py\n+++ b/{tag}.py\n@@ -1 +1 @@\n-{tag}\n+{tag}x\n",
        ),
        baseline_passed=base,
        failed_after=tuple(f"t{i}" for i in range(n_fail)),
        passed_after=tuple(f"t{i}" for i in range(n_fail, 8)),
    )


def test_curation_is_deterministic_and_config_driven() -> None:
    reports = [_valid_report(BugStrategy.PROCEDURAL_AST, n, f"p{n}") for n in (1, 2, 6)]
    reports.append(_valid_report(BugStrategy.PR_MIRROR, 2, "m1"))
    config = CurationConfig(seed=9, fail_to_pass_min=1, fail_to_pass_max=4,
                            max_instances=10)
    first = select_subset(config, reports, "digest")
    second = select_subset(config, reports, "digest")
    assert [i.instance_id for i in first] == [i.instance_id for i in second]
    # n_fail=6 is outside the configured band: excluded, not hand-picked.
    assert all(len(i.fail_to_pass) <= 4 for i in first)
    assert len(first) == 3


def test_curation_respects_strategy_quota() -> None:
    reports = [_valid_report(BugStrategy.PROCEDURAL_AST, 1, f"p{i}") for i in range(5)]
    reports.append(_valid_report(BugStrategy.PR_MIRROR, 1, "m"))
    config = CurationConfig(seed=9, max_instances=10,
                            strategy_quota={"procedural_ast": 2})
    chosen = select_subset(config, reports, "digest")
    assert sum(1 for i in chosen if i.record.strategy is BugStrategy.PROCEDURAL_AST) == 2
    assert any(i.record.strategy is BugStrategy.PR_MIRROR for i in chosen)


def test_curation_dedupes_identical_patches() -> None:
    r = _valid_report(BugStrategy.PROCEDURAL_AST, 1, "dup")
    config = CurationConfig(seed=9, max_instances=10)
    chosen = select_subset(config, [r, r], "digest")
    assert len(chosen) == 1


def test_curation_rejects_invalid_config() -> None:
    with pytest.raises(BugsmithError):
        CurationConfig(seed=1, fail_to_pass_min=0)
    with pytest.raises(BugsmithError):
        CurationConfig(seed=1, fail_to_pass_min=3, fail_to_pass_max=2)
    with pytest.raises(BugsmithError):
        CurationConfig(seed=1, strategy_quota={"nope": 1})
    with pytest.raises(BugsmithError):
        CurationConfig(seed=1, strategy_quota={"procedural_ast": 0})
    with pytest.raises(BugsmithError):
        CurationConfig(seed=1, strategy_quota={"procedural_ast": -2})


def test_to_instance_refuses_unvalidated() -> None:
    with pytest.raises(BugsmithError):
        to_instance(_report(), "digest")


def test_instance_roundtrip() -> None:
    from bugsmith.types import BenchmarkInstance

    r = _valid_report(BugStrategy.PR_MIRROR, 2, "rt")
    inst = to_instance(r, "digest123")
    assert inst.instance_id.startswith("pr_mirror__")
    back = BenchmarkInstance.from_dict(inst.to_dict())
    assert back == inst


# --- docker image contract (no daemon needed) ------------------------------


def test_dockerfile_is_a_sealed_environment_not_a_repo_snapshot() -> None:
    # The image must not bake in a repo copy: validation bind-mounts the
    # patched tree, and a stale in-image copy could shadow it.
    text = dockerfile_text()
    assert '"pytest>=8,<10"' in text
    assert "FROM python:3.12-slim" in text
    assert "COPY repo" not in text
    assert "/target" not in text


def test_dockerfile_installs_declared_dependencies() -> None:
    text = dockerfile_text(has_dependencies=True)
    assert "COPY deps/requirements.txt" in text
    assert "pip install --no-cache-dir -r /deps/requirements.txt" in text
    plain = dockerfile_text(has_dependencies=False)
    assert "requirements.txt" not in plain


def test_build_context_materializes_manifests_not_source(tmp_path: Path) -> None:
    ctx = tmp_path / "ctx"
    build_context(FIXTURE, ctx)
    assert (ctx / "Dockerfile").is_file()
    assert (ctx / "deps" / "requirements.txt").is_file()
    # No repo source in the build context.
    assert not (ctx / "repo").exists()


def test_collect_requirements_from_pyproject_and_txt(tmp_path: Path) -> None:
    from bugsmith.images import collect_requirements

    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["pyyaml>=6.0", "rich"]\n',
        encoding="utf-8",
    )
    (repo / "requirements-test.txt").write_text(
        "# comment\npytest>=8.0\npyyaml>=6.0\n", encoding="utf-8"
    )
    reqs = collect_requirements(repo)
    assert reqs == ["pyyaml>=6.0", "rich", "pytest>=8.0"]


def test_collect_requirements_empty_for_fixture() -> None:
    from bugsmith.images import collect_requirements

    assert collect_requirements(FIXTURE) == []


def test_procedural_skips_docs_and_examples(tmp_path: Path) -> None:
    import shutil as _shutil

    repo = tmp_path / "r"
    _shutil.copytree(FIXTURE, repo, ignore=_shutil.ignore_patterns("__pycache__"))
    (repo / "docs").mkdir()
    (repo / "docs" / "conf.py").write_text("x = 1 + 2\n", encoding="utf-8")
    gen = ProceduralBugGenerator(3)
    files = gen._source_files(repo)
    assert all("docs" not in f.parts for f in files)


def test_repo_digest_changes_with_content(tmp_path: Path) -> None:
    d1 = repo_digest(FIXTURE)
    other = tmp_path / "o"
    shutil.copytree(FIXTURE, other, ignore=shutil.ignore_patterns("__pycache__"))
    (other / "src" / "mcalc" / "arithmetic.py").write_text("# changed\n")
    assert repo_digest(other) != d1
    assert len(d1) == 16


def test_fixture_baseline_is_green() -> None:
    # Guard: the fixture target must stay a green, honest substrate.
    import subprocess

    proc = subprocess.run(
        ["python3", "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"],
        cwd=str(FIXTURE), check=False,
        env={**os.environ, "PYTHONPATH": str(FIXTURE / "src")},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout[-500:]
    assert "16 passed" in proc.stdout


def test_generated_patch_bytes_are_stable_across_processes(tmp_path: Path) -> None:
    # Reproducibility must survive process boundaries (criterion 3), not
    # just in-memory RNG state.
    import subprocess

    code = ("from bugsmith.buggen import ProceduralBugGenerator;"
            "cs=ProceduralBugGenerator(42).generate('src/bugsmith/fixtures/target',5);"
            "print('\\x00'.join(c.patch_sha for c in cs))")
    env = {"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"}
    p1 = subprocess.run(["python3", "-c", code], cwd=str(REPO_ROOT),
                        capture_output=True, text=True, env=env, timeout=60,
                        check=False)
    p2 = subprocess.run(["python3", "-c", code], cwd=str(REPO_ROOT),
                        capture_output=True, text=True, env=env, timeout=60,
                        check=False)
    assert p1.returncode == 0 and p2.returncode == 0
    assert p1.stdout == p2.stdout
    assert len(p1.stdout.strip().split("\x00")) == 5


def test_patch_sha_is_content_addressed() -> None:
    # Sanity: content-addressing uses the patch bytes, not file paths.
    a = BugCandidate(record=_record(), patch="--- a/1.py\n+++ b/1.py\n")
    b = BugCandidate(record=_record(target_file="other.py"), patch="--- a/1.py\n+++ b/1.py\n")
    assert a.patch_sha == b.patch_sha
    assert hashlib.sha256(b"--- a/1.py\n+++ b/1.py\n").hexdigest()[:16] == a.patch_sha


def test_docker_run_and_collect_share_pytest_construction(tmp_path, monkeypatch) -> None:
    # Regression: the CI failure where DockerRunner.run set PYTHONPATH but
    # _collect_node_ids did not, so baseline collected zero tests. Both
    # paths must go through the single _container_pytest_script site.
    import subprocess as sp

    import bugsmith.harness as h

    seen: list[list[str]] = []

    class FakeProc:
        stdout = ""
        returncode = 0

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return FakeProc()

    monkeypatch.setattr(sp, "run", fake_run)
    image = h.TargetImage(tag="bugsmith-target:fake", repo_digest="fake")
    runner = h.DockerRunner(image)
    runner.run(tmp_path)
    runner.collect(tmp_path)
    assert len(seen) == 2
    for cmd in seen:
        script = cmd[-1]
        assert "PYTHONPATH=$(" in script, f"PYTHONPATH missing from: {script[:80]}"
        assert "--user" in cmd, "container must run as the host user (file ownership)"
        assert "PYTHONDONTWRITEBYTECODE=1" in cmd
    # The two scripts differ only in the pytest args, not the setup.
    assert seen[0][-1].split("python3 -m pytest")[0] == seen[1][-1].split("python3 -m pytest")[0]


def test_off_by_one_range_mutates_stop_never_step() -> None:
    # Regression (implementation review): range(a, b, step) must mutate the
    # stop bound; touching the step is not an off-by-one fault.
    from bugsmith.buggen import _collect_sites, _mutate_source

    src = "def f(n):\n    return list(range(0, n, 2))\n"
    sites = [s for s in _collect_sites(ast.parse(src)) if s.kind is ast.Call]
    assert len(sites) == 1
    mutated = _mutate_source(src, sites[0])
    assert "range(0, n - 1, 2)" in mutated, mutated
    assert ", 1)" not in mutated  # the step is untouched


def test_from_dict_verifies_instance_id() -> None:
    from bugsmith.types import BenchmarkInstance

    r = _valid_report(BugStrategy.PR_MIRROR, 2, "rt")
    inst = to_instance(r, "digest123")
    back = BenchmarkInstance.from_dict(inst.to_dict())
    assert back == inst
    tampered = inst.to_dict()
    tampered["instance_id"] = "forged__id"
    with pytest.raises(BugsmithError):
        BenchmarkInstance.from_dict(tampered)


def test_instance_id_binds_repo_digest() -> None:
    r = _valid_report(BugStrategy.PROCEDURAL_AST, 1, "d1")
    a = to_instance(r, "digest-aaa")
    b = to_instance(r, "digest-bbb")
    assert a.instance_id != b.instance_id
    assert "digest-a"[:8] in a.instance_id


def test_validated_record_uses_canonical_instance_id() -> None:
    # The CLI's all_validated.jsonl writer must use the single canonical
    # formula: the record joins with the curated instance JSONs on
    # instance_id, so a second formula here silently breaks the audit trail.
    from agenteval_bench.cli import _validated_record

    r = _valid_report(BugStrategy.PROCEDURAL_AST, 1, "d1")
    digest = "digest-aaa"
    rec = _validated_record(r, digest)
    assert rec["instance_id"] == instance_id(r, digest)
    assert rec["instance_id"] == to_instance(r, digest).instance_id
    assert rec["patch_sha"] == r.candidate.patch_sha
    assert rec["strategy"] == r.candidate.record.strategy.value


def test_baseline_rejects_abnormal_exit_code(tmp_path: Path, monkeypatch) -> None:
    work = tmp_path / "w"

    def fake_run(self, work_tree: Path, known_ids: frozenset = frozenset()) -> HarnessOutcome:
        return HarnessOutcome(failed=frozenset(), collection_error=False,
                              returncode=139)

    monkeypatch.setattr(LocalRunner, "run", fake_run)
    with pytest.raises(BugsmithError):
        baseline(FIXTURE, LocalRunner(), work)


def test_local_runner_prepends_pythonpath(tmp_path: Path, monkeypatch) -> None:
    import os
    import subprocess as sp

    import bugsmith.harness as h

    captured: dict[str, str] = {}

    class FakeProc:
        stdout = ""
        returncode = 0

    def fake_run(cmd, **kw):
        captured.update(kw["env"])
        return FakeProc()

    monkeypatch.setattr(sp, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/existing/entry")
    h.LocalRunner().run(tmp_path)
    assert captured["PYTHONPATH"].endswith(f"{os.pathsep}/existing/entry")
    assert str(tmp_path / "src") in captured["PYTHONPATH"] or str(tmp_path) in captured["PYTHONPATH"]


def test_patch_applier_honors_no_trailing_newline(tmp_path: Path) -> None:
    from bugsmith.patch import apply_hunks

    original = ["line1\n", "line2"]  # no trailing newline
    hunks = ["@@ -1,2 +1,2 @@", " line1", "-line2", "+line2x", "\\ No newline at end of file"]
    out = apply_hunks(original, hunks)
    assert "".join(out) == "line1\nline2x"
    assert not "".join(out).endswith("\n")
