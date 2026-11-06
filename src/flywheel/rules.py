"""Rule-kind registry: predicates, injectors, and PR wiring (arena#40).

The registry is CLOSED: the meta-agent proposes instances of known rule
kinds, never arbitrary generated code. Free-form rule synthesis needs a
model transport this repo does not have; inventing one here would be
theater, so the registry documents it as a follow-up instead.

Each kind carries:
- ``predicate``: a pure function over a repo tree (AST scans, no I/O
  beyond reading the tree it is given).
- ``injector``: a deterministic fault of the kind's class, applied to a
  throwaway copy of the validation corpus. The injector returns the new
  file text; the caller diffs it into a bugsmith BugCandidate so the
  injected fault is validated as a real bug (fail_to_pass evidence).
- ``wiring``: the file contents a PR payload carries to make the rule
  blocking in the target repo, against a live consumer (ruff CI step, or
  the repo's own mock-rejection test).
- ``already_enforced``: the live check that, when present in the target,
  makes proposing the kind redundant — Reflect skips it with the reason.

Alignment rule (enforced by tests): a kind's flywheel predicate and the
test its wiring installs MUST match the live enforcer they mirror, same
detection shape and same file set. A scan that passes what the enforcer
fails is a promoted rule that breaks the target's CI on merge; a scan
that flags what the enforcer passes is a veto on a rule that would have
been fine. Both directions are pinned by probe tests against the live
enforcer in tests/test_no_mocks_on_boundaries.py.
"""

from __future__ import annotations

import ast
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from flywheel.types import Finding, FlywheelError, RuleKind

# Directories a rule scan never descends into. The scan reads the target
# tree; it must not trip over its own work products or VCS metadata.
_SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".venv", "venv", ".hypothesis", ".ruff_cache", ".tox", ".mypy_cache"}
)

# A mock import inside tests/ that patches production code is the
# boundary violation; production code importing mock is a different smell
# and out of this kind's scope.
_MOCK_MODULES = frozenset({"mock", "unittest.mock"})


@dataclass(frozen=True)
class RuleSpec:
    """One rule kind: what it forbids, how it is detected, what it costs."""

    kind: RuleKind
    description: str
    signal_keywords: tuple[str, ...]  # Reflect signals in fix-commit messages
    requires_behavioral_break: bool  # injected faults must break a test
    injection_sites: tuple[str, ...]  # deterministic injection case names
    requires_enforcer_check: bool = False  # the installed enforcer test
    # must fail on the injected tree (the evidence leg for stylistic
    # kinds whose faults break no test)


def _iter_py_files(tree: Path, *, tests_only: bool = False) -> list[Path]:
    tree = tree.resolve()
    out: list[Path] = []
    for path in sorted(tree.rglob("*.py")):
        rel = path.relative_to(tree)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if tests_only and "tests" not in rel.parts and "test" not in rel.parts:
            continue
        out.append(path)
    return out


def _iter_mock_files(tree: Path) -> list[Path]:
    """Top-level ``*.py`` files directly under a ``tests`` directory —
    exactly the file set the live enforcer scans
    (``TESTS_DIR.glob("*.py")``), so the scan and the enforcer cannot
    disagree about which files are in scope."""
    tree = tree.resolve()
    out: list[Path] = []
    for tests_dir in sorted(tree.rglob("tests")):
        if not tests_dir.is_dir():
            continue
        rel = tests_dir.relative_to(tree)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        out.extend(sorted(p for p in tests_dir.glob("*.py") if p.is_file()))
    return out


def _parse_or_skip(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None


def _scan_bare_except(tree: Path) -> list[Finding]:
    """A bare ``except:`` swallows BaseException (KeyboardInterrupt,
    SystemExit) along with the error it meant to catch."""
    findings: list[Finding] = []
    for path in _iter_py_files(tree):
        parsed = _parse_or_skip(path)
        if parsed is None:
            continue
        for node in ast.walk(parsed):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append(
                    Finding(
                        file=str(path.relative_to(tree.resolve())),
                        line=node.lineno,
                        message="bare except: swallows BaseException; name the exception",
                    )
                )
    return findings


def _mock_violations(source: str) -> list[str]:
    """Violation descriptions for mock machinery in one module's source.

    The same detection shape as the live enforcer
    (``tests/test_no_mocks_on_boundaries.py::find_mock_usage``): any
    ``unittest.mock`` / ``mock`` import, any ``mock.*`` / ``unittest.mock.*``
    attribute use, or the ``mocker`` fixture. The two implementations are
    kept in sync by probe tests; if they drift, a test fails.
    """
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("mock", "unittest.mock") or alias.name.startswith(
                    ("mock.", "unittest.mock.")
                ):
                    violations.append(f"imports mock module ({alias.name})")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(("unittest.mock", "mock.")) or module == "mock":
                violations.append(f"imports from {node.module}")
            elif node.module == "unittest":
                for alias in node.names:
                    if alias.name == "mock":
                        violations.append("imports mock from unittest")
        elif isinstance(node, ast.Attribute):
            value = node.value
            if isinstance(value, ast.Name) and value.id == "mock":
                violations.append(f"uses mock.{node.attr}")
            elif (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "unittest"
                and value.attr == "mock"
            ):
                violations.append(f"uses unittest.mock.{node.attr}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                if arg.arg == "mocker":
                    violations.append(f"uses the mocker fixture in {node.name}()")
    return violations


def _scan_mock_on_boundary(tree: Path) -> list[Finding]:
    """``unittest.mock`` use inside tests: the mock patches production
    code, so the test can pass while the boundary is broken. Mirrors the
    live enforcer in tests/test_no_mocks_on_boundaries.py (same detection
    shape AND same file set — top-level tests/*.py only). The enforcer is
    a test, not a library the flywheel may import, so the shape is
    duplicated here and pinned equal by probe tests."""
    findings: list[Finding] = []
    for path in _iter_mock_files(tree):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            violations = _mock_violations(source)
        except SyntaxError:
            continue
        rel = str(path.relative_to(tree.resolve()))
        for violation in violations:
            findings.append(Finding(file=rel, line=0, message=violation))
    # Line numbers are not part of the mock kind's evidence (the live
    # enforcer reports per-file violations); keep file-level findings so
    # a scan finding and an enforcer failure mean the same thing.
    return findings


_SCAN: dict[RuleKind, Callable[[Path], list[Finding]]] = {
    "no-bare-except": _scan_bare_except,
    "no-mock-on-boundary": _scan_mock_on_boundary,
}


def scan(kind: RuleKind, tree: Path) -> list[Finding]:
    """Run a rule kind's predicate over a repo tree. Pure over the tree it
    is given; it reads, never writes."""
    return _SCAN[kind](tree)


# --- injectors: kind-specific faults, applied to a throwaway copy -------
#
# Each injector takes the corpus copy root and returns (repo-relative
# path, new file text). It refuses (raises) when the expected site is
# absent instead of silently producing a no-op injection: a no-op would
# make the dry-run vacuous, and vacuous dry-runs must fail loudly.


def _wrap_function_body_in_bare_except(source: str, func_name: str) -> str:
    """Wrap one function's body in ``try: ... except: return 0.0`` via AST
    rewrite. The result is a real behavioral fault: the function swallows
    the ValueError its own tests assert on (e.g. median([]))."""
    try:
        parsed = ast.parse(source)
    except SyntaxError as e:
        raise FlywheelError(f"injection site is not parseable Python: {e}") from e
    target: ast.FunctionDef | None = None
    for node in ast.walk(parsed):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target = node
            break
    if target is None:
        raise FlywheelError(f"injection site {func_name!r} not found in target file")
    if not target.body:
        raise FlywheelError(f"injection site {func_name!r} has an empty body")
    swallowed = ast.Try(
        body=target.body,
        handlers=[
            ast.ExceptHandler(
                type=None,
                name=None,
                body=[ast.Return(value=ast.Constant(value=0.0))],
            )
        ],
        orelse=[],
        finalbody=[],
    )
    target.body = [swallowed]
    ast.fix_missing_locations(parsed)
    return ast.unparse(parsed) + "\n"


def _inject_bare_except(corpus_copy: Path, case: str) -> tuple[str, str]:
    rel = "src/mcalc/stats.py"
    path = corpus_copy / rel
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as e:
        raise FlywheelError(f"injection site file {rel} unreadable: {e}") from e
    func_by_case = {"bare-except-median": "median", "bare-except-mean": "mean"}
    func_name = func_by_case.get(case)
    if func_name is None:
        raise FlywheelError(f"unknown injection case {case!r} for no-bare-except")
    return rel, _wrap_function_body_in_bare_except(source, func_name)


_MOCK_INJECTION = '''

from unittest.mock import patch


def test_mocked_mean_returns_canned_value() -> None:
    with patch("mcalc.stats.mean", return_value=42.0):
        assert mean([1.0, 2.0]) == 42.0
'''


def _inject_mock_on_boundary(corpus_copy: Path, case: str) -> tuple[str, str]:
    rel = "tests/test_mcalc.py"
    path = corpus_copy / rel
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as e:
        raise FlywheelError(f"injection site file {rel} unreadable: {e}") from e
    if case != "mock-in-test":
        raise FlywheelError(f"unknown injection case {case!r} for no-mock-on-boundary")
    if "unittest.mock" in source:
        raise FlywheelError("mock injection site already contains mock usage: refusing")
    return rel, source.rstrip("\n") + "\n" + _MOCK_INJECTION


_INJECT: dict[RuleKind, Callable[[Path, str], tuple[str, str]]] = {
    "no-bare-except": _inject_bare_except,
    "no-mock-on-boundary": _inject_mock_on_boundary,
}


def inject(kind: RuleKind, corpus_copy: Path, case: str) -> tuple[str, str]:
    """Inject one fault of the kind's class into a throwaway corpus copy.
    Returns (repo-relative path, new file text)."""
    return _INJECT[kind](corpus_copy, case)


# --- already-enforced probes --------------------------------------------
#
# A kind is already enforced when the target repo has a live check for it.
# The probe reads the target's own config/tests; it never claims
# enforcement it cannot see. A config it cannot parse is "not enforced",
# never an exception: the probe must not crash Reflect.


def _as_mapping(value: object) -> dict[str, object] | None:
    return value if isinstance(value, dict) else None


def _covers_e722(item: object) -> bool:
    """One ruff select item enables E722: the code itself, a prefix that
    contains it (``E``, ``E7``, ``E72``), or ``ALL``. ``BLE`` / ``BLE001``
    do NOT flag a bare ``except:`` (verified against ruff 0.17.4: BLE001
    flags ``except Exception``, never a bare handler), so they never
    count as enforcing this kind."""
    if not isinstance(item, str) or not item:
        return False
    if item == "ALL":
        return True
    return "E722".startswith(item)


def _lint_selects_bare_except(lint: dict[str, object] | None) -> bool:
    if lint is None:
        return False
    for key in ("select", "extend-select"):
        items = lint.get(key)
        if isinstance(items, list) and any(_covers_e722(item) for item in items):
            return True
    return False


def _ruff_selects_bare_except(target_root: Path) -> bool:
    for config_name in ("pyproject.toml", "ruff.toml"):
        config = target_root / config_name
        try:
            doc = tomllib.loads(config.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        if config_name == "pyproject.toml":
            tool = _as_mapping(doc.get("tool"))
            ruff = _as_mapping(tool.get("ruff")) if tool is not None else None
            lint = _as_mapping(ruff.get("lint")) if ruff is not None else None
        else:
            lint = _as_mapping(doc.get("lint"))
        if _lint_selects_bare_except(lint):
            return True
    return False


def already_enforced(kind: RuleKind, target_root: Path) -> str | None:
    """The live check that already enforces this kind in the target repo,
    or None when the kind is not enforced there."""
    if kind == "no-mock-on-boundary":
        enforcer = target_root / "tests" / "test_no_mocks_on_boundaries.py"
        if enforcer.is_file():
            try:
                text = enforcer.read_text(encoding="utf-8")
            except OSError:
                return None
            # Existence alone is not enforcement: the file must carry the
            # detector the loop mirrors, or a foreign test of the same
            # name would silently stand in for it.
            if "def find_mock_usage" in text:
                return "enforced by tests/test_no_mocks_on_boundaries.py (CI runs it)"
        return None
    if _ruff_selects_bare_except(target_root):
        return "enforced by ruff select E722 (or a prefix covering it) in pyproject.toml/ruff.toml (CI runs ruff check)"
    return None


# --- PR wiring ------------------------------------------------------------
#
# The file contents a PR payload carries to make the rule blocking. Each
# wiring targets a live consumer: the ruff CI step, or a new test the CI
# suite runs. A wiring that no consumer runs would be shelfware, and the
# PR payload refuses to carry it.
#
# The bare-except wiring selects E722 ONLY. Selecting a group (``E``,
# ``F``) would turn on checks the dry-run never validated (e.g. E501 on
# pre-existing long lines) and break the target's CI on merge — exactly
# the failure the dry-run gate exists to prevent.


def _insert_e722_into_select(text: str) -> str:
    """Insert ``"E722"`` into the ``select = [...]`` list of a
    ``[tool.ruff.lint]`` section. Returns the edited text, or raises
    FlywheelError when the select is not a plain list literal this
    wiring knows how to extend safely."""
    section = re.search(r"(?m)^\[tool\.ruff\.lint\]\s*$", text)
    if section is None:
        raise FlywheelError("target pyproject.toml has no [tool.ruff.lint] section")
    rest = text[section.end():]
    next_header = re.search(r"(?m)^\[", rest)
    section_text = rest[: next_header.start()] if next_header else rest
    match = re.search(r"select\s*=\s*\[([^\]]*)\]", section_text, re.DOTALL)
    if match is None:
        raise FlywheelError(
            "target [tool.ruff.lint] select is not a plain list literal: "
            "wiring refuses to edit it (extend by hand, then re-run the loop)"
        )
    inner = match.group(1)
    stripped = inner.strip()
    insertion = '"E722"' if not stripped else stripped.rstrip(",").rstrip() + ', "E722"'
    new_section = (
        section_text[: match.start(1)] + " " + insertion + " " + section_text[match.end(1):]
    )
    return text[: section.end()] + new_section + text[section.end() + len(section_text):]


def _wire_bare_except(target_root: Path) -> dict[str, str]:
    pyproject = target_root / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError as e:
        raise FlywheelError(f"cannot read target pyproject.toml for wiring: {e}") from e
    if "[tool.ruff.lint]" not in text:
        new_text = text.rstrip("\n") + '\n\n[tool.ruff.lint]\nselect = ["E722"]\n'
        return {"pyproject.toml": new_text}
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise FlywheelError(f"cannot parse target pyproject.toml for wiring: {e}") from e
    tool = _as_mapping(doc.get("tool")) if isinstance(doc, dict) else None
    ruff = _as_mapping(tool.get("ruff")) if tool is not None else None
    lint = _as_mapping(ruff.get("lint")) if ruff is not None else None
    if lint is None:
        raise FlywheelError(
            "target [tool.ruff.lint] is not a table: wiring refuses to edit it "
            "(extend by hand, then re-run the loop)"
        )
    if _lint_selects_bare_except(lint):
        # Already covered: the wiring is a no-op, not a second edit.
        return {"pyproject.toml": text}
    select = lint.get("select")
    if select is None and lint.get("extend-select") is None:
        # A lint section with no select at all: add one right after the
        # section header (keys may appear in any order within a section).
        new_text = text.replace(
            "[tool.ruff.lint]", '[tool.ruff.lint]\nselect = ["E722"]', 1
        )
        return {"pyproject.toml": new_text}
    if not isinstance(select, list):
        raise FlywheelError(
            "target [tool.ruff.lint] select is not a list: wiring refuses "
            "to edit it (extend by hand, then re-run the loop)"
        )
    return {"pyproject.toml": _insert_e722_into_select(text)}


_MOCK_ENFORCER_TEST = '''"""Mock-rejection enforcement (flywheel-proposed rule no-mock-on-boundary).

Mocks on real boundaries are rejected: the repo's test style is recorded
replay outputs and hand-written stubs, never unittest.mock. This test is
the live consumer of the rule: CI runs it, so a mock import in tests/
fails the suite instead of living as prose.

This file is the same detector as the flywheel's mock predicate
(``flywheel.rules._mock_violations``): same detection shape, same file
set (top-level tests/*.py). If the two drift, the flywheel's probe tests
fail. The positive/negative controls below prove the detector fires on
mock machinery and stays quiet on hand-written stubs.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def find_mock_usage(source: str) -> list[str]:
    """Violation descriptions for mock machinery in one module's source."""
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("mock", "unittest.mock") or alias.name.startswith(
                    ("mock.", "unittest.mock.")
                ):
                    violations.append(f"imports mock module ({alias.name})")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(("unittest.mock", "mock.")) or module == "mock":
                violations.append(f"imports from {node.module}")
            elif node.module == "unittest":
                for alias in node.names:
                    if alias.name == "mock":
                        violations.append("imports mock from unittest")
        elif isinstance(node, ast.Attribute):
            value = node.value
            if isinstance(value, ast.Name) and value.id == "mock":
                violations.append(f"uses mock.{node.attr}")
            elif (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "unittest"
                and value.attr == "mock"
            ):
                violations.append(f"uses unittest.mock.{node.attr}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                if arg.arg == "mocker":
                    violations.append(f"uses the mocker fixture in {node.name}()")
    return violations


def test_no_mocks_in_test_suite() -> None:
    offenders: dict[str, list[str]] = {}
    # Every .py file under tests/, not just test_*.py: conftest.py and
    # helper modules are where fixtures (and mocks) normally live.
    for path in sorted(TESTS_DIR.glob("*.py")):
        violations = find_mock_usage(path.read_text(encoding="utf-8"))
        if violations:
            offenders[path.name] = violations
    assert not offenders, (
        "mock machinery is rejected on real boundaries (flywheel rule "
        "no-mock-on-boundary) — use a hand-written stub or recorded data "
        f"instead: {offenders}"
    )


# --- controls: the detector must fire on mocks and stay quiet on stubs ---


def test_detector_flags_mock_imports() -> None:
    assert find_mock_usage("from unittest.mock import patch\\n") != []
    assert find_mock_usage("import mock\\n") != []
    assert find_mock_usage("from mock import patch\\n") != []
    assert find_mock_usage("from mock import MagicMock\\n") != []
    assert find_mock_usage("from unittest import mock\\n") != []
    assert find_mock_usage("import unittest.mock as m\\n") != []


def test_detector_flags_mock_attribute_use_and_mocker_fixture() -> None:
    assert find_mock_usage("x = mock.patch('os.path.join')\\n") != []
    assert find_mock_usage("x = unittest.mock.MagicMock()\\n") != []
    assert find_mock_usage("def test_something(mocker):\\n    pass\\n") != []
    assert find_mock_usage("async def test_something(mocker):\\n    pass\\n") != []


def test_detector_allows_hand_written_stubs() -> None:
    stub = "def replay_fn(input_text: str) -> str:\\n    return recorded.get(input_text, '')\\n"
    assert find_mock_usage(stub) == []
    # A local named `patch` (a diff string) is not mock machinery: only
    # the mock module / fixture trip the detector.
    assert find_mock_usage("patch = open('fix.patch').read()\\n") == []
'''


def _wire_mock_on_boundary(target_root: Path) -> dict[str, str]:
    path = target_root / "tests" / "test_no_mocks_on_boundaries.py"
    if path.is_file():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as e:
            raise FlywheelError(f"cannot read target enforcer test for wiring: {e}") from e
        if existing == _MOCK_ENFORCER_TEST:
            return {"tests/test_no_mocks_on_boundaries.py": existing}
        # A foreign test of the same name: refuse to clobber it.
        raise FlywheelError(
            "target already has tests/test_no_mocks_on_boundaries.py with "
            "different content: wiring refuses to overwrite it (reconcile "
            "by hand, then re-run the loop)"
        )
    return {"tests/test_no_mocks_on_boundaries.py": _MOCK_ENFORCER_TEST}


_WIRE: dict[RuleKind, Callable[[Path], dict[str, str]]] = {
    "no-bare-except": _wire_bare_except,
    "no-mock-on-boundary": _wire_mock_on_boundary,
}


def wiring(kind: RuleKind, target_root: Path) -> dict[str, str]:
    """File contents a PR payload carries to make this kind blocking."""
    return _WIRE[kind](target_root)


RULES: dict[RuleKind, RuleSpec] = {
    "no-bare-except": RuleSpec(
        kind="no-bare-except",
        description="A bare except: swallows BaseException; name the exception (ruff E722).",
        signal_keywords=("bare except", "blind except", "e722", "ble001", "swallowed exception"),
        requires_behavioral_break=True,
        injection_sites=("bare-except-median", "bare-except-mean"),
        requires_enforcer_check=False,
    ),
    "no-mock-on-boundary": RuleSpec(
        kind="no-mock-on-boundary",
        description="unittest.mock in tests patches production code; use stubs or recorded data.",
        signal_keywords=("mock", "unittest.mock", "patch("),
        requires_behavioral_break=False,
        injection_sites=("mock-in-test",),
        requires_enforcer_check=True,
    ),
}
