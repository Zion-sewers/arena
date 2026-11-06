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
"""

from __future__ import annotations

import ast
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


def _scan_mock_on_boundary(tree: Path) -> list[Finding]:
    """``unittest.mock`` use inside tests: the mock patches production
    code, so the test can pass while the boundary is broken. Mirrors the
    live enforcer in tests/test_no_mocks_on_boundaries.py (same detection
    shape, independent implementation — the enforcer is a test, not a
    library the flywheel may import)."""
    findings: list[Finding] = []
    for path in _iter_py_files(tree, tests_only=True):
        parsed = _parse_or_skip(path)
        if parsed is None:
            continue
        rel = str(path.relative_to(tree.resolve()))
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _MOCK_MODULES or alias.name.startswith("mock."):
                        findings.append(
                            Finding(file=rel, line=node.lineno, message=f"imports {alias.name}")
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in _MOCK_MODULES or module.startswith("mock."):
                    findings.append(
                        Finding(file=rel, line=node.lineno, message=f"imports from {module}")
                    )
            elif isinstance(node, ast.Attribute):
                value = node.value
                if isinstance(value, ast.Name) and value.id == "mock":
                    findings.append(
                        Finding(file=rel, line=node.lineno, message=f"uses mock.{node.attr}")
                    )
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
    parsed = ast.parse(source)
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
    func_name = {"bare-except-median": "median", "bare-except-mean": "mean"}[case]
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
# enforcement it cannot see.


def _ruff_selects_bare_except(target_root: Path) -> bool:
    pyproject = target_root / "pyproject.toml"
    try:
        doc = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    lint = doc.get("tool", {}).get("ruff", {}).get("lint", {})
    select = lint.get("select", [])
    return isinstance(select, list) and any(
        isinstance(item, str) and item in ("E722", "BLE", "BLE001") for item in select
    )


def already_enforced(kind: RuleKind, target_root: Path) -> str | None:
    """The live check that already enforces this kind in the target repo,
    or None when the kind is not enforced there."""
    if kind == "no-mock-on-boundary":
        enforcer = target_root / "tests" / "test_no_mocks_on_boundaries.py"
        if enforcer.is_file():
            return "enforced by tests/test_no_mocks_on_boundaries.py (CI runs it)"
        return None
    if _ruff_selects_bare_except(target_root):
        return "enforced by ruff select E722/BLE in pyproject.toml (CI runs ruff check)"
    return None


# --- PR wiring ------------------------------------------------------------
#
# The file contents a PR payload carries to make the rule blocking. Each
# wiring targets a live consumer: the ruff CI step, or a new test the CI
# suite runs. A wiring that no consumer runs would be shelfware, and the
# PR payload refuses to carry it.


def _wire_bare_except(target_root: Path) -> dict[str, str]:
    pyproject = target_root / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError as e:
        raise FlywheelError(f"cannot read target pyproject.toml for wiring: {e}") from e
    if "[tool.ruff.lint]" in text:
        # A lint section with an existing select is a shape this wiring
        # does not know how to extend safely: refuse rather than corrupt.
        raise FlywheelError(
            "target pyproject.toml already has [tool.ruff.lint]: wiring refuses "
            "to edit an existing select (extend by hand, then re-run the loop)"
        )
    new_text = text.rstrip("\n") + '\n\n[tool.ruff.lint]\nselect = ["E", "F", "E722"]\n'
    return {"pyproject.toml": new_text}


_MOCK_ENFORCER_TEST = '''"""Mock-rejection enforcement (flywheel-proposed rule no-mock-on-boundary).

Mocks on real boundaries are rejected: the repo's test style is recorded
replay outputs and hand-written stubs, never unittest.mock. This test is
the live consumer of the rule: CI runs it, so a mock import in tests/
fails the suite instead of living as prose.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def test_no_mock_imports_in_tests() -> None:
    violations: list[str] = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("mock", "unittest.mock") or alias.name.startswith("mock."):
                        violations.append(f"{path.name}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in ("mock", "unittest.mock") or module.startswith("mock."):
                    violations.append(f"{path.name}:{node.lineno} imports from {module}")
    assert not violations, "mock usage in tests/: " + "; ".join(violations)
'''


def _wire_mock_on_boundary(target_root: Path) -> dict[str, str]:
    del target_root  # the enforcer test is target-independent
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
    ),
    "no-mock-on-boundary": RuleSpec(
        kind="no-mock-on-boundary",
        description="unittest.mock in tests patches production code; use stubs or recorded data.",
        signal_keywords=("mock", "unittest.mock", "patch("),
        requires_behavioral_break=False,
        injection_sites=("mock-in-test",),
    ),
}
