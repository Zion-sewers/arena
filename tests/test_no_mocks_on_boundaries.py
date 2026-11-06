"""Mock-rejection enforcement (arena#39): mocks on real boundaries are rejected.

The tiered pipeline's acceptance criterion: mocks on business logic, the
database, files, or external services are rejected. This repo's test style
is recorded replay outputs and hand-written stub functions (e.g. the
``replay_fn`` in ``cli.py``), never ``unittest.mock`` — so the enforcement
is a blanket ban on the mock machinery in ``tests/``: any ``unittest.mock``
/ ``mock`` import, any ``mock.*`` attribute use, or the ``mocker`` fixture.

The detector is proven, not just vacuously green: the positive/negative
controls below feed it synthetic modules that must and must not trip it.
If a future test genuinely needs a double, the honest route is a
hand-written stub (like ``replay_fn``) or recorded data — not editing this
test to allow an import.
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
        "mock machinery is rejected on real boundaries (arena#39) — "
        "use a hand-written stub or recorded data instead: "
        f"{offenders}"
    )


# --- controls: the detector must fire on mocks and stay quiet on stubs ---


def test_detector_flags_mock_imports() -> None:
    assert find_mock_usage("from unittest.mock import patch\n") != []
    assert find_mock_usage("import mock\n") != []
    assert find_mock_usage("from mock import patch\n") != []
    assert find_mock_usage("from mock import MagicMock\n") != []
    assert find_mock_usage("from unittest import mock\n") != []
    assert find_mock_usage("import unittest.mock as m\n") != []


def test_detector_flags_mock_attribute_use_and_mocker_fixture() -> None:
    assert find_mock_usage("x = mock.patch('os.path.join')\n") != []
    assert find_mock_usage("x = unittest.mock.MagicMock()\n") != []
    assert find_mock_usage("def test_something(mocker):\n    pass\n") != []
    assert find_mock_usage("async def test_something(mocker):\n    pass\n") != []


def test_detector_allows_hand_written_stubs() -> None:
    stub = "def replay_fn(input_text: str) -> str:\n    return recorded.get(input_text, '')\n"
    assert find_mock_usage(stub) == []
    # A local named `patch` (a diff string, as in test_bugsmith.py) is not
    # mock machinery: only the mock module / fixture trip the detector.
    assert find_mock_usage("patch = open('fix.patch').read()\n") == []
