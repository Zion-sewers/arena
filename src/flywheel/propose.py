"""Propose: render a validated rule as a postable PR payload (arena#40).

Only a ValidatedRule reaches this module — the function signature is the
dry-run gate. The payload carries the wiring file contents for a live
consumer plus the dry-run evidence in the body, so a reviewer can see
exactly what was measured (injected faults, gate-catch rate, false
positives) and on what corpus. The flywheel never writes the payload
into the target tree; the CLI writes payload files out for the fleet
runner, which posts them via the GitHub API (the poster itself is a
follow-up, not shipped here).

Every dry-run, promotable or not, is appended to the gate-catch ledger
(JSONL, caller-chosen path; the CLI defaults it to a durable path under
the target repo). The gate-catch rate in the ledger is the loop's
leading metric: a loop that proposes rules nobody measures is
not a loop.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from flywheel.rules import wiring
from flywheel.types import DryRunReport, FlywheelError, PrPayload, ValidatedRule

LEDGER_SCHEMA_VERSION = 1
_LEDGER_KEYS = frozenset(
    {
        "schema_version",
        "ts",
        "rule_kind",
        "target_repo",
        "injected",
        "caught",
        "catch_rate",
        "false_positives",
        "promotable",
    }
)


def render_pr_payload(rule: ValidatedRule, target_root: Path) -> PrPayload:
    """Render the PR payload for a validated rule. ``target_root`` is read
    (pyproject text for the wiring); it is never written."""
    report = rule.report
    files = wiring(rule.draft.kind, target_root)
    case_lines = "\n".join(
        f"- {c.name} ({c.file}): flagged={c.flagged}, "
        f"fail_to_pass={list(c.fail_to_pass) or 'none'}"
        + (f", enforcer_failed={c.enforcer_failed}" if report.requires_enforcer_check else "")
        for c in report.cases
    )
    body = (
        f"Flywheel-proposed blocking rule: **{rule.draft.kind}**\n\n"
        f"Evidence: {rule.draft.support} sampled fix-commits "
        f"({', '.join(sorted({s[:7] for s in rule.draft.evidence_shas}))}); "
        f"{rule.draft.rationale}\n\n"
        "Fix-commit evidence is message-keyword signal only (no file or "
        "diff evidence — the list endpoint carries no files). The dry-run "
        "below is the evidence that the rule catches its class, not the "
        "cited commits.\n\n"
        "Dry-run (this rule was validated before it could become blocking):\n"
        f"- injected faults of the class: {report.injected}\n"
        f"- caught by the rule: {report.caught} "
        f"(gate-catch rate {report.catch_rate:.0%})\n"
        f"- false positives on the clean tree: {report.false_positives}\n"
        f"{case_lines}\n\n"
        "Dry-run corpus: the vendored mcalc fixture (bugsmith), suite outcome "
        "via the bugsmith harness (fail_to_pass per case above). A rule that "
        "had not passed this dry-run could not have produced this PR: the "
        "flywheel renders payloads from ValidatedRule only.\n"
    )
    return PrPayload(
        title=(
            f"feat: blocking rule {rule.draft.kind} "
            f"(flywheel dry-run, gate-catch {report.catch_rate:.0%})"
        ),
        body=body,
        files=files,
        target_repo=rule.draft.target_repo,
        rule_kind=rule.draft.kind,
    )


def append_ledger(ledger_path: Path, report: DryRunReport) -> None:
    """Append one dry-run to the gate-catch ledger (JSONL). The ledger
    path is caller-chosen; the flywheel never writes into a repo tree."""
    entry = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "ts": datetime.now(UTC).isoformat(),
        "rule_kind": report.kind,
        "target_repo": report.target_repo,
        "injected": report.injected,
        "caught": report.caught,
        "catch_rate": report.catch_rate,
        "false_positives": report.false_positives,
        "promotable": report.promotable,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _is_count(value: object) -> bool:
    """A ledger count: a real int, never a bool (bool is an int subclass)
    and never negative."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_ledger_entry(entry: dict[str, object], line_no: int) -> None:
    injected = entry.get("injected")
    caught = entry.get("caught")
    false_positives = entry.get("false_positives")
    catch_rate = entry.get("catch_rate")
    for name, value in (
        ("injected", injected),
        ("caught", caught),
        ("false_positives", false_positives),
    ):
        if not _is_count(value):
            raise FlywheelError(
                f"ledger line {line_no} has an invalid {name}: {value!r}"
            )
    assert isinstance(injected, int) and isinstance(caught, int)
    assert isinstance(false_positives, int)
    if caught > injected:
        raise FlywheelError(
            f"ledger line {line_no} is inconsistent: caught={caught} "
            f"exceeds injected={injected}"
        )
    if isinstance(catch_rate, bool) or not isinstance(catch_rate, (int, float)):
        raise FlywheelError(
            f"ledger line {line_no} has a non-numeric catch_rate: {catch_rate!r}"
        )
    if not 0.0 <= float(catch_rate) <= 1.0:
        raise FlywheelError(
            f"ledger line {line_no} has catch_rate outside [0, 1]: {catch_rate!r}"
        )
    expected = (caught / injected) if injected else 0.0
    if abs(float(catch_rate) - expected) > 1e-9:
        raise FlywheelError(
            f"ledger line {line_no} is inconsistent: catch_rate={catch_rate} "
            f"but caught/injected = {expected}"
        )


def read_ledger(ledger_path: Path) -> list[dict[str, object]]:
    """Read and schema-validate the ledger. A corrupt line fails loudly:
    the leading metric must never be computed from half-parsed data.
    Validation covers value ranges and internal consistency, not just
    key presence: a ledger line claiming catches it does not carry is
    refused here, the same way DryRunReport refuses it at construction."""
    if not ledger_path.is_file():
        return []
    entries: list[dict[str, object]] = []
    for line_no, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            raise FlywheelError(f"ledger line {line_no} is not valid JSON: {e}") from e
        if not isinstance(entry, dict) or not _LEDGER_KEYS.issubset(entry.keys()):
            raise FlywheelError(
                f"ledger line {line_no} is missing required fields: {sorted(entry.keys()) if isinstance(entry, dict) else type(entry)}"
            )
        if entry.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise FlywheelError(
                f"ledger line {line_no} has unsupported schema_version "
                f"{entry.get('schema_version')}"
            )
        _validate_ledger_entry(entry, line_no)
        entries.append(entry)
    return entries


def leading_gate_catch_rate(ledger_path: Path) -> float | None:
    """The loop's leading metric: pooled gate-catch rate across ledgered
    dry-runs (total caught / total injected), so a 1-case run does not
    count the same as a 2-case run. None when the ledger is empty (no
    measurement yet, not zero)."""
    entries = read_ledger(ledger_path)
    if not entries:
        return None
    injected = sum(cast(int, e["injected"]) for e in entries)
    caught = sum(cast(int, e["caught"]) for e in entries)
    if injected == 0:
        return None
    return caught / injected
