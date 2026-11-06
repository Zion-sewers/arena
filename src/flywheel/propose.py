"""Propose: render a validated rule as a postable PR payload (arena#40).

Only a ValidatedRule reaches this module — the function signature is the
dry-run gate. The payload carries the wiring file contents for a live
consumer plus the dry-run evidence in the body, so a reviewer can see
exactly what was measured (injected faults, gate-catch rate, false
positives) and on what corpus. The flywheel never writes the payload
into the target tree; the fleet runner posts it via the GitHub API.

Every dry-run, promotable or not, is appended to the gate-catch ledger
(JSONL, caller-chosen path). The gate-catch rate in the ledger is the
loop's leading metric: a loop that proposes rules nobody measures is
not a loop.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

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
        for c in report.cases
    )
    body = (
        f"Flywheel-proposed blocking rule: **{rule.draft.kind}**\n\n"
        f"Evidence: {rule.draft.support} sampled fix-commits "
        f"({', '.join(sorted({s[:7] for s in rule.draft.evidence_shas}))}); "
        f"{rule.draft.rationale}\n\n"
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


def read_ledger(ledger_path: Path) -> list[dict[str, object]]:
    """Read and schema-validate the ledger. A corrupt line fails loudly:
    the leading metric must never be computed from half-parsed data."""
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
        entries.append(entry)
    return entries


def leading_gate_catch_rate(ledger_path: Path) -> float | None:
    """The loop's leading metric: mean gate-catch rate across ledgered
    dry-runs. None when the ledger is empty (no measurement yet, not
    zero)."""
    entries = read_ledger(ledger_path)
    if not entries:
        return None
    rates = [entry["catch_rate"] for entry in entries]
    numeric = [r for r in rates if isinstance(r, (int, float))]
    if len(numeric) != len(rates):
        raise FlywheelError("ledger holds a non-numeric catch_rate: refusing the metric")
    return sum(numeric) / len(numeric)
