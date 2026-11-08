"""Seeded end-to-end demo of the metric-assumption ledger (arena#56).

Runs the full ledger pipeline offline on the shipped registry. Exit 0
only if every acceptance expectation holds:
- every shipped metric lists at least one assumption, each with a
  falsifier;
- the shipped ledger audits clean, and every per-command metric set
  audits clean when fed to audit_report;
- audit_report flags a reported metric with no ledger entry;
- an assumption-free metric is refused at construction, and a
  whitespace-only assumption, name, or phenomenon is refused too;
- a list passed as assumptions is copied: mutating the source list
  leaves the frozen spec unchanged;
- a soft metric without a hard backup is refused at construction;
- the audit flags a soft metric whose backup names a missing metric,
  and one whose backup names a SOFT metric;
- the two circular backups from the first round are gone:
  scorer_validity is backed by label_agreement (ground-truth
  validation, not the scorer's own output) and calibration_gap by
  agreement_rate (its own measured component);
- the fail-closed parser rebuilds the shipped ledger bit-for-bit and
  refuses a soft spec with no backup and a spec with unknown keys;
- the audit summary prints the assumption texts, and to_dict carries
  the full specs;
- an empty audit is not clean, and require_ledger_ok raises on a
  flagged audit.

Any failure raises, so CI can gate on this demo directly.
"""

from __future__ import annotations

import sys

from agenteval_bench.metric_ledger import (
    METRIC_LEDGER,
    REPORT_METRIC_SETS,
    LedgerStatus,
    MetricAssumption,
    MetricKind,
    MetricSpec,
    audit_ledger,
    audit_report,
    ledger_from_dicts,
    require_ledger_ok,
    spec_from_dict,
)


def _expect(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(f"metric-ledger demo expectation failed: {label}")
    print(f"  ok: {label}")


def main() -> int:
    print("metric-ledger demo: Berman cash test for every reported metric")
    audit = audit_ledger()

    _expect(bool(METRIC_LEDGER), "the shipped ledger is non-empty")
    for name, spec in METRIC_LEDGER.items():
        _expect(
            len(spec.assumptions) >= 1,
            f"{name} lists its assumptions ({len(spec.assumptions)})",
        )
        for assumption in spec.assumptions:
            _expect(
                bool(assumption.text) and bool(assumption.falsifier),
                f"{name} assumption carries a falsifier",
            )
    _expect(audit.all_ok, "shipped ledger audits clean")
    _expect(audit.flagged == [], "no shipped metric is flagged")
    kinds = {spec.kind for spec in METRIC_LEDGER.values()}
    _expect(
        kinds == {MetricKind.HARD, MetricKind.SOFT},
        "ledger covers both HARD and SOFT metrics",
    )
    for name, spec in METRIC_LEDGER.items():
        if spec.kind is MetricKind.SOFT:
            backup = spec.hard_backup
            _expect(
                backup is not None
                and METRIC_LEDGER[backup].kind is MetricKind.HARD,
                f"soft {name} is backed by hard {backup}",
            )
    _expect(
        METRIC_LEDGER["scorer_validity"].hard_backup == "label_agreement",
        "scorer_validity is backed by ground-truth label agreement, "
        "not the scorer's own output",
    )
    _expect(
        METRIC_LEDGER["calibration_gap"].hard_backup == "agreement_rate",
        "calibration_gap is backed by its own measured agreement rate",
    )
    for command, metric_set in REPORT_METRIC_SETS.items():
        report_audit = audit_report(metric_set)
        _expect(
            report_audit.all_ok,
            f"the {command} report's metric set audits clean "
            f"({len(metric_set)} metrics)",
        )
    summary = audit.summary()
    for name in METRIC_LEDGER:
        _expect(name in summary, f"audit summary names {name}")
    _expect(
        "The scorer's matchers decide pass/fail exactly" in summary,
        "audit summary prints assumption texts, not just counts",
    )
    data = audit.to_dict()
    _expect(
        data["metrics"]["pass_rate"]["assumptions"],
        "audit JSON carries the full specs with assumptions",
    )

    ghost = audit_report(["ghost_metric"])
    _expect(
        ghost.flagged == ["ghost_metric"]
        and ghost.findings[0].status is LedgerStatus.FLAGGED_NOT_IN_LEDGER,
        "audit_report flags a reported metric with no ledger entry",
    )
    _expect(not ghost.all_ok, "a flagged report audit is not clean")

    try:
        MetricSpec(
            name="naked",
            kind=MetricKind.HARD,
            phenomenon="a number with no stated assumptions",
            assumptions=(),
        )
    except ValueError as exc:
        _expect("assumption-free" in str(exc), "assumption-free refused")
    else:
        raise AssertionError("assumption-free metric was accepted")

    try:
        MetricAssumption(text=" ", falsifier="  ")
    except ValueError:
        _expect(True, "whitespace-only assumption refused")
    else:
        raise AssertionError("whitespace-only assumption was accepted")

    try:
        MetricSpec(
            name=" ",
            kind=MetricKind.HARD,
            phenomenon="x",
            assumptions=(MetricAssumption(text="t", falsifier="f"),),
        )
    except ValueError:
        _expect(True, "whitespace-only name refused")
    else:
        raise AssertionError("whitespace-only name was accepted")

    source = [MetricAssumption(text="t", falsifier="f")]
    spec = MetricSpec(
        name="frozen",
        kind=MetricKind.HARD,
        phenomenon="a count",
        assumptions=source,  # type: ignore[arg-type]
    )
    source.clear()
    _expect(
        len(spec.assumptions) == 1,
        "mutating the source list leaves the frozen spec unchanged",
    )
    _expect(isinstance(spec.assumptions, tuple), "assumptions stored as tuple")

    try:
        MetricSpec(
            name="vibes",
            kind=MetricKind.SOFT,
            phenomenon="how good it felt",
            assumptions=(
                MetricAssumption(text="it felt good", falsifier="it felt bad"),
            ),
        )
    except ValueError as exc:
        _expect("hard backup" in str(exc), "soft without backup refused")
    else:
        raise AssertionError("soft metric without backup was accepted")

    ghost_backup = audit_ledger(
        {
            "vibes": MetricSpec(
                name="vibes",
                kind=MetricKind.SOFT,
                phenomenon="how good it felt",
                assumptions=(
                    MetricAssumption(
                        text="it felt good", falsifier="it felt bad"
                    ),
                ),
                hard_backup="ghost",
            )
        }
    )
    _expect(
        ghost_backup.flagged == ["vibes"]
        and ghost_backup.findings[0].status is LedgerStatus.FLAGGED_BAD_BACKUP,
        "audit flags a soft metric whose backup is missing",
    )
    soft_backed = audit_ledger(
        {
            "vibes": MetricSpec(
                name="vibes",
                kind=MetricKind.SOFT,
                phenomenon="how good it felt",
                assumptions=(
                    MetricAssumption(
                        text="it felt good", falsifier="it felt bad"
                    ),
                ),
                hard_backup="feelings",
            ),
            "feelings": MetricSpec(
                name="feelings",
                kind=MetricKind.SOFT,
                phenomenon="a feeling",
                assumptions=(
                    MetricAssumption(text="t", falsifier="f"),
                ),
                hard_backup="pass_rate",
            ),
            "pass_rate": METRIC_LEDGER["pass_rate"],
        }
    )
    _expect(
        soft_backed.flagged == ["vibes"],
        "audit flags a soft metric backed by a SOFT metric",
    )
    empty = audit_ledger({})
    _expect(not empty.all_ok, "an empty ledger audit is not clean")
    try:
        require_ledger_ok(ghost)
    except ValueError:
        _expect(True, "require_ledger_ok raises on a flagged audit")
    else:
        raise AssertionError("require_ledger_ok passed a flagged audit")

    payloads = [
        dict(METRIC_LEDGER[name].to_dict(), name=name) for name in METRIC_LEDGER
    ]
    rebuilt = ledger_from_dicts(payloads)
    _expect(
        audit_ledger(rebuilt).all_ok,
        "fail-closed parser rebuilds the shipped ledger",
    )
    _expect(
        spec_from_dict(payloads[0]) == METRIC_LEDGER[payloads[0]["name"]],
        "parser round-trips one shipped spec",
    )
    try:
        spec_from_dict(
            {
                "name": "vibes",
                "kind": "soft",
                "phenomenon": "how good it felt",
                "assumptions": [{"text": "t", "falsifier": "f"}],
            }
        )
    except ValueError as exc:
        _expect("hard backup" in str(exc), "parser refuses soft without backup")
    else:
        raise AssertionError("parser accepted a soft spec with no backup")
    try:
        spec_from_dict(
            {
                "name": "vibes",
                "kind": "soft",
                "phenomenon": "how good it felt",
                "assumptions": [{"text": "t", "falsifier": "f"}],
                "hard_backup": "pass_rate",
                "surprise": True,
            }
        )
    except ValueError as exc:
        _expect("unknown keys" in str(exc), "parser refuses unknown keys")
    else:
        raise AssertionError("parser accepted unknown keys")

    print("metric-ledger demo: all expectations hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
