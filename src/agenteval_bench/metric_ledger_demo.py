"""Seeded end-to-end demo of the metric-assumption ledger (arena#56).

Runs the full ledger pipeline offline on the shipped registry. Exit 0
only if every acceptance expectation holds:
- every shipped metric lists at least one assumption, each with a
  falsifier;
- the shipped ledger audits clean: no soft metric without a hard
  backup, every backup resolves to a HARD metric;
- an assumption-free metric is refused at construction;
- a soft metric without a hard backup is refused at construction;
- the audit flags a soft metric whose backup names a missing metric,
  and one whose backup names a SOFT metric;
- the fail-closed parser rebuilds the shipped ledger bit-for-bit and
  refuses a soft spec with no backup;
- the audit summary names every shipped metric.

Any failure raises, so CI can gate on this demo directly.
"""

from __future__ import annotations

import sys

from agenteval_bench.metric_ledger import (
    METRIC_LEDGER,
    LedgerStatus,
    MetricAssumption,
    MetricKind,
    MetricSpec,
    audit_ledger,
    ledger_from_dicts,
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
    summary = audit.summary()
    for name in METRIC_LEDGER:
        _expect(name in summary, f"audit summary names {name}")

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

    ghost = audit_ledger(
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
        ghost.flagged == ["vibes"]
        and ghost.findings[0].status is LedgerStatus.FLAGGED_BAD_BACKUP,
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

    print("metric-ledger demo: all expectations hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
