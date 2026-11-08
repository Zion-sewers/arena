"""Seeded end-to-end demo of the metric-assumption ledger (arena#56).

Runs the full ledger pipeline offline on the shipped registry. Exit 0
only if every acceptance expectation holds:
- every shipped metric lists at least one assumption, each with a
  falsifier;
- the shipped ledger audit flags exactly the three honestly-unbacked
  soft metrics (alt_conservative_score, calibration_gap,
  perturbation_stability) and nothing else: every other soft metric
  is backed by a HARD metric that is independently measured;
- kind is coerced and validated at construction: a plain "soft"
  string becomes SOFT, a garbage kind raises, and a hand-built spec
  can never slip a non-Enum kind past the audit as clean;
- audit_measured audits only what the report measured: a soft metric
  whose backup was not measured in this report is flagged, and a
  metric the report did not print is never audited;
- audit_report flags a reported metric with no ledger entry, and
  refuses a bare string;
- an assumption-free metric is refused at construction, and a
  whitespace-only assumption, name, or phenomenon is refused too;
- a list passed as assumptions is copied: mutating the source list
  leaves the frozen spec unchanged;
- a soft metric without a hard backup constructs and is flagged by
  the audit (the honest unbacked filing), never silently clean;
- the audit flags a soft metric whose backup names a missing metric,
  and one whose backup names a SOFT metric;
- the circular backups from the first rounds are gone: the gap and
  the stability drop are filed UNBACKED, and the matcher proxies are
  backed by label_agreement (ground-truth validation, not the
  scorer's own output);
- the fail-closed parser rebuilds the shipped ledger bit-for-bit and
  refuses a spec with unknown keys;
- the audit summary prints the assumption texts, and to_dict carries
  the full specs;
- an empty audit measures nothing, so it passes through: its
  summary says "no metrics measured", never "0/0 metrics clean",
  and require_ledger_ok raises on a flagged audit, not an empty one.

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
    audit_measured,
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
    _expect(
        audit.flagged
        == [
            "alt_conservative_score",
            "calibration_gap",
            "perturbation_stability",
        ],
        "shipped ledger flags exactly the three honestly-unbacked soft "
        "metrics, nothing else",
    )
    kinds = {spec.kind for spec in METRIC_LEDGER.values()}
    _expect(
        kinds == {MetricKind.HARD, MetricKind.SOFT},
        "ledger covers both HARD and SOFT metrics",
    )
    for name, spec in METRIC_LEDGER.items():
        if spec.kind is MetricKind.SOFT:
            backup = spec.hard_backup
            if backup is None:
                _expect(
                    name in audit.flagged,
                    f"unbacked soft {name} is flagged by the audit",
                )
            else:
                _expect(
                    METRIC_LEDGER[backup].kind is MetricKind.HARD,
                    f"soft {name} is backed by hard {backup}",
                )
    _expect(
        METRIC_LEDGER["calibration_gap"].hard_backup is None,
        "calibration_gap is filed UNBACKED: the gap is computed from "
        "the agreement rate itself, so it cannot check itself",
    )
    _expect(
        METRIC_LEDGER["perturbation_stability"].hard_backup is None,
        "perturbation_stability is filed UNBACKED: the drops are "
        "differences of the pass rates themselves",
    )
    _expect(
        METRIC_LEDGER["contains"].hard_backup == "label_agreement"
        and METRIC_LEDGER["regex"].hard_backup == "label_agreement"
        and METRIC_LEDGER["json_schema"].hard_backup == "label_agreement",
        "the soft matchers are backed by ground-truth label agreement, "
        "not the scorer's own output",
    )
    _expect(
        METRIC_LEDGER["p_conventional"].hard_backup == "conventional_share",
        "p_conventional is backed by the measured cluster share",
    )
    # Live derivation: the audit covers only what the report measured.
    run_audit = audit_measured({"pass_rate": True})
    _expect(
        run_audit.all_ok and [f.metric for f in run_audit.findings] == ["pass_rate"],
        "run without labels audits pass_rate only — never a metric "
        "the report never showed",
    )
    labels_audit = audit_measured(
        {"pass_rate": True, "label_agreement": True, "contains": True}
    )
    _expect(
        labels_audit.all_ok,
        "run with validity labels audits the measured matcher and its "
        "backup clean",
    )
    unmeasured_backup = audit_measured({"contains": True})
    _expect(
        unmeasured_backup.flagged == ["contains"]
        and unmeasured_backup.findings[0].status
        is LedgerStatus.FLAGGED_NO_BACKUP,
        "a soft metric whose backup was not measured in this report "
        "is flagged, even though the registry names one",
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
        audit_report("pass_rate")  # type: ignore[arg-type]
    except TypeError:
        _expect(True, "audit_report refuses a bare string")
    else:
        raise AssertionError("audit_report audited a bare string's characters")

    # Kind is coerced and validated at construction (round-2 B1).
    plain = MetricSpec(
        name="vibes",
        kind="soft",  # type: ignore[arg-type]
        phenomenon="how good it felt",
        assumptions=(MetricAssumption(text="it felt good", falsifier="it felt bad"),),
    )
    _expect(plain.kind is MetricKind.SOFT, 'plain "soft" string coerces to SOFT')
    _expect(
        audit_measured({"vibes": True}, ledger={"vibes": plain}).flagged == ["vibes"],
        "a coerced soft metric with no backup audits flagged, never clean",
    )
    try:
        MetricSpec(
            name="vibes",
            kind="banana",  # type: ignore[arg-type]
            phenomenon="how good it felt",
            assumptions=(MetricAssumption(text="t", falsifier="f"),),
        )
    except ValueError:
        _expect(True, "a garbage kind raises at construction")
    else:
        raise AssertionError("a garbage kind constructed")

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

    unbacked = MetricSpec(
        name="vibes",
        kind=MetricKind.SOFT,
        phenomenon="how good it felt",
        assumptions=(
            MetricAssumption(text="it felt good", falsifier="it felt bad"),
        ),
    )
    _expect(
        unbacked.hard_backup is None,
        "soft without backup constructs (the honest unbacked filing)",
    )
    _expect(
        audit_measured({"vibes": True}, ledger={"vibes": unbacked}).flagged
        == ["vibes"],
        "soft without backup audits flagged, never silently clean",
    )

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
    _expect(empty.all_ok, "an empty ledger audit passes through: "
            "nothing was measured, so there is nothing to flag")
    _expect(
        "no metrics measured" in empty.summary()
        and "0/0 metrics clean" not in empty.summary(),
        "an empty audit never prints an 'N/N clean' headline",
    )
    _expect(
        empty.to_dict()["all_ok"] is True,
        "an empty audit's JSON never pairs a PASS with all_ok false",
    )
    try:
        require_ledger_ok(empty)
    except ValueError:  # pragma: no cover - the pass is the expectation
        raise AssertionError("require_ledger_ok refused an empty audit")
    else:
        _expect(True, "require_ledger_ok passes an empty audit")
    # Round 4 (impl B1): a spec built by bypassing __post_init__ with
    # a plain-string kind is audited as the kind it names, and the
    # summary renders it without crashing.
    sneaky = object.__new__(MetricSpec)
    object.__setattr__(sneaky, "name", "sneaky_soft")
    object.__setattr__(sneaky, "kind", "soft")
    object.__setattr__(sneaky, "phenomenon", "how good it felt")
    object.__setattr__(
        sneaky, "assumptions", (MetricAssumption(text="t", falsifier="f"),)
    )
    object.__setattr__(sneaky, "hard_backup", None)
    sneaky_audit = audit_measured(
        {"sneaky_soft": True}, ledger={"sneaky_soft": sneaky}
    )
    _expect(
        sneaky_audit.flagged == ["sneaky_soft"],
        "a bypassed soft kind audits flagged, never clean",
    )
    _expect(
        "sneaky_soft" in sneaky_audit.summary(),
        "the summary renders a bypassed-kind audit without crashing",
    )
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
        audit_ledger(rebuilt).flagged
        == [
            "alt_conservative_score",
            "calibration_gap",
            "perturbation_stability",
        ],
        "fail-closed parser rebuilds the shipped ledger bit-for-bit",
    )
    _expect(
        spec_from_dict(payloads[0]) == METRIC_LEDGER[payloads[0]["name"]],
        "parser round-trips one shipped spec",
    )
    parsed_unbacked = spec_from_dict(
        {
            "name": "vibes",
            "kind": "soft",
            "phenomenon": "how good it felt",
            "assumptions": [{"text": "t", "falsifier": "f"}],
        }
    )
    _expect(
        parsed_unbacked.hard_backup is None,
        "parser accepts a soft spec with no backup (audit flags it)",
    )
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
