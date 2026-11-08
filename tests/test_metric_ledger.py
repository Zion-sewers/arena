"""Tests for the metric-assumption ledger (arena#56).

Adversarial cases: assumption-free specs refused at construction, at
parsing, and at audit; soft metrics without a backup, with a missing
backup, or with a SOFT backup refused at build and flagged at audit;
a hand-built ledger with a mis-filed spec flagged; duplicate names
refused.
"""

from __future__ import annotations

import pytest

from agenteval_bench.metric_ledger import (
    METRIC_LEDGER,
    LedgerAudit,
    LedgerStatus,
    MetricAssumption,
    MetricKind,
    MetricSpec,
    audit_ledger,
    ledger_from_dicts,
    require_ledger_ok,
    spec_from_dict,
)


def _assumption(text: str = "the scorer decides pass/fail exactly") -> MetricAssumption:
    return MetricAssumption(
        text=text,
        falsifier="a case passes whose output fails on manual inspection",
    )


def _hard(name: str) -> MetricSpec:
    return MetricSpec(
        name=name,
        kind=MetricKind.HARD,
        phenomenon=f"measured {name}",
        assumptions=(_assumption(),),
    )


def _soft(name: str, backup: str) -> MetricSpec:
    return MetricSpec(
        name=name,
        kind=MetricKind.SOFT,
        phenomenon=f"inferred {name}",
        assumptions=(_assumption(),),
        hard_backup=backup,
    )


class TestAssumptionFreeRefused:
    def test_empty_assumptions_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="assumption-free"):
            MetricSpec(
                name="naked",
                kind=MetricKind.HARD,
                phenomenon="a number with no stated assumptions",
                assumptions=(),
            )

    def test_empty_assumption_text_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty text"):
            MetricAssumption(text="", falsifier="something breaks")

    def test_assumption_without_falsifier_refused(self) -> None:
        with pytest.raises(ValueError, match="falsifier"):
            MetricAssumption(text="the scorer is exact", falsifier="")

    def test_empty_phenomenon_refused(self) -> None:
        with pytest.raises(ValueError, match="phenomenon"):
            MetricSpec(
                name="naked",
                kind=MetricKind.HARD,
                phenomenon="",
                assumptions=(_assumption(),),
            )


class TestSoftNeedsHardBackup:
    def test_soft_without_backup_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="hard backup"):
            MetricSpec(
                name="vibes",
                kind=MetricKind.SOFT,
                phenomenon="how good it felt",
                assumptions=(_assumption(),),
            )

    def test_hard_with_backup_refused(self) -> None:
        with pytest.raises(ValueError, match="carries no backup"):
            MetricSpec(
                name="count",
                kind=MetricKind.HARD,
                phenomenon="a direct count",
                assumptions=(_assumption(),),
                hard_backup="pass_rate",
            )

    def test_audit_flags_soft_with_missing_backup(self) -> None:
        ledger = {"vibes": _soft("vibes", "no_such_metric")}
        audit = audit_ledger(ledger)
        assert audit.flagged == ["vibes"]
        finding = audit.findings[0]
        assert finding.status is LedgerStatus.FLAGGED_BAD_BACKUP
        assert "not in the ledger" in finding.detail

    def test_audit_flags_soft_backed_by_soft(self) -> None:
        ledger = {
            "vibes": _soft("vibes", "feelings"),
            "feelings": _soft("feelings", "pass_rate"),
            "pass_rate": _hard("pass_rate"),
        }
        audit = audit_ledger(ledger)
        assert audit.flagged == ["vibes"]
        finding = next(f for f in audit.findings if f.metric == "vibes")
        assert finding.status is LedgerStatus.FLAGGED_BAD_BACKUP
        assert "not HARD" in finding.detail

    def test_audit_flags_misfiled_spec(self) -> None:
        # A spec filed under a different key than its own name.
        ledger = {"wrong_key": _hard("right_name")}
        audit = audit_ledger(ledger)
        assert audit.flagged == ["wrong_key"]


class TestShippedLedger:
    def test_every_shipped_metric_lists_assumptions(self) -> None:
        assert METRIC_LEDGER
        for name, spec in METRIC_LEDGER.items():
            assert spec.assumptions, f"{name} lists no assumptions"
            for assumption in spec.assumptions:
                assert assumption.text
                assert assumption.falsifier

    def test_shipped_ledger_audits_clean(self) -> None:
        audit = audit_ledger()
        assert audit.all_ok
        assert audit.flagged == []
        require_ledger_ok(audit)

    def test_shipped_ledger_has_both_kinds(self) -> None:
        kinds = {spec.kind for spec in METRIC_LEDGER.values()}
        assert kinds == {MetricKind.HARD, MetricKind.SOFT}

    def test_every_soft_metric_backed_by_hard(self) -> None:
        for spec in METRIC_LEDGER.values():
            if spec.kind is MetricKind.SOFT:
                backup = spec.hard_backup
                assert backup is not None
                assert METRIC_LEDGER[backup].kind is MetricKind.HARD
            else:
                assert spec.hard_backup is None

    def test_audit_summary_names_every_metric(self) -> None:
        summary = audit_ledger().summary()
        for name in METRIC_LEDGER:
            assert name in summary
        assert "metrics clean" in summary

    def test_audit_to_dict_round_trips(self) -> None:
        data = audit_ledger().to_dict()
        assert data["all_ok"] is True
        assert data["flagged"] == []
        assert len(data["findings"]) == len(METRIC_LEDGER)

    def test_spec_to_dict_carries_assumptions_and_backup(self) -> None:
        data = METRIC_LEDGER["judge_score"].to_dict()
        assert data["kind"] == "soft"
        assert data["hard_backup"] == "quality"
        assert len(data["assumptions"]) == 3
        assert data["assumptions"][0]["falsifier"]


class TestFailClosedParsing:
    def test_spec_from_dict_round_trip(self) -> None:
        payload = METRIC_LEDGER["calibration_gap"].to_dict()
        payload["name"] = "calibration_gap"
        parsed = spec_from_dict(payload)
        assert parsed == METRIC_LEDGER["calibration_gap"]

    def test_parser_refuses_missing_assumptions(self) -> None:
        with pytest.raises(ValueError, match="assumptions"):
            spec_from_dict(
                {
                    "name": "naked",
                    "kind": "hard",
                    "phenomenon": "a number",
                    "assumptions": [],
                }
            )

    def test_parser_refuses_soft_without_backup(self) -> None:
        with pytest.raises(ValueError, match="hard backup"):
            spec_from_dict(
                {
                    "name": "vibes",
                    "kind": "soft",
                    "phenomenon": "how good it felt",
                    "assumptions": [
                        {"text": "it felt good", "falsifier": "it felt bad"}
                    ],
                }
            )

    def test_parser_refuses_bad_kind(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            spec_from_dict(
                {
                    "name": "x",
                    "kind": "medium",
                    "phenomenon": "a number",
                    "assumptions": [
                        {"text": "t", "falsifier": "f"},
                    ],
                }
            )

    def test_parser_refuses_non_string_backup(self) -> None:
        with pytest.raises(ValueError, match="hard_backup"):
            spec_from_dict(
                {
                    "name": "vibes",
                    "kind": "soft",
                    "phenomenon": "how good it felt",
                    "assumptions": [
                        {"text": "t", "falsifier": "f"},
                    ],
                    "hard_backup": 42,
                }
            )

    def test_ledger_from_dicts_refuses_missing_backup(self) -> None:
        with pytest.raises(ValueError, match="not in the ledger"):
            ledger_from_dicts(
                [
                    {
                        "name": "vibes",
                        "kind": "soft",
                        "phenomenon": "how good it felt",
                        "assumptions": [
                            {"text": "t", "falsifier": "f"},
                        ],
                        "hard_backup": "ghost",
                    }
                ]
            )

    def test_ledger_from_dicts_refuses_soft_backup(self) -> None:
        with pytest.raises(ValueError, match="is not HARD"):
            ledger_from_dicts(
                [
                    {
                        "name": "vibes",
                        "kind": "soft",
                        "phenomenon": "how good it felt",
                        "assumptions": [
                            {"text": "t", "falsifier": "f"},
                        ],
                        "hard_backup": "feelings",
                    },
                    {
                        "name": "feelings",
                        "kind": "soft",
                        "phenomenon": "a feeling",
                        "assumptions": [
                            {"text": "t", "falsifier": "f"},
                        ],
                        "hard_backup": "pass_rate",
                    },
                    {
                        "name": "pass_rate",
                        "kind": "hard",
                        "phenomenon": "a count",
                        "assumptions": [
                            {"text": "t", "falsifier": "f"},
                        ],
                    },
                ]
            )

    def test_ledger_from_dicts_refuses_duplicate_names(self) -> None:
        payload = {
            "name": "pass_rate",
            "kind": "hard",
            "phenomenon": "a count",
            "assumptions": [{"text": "t", "falsifier": "f"}],
        }
        with pytest.raises(ValueError, match="duplicate"):
            ledger_from_dicts([payload, payload])

    def test_ledger_from_dicts_builds_clean_ledger(self) -> None:
        payloads = [
            dict(METRIC_LEDGER[name].to_dict(), name=name)
            for name in METRIC_LEDGER
        ]
        built = ledger_from_dicts(payloads)
        assert audit_ledger(built).all_ok
        assert set(built) == set(METRIC_LEDGER)

    def test_ledger_audit_is_frozen(self) -> None:
        audit = audit_ledger()
        with pytest.raises(AttributeError):
            audit.findings = ()  # type: ignore[misc]
        assert isinstance(audit, LedgerAudit)
