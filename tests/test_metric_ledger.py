"""Tests for the metric-assumption ledger (arena#56).

Adversarial cases: assumption-free and whitespace-only specs refused
at construction, at parsing, and at audit; a mutable source list
cannot mutate a frozen spec; soft metrics without a backup, with a
missing backup, or with a SOFT backup refused at build and flagged at
audit; a reported metric with no ledger entry flagged by audit_report;
a hand-built ledger with a mis-filed spec flagged; duplicate names
refused; an empty audit is not clean.
"""

from __future__ import annotations

import pytest

from agenteval_bench.metric_ledger import (
    METRIC_LEDGER,
    REPORT_METRIC_SETS,
    LedgerAudit,
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
        with pytest.raises(ValueError, match="non-empty"):
            MetricAssumption(text="", falsifier="something breaks")

    def test_whitespace_assumption_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            MetricAssumption(text=" ", falsifier="  ")

    def test_whitespace_name_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            MetricSpec(
                name=" ",
                kind=MetricKind.HARD,
                phenomenon="a number",
                assumptions=(_assumption(),),
            )

    def test_whitespace_phenomenon_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            MetricSpec(
                name="x",
                kind=MetricKind.HARD,
                phenomenon="   ",
                assumptions=(_assumption(),),
            )

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

    def test_source_list_mutation_leaves_spec_unchanged(self) -> None:
        source = [_assumption()]
        spec = MetricSpec(
            name="frozen",
            kind=MetricKind.HARD,
            phenomenon="a count",
            assumptions=source,  # type: ignore[arg-type]
        )
        source.clear()
        assert len(spec.assumptions) == 1
        assert isinstance(spec.assumptions, tuple)
        assert hash(spec) is not None

    def test_non_assumption_element_refused(self) -> None:
        with pytest.raises(TypeError, match="MetricAssumption"):
            MetricSpec(
                name="bad",
                kind=MetricKind.HARD,
                phenomenon="a count",
                assumptions=("not an assumption",),  # type: ignore[arg-type]
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
        assert (
            audit.findings[0].status is LedgerStatus.FLAGGED_MISFILED
        )

    def test_audit_flags_backup_cycle(self) -> None:
        # Two soft metrics backing each other: neither resolves to HARD.
        ledger = {
            "a": _soft("a", "b"),
            "b": _soft("b", "a"),
        }
        audit = audit_ledger(ledger)
        assert audit.flagged == ["a", "b"]

    def test_audit_flags_hard_carrying_backup(self) -> None:
        spec = _hard("count")
        object.__setattr__(spec, "hard_backup", "pass_rate")
        audit = audit_ledger({"count": spec, "pass_rate": _hard("pass_rate")})
        assert audit.flagged == ["count"]

    def test_audit_rejects_wrong_typed_value(self) -> None:
        with pytest.raises(TypeError, match="not a MetricSpec"):
            audit_ledger({"x": "not a spec"})  # type: ignore[arg-type]

    def test_empty_ledger_audit_is_not_clean(self) -> None:
        audit = audit_ledger({})
        assert not audit.all_ok
        assert "no metrics audited" in audit.summary()
        with pytest.raises(ValueError, match="no metrics"):
            require_ledger_ok(audit)

    def test_require_ledger_ok_raises_on_flagged(self) -> None:
        audit = audit_ledger({"vibes": _soft("vibes", "ghost")})
        with pytest.raises(ValueError, match="vibes"):
            require_ledger_ok(audit)


class TestAuditReport:
    def test_every_report_set_audits_clean(self) -> None:
        for command, metric_set in REPORT_METRIC_SETS.items():
            audit = audit_report(metric_set)
            assert audit.all_ok, f"{command} set flagged: {audit.flagged}"

    def test_report_sets_cover_the_shipped_ledger(self) -> None:
        covered = set().union(*REPORT_METRIC_SETS.values())
        assert covered == set(METRIC_LEDGER)

    def test_unknown_metric_flagged_not_in_ledger(self) -> None:
        audit = audit_report(["ghost_metric"])
        assert audit.flagged == ["ghost_metric"]
        assert (
            audit.findings[0].status is LedgerStatus.FLAGGED_NOT_IN_LEDGER
        )
        assert not audit.all_ok

    def test_report_audit_renders_from_the_audited_ledger(self) -> None:
        # The summary must describe the ledger actually audited, not
        # the shipped one: a custom ledger where quality is SOFT.
        custom = {
            "quality": _soft("quality", "pass_rate"),
            "pass_rate": _hard("pass_rate"),
        }
        summary = audit_ledger(custom).summary()
        assert "quality" in summary
        assert "soft" in summary
        assert "backup=pass_rate" in summary

    def test_report_audit_deduplicates_names(self) -> None:
        audit = audit_report(["pass_rate", "pass_rate"])
        assert len(audit.findings) == 1


class TestShippedLedger:
    def test_every_shipped_metric_lists_assumptions(self) -> None:
        assert METRIC_LEDGER
        for name, spec in METRIC_LEDGER.items():
            assert spec.assumptions, f"{name} lists no assumptions"
            for assumption in spec.assumptions:
                assert assumption.text
                assert assumption.text == assumption.text.strip()
                assert assumption.falsifier
                assert assumption.falsifier == assumption.falsifier.strip()

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

    def test_backups_are_not_circular(self) -> None:
        # The two circular backups from the first round: the scorer's
        # validity must not be backed by the scorer's own output, and
        # the calibration gap must be backed by its own measured
        # component, not by an unrelated rate.
        assert (
            METRIC_LEDGER["scorer_validity"].hard_backup == "label_agreement"
        )
        assert (
            METRIC_LEDGER["calibration_gap"].hard_backup == "agreement_rate"
        )

    def test_audit_summary_prints_assumption_texts(self) -> None:
        summary = audit_ledger().summary()
        for name in METRIC_LEDGER:
            assert name in summary
        assert "The scorer's matchers decide pass/fail exactly" in summary
        assert "metrics clean" in summary

    def test_audit_to_dict_carries_full_specs(self) -> None:
        data = audit_ledger().to_dict()
        assert data["all_ok"] is True
        assert data["flagged"] == []
        assert len(data["findings"]) == len(METRIC_LEDGER)
        metrics = data["metrics"]
        assert metrics["pass_rate"]["assumptions"]
        assert metrics["pass_rate"]["phenomenon"]
        assert metrics["calibration_gap"]["kind"] == "soft"
        assert metrics["calibration_gap"]["hard_backup"] == "agreement_rate"

    def test_spec_to_dict_carries_name_and_round_trips(self) -> None:
        for name, spec in METRIC_LEDGER.items():
            data = spec.to_dict()
            assert data["name"] == name
            assert spec_from_dict(data) == spec

    def test_ledger_audit_is_frozen(self) -> None:
        audit = audit_ledger()
        with pytest.raises(AttributeError):
            audit.findings = ()  # type: ignore[misc]
        assert isinstance(audit, LedgerAudit)


class TestFailClosedParsing:
    def test_spec_from_dict_round_trip(self) -> None:
        payload = METRIC_LEDGER["calibration_gap"].to_dict()
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

    def test_parser_refuses_whitespace_assumption(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            spec_from_dict(
                {
                    "name": "naked",
                    "kind": "hard",
                    "phenomenon": "a number",
                    "assumptions": [{"text": " ", "falsifier": "f"}],
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

    def test_parser_refuses_unknown_keys(self) -> None:
        with pytest.raises(ValueError, match="unknown keys"):
            spec_from_dict(
                {
                    "name": "vibes",
                    "kind": "soft",
                    "phenomenon": "how good it felt",
                    "assumptions": [
                        {"text": "t", "falsifier": "f"},
                    ],
                    "hard_backup": "pass_rate",
                    "surprise": True,
                }
            )

    def test_parser_accepts_tuple_assumptions(self) -> None:
        parsed = spec_from_dict(
            {
                "name": "x",
                "kind": "hard",
                "phenomenon": "a count",
                "assumptions": ({"text": "t", "falsifier": "f"},),
            }
        )
        assert parsed.assumptions[0].text == "t"

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
        payloads = [spec.to_dict() for spec in METRIC_LEDGER.values()]
        built = ledger_from_dicts(payloads)
        assert audit_ledger(built).all_ok
        assert set(built) == set(METRIC_LEDGER)
