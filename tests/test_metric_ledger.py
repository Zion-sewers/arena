"""Tests for the metric-assumption ledger (arena#56).

Adversarial cases: assumption-free and whitespace-only specs refused
at construction, at parsing, and at audit; a mutable source list
cannot mutate a frozen spec; a garbage kind is refused and a plain
"soft" string coerces (round-2 B1: no fail-open kind); soft metrics
without a backup, with a missing backup, or with a SOFT backup are
flagged at audit (an unbacked soft metric is the honest filing, not a
refusal); a soft metric whose backup was not measured in this report
is flagged; a reported metric with no ledger entry flagged by
audit_report; a hand-built ledger with a mis-filed spec flagged;
duplicate names refused; an empty audit is not clean.
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
    audit_measured,
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
    def test_soft_without_backup_constructs_but_audits_flagged(self) -> None:
        # Round 3: the honest filing for a soft metric with no
        # independent backup is UNBACKED, flagged in every report that
        # prints it — not a refusal, and never silently clean.
        spec = MetricSpec(
            name="vibes",
            kind=MetricKind.SOFT,
            phenomenon="how good it felt",
            assumptions=(_assumption(),),
        )
        assert spec.hard_backup is None
        audit = audit_measured({"vibes": True}, ledger={"vibes": spec})
        assert audit.flagged == ["vibes"]
        assert (
            audit.findings[0].status is LedgerStatus.FLAGGED_NO_BACKUP
        )

    def test_plain_string_kind_coerces_to_enum(self) -> None:
        # Round-2 B1: kind="soft" (a plain str, no backup) must not
        # slip past the backup rule as clean — it coerces to SOFT and
        # the audit flags it.
        spec = MetricSpec(
            name="vibes",
            kind="soft",  # type: ignore[arg-type]
            phenomenon="how good it felt",
            assumptions=(_assumption(),),
        )
        assert spec.kind is MetricKind.SOFT
        audit = audit_measured({"vibes": True}, ledger={"vibes": spec})
        assert audit.flagged == ["vibes"]
        plain_hard = MetricSpec(
            name="count",
            kind="hard",  # type: ignore[arg-type]
            phenomenon="a direct count",
            assumptions=(_assumption(),),
        )
        assert plain_hard.kind is MetricKind.HARD

    def test_garbage_kind_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            MetricSpec(
                name="vibes",
                kind="banana",  # type: ignore[arg-type]
                phenomenon="how good it felt",
                assumptions=(_assumption(),),
            )

    def test_non_string_kind_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            MetricSpec(
                name="vibes",
                kind=7,  # type: ignore[arg-type]
                phenomenon="how good it felt",
                assumptions=(_assumption(),),
            )

    def test_audit_flags_soft_with_unmeasured_backup(self) -> None:
        # Round-2 B3: naming a backup in the registry is not proof it
        # was measured in this report.
        ledger = {
            "contains": _soft("contains", "label_agreement"),
            "label_agreement": _hard("label_agreement"),
        }
        audit = audit_measured({"contains": True}, ledger=ledger)
        assert audit.flagged == ["contains"]
        finding = audit.findings[0]
        assert finding.status is LedgerStatus.FLAGGED_NO_BACKUP
        assert "not measured in this report" in finding.detail
        audit = audit_measured(
            {"contains": True, "label_agreement": True}, ledger=ledger
        )
        assert audit.all_ok

    def test_audit_measured_skips_unmeasured_names(self) -> None:
        # Round-2 B1/B3: a metric the report did not print is never
        # audited, so the headline never certifies a phantom number.
        audit = audit_measured(
            {"pass_rate": True, "label_agreement": False},
            ledger={"pass_rate": _hard("pass_rate")},
        )
        assert [f.metric for f in audit.findings] == ["pass_rate"]
        assert audit.all_ok

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

    def test_empty_ledger_audit_is_pass_through(self) -> None:
        # Round 4: an audit that measured nothing is not a failure —
        # it is nothing to gate on. Its summary never prints "0/0
        # metrics clean", its JSON never pairs a PASS with all_ok
        # false, and --strict-ledger passes through on it.
        audit = audit_ledger({})
        assert audit.all_ok
        assert audit.flagged == []
        assert "no metrics measured" in audit.summary()
        assert "0/0 metrics clean" not in audit.summary()
        require_ledger_ok(audit)  # must not raise
        assert audit.to_dict()["all_ok"] is True

    def test_audit_measured_rejects_non_spec_entry(self) -> None:
        # Round-4 B1 (impl): a hand-built non-MetricSpec entry fails
        # closed in audit_measured, exactly as audit_ledger already
        # refused it — a duck-typed soft metric with no backup must
        # never audit clean.
        import types

        fake = types.SimpleNamespace(
            name="my_soft",
            kind="soft",
            phenomenon="x",
            assumptions=[_assumption()],
            hard_backup=None,
        )
        with pytest.raises(TypeError, match="not a MetricSpec"):
            audit_measured({"my_soft": True}, ledger={"my_soft": fake})  # type: ignore[arg-type]

    def test_audit_coerces_bypassed_kind_and_never_crashes_summary(self) -> None:
        # Round-4 B1 (impl): a MetricSpec built by bypassing
        # __post_init__ with a plain-string kind is audited as the
        # kind it names, and the summary renders it without crashing.
        sneaky = object.__new__(MetricSpec)
        object.__setattr__(sneaky, "name", "sneaky_soft")
        object.__setattr__(sneaky, "kind", "soft")
        object.__setattr__(sneaky, "phenomenon", "how good it felt")
        object.__setattr__(sneaky, "assumptions", (_assumption(),))
        object.__setattr__(sneaky, "hard_backup", None)
        audit = audit_measured({"sneaky_soft": True}, ledger={"sneaky_soft": sneaky})
        assert audit.flagged == ["sneaky_soft"]
        assert (
            audit.findings[0].status is LedgerStatus.FLAGGED_NO_BACKUP
        )
        summary = audit.summary("round-4")
        assert "sneaky_soft" in summary
        assert "soft" in summary
        assert audit.to_dict()["metrics"]["sneaky_soft"]["kind"] == "soft"

    def test_audit_raises_on_bypassed_junk_assumption_items(self) -> None:
        # Round-5 N1 (impl): a MetricSpec built by bypassing
        # __post_init__ with a proper kind and a tuple of
        # non-MetricAssumption items must raise loudly at audit time,
        # never audit clean and crash at render time.
        sneaky = object.__new__(MetricSpec)
        object.__setattr__(sneaky, "name", "mixed")
        object.__setattr__(sneaky, "kind", MetricKind.HARD)
        object.__setattr__(sneaky, "phenomenon", "a count of things")
        object.__setattr__(
            sneaky, "assumptions", (_assumption(), "junk")
        )
        object.__setattr__(sneaky, "hard_backup", None)
        with pytest.raises(TypeError, match="MetricAssumption"):
            audit_measured({"mixed": True}, ledger={"mixed": sneaky})

    def test_audit_raises_on_bypassed_non_string_backup(self) -> None:
        # Round-7 (impl N1): a MetricSpec built by bypassing
        # __post_init__ with a non-string hard_backup must raise
        # loudly at audit time. An int backup flags
        # FLAGGED_BAD_BACKUP and then crashes the summary's string
        # render; a list backup raises unhashable inside the audit
        # itself. Neither is reachable through the constructor or
        # parser (both refuse non-string backups) — only via
        # object.__new__ bypass.
        for bad_backup in (5, ["pass_rate"]):
            sneaky = object.__new__(MetricSpec)
            object.__setattr__(sneaky, "name", "sneaky")
            object.__setattr__(sneaky, "kind", MetricKind.SOFT)
            object.__setattr__(sneaky, "phenomenon", "a soft number")
            object.__setattr__(sneaky, "assumptions", (_assumption(),))
            object.__setattr__(sneaky, "hard_backup", bad_backup)
            with pytest.raises(TypeError, match="hard_backup"):
                audit_measured({"sneaky": True}, ledger={"sneaky": sneaky})

    def test_validated_share_is_hard_and_audits_clean(self) -> None:
        # Round-5 (arch B2): bugsmith's validated share is a reported
        # proportion over recorded candidate reports — HARD, no backup.
        audit = audit_measured({"validated_share": True})
        assert audit.all_ok
        assert audit.flagged == []
        spec = METRIC_LEDGER["validated_share"]
        assert spec.kind is MetricKind.HARD
        assert spec.hard_backup is None

    def test_audit_raises_on_bypassed_garbage_kind(self) -> None:
        sneaky = object.__new__(MetricSpec)
        object.__setattr__(sneaky, "name", "sneaky")
        object.__setattr__(sneaky, "kind", "banana")
        object.__setattr__(sneaky, "phenomenon", "x")
        object.__setattr__(sneaky, "assumptions", (_assumption(),))
        object.__setattr__(sneaky, "hard_backup", None)
        with pytest.raises(ValueError, match="kind"):
            audit_measured({"sneaky": True}, ledger={"sneaky": sneaky})

    def test_require_ledger_ok_raises_on_flagged(self) -> None:
        audit = audit_ledger({"vibes": _soft("vibes", "ghost")})
        with pytest.raises(ValueError, match="vibes"):
            require_ledger_ok(audit)


class TestAuditReport:
    def test_audit_report_refuses_bare_string(self) -> None:
        # Auditing "pass_rate" as characters would flag each letter:
        # a latent footgun, refused outright.
        with pytest.raises(TypeError, match="bare string"):
            audit_report("pass_rate")  # type: ignore[arg-type]

    def test_audit_measured_refuses_bare_string(self) -> None:
        with pytest.raises(TypeError, match="mapping"):
            audit_measured("pass_rate")  # type: ignore[arg-type]

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

    def test_shipped_ledger_flags_only_the_unbacked_soft_metrics(self) -> None:
        # Round 3: the shipped ledger is NOT all-clean by design —
        # calibration_gap and perturbation_stability are filed UNBACKED
        # on purpose (no independent backup exists) and flagged here,
        # exactly as in the reports that print them. Round 6 adds
        # alt_conservative_score: a mean of Wilson bounds is SOFT
        # under the ledger rule, and the win rate cannot back it
        # (same wins), so it is filed UNBACKED too. Round 7 adds
        # the ship test statistics (oec_z, srm_chi2, aa_z — test
        # statistics computed through an estimated standard error)
        # and the guardrail lower bound (the bound the budget is
        # checked against): seven unbacked filings in total.
        audit = audit_ledger()
        assert audit.flagged == [
            "aa_z",
            "alt_conservative_score",
            "calibration_gap",
            "guardrail_lower_bound",
            "oec_z",
            "perturbation_stability",
            "srm_chi2",
        ]
        assert not audit.all_ok
        with pytest.raises(ValueError, match="flagged metrics"):
            require_ledger_ok(audit)

    def test_shipped_ledger_has_both_kinds(self) -> None:
        kinds = {spec.kind for spec in METRIC_LEDGER.values()}
        assert kinds == {MetricKind.HARD, MetricKind.SOFT}

    def test_round4_entries_exist_with_assumptions(self) -> None:
        # Round 4: ece (the headline calibrate number, own entry),
        # treated_share (simpson), the alt-test ranking numbers,
        # prior_score (snapshot promotions), and the tracebench
        # regression aggregates — every one with assumptions and a
        # falsifier. Round 6: alt_conservative_score is SOFT, not
        # HARD — a mean of Wilson lower bounds consumes an inferred
        # quantity — and filed UNBACKED, like the gap.
        for name in (
            "ece",
            "treated_share",
            "alt_annotator_rho",
            "prior_score",
            "regression_counts",
            "regression_streak",
        ):
            assert name in METRIC_LEDGER, f"{name} missing from the ledger"
            spec = METRIC_LEDGER[name]
            assert spec.kind is MetricKind.HARD
            assert spec.assumptions
            assert spec.hard_backup is None
        score = METRIC_LEDGER["alt_conservative_score"]
        assert score.kind is MetricKind.SOFT
        assert score.hard_backup is None
        assert score.assumptions
        assert "Wilson" in score.phenomenon

    def test_conservative_score_is_soft_because_it_consumes_a_bound(self) -> None:
        # Round-6 B1 (product + arch): the score is a mean of Wilson
        # 95% lower bounds, so the ledger rule files it SOFT; the win
        # rate is computed from the same wins and cannot back it, so
        # it is UNBACKED and every alt-test report flags it.
        score = METRIC_LEDGER["alt_conservative_score"]
        assert score.kind is MetricKind.SOFT
        assert score.hard_backup is None
        audit = audit_measured({"alt_conservative_score": True})
        assert audit.flagged == ["alt_conservative_score"]
        assert not audit.all_ok

    def test_ship_test_statistics_are_soft_and_unbacked(self) -> None:
        # Round-7 B1 (arch): a z-score is none of the rule's HARD
        # forms (count/sum/proportion/weighted mean) and takes an
        # inferential step — a pooled standard error estimated from
        # the recorded proportions, read through a normal/chi-square
        # approximation the entries' own assumptions admit. The
        # stipulated null does not save it: the null is stipulated,
        # the standard error is estimated. No HARD entry could back
        # a z-score (its check would be the p-value — the same
        # inference), so the filings are SOFT/UNBACKED and every
        # ship report flags them.
        for name in ("oec_z", "srm_chi2", "aa_z"):
            spec = METRIC_LEDGER[name]
            assert spec.kind is MetricKind.SOFT
            assert spec.hard_backup is None
            assert spec.assumptions
        audit = audit_measured(
            {
                "oec_z": True,
                "srm_chi2": True,
                "aa_z": True,
                "guardrail_delta": True,
                "guardrail_lower_bound": True,
            }
        )
        assert audit.flagged == [
            "aa_z",
            "guardrail_lower_bound",
            "oec_z",
            "srm_chi2",
        ]
        assert not audit.all_ok

    def test_guardrail_lower_bound_is_its_own_soft_entry(self) -> None:
        # Round-7 B2 (arch): the gate rests on the interval lower
        # bound (oec.py: "grade rests on interval BOUNDS, never point
        # estimates"), so the bound is its own SOFT/UNBACKED entry —
        # the point delta is the quantity the bound is built from
        # and cannot check it. The delta itself stays HARD: a plain
        # difference of recorded observations.
        delta = METRIC_LEDGER["guardrail_delta"]
        assert delta.kind is MetricKind.HARD
        assert delta.hard_backup is None
        assert "lower bound" not in delta.phenomenon
        bound = METRIC_LEDGER["guardrail_lower_bound"]
        assert bound.kind is MetricKind.SOFT
        assert bound.hard_backup is None
        assert "Newcombe" in bound.phenomenon
        audit = audit_measured(
            {"guardrail_delta": True, "guardrail_lower_bound": True}
        )
        assert audit.flagged == ["guardrail_lower_bound"]

    def test_ece_is_not_the_calibration_gap(self) -> None:
        # Round-4 B1 (arch): the per-bin gap (confidence minus the
        # Wilson lower bound of agreement) and ECE (bin-weighted mean
        # absolute difference) are different formulas — the gap's
        # phenomenon must not claim ECE is its aggregate.
        gap = METRIC_LEDGER["calibration_gap"]
        assert "ECE is the bin-weighted aggregate" not in gap.phenomenon
        assert "different aggregate" in gap.phenomenon
        assert METRIC_LEDGER["ece"].phenomenon != gap.phenomenon

    def test_every_soft_metric_backed_by_hard_or_flagged_unbacked(self) -> None:
        for spec in METRIC_LEDGER.values():
            if spec.kind is MetricKind.SOFT:
                backup = spec.hard_backup
                if backup is None:
                    assert spec.name in audit_ledger().flagged
                else:
                    assert METRIC_LEDGER[backup].kind is MetricKind.HARD
            else:
                assert spec.hard_backup is None

    def test_backups_are_not_circular(self) -> None:
        # Round-2 B2: the two circular backups are gone — the gap and
        # the stability drop are filed UNBACKED, and the matcher
        # proxies are backed by ground-truth label agreement (labels
        # the scorer never sees), not by the scorer's own output.
        assert METRIC_LEDGER["calibration_gap"].hard_backup is None
        assert METRIC_LEDGER["perturbation_stability"].hard_backup is None
        for matcher in ("contains", "regex", "json_schema"):
            assert (
                METRIC_LEDGER[matcher].hard_backup == "label_agreement"
            )
        assert (
            METRIC_LEDGER["p_conventional"].hard_backup
            == "conventional_share"
        )

    def test_quality_safety_name_the_matcher_dependence(self) -> None:
        # Round-2 B5: the dimension proportions inherit the soft
        # matchers' proxy error — the dependence is named, not hidden
        # behind the HARD label.
        for dimension in ("quality", "safety"):
            texts = " ".join(
                a.text for a in METRIC_LEDGER[dimension].assumptions
            )
            assert "soft matchers" in texts

    def test_reliability_names_the_replay_path(self) -> None:
        # Round-2 B5: decide replays recorded outputs, so the
        # phenomenon says replay-agreement, never live stability.
        phenomenon = METRIC_LEDGER["reliability"].phenomenon
        assert "replay-agreement" in phenomenon
        assert "ignores the seed" in " ".join(
            a.text for a in METRIC_LEDGER["reliability"].assumptions
        )

    def test_no_scorer_validity_entry(self) -> None:
        # Round-2 B3 (product): scorer_validity is a validity table,
        # not a printed number — auditing it would certify a number
        # the report never shows.
        assert "scorer_validity" not in METRIC_LEDGER

    def test_audit_summary_prints_assumption_texts(self) -> None:
        summary = audit_ledger().summary()
        for name in METRIC_LEDGER:
            assert name in summary
        assert "The scorer's matchers decide pass/fail exactly" in summary
        assert "metrics clean" in summary

    def test_audit_to_dict_carries_full_specs(self) -> None:
        data = audit_ledger().to_dict()
        assert data["all_ok"] is False
        assert data["flagged"] == [
            "aa_z",
            "alt_conservative_score",
            "calibration_gap",
            "guardrail_lower_bound",
            "oec_z",
            "perturbation_stability",
            "srm_chi2",
        ]
        assert len(data["findings"]) == len(METRIC_LEDGER)
        metrics = data["metrics"]
        assert metrics["pass_rate"]["assumptions"]
        assert metrics["pass_rate"]["phenomenon"]
        assert metrics["calibration_gap"]["kind"] == "soft"
        assert metrics["calibration_gap"]["hard_backup"] is None

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

    def test_parser_accepts_soft_without_backup(self) -> None:
        # Round 3: the parser accepts an unbacked soft spec — the
        # audit flags it, the parser does not refuse it.
        parsed = spec_from_dict(
            {
                "name": "vibes",
                "kind": "soft",
                "phenomenon": "how good it felt",
                "assumptions": [
                    {"text": "it felt good", "falsifier": "it felt bad"}
                ],
            }
        )
        assert parsed.hard_backup is None
        audit = audit_measured({"vibes": True}, ledger={"vibes": parsed})
        assert audit.flagged == ["vibes"]

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

    def test_parser_refuses_unknown_assumption_keys(self) -> None:
        with pytest.raises(ValueError, match="unknown keys"):
            spec_from_dict(
                {
                    "name": "x",
                    "kind": "hard",
                    "phenomenon": "a count",
                    "assumptions": [
                        {"text": "t", "falsifier": "f", "surprise": True},
                    ],
                }
            )

    def test_parser_refuses_duplicate_assumptions(self) -> None:
        with pytest.raises(ValueError, match="duplicate assumption"):
            spec_from_dict(
                {
                    "name": "x",
                    "kind": "hard",
                    "phenomenon": "a count",
                    "assumptions": [
                        {"text": "t", "falsifier": "f"},
                        {"text": "t", "falsifier": "f"},
                    ],
                }
            )

    def test_ledger_from_dicts_builds_shipped_ledger(self) -> None:
        payloads = [spec.to_dict() for spec in METRIC_LEDGER.values()]
        built = ledger_from_dicts(payloads)
        assert set(built) == set(METRIC_LEDGER)
        assert audit_ledger(built).flagged == [
            "aa_z",
            "alt_conservative_score",
            "calibration_gap",
            "guardrail_lower_bound",
            "oec_z",
            "perturbation_stability",
            "srm_chi2",
        ]
