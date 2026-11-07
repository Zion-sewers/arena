"""Tests for analysis pre-specification (agenteval-bench#48).

Style note: no mocks anywhere in this repo (see
test_no_mocks_on_boundaries.py) — fixtures are hand-built plans and
findings, registries are real temp files.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from experiment.prespec import (
    AnalysisFinding,
    AnalysisPlan,
    AnalysisRegistryCorrupted,
    AnalysisRegistryMalformed,
    AnalysisViolation,
    Assumption,
    check_analysis,
    find_analysis_plan,
    finding_from_dict,
    finding_to_dict,
    load_analysis_plan,
    plan_from_dict,
    plan_to_dict,
    register_analysis_plan,
    review_analysis,
)


def _assumption() -> Assumption:
    return Assumption(
        text="the eval suite's citation cases represent production traffic",
        falsifier="suite citation cases differ from production by >20pp",
    )


def _plan(**overrides: object) -> AnalysisPlan:
    kwargs: dict[str, object] = {
        "name": "reranker-v2-failure-modes",
        "mode": "confirmatory",
        "question": "Does reranker v2 reduce hallucinated citations?",
        "hypotheses": (
            "reranker v2 cuts hallucinated citations by at least 10pp vs v1",
        ),
        "assumptions": (_assumption(),),
        "n_planned": 200,
        "decision_rule": "confirm iff the 95% interval on the delta excludes 0",
        "data_source": "golden replay suite, seed 42",
    }
    kwargs.update(overrides)
    return AnalysisPlan(**kwargs)  # type: ignore[arg-type]


def _eplan(**overrides: object) -> AnalysisPlan:
    kwargs: dict[str, object] = {
        "name": "reranker-v2-explore",
        "mode": "exploratory",
        "question": "Which failure modes appear most in reranker v2 outputs?",
        "hypotheses": (),
        "assumptions": (_assumption(),),
        "n_planned": 200,
        "decision_rule": "",
        "data_source": "golden replay suite, seed 42",
    }
    kwargs.update(overrides)
    return AnalysisPlan(**kwargs)  # type: ignore[arg-type]


def _after(registered_at: str, hours: float = 1.0) -> str:
    parsed = datetime.fromisoformat(registered_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (parsed + timedelta(hours=hours)).isoformat()


def _finding(plan_digest: str, accessed: str, **overrides: object) -> AnalysisFinding:
    kwargs: dict[str, object] = {
        "plan_digest": plan_digest,
        "mode": "confirmatory",
        "hypothesis": "reranker v2 cuts hallucinated citations by at least 10pp vs v1",
        "claim": "v2 cuts hallucinated citations by 14pp",
        "n_observed": 200,
        "interval": (0.08, 0.20),
        "data_digest": "sha256:golden-suite-seed42",
        "data_accessed_at": accessed,
        "code_version": "arena@dbf6909",
    }
    kwargs.update(overrides)
    return AnalysisFinding(**kwargs)  # type: ignore[arg-type]


def _register(plan: AnalysisPlan, tmp_path: Path) -> tuple[str, str, str]:
    registry = str(tmp_path / "analyses.jsonl")
    digest = register_analysis_plan(plan, registry)
    _, registered_at = load_analysis_plan(digest, registry)
    return digest, registered_at, registry


# --- construction: the mode split is structural, not a label ---


def test_confirmatory_plan_without_hypothesis_refused() -> None:
    with pytest.raises(ValueError, match="hypothesis"):
        _plan(hypotheses=())


def test_confirmatory_plan_without_decision_rule_refused() -> None:
    with pytest.raises(ValueError, match="decision_rule"):
        _plan(decision_rule="")


def test_exploratory_plan_with_hypothesis_refused() -> None:
    with pytest.raises(ValueError, match="no hypotheses"):
        _eplan(hypotheses=("v2 is better",))


def test_exploratory_plan_with_decision_rule_refused() -> None:
    with pytest.raises(ValueError, match="no decision_rule"):
        _eplan(decision_rule="confirm iff better")


def test_plan_without_assumptions_refused() -> None:
    with pytest.raises(ValueError, match="assumption"):
        _plan(assumptions=())


def test_assumption_without_falsifier_refused() -> None:
    with pytest.raises(ValueError, match="falsifier"):
        Assumption(text="something", falsifier="")


def test_plan_below_min_n_refused() -> None:
    with pytest.raises(ValueError, match="below minimum"):
        _plan(n_planned=10)


def test_confirmatory_finding_without_interval_refused() -> None:
    with pytest.raises(ValueError, match="interval"):
        _finding("0" * 64, "2026-11-07T00:00:00+00:00", interval=None)


def test_confirmatory_finding_without_hypothesis_refused() -> None:
    with pytest.raises(ValueError, match="hypothesis"):
        _finding("0" * 64, "2026-11-07T00:00:00+00:00", hypothesis="")


def test_exploratory_finding_with_hypothesis_refused() -> None:
    with pytest.raises(ValueError, match="no hypothesis"):
        _finding(
            "0" * 64,
            "2026-11-07T00:00:00+00:00",
            mode="exploratory",
            hypothesis="v2 is better",
        )


def test_finding_interval_inverted_refused() -> None:
    with pytest.raises(ValueError, match="lower bound"):
        _finding("0" * 64, "2026-11-07T00:00:00+00:00", interval=(0.20, 0.08))


def test_finding_without_data_digest_refused() -> None:
    with pytest.raises(ValueError, match="data_digest"):
        _finding("0" * 64, "2026-11-07T00:00:00+00:00", data_digest="")


def test_finding_with_unparseable_timestamp_refused() -> None:
    with pytest.raises(ValueError, match="ISO timestamp"):
        _finding("0" * 64, "not-a-timestamp")


# --- registry: hash-chained, round-trips, tamper-evident ---


def test_register_load_round_trip(tmp_path: Path) -> None:
    plan = _plan()
    digest, _, registry = _register(plan, tmp_path)
    loaded, _ = load_analysis_plan(digest, registry)
    assert loaded == plan


def test_find_plan_matches_registered_only(tmp_path: Path) -> None:
    plan = _plan()
    digest, _, registry = _register(plan, tmp_path)
    assert find_analysis_plan(plan, registry) == digest
    assert find_analysis_plan(_plan(name="different-name"), registry) is None


def test_rewritten_registry_refused(tmp_path: Path) -> None:
    plan = _plan()
    digest, _, registry = _register(plan, tmp_path)
    path = Path(registry)
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("reranker-v2-failure-modes", "reranker-v9-failure-modes"),
        encoding="utf-8",
    )
    with pytest.raises(AnalysisRegistryCorrupted, match="hash chain broken"):
        load_analysis_plan(digest, registry)


def test_malformed_registry_record(tmp_path: Path) -> None:
    registry = str(tmp_path / "analyses.jsonl")
    Path(registry).write_text(json.dumps({"foo": 1}) + "\n", encoding="utf-8")
    with pytest.raises(AnalysisRegistryMalformed):
        load_analysis_plan("0" * 64, registry)


def test_unknown_digest_raises_key_error(tmp_path: Path) -> None:
    _, _, registry = _register(_plan(), tmp_path)
    with pytest.raises(KeyError):
        load_analysis_plan("f" * 64, registry)


# --- review gate: happy paths ---


def test_confirmatory_all_ten_pass(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_plan(), tmp_path)
    report = review_analysis(
        digest, registry, _finding(digest, _after(registered_at))
    )
    assert report.verdict == "CONFIRMED"
    assert len(report.questions) == 10
    assert [q.question_id for q in report.questions] == [f"Q{i}" for i in range(1, 11)]
    assert all(q.passed for q in report.questions)
    assert report.violations == ()


def test_exploratory_all_ten_pass_is_never_confirmed(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_eplan(), tmp_path)
    report = review_analysis(
        digest,
        registry,
        _finding(
            digest,
            _after(registered_at),
            mode="exploratory",
            hypothesis="",
            claim="citation errors cluster on long-context cases",
            interval=None,
        ),
    )
    assert report.verdict == "EXPLORATORY_ONLY"
    assert all(q.passed for q in report.questions)
    assert any("never a conclusion" in n for n in report.notes)


def test_exploratory_sample_tolerance_not_enforced(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_eplan(), tmp_path)
    report = review_analysis(
        digest,
        registry,
        _finding(
            digest,
            _after(registered_at),
            mode="exploratory",
            hypothesis="",
            claim="citation errors cluster on long-context cases",
            interval=None,
            n_observed=50,
        ),
    )
    assert report.verdict == "EXPLORATORY_ONLY"
    q6 = next(q for q in report.questions if q.question_id == "Q6")
    assert q6.passed


# --- review gate: refusals name the violation ---


def test_unregistered_plan_refused(tmp_path: Path) -> None:
    _, _, registry = _register(_plan(), tmp_path)
    report = review_analysis(
        "0" * 64,
        registry,
        _finding("0" * 64, "2026-11-07T00:00:00+00:00"),
    )
    assert report.verdict == "REFUSED"
    assert report.violations == ("UNREGISTERED_PLAN",)
    assert report.questions == ()


def test_malformed_registry_is_unreadable_not_unregistered(tmp_path: Path) -> None:
    registry = str(tmp_path / "analyses.jsonl")
    Path(registry).write_text(json.dumps({"foo": 1}) + "\n", encoding="utf-8")
    report = review_analysis(
        "0" * 64,
        registry,
        _finding("0" * 64, "2026-11-07T00:00:00+00:00"),
    )
    assert report.verdict == "REFUSED"
    assert report.violations == ("REGISTRY_UNREADABLE",)


def test_plan_after_data_refused(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_plan(), tmp_path)
    parsed = datetime.fromisoformat(registered_at)
    before = (parsed - timedelta(hours=1)).isoformat()
    report = review_analysis(digest, registry, _finding(digest, before))
    assert report.verdict == "REFUSED"
    assert "PLAN_AFTER_DATA" in report.violations
    q2 = next(q for q in report.questions if q.question_id == "Q2")
    assert not q2.passed


def test_post_hoc_reworded_hypothesis_refused(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_plan(), tmp_path)
    report = review_analysis(
        digest,
        registry,
        _finding(
            digest,
            _after(registered_at),
            hypothesis="reranker v2 cuts hallucinated citations a lot",
        ),
    )
    assert report.verdict == "REFUSED"
    assert "POST_HOC_CONFIRMATION" in report.violations
    q5 = next(q for q in report.questions if q.question_id == "Q5")
    assert not q5.passed


def test_mode_mismatch_refused(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_eplan(), tmp_path)
    report = review_analysis(
        digest,
        registry,
        _finding(
            digest,
            _after(registered_at),
            mode="confirmatory",
            hypothesis="citation errors cluster on long-context cases",
            claim="citation errors cluster on long-context cases",
        ),
    )
    assert report.verdict == "REFUSED"
    assert "MODE_MISMATCH" in report.violations
    assert "POST_HOC_CONFIRMATION" in report.violations


def test_sample_too_small_refused(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_plan(), tmp_path)
    report = review_analysis(
        digest, registry, _finding(digest, _after(registered_at), n_observed=10)
    )
    assert report.verdict == "REFUSED"
    assert "SAMPLE_TOO_SMALL" in report.violations


def test_confirmatory_sample_deviation_refused(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_plan(), tmp_path)
    report = review_analysis(
        digest, registry, _finding(digest, _after(registered_at), n_observed=150)
    )
    assert report.verdict == "REFUSED"
    assert "SAMPLE_DEVIATION" in report.violations


def test_confirmatory_sample_within_tolerance_passes(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_plan(), tmp_path)
    report = review_analysis(
        digest, registry, _finding(digest, _after(registered_at), n_observed=204)
    )
    assert report.verdict == "CONFIRMED"


def test_naive_timestamp_read_as_utc(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_plan(), tmp_path)
    parsed = datetime.fromisoformat(registered_at).replace(tzinfo=None)
    accessed = (parsed + timedelta(hours=1)).isoformat()
    report = review_analysis(digest, registry, _finding(digest, accessed))
    assert report.verdict == "CONFIRMED"


# --- check_analysis: the raising CI form ---


def test_check_analysis_raises_on_refusal(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_plan(), tmp_path)
    with pytest.raises(AnalysisViolation, match="POST_HOC_CONFIRMATION"):
        check_analysis(
            digest,
            registry,
            _finding(
                digest,
                _after(registered_at),
                hypothesis="a hypothesis nobody registered",
            ),
        )


def test_check_analysis_returns_report_on_pass(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_plan(), tmp_path)
    report = check_analysis(digest, registry, _finding(digest, _after(registered_at)))
    assert report.verdict == "CONFIRMED"


# --- review gate: fix-forward regressions (round-1 critiques) ---


def test_digest_mismatch_refused(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_plan(), tmp_path)
    report = review_analysis(
        digest,
        registry,
        _finding("0" * 64, _after(registered_at)),
    )
    assert report.verdict == "REFUSED"
    assert report.violations == ("DIGEST_MISMATCH",)
    assert report.questions == ()


def test_check_analysis_digest_mismatch(tmp_path: Path) -> None:
    digest, registered_at, registry = _register(_plan(), tmp_path)
    with pytest.raises(AnalysisViolation, match="DIGEST_MISMATCH"):
        check_analysis(
            digest,
            registry,
            _finding("0" * 64, _after(registered_at)),
        )


def test_registered_at_tamper_breaks_chain(tmp_path: Path) -> None:
    from experiment.prespec import AnalysisRegistryCorrupted

    digest, _, registry = _register(_plan(), tmp_path)
    path = Path(registry)
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    record["registered_at"] = "2020-01-01T00:00:00+00:00"
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(AnalysisRegistryCorrupted, match="hash chain broken"):
        load_analysis_plan(digest, registry)


def test_invalid_plan_record_is_unreadable_not_a_raise(tmp_path: Path) -> None:
    from experiment.prespec import _record_digest

    plan = _plan()
    canonical = dict(plan_to_dict(plan))
    canonical["mode"] = "bogus"
    registered_at = "2026-11-07T18:00:00+00:00"
    digest = _record_digest("GENESIS", canonical, registered_at)
    registry = str(tmp_path / "analyses.jsonl")
    record = {
        "digest": digest,
        "registered_at": registered_at,
        "analysis_plan": canonical,
    }
    Path(registry).write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    report = review_analysis(
        digest,
        registry,
        _finding(digest, "2026-11-07T19:00:00+00:00"),
    )
    assert report.verdict == "REFUSED"
    assert report.violations == ("REGISTRY_UNREADABLE",)


def test_finding_from_dict_rejects_short_interval() -> None:
    from experiment.prespec import finding_from_dict

    doc = {
        "plan_digest": "0" * 64,
        "mode": "confirmatory",
        "hypothesis": "h",
        "claim": "c",
        "n_observed": 200,
        "interval": [0.08],
        "data_digest": "sha256:x",
        "data_accessed_at": "2026-11-07T00:00:00+00:00",
        "code_version": "arena@dbf6909",
    }
    with pytest.raises(ValueError, match="invalid analysis finding doc"):
        finding_from_dict(doc)


def test_finding_from_dict_rejects_long_interval() -> None:
    from experiment.prespec import finding_from_dict

    doc = {
        "plan_digest": "0" * 64,
        "mode": "confirmatory",
        "hypothesis": "h",
        "claim": "c",
        "n_observed": 200,
        "interval": [0.08, 0.20, 0.30],
        "data_digest": "sha256:x",
        "data_accessed_at": "2026-11-07T00:00:00+00:00",
        "code_version": "arena@dbf6909",
    }
    with pytest.raises(ValueError, match="invalid analysis finding doc"):
        finding_from_dict(doc)


def test_finding_from_dict_rejects_coerced_sample_size() -> None:
    from experiment.prespec import finding_from_dict

    base = {
        "plan_digest": "0" * 64,
        "mode": "confirmatory",
        "hypothesis": "h",
        "claim": "c",
        "interval": [0.08, 0.20],
        "data_digest": "sha256:x",
        "data_accessed_at": "2026-11-07T00:00:00+00:00",
        "code_version": "arena@dbf6909",
    }
    with pytest.raises(ValueError, match="invalid analysis finding doc"):
        finding_from_dict({**base, "n_observed": 200.9})
    with pytest.raises(ValueError, match="invalid analysis finding doc"):
        finding_from_dict({**base, "n_observed": "200"})


# --- JSON round-trips for the CLI ---


def test_plan_json_round_trip() -> None:
    plan = _plan(experiment_digest="abc123")
    assert plan_from_dict(plan_to_dict(plan)) == plan


def test_finding_json_round_trip(tmp_path: Path) -> None:
    digest, registered_at, _registry = _register(_plan(), tmp_path)
    finding = _finding(digest, _after(registered_at))
    assert finding_from_dict(finding_to_dict(finding)) == finding
    assert finding_from_dict(finding_to_dict(finding)) != _finding(
        digest, _after(registered_at), n_observed=199
    )


def test_finding_from_dict_rejects_bad_doc() -> None:
    with pytest.raises(ValueError, match="invalid analysis finding doc"):
        finding_from_dict({"claim": "no fields at all"})


def test_plan_from_dict_rejects_bad_doc() -> None:
    with pytest.raises(ValueError, match="invalid analysis plan doc"):
        plan_from_dict({"name": "no fields at all"})


@pytest.mark.parametrize("doc", [["x"], "abc", 42, None, True])
def test_from_dict_rejects_non_object_json_doc(doc: object) -> None:
    """A JSON doc that is not an object is unusable input (exit 2 via the
    CLI), never a crash: .get on a list/str/int raises AttributeError,
    which no catch tuple covers, so the guard must fire first."""
    with pytest.raises(TypeError, match="invalid analysis plan doc"):
        plan_from_dict(doc)
    with pytest.raises(TypeError, match="invalid analysis finding doc"):
        finding_from_dict(doc)
