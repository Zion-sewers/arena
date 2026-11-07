"""Seeded end-to-end demo of analysis pre-specification (agenteval-bench#48).

Runs entirely on fixed strings and timestamps derived from the registry
itself — no randomness, no network:

1. The compliant confirmatory path: register a plan, report a finding
   that quotes a registered hypothesis verbatim with an interval, and
   the gate returns CONFIRMED with all ten questions passing.
2. The compliant exploratory path: same shape, no hypothesis, no
   interval required — the gate returns EXPLORATORY_ONLY, never
   CONFIRMED.
3. Refusal drills: a post-hoc hypothesis (reworded or unregistered) is
   refused as POST_HOC_CONFIRMATION; an exploratory finding presented
   under the confirmatory mode is refused as MODE_MISMATCH; a plan
   registered after the data was touched is refused as PLAN_AFTER_DATA;
   a confirmatory finding without an interval is refused at
   construction; a finding on an unregistered digest is refused as
   UNREGISTERED_PLAN; a finding naming a different digest than the one
   under review is refused as DIGEST_MISMATCH; a rewritten registry is
   refused as a broken chain, including a backdated registered_at.

Exit 0 only if every expectation holds; any failure raises, so CI can
gate on this demo directly.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta

from experiment.prespec import (
    AnalysisFinding,
    AnalysisPlan,
    AnalysisRegistryCorrupted,
    AnalysisViolation,
    Assumption,
    check_analysis,
    find_analysis_plan,
    register_analysis_plan,
    review_analysis,
)


def _after(registered_at: str, hours: float = 1.0) -> str:
    """An ISO timestamp safely after a registered_at timestamp."""
    parsed = datetime.fromisoformat(registered_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (parsed + timedelta(hours=hours)).isoformat()


def _before(registered_at: str, hours: float = 1.0) -> str:
    parsed = datetime.fromisoformat(registered_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (parsed - timedelta(hours=hours)).isoformat()


def _confirmatory_plan() -> AnalysisPlan:
    return AnalysisPlan(
        name="reranker-v2-failure-modes",
        mode="confirmatory",
        question="Does reranker v2 reduce the hallucinated-citation failure mode?",
        hypotheses=(
            "reranker v2 cuts hallucinated citations by at least 10pp vs v1",
        ),
        assumptions=(
            Assumption(
                text="the eval suite's citation cases represent production traffic",
                falsifier="suite citation cases differ from production by >20pp",
            ),
        ),
        n_planned=200,
        decision_rule="confirm iff the 95% interval on the citation-error "
        "delta excludes 0 in v2's favour",
        data_source="golden replay suite, seed 42",
    )


def _exploratory_plan() -> AnalysisPlan:
    return AnalysisPlan(
        name="reranker-v2-failure-modes-explore",
        mode="exploratory",
        question="Which failure modes appear most in reranker v2 outputs?",
        hypotheses=(),
        assumptions=(
            Assumption(
                text="the eval suite's cases cover the failure modes of interest",
                falsifier="a production failure mode has no suite case",
            ),
        ),
        n_planned=200,
        decision_rule="",
        data_source="golden replay suite, seed 42",
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        registry = os.path.join(tmp, "plans", "analyses.jsonl")

        # 1. Compliant confirmatory path -> CONFIRMED, all ten pass.
        plan = _confirmatory_plan()
        digest = register_analysis_plan(plan, registry)
        assert find_analysis_plan(plan, registry) == digest
        from experiment.prespec import load_analysis_plan

        _, registered_at = load_analysis_plan(digest, registry)
        accessed = _after(registered_at)
        report = review_analysis(
            digest,
            registry,
            AnalysisFinding(
                plan_digest=digest,
                mode="confirmatory",
                hypothesis="reranker v2 cuts hallucinated citations by at least 10pp vs v1",
                claim="v2 cuts hallucinated citations by 14pp",
                n_observed=200,
                interval=(0.08, 0.20),
                data_digest="sha256:golden-suite-seed42",
                data_accessed_at=accessed,
                code_version="arena@dbf6909",
            ),
        )
        assert report.verdict == "CONFIRMED", report
        assert len(report.questions) == 10, report
        assert all(q.passed for q in report.questions), report
        assert not report.violations, report

        # 2. Compliant exploratory path -> EXPLORATORY_ONLY, never CONFIRMED.
        eplan = _exploratory_plan()
        edigest = register_analysis_plan(eplan, registry)
        _, eregistered_at = load_analysis_plan(edigest, registry)
        ereport = review_analysis(
            edigest,
            registry,
            AnalysisFinding(
                plan_digest=edigest,
                mode="exploratory",
                hypothesis="",
                claim="citation errors cluster on long-context cases",
                n_observed=200,
                interval=None,
                data_digest="sha256:golden-suite-seed42",
                data_accessed_at=_after(eregistered_at),
                code_version="arena@dbf6909",
            ),
        )
        assert ereport.verdict == "EXPLORATORY_ONLY", ereport
        assert all(q.passed for q in ereport.questions), ereport

        # 3a. Post-hoc hypothesis: reworded, unregistered -> refused.
        posthoc = review_analysis(
            digest,
            registry,
            AnalysisFinding(
                plan_digest=digest,
                mode="confirmatory",
                hypothesis="reranker v2 cuts hallucinated citations a lot",
                claim="v2 cuts hallucinated citations a lot",
                n_observed=200,
                interval=(0.08, 0.20),
                data_digest="sha256:golden-suite-seed42",
                data_accessed_at=accessed,
                code_version="arena@dbf6909",
            ),
        )
        assert posthoc.verdict == "REFUSED", posthoc
        assert "POST_HOC_CONFIRMATION" in posthoc.violations, posthoc
        assert not any(q.passed for q in posthoc.questions if q.question_id == "Q5"), posthoc

        # 3b. Exploratory finding presented as confirmatory -> refused.
        relabeled = review_analysis(
            edigest,
            registry,
            AnalysisFinding(
                plan_digest=edigest,
                mode="confirmatory",
                hypothesis="citation errors cluster on long-context cases",
                claim="citation errors cluster on long-context cases",
                n_observed=200,
                interval=(0.01, 0.05),
                data_digest="sha256:golden-suite-seed42",
                data_accessed_at=_after(eregistered_at),
                code_version="arena@dbf6909",
            ),
        )
        assert relabeled.verdict == "REFUSED", relabeled
        assert "MODE_MISMATCH" in relabeled.violations, relabeled

        # 3c. Plan registered after the data was touched -> refused.
        late = review_analysis(
            digest,
            registry,
            AnalysisFinding(
                plan_digest=digest,
                mode="confirmatory",
                hypothesis="reranker v2 cuts hallucinated citations by at least 10pp vs v1",
                claim="v2 cuts hallucinated citations by 14pp",
                n_observed=200,
                interval=(0.08, 0.20),
                data_digest="sha256:golden-suite-seed42",
                data_accessed_at=_before(registered_at),
                code_version="arena@dbf6909",
            ),
        )
        assert late.verdict == "REFUSED", late
        assert "PLAN_AFTER_DATA" in late.violations, late

        # 3d. Confirmatory finding without an interval: refused at construction.
        try:
            AnalysisFinding(
                plan_digest=digest,
                mode="confirmatory",
                hypothesis="reranker v2 cuts hallucinated citations by at least 10pp vs v1",
                claim="v2 cuts hallucinated citations by 14pp",
                n_observed=200,
                interval=None,
                data_digest="sha256:golden-suite-seed42",
                data_accessed_at=accessed,
                code_version="arena@dbf6909",
            )
        except ValueError as e:
            assert "interval" in str(e), e
        else:
            raise AssertionError("confirmatory finding without interval constructed")

        # 3e. Finding on an unregistered digest -> refused.
        unknown = review_analysis(
            "0" * 64,
            registry,
            AnalysisFinding(
                plan_digest="0" * 64,
                mode="confirmatory",
                hypothesis="reranker v2 cuts hallucinated citations by at least 10pp vs v1",
                claim="v2 cuts hallucinated citations by 14pp",
                n_observed=200,
                interval=(0.08, 0.20),
                data_digest="sha256:golden-suite-seed42",
                data_accessed_at=accessed,
                code_version="arena@dbf6909",
            ),
        )
        assert unknown.verdict == "REFUSED", unknown
        assert "UNREGISTERED_PLAN" in unknown.violations, unknown

        # 3f. A rewritten registry is refused as a broken chain.
        with open(registry, encoding="utf-8") as f:
            lines = f.readlines()
        tampered = lines[0].replace("reranker-v2-failure-modes", "reranker-v9-failure-modes")
        with open(registry, "w", encoding="utf-8") as f:
            f.write(tampered + "".join(lines[1:]))
        try:
            load_analysis_plan(digest, registry)
        except AnalysisRegistryCorrupted:
            pass
        else:
            raise AssertionError("rewritten registry loaded clean")
        rewritten = review_analysis(
            digest,
            registry,
            AnalysisFinding(
                plan_digest=digest,
                mode="confirmatory",
                hypothesis="reranker v2 cuts hallucinated citations by at least 10pp vs v1",
                claim="v2 cuts hallucinated citations by 14pp",
                n_observed=200,
                interval=(0.08, 0.20),
                data_digest="sha256:golden-suite-seed42",
                data_accessed_at=accessed,
                code_version="arena@dbf6909",
            ),
        )
        assert rewritten.verdict == "REFUSED", rewritten
        assert "UNREGISTERED_PLAN" in rewritten.violations, rewritten

        # Fresh registry for the raising-form checks below; the 3g drill
        # runs against it first.
        registry2 = os.path.join(tmp, "plans2.jsonl")
        digest2 = register_analysis_plan(_confirmatory_plan(), registry2)
        from experiment.prespec import load_analysis_plan as _load

        _, registered_at2 = _load(digest2, registry2)

        # 3g. A finding naming a different digest than the one under
        # review is refused as DIGEST_MISMATCH — even when the named
        # digest is unregistered: the plan under review loads first, then
        # the binding is checked against the finding's digest.
        mismatch = review_analysis(
            digest2,
            registry2,
            AnalysisFinding(
                plan_digest="0" * 64,
                mode="confirmatory",
                hypothesis="reranker v2 cuts hallucinated citations by at least 10pp vs v1",
                claim="v2 cuts hallucinated citations by 14pp",
                n_observed=200,
                interval=(0.08, 0.20),
                data_digest="sha256:golden-suite-seed42",
                data_accessed_at=_after(registered_at2),
                code_version="arena@dbf6909",
            ),
        )
        assert mismatch.verdict == "REFUSED", mismatch
        assert "DIGEST_MISMATCH" in mismatch.violations, mismatch

        # 3h. Backdating registered_at breaks the chain: the timestamp
        # is part of the hashed payload.
        registry3 = os.path.join(tmp, "plans3.jsonl")
        digest3 = register_analysis_plan(_confirmatory_plan(), registry3)
        with open(registry3, encoding="utf-8") as f:
            lines3 = f.readlines()
        rec3 = json.loads(lines3[0])
        rec3["registered_at"] = "2020-01-01T00:00:00+00:00"
        with open(registry3, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec3, sort_keys=True) + "\n")
        try:
            load_analysis_plan(digest3, registry3)
        except AnalysisRegistryCorrupted:
            pass
        else:
            raise AssertionError("backdated registered_at loaded clean")

        # The raising form: check_analysis raises on refusal, returns on pass.
        ok = check_analysis(
            digest2,
            registry2,
            AnalysisFinding(
                plan_digest=digest2,
                mode="confirmatory",
                hypothesis="reranker v2 cuts hallucinated citations by at least 10pp vs v1",
                claim="v2 cuts hallucinated citations by 14pp",
                n_observed=200,
                interval=(0.08, 0.20),
                data_digest="sha256:golden-suite-seed42",
                data_accessed_at=_after(registered_at2),
                code_version="arena@dbf6909",
            ),
        )
        assert ok.verdict == "CONFIRMED", ok
        try:
            check_analysis(
                digest2,
                registry2,
                AnalysisFinding(
                    plan_digest=digest2,
                    mode="confirmatory",
                    hypothesis="some other hypothesis entirely",
                    claim="v2 is better somehow",
                    n_observed=200,
                    interval=(0.08, 0.20),
                    data_digest="sha256:golden-suite-seed42",
                    data_accessed_at=_after(registered_at2),
                    code_version="arena@dbf6909",
                ),
            )
        except AnalysisViolation as e:
            assert "POST_HOC_CONFIRMATION" in str(e), e
        else:
            raise AssertionError("check_analysis passed a post-hoc finding")

    print("prespec demo: CONFIRMED + EXPLORATORY_ONLY paths verify; "
          "all refusal drills fire (POST_HOC_CONFIRMATION, MODE_MISMATCH, "
          "PLAN_AFTER_DATA, UNREGISTERED_PLAN, DIGEST_MISMATCH, broken chain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
