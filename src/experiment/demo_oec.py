"""Seeded end-to-end demo of the OEC ship gate (agenteval-bench#46).

Runs entirely on hand-built reported evidence (no randomness, no
network):

1. Registers a plan (n=800, alpha=0.05, looks at t=0.5 and t=1.0)
   and a design naming one OEC (task_success, higher_better) and one
   guardrail (safety_violations, lower_better, max regression 0.02).
   The design itself is registered in the hash chain before any
   evidence is graded.
2. Shows the compliant ship path: SRM clean, A/A clean, guardrail
   unregressed, OEC won at the final look (z=2.5, below the surprise
   threshold) — SHIP.
3. Violation drills: every refusal the issue demands is exercised and
   must fire — SRM failure blocks, A/A failure blocks, guardrail
   regression blocks, a surprising win without replication lands in
   NEEDS_REPLICATION, a surprising win with confirming replication
   ships, a trust-failing experiment with a huge OEC win still reports
   the trust failure (trust tests run before results are read), an
   OEC-switched report cannot ship, a replication that does not
   confirm blocks the ship, an unmeasured guardrail blocks, a design
   that was never registered is refused, a design rewritten after
   registration (wider guardrail budget) is refused, a zero-event
   guardrail on one sample is refused (the Newcombe interval fails
   closed on thin data), and an early stop at t=0.5 with a decisive
   look ships.

Exit 0 only if every expectation holds; any failure raises, so CI can
gate on this demo directly.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import replace

from experiment.oec import (
    AAEvidence,
    ExperimentDesign,
    MetricObservation,
    ReplicationEvidence,
    ShipEvidence,
    ShipRefused,
    check_ship,
    default_design,
    register_design,
    ship_decision,
)
from experiment.registry import (
    ExperimentEvidence,
    LookEvidence,
    register_plan,
)
from experiment.types import ExperimentPlan

#: Default A/A split and guardrail observation for the demo's base
#: evidence (module-level singletons: B008 forbids calls in defaults).
_DEFAULT_AA = AAEvidence(250, 500, 260, 500)
_DEFAULT_GUARDRAILS = (MetricObservation("safety_violations", 24, 400, 16, 400),)


def _base_evidence(
    digest: str,
    final_z: float = 2.5,
    final_rejected: bool = True,
    srm_counts: tuple[int, ...] = (398, 402),
    aa: AAEvidence | None = _DEFAULT_AA,
    guardrails: tuple[MetricObservation, ...] = _DEFAULT_GUARDRAILS,
    replication: ReplicationEvidence | None = None,
    oec_metric: str = "task_success",
    run_id: str = "run-1",
    n_final: int = 800,
    looks: tuple[LookEvidence, ...] | None = None,
    stopped_at: float | None = None,
) -> ShipEvidence:
    return ShipEvidence(
        plan_digest=digest,
        oec_metric=oec_metric,
        run_id=run_id,
        experiment=ExperimentEvidence(
            n_final=n_final,
            looks=(
                looks
                if looks is not None
                else (
                    LookEvidence(0.5, 1.0, False, metric="task_success"),
                    LookEvidence(1.0, final_z, final_rejected, metric="task_success"),
                )
            ),
            stopped_at=stopped_at,
        ),
        srm_counts=srm_counts,
        aa=aa,
        guardrails=guardrails,
        replication=replication,
    )


def _check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'ok' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"demo drill failed: {label} {detail}")


def main() -> int:
    print("OEC ship gate demo (agenteval-bench#46)")
    with tempfile.TemporaryDirectory() as tmp:
        registry = os.path.join(tmp, "plans.jsonl")
        plan = ExperimentPlan(name="reranker-v2-vs-v1", n_planned=800, alpha=0.05, looks=(0.5, 1.0))
        digest = register_plan(plan, registry)
        design = default_design(digest)
        assert isinstance(design, ExperimentDesign)
        design_digest = register_design(design, registry)
        _check("design registers in the hash chain", bool(design_digest))

        print("1. compliant path ships")
        verdict = ship_decision(design, _base_evidence(digest), registry)
        _check("compliant evidence ships", verdict.status == "ship", str(verdict.status))
        _check(
            "verdict records the registered design digest",
            verdict.design_digest == design_digest,
            verdict.design_digest,
        )
        _check(
            "check_ship passes the compliant path",
            check_ship(design, _base_evidence(digest), registry).status == "ship",
        )

        print("2. SRM failure blocks the ship")
        verdict = ship_decision(design, _base_evidence(digest, srm_counts=(350, 450)), registry)
        _check("SRM-broken evidence is NO_SHIP", verdict.status == "no_ship")
        _check(
            "reason names SRM",
            any("SRM_FAIL" in r for r in verdict.reasons),
            "; ".join(verdict.reasons),
        )

        print("3. A/A failure blocks the ship")
        verdict = ship_decision(
            design,
            _base_evidence(digest, aa=AAEvidence(200, 500, 300, 500)),
            registry,
        )
        _check("A/A-broken evidence is NO_SHIP", verdict.status == "no_ship")
        _check(
            "reason names A/A",
            any("AA_FAIL" in r for r in verdict.reasons),
            "; ".join(verdict.reasons),
        )

        print("4. guardrail regression blocks the ship")
        verdict = ship_decision(
            design,
            _base_evidence(
                digest,
                guardrails=(MetricObservation("safety_violations", 8, 400, 40, 400),),
            ),
            registry,
        )
        _check("regressed guardrail is NO_SHIP", verdict.status == "no_ship")
        _check(
            "reason names the guardrail",
            any("GUARDRAIL_REGRESSED" in r for r in verdict.reasons),
            "; ".join(verdict.reasons),
        )

        print("5. surprising win without replication needs replication")
        verdict = ship_decision(design, _base_evidence(digest, final_z=4.0), registry)
        _check(
            "surprising win is NEEDS_REPLICATION, not SHIP",
            verdict.status == "needs_replication",
            str(verdict.status),
        )
        _check(
            "reason carries the replication recipe",
            any("fresh run" in r for r in verdict.reasons),
            "; ".join(verdict.reasons),
        )

        print("6. surprising win with confirming replication ships")
        verdict = ship_decision(
            design,
            _base_evidence(
                digest,
                final_z=4.0,
                replication=ReplicationEvidence(z=2.5, n=400, run_id="run-2"),
            ),
            registry,
        )
        _check(
            "replicated surprise ships",
            verdict.status == "ship",
            "; ".join(verdict.reasons),
        )

        print("7. trust tests run before results are read")
        verdict = ship_decision(
            design,
            _base_evidence(digest, final_z=4.0, srm_counts=(350, 450)),
            registry,
        )
        _check("trust failure still NO_SHIP despite a huge win", verdict.status == "no_ship")
        _check(
            "reason names the trust failure, not the win",
            any("SRM_FAIL" in r for r in verdict.reasons),
            "; ".join(verdict.reasons),
        )

        print("8. OEC-switched evidence cannot ship")
        verdict = ship_decision(
            design,
            _base_evidence(digest, oec_metric="latency_p99"),
            registry,
        )
        _check("switched OEC is NO_SHIP", verdict.status == "no_ship")
        _check(
            "reason names OEC_MISMATCH",
            any("OEC_MISMATCH" in r for r in verdict.reasons),
            "; ".join(verdict.reasons),
        )

        print("9. a replication that does not confirm blocks the ship")
        verdict = ship_decision(
            design,
            _base_evidence(
                digest,
                final_z=4.0,
                replication=ReplicationEvidence(z=0.5, n=400, run_id="run-2"),
            ),
            registry,
        )
        _check("unconfirmed replication is NO_SHIP", verdict.status == "no_ship")
        _check(
            "reason names the failed confirmation",
            any("REPLICATION_NOT_CONFIRMED" in r for r in verdict.reasons),
            "; ".join(verdict.reasons),
        )

        print("10. an unmeasured guardrail blocks the ship")
        verdict = ship_decision(design, _base_evidence(digest, guardrails=()), registry)
        _check("unmeasured guardrail is NO_SHIP", verdict.status == "no_ship")
        _check(
            "reason names the unmeasured guardrail",
            any("GUARDRAIL_UNMEASURED" in r for r in verdict.reasons),
            "; ".join(verdict.reasons),
        )

        print("11. check_ship raises on a refused ship")
        try:
            check_ship(
                design,
                _base_evidence(digest, srm_counts=(350, 450)),
                registry,
            )
        except ShipRefused as e:
            _check("check_ship raises ShipRefused", "SRM_FAIL" in str(e), str(e))
        else:
            _check("check_ship raises ShipRefused", False, "no raise")

        print("12. a design that was never registered is refused")
        unregistered = replace(design, surprise_z=5.0)
        verdict = ship_decision(unregistered, _base_evidence(digest), registry)
        _check("unregistered design is NO_SHIP", verdict.status == "no_ship")
        _check(
            "reason names DESIGN_NOT_REGISTERED",
            any("DESIGN_NOT_REGISTERED" in r for r in verdict.reasons),
            "; ".join(verdict.reasons),
        )

        print("13. a design rewritten after registration is refused")
        rewritten = replace(
            design,
            metrics=(
                design.metrics[0],
                replace(design.metrics[1], max_regression=0.20),
            ),
        )
        verdict = ship_decision(rewritten, _base_evidence(digest), registry)
        _check("rewritten design is NO_SHIP", verdict.status == "no_ship")
        _check(
            "reason names DESIGN_NOT_REGISTERED",
            any("DESIGN_NOT_REGISTERED" in r for r in verdict.reasons),
            "; ".join(verdict.reasons),
        )

        print("14. a zero-event guardrail on one sample is refused")
        verdict = ship_decision(
            design,
            _base_evidence(
                digest,
                guardrails=(MetricObservation("safety_violations", 0, 1, 0, 1),),
            ),
            registry,
        )
        _check("zero-event guardrail on n=1 is NO_SHIP", verdict.status == "no_ship")
        _check(
            "reason names the guardrail",
            any("GUARDRAIL_REGRESSED" in r for r in verdict.reasons),
            "; ".join(verdict.reasons),
        )

        print("15. an early stop at t=0.5 with a decisive look ships")
        verdict = ship_decision(
            design,
            _base_evidence(
                digest,
                n_final=400,
                looks=(LookEvidence(0.5, 3.0, True, metric="task_success"),),
                stopped_at=0.5,
                srm_counts=(198, 202),
            ),
            registry,
        )
        _check("early-stop evidence ships", verdict.status == "ship", "; ".join(verdict.reasons))

    print("OEC demo: all 15 drills passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
