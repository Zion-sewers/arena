"""CLI entry point for agenteval-bench."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from agenteval_bench.decision import DecisionPolicy, RiskLevel
from agenteval_bench.engine import DEFAULT_SEED, EvalRunner
from agenteval_bench.models import EvalSuite
from agenteval_bench.replay import ReplayCase, ReplayLog, find_mismatches, suite_digest


def _flag(args: list[str], name: str, default: str | None = None) -> tuple[list[str], str | None]:
    """Pop ``--name value`` from args, returning (rest, value)."""
    rest: list[str] = []
    value = default
    i = 0
    while i < len(args):
        if args[i] == name and i + 1 < len(args):
            value = args[i + 1]
            i += 2
        else:
            rest.append(args[i])
            i += 1
    return rest, value


def _has(args: list[str], name: str) -> tuple[list[str], bool]:
    rest = [a for a in args if a != name]
    return rest, len(rest) != len(args)


def _load_suite_for_replay(suite_path: str):
    suite = EvalSuite.from_yaml(suite_path)
    with open(suite_path, "rb") as f:
        digest = suite_digest(f.read())
    recorded = {c.id: c.output for c in suite.cases}
    missing = [c.id for c in suite.cases if c.output is None and not c.skip]
    for c in suite.cases:
        if c.output is None and not c.skip:
            c.skip = True
    return suite, recorded, missing, digest


def _run_suite(suite: EvalSuite, recorded: dict[str, str], seed: int):
    """Run a replay suite (recorded outputs only) and return (result, outputs)."""
    replay_ids = iter([c.id for c in suite.cases if not c.skip])
    outputs: dict[str, str] = {}

    def replay_fn(_input: str) -> str:
        cid = next(replay_ids)
        out = recorded.get(cid) or ""
        outputs[cid] = out
        return out

    runner = EvalRunner()
    return runner.run(suite, replay_fn, seed=seed), outputs


def _build_log(suite: EvalSuite, suite_path: str, digest: str,
               result, outputs: dict[str, str], seed: int) -> ReplayLog:
    by_id = {r.case_id: r for r in result.results}
    cases = []
    for c in suite.cases:
        r = by_id[c.id]
        cases.append(ReplayCase(
            case_id=c.id,
            input=c.input,
            output=outputs.get(c.id, "") if not r.details.get("skipped") else "",
            score=r.score,
            passed=r.passed,
            skipped=bool(r.details.get("skipped")),
            rng_draw=r.details.get("rng_draw"),
        ))
    return ReplayLog(
        seed=seed,
        suite_name=suite.name,
        suite_file=os.path.basename(suite_path),
        digest=digest,
        cases=cases,
    )


def cmd_run(args: list[str]) -> int:
    args, suite_path = _flag(args, "--suite")
    args, ci_mode = _has(args, "--ci")
    args, threshold_s = _flag(args, "--threshold", "1.0")
    args, seed_s = _flag(args, "--seed", str(DEFAULT_SEED))
    args, replay_log = _flag(args, "--replay-log")
    args, labels_path = _flag(args, "--validity-labels")

    if not suite_path:
        print("Error: --suite <file> is required", file=sys.stderr)
        return 1

    try:
        threshold = float(threshold_s or "1.0")
    except ValueError:
        print(f"Error: --threshold must be a number, got {threshold_s}", file=sys.stderr)
        return 1
    if not 0.0 <= threshold <= 1.0:
        print(f"Error: --threshold must be between 0.0 and 1.0, got {threshold}", file=sys.stderr)
        return 1
    try:
        seed = int(seed_s or str(DEFAULT_SEED))
    except ValueError:
        print(f"Error: --seed must be an integer, got {seed_s}", file=sys.stderr)
        return 1

    suite, recorded, missing, digest = _load_suite_for_replay(suite_path)
    # Standalone CLI scores a replay suite: each case carries a recorded
    # `output`. Cases without one are skipped (a live agent would fill them
    # via the Python API). This is what CI runs against a golden set.
    result, outputs = _run_suite(suite, recorded, seed)
    print(result.summary())
    if missing:
        print(f"Note: {len(missing)} case(s) had no recorded output and were skipped.")

    # Every metric carries a measurement-reliability column (arena#43):
    # the run report prints the scorer's validity table next to the pass
    # rate, so a 100% pass rate never reads as trusted on its own. With
    # --validity-labels, the printed grade for that matcher is a measured
    # validation against ground truth, not the built-in note.
    from agenteval_bench.validity import assess_scorer, validate_matcher_labels

    scorer_report = assess_scorer()
    if labels_path:
        try:
            measured = validate_matcher_labels(labels_path)
        except (OSError, ValueError, TypeError) as exc:
            print(
                f"Error: cannot load validity labels {labels_path}: {exc}",
                file=sys.stderr,
            )
            return 1
        scorer_report = assess_scorer(notes={measured.metric: measured})
        print(f"Validity labels: {labels_path} — {measured.metric} measured: {measured.basis}")
    print(scorer_report.summary())

    if replay_log:
        log = _build_log(suite, suite_path, digest, result, outputs, seed)
        log.write(replay_log)
        print(f"Replay log: {replay_log} (seed={seed})")

    if ci_mode:
        print(
            "Warning: --ci is a legacy point-estimate gate (pass_rate vs "
            "threshold, no dimensions or uncertainty). Superseded by "
            "`agenteval-bench decide`; see docs/DECISION-SYSTEM.md."
        )
        ok = result.pass_rate >= threshold
        print(f"CI gate: pass_rate {result.pass_rate:.1%} vs threshold {threshold:.0%} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 2
    return 0


def _policy_number(value: object, field_name: str, level_name: str) -> float:
    """One threshold value from a policy file: finite number, never a bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"Invalid {field_name} for risk level {level_name!r}: "
            f"expected a number, got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"Invalid {field_name} for risk level {level_name!r}: "
            f"expected a finite number, got {value!r}"
        )
    return number


def load_policy(policy_path: str | None) -> tuple[DecisionPolicy, RiskLevel]:
    """Load a decision policy YAML, filling unspecified levels from defaults.

    Policy file shape (see examples/decision-policy.yaml):
        risk: medium
        thresholds:
          low: {quality: 0.7, safety: 0.8, reliability: 0.7, cost_budget: 100.0}
          medium: {...}
          high: {...}
          critical: {...}
    Returns (policy, default_risk). A missing file means the default
    policy at medium risk. Unknown keys and non-numeric values are
    rejected: a typo'd threshold must fail loudly, not silently fall
    back to a default the operator did not intend.
    """
    from agenteval_bench.decision import RiskThresholds, default_policy

    policy = default_policy()
    risk = RiskLevel.MEDIUM
    if policy_path is None:
        return policy, risk
    import yaml

    try:
        with open(policy_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid policy {policy_path}: malformed YAML ({exc})") from exc
    if not isinstance(data, dict):
        raise TypeError(f"Invalid policy {policy_path}: expected a mapping")
    unknown = sorted(set(data) - {"risk", "thresholds"})
    if unknown:
        raise ValueError(f"Unknown top-level keys in {policy_path}: {unknown}")
    if "risk" in data:
        try:
            risk = RiskLevel(str(data["risk"]))
        except ValueError:
            raise ValueError(
                f"Invalid risk {data['risk']!r} in {policy_path}: "
                f"expected one of {[r.value for r in RiskLevel]}"
            ) from None
    overrides = data.get("thresholds", {})
    if not isinstance(overrides, dict):
        raise TypeError(f"Invalid thresholds in {policy_path}: expected a mapping")
    thresholds = dict(policy.thresholds)
    for level_name, values in overrides.items():
        try:
            level = RiskLevel(str(level_name))
        except ValueError:
            raise ValueError(
                f"Invalid risk level {level_name!r} in {policy_path}: "
                f"expected one of {[r.value for r in RiskLevel]}"
            ) from None
        if not isinstance(values, dict):
            raise TypeError(f"Invalid thresholds for {level_name!r}: expected a mapping")
        unknown_fields = sorted(
            set(values) - {"quality", "safety", "reliability", "cost_budget"}
        )
        if unknown_fields:
            raise ValueError(
                f"Unknown threshold fields for {level_name!r} in {policy_path}: "
                f"{unknown_fields}"
            )
        base = thresholds[level]
        thresholds[level] = RiskThresholds(
            quality=_policy_number(values.get("quality", base.quality), "quality", level_name),
            safety=_policy_number(values.get("safety", base.safety), "safety", level_name),
            reliability=_policy_number(
                values.get("reliability", base.reliability), "reliability", level_name
            ),
            cost_budget=_policy_number(
                values.get("cost_budget", base.cost_budget), "cost_budget", level_name
            ),
        )
    return DecisionPolicy(thresholds=thresholds), risk


def cmd_decide(args: list[str]) -> int:
    """Map a replay suite to a decision: evidence -> uncertainty -> threshold -> action.

    Exit codes: 0 GATE, 1 ROLLBACK, 2 INVESTIGATE, 3 usage error.
    Scores a replay suite (recorded outputs + recorded per-case costs),
    so the decision is reproducible in CI without a live agent.

    Reliability in this mode is replay-agreement: the recorded outputs
    are re-scored across runs, so the runs cannot disagree the way a
    live stochastic agent's runs can. The summary and the report label
    it as such; it must not be read as live-agent reliability.

    --strict-validity downgrades a GATE to INVESTIGATE when any metric in
    the validity report is untrusted. Default off: replay mode cannot
    trust reliability by construction, so a strict default would make
    `decide` unable to gate at all.
    """
    from agenteval_bench.decision import (
        CostEvidence,
        DecisionAction,
        Dimension,
        ProportionEvidence,
        RiskLevel,
        UnmeasuredEvidence,
        policy_fingerprint,
    )
    from agenteval_bench.validity import assess_decision, assess_scorer

    args, suite_path = _flag(args, "--suite")
    args, policy_path = _flag(args, "--policy")
    args, risk_s = _flag(args, "--risk")
    args, seed_s = _flag(args, "--seed", str(DEFAULT_SEED))
    args, reliability_runs_s = _flag(args, "--reliability-runs", "2")
    args, out = _flag(args, "--out")
    args, strict_validity = _has(args, "--strict-validity")
    if not suite_path:
        print("Error: decide needs --suite <file>", file=sys.stderr)
        return 3
    try:
        seed = int(seed_s or str(DEFAULT_SEED))
    except ValueError:
        print(f"Error: --seed must be an integer, got {seed_s}", file=sys.stderr)
        return 3
    try:
        reliability_runs = int(reliability_runs_s or "2")
    except ValueError:
        print(
            f"Error: --reliability-runs must be an integer, got {reliability_runs_s}",
            file=sys.stderr,
        )
        return 3
    if reliability_runs < 1:
        print("Error: --reliability-runs must be >= 1", file=sys.stderr)
        return 3
    try:
        policy, file_risk = load_policy(policy_path)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    risk = file_risk
    if risk_s is not None:
        try:
            risk = RiskLevel(risk_s)
        except ValueError:
            print(
                f"Error: --risk must be one of {[r.value for r in RiskLevel]}, got {risk_s}",
                file=sys.stderr,
            )
            return 3

    import yaml

    try:
        suite, _recorded, missing, digest = _load_suite_for_replay(suite_path)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        print(f"Error: cannot load suite {suite_path}: {exc}", file=sys.stderr)
        return 3
    if missing:
        print(f"Note: {len(missing)} case(s) had no recorded output and were skipped.")
    inputs = [c.input for c in suite.cases if not c.skip]
    if len(set(inputs)) != len(inputs):
        print(
            "Error: decide replay needs unique case inputs (the replay agent "
            "is keyed by input); duplicate inputs found",
            file=sys.stderr,
        )
        return 3
    by_input = {c.input: c.output or "" for c in suite.cases if not c.skip}

    def replay_fn(prompt: str) -> str:
        return by_input[prompt]

    runner = EvalRunner()
    try:
        decision = runner.run_decision(
            suite,
            replay_fn,
            policy,
            risk,
            cost_fn=lambda case, _output: case.cost,
            reliability_runs=reliability_runs,
            seed=seed,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    print(decision.summary())
    # Every metric carries a measurement-reliability column (arena#43):
    # this mode replays recorded outputs, so reliability is
    # replay-agreement, graded UNVALIDATED_PROXY — never as live stability.
    validity = assess_decision(decision, reliability_kind="replay")
    print(validity.summary())
    # The decision grades quality/safety DIRECT for the proportion, but the
    # verdicts underneath rest on the scorer's matchers: the scorer report
    # is printed in the same output, so the two grades are never read apart.
    scorer_report = assess_scorer()
    print(scorer_report.summary())
    # --strict-validity is the enforcement switch: a GATE that rests on an
    # untrusted metric is downgraded to INVESTIGATE. Default off, because
    # replay mode structurally cannot trust reliability (see above), so a
    # strict default would make `decide` unable to gate at all; the switch
    # exists for live-assessed decisions and for callers that want the
    # validity gate to bite. The evidence action is recorded, never hidden.
    action = decision.action
    evidence_action = None
    if strict_validity and action is DecisionAction.GATE and not validity.all_trusted:
        evidence_action = action
        action = DecisionAction.INVESTIGATE
        print(
            "Validity gate (strict): GATE downgraded to INVESTIGATE — "
            f"untrusted: {', '.join(validity.untrusted_metrics())}"
        )
    print(
        "Note: reliability here is replay-agreement (recorded outputs "
        "re-scored across runs); it cannot detect live-agent flakiness."
    )
    if out:
        report: dict[str, Any] = {
            "action": action.value,
            "risk": decision.risk.value,
            "suite": suite.name,
            "suite_digest": digest,
            "seed": seed,
            "reliability_runs": reliability_runs,
            "policy_fingerprint": policy_fingerprint(policy),
            "validity": validity.to_dict(),
            "scorer_validity": scorer_report.to_dict(),
            "dimensions": {},
        }
        if evidence_action is not None:
            report["evidence_action"] = evidence_action.value
        for dimension, verdict in decision.verdicts.items():
            evidence = verdict.evidence
            if isinstance(evidence, UnmeasuredEvidence):
                entry: dict[str, Any] = {
                    "kind": "unmeasured",
                    "action": verdict.action.value,
                    "point": None,
                    "lower": None,
                    "upper": None,
                    "threshold": verdict.threshold,
                    "reason": verdict.reason,
                }
            elif isinstance(evidence, CostEvidence):
                entry = {
                    "kind": "measured",
                    "action": verdict.action.value,
                    "point": evidence.point,
                    "lower": evidence.lower,
                    "upper": evidence.upper,
                    "n": evidence.n,
                    "threshold": verdict.threshold,
                    "reason": verdict.reason,
                }
            elif isinstance(evidence, ProportionEvidence):
                entry = {
                    "kind": (
                        "replay-agreement"
                        if dimension is Dimension.RELIABILITY
                        else "measured"
                    ),
                    "action": verdict.action.value,
                    "point": evidence.point,
                    "lower": evidence.lower,
                    "upper": evidence.upper,
                    "successes": evidence.successes,
                    "trials": evidence.trials,
                    "threshold": verdict.threshold,
                    "reason": verdict.reason,
                }
                if dimension is Dimension.RELIABILITY:
                    entry["reason"] = (
                        "replay-agreement: recorded outputs re-scored across "
                        "runs; cannot detect live-agent flakiness"
                    )
            else:  # pragma: no cover - exhaustive union, kept for mypy
                raise TypeError(f"unknown evidence type {type(evidence).__name__}")
            report["dimensions"][dimension.value] = entry
        try:
            with open(out, "w") as f:
                json.dump(report, f, indent=2)
        except OSError as exc:
            print(f"Error: cannot write report {out}: {exc}", file=sys.stderr)
            return 3
        print(f"Decision report: {out}")
    if action is DecisionAction.GATE:
        return 0
    if action is DecisionAction.ROLLBACK:
        return 1
    return 2


def cmd_baseline(args: list[str]) -> int:
    """Freeze a clean-baseline snapshot: per-case scores pinned to seed + suite digest."""
    args, suite_path = _flag(args, "--suite")
    args, seed_s = _flag(args, "--seed", str(DEFAULT_SEED))
    args, out = _flag(args, "--out")
    if not suite_path or not out:
        print("Error: baseline needs --suite <file> --out baseline.json", file=sys.stderr)
        return 1
    try:
        seed = int(seed_s or str(DEFAULT_SEED))
    except ValueError:
        print(f"Error: --seed must be an integer, got {seed_s}", file=sys.stderr)
        return 1

    suite, recorded, _missing, digest = _load_suite_for_replay(suite_path)
    result, _ = _run_suite(suite, recorded, seed)
    scores = [
        {"case_id": r.case_id, "score": r.score, "passed": r.passed}
        for r in result.results if not r.details.get("skipped")
    ]
    doc = {
        "tool": "agenteval-bench",
        "snapshot_version": 1,
        "seed": seed,
        "suite_name": suite.name,
        "suite_file": os.path.basename(suite_path),
        "suite_digest": digest,
        "scorer": "DeterministicScorer",
        "scores": scores,
    }
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, sort_keys=True, indent=2) + "\n")
    print(f"Baseline snapshot: {len(scores)} scores (seed={seed}) -> {out}")
    return 0


def cmd_alt_test(args: list[str]) -> int:
    """Run the Alternative Annotator Test on a recorded annotation dataset.

    Judge labels are recorded data (like replay outputs), never live LLM
    calls. Prints the winning-rate leaderboard; --out writes the JSON report.
    """
    args, data_path = _flag(args, "--data")
    args, epsilon_s = _flag(args, "--epsilon", "0.1")
    args, q_s = _flag(args, "--q", "0.05")
    args, seed_s = _flag(args, "--seed", str(DEFAULT_SEED))
    args, out = _flag(args, "--out")

    if not data_path:
        print("Error: --data <file.jsonl> is required", file=sys.stderr)
        return 1
    try:
        epsilon = float(epsilon_s or "0.1")
        q = float(q_s or "0.05")
        seed = int(seed_s or str(DEFAULT_SEED))
    except ValueError:
        print("Error: --epsilon/--q must be numbers, --seed an integer", file=sys.stderr)
        return 1

    from alt_test.runner import (
        InsufficientCoverage,
        dataset_digest,
        load_jsonl,
        render_text,
        run_alt_test,
    )
    from alt_test.types import AltTestConfig

    try:
        config = AltTestConfig(epsilon=epsilon, q=q, seed=seed)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    try:
        items = load_jsonl(data_path)
    except (OSError, ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if items and items[0].task == "text":
        print(
            "Note: text task uses the deterministic trigram-Jaccard similarity "
            "surrogate (see docs/ALT-TEST.md); inject a custom sim via the "
            "Python API for embedding-based SIM.",
            file=sys.stderr,
        )
    try:
        report = run_alt_test(items, config, input_digest=dataset_digest(items))
    except (OSError, ValueError, TypeError, InsufficientCoverage) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(render_text(report))
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
        print(f"Report: {out}")
    return 0


def cmd_simpson(args: list[str]) -> int:
    """Rule on a "treated beats control overall" claim from per-slice aggregates.

    Reads a JSON doc of per-stratum counts (periods, segments, task
    categories), prints the mandatory per-stratum disaggregation, and
    exits 0 only if the pooled win claim survives Simpson-safe
    aggregation (agenteval-bench#36). Exit 1 = claim REJECTED or pooled
    reporting refused (violations named); exit 2 = unusable input.

    Input schema:
      {"strata": [{"name", "n_treated", "n_control",
                   "treated_successes", "control_successes"}],
       "pooled_n_treated": int, "pooled_n_control": int,
       "allocation_tolerance": float (optional)}
    """
    from experiment.simpson import (
        DEFAULT_ALLOCATION_TOLERANCE,
        Stratum,
        win_claim,
    )

    args, slices_path = _flag(args, "--slices")
    args, tol_s = _flag(args, "--allocation-tolerance")
    args, out = _flag(args, "--out")
    if not slices_path:
        print("Error: simpson needs --slices <file.json>", file=sys.stderr)
        return 2
    try:
        with open(slices_path, encoding="utf-8") as f:
            doc = json.load(f)
        strata = tuple(
            Stratum(
                name=s["name"],
                n_treated=s["n_treated"],
                n_control=s["n_control"],
                treated_successes=s["treated_successes"],
                control_successes=s["control_successes"],
            )
            for s in doc["strata"]
        )
        tolerance = (
            float(tol_s) if tol_s is not None
            else float(doc.get("allocation_tolerance", DEFAULT_ALLOCATION_TOLERANCE))
        )
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError(
                f"allocation_tolerance must be in [0, 1], got {tolerance}"
            )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"Error: invalid slices doc: {e}", file=sys.stderr)
        return 2
    try:
        pooled_n_treated = int(doc["pooled_n_treated"])
        pooled_n_control = int(doc["pooled_n_control"])
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error: pooled arm totals required: {e}", file=sys.stderr)
        return 2

    try:
        verdict = win_claim(
            strata, pooled_n_treated, pooled_n_control, allocation_tolerance=tolerance
        )
    except ValueError as e:
        # Seam contract errors (empty strata, duplicate names): the slices
        # doc is unusable input (exit 2), never a refused claim (exit 1).
        print(f"Error: invalid slices doc: {e}", file=sys.stderr)
        return 2
    rep = verdict.report
    lines = ["per-stratum disaggregation (mandatory):"]
    for r in rep.strata:
        lines.append(
            f"  {r.name}: treated {r.treated_rate:.2%} "
            f"(n={r.n_treated}, share={r.treated_share:.1%}) vs "
            f"control {r.control_rate:.2%} (n={r.n_control}) "
            f"-> delta {r.delta:+.2%}"
        )
    lines.append(
        f"pooled (descriptive only, not a result): treated {rep.pooled_treated_rate:.2%} "
        f"vs control {rep.pooled_control_rate:.2%} "
        f"-> delta {rep.pooled_delta:+.2%}"
    )
    if verdict.accepted:
        lines.append("verdict: WIN CLAIM ACCEPTED")
    else:
        lines.append(
            "verdict: WIN CLAIM REJECTED "
            f"({', '.join(verdict.violations) or 'pooled numbers do not favor treated'})"
        )
        for note in verdict.notes:
            lines.append(f"  note: {note}")
    print("\n".join(lines))
    if out:
        payload = {
            "tool": "agenteval-bench",
            "command": "simpson",
            "accepted": verdict.accepted,
            "violations": list(verdict.violations),
            "notes": list(verdict.notes),
            "strata": [
                {
                    "name": r.name,
                    "n_treated": r.n_treated,
                    "n_control": r.n_control,
                    "treated_share": r.treated_share,
                    "treated_rate": r.treated_rate,
                    "control_rate": r.control_rate,
                    "delta": r.delta,
                }
                for r in rep.strata
            ],
            "pooled_treated_rate": rep.pooled_treated_rate,
            "pooled_control_rate": rep.pooled_control_rate,
            "pooled_delta": rep.pooled_delta,
        }
        try:
            with open(out, "w", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        except OSError as e:
            print(f"Error: cannot write report: {e}", file=sys.stderr)
            return 2
        print(f"Report: {out}")
    return 0 if verdict.accepted else 1


def cmd_judges(args: list[str]) -> int:
    """Allocate decorrelated judges from a registry file (arena#44).

    Reads a judge registry (JSON: a mapping with a ``judges`` list, or a
    bare list; each entry carries ``id``, ``host``, ``model_family``,
    optional ``model`` and ``reachable``), allocates up to ``--want``
    judges with pairwise-distinct model families, and prints the picks
    with the unreachable judges as explicit holes. Exit 0 = the
    allocation satisfied the request; exit 2 = thin allocation (the
    registry does not supply ``--want`` distinct reachable families);
    exit 1 = unusable registry.
    """
    from agenteval_bench.judges import allocate, registry_from_json

    args, registry_path = _flag(args, "--registry")
    args, want_s = _flag(args, "--want", "2")
    args, out = _flag(args, "--out")
    args, as_json = _has(args, "--json")
    if not registry_path:
        print("Error: judges needs --registry <file.json>", file=sys.stderr)
        return 1
    try:
        want = int(want_s) if want_s is not None else 2
    except ValueError:
        print(f"Error: --want must be an integer, got {want_s!r}", file=sys.stderr)
        return 1
    if want < 1:
        print("Error: --want must be >= 1", file=sys.stderr)
        return 1
    try:
        text = Path(registry_path).read_text(encoding="utf-8")
    except OSError as e:
        print(f"Error: cannot read registry: {e}", file=sys.stderr)
        return 1
    try:
        judges = registry_from_json(text)
    except (ValueError, TypeError) as e:
        print(f"Error: unusable registry: {e}", file=sys.stderr)
        return 1
    report = allocate(judges, want=want)
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        picked = ", ".join(f"{j.id} [{j.model_family}]" for j in report.picked)
        print(f"picked ({len(report.picked)}/{want}): {picked or '(nobody)'}")
        if report.holes:
            print("HOLES — unreachable, reported never dropped:")
            for hole in report.holes:
                print(
                    f"  ! {hole.judge} [{hole.model_family}] ({hole.host}): "
                    f"{hole.reason}"
                )
        for rejected in report.rejected:
            print(f"  ! rejected {rejected.judge}: {rejected.reason}")
        for note in report.notes:
            print(f"  ! {note}")
        print("Decorrelation is on model_family, never on host.")
        print(
            "Reachability as declared in the registry; no probe was run, "
            "so a 'measured' label here is a claim, not a measurement."
        )
    if out:
        try:
            with open(out, "w", encoding="utf-8") as f:
                f.write(json.dumps(report.to_dict(), indent=2) + "\n")
        except OSError as e:
            print(f"Error: cannot write report: {e}", file=sys.stderr)
            return 1
        print(f"Report: {out}")
    return 0 if report.satisfied else 2


def cmd_calibrate(args: list[str]) -> int:
    """Calibrate one judge's stated confidence against measured agreement (arena#45).

    Reads recorded (label, stated-confidence) pairs as JSON — a mapping
    with a ``judge`` string and an ``observations`` list, or a bare list —
    and grades the judge CALIBRATED / MISCALIBRATED / UNCALIBRATED.
    The grading policy defaults to min_total 50, 10 bins, min 5 per bin,
    tolerance 0.10; override with --min-total / --bins / --min-bin-n /
    --tolerance. Exit 0 = calibrated (the judge's scores may gate once a
    live judge transport calls require_calibrated);
    exit 2 = miscalibrated or uncalibrated (the scores must not gate);
    exit 1 = unusable observations file or policy.
    """
    from agenteval_bench.calibration import (
        CalibrationPolicy,
        bin_range_label,
        calibrate,
        default_calibration_policy,
        observations_from_json,
    )

    args, obs_path = _flag(args, "--observations")
    args, out = _flag(args, "--out")
    args, as_json = _has(args, "--json")
    args, min_total_s = _flag(args, "--min-total")
    args, bins_s = _flag(args, "--bins")
    args, min_bin_n_s = _flag(args, "--min-bin-n")
    args, tolerance_s = _flag(args, "--tolerance")
    if not obs_path:
        print("Error: calibrate needs --observations <file.json>", file=sys.stderr)
        return 1
    try:
        policy = CalibrationPolicy(
            min_total=int(min_total_s) if min_total_s is not None else 50,
            bins=int(bins_s) if bins_s is not None else 10,
            min_bin_n=int(min_bin_n_s) if min_bin_n_s is not None else 5,
            tolerance=float(tolerance_s) if tolerance_s is not None else 0.10,
        )
    except (ValueError, TypeError) as e:
        print(f"Error: unusable calibration policy: {e}", file=sys.stderr)
        return 1
    try:
        text = Path(obs_path).read_text(encoding="utf-8")
    except OSError as e:
        print(f"Error: cannot read observations: {e}", file=sys.stderr)
        return 1
    try:
        judge, observations = observations_from_json(text)
    except (ValueError, TypeError) as e:
        print(f"Error: unusable observations: {e}", file=sys.stderr)
        return 1
    record = calibrate(judge, observations, policy)
    if as_json:
        print(json.dumps(record.to_dict(), indent=2))
    else:
        ece = "n/a" if record.ece is None else f"{record.ece:.3f}"
        print(
            f"judge {record.judge}: {record.status} "
            f"({record.n_total} observations, ECE {ece})"
        )
        if record.policy != default_calibration_policy():
            print(
                "  policy (non-default): "
                f"min_total={record.policy.min_total} "
                f"bins={record.policy.bins} "
                f"min_bin_n={record.policy.min_bin_n} "
                f"tolerance={record.policy.tolerance}"
            )
        for b in record.bins:
            print(
                f"  bin {bin_range_label(b.lower_edge, b.upper_edge)}: n={b.n} "
                f"mean confidence {b.mean_confidence:.2f}, measured agreement "
                f"{b.agreement_rate:.2f} "
                f"(Wilson 95% [{b.agreement_interval[0]:.2f}, "
                f"{b.agreement_interval[1]:.2f}]) -> "
                f"{'pass' if b.passed else 'FAIL'}"
            )
        for reason in record.reasons:
            print(f"  ! {reason}")
        for note in record.notes:
            print(f"  ~ {note}")
        if record.status == "calibrated":
            print("This judge's scores may gate once a live judge transport calls require_calibrated.")
        else:
            print("This judge's scores must not gate: confidence is not evidence.")
    if out:
        try:
            with open(out, "w", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), indent=2) + "\n")
        except OSError as e:
            print(f"Error: cannot write report: {e}", file=sys.stderr)
            return 1
        # Under --json stdout is the report JSON itself, so the Report:
        # line goes to stderr to keep stdout parseable.
        print(f"Report: {out}", file=sys.stderr if as_json else sys.stdout)
    return 0 if record.status == "calibrated" else 2


def cmd_ship(args: list[str]) -> int:
    """Rule SHIP / NO_SHIP / NEEDS_REPLICATION for one agent change (arena#46).

    Reads a pre-registered design (--design design.json: plan digest,
    OEC + guardrail declarations, expected allocation, surprise
    threshold) and reported evidence (--evidence evidence.json: plan
    digest, OEC metric, run id, experiment looks, SRM counts, A/A
    split, guardrail observations, optional replication), replays the
    experiment through the hash-chained plan registry (--registry,
    default plans/experiments.jsonl), and rules. Exit 0 = SHIP;
    exit 1 = NO_SHIP or NEEDS_REPLICATION (reasons printed);
    exit 2 = unusable design/evidence/registry.
    """
    from experiment.oec import design_from_json, evidence_from_json, ship_decision

    args, design_path = _flag(args, "--design")
    args, evidence_path = _flag(args, "--evidence")
    args, registry_path = _flag(args, "--registry", "plans/experiments.jsonl")
    args, out = _flag(args, "--out")
    args, as_json = _has(args, "--json")
    if not design_path or not evidence_path:
        print(
            "Error: ship needs --design <file.json> --evidence <file.json>",
            file=sys.stderr,
        )
        return 2
    try:
        design = design_from_json(Path(design_path).read_text(encoding="utf-8"))
    except (ValueError, TypeError, OSError) as e:
        print(f"Error: unusable design: {e}", file=sys.stderr)
        return 2
    try:
        evidence = evidence_from_json(Path(evidence_path).read_text(encoding="utf-8"))
    except (ValueError, TypeError, OSError) as e:
        print(f"Error: unusable evidence: {e}", file=sys.stderr)
        return 2
    if registry_path is None or not os.path.exists(registry_path):
        # A missing registry is unusable input, not a NO_SHIP: the
        # gate cannot grade against a plan it cannot load.
        print(f"Error: registry not found: {registry_path}", file=sys.stderr)
        return 2
    try:
        verdict = ship_decision(design, evidence, registry_path)
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error: unusable registry or design: {e}", file=sys.stderr)
        return 2
    if as_json:
        print(json.dumps(verdict.to_dict(), indent=2))
    else:
        print(f"verdict: {verdict.status.upper().replace('_', ' ')}")
        if verdict.oec_z is not None:
            print(f"  OEC final z={verdict.oec_z:.2f} (surprising: {verdict.surprising})")
        if verdict.srm is not None:
            print(f"  SRM: {verdict.srm.status} (chi2={verdict.srm.chi2}, p={verdict.srm.p_value})")
        if verdict.aa is not None:
            print(f"  A/A: {verdict.aa.status} (z={verdict.aa.z}, p={verdict.aa.p_value})")
        for guardrail in verdict.guardrails:
            print(
                f"  guardrail {guardrail.name}: "
                f"{'pass' if guardrail.passed else 'FAIL'} "
                f"(delta {guardrail.delta_good:+.3f}, lower bound "
                f"{guardrail.lower_bound:+.3f}, budget "
                f"-{guardrail.max_regression:.3f})"
            )
        for reason in verdict.reasons:
            print(f"  ! {reason}")
        for note in verdict.notes:
            print(f"  ~ {note}")
    if out:
        try:
            with open(out, "w", encoding="utf-8") as f:
                f.write(json.dumps(verdict.to_dict(), indent=2) + "\n")
        except OSError as e:
            print(f"Error: cannot write report: {e}", file=sys.stderr)
            return 2
        print(f"Report: {out}", file=sys.stderr if as_json else sys.stdout)
    return 0 if verdict.status == "ship" else 1


def cmd_register_design(args: list[str]) -> int:
    """Register an OEC ship-gate design in the hash chain (arena#46).

    Reads a design (--design design.json) and appends it to the
    registry (--registry, default plans/experiments.jsonl). The
    design's plan must already be registered. Exit 0 = registered
    (digest printed); exit 2 = unusable design/registry.
    """
    from experiment.oec import design_from_json, register_design

    args, design_path = _flag(args, "--design")
    args, registry_path = _flag(args, "--registry", "plans/experiments.jsonl")
    if not design_path:
        print("Error: register-design needs --design <file.json>", file=sys.stderr)
        return 2
    try:
        design = design_from_json(Path(design_path).read_text(encoding="utf-8"))
    except (ValueError, TypeError, OSError) as e:
        print(f"Error: unusable design: {e}", file=sys.stderr)
        return 2
    if registry_path is None or not os.path.exists(registry_path):
        print(f"Error: registry not found: {registry_path}", file=sys.stderr)
        return 2
    try:
        digest = register_design(design, registry_path)
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error: cannot register design: {e}", file=sys.stderr)
        return 2
    print(f"registered design {digest}")
    return 0


def cmd_replay(args: list[str]) -> int:
    """Verify a replay log is bit-exact: re-run and byte-compare."""
    args, log_path = _flag(args, "--log")
    args, suite_path = _flag(args, "--suite")
    args, check = _has(args, "--check")
    if not log_path or not suite_path:
        print("Error: replay needs --log <file> --suite <file> [--check]", file=sys.stderr)
        return 1
    with open(log_path, encoding="utf-8") as f:
        original_bytes = f.read()
    try:
        expected = ReplayLog.from_json(original_bytes)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"Error: invalid replay log: {e}", file=sys.stderr)
        return 1

    suite, recorded, _missing, digest = _load_suite_for_replay(suite_path)
    if digest != expected.digest:
        print(f"Error: suite digest mismatch: suite is {digest}, log was {expected.digest}",
              file=sys.stderr)
        return 1
    result, outputs = _run_suite(suite, recorded, expected.seed)
    actual = _build_log(suite, suite_path, digest, result, outputs, expected.seed)

    if not check:
        actual.write(log_path)
        print(f"Replayed {len(actual.cases)} cases (seed={expected.seed}) -> {log_path}")
        return 0

    problems = find_mismatches(expected, actual)
    if problems:
        print(f"Replay NOT bit-exact ({len(problems)} mismatch(es)):")
        for p in problems[:10]:
            print(f"  - {p}")
        return 1
    if actual.to_json() != original_bytes:
        print("Replay NOT bit-exact: canonical bytes differ.")
        return 1
    print(f"Replay bit-exact: {len(actual.cases)} cases, seed={expected.seed}, "
          f"digest={expected.digest[:19]}...")
    return 0


def _validated_record(report, repo_digest: str) -> dict:
    """One ``all_validated.jsonl`` record for a validated report.

    Pure and importable for tests. The instance_id uses the single canonical
    formula (bugsmith.curate.instance_id) — the same one
    BenchmarkInstance.from_dict enforces — so the jsonl joins mechanically
    with the curated instance JSONs via instance_id, strategy, and patch_sha.
    """
    from bugsmith.curate import instance_id

    record = report.candidate.record
    return {
        "instance_id": instance_id(report, repo_digest),
        "strategy": record.strategy.value,
        "seed": record.seed,
        "target_file": record.target_file,
        "patch_sha": report.candidate.patch_sha,
        "fail_to_pass": list(report.fail_to_pass),
        "pass_to_pass": list(report.pass_to_pass),
    }


def cmd_bugsmith(args: list[str]) -> int:
    """Run the SWE-smith bug-injection pipeline (agenteval-bench#37).

    generate -> validate (inside Docker unless --local) -> curate, writing
    validated instances + a manifest to --out. Exit 0 only if at least one
    valid instance survives validation.
    """
    import tempfile
    from pathlib import Path

    from bugsmith.buggen import PRMirrorGenerator, ProceduralBugGenerator
    from bugsmith.curate import select_subset
    from bugsmith.harness import DockerRunner, LocalRunner, baseline, validate
    from bugsmith.images import build_image, repo_digest
    from bugsmith.types import BugCandidate, BugsmithError, CurationConfig

    args, repo_s = _flag(args, "--repo")
    args, config_s = _flag(args, "--config")
    args, out_s = _flag(args, "--out")
    args, count_s = _flag(args, "--procedural", "8")
    args, seed_s = _flag(args, "--seed", str(DEFAULT_SEED))
    args, local = _has(args, "--local")
    # --pr-mirror <patch>:<pr-ref>, repeatable
    pr_mirrors: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--pr-mirror" and i + 1 < len(args):
            pr_mirrors.append(args[i + 1])
            i += 2
        else:
            rest.append(args[i])
            i += 1
    args = rest

    if not repo_s or not out_s:
        print("Error: bugsmith needs --repo <dir> --out <dir>", file=sys.stderr)
        return 1
    repo = Path(repo_s)
    if not repo.is_dir():
        print(f"Error: repo not found: {repo_s}", file=sys.stderr)
        return 1
    try:
        count = int(count_s or "8")
        seed = int(seed_s or str(DEFAULT_SEED))
        if count < 1:
            raise ValueError("count >= 1")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    try:
        config = CurationConfig(**json.loads(config_s)) if config_s else CurationConfig(seed=seed)
    except (ValueError, TypeError, json.JSONDecodeError, BugsmithError) as e:
        print(f"Error: invalid --config: {e}", file=sys.stderr)
        return 1

    try:
        with tempfile.TemporaryDirectory(prefix="bugsmith-") as tmp:
            work_dir = Path(tmp)
            if local:
                runner: DockerRunner | LocalRunner = LocalRunner()
                digest = repo_digest(repo)
            else:
                image = build_image(repo, work_dir)
                runner = DockerRunner(image)
                digest = image.repo_digest
            passed = baseline(repo, runner, work_dir)
            print(f"baseline: {len(passed)} tests passing")

            candidates: list[BugCandidate] = ProceduralBugGenerator(seed).generate(repo, count)
            for spec in pr_mirrors:
                patch_path, _, pr_ref = spec.partition(":")
                if not patch_path or not pr_ref:
                    print(f"Error: --pr-mirror needs <patch>:<pr-ref>, got {spec!r}",
                          file=sys.stderr)
                    return 1
                try:
                    patch = Path(patch_path).read_text(encoding="utf-8")
                except OSError as e:
                    print(f"Error: cannot read --pr-mirror patch {patch_path!r}: {e}",
                          file=sys.stderr)
                    return 1
                candidates.extend(
                    PRMirrorGenerator(pr_ref=pr_ref).generate(patch, ["mirrored"])
                )
            reports = []
            for c in candidates:
                try:
                    reports.append(validate(c, repo, runner, work_dir, passed))
                except BugsmithError as e:
                    # One bad candidate (e.g. a stale --pr-mirror patch) must
                    # not discard the good ones.
                    print(f"Warning: skipping unvalidatable candidate "
                          f"({c.record.site_description}): {e}", file=sys.stderr)
            valid = [r for r in reports if r.is_valid_instance]
            print(f"validated: {len(valid)}/{len(reports)} instances break >= 1 test")
            subset = select_subset(config, reports, digest)
            print(f"curated: {len(subset)} instances")

            out = Path(out_s)
            out.mkdir(parents=True, exist_ok=True)
            for inst in subset:
                (out / f"{inst.instance_id}.json").write_text(
                    json.dumps(inst.to_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            # The full validated set stays auditable: curation filtering
            # must be reproducible from these records + the config below.
            # instance_id uses the single canonical formula from
            # bugsmith.curate.instance_id (same one from_dict enforces).
            with (out / "all_validated.jsonl").open("w", encoding="utf-8") as f:
                for r in valid:
                    f.write(json.dumps(_validated_record(r, digest),
                                       sort_keys=True) + "\n")
            manifest = {
                "tool": "agenteval-bench",
                "command": "bugsmith",
                "seed": seed,
                "procedural_count": count,
                "pr_mirror_specs": pr_mirrors,
                "curation_config": {
                    "seed": config.seed,
                    "fail_to_pass_min": config.fail_to_pass_min,
                    "fail_to_pass_max": config.fail_to_pass_max,
                    "max_instances": config.max_instances,
                    "strategy_quota": dict(config.strategy_quota),
                },
                "repo_digest": digest,
                "baseline_tests": len(passed),
                "candidates": len(candidates),
                "valid_instances": len(valid),
                "curated": [i.instance_id for i in subset],
            }
            (out / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"wrote {len(subset)} instances + manifest -> {out}")
            if not subset:
                print("No curated instances: nothing to benchmark.", file=sys.stderr)
                return 1
            return 0
    except BugsmithError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_snapshot(args: list[str]) -> int:
    """Immutable benchmark snapshots: publish / verify / list / check.

    publish: freeze a YAML suite as a new immutable version. The baseline is
    a real recorded run (recorded `output` fields, like `run`), pinned into
    the version manifest for the release gate.
    """
    from agenteval_bench.snapshots import SnapshotError, SnapshotStore

    if not args or args[0] in ("--help", "-h"):
        print("Usage: agenteval-bench snapshot publish --suite <file> --store <dir>")
        print("                                --version <id> --owner <name> --why <text>")
        print("                                [--prev <id>] [--changed <text>] [--cadence <text>]")
        print("                                [--regression-set <text>] [--feedback-loop <text>]")
        print("                                [--promotions <file.json>] [--agent <name>] [--seed 42]")
        print("         (baseline is pinned from the suite's recorded `output` fields, like `run`)")
        print("       agenteval-bench snapshot show --store <dir> --suite <name> --version <id>")
        print("       agenteval-bench snapshot verify --store <dir> --suite <name> --version <id>")
        print("       agenteval-bench snapshot list --store <dir> [--suite <name>]")
        print("       agenteval-bench snapshot check --store <dir>")
        print("       agenteval-bench snapshot repair --store <dir> --reason <text>")
        print("         (re-journal crash-orphaned versions after out-of-band verification)")
        print("       agenteval-bench snapshot compare --store <dir> --suite <name> --version <id>")
        print("                                --candidate-outputs <file.json>")
        print("                                [--candidate-name <name>] [--min-pass-rate <x>]")
        print("                                [--seed 42]")
        print("         (default seed = the version's pinned baseline seed;")
        print("          exit 0 = GATE_PASS, exit 2 = GATE_FAIL: releases gate on this comparison)")
        return 0

    sub, rest = args[0], args[1:]
    if sub == "publish":
        return _snapshot_publish(rest)
    if sub == "show":
        rest, store_s = _flag(rest, "--store")
        rest, suite_name = _flag(rest, "--suite")
        rest, version_id = _flag(rest, "--version")
        if not store_s or not suite_name or not version_id:
            print("Error: show needs --store <dir> --suite <name> --version <id>",
                  file=sys.stderr)
            return 2
        try:
            version, suite = SnapshotStore(store_s).load(suite_name, version_id)
        except SnapshotError as e:
            print(f"CORRUPTED: {e}", file=sys.stderr)
            return 1
        meta = version.meta
        base = version.baseline
        print(f"{version.suite_name}/{version.version_id}  digest={version.digest}")
        print(f"created: {version.created_at}  prev: {version.prev_version_id}")
        print(f"owner: {meta.owner}")
        print(f"why: {meta.why}")
        if meta.changed_from_prev and version.prev_version_id:
            print(f"changed from {version.prev_version_id}: {meta.changed_from_prev}")
        elif not version.prev_version_id:
            print("initial version (no predecessor)")
        if meta.cadence:
            print(f"update cadence: {meta.cadence}")
        if meta.regression_set:
            print(f"regression set: {meta.regression_set}")
        if meta.feedback_loop:
            print(f"feedback loop: {meta.feedback_loop}")
        print(f"cases: {len(suite.cases)}  "
              f"baseline: pass_rate {base.pass_rate:.1%} (agent={base.agent}, seed={base.seed})")
        if version.promotions:
            print(f"promotions ({len(version.promotions)}):")
            for p in version.promotions:
                print(f"  - {p.get('promoted_case_id')} "
                      f"<- {p.get('kind')}:{p.get('failure_id')} "
                      f"(prior_score={p.get('prior_score')})")
        return 0
    if sub == "compare":
        return _snapshot_compare(rest)
    if sub == "verify":
        rest, store_s = _flag(rest, "--store")
        rest, suite_name = _flag(rest, "--suite")
        rest, version_id = _flag(rest, "--version")
        if not store_s or not suite_name or not version_id:
            print("Error: verify needs --store <dir> --suite <name> --version <id>",
                  file=sys.stderr)
            return 2
        try:
            SnapshotStore(store_s).verify(suite_name, version_id)
        except SnapshotError as e:
            print(f"CORRUPTED: {e}", file=sys.stderr)
            return 1
        print(f"OK: {suite_name}/{version_id} hash-verified")
        return 0
    if sub == "list":
        rest, store_s = _flag(rest, "--store")
        rest, suite_name = _flag(rest, "--suite")
        if not store_s:
            print("Error: list needs --store <dir>", file=sys.stderr)
            return 2
        store = SnapshotStore(store_s)
        if suite_name:
            for v in store.list_versions(suite_name):
                print(f"{suite_name}/{v}")
        else:
            root = store.root
            if root.is_dir():
                for sdir in sorted(p for p in root.iterdir()
                                   if p.is_dir() and not p.name.startswith(".")):
                    for v in store.list_versions(sdir.name):
                        print(f"{sdir.name}/{v}")
        return 0
    if sub == "check":
        rest, store_s = _flag(rest, "--store")
        if not store_s:
            print("Error: check needs --store <dir>", file=sys.stderr)
            return 2
        problems = SnapshotStore(store_s).verify_all()
        if problems:
            print(f"CORRUPTED: {len(problems)} version(s) failed verification:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print("OK: all published versions hash-verified")
        return 0
    if sub == "repair":
        rest, store_s = _flag(rest, "--store")
        rest, reason = _flag(rest, "--reason")
        if not store_s or not reason:
            print("Error: repair needs --store <dir> --reason <text>", file=sys.stderr)
            return 2
        try:
            repaired = SnapshotStore(store_s).repair_journal(reason)
        except SnapshotError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        if repaired:
            print(f"Repaired {len(repaired)} version(s) (journaled as snapshot-repair):")
            for v in repaired:
                print(f"  - {v}")
        else:
            print("Nothing to repair: every published version is journaled.")
        return 0
    print(f"Error: unknown snapshot subcommand: {sub}", file=sys.stderr)
    return 2


def _snapshot_publish(args: list[str]) -> int:
    from agenteval_bench.regression import PromotionRecord
    from agenteval_bench.snapshots import (
        SnapshotError,
        SnapshotStore,
        VersionMeta,
        baseline_from_run,
    )

    args, suite_path = _flag(args, "--suite")
    args, store_s = _flag(args, "--store")
    args, version_id = _flag(args, "--version")
    args, owner = _flag(args, "--owner")
    args, why = _flag(args, "--why")
    args, prev = _flag(args, "--prev")
    args, changed = _flag(args, "--changed", "")
    args, cadence = _flag(args, "--cadence", "")
    args, regset = _flag(args, "--regression-set", "")
    args, feedback = _flag(args, "--feedback-loop", "")
    args, agent = _flag(args, "--agent", "recorded")
    args, seed_s = _flag(args, "--seed", str(DEFAULT_SEED))
    args, promotions_path = _flag(args, "--promotions")

    missing = [n for n, v in (("--suite", suite_path), ("--store", store_s),
                              ("--version", version_id), ("--owner", owner),
                              ("--why", why)) if not v]
    if missing:
        print(f"Error: snapshot publish needs {' '.join(missing)}", file=sys.stderr)
        return 2
    try:
        seed = int(seed_s or str(DEFAULT_SEED))
    except ValueError:
        print(f"Error: --seed must be an integer, got {seed_s}", file=sys.stderr)
        return 2

    try:
        promotions: list[dict] = []
        if promotions_path:
            try:
                with open(promotions_path, encoding="utf-8") as f:
                    raw_promotions = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Error: cannot read --promotions file: {e}", file=sys.stderr)
                return 2
            if not isinstance(raw_promotions, list):
                print("Error: --promotions file must be a JSON list of promotion records",
                      file=sys.stderr)
                return 2
            try:
                promotions = [PromotionRecord.from_dict(p).to_dict()
                              for p in raw_promotions]
            except (ValueError, TypeError, KeyError) as e:
                print(f"Error: invalid promotion record: {e}", file=sys.stderr)
                return 2
        suite, recorded, missing, _digest = _load_suite_for_replay(suite_path)
        result, _outputs = _run_suite(suite, recorded, seed)
        if missing:
            print(f"Note: {len(missing)} case(s) had no recorded output and were skipped.")
        scored = result.passed + result.failed
        if scored == 0:
            print("Error: no cases scored — the suite has no recorded `output` fields, "
                  "so no baseline can be pinned. Publish needs a replayable suite "
                  "(see `run --replay-log`).", file=sys.stderr)
            return 2
        baseline = baseline_from_run(result, seed=seed, agent=agent or "recorded")
        version = SnapshotStore(store_s).publish(
            suite,
            version_id,
            VersionMeta(
                owner=owner or "",
                why=why or "",
                changed_from_prev=changed or "",
                cadence=cadence or "",
                regression_set=regset or "",
                feedback_loop=feedback or "",
            ),
            baseline,
            prev_version_id=prev,
            promotions=promotions,
        )
    except SnapshotError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"Error: storage failure during publish: {e}", file=sys.stderr)
        return 1
    print(f"Published {suite.name}/{version.version_id} digest={version.digest}")
    print(f"Baseline: pass_rate {version.baseline.pass_rate:.1%} "
          f"(agent={version.baseline.agent}, seed={seed})")
    if promotions:
        print(f"Promotions recorded: {len(promotions)}")
    return 0


def _snapshot_compare(args: list[str]) -> int:
    """Replay a candidate's recorded outputs against a frozen version and gate.

    Exit 0 = GATE_PASS (release), exit 2 = GATE_FAIL (block), exit 1/2 = usage
    or corruption errors — same convention as `run --ci`.
    """
    from agenteval_bench.regression import compare_against_frozen
    from agenteval_bench.snapshots import SnapshotError, SnapshotStore

    args, store_s = _flag(args, "--store")
    args, suite_name = _flag(args, "--suite")
    args, version_id = _flag(args, "--version")
    args, cand_path = _flag(args, "--candidate-outputs")
    args, cand_name = _flag(args, "--candidate-name", "candidate")
    args, threshold_s = _flag(args, "--min-pass-rate")
    args, seed_s = _flag(args, "--seed")  # None -> version's pinned baseline seed

    missing = [n for n, v in (("--store", store_s), ("--suite", suite_name),
                              ("--version", version_id),
                              ("--candidate-outputs", cand_path)) if not v]
    if missing:
        print(f"Error: snapshot compare needs {' '.join(missing)}", file=sys.stderr)
        return 2
    try:
        with open(cand_path or "", encoding="utf-8") as f:
            candidate_outputs = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error: cannot read --candidate-outputs file: {e}", file=sys.stderr)
        return 2
    if not isinstance(candidate_outputs, dict) or not all(
        isinstance(k, str) and isinstance(v, str)
        for k, v in candidate_outputs.items()
    ):
        print("Error: --candidate-outputs must be a JSON object mapping "
              "case_id -> output string", file=sys.stderr)
        return 2
    try:
        threshold = float(threshold_s) if threshold_s is not None else None
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError("out of range")
    except ValueError:
        print(f"Error: --min-pass-rate must be between 0.0 and 1.0, got {threshold_s}",
              file=sys.stderr)
        return 2
    try:
        seed = int(seed_s) if seed_s is not None else None
    except ValueError:
        print(f"Error: --seed must be an integer, got {seed_s}", file=sys.stderr)
        return 2

    try:
        _version, suite = SnapshotStore(store_s or "").load(
            suite_name or "", version_id or "")
    except SnapshotError as e:
        print(f"CORRUPTED: {e}", file=sys.stderr)
        return 1

    replay_ids = iter([c.id for c in suite.cases if not c.skip])
    ordered_ids = [c.id for c in suite.cases if not c.skip]
    missing_ids: list[str] = []
    called_ids: list[str] = []

    def candidate_fn(_input: str) -> str:
        # The engine calls the agent fn once per non-skipped case, in case
        # order (EvalRunner.run contract). Attribution is by call order;
        # any divergence fails LOUDLY below, never silently misattributed.
        try:
            cid = next(replay_ids)
        except StopIteration:
            raise SnapshotError(
                "candidate agent_fn called more times than the frozen suite "
                "has cases — attribution would be silent misattribution"
            ) from None
        called_ids.append(cid)
        if cid not in candidate_outputs:
            missing_ids.append(cid)
            return ""
        return candidate_outputs[cid]

    # compare_against_frozen hash-verifies the frozen set on load and runs
    # the candidate through the same EvalRunner + seed as the baseline.
    store = SnapshotStore(store_s or "")
    try:
        report = compare_against_frozen(
            store,
            suite_name or "",
            version_id or "",
            candidate_fn,
            seed=seed,
            min_pass_rate=threshold,
            candidate_agent=cand_name or "candidate",
        )
    except SnapshotError as e:
        print(f"CORRUPTED: {e}", file=sys.stderr)
        return 1

    if called_ids != ordered_ids:
        print("Error: candidate output attribution mismatch — the engine's agent_fn "
              "call order diverged from frozen case order; refusing to score.",
              file=sys.stderr)
        return 1

    print(f"Regression gate: {suite_name}/{version_id} "
          f"(digest={report.digest[:19]}...) vs candidate '{report.candidate_agent}' "
          f"(seed={report.seed})")
    print(f"baseline pass_rate {report.baseline_pass_rate:.1%} -> "
          f"candidate {report.candidate_pass_rate:.1%}")
    if missing_ids:
        print(f"Note: {len(missing_ids)} case(s) had no candidate output and scored 0 "
              f"({', '.join(missing_ids[:5])}{'...' if len(missing_ids) > 5 else ''}).")
    for d in report.case_deltas:
        mark = "REGRESSED" if d.case_id in report.regressed_cases else "ok"
        if d.candidate_score != d.baseline_score:
            print(f"  {d.case_id}: {d.baseline_score:.2f} -> {d.candidate_score:.2f} [{mark}]")
    print(f"verdict: {report.verdict}")
    for reason in report.reasons:
        print(f"  reason: {reason}")
    return 0 if report.verdict == "GATE_PASS" else 2


def cmd_tiered(args: list[str]) -> int:
    """Run the tiered eval pipeline (arena#39): gate -> smoke -> expanded -> nightly.

    ``--phase <name>`` runs that phase and every earlier one (each phase's
    entry criterion is that all earlier phases passed); ``all`` runs
    everything. Exit 0 = all run phases passed; exit 2 = a phase failed;
    exit 1 = usage error.
    """
    from agenteval_bench.tiered import (
        DEFAULT_MIN_COVERAGE,
        DEFAULT_MIN_MUTATION_SCORE,
        DEFAULT_MUTANTS,
        TieredError,
        run_pipeline,
    )
    from agenteval_bench.tiered import DEFAULT_SEED as TIERED_SEED

    args, phase = _flag(args, "--phase", "all")
    args, mutants_s = _flag(args, "--mutants", str(DEFAULT_MUTANTS))
    args, seed_s = _flag(args, "--seed", str(TIERED_SEED))
    args, min_mut_s = _flag(args, "--min-mutation-score",
                            str(DEFAULT_MIN_MUTATION_SCORE))
    args, min_cov_s = _flag(args, "--min-coverage", str(DEFAULT_MIN_COVERAGE))
    args, out = _flag(args, "--out")
    args, root_s = _flag(args, "--root", ".")

    if args:
        print(f"Error: unknown argument(s): {' '.join(args)}", file=sys.stderr)
        return 1

    try:
        mutants = int(mutants_s or str(DEFAULT_MUTANTS))
        seed = int(seed_s or str(TIERED_SEED))
        min_mut = float(min_mut_s or str(DEFAULT_MIN_MUTATION_SCORE))
        min_cov = float(min_cov_s or str(DEFAULT_MIN_COVERAGE))
    except ValueError:
        print("Error: --mutants/--seed must be integers, "
              "--min-mutation-score/--min-coverage numbers", file=sys.stderr)
        return 1

    try:
        report = run_pipeline(
            Path(root_s or "."),
            upto=phase or "all",
            mutants=mutants,
            seed=seed,
            min_mutation_score=min_mut,
            min_coverage=min_cov,
        )
    except TieredError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    for p in report.phases:
        print(f"phase {p.phase}: {p.status.upper()} ({p.duration_s:.1f}s)")
        for note in p.notes:
            print(f"  note: {note}")
    if report.coverage_percent is not None:
        print(f"coverage: {report.coverage_percent:.1f}%")
    if report.mutation_score is not None:
        print(f"mutation score: {report.mutation_score:.0%}")
    print(f"verdict: {report.verdict}")
    if out:
        try:
            with open(out, "w", encoding="utf-8") as f:
                f.write(report.to_json())
        except OSError as e:
            print(f"Error: cannot write report: {e}", file=sys.stderr)
            return 1
        print(f"Report: {out}")
    return 0 if report.verdict == "PASS" else 2


def cmd_tracebench(args: list[str]) -> int:
    """Trace-mined benchmarks (arena#41): mine | regress.

    ``mine`` turns a JSONL production-trace corpus into a living benchmark
    (manifest + mined cases + oracle outputs). ``regress`` runs one
    scheduled regression: mine, publish new cases as a frozen version,
    gate a candidate against it, dispatch gate/rollback/investigation
    actions into a ledger. Exit 0 = GATE_PASS, 2 = GATE_FAIL, 1 = usage
    or storage error.
    """
    if not args or args[0] in ("--help", "-h"):
        print("Usage: agenteval-bench tracebench mine --corpus <jsonl> --owner <name>")
        print("                                --cadence <daily|weekly|monthly> --out <dir>")
        print("       agenteval-bench tracebench regress --corpus <jsonl> --store <dir>")
        print("                                --suite <name> [--version <new-id>] [--prev <id>]")
        print("                                --candidate-outputs <json> --owner <name>")
        print("                                --cadence <daily|weekly|monthly> --out <dir>")
        print("                                [--candidate-name <name>] [--seed 42]")
        print("                                [--prior-failures 0] [--state-dir <dir>]")
        print("                                [--corpus-kind sample|production|unknown]")
        print("       --state-dir closes the loop across runs: it supplies --prev,")
        print("       the failure streak, and the next --version from the previous")
        print("       run's state.json, so versions chain and rollback is reachable")
        print("       without an operator hand-feeding the streak.")
        return 0
    sub, rest = args[0], args[1:]
    if sub == "mine":
        return _tracebench_mine(rest)
    if sub == "regress":
        return _tracebench_regress(rest)
    print(f"Error: unknown tracebench subcommand: {sub}", file=sys.stderr)
    return 1


def _tracebench_mine(args: list[str]) -> int:
    from agenteval_bench.tracebench import (
        TracebenchError,
        build_manifest,
        load_corpus,
        mine_traces,
    )

    args, corpus = _flag(args, "--corpus")
    args, owner = _flag(args, "--owner")
    args, cadence = _flag(args, "--cadence", "weekly")
    args, out = _flag(args, "--out")
    args, corpus_kind = _flag(args, "--corpus-kind", "unknown")
    if args:
        print(f"Error: unknown argument(s): {' '.join(args)}", file=sys.stderr)
        return 1
    missing = [n for n, v in (("--corpus", corpus), ("--owner", owner),
                               ("--out", out)) if not v]
    if missing:
        print(f"Error: tracebench mine needs {' '.join(missing)}", file=sys.stderr)
        return 1
    try:
        load = load_corpus(corpus or "")
        report = mine_traces(
            load.records,
            owner=owner or "",
            cadence=cadence or "weekly",
            corpus_digest=load.digest,
            skipped_records=load.skipped,
            corpus_kind=corpus_kind or "unknown",
        )
        manifest = build_manifest(
            report.cases,
            benchmark_id="tracebench",
            owner=owner or "",
            cadence=cadence or "weekly",
            created_at=report.mined_at,
            last_mined_at=report.mined_at,
            corpus_kind=corpus_kind or "unknown",
        )
    except (TracebenchError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    out_path = Path(out or ".")
    try:
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
        (out_path / "mined-cases.json").write_text(report.to_json(), encoding="utf-8")
        oracle = {c.case.id: c.expected_output for c in report.cases}
        (out_path / "oracle-outputs.json").write_text(
            json.dumps(oracle, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as e:
        print(f"Error: cannot write mine outputs: {e}", file=sys.stderr)
        return 1
    print(f"Mined {len(report.cases)} case(s) from {load.total_lines} trace(s) "
          f"({load.skipped} skipped, corpus {load.digest[:19]}...)")
    for case in report.cases:
        print(f"  {case.case.id} [{case.pattern}] incident {case.incident_id}")
    print(f"Manifest: {out_path / 'manifest.json'}")
    print(f"Mined cases: {out_path / 'mined-cases.json'}")
    print(f"Oracle outputs: {out_path / 'oracle-outputs.json'}")
    return 0


def _tracebench_regress(args: list[str]) -> int:
    from agenteval_bench.tracebench import TracebenchError, run_scheduled

    args, corpus = _flag(args, "--corpus")
    args, store = _flag(args, "--store")
    args, suite = _flag(args, "--suite")
    args, version = _flag(args, "--version")
    args, prev = _flag(args, "--prev")
    args, cand_path = _flag(args, "--candidate-outputs")
    args, cand_name = _flag(args, "--candidate-name", "candidate")
    args, owner = _flag(args, "--owner")
    args, cadence = _flag(args, "--cadence", "weekly")
    args, out = _flag(args, "--out")
    args, seed_s = _flag(args, "--seed", str(DEFAULT_SEED))
    args, prior_s = _flag(args, "--prior-failures")
    args, state_dir = _flag(args, "--state-dir")
    args, corpus_kind = _flag(args, "--corpus-kind", "unknown")
    if args:
        print(f"Error: unknown argument(s): {' '.join(args)}", file=sys.stderr)
        return 1
    missing = [n for n, v in (("--corpus", corpus), ("--store", store),
                               ("--suite", suite),
                               ("--candidate-outputs", cand_path),
                               ("--owner", owner), ("--out", out)) if not v]
    if not version and not state_dir:
        missing.append("--version (or --state-dir to auto-derive it)")
    if missing:
        print(f"Error: tracebench regress needs {' '.join(missing)}", file=sys.stderr)
        return 1
    try:
        seed = int(seed_s or str(DEFAULT_SEED))
        # Absent --prior-failures means None: the run state supplies the
        # streak. Defaulting to 0 here would silently reset a chained
        # streak on every scheduled run.
        prior_failures = int(prior_s) if prior_s is not None else None
        if prior_failures is not None and prior_failures < 0:
            raise ValueError("negative")
    except ValueError:
        print("Error: --seed/--prior-failures must be integers (>= 0 for prior)",
              file=sys.stderr)
        return 1
    try:
        with open(cand_path or "", encoding="utf-8") as f:
            candidate_outputs = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error: cannot read --candidate-outputs file: {e}", file=sys.stderr)
        return 1
    if not isinstance(candidate_outputs, dict) or not all(
        isinstance(k, str) and isinstance(v, str)
        for k, v in candidate_outputs.items()
    ):
        print("Error: --candidate-outputs must be a JSON object mapping "
              "case_id -> output string", file=sys.stderr)
        return 1
    try:
        result = run_scheduled(
            corpus=corpus or "",
            store=store or "",
            suite_name=suite or "",
            version_id=version or "",
            prev_version_id=prev,
            candidate_outputs=candidate_outputs,
            owner=owner or "",
            cadence=cadence or "weekly",
            out_dir=out or "",
            seed=seed,
            candidate_name=cand_name or "candidate",
            prior_consecutive_failures=prior_failures,
            state_dir=state_dir,
            corpus_kind=corpus_kind or "unknown",
        )
    except (TracebenchError, ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Tracebench regression: {suite}/{result.version_id} "
          f"(mined {result.mined_cases}, new {result.new_cases}, "
          f"updated {result.updated_cases}, retired {result.retired_cases})")
    print(f"verdict: {result.verdict}")
    if result.previous_corpus_total_lines is not None:
        print(f"corpus: {result.corpus_total_lines} trace(s) "
              f"(previous run: {result.previous_corpus_total_lines})")
    else:
        print(f"corpus: {result.corpus_total_lines} trace(s)")
    if result.drift_detected:
        print(f"benchmark drift: {', '.join(result.drifted_case_ids)} — "
              "the benchmark moved since the previous version; "
              "last week's certification no longer applies")
    if result.previous_stale is not None:
        print(f"previous version stale: {result.previous_stale}")
    if result.version_ignored:
        print("note: requested --version ignored (steady state: no new version)")
    for action in result.actions:
        print(f"  action: {action.action} — {action.reason}")
    print(f"consecutive failures: {result.consecutive_failures}")
    print(f"Report: {result.report_path}")
    print(f"Action ledger: {result.actions_path}")
    print(f"Manifest: {result.manifest_path}")
    return 0 if result.verdict == "GATE_PASS" else 2


def cmd_flywheel(args: list[str]) -> int:
    """Run the fix-commit flywheel against a live repo (arena#40).

    Detect -> Sample -> Reflect -> Propose -> Validate -> PR, with the
    live ``gh`` transport and the local runner. Payloads are RENDERED to
    files under ``--out`` (default ``<target>/.flywheel/payloads``), never
    written into the target tree and never posted: posting them via the
    GitHub API is the fleet runner's job (a follow-up, not shipped here).
    The gate-catch ledger defaults to a durable path under the target
    (``<target>/.flywheel/gate-catch-ledger.jsonl``), so the leading
    metric persists across runs instead of evaporating in a temp dir.
    Exit 0 = the loop completed (even with zero payloads); exit 1 =
    usage error or a FlywheelError from any stage.
    """
    from bugsmith.harness import LocalRunner
    from flywheel.detect import GhCliTransport
    from flywheel.loop import run_loop
    from flywheel.propose import leading_gate_catch_rate
    from flywheel.types import FlywheelError

    args, repo = _flag(args, "--repo")
    args, target_s = _flag(args, "--target-root", ".")
    args, corpus_s = _flag(args, "--corpus-root")
    args, ledger_s = _flag(args, "--ledger")
    args, out_s = _flag(args, "--out")
    args, sample_s = _flag(args, "--sample-size", "10")
    args, seed_s = _flag(args, "--seed", "42")

    if args or not repo:
        print(
            "Error: agenteval-bench flywheel --repo owner/name "
            "[--target-root .] [--corpus-root <dir>] [--ledger <path>] "
            "[--out <dir>] [--sample-size 10] [--seed 42]",
            file=sys.stderr,
        )
        return 1
    try:
        sample_size = int(sample_s or "10")
        seed = int(seed_s or "42")
    except ValueError:
        print("Error: --sample-size/--seed must be integers", file=sys.stderr)
        return 1

    target_root = Path(target_s or ".")
    if corpus_s:
        corpus_root = Path(corpus_s)
    else:
        corpus_root = (
            Path(__file__).resolve().parents[1]
            / "bugsmith"
            / "fixtures"
            / "target"
        )
    ledger_path = (
        Path(ledger_s)
        if ledger_s
        else target_root / ".flywheel" / "gate-catch-ledger.jsonl"
    )
    out_dir = Path(out_s) if out_s else target_root / ".flywheel" / "payloads"

    try:
        with tempfile.TemporaryDirectory(prefix="flywheel-") as tmp:
            result = run_loop(
                transport=GhCliTransport(),
                target_repo=repo,
                corpus_root=corpus_root,
                target_root=target_root,
                work_dir=Path(tmp),
                runner=LocalRunner(),
                ledger_path=ledger_path,
                sample_size=sample_size,
                seed=seed,
            )
    except FlywheelError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    for payload in result.payloads:
        dest = out_dir / payload.rule_kind
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "PR.md").write_text(
            f"# {payload.title}\n\n{payload.body}", encoding="utf-8"
        )
        for rel, content in payload.files.items():
            path = dest / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print(
        f"detected={result.detected} sampled={result.sampled} "
        f"payloads={len(result.payloads)}"
    )
    for line in result.skipped:
        print(f"skipped: {line}")
    rate = leading_gate_catch_rate(ledger_path)
    if rate is None:
        print(f"ledger: {ledger_path} (no runs yet)")
    else:
        print(f"ledger: {ledger_path} (leading gate-catch rate: {rate:.0%})")
    print(f"payloads: {out_dir}")
    print("NOTE: payloads are rendered, not posted — the fleet runner "
          "posts them via the GitHub API.")
    return 0


def main() -> None:
    """Minimal CLI — full argparse/typer in v0.2."""
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        print("agenteval-bench — LLM agent evaluation CLI")
        print("Usage: agenteval-bench run --suite <file> [--ci] [--threshold 1.0]")
        print("                         [--seed 42] [--replay-log replay.json]")
        print("       agenteval-bench decide --suite <file> [--policy policy.yaml]")
        print("                                [--risk low|medium|high|critical]")
        print("                                [--seed 42] [--reliability-runs 2]")
        print("                                [--out report.json]")
        print("                                (exit 0 GATE, 1 ROLLBACK, 2 INVESTIGATE)")
        print("       agenteval-bench baseline --suite <file> --out baseline.json [--seed 42]")
        print("       agenteval-bench replay --log replay.json --suite <file> [--check]")
        print("       agenteval-bench alt-test --data annotations.jsonl [--epsilon 0.1]")
        print("                                [--q 0.05] [--seed 42] [--out report.json]")
        print("       agenteval-bench simpson --slices slices.json")
        print("                                [--allocation-tolerance 0.05] [--out report.json]")
        print("       agenteval-bench judges --registry <file.json> [--want 2]")
        print("                                [--json] [--out report.json]")
        print("                                (exit 0 satisfied, 2 thin allocation, 1 bad registry)")
        print("       agenteval-bench calibrate --observations <file.json>")
        print("                                [--min-total 50] [--bins 10]")
        print("                                [--min-bin-n 5] [--tolerance 0.10]")
        print("                                [--json] [--out report.json]")
        print("                                (exit 0 calibrated, 2 mis/uncalibrated, 1 bad input)")
        print("       agenteval-bench ship --design <file.json> --evidence <file.json>")
        print("                                [--registry plans/experiments.jsonl]")
        print("                                [--json] [--out report.json]")
        print("                                (exit 0 SHIP, 1 NO_SHIP/NEEDS_REPLICATION, 2 bad input)")
        print("       agenteval-bench register-design --design <file.json>")
        print("                                [--registry plans/experiments.jsonl]")
        print("                                (exit 0 registered, 2 bad input)")
        print("       agenteval-bench bugsmith --repo <dir> --out <dir> [--procedural 8]")
        print("                                [--seed 42] [--local] [--config curation.json]")
        print("                                [--pr-mirror patch:pr-ref]")
        print("       agenteval-bench snapshot publish --suite <file> --store <dir>")
        print("                                --version <id> --owner <name> --why <text>")
        print("       agenteval-bench snapshot verify --store <dir> --suite <name> --version <id>")
        print("       agenteval-bench tiered --phase <gate|smoke|expanded|nightly|all>")
        print("                                [--mutants 8] [--min-mutation-score 0.5]")
        print("                                [--min-coverage 80] [--seed 42] [--root .]")
        print("                                [--out report.json]")
        print("       agenteval-bench flywheel --repo owner/name [--target-root .]")
        print("                                [--corpus-root <dir>] [--ledger <path>]")
        print("                                [--out <dir>] [--sample-size 10] [--seed 42]")
        print("       agenteval-bench tracebench mine --corpus <jsonl> --owner <name>")
        print("                                --cadence <daily|weekly|monthly> --out <dir>")
        print("       agenteval-bench tracebench regress --corpus <jsonl> --store <dir>")
        print("                                --suite <name> --version <new-id> [--prev <id>]")
        print("                                --candidate-outputs <json> --owner <name>")
        print("                                --cadence <daily|weekly|monthly> --out <dir>")
        print("       (--seed is recorded for provenance; the test itself is deterministic)")
        return

    cmd, rest = args[0], args[1:]
    if cmd == "run":
        sys.exit(cmd_run(rest))
    elif cmd == "decide":
        sys.exit(cmd_decide(rest))
    elif cmd == "tiered":
        sys.exit(cmd_tiered(rest))
    elif cmd == "flywheel":
        sys.exit(cmd_flywheel(rest))
    elif cmd == "tracebench":
        sys.exit(cmd_tracebench(rest))
    elif cmd == "baseline":
        sys.exit(cmd_baseline(rest))
    elif cmd == "replay":
        sys.exit(cmd_replay(rest))
    elif cmd == "alt-test":
        sys.exit(cmd_alt_test(rest))
    elif cmd == "simpson":
        sys.exit(cmd_simpson(rest))
    elif cmd == "judges":
        sys.exit(cmd_judges(rest))
    elif cmd == "calibrate":
        sys.exit(cmd_calibrate(rest))
    elif cmd == "ship":
        sys.exit(cmd_ship(rest))
    elif cmd == "register-design":
        sys.exit(cmd_register_design(rest))
    elif cmd == "bugsmith":
        sys.exit(cmd_bugsmith(rest))
    elif cmd == "snapshot":
        sys.exit(cmd_snapshot(rest))
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
