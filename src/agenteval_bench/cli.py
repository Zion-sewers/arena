"""CLI entry point for agenteval-bench."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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

    if replay_log:
        log = _build_log(suite, suite_path, digest, result, outputs, seed)
        log.write(replay_log)
        print(f"Replay log: {replay_log} (seed={seed})")

    if ci_mode:
        ok = result.pass_rate >= threshold
        print(f"CI gate: pass_rate {result.pass_rate:.1%} vs threshold {threshold:.0%} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 2
    return 0


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


def main() -> None:
    """Minimal CLI — full argparse/typer in v0.2."""
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        print("agenteval-bench — LLM agent evaluation CLI")
        print("Usage: agenteval-bench run --suite <file> [--ci] [--threshold 1.0]")
        print("                         [--seed 42] [--replay-log replay.json]")
        print("       agenteval-bench baseline --suite <file> --out baseline.json [--seed 42]")
        print("       agenteval-bench replay --log replay.json --suite <file> [--check]")
        print("       agenteval-bench alt-test --data annotations.jsonl [--epsilon 0.1]")
        print("                                [--q 0.05] [--seed 42] [--out report.json]")
        print("       agenteval-bench simpson --slices slices.json")
        print("                                [--allocation-tolerance 0.05] [--out report.json]")
        print("       agenteval-bench bugsmith --repo <dir> --out <dir> [--procedural 8]")
        print("                                [--seed 42] [--local] [--config curation.json]")
        print("                                [--pr-mirror patch:pr-ref]")
        print("       agenteval-bench snapshot publish --suite <file> --store <dir>")
        print("                                --version <id> --owner <name> --why <text>")
        print("       agenteval-bench snapshot verify --store <dir> --suite <name> --version <id>")
        print("       agenteval-bench tiered --phase <gate|smoke|expanded|nightly|all>")
        print("                                [--mutants 8] [--min-mutation-score 0.5]")
        print("                                [--min-coverage 80] [--out report.json]")
        print("       (--seed is recorded for provenance; the test itself is deterministic)")
        return

    cmd, rest = args[0], args[1:]
    if cmd == "run":
        sys.exit(cmd_run(rest))
    elif cmd == "tiered":
        sys.exit(cmd_tiered(rest))
    elif cmd == "baseline":
        sys.exit(cmd_baseline(rest))
    elif cmd == "replay":
        sys.exit(cmd_replay(rest))
    elif cmd == "alt-test":
        sys.exit(cmd_alt_test(rest))
    elif cmd == "simpson":
        sys.exit(cmd_simpson(rest))
    elif cmd == "bugsmith":
        sys.exit(cmd_bugsmith(rest))
    elif cmd == "snapshot":
        sys.exit(cmd_snapshot(rest))
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
