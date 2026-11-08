"""Immutable benchmark snapshots: version, never mutate (BetterBench discipline).

Every established benchmark set is an immutable, versioned snapshot on disk.
``SnapshotStore.publish`` writes a new version under
``<root>/<suite-name>/<version-id>/`` (``suite.json`` + ``manifest.json``)
and refuses to overwrite an existing version id — writes are append-only
through this API. ``SnapshotStore.load`` hash-verifies the stored bytes
against the manifest digest and cross-checks the manifest's semantic
content (baseline, version metadata) against a hash-chained publish
journal; any mismatch raises :class:`SnapshotCorrupted`.

Honesty boundary: append-only through the API is enforcement (the store
itself never mutates a version); protection against out-of-band mutation
(``rm``, manual edits) is *detection*, not prevention — a tampered version
fails closed at load time with the offending path named. The journal
detects silent rewrites of the manifest's release-gate inputs (baseline
scores, version metadata) that a suite-bytes hash alone would miss.
Detection is the strongest guarantee a single-machine file layout can
honestly offer: a party with full write access to the store directory
could rewrite store + journal together, so the journal is tamper-*evidence*
against partial/accidental mutation, not Byzantine tamper-*proofing*.
Committing the store to version control is the outer trust anchor.

Each version's manifest records what changed from the previous version and
why (owner, update cadence, regression set, feedback loop), forming an
auditable version chain: ``prev_version_id`` links every version back to
its predecessor.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agenteval_bench.models import (
    CostBound,
    EvalCase,
    EvalSuite,
    ExpectedOutput,
    RubricCriterion,
    RunResult,
)
from agenteval_bench.replay import suite_digest

SNAPSHOT_FORMAT = 1
JOURNAL_GENESIS = "GENESIS"


def _canonical_json(obj: Any) -> bytes:
    """Canonical JSON bytes (same style as suite canonical form)."""
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SnapshotError(Exception):
    """Base error for snapshot store failures."""


class VersionExistsError(SnapshotError):
    """Raised when publishing a version id that already exists."""


class VersionNotFoundError(SnapshotError):
    """Raised when loading a version id that does not exist."""


class SnapshotCorrupted(SnapshotError):
    """Raised when a stored version fails hash or schema verification."""


@dataclass(frozen=True)
class VersionMeta:
    """Why this version exists — the BetterBench maintenance record."""

    owner: str
    why: str
    changed_from_prev: str = ""
    cadence: str = ""
    regression_set: str = ""
    feedback_loop: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "why": self.why,
            "changed_from_prev": self.changed_from_prev,
            "cadence": self.cadence,
            "regression_set": self.regression_set,
            "feedback_loop": self.feedback_loop,
        }

    def __post_init__(self) -> None:
        if not self.owner or not self.why:
            raise ValueError("VersionMeta requires non-empty owner and why")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VersionMeta:
        if not isinstance(d, dict):
            raise SnapshotCorrupted(f"version meta must be a mapping, got {type(d).__name__}")
        for key in ("owner", "why"):
            if not isinstance(d.get(key), str) or not d[key]:
                raise SnapshotCorrupted(f"version meta missing non-empty {key!r}")
        optional = {}
        for key in ("changed_from_prev", "cadence", "regression_set", "feedback_loop"):
            value = d.get(key, "")
            if not isinstance(value, str):
                raise SnapshotCorrupted(f"version meta {key!r} must be a string")
            optional[key] = value
        return cls(owner=d["owner"], why=d["why"], **optional)


@dataclass(frozen=True)
class BaselineRecord:
    """Aggregate scores pinned at publish time — the release-gate reference."""

    agent: str
    seed: int
    pass_rate: float
    total: int
    passed: int
    failed: int
    skipped: int
    per_case: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "seed": self.seed,
            "pass_rate": self.pass_rate,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "per_case": dict(self.per_case),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BaselineRecord:
        if not isinstance(d, dict):
            raise SnapshotCorrupted(f"baseline must be a mapping, got {type(d).__name__}")
        if not isinstance(d.get("agent"), str):
            raise SnapshotCorrupted("baseline missing agent")
        if not isinstance(d.get("seed"), int):
            raise SnapshotCorrupted("baseline missing integer seed")
        for key in ("pass_rate",):
            if not isinstance(d.get(key), (int, float)):
                raise SnapshotCorrupted(f"baseline missing numeric {key!r}")
        for key in ("total", "passed", "failed", "skipped"):
            if not isinstance(d.get(key), int):
                raise SnapshotCorrupted(f"baseline missing integer {key!r}")
        per_case = d.get("per_case", {})
        if not isinstance(per_case, dict) or not all(
            isinstance(k, str) and isinstance(v, (int, float)) for k, v in per_case.items()
        ):
            raise SnapshotCorrupted("baseline per_case must map case id -> number")
        return cls(
            agent=d["agent"],
            seed=d["seed"],
            pass_rate=float(d["pass_rate"]),
            total=d["total"],
            passed=d["passed"],
            failed=d["failed"],
            skipped=d["skipped"],
            per_case={k: float(v) for k, v in per_case.items()},
        )


def baseline_from_run(result: RunResult, seed: int, agent: str) -> BaselineRecord:
    """Pin a recorded run's aggregates as a version's release-gate baseline."""
    per_case = {
        r.case_id: r.score for r in result.results if not r.details.get("skipped")
    }
    return BaselineRecord(
        agent=agent,
        seed=seed,
        pass_rate=result.pass_rate,
        total=result.total,
        passed=result.passed,
        failed=result.failed,
        skipped=result.skipped,
        per_case=per_case,
    )


@dataclass(frozen=True)
class BenchmarkVersion:
    """A published immutable benchmark snapshot."""

    suite_name: str
    version_id: str
    digest: str
    prev_version_id: str | None
    created_at: str
    meta: VersionMeta
    baseline: BaselineRecord
    promotions: tuple[dict[str, Any], ...] = ()


def _expected_to_dict(expected: ExpectedOutput) -> dict[str, Any]:
    return {
        "exact": expected.exact,
        "contains": list(expected.contains),
        "regex": expected.regex,
        "json_schema": (
            dict(expected.json_schema) if expected.json_schema is not None else None
        ),
    }


def _case_to_dict(case: EvalCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "input": case.input,
        "expected": _expected_to_dict(case.expected),
        "rubric": [
            {
                "criterion": r.criterion,
                "weight": r.weight,
                "description": r.description,
            }
            for r in case.rubric
        ],
        "skip": case.skip,
        "output": case.output,
        # Triple-regime replay evidence (arena#53): the recorded
        # outputs and inputs under the perturbed regimes. A suite
        # that passes through snapshots must keep them, or re-grading
        # it yields UNMEASURED with no error saying why.
        "output_light": case.output_light,
        "output_adversarial": case.output_adversarial,
        "input_light": case.input_light,
        "input_adversarial": case.input_adversarial,
    }


def suite_to_canonical(suite: EvalSuite) -> bytes:
    """Deterministic byte form of a suite — the snapshot's hashed identity.

    Built from the parsed ``EvalSuite`` object (not the source YAML), so
    ``canonical(parse(canonical(suite))) == canonical(suite)`` is a fixed
    point the store enforces at every load.
    """
    doc: dict[str, Any] = {
        "tool": "agenteval-bench",
        "kind": "benchmark-snapshot",
        "format": SNAPSHOT_FORMAT,
        "name": suite.name,
        "suite_version": suite.version,
        "cases": [_case_to_dict(c) for c in suite.cases],
        "cost_bound": {
            "max_input_tokens": suite.cost_bound.max_input_tokens,
            "max_output_tokens": suite.cost_bound.max_output_tokens,
        },
    }
    return (json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _require_str(d: dict[str, Any], key: str, where: str) -> str:
    value = d.get(key)
    if not isinstance(value, str):
        raise SnapshotCorrupted(f"{where}: {key!r} must be a string")
    return value


def _expected_from_dict(d: dict[str, Any]) -> ExpectedOutput:
    if not isinstance(d, dict):
        raise SnapshotCorrupted("case: expected must be a mapping")
    for key, kind in (("exact", str), ("regex", str)):
        value = d.get(key)
        if value is not None and not isinstance(value, kind):
            raise SnapshotCorrupted(f"case: expected.{key} must be a string or null")
    contains = d.get("contains", [])
    if not isinstance(contains, list) or not all(isinstance(x, str) for x in contains):
        raise SnapshotCorrupted("case: expected.contains must be a list of strings")
    schema = d.get("json_schema")
    if schema is not None and not isinstance(schema, dict):
        raise SnapshotCorrupted("case: expected.json_schema must be a mapping or null")
    return ExpectedOutput(
        exact=d.get("exact"),
        contains=list(contains),
        regex=d.get("regex"),
        json_schema=dict(schema) if schema is not None else None,
    )


def _case_from_dict(d: dict[str, Any]) -> EvalCase:
    if not isinstance(d, dict):
        raise SnapshotCorrupted("case must be a mapping")
    case_id = _require_str(d, "id", "case")
    if not case_id:
        raise SnapshotCorrupted("case: id must be non-empty")
    text = _require_str(d, "input", "case")
    rubric_raw = d.get("rubric", [])
    if not isinstance(rubric_raw, list):
        raise SnapshotCorrupted("case: rubric must be a list")
    rubric = []
    for r in rubric_raw:
        if not isinstance(r, dict):
            raise SnapshotCorrupted("case: rubric entries must be mappings")
        criterion = _require_str(r, "criterion", "rubric entry")
        weight = r.get("weight", 1.0)
        if not isinstance(weight, (int, float)):
            raise SnapshotCorrupted("rubric entry: weight must be numeric")
        description = r.get("description", "")
        if not isinstance(description, str):
            raise SnapshotCorrupted("rubric entry: description must be a string")
        rubric.append(
            RubricCriterion(criterion=criterion, weight=float(weight), description=description)
        )
    skip = d.get("skip", False)
    if not isinstance(skip, bool):
        raise SnapshotCorrupted("case: skip must be a boolean")
    output = d.get("output")
    if output is not None and not isinstance(output, str):
        raise SnapshotCorrupted("case: output must be a string or null")
    triple: dict[str, str | None] = {}
    for key in (
        "output_light",
        "output_adversarial",
        "input_light",
        "input_adversarial",
    ):
        value = d.get(key)
        if value is not None and not isinstance(value, str):
            raise SnapshotCorrupted(f"case: {key} must be a string or null")
        triple[key] = value
    return EvalCase(
        id=case_id,
        input=text,
        expected=_expected_from_dict(d.get("expected", {})),
        rubric=rubric,
        skip=skip,
        output=output,
        output_light=triple["output_light"],
        output_adversarial=triple["output_adversarial"],
        input_light=triple["input_light"],
        input_adversarial=triple["input_adversarial"],
    )


def suite_from_canonical(data: bytes) -> EvalSuite:
    """Parse canonical bytes back into an ``EvalSuite`` (strict schema)."""
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SnapshotCorrupted(f"suite.json is not valid canonical JSON: {e}") from e
    if not isinstance(doc, dict):
        raise SnapshotCorrupted("suite.json top level must be a mapping")
    if doc.get("kind") != "benchmark-snapshot":
        raise SnapshotCorrupted("suite.json: kind must be 'benchmark-snapshot'")
    if doc.get("format") != SNAPSHOT_FORMAT:
        raise SnapshotCorrupted(
            f"suite.json: unsupported format {doc.get('format')!r} "
            f"(this tool reads {SNAPSHOT_FORMAT})"
        )
    name = _require_str(doc, "name", "suite")
    suite_version = doc.get("suite_version", 1)
    if not isinstance(suite_version, int):
        raise SnapshotCorrupted("suite: suite_version must be an integer")
    cases_raw = doc.get("cases", [])
    if not isinstance(cases_raw, list):
        raise SnapshotCorrupted("suite: cases must be a list")
    cases = [_case_from_dict(c) for c in cases_raw]
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise SnapshotCorrupted(f"suite: duplicate case id {c.id!r}")
        seen.add(c.id)
    cost_raw = doc.get("cost_bound", {})
    if not isinstance(cost_raw, dict):
        raise SnapshotCorrupted("suite: cost_bound must be a mapping")
    cost_bound = CostBound(
        max_input_tokens=int(cost_raw.get("max_input_tokens", 1500)),
        max_output_tokens=int(cost_raw.get("max_output_tokens", 120)),
    )
    return EvalSuite(name=name, version=suite_version, cases=cases, cost_bound=cost_bound)


class SnapshotStore:
    """Append-only on-disk store of immutable benchmark versions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- paths -----------------------------------------------------------
    def _suite_dir(self, suite_name: str) -> Path:
        return self.root / suite_name

    def _version_dir(self, suite_name: str, version_id: str) -> Path:
        return self._suite_dir(suite_name) / version_id

    @staticmethod
    def _check_path_segment(segment: str, what: str) -> None:
        if not segment or not isinstance(segment, str):
            raise SnapshotError(f"{what} must be a non-empty string")
        if segment in (".", "..") or "/" in segment or "\\" in segment:
            raise SnapshotError(f"{what} {segment!r} is not a safe path segment")
        if segment.startswith("."):
            raise SnapshotError(
                f"{what} {segment!r} must not start with '.' "
                "(dot-directories are hidden from listing/verification)"
            )

    # Crash-orphaned claim dirs (SIGKILL between the exclusive mkdir and
    # the atomic rename) are reclaimed only once stale: the claim->rename
    # window is two small file writes, so a claim dir older than this is
    # essentially certainly orphaned, not a concurrent publisher mid-flight.
    _STALE_CLAIM_SECONDS = 60.0

    def _claim_version_dir(self, suite_name: str, version_id: str) -> Path:
        """Exclusively claim the version directory for a new publish.

        Returns the claimed (empty, owned) directory. Raises
        :class:`VersionExistsError` if the id is taken or a live claim is
        in flight; reclaims a stale crash-orphaned claim dir.
        """
        self._suite_dir(suite_name).mkdir(parents=True, exist_ok=True)
        vdir = self._version_dir(suite_name, version_id)
        if vdir.exists():
            if (vdir / "manifest.json").is_file():
                raise VersionExistsError(
                    f"version {version_id!r} of suite {suite_name!r} already exists — "
                    "versions are immutable; publish a new version id instead"
                )
            try:
                empty = not any(vdir.iterdir())
            except OSError:
                empty = False
            if empty:
                try:
                    age = time.time() - vdir.stat().st_mtime
                except OSError:
                    age = 0.0
                if age >= self._STALE_CLAIM_SECONDS:
                    try:
                        vdir.rmdir()
                    except OSError:
                        pass
                    else:
                        return self._claim_version_dir(suite_name, version_id)
            raise VersionExistsError(
                f"version {version_id!r} of suite {suite_name!r} is already claimed "
                "(another publish in flight?)"
            )
        try:
            vdir.mkdir(exist_ok=False)
        except FileExistsError:
            # Lost the race: recurse once to get the precise reason.
            return self._claim_version_dir(suite_name, version_id)
        return vdir

    @staticmethod
    def _check_version_id(version_id: str) -> None:
        SnapshotStore._check_path_segment(version_id, "version_id")

    # -- publish ---------------------------------------------------------
    def publish(
        self,
        suite: EvalSuite,
        version_id: str,
        meta: VersionMeta,
        baseline: BaselineRecord,
        *,
        prev_version_id: str | None = None,
        promotions: list[dict[str, Any]] | None = None,
        created_at: str | None = None,
    ) -> BenchmarkVersion:
        """Freeze ``suite`` as a new immutable version.

        Raises :class:`VersionExistsError` if ``version_id`` is already
        published (never overwrites), and :class:`VersionNotFoundError` if
        ``prev_version_id`` names a version that does not exist — the chain
        must be unbroken.
        """
        self._check_version_id(version_id)
        self._check_path_segment(suite.name, "suite name")
        if prev_version_id is not None:
            self._check_version_id(prev_version_id)
            if not self._version_dir(suite.name, prev_version_id).exists():
                raise VersionNotFoundError(
                    f"prev version {prev_version_id!r} of suite {suite.name!r} "
                    "does not exist — cannot extend a broken chain"
                )

        suite_bytes = suite_to_canonical(suite)
        digest = suite_digest(suite_bytes)
        # Publish enforces exactly what load enforces. A suite that would
        # fail the schema or canonical fixed-point check at load (duplicate
        # case ids, non-normalized types) is rejected HERE — otherwise the
        # version id would be bricked: unloadable and unrepublishable.
        try:
            probe = suite_from_canonical(suite_bytes)
        except SnapshotCorrupted as e:
            raise SnapshotError(
                f"suite {suite.name!r} failed pre-publish validation: {e}"
            ) from e
        if suite_to_canonical(probe) != suite_bytes:
            raise SnapshotError(
                f"suite {suite.name!r} is not canonical-stable "
                "(check declared dataclass types, e.g. float weights) — "
                "refusing to publish an unloadable version"
            )
        # Same bricking class for the manifest payloads: a type-violating
        # BaselineRecord/VersionMeta (built by bypassing the typed
        # constructors) would write a manifest that load() rejects, with the
        # id already claimed. Round-trip through the validators pre-claim.
        try:
            VersionMeta.from_dict(meta.to_dict())
            BaselineRecord.from_dict(baseline.to_dict())
        except (SnapshotCorrupted, ValueError, TypeError) as e:
            raise SnapshotError(
                f"version metadata/baseline failed pre-publish validation: {e}"
            ) from e
        # Structural validation of promotion records (semantic kind
        # validation lives in regression.PromotionRecord — snapshots must not
        # import the workflow layer). A garbage dict would otherwise land in
        # the manifest unchecked.
        for p in promotions or []:
            if not isinstance(p, dict):
                raise SnapshotError(
                    f"promotion records must be mappings, got {type(p).__name__}"
                )
            for key in ("failure_id", "kind", "promoted_case_id"):
                value = p.get(key)
                if not isinstance(value, str) or not value:
                    raise SnapshotError(
                        f"promotion record missing non-empty {key!r}"
                    )
        created = created_at or datetime.now(UTC).isoformat()
        manifest: dict[str, Any] = {
            "tool": "agenteval-bench",
            "kind": "benchmark-snapshot",
            "format": SNAPSHOT_FORMAT,
            "suite_name": suite.name,
            "version_id": version_id,
            "digest": digest,
            "prev_version_id": prev_version_id,
            "created_at": created,
            "meta": meta.to_dict(),
            "baseline": baseline.to_dict(),
            "promotions": list(promotions or []),
        }
        manifest_bytes = (
            json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8")

        # Stage in a temp dir inside the suite dir, then atomically rename —
        # a crash mid-publish never leaves a half-written version under its
        # final id. The version id is claimed EXCLUSIVELY via _claim_version_dir
        # (atomic mkdir gate; stale crash-orphaned claims reclaimed): two
        # concurrent publishers of the same id cannot both succeed (TOCTOU closed).
        vdir = self._claim_version_dir(suite.name, version_id)
        # Drop crash-orphaned staging dirs for this version id (a SIGKILL
        # between mkdtemp and os.replace leaves them; they are hidden from
        # listing/verification by design, so clean them here, never silently).
        staging_prefix = f".{version_id}."
        for child in self._suite_dir(suite.name).iterdir():
            if child.is_dir() and child.name.startswith(staging_prefix):
                shutil.rmtree(child, ignore_errors=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{version_id}.", dir=self._suite_dir(suite.name))
        )
        try:
            (staging / "suite.json").write_bytes(suite_bytes)
            (staging / "manifest.json").write_bytes(manifest_bytes)
            # vdir is an empty dir we own: rename replaces it atomically.
            os.replace(staging, vdir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            # Release the claim so a retry (or an operator) is not stuck
            # behind our empty directory.
            try:
                vdir.rmdir()
            except OSError:
                pass
            raise

        self._journal_append(
            suite_name=suite.name,
            version_id=version_id,
            digest=digest,
            manifest_digest=_sha256_hex(manifest_bytes),
            meta_digest=_sha256_hex(_canonical_json(meta.to_dict())),
            baseline_digest=_sha256_hex(_canonical_json(baseline.to_dict())),
        )

        return BenchmarkVersion(
            suite_name=suite.name,
            version_id=version_id,
            digest=digest,
            prev_version_id=prev_version_id,
            created_at=created,
            meta=meta,
            baseline=baseline,
            promotions=tuple(promotions or []),
        )

    # -- publish journal -------------------------------------------------
    # The journal is a hash-chained append-only log of every publish. It
    # binds the manifest's semantic content (version metadata, baseline)
    # to the suite digest: editing the manifest's release-gate inputs
    # without touching suite.json is detected at load time. It is
    # tamper-*evidence* against partial mutation, not a defense against a
    # party that rewrites the whole store directory (documented above).
    @property
    def _journal_path(self) -> Path:
        return self.root / "journal.jsonl"

    def _read_journal_lines(self) -> list[str]:
        path = self._journal_path
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8").splitlines()

    def _journal_entries(self) -> list[dict[str, Any]]:
        """Parse journal entries.

        A corrupt FINAL line is a torn tail (the single-line append never
        completed after a crash) and is dropped from reads — the entry it
        belonged to then counts as missing, which fails closed at load.
        A corrupt non-final line cannot come from a crashed append (appends
        only touch the tail), so it is tampering and raises.
        """
        lines = [l for l in self._read_journal_lines() if l.strip()]
        entries: list[dict[str, Any]] = []
        for i, line in enumerate(lines):
            last = i == len(lines) - 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                if last:
                    continue  # torn tail
                raise SnapshotCorrupted(
                    f"journal.jsonl line {i + 1} is not valid JSON: {e}"
                ) from e
            if not isinstance(entry, dict):
                if last:
                    continue  # torn tail
                raise SnapshotCorrupted(f"journal.jsonl line {i + 1} is not a mapping")
            entries.append(entry)
        return entries

    def _journal_has_torn_tail(self) -> bool:
        lines = [l for l in self._read_journal_lines() if l.strip()]
        if not lines:
            return False
        try:
            entry = json.loads(lines[-1])
        except json.JSONDecodeError:
            return True
        return not isinstance(entry, dict)

    def _truncate_torn_tail(self) -> bool:
        """Physically remove a torn final journal line. Returns True if cut."""
        lines = self._read_journal_lines()
        idx = len(lines) - 1
        while idx >= 0 and not lines[idx].strip():
            idx -= 1
        if idx < 0:
            return False
        try:
            json.loads(lines[idx])
            return False
        except json.JSONDecodeError:
            pass
        kept = lines[:idx]
        self._journal_path.write_text(
            "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8"
        )
        return True

    def _journal_append(
        self,
        *,
        suite_name: str,
        version_id: str,
        digest: str,
        manifest_digest: str,
        meta_digest: str,
        baseline_digest: str,
        kind: str = "snapshot-publish",
        reason: str = "",
        pre_repair_chain_tip: str | None = None,
    ) -> None:
        def build(prev_chain: str) -> dict[str, Any]:
            entry: dict[str, Any] = {
                "tool": "agenteval-bench",
                "kind": kind,
                "format": SNAPSHOT_FORMAT,
                "suite_name": suite_name,
                "version_id": version_id,
                "digest": digest,
                "manifest_digest": "sha256:" + manifest_digest,
                "meta_digest": "sha256:" + meta_digest,
                "baseline_digest": "sha256:" + baseline_digest,
                "prev_chain": prev_chain,
            }
            if kind != "snapshot-publish":
                # Repairs are forensic events, not publishes: the operator's
                # attestation and the chain tip they saw are recorded so a
                # recovery can never silently manufacture a publish.
                entry["reason"] = reason
                entry["pre_repair_chain_tip"] = pre_repair_chain_tip
            entry["chain"] = _sha256_hex(
                _canonical_json({k: v for k, v in entry.items()})
            )
            return entry

        self._journal_append_locked(build)

    def _journal_append_locked(self, build: Callable[[str], dict[str, Any]]) -> None:
        """Append one journal entry holding an exclusive lock (POSIX).

        The tail chain is read and the entry appended under the same lock,
        so two concurrent publishers can never append with the same
        prev_chain (journal fork). A torn tail (provably incomplete final
        line from a crashed append) is truncated — it can never become a
        valid entry. A corrupt non-tail line is tampering: the append is
        refused rather than extending a compromised journal. Falls back
        to an unlocked append where fcntl is unavailable.
        """
        path = self._journal_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import fcntl

            have_fcntl = True
        except ImportError:  # pragma: no cover - non-POSIX platforms
            have_fcntl = False
        with open(path, "a+", encoding="utf-8") as f:
            if have_fcntl:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                lines = [l for l in f.read().splitlines() if l.strip()]
                clean: list[str] = []
                for i, raw in enumerate(lines):
                    try:
                        json.loads(raw)
                    except json.JSONDecodeError:
                        if i == len(lines) - 1:
                            break  # torn tail: provably incomplete, dropped below
                        raise SnapshotCorrupted(
                            "journal.jsonl: corrupt non-tail entry — refusing to "
                            "extend a tampered journal"
                        ) from None
                    clean.append(raw)
                if len(clean) != len(lines):
                    f.seek(0)
                    f.truncate()
                    if clean:
                        f.write("\n".join(clean) + "\n")
                prev_chain = (
                    json.loads(clean[-1])["chain"] if clean else JOURNAL_GENESIS
                )
                entry = build(prev_chain)
                f.seek(0, 2)
                f.write(json.dumps(entry, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                if have_fcntl:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _journal_lookup(self, suite_name: str, version_id: str) -> dict[str, Any]:
        matches = [
            e
            for e in self._journal_entries()
            if e.get("suite_name") == suite_name and e.get("version_id") == version_id
        ]
        if not matches:
            raise SnapshotCorrupted(
                f"{suite_name}/{version_id}: no publish journal entry — "
                "the version directory exists but was never journaled "
                "(crash between publish and journal append? verify the version "
                "out of band, then re-journal with "
                "`snapshot repair --reason <why-you-trust-it>`)"
            )
        if len(matches) > 1:
            raise SnapshotCorrupted(
                f"{suite_name}/{version_id}: {len(matches)} journal entries — "
                "journal forked, manual inspection required"
            )
        return matches[0]

    def _journal_verify_chain(self) -> list[str]:
        """Verify the journal's hash chain; returns problem descriptions."""
        problems: list[str] = []
        prev = JOURNAL_GENESIS
        for lineno, entry in enumerate(self._journal_entries(), 1):
            if entry.get("prev_chain") != prev:
                problems.append(
                    f"journal.jsonl line {lineno}: chain break "
                    f"(prev_chain != expected)"
                )
                # Re-anchor so one break doesn't cascade into noise.
                prev = entry.get("chain", prev)
                continue
            recomputed = _sha256_hex(
                _canonical_json({k: v for k, v in entry.items() if k != "chain"})
            )
            if entry.get("chain") != recomputed:
                problems.append(f"journal.jsonl line {lineno}: chain hash mismatch")
            prev = entry.get("chain", prev)
        return problems

    def repair_journal(self, reason: str) -> list[str]:
        """Re-append missing journal entries after full re-verification.

        Operator action for crash recovery (publish succeeded but the
        process died before the journal append). Each repaired version is
        fully re-verified (suite bytes vs manifest digest + canonical
        fixed point) before its entry is appended.

        The repair is journaled as ``snapshot-repair`` — never as a publish —
        with the operator's ``reason`` (required: the attestation for why
        this version is trusted) and the chain tip seen before repair, so
        recovery leaves a forensic trace instead of silently manufacturing
        a publish. Returns the repaired version ids.
        """
        if not reason:
            raise SnapshotError("repair_journal requires a reason (operator attestation)")
        # Heal a torn tail first: the incomplete line is provably not a
        # completed entry, so truncating it loses nothing.
        self._truncate_torn_tail()
        repaired: list[str] = []
        if not self.root.is_dir():
            return repaired
        entries = self._journal_entries()
        pre_repair_tip = entries[-1]["chain"] if entries else JOURNAL_GENESIS
        known = {
            (e.get("suite_name"), e.get("version_id")) for e in entries
        }
        for sdir in sorted(self.root.iterdir()):
            if not sdir.is_dir() or sdir.name.startswith("."):
                continue
            for vdir in sorted(sdir.iterdir()):
                if not vdir.is_dir() or vdir.name.startswith("."):
                    continue
                if (sdir.name, vdir.name) in known:
                    continue
                # Full verification first: never journal a corrupt version.
                version, _suite, manifest_bytes = self._load_unjournaled(
                    sdir.name, vdir.name
                )
                self._journal_append(
                    suite_name=version.suite_name,
                    version_id=version.version_id,
                    digest=version.digest,
                    manifest_digest=_sha256_hex(manifest_bytes),
                    meta_digest=_sha256_hex(_canonical_json(version.meta.to_dict())),
                    baseline_digest=_sha256_hex(
                        _canonical_json(version.baseline.to_dict())
                    ),
                    kind="snapshot-repair",
                    reason=reason,
                    pre_repair_chain_tip=pre_repair_tip,
                )
                pre_repair_tip = self._journal_entries()[-1]["chain"]
                repaired.append(f"{sdir.name}/{vdir.name}")
        return repaired

    def _load_unjournaled(
        self, suite_name: str, version_id: str
    ) -> tuple[BenchmarkVersion, EvalSuite, bytes]:
        """load() without the journal cross-check (for repair only).

        Returns the version, the suite, and the raw manifest bytes (whose
        hash the journal binds).
        """
        self._check_path_segment(suite_name, "suite name")
        self._check_version_id(version_id)
        vdir = self._version_dir(suite_name, version_id)
        if not vdir.is_dir():
            raise VersionNotFoundError(
                f"version {version_id!r} of suite {suite_name!r} is not published"
            )
        try:
            suite_bytes = (vdir / "suite.json").read_bytes()
        except OSError as e:
            raise SnapshotCorrupted(f"{vdir}/suite.json unreadable: {e}") from e
        try:
            manifest_bytes = (vdir / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise SnapshotCorrupted(f"{vdir}/manifest.json unreadable: {e}") from e

        if not isinstance(manifest, dict) or manifest.get("format") != SNAPSHOT_FORMAT:
            raise SnapshotCorrupted(f"{vdir}/manifest.json has wrong format marker")
        recorded = manifest.get("digest")
        if suite_digest(suite_bytes) != recorded:
            raise SnapshotCorrupted(
                f"hash mismatch for {suite_name}/{version_id}: on-disk suite bytes "
                f"no longer match manifest digest {recorded!r} — the snapshot was mutated"
            )
        suite = suite_from_canonical(suite_bytes)
        if suite_to_canonical(suite) != suite_bytes:
            raise SnapshotCorrupted(
                f"canonical fixed-point broken for {suite_name}/{version_id}"
            )
        if suite.name != suite_name or manifest.get("version_id") != version_id:
            raise SnapshotCorrupted(f"manifest identity drift for {suite_name}/{version_id}")
        promotions_raw = manifest.get("promotions", [])
        if not isinstance(promotions_raw, list):
            raise SnapshotCorrupted(f"{vdir}/manifest.json: promotions must be a list")
        version = BenchmarkVersion(
            suite_name=suite_name,
            version_id=version_id,
            digest=str(recorded),
            prev_version_id=manifest.get("prev_version_id"),
            created_at=str(manifest.get("created_at", "")),
            meta=VersionMeta.from_dict(manifest.get("meta", {})),
            baseline=BaselineRecord.from_dict(manifest.get("baseline", {})),
            promotions=tuple(promotions_raw),
        )
        return version, suite, manifest_bytes

    # -- load ------------------------------------------------------------
    def load(self, suite_name: str, version_id: str) -> tuple[BenchmarkVersion, EvalSuite]:
        """Load a version, hash-verifying immutability.

        Verifies the suite bytes against the manifest digest, the canonical
        fixed point, and the manifest's semantic content (version metadata,
        baseline) against the hash-chained publish journal. Raises
        :class:`VersionNotFoundError` if the version was never published;
        :class:`SnapshotCorrupted` on any tampering, schema violation, or
        journal mismatch.
        """
        version, suite, manifest_bytes = self._load_unjournaled(suite_name, version_id)
        entry = self._journal_lookup(suite_name, version_id)
        manifest_digest = "sha256:" + _sha256_hex(manifest_bytes)
        if entry.get("manifest_digest") != manifest_digest:
            raise SnapshotCorrupted(
                f"{suite_name}/{version_id}: journal manifest digest mismatch — "
                "manifest.json was edited after publishing "
                "(chain, timestamps, promotions, or metadata)"
            )
        if entry.get("digest") != version.digest:
            raise SnapshotCorrupted(
                f"{suite_name}/{version_id}: journal digest != manifest digest — "
                "the version was replaced after publishing"
            )
        meta_digest = "sha256:" + _sha256_hex(_canonical_json(version.meta.to_dict()))
        if entry.get("meta_digest") != meta_digest:
            raise SnapshotCorrupted(
                f"{suite_name}/{version_id}: journal meta digest mismatch — "
                "the version record (owner/why/changes) was edited after publishing"
            )
        baseline_digest = "sha256:" + _sha256_hex(
            _canonical_json(version.baseline.to_dict())
        )
        if entry.get("baseline_digest") != baseline_digest:
            raise SnapshotCorrupted(
                f"{suite_name}/{version_id}: journal baseline digest mismatch — "
                "the pinned baseline was edited after publishing; "
                "the release gate refuses to run on a weakened baseline"
            )
        return version, suite

    # -- listing / verification -------------------------------------------
    def list_versions(self, suite_name: str) -> list[str]:
        """Published version ids for a suite, sorted."""
        self._check_path_segment(suite_name, "suite name")
        sdir = self._suite_dir(suite_name)
        if not sdir.is_dir():
            return []
        return sorted(p.name for p in sdir.iterdir() if p.is_dir() and not p.name.startswith("."))

    def verify(self, suite_name: str, version_id: str) -> None:
        """Hash-verify one version; raises on any problem."""
        self.load(suite_name, version_id)

    def verify_all(self) -> list[str]:
        """Hash-verify every published version + the journal chain.

        Empty list means the whole store is healthy.
        """
        problems: list[str] = []
        if not self.root.is_dir():
            return problems
        try:
            problems.extend(self._journal_verify_chain())
        except SnapshotError as e:
            problems.append(f"journal.jsonl: {e}")
        if self._journal_has_torn_tail():
            problems.append(
                "journal.jsonl: torn tail line (incomplete append; dropped from reads; "
                "run `snapshot repair` after out-of-band verification)"
            )
        for sdir in sorted(self.root.iterdir()):
            if not sdir.is_dir() or sdir.name.startswith("."):
                continue
            for vdir in sorted(sdir.iterdir()):
                if not vdir.is_dir() or vdir.name.startswith("."):
                    continue
                try:
                    self.verify(sdir.name, vdir.name)
                except SnapshotError as e:
                    problems.append(f"{sdir.name}/{vdir.name}: {e}")
        return problems


__all__ = [
    "SNAPSHOT_FORMAT",
    "BaselineRecord",
    "BenchmarkVersion",
    "SnapshotCorrupted",
    "SnapshotError",
    "SnapshotStore",
    "VersionExistsError",
    "VersionMeta",
    "VersionNotFoundError",
    "baseline_from_run",
    "suite_from_canonical",
    "suite_to_canonical",
]
