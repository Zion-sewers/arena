"""Judge decorrelation: HOST != FAMILY (arena#44).

Judges whose findings are compared must come from different MODEL FAMILIES.
Two hosts serving the same weights carry zero information: the documented
trap is nvidia-nim + openrouter silently serving the same model twice while
the gate reports information-free agreement. Provider-name diversity is
NOT decorrelation, so allocation is keyed on ``model_family``, never on
``host`` or provider name.

Rules (adapted from the issue's named source, claude-setup
``tools/review/actors.json`` catalog A.5 + ``allocate.py`` — the semantics,
not the code):
* allocation is keyed on model family; same-family pairs are rejected and
  the rejection is named in the report;
* ``VARIES-BY-MODEL`` is not a family: a reseller whose served model varies
  can never fill a second slot, because its family cannot be shown to
  differ from a partner it may itself be serving. It may be picked only as
  a single opinion when nothing else is available;
* unreachable judges appear in the report as explicit holes — never picked,
  never silently dropped;
* reachability is a declared property of the registry (``measured`` /
  ``declared`` / ``unreachable``), or a live measurement when the caller
  supplies a ``probe``. A probe that returns True stamps the judge's
  effective reachability as ``"measured"`` in the report (and the override
  is noted when it differs from the declaration), so a picked judge is
  never labelled unreachable. The module itself makes no network call:
  arena has no live judge transport (README: "LLM-as-judge is planned, not
  implemented"), so a probe is an injectable callable the caller owns.
* the report accounts for every registry judge exactly once — picked,
  holes, rejected, or surplus (reachable judges with distinct families the
  quota did not need) — so nothing is silently dropped.

Everything is stdlib-only and deterministic: candidates are ordered by
(reachability rank, id), so the same registry always allocates the same way.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

#: A family that varies by model is not a family. Pairing on it is the
#: correlated pair the registry's own contract warns about.
UNKNOWN_FAMILY = "VARIES-BY-MODEL"

#: Reachability ranks; unreachable judges are ordered last and REPORTED,
#: never silently dropped. Frozen: the ordering the determinism and
#: hole-reporting guarantees rest on cannot be redefined by an importer.
REACH_RANK: Mapping[str, int] = MappingProxyType(
    {"measured": 0, "declared": 1, "unreachable": 2}
)

Reachability = Literal["measured", "declared", "unreachable"]


@dataclass(frozen=True)
class Judge:
    """One judge: where it runs, and which weights it runs.

    ``host`` names the provider serving the judge; ``model_family`` names
    the weights. The two are independent: one host can serve many families
    (nvidia-nim serves deepseek, qwen, llama) and one family can be served
    by many hosts (deepseek via nvidia-nim and via openrouter).
    """

    id: str
    host: str
    model_family: str
    model: str = ""
    reachable: Reachability = "declared"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("judge id must be non-empty")
        if not self.host:
            raise ValueError(f"judge {self.id!r} host must be non-empty")
        if not self.model_family:
            raise ValueError(f"judge {self.id!r} model_family must be non-empty")
        if self.reachable not in REACH_RANK:
            raise ValueError(
                f"judge {self.id!r} reachable must be one of "
                f"{sorted(REACH_RANK)}, got {self.reachable!r}"
            )


@dataclass(frozen=True)
class JudgeHole:
    """An unreachable judge, reported as an explicit hole in the allocation.

    A hole is never a silent drop: the report names the judge, where it
    runs, which family it would have been, and why it is unreachable.
    """

    judge: str
    host: str
    model_family: str
    reason: str


@dataclass(frozen=True)
class RejectedJudge:
    """A reachable judge not picked because its family was already taken."""

    judge: str
    family: str
    reason: str


@dataclass(frozen=True)
class AllocationReport:
    """The result of one allocation: picks, holes, rejections, notes.

    The report accounts for every registry judge exactly once across
    ``picked``, ``holes``, ``rejected`` and ``surplus`` — nothing is
    silently dropped and nothing is double-counted. ``surplus`` holds
    reachable judges with distinct families the quota did not need.
    """

    want: int
    picked: tuple[Judge, ...] = ()
    families: tuple[str, ...] = ()
    holes: tuple[JudgeHole, ...] = ()
    rejected: tuple[RejectedJudge, ...] = ()
    surplus: tuple[Judge, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        picked_families = [j.model_family for j in self.picked]
        if len(set(picked_families)) != len(picked_families):
            raise ValueError("picked judges must carry distinct model families")
        if tuple(picked_families) != self.families:
            raise ValueError("families must match the picked judges' model families")
        buckets = {
            "picked": {j.id for j in self.picked},
            "holes": {h.judge for h in self.holes},
            "rejected": {r.judge for r in self.rejected},
            "surplus": {j.id for j in self.surplus},
        }
        seen: dict[str, str] = {}
        for name, ids in buckets.items():
            for judge_id in ids:
                if judge_id in seen:
                    raise ValueError(
                        f"judge {judge_id!r} appears in both {seen[judge_id]} and {name}"
                    )
                seen[judge_id] = name
        for rejected in self.rejected:
            if (
                rejected.family not in picked_families
                and rejected.family != UNKNOWN_FAMILY
            ):
                raise ValueError(
                    f"rejected judge {rejected.judge!r} names family "
                    f"{rejected.family!r} that was not picked"
                )

    @property
    def satisfied(self) -> bool:
        """True when the allocation supplied the requested number of judges."""
        return len(self.picked) >= self.want

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form of the report."""
        return {
            "want": self.want,
            "satisfied": self.satisfied,
            "picked": [
                {
                    "id": j.id,
                    "host": j.host,
                    "model_family": j.model_family,
                    "model": j.model,
                    "reachable": j.reachable,
                }
                for j in self.picked
            ],
            "families": list(self.families),
            "holes": [
                {
                    "judge": h.judge,
                    "host": h.host,
                    "model_family": h.model_family,
                    "reason": h.reason,
                }
                for h in self.holes
            ],
            "rejected": [
                {"judge": r.judge, "family": r.family, "reason": r.reason} for r in self.rejected
            ],
            "surplus": [
                {
                    "id": j.id,
                    "host": j.host,
                    "model_family": j.model_family,
                    "model": j.model,
                    "reachable": j.reachable,
                }
                for j in self.surplus
            ],
            "notes": list(self.notes),
        }


def _sort_key(judge: Judge) -> tuple[int, str]:
    return (REACH_RANK[judge.reachable], judge.id)


def allocate(
    judges: Sequence[Judge],
    want: int = 2,
    probe: Callable[[Judge], bool] | None = None,
) -> AllocationReport:
    """Pick up to ``want`` judges, no two sharing a model_family.

    ``probe``, when supplied, is the authority on reachability: it is
    called exactly once per judge, BEFORE family filtering, so a judge
    that ends up surplus or rejected is probed too. A judge it reports
    unreachable (or one it raises on) becomes a hole with the reason
    recorded. A judge it reports reachable has its effective reachability
    stamped ``"measured"`` in the report, and the override is noted when
    it differs from the declaration — a picked judge is never labelled
    unreachable. Candidates are ordered by EFFECTIVE reachability, so a
    probe-revived judge competes on its measured rank, not its stale
    declared rank. A probe that returns a non-bool is a programming
    error and raises ``TypeError`` loudly instead of being read as
    truthiness. Without a probe, the registry's declared reachability
    rules. Either way an unreachable judge is reported, never dropped.
    """
    if want < 1:
        raise ValueError(f"want must be >= 1, got {want}")
    ids = [j.id for j in judges]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate judge id(s): {', '.join(dupes)}")
    if not judges:
        return AllocationReport(
            want=want,
            notes=("no judges in the registry; nothing to allocate",),
        )

    # Phase 1 — effective reachability: the probe, when given, overrides
    # the declaration (a declared-unreachable judge a probe can reach is
    # reachable again); a probe failure is a hole, never a crash.
    reachable: list[Judge] = []
    holes: list[JudgeHole] = []
    notes: list[str] = []
    for judge in judges:
        if probe is not None:
            try:
                alive = probe(judge)
            except Exception as exc:  # noqa: BLE001 — probe failure is a hole
                holes.append(
                    JudgeHole(
                        judge.id,
                        judge.host,
                        judge.model_family,
                        f"probe raised {type(exc).__name__}: {exc}",
                    )
                )
                continue
            if not isinstance(alive, bool):
                raise TypeError(
                    f"probe for judge {judge.id!r} must return bool, "
                    f"got {type(alive).__name__}"
                )
            if not alive:
                holes.append(
                    JudgeHole(
                        judge.id,
                        judge.host,
                        judge.model_family,
                        "probe reported unreachable",
                    )
                )
                continue
            if judge.reachable != "measured":
                notes.append(
                    f"{judge.id}: probe measured reachable "
                    f"(declared {judge.reachable})"
                )
                reachable.append(replace(judge, reachable="measured"))
            else:
                reachable.append(judge)
        elif judge.reachable == "unreachable":
            holes.append(
                JudgeHole(
                    judge.id,
                    judge.host,
                    judge.model_family,
                    "declared unreachable",
                )
            )
        else:
            reachable.append(judge)
    holes.sort(key=lambda h: h.judge)

    rejected: list[RejectedJudge] = []
    surplus: list[Judge] = []
    picked: list[Judge] = []
    used_families: set[str] = set()

    # An UNKNOWN_FAMILY judge is admissible ONLY as a solitary reviewer.
    # Its family cannot be shown to DIFFER from anything, so it cannot
    # satisfy decorrelation against any partner, in either slot order.
    # Phase 2 orders by effective reachability, so the probe's verdict —
    # not the stale declaration — decides who competes first.
    known = [
        j for j in sorted(reachable, key=_sort_key) if j.model_family != UNKNOWN_FAMILY
    ]
    unknown = [
        j for j in sorted(reachable, key=_sort_key) if j.model_family == UNKNOWN_FAMILY
    ]

    for candidate in known:
        if candidate.model_family in used_families:
            first = next(p.id for p in picked if p.model_family == candidate.model_family)
            rejected.append(
                RejectedJudge(
                    candidate.id,
                    candidate.model_family,
                    f"same model_family {candidate.model_family!r} as {first!r}; "
                    "host diversity is not decorrelation",
                )
            )
            continue
        if len(picked) >= want:
            surplus.append(candidate)
            continue
        picked.append(candidate)
        used_families.add(candidate.model_family)

    if not picked and unknown:
        solo = unknown[0]
        picked.append(solo)
        used_families.add(UNKNOWN_FAMILY)
        notes.append(
            f"{solo.id} is the only reachable judge and its model_family is "
            f"{UNKNOWN_FAMILY}. Usable as a single opinion; it can never be "
            "half of a decorrelated pair, because a reseller family cannot "
            "be shown to differ from a partner it may itself be serving."
        )
        for extra in unknown[1:]:
            rejected.append(
                RejectedJudge(
                    extra.id,
                    UNKNOWN_FAMILY,
                    f"same model_family {UNKNOWN_FAMILY!r} as {solo.id!r}; "
                    "a reseller family cannot be shown to differ from a "
                    "partner it may itself be serving",
                )
            )
    elif unknown:
        for extra in unknown:
            rejected.append(
                RejectedJudge(
                    extra.id,
                    UNKNOWN_FAMILY,
                    f"a {UNKNOWN_FAMILY} family cannot be shown to differ "
                    "from the judges picked",
                )
            )
        notes.append(
            f"{', '.join(j.id for j in unknown)} excluded: a "
            f"{UNKNOWN_FAMILY} family cannot be shown to differ from the "
            "judges picked."
        )

    if len(picked) < want:
        notes.append(
            f"wanted {want} decorrelated judge(s), found {len(picked)}. The "
            f"registry does not supply {want} distinct reachable model "
            "families."
        )

    return AllocationReport(
        want=want,
        picked=tuple(picked),
        families=tuple(j.model_family for j in picked),
        holes=tuple(holes),
        rejected=tuple(rejected),
        surplus=tuple(surplus),
        notes=tuple(notes),
    )


def load_registry(data: object) -> tuple[Judge, ...]:
    """Parse a judge registry into typed judges, failing closed.

    Accepts a mapping with a ``judges`` list, or a bare list of judge
    entries. Each entry must be a mapping with ``id``, ``host`` and
    ``model_family`` (non-empty strings); ``model`` and ``reachable``
    are optional (defaults ``""`` and ``"declared"``). Unknown keys are
    rejected so a typo (``model_familiy``) cannot silently become a judge
    with no declared family.
    """
    if isinstance(data, Mapping):
        raw = data.get("judges")
        if raw is None:
            raise ValueError("registry mapping must carry a 'judges' list")
    elif isinstance(data, Sequence):
        raw = data
    else:
        raise TypeError("registry must be a mapping with 'judges' or a list")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("'judges' must be a list of judge entries")

    allowed = {"id", "host", "model_family", "model", "reachable"}
    judges: list[Judge] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise TypeError(f"judge entry must be a mapping, got {entry!r}")
        unknown_keys = sorted(set(entry) - allowed)
        if unknown_keys:
            raise ValueError(
                f"judge entry {entry.get('id', '?')!r} has unknown key(s): "
                f"{', '.join(unknown_keys)}"
            )
        for field_name in ("id", "host", "model_family"):
            if field_name not in entry:
                raise ValueError(f"judge entry {entry!r} is missing required field {field_name!r}")
            if not isinstance(entry[field_name], str):
                raise TypeError(
                    f"judge {entry.get('id', '?')!r} field {field_name!r} must be a string"
                )
        reachable = entry.get("reachable", "declared")
        if reachable not in REACH_RANK:
            raise ValueError(
                f"judge {entry.get('id', '?')!r} reachable must be one of "
                f"{sorted(REACH_RANK)}, got {reachable!r}"
            )
        model = entry.get("model", "")
        if not isinstance(model, str):
            raise TypeError(f"judge {entry.get('id', '?')!r} field 'model' must be a string")
        judges.append(
            Judge(
                id=entry["id"],
                host=entry["host"],
                model_family=entry["model_family"],
                model=model,
                reachable=reachable,
            )
        )
    ids = [j.id for j in judges]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate judge id(s): {', '.join(dupes)}")
    return tuple(judges)


def registry_from_json(text: str) -> tuple[Judge, ...]:
    """Parse a registry JSON document (see :func:`load_registry`)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"registry is not valid JSON: {exc}") from exc
    return load_registry(data)
