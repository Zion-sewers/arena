"""Bug generators for the SWE-smith pipeline (agenteval-bench#37).

Three strategies, one contract: every generator yields :class:`BugCandidate`
objects whose provenance is recorded in a :class:`GenerationRecord`.

* :class:`ProceduralBugGenerator` — deterministic AST-level fault
  injection. ``random.Random(seed)`` drives site selection, so the same
  ``(seed, repo)`` always yields byte-identical patches.
* :class:`LLMGeneratedBugGenerator` — model-rewritten functions. The model
  itself is an *injected* callable: constructing without a ``rewrite_fn``
  raises instead of silently falling back to another strategy (criterion 4
  of the issue: no silent substitution of the generation mechanism).
* :class:`PRMirrorGenerator` — wraps an explicit, real patch (a reverted
  PR) as a candidate. Not seeded; reproducibility comes from the recorded
  patch bytes themselves.
"""

from __future__ import annotations

import ast
import copy
import difflib
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bugsmith.types import BugCandidate, BugsmithError, BugStrategy, GenerationRecord

GENERATOR_VERSION = "bugsmith-buggen/1"


class RewriteFn(Protocol):
    """Injected model rewrite: (source, file, hint) -> mutated source."""

    def __call__(self, source: str, target_file: str, hint: str) -> str: ...


@dataclass(frozen=True)
class _Site:
    lineno: int
    col_offset: int
    kind: type
    description: str
    mutate: Callable[[ast.AST], ast.AST]  # node -> mutated node (in place copy)


def _swap_binop(node: ast.BinOp) -> ast.BinOp:
    pairs = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div,
             ast.Div: ast.Mult, ast.FloorDiv: ast.Mod, ast.Mod: ast.FloorDiv}
    cls = pairs.get(type(node.op))
    if cls is None:
        raise BugsmithError(f"no binop swap for {type(node.op).__name__}")
    return ast.BinOp(left=node.left, op=cls(), right=node.right)


def _flip_compare(node: ast.Compare) -> ast.Compare:
    pairs = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE,
             ast.GtE: ast.LtE, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
    ops = [pairs.get(type(op)) for op in node.ops]
    if any(o is None for o in ops):
        raise BugsmithError("unsupported comparison operator")
    return ast.Compare(left=node.left, ops=[o() for o in ops if o], comparators=node.comparators)


def _flip_boolop(node: ast.BoolOp) -> ast.BoolOp:
    cls = ast.Or if isinstance(node.op, ast.And) else ast.And
    return ast.BoolOp(op=cls(), values=node.values)


def _off_by_one_range(node: ast.Call) -> ast.Call:
    # range(n) -> range(n + 1); range(a, b[, step]) mutates the STOP bound,
    # never the step (mutating the step is not an off-by-one fault).
    args = list(node.args)
    if len(args) == 1:
        args[0] = ast.BinOp(left=args[0], op=ast.Add(), right=ast.Constant(value=1))
    elif len(args) >= 2:
        args[1] = ast.BinOp(left=args[1], op=ast.Sub(), right=ast.Constant(value=1))
    else:
        raise BugsmithError("range() with no args")
    return ast.Call(func=node.func, args=args, keywords=node.keywords)


def _negate_constant(node: ast.Constant) -> ast.Constant:
    v = node.value
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BugsmithError("not a numeric constant")
    return ast.Constant(value=-v if v != 0 else 1)


def _mutate_return_number(node: ast.Return) -> ast.Return:
    if not isinstance(node.value, ast.Constant):
        raise BugsmithError("return value is not a constant")
    return ast.Return(value=_negate_constant(node.value))


def _collect_sites(tree: ast.AST) -> list[_Site]:
    sites: list[_Site] = []

    def desc(node: ast.AST, what: str) -> str:
        return f"{what} at line {node.lineno}:{node.col_offset}"

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in (
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
        ):
            sites.append(_Site(node.lineno, node.col_offset, ast.BinOp, desc(node, "binop-swap"), _swap_binop))
        elif isinstance(node, ast.Compare) and all(
            type(op) in (ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq)
            for op in node.ops
        ):
            sites.append(_Site(node.lineno, node.col_offset, ast.Compare, desc(node, "compare-flip"), _flip_compare))
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            sites.append(_Site(node.lineno, node.col_offset, ast.BoolOp, desc(node, "boolop-flip"), _flip_boolop))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and 1 <= len(node.args) <= 3
            and not node.keywords
        ):
            sites.append(_Site(node.lineno, node.col_offset, ast.Call, desc(node, "range-off-by-one"), _off_by_one_range))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            sites.append(
                _Site(node.lineno, node.col_offset, ast.Constant, desc(node, "constant-negate"), _negate_constant)
            )
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            sites.append(
                _Site(node.lineno, node.col_offset, ast.Return, desc(node, "return-constant-mutate"), _mutate_return_number)
            )
    return sites


def _find_node(tree: ast.AST, site: _Site) -> ast.AST | None:
    for node in ast.walk(tree):
        if (
            type(node) is site.kind
            and getattr(node, "lineno", None) == site.lineno
            and getattr(node, "col_offset", None) == site.col_offset
        ):
            return node
    return None


def _mutate_source(source: str, site: _Site) -> str:
    """Splice a single mutated node back into the source text.

    Only the node's own source segment (lineno/col_offset ..
    end_lineno/end_col_offset) is replaced, so generated patches show
    exactly the injected fault and the surrounding statement stays
    intact. The result is re-parsed: the generator never emits
    unparseable code.
    """
    tree = ast.parse(source)
    target = _find_node(tree, site)
    if target is None:
        raise BugsmithError(f"site not found: {site.description}")
    mutated = site.mutate(copy.deepcopy(target))
    ast.fix_missing_locations(mutated)
    new_segment = ast.unparse(mutated)
    lines = source.splitlines(keepends=True)
    s_line, s_col = target.lineno - 1, target.col_offset
    e_line, e_col = target.end_lineno - 1, target.end_col_offset
    prefix = lines[s_line][:s_col]
    suffix = lines[e_line][e_col:]
    cont_indent = " " * s_col
    seg_lines = new_segment.split("\n")
    spliced = prefix + seg_lines[0]
    for ln in seg_lines[1:]:
        spliced += "\n" + (cont_indent + ln if ln.strip() else ln)
    new_text = "".join(lines[:s_line]) + spliced + suffix + "".join(lines[e_line + 1 :])
    ast.parse(new_text)  # the generator never emits unparseable code
    return new_text


def _unified_patch(rel_path: str, original: str, mutated: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            mutated.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
        )
    )


class ProceduralBugGenerator:
    """Deterministic AST fault injection. Seeded: same seed -> same patches."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._seed = seed

    @property
    def strategy(self) -> BugStrategy:
        return BugStrategy.PROCEDURAL_AST

    def _source_files(
        self, repo_root: Path, exclude: tuple[str, ...] = ()
    ) -> list[Path]:
        def is_test_path(p: Path) -> bool:
            rel_parts = p.relative_to(repo_root).parts
            return any(
                part in ("test", "tests") or part.startswith("test_")
                for part in rel_parts
            )

        def is_excluded(p: Path) -> bool:
            rel = p.relative_to(repo_root).as_posix()
            return any(rel == e or rel.startswith(e.rstrip("/") + "/") for e in exclude)

        # docs/ and examples/ are not library code: mutating them is noise,
        # not a benchmark instance.
        files = sorted(
            p for p in repo_root.rglob("*.py")
            if not is_test_path(p)
            and not is_excluded(p)
            and ".git" not in p.relative_to(repo_root).parts
            and "docs" not in p.relative_to(repo_root).parts
            and "examples" not in p.relative_to(repo_root).parts
        )
        if not files:
            raise BugsmithError(f"no source files under {repo_root}")
        return files

    def generate(
        self, repo_root: Path | str, count: int, *, exclude: tuple[str, ...] = ()
    ) -> list[BugCandidate]:
        """Yield up to ``count`` candidates, each from a distinct (file, site).

        ``exclude`` lists repo-relative paths (files or directories) that
        are never mutated. The tiered pipeline excludes its own scoring
        path (``src/bugsmith``) so a mutant there cannot fabricate the
        score it is being measured by.
        """
        if count < 1:
            raise BugsmithError("count must be >= 1")
        repo_root = Path(repo_root).resolve()
        candidates: list[BugCandidate] = []
        seen: set[tuple[str, str]] = set()
        files = self._source_files(repo_root, exclude)
        attempts = 0
        # Bounded: every attempt consumes RNG state deterministically.
        while len(candidates) < count and attempts < count * 25:
            attempts += 1
            path = self._rng.choice(files)
            try:
                source = path.read_text(encoding="utf-8")
                sites = _collect_sites(ast.parse(source))
            except (OSError, SyntaxError):
                continue
            if not sites:
                continue
            site = self._rng.choice(sites)
            key = (str(path.relative_to(repo_root)), site.description)
            if key in seen:
                continue
            try:
                mutated = _mutate_source(source, site)
            except (BugsmithError, SyntaxError):
                continue
            if mutated == source:
                continue
            seen.add(key)
            rel = str(path.relative_to(repo_root))
            patch = _unified_patch(rel, source, mutated)
            if not patch.strip():
                continue
            candidates.append(
                BugCandidate(
                    record=GenerationRecord(
                        strategy=BugStrategy.PROCEDURAL_AST,
                        seed=self._seed,
                        generator_version=GENERATOR_VERSION,
                        target_file=rel,
                        site_description=site.description,
                    ),
                    patch=patch,
                )
            )
        return candidates


class LLMGeneratedBugGenerator:
    """Model-rewritten bug injection. The model is injected, never defaulted.

    Constructing without ``rewrite_fn`` raises :class:`BugsmithError`:
    silently substituting a different generation mechanism would violate
    the issue's criterion 4 (no fallback to an unrecorded mechanism).
    """

    def __init__(self, rewrite_fn: RewriteFn | None = None, seed: int | None = None) -> None:
        if rewrite_fn is None:
            raise BugsmithError(
                "LLMGeneratedBugGenerator requires an explicit rewrite_fn "
                "(no model backend configured; refusing to silently substitute "
                "another strategy)"
            )
        self._rewrite_fn = rewrite_fn
        self._seed = seed

    @property
    def strategy(self) -> BugStrategy:
        return BugStrategy.LLM_GENERATED

    def generate(
        self, repo_root: Path | str, target_files: list[str], hint: str = ""
    ) -> list[BugCandidate]:
        repo_root = Path(repo_root).resolve()
        candidates = []
        for rel in target_files:
            path = repo_root / rel
            if not path.is_file():
                raise BugsmithError(f"target file not found: {rel}")
            source = path.read_text(encoding="utf-8")
            mutated = self._rewrite_fn(source, rel, hint)
            ast.parse(mutated)  # model output must still be valid Python
            if mutated == source:
                raise BugsmithError(f"rewrite_fn returned identical source for {rel}")
            candidates.append(
                BugCandidate(
                    record=GenerationRecord(
                        strategy=BugStrategy.LLM_GENERATED,
                        seed=self._seed,
                        generator_version=GENERATOR_VERSION,
                        target_file=rel,
                        site_description=f"model rewrite (hint: {hint[:80]})",
                    ),
                    patch=_unified_patch(rel, source, mutated),
                )
            )
        return candidates


class PRMirrorGenerator:
    """Wraps an explicit real patch (a reverted PR) as a bug candidate."""

    def __init__(self, pr_ref: str) -> None:
        if not pr_ref:
            raise BugsmithError("pr_ref is required: a mirrored patch must name its PR")
        self._pr_ref = pr_ref

    @property
    def strategy(self) -> BugStrategy:
        return BugStrategy.PR_MIRROR

    def generate(self, patch: str, target_files: list[str]) -> list[BugCandidate]:
        if not patch.strip():
            raise BugsmithError("mirrored patch is empty")
        return [
            BugCandidate(
                record=GenerationRecord(
                    strategy=BugStrategy.PR_MIRROR,
                    seed=None,
                    generator_version=GENERATOR_VERSION,
                    target_file=",".join(target_files),
                    site_description=f"reverted PR {self._pr_ref}",
                    extra={"pr_ref": self._pr_ref},
                ),
                patch=patch if patch.endswith("\n") else patch + "\n",
            )
        ]
