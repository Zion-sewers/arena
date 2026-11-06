"""Detect: find fix-commits in an agent repo (arena#40).

Detect reads commit history through an injected transport. The live
transport shells out to the fleet's ``gh`` CLI (authentication lives
there, not in this repo); unit tests and the demo use hand-written stub
transports carrying recorded payloads. The parse function is pure and
tested on recorded JSON, so the live path's only untested part is the
two-line subprocess call.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
from collections.abc import Mapping
from typing import Protocol

from flywheel.types import FixCommit, FlywheelError

_FIX_PREFIX_RE = re.compile(r"^fix(\([^)]*\))?:", re.IGNORECASE)
_FIXES_ISSUE_RE = re.compile(r"\bfix(?:es|ed)?\s+#\d+", re.IGNORECASE)


def is_fix_commit(message: str) -> bool:
    """Conventional-Commit ``fix:``/``fix(scope):`` prefix, or a ``fixes
    #N`` reference. A commit that merely mentions 'fix' in passing (e.g.
    'prefix fix for the demo') is not a fix-commit."""
    first_line = message.strip().splitlines()[0] if message.strip() else ""
    return bool(_FIX_PREFIX_RE.match(first_line) or _FIXES_ISSUE_RE.search(first_line))


def sample_commits(
    commits: list[FixCommit], count: int, *, seed: int
) -> tuple[FixCommit, ...]:
    """Seeded sample: same input + seed -> same sample. Deterministic
    order (sorted by sha) before sampling, so the sample does not depend
    on the transport's listing order."""
    if count < 1:
        raise FlywheelError("sample count must be >= 1")
    ordered = sorted(commits, key=lambda c: c.sha)
    if len(ordered) <= count:
        return tuple(ordered)
    return tuple(random.Random(seed).sample(ordered, count))


class GitHubTransport(Protocol):
    """Reads fix-commits from a repo. Implementations own authentication
    and transport; this module owns classification and sampling."""

    def list_fix_commits(self, repo: str, since: str | None = None) -> list[FixCommit]:
        """All fix-commits in ``repo`` (``owner/name``), newest last."""
        ...


def parse_commits_payload(
    payload: list[Mapping[str, object]], repo: str
) -> list[FixCommit]:
    """Parse a GitHub list-commits response into FixCommits. Pure: testable
    on recorded payloads without network. Entries that are not objects,
    lack a sha, or lack a commit message are skipped, not guessed at."""
    commits: list[FixCommit] = []
    for entry in payload:
        sha = entry.get("sha")
        commit = entry.get("commit")
        if not isinstance(sha, str) or not sha:
            continue
        if not isinstance(commit, Mapping):
            continue
        message = commit.get("message")
        if not isinstance(message, str) or not message.strip():
            continue
        commits.append(FixCommit(sha=sha, message=message, repo=repo))
    return commits


class GhCliTransport:
    """Live transport: the fleet's ``gh`` CLI. Two lines of subprocess;
    everything else is the pure parse function above."""

    def list_fix_commits(self, repo: str, since: str | None = None) -> list[FixCommit]:
        url = f"/repos/{repo}/commits?per_page=100"
        if since:
            url += f"&since={since}"
        try:
            proc = subprocess.run(
                ["gh", "GET", url],
                capture_output=True,
                timeout=120,
                check=False,
                text=True,
            )
        except OSError as e:
            raise FlywheelError(f"gh CLI unavailable for Detect: {e}") from e
        if proc.returncode != 0:
            raise FlywheelError(
                f"gh GET {url} failed (exit {proc.returncode}): {proc.stderr.strip()[:300]}"
            )
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise FlywheelError(f"gh returned unparseable commits JSON: {e}") from e
        if not isinstance(payload, list):
            raise FlywheelError("gh returned a non-list commits payload: refusing to guess")
        return [
            c for c in parse_commits_payload(payload, repo) if is_fix_commit(c.message)
        ]
