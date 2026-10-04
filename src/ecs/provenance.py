"""What produced a results file, recorded so that drift is detectable.

A commit alone cannot be checked against the code: it stays whatever it was
while the code that wrote the file goes on changing.  The block this module
writes records the digest of every file the result depends on as well, so
``tests/test_provenance.py`` can compare the record against the working tree and
fail when they part.  Regenerating the result is then the only way back to a
passing suite.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

__all__ = ["REPO_ROOT", "digest_of", "head_commit", "provenance_block"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def digest_of(path: Path) -> str:
    """SHA-256 of one file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head_commit() -> str:
    """The commit checked out in this repository, or ``unknown`` outside a git checkout."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, cwd=REPO_ROOT
    )
    return out.stdout.strip() or "unknown"


def provenance_block(producers: list[str]) -> dict[str, Any]:
    """The commit, and the digest of every source file the result rests on.

    ``producers`` are repository-relative paths: the script that wrote the file
    and the modules whose behaviour its numbers depend on. Listing a module that
    cannot change the numbers costs a regeneration for nothing; leaving out one
    that can is the failure this block exists to catch.
    """
    return {
        "commit": head_commit(),
        "producers": {path: digest_of(REPO_ROOT / path) for path in sorted(producers)},
    }
