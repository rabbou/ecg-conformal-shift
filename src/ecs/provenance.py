"""What produced a results file, recorded so that drift is detectable.

A results file used to carry one field, the commit that was checked out when it
was written.  That field cannot be checked: it stays whatever it was while the
code that wrote the file goes on changing underneath it, and four of the eight
files carried a commit older than the last change to their own producer without
anything saying so.

The block this module writes records the digest of every file the result
depends on, so ``tests/test_provenance.py`` can compare the record against the
working tree and fail when they part.  Regenerating the result is then the only
way to make the test pass again, which is the property that was missing.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

__all__ = ["REPO_ROOT", "digest_of", "provenance_block"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def digest_of(path: Path) -> str:
    """SHA-256 of one file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance_block(producers: list[str]) -> dict[str, Any]:
    """The commit, and the digest of every source file the result rests on.

    ``producers`` are repository-relative paths: the script that wrote the file
    and the modules whose behaviour its numbers depend on. Listing a module that
    cannot change the numbers costs a regeneration for nothing; leaving out one
    that can is the failure this block exists to catch.
    """
    return {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
        ).stdout.strip(),
        "producers": {path: digest_of(REPO_ROOT / path) for path in sorted(producers)},
    }
