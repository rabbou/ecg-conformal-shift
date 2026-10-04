"""How the report and the README print a number, and how a test finds one in their text."""

from __future__ import annotations

import re

from ecs.config import REPO_ROOT

REPORT = REPO_ROOT / "REPORT.md"
README = REPO_ROOT / "README.md"

# A number as prose prints it: digits with thousands commas, an optional decimal
# part, an optional per cent sign; a leading minus sign is part of the number.
NUMBER = re.compile(r"(?<![\w.])[-−]?\d[\d,]*(?:\.\d+)?%?")


def pct(x: float | None, digits: int = 1) -> str:
    """A share as the text prints it: ``0.716`` is ``71.6%``."""
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"
