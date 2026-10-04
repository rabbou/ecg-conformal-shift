"""Render REPORT.md and README.md from their templates and the files under ``results/``.

A template writes each computed number as ``{{name}}``, a table's rows as
``{{rows:builder}}``, and ``{{cap:name}}`` or ``{{low:name}}`` where a sentence
needs the figure's first letter in the other case.  Every name resolves through
``report_values``; a name nothing defines is an error, never an empty string.

Usage: uv run python tests/report_render.py --write
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from report_text import README, REPORT

from ecs.config import REPO_ROOT

TEMPLATES = REPO_ROOT / "docs/templates"
PAIRS = {REPORT: TEMPLATES / "REPORT.md", README: TEMPLATES / "README.md"}
PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}")


def render(template: str) -> str:
    from report_values import ROWS, values

    figures = values()
    missing: list[str] = []

    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        if name.startswith("rows:"):
            return "\n".join(ROWS[name.removeprefix("rows:")]())
        case, _, key = name.rpartition(":") if name[:4] in ("cap:", "low:") else ("", "", name)
        if key not in figures:
            missing.append(key)
            return match.group(0)
        value = figures[key]
        if case == "cap":
            return value[:1].upper() + value[1:]
        if case == "low":
            return value[:1].lower() + value[1:]
        return value

    text = PLACEHOLDER.sub(fill, template)
    if missing:
        raise KeyError(f"names no results file defines: {sorted(set(missing))}")
    return text


def rendered(target: Path) -> str:
    return render(PAIRS[target].read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write REPORT.md and README.md")
    args = parser.parse_args(argv)
    stale = []
    for target in PAIRS:
        text = rendered(target)
        if text != target.read_text():
            stale.append(target.name)
            if args.write:
                target.write_text(text)
    verb = "wrote" if args.write else "would rewrite"
    print(f"{verb}: {', '.join(stale)}" if stale else "REPORT.md and README.md are current")
    return 0 if args.write or not stale else 1


if __name__ == "__main__":
    sys.exit(main())
