"""Render REPORT.md, SUPPLEMENT.md, README.md and CITATION.cff from their templates.

A template in ``docs/templates/`` writes each computed number as ``{{name}}``,
``{{cap:name}}`` where a sentence needs the figure's first letter in upper
case, and a table's rows as ``{{rows:builder}}``.  Every name resolves through
``paper_values``, which reads the files under ``results/``; a name nothing
defines is an error, never an empty string.

Usage: uv run python tests/paper_render.py --write
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "docs/templates"
REPORT = ROOT / "REPORT.md"
SUPPLEMENT = ROOT / "SUPPLEMENT.md"
README = ROOT / "README.md"
CITATION = ROOT / "CITATION.cff"
TARGETS = (REPORT, SUPPLEMENT, README, CITATION)
PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}")


def render(template: str) -> str:
    from paper_values import ROWS, values

    figures = values()
    missing: list[str] = []

    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        if name.startswith("rows:"):
            return "\n".join(ROWS[name.removeprefix("rows:")]())
        capital = name.startswith("cap:")
        key = name.removeprefix("cap:")
        if key not in figures:
            missing.append(key)
            return match.group(0)
        value = figures[key]
        return value[:1].upper() + value[1:] if capital else value

    text = PLACEHOLDER.sub(fill, template)
    if missing:
        raise KeyError(f"names no results file defines: {sorted(set(missing))}")
    return text


def template_of(target: Path) -> Path:
    return TEMPLATES / target.name


def rendered(target: Path) -> str:
    return render(template_of(target).read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the rendered files")
    args = parser.parse_args(argv)
    stale = []
    for target in TARGETS:
        text = rendered(target)
        if not target.exists() or text != target.read_text():
            stale.append(target.name)
            if args.write:
                target.write_text(text)
    verb = "wrote" if args.write else "would rewrite"
    print(f"{verb}: {', '.join(stale)}" if stale else "every rendered file is current")
    return 0 if args.write or not stale else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(ROOT / "src"))
    sys.exit(main())
