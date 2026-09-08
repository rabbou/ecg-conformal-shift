"""What the splits were leaking, and what they leak once the repeats are joined.

A patient-level split keeps one patient's tracings on one side of a boundary.
It cannot keep one *tracing* on one side when the corpus files it twice under
different record identifiers and says nothing about the patient: the Challenge
bundle ships no patient key at all, so each record was its own patient and a
repeated tracing went wherever the shuffle sent it. Where that boundary is
train against test, the model is scored on a recording it was fitted on.

This counts, per corpus, the groups of identical tracings whose members sit in
two different parts -- first under the patient key alone, then under the key
widened to hold a repeated tracing together. The second column is the one that
has to be zero.

``unused`` is not a side: those records feed no figure, so a group split only
between a used part and the unused remainder leaks nothing. Both counts are
reported so the difference is visible.

Usage: .venv/bin/python scripts/split_leak.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ecs.config import RESULTS_DIR
from ecs.duplicates import straddling_groups
from ecs.rotation import PARTS, SOURCES, corpus_index, duplicate_groups_of


def commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def leak(corpus: str, widen: bool) -> dict[str, Any]:
    index = corpus_index(corpus, widen_duplicates=widen)
    parts = {str(record): str(part) for record, part in index.frame["part"].items()}
    groups = duplicate_groups_of(corpus)
    across_any = straddling_groups(parts, groups)
    across_used = straddling_groups(parts, groups, ignore=["unused"])
    records = sum(len(g) for g in across_used)
    return {
        "n_groups": len(groups),
        "n_groups_across_any_part": len(across_any),
        "n_groups_across_two_used_parts": len(across_used),
        "n_records_in_those_groups": records,
        "part_sizes": {
            part: int((index.frame["part"] == part).sum()) for part in (*PARTS, "unused")
        },
        "worst_boundary": sorted(
            {
                " / ".join(sorted({parts[m] for m in members} - {"unused"}))
                for members in across_used
            }
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RESULTS_DIR / "split_leak.json"))
    args = parser.parse_args(argv)

    out: dict[str, Any] = {
        "written_by": "scripts/split_leak.py",
        "commit": commit(),
        "reading": (
            "before: every record is its own patient where the corpus ships no patient key, "
            "so a tracing filed twice is two patients. after: a group of identical tracings "
            "is one splitting unit."
        ),
        "corpora": {},
    }
    for corpus in SOURCES:
        before = leak(corpus, widen=False)
        after = leak(corpus, widen=True)
        out["corpora"][corpus] = {"before": before, "after": after}
        print(
            f"{corpus:<16} groups {before['n_groups']:>4} | "
            f"across two used parts before {before['n_groups_across_two_used_parts']:>4} "
            f"after {after['n_groups_across_two_used_parts']:>4} | "
            f"boundaries before {before['worst_boundary']}",
            flush=True,
        )

    total_after = sum(
        block["after"]["n_groups_across_two_used_parts"] for block in out["corpora"].values()
    )
    out["n_groups_still_across_two_used_parts"] = total_after
    path = Path(args.out)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"written {path}; still straddling after the fix: {total_after}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
