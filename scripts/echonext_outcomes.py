"""EchoNext: what each patient gets, under the sensitivity threshold and the per-label rule.

Coverage of the ill counts a patient sent to a human with both labels as covered,
so it is not the share of the ill the model recognised.  This splits every ill
and every healthy patient of each care setting into what they were given: the
disease label alone, both labels (referred to a human reader), or the other label
alone.  Under the sensitivity threshold nobody is referred, and the share of the
ill recognised is the sensitivity.

The split is read off the committed coverage grid, two rows per cell, by
``ecs.transfer.outcomes_from_coverage``; it needs no tracing and no score, so it
regenerates in a second from ``results/echonext_transfer.json``.

Usage: .venv/bin/python scripts/echonext_outcomes.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ecs.config import RESULTS_DIR
from ecs.provenance import provenance_block
from ecs.transfer import outcomes_from_coverage

LABEL = "shd_moderate_or_greater_flag"
CONTEXTS = ("inpatient", "emergency", "outpatient")


def cell(rows: list[dict[str, Any]], context: str, method: str) -> dict[str, Any]:
    return next(
        r for r in rows if (r["label"], r["context"], r["method"]) == (LABEL, context, method)
    )


def arm_outcomes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per care setting: the per-label split and the sensitivity threshold's operating point."""
    out: dict[str, Any] = {}
    for context in CONTEXTS:
        plain, perlabel = cell(rows, context, "plain"), cell(rows, context, "perlabel")
        split = outcomes_from_coverage(plain, perlabel)
        recognised = split["ill"]["recognised"]["count"] + split["ill"]["referred"]["count"]
        flagged_healthy = split["healthy"]["n"] - split["healthy"]["cleared"]["count"]
        missed = split["ill"]["missed"]["count"]
        out[context] = {
            "n": plain["n"],
            "n_ill": plain["n_pos"],
            "prevalence": plain["n_pos"] / plain["n"],
            "sensitivity_threshold": {
                "sensitivity": plain["coverage_pos"],
                "specificity": plain["coverage_neg"],
                "ppv": recognised / (recognised + flagged_healthy),
                "npv": split["healthy"]["cleared"]["count"]
                / (split["healthy"]["cleared"]["count"] + missed),
            },
            "perlabel": {**split, "referred_all": perlabel["abstention"]},
        }
    return out


def build(transfer: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": (
            "Of the ill and the healthy in each care setting, how many are given the "
            "disease alone, both labels, or the other label alone."
        ),
        "label": LABEL,
        "alpha": transfer["alpha"],
        "definitions": {
            "recognised": "the ill given the disease label alone",
            "referred": "given both labels: no machine answer, a human reads the ECG",
            "missed": "the ill given the healthy label alone",
            "cleared": "the healthy given the healthy label alone",
            "false_alarm": "the healthy given the disease label alone",
        },
        "provenance": provenance_block(
            [
                "results/echonext_transfer.json",
                "scripts/echonext_outcomes.py",
                "src/ecs/transfer.py",
            ]
        ),
        "arms": {arm: arm_outcomes(m["coverage"]) for arm, m in transfer["arms"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, default=RESULTS_DIR / "echonext_transfer.json")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "echonext_outcomes.json")
    args = parser.parse_args()
    table = build(json.loads(args.transfer.read_text()))
    args.out.write_text(json.dumps(table, indent=2) + "\n")
    print(f"wrote {args.out}")
    for arm, contexts in table["arms"].items():
        ill = contexts["outpatient"]["perlabel"]["ill"]
        shares = {k: round(100 * ill[k]["share"], 1) for k in ("recognised", "referred", "missed")}
        print(f"  {arm:<14s} outpatients with the disease: {shares}")


if __name__ == "__main__":
    main()
