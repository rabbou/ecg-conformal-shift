"""Does the positive predictive value recomputed by Bayes' rule predict the one observed?

For every pair in ``score_pairs``: one threshold is fitted on the source at 90%
sensitivity (the study's plain threshold), the source's sensitivity and
specificity at it are carried to the target's prevalence by Bayes' rule, and the
result is set against the share of flagged target patients who are ill.

Writes ``results/ppv_gap.json`` from the infarction and rotation pairs, which
this repository carries, and, where EchoNext and its stored scores are on disk,
``results/echonext_ppv_gap.json`` with the EchoNext pairs and the summary over
all three families.  ``results/ppv_gap.csv`` holds every row of the run.

Usage: .venv/bin/python scripts/ppv_gap.py
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from score_pairs import Pair, echonext, echonext_available, infarction, rotation, rounded

from ecs.config import RESULTS_DIR
from ecs.ppv_gap import gap_row, label_free, label_shift_check, table_of
from ecs.provenance import provenance_block
from ecs.transfer import ALPHA, conformal_sets

# A cell is summarised when its observed PPV rests on at least this many flagged
# patients and its target holds at least this many ill ones; below that the
# Wilson interval is wider than the gaps the summary is about.
MIN_FLAGGED = 20
MIN_ILL = 10
CLOSE_POINTS = 0.02  # "one or two points", the margin T-067's card names

PRODUCERS = [
    "scripts/ppv_gap.py",
    "scripts/score_pairs.py",
    "src/ecs/ppv_gap.py",
    "src/ecs/repairs.py",
    "src/ecs/calibration.py",
    "src/ecs/conformal.py",
    "src/ecs/transfer.py",
    "src/ecs/metrics.py",
]


def measure(pair: Pair) -> dict[str, Any]:
    flags_source = conformal_sets(pair.p_source, pair.y_source, pair.p_source, ALPHA)["plain"]
    flags_target = conformal_sets(pair.p_source, pair.y_source, pair.p_target, ALPHA)["plain"]
    source = table_of(pair.y_source, flags_source[:, 1])
    target = table_of(pair.y_target, flags_target[:, 1])
    row: dict[str, Any] = {
        "family": pair.family,
        "model": pair.model,
        "source": pair.source,
        "target": pair.target,
        "label": pair.label,
        "in_distribution": pair.in_distribution,
        **gap_row(source, target),
    }
    if row["defined"]:
        row |= label_free(
            pair.p_source,
            pair.y_source,
            pair.p_target,
            flags_target[:, 1],
            row["sens_source"],
            row["spec_source"],
        )
        row |= label_shift_check(pair.p_source, pair.y_source, pair.p_target, pair.y_target)
    row["summarised"] = bool(
        row["defined"] and row["n_flagged"] >= MIN_FLAGGED and row["n_target_ill"] >= MIN_ILL
    )
    return row


def _error(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    errors = np.array(
        [abs(r[key] - r["ppv_observed"]) for r in rows if r.get(key) is not None], dtype=float
    )
    if errors.size == 0:
        return {"n": 0}
    return {
        "n": int(errors.size),
        "median_abs_points": float(np.median(errors)),
        "share_within_two_points": float((errors <= CLOSE_POINTS).mean()),
    }


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(values)) if values else None


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How far the recipe lands from the observation, over the cells that can say."""
    kept = [r for r in rows if r["summarised"]]
    if not kept:
        return {"cells": 0, "cells_not_summarised": len(rows)}
    gap = np.array([r["gap"] for r in kept])
    ratio = np.array([r["ppv_recomputed"] / r["ppv_observed"] for r in kept])
    over = np.array([r["gap"] > 0 for r in kept])
    spec_moved = np.array([r["spec_target"] - r["spec_source"] for r in kept])
    sens_moved = np.array([r["sens_target"] - r["sens_source"] for r in kept])
    return {
        "cells": len(kept),
        "cells_not_summarised": len(rows) - len(kept),
        "median_abs_gap_points": float(np.median(np.abs(gap))),
        "abs_gap_quartiles": [float(q) for q in np.percentile(np.abs(gap), [25, 75])],
        "max_abs_gap_points": float(np.max(np.abs(gap))),
        "share_within_two_points": float((np.abs(gap) <= CLOSE_POINTS).mean()),
        "share_outside_observed_interval": float(
            np.mean([not r["recomputed_inside_interval"] for r in kept])
        ),
        "share_recipe_too_high": float(over.mean()),
        "median_ratio_recomputed_to_observed": float(np.median(ratio)),
        "share_ratio_off_by_a_quarter_or_more": float(np.mean((ratio >= 1.25) | (ratio <= 0.8))),
        "median_abs_specificity_change": float(np.median(np.abs(spec_moved))),
        "median_abs_sensitivity_change": float(np.median(np.abs(sens_moved))),
        "median_likelihood_ratio_healthy_source": _median(kept, "likelihood_ratio_healthy_source"),
        "median_likelihood_ratio_healthy_target": _median(kept, "likelihood_ratio_healthy_target"),
        "median_likelihood_ratio_ill_source": _median(kept, "likelihood_ratio_ill_source"),
        "median_likelihood_ratio_ill_target": _median(kept, "likelihood_ratio_ill_target"),
        "label_free_predictors": {
            "recipe_true_prevalence": _error(kept, "ppv_recomputed"),
            "recipe_estimated_prevalence": _error(kept, "ppv_recipe_estimated_prevalence"),
            "mean_probability": _error(kept, "ppv_mean_probability"),
            "mean_probability_prior_corrected": _error(
                kept, "ppv_mean_probability_prior_corrected"
            ),
        },
    }


def summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per family, in-distribution controls apart from transfers, and per target."""
    out: dict[str, Any] = {}
    for family in sorted({r["family"] for r in rows}):
        mine = [r for r in rows if r["family"] == family]
        out[family] = {
            "transfer": summary([r for r in mine if not r["in_distribution"]]),
            "in_distribution": summary([r for r in mine if r["in_distribution"]]),
            "by_target": {
                t: summary([r for r in mine if r["target"] == t])
                for t in sorted({r["target"] for r in mine})
            },
        }
    return out


def header() -> dict[str, Any]:
    return {
        "question": (
            "Does the PPV recomputed from the source's sensitivity and specificity at the "
            "target's prevalence predict the PPV observed at the target?"
        ),
        "threshold": (
            f"one per pair, fitted on the source so that {1 - ALPHA:.0%} of its ill patients "
            "are flagged (the conformal quantile of the positive class)"
        ),
        "prevalence_given_to_the_recipe": "the target's true prevalence",
        "gap": "recomputed minus observed, in proportions (0.01 is one percentage point)",
        "gap_interval": "2.5th and 97.5th percentiles of 2,000 multinomial redraws of both tables",
        "observed_interval": "Wilson, 95%",
        "summarised_when": f"at least {MIN_FLAGGED} flagged and {MIN_ILL} ill target patients",
    }


def main() -> None:
    started = time.time()
    public = [measure(p) for p in (*infarction(), *rotation())]
    result = header() | {
        "rows": public,
        "summary": summaries(public),
        "seconds": round(time.time() - started, 1),
        "provenance": provenance_block(PRODUCERS),
    }
    (RESULTS_DIR / "ppv_gap.json").write_text(json.dumps(rounded(result), indent=1) + "\n")
    rows = public
    if echonext_available():
        private = [measure(p) for p in echonext()]
        rows = public + private
        (RESULTS_DIR / "echonext_ppv_gap.json").write_text(
            json.dumps(
                rounded(
                    header()
                    | {
                        "rows": private,
                        "summary": summaries(private),
                        "all_families": {
                            "transfer": summary([r for r in rows if not r["in_distribution"]]),
                            "in_distribution": summary([r for r in rows if r["in_distribution"]]),
                        },
                        "commit": provenance_block(PRODUCERS)["commit"],
                    }
                ),
                indent=1,
            )
            + "\n"
        )
    else:
        print("EchoNext or its stored scores are absent; echonext_ppv_gap.json not written")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "ppv_gap.csv", index=False)
    print(f"{len(rows)} rows in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
