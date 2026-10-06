"""95% intervals for the shares REPORT.md quotes that no other results file carries.

Two kinds.  Across transfers, each share (the recomputed PPV outside the observed
PPV's interval, within two points of it, too high, off by a quarter or more) and
the median gap are summaries over cells that are not independent: the cells of one
model on one pair of populations share a threshold rule, a model and a target.  The
interval therefore resamples whole clusters, one cluster per model and pair of
populations, 2,000 times, and takes the 2.5th and 97.5th percentiles.  With few
clusters (four for each Columbia block) the interval is rough, and it says so by
carrying the cluster count beside it.

Among Columbia patients, the shares the article states as proportions of patients
(prevalence, share flagged, negative predictive value) get Wilson intervals from
the counts in ``results/echonext_clinical.json``.

Writes ``results/ppv_intervals.json``.  A few seconds.

Usage: uv run python scripts/ppv_intervals.py
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ecs.config import REPO_ROOT, RESULTS_DIR
from ecs.metrics import wilson_interval
from ecs.provenance import provenance_block

DRAWS = 2000
CLOSE_POINTS = 0.02
INPUTS = ("results/ppv_gap.json", "results/echonext_ppv_gap.json", "results/echonext_clinical.json")
PRODUCERS = ["scripts/ppv_intervals.py", "src/ecs/metrics.py", *INPUTS]
Row = dict[str, Any]

STATISTICS: dict[str, Callable[[list[Row]], float]] = {
    "median_abs_gap_points": lambda rows: float(np.median([abs(r["gap"]) for r in rows])),
    "share_outside_observed_interval": lambda rows: float(
        np.mean([not r["recomputed_inside_interval"] for r in rows])
    ),
    "share_within_two_points": lambda rows: float(
        np.mean([abs(r["gap"]) <= CLOSE_POINTS for r in rows])
    ),
    "share_recipe_too_high": lambda rows: float(np.mean([r["gap"] > 0 for r in rows])),
    "share_ratio_off_by_a_quarter_or_more": lambda rows: float(
        np.mean(
            [
                (r["ppv_recomputed"] / r["ppv_observed"] >= 1.25)
                or (r["ppv_recomputed"] / r["ppv_observed"] <= 0.8)
                for r in rows
            ]
        )
    ),
}


def cluster_of(row: Row) -> tuple[str, str, str, str]:
    return (row["family"], row["model"], row["source"], row["target"])


def clustered(rows: list[Row], seed: int = 0) -> dict[str, Any]:
    """Each statistic on the rows, with the percentile interval of a cluster bootstrap."""
    clusters: dict[tuple[str, str, str, str], list[Row]] = {}
    for r in rows:
        clusters.setdefault(cluster_of(r), []).append(r)
    keys = sorted(clusters)
    rng = np.random.default_rng(seed)
    drawn: dict[str, list[float]] = {name: [] for name in STATISTICS}
    for _ in range(DRAWS):
        picked = rng.integers(0, len(keys), len(keys))
        sample = [r for i in picked for r in clusters[keys[i]]]
        for name, statistic in STATISTICS.items():
            drawn[name].append(statistic(sample))
    out: dict[str, Any] = {"cells": len(rows), "clusters": len(keys)}
    for name, statistic in STATISTICS.items():
        low, high = np.percentile(drawn[name], [2.5, 97.5])
        out[name] = {"estimate": statistic(rows), "low": float(low), "high": float(high)}
    return out


def groups(rows: list[Row]) -> dict[str, list[Row]]:
    kept = [r for r in rows if r["summarised"]]
    transfer = [r for r in kept if not r["in_distribution"]]
    return {
        "transfer": transfer,
        "control": [r for r in kept if r["in_distribution"]],
        "columbia_emergency": [
            r for r in transfer if r["family"] == "echonext" and r["target"] == "emergency"
        ],
        "columbia_outpatient": [
            r for r in transfer if r["family"] == "echonext" and r["target"] == "outpatient"
        ],
        "rotation": [r for r in transfer if r["family"] == "rotation"],
        "infarction": [r for r in transfer if r["family"] == "infarction"],
    }


def wilson(successes: int, n: int) -> dict[str, float]:
    low, high = wilson_interval(successes, n)
    return {"share": successes / n, "successes": successes, "n": n, "low": low, "high": high}


def columbia(clinical: dict[str, Any]) -> dict[str, Any]:
    """Prevalence, share flagged and NPV among the trained network's test patients."""
    out: dict[str, Any] = {}
    for context in ("inpatient", "outpatient"):
        m = clinical["arms"]["resnet"][context]
        n, ill = m["n"], m["n_ill"]
        caught = round(m["sensitivity"] * ill)
        healthy_flagged = round((1 - m["specificity"]) * (n - ill))
        flagged = caught + healthy_flagged
        assert abs(flagged / n - m["label_free"]["share_flagged"]) < 1e-9, context
        cleared = n - flagged
        out[context] = {
            "prevalence": wilson(ill, n),
            "share_flagged": wilson(flagged, n),
            "npv": wilson(cleared - (ill - caught), cleared),
        }
    return out


def main() -> None:
    start = time.time()
    rows = (
        json.loads((RESULTS_DIR / "ppv_gap.json").read_text())["rows"]
        + json.loads((RESULTS_DIR / "echonext_ppv_gap.json").read_text())["rows"]
    )
    clinical = json.loads((RESULTS_DIR / "echonext_clinical.json").read_text())
    result: dict[str, Any] = {
        "question": (
            "95% intervals for the shares across transfers and the Columbia proportions the "
            "report quotes without one elsewhere."
        ),
        "interval_rule": (
            "across transfers: 2.5th and 97.5th percentiles of 2,000 draws resampling whole "
            "clusters, one per model and pair of populations; among patients: Wilson, 95%"
        ),
        "close_points": CLOSE_POINTS,
        "across_transfers": {name: clustered(g) for name, g in groups(rows).items()},
        "columbia": columbia(clinical),
        "provenance": provenance_block(PRODUCERS),
    }
    result["seconds"] = round(time.time() - start, 1)
    path = RESULTS_DIR / "ppv_intervals.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
