"""Infarction at the three sites: how well the model separates, and how far one threshold moves.

For PTB-XL fold 10, Shandong and Chongqing, from the committed scores: the
AUROC for infarction with a 95% bootstrap interval, and a 95% interval around
the sensitivity and specificity ``results/outcomes.json`` reports for the
threshold set to 90% at PTB-XL.

That figure is a mean over 200 draws, each setting the threshold on one half of
PTB-XL's fold 10.  At PTB-XL each draw reads the other half (about 1,099
tracings, 276 infarctions), so the spread between draws already holds both the
threshold's movement and the sampling of the tracings read: the interval is the
mean plus or minus 1.96 standard deviations of the draws.  Shandong and
Chongqing are read whole in every draw, so their interval adds the binomial
variance of the site's own count to the variance between draws.

Writes ``results/infarction_sites.json``.  A few seconds.

Usage: uv run python scripts/infarction_sites.py
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from ecs.config import REPO_ROOT, RESULTS_DIR
from ecs.provenance import provenance_block
from ecs.transfer import auroc_row

SCORES = {
    "ptbxl": "results/baseline/scores.npz",
    "sph": "results/external/sph.npz",
    "acs": "results/external/acs.npz",
}
PRODUCERS = [
    "scripts/infarction_sites.py",
    "src/ecs/transfer.py",
    "results/outcomes.json",
    *SCORES.values(),
]


Z = 1.959964


def draw_interval(mean: float, sd_between: float, n_read: float | None) -> list[float]:
    """Mean plus or minus 1.96 standard deviations: the spread between draws, plus the
    binomial variance of ``n_read`` patients when every draw reads the same ones."""
    variance = sd_between**2 + (mean * (1 - mean) / n_read if n_read else 0.0)
    half = Z * math.sqrt(variance)
    return [max(0.0, mean - half), min(1.0, mean + half)]


def site(corpus: str, outcomes: dict[str, Any]) -> dict[str, Any]:
    with np.load(REPO_ROOT / SCORES[corpus]) as data:
        y = data["labels"].astype(int)
        p = data["probs"][:, 1].astype(np.float64)
    by = outcomes["by_corpus"][corpus]
    plain = by["schemes"]["plain"]
    sens, spec = plain["1"]["correct"], plain["0"]["correct"]
    n_mi, n_other = int(y.sum()), int((1 - y).sum())
    # A site read whole in every draw adds its own count's sampling; PTB-XL's draws
    # each read a different half, so their spread already carries it.
    read_whole = by["n_scored"]["sd"] == 0
    return {
        "n": int(len(y)),
        "n_mi": n_mi,
        "n_read_per_draw": by["n_scored"]["mean"],
        "n_mi_read_per_draw": by["n_positive"]["mean"],
        "auroc": auroc_row(y, p),
        "sensitivity": sens["mean"],
        "sensitivity_interval": draw_interval(
            sens["mean"], sens["sd"], n_mi if read_whole else None
        ),
        "specificity": spec["mean"],
        "specificity_interval": draw_interval(
            spec["mean"], spec["sd"], n_other if read_whole else None
        ),
    }


def main() -> None:
    start = time.time()
    outcomes = json.loads((RESULTS_DIR / "outcomes.json").read_text())
    result: dict[str, Any] = {
        "question": (
            "How well the infarction model separates at each site, and how far the sensitivity "
            "and specificity of the threshold set at PTB-XL move between draws."
        ),
        "interval_rule": (
            "mean +/- 1.96 sd; the sd is the spread between the 200 draws, plus the binomial "
            "variance of the site's count where every draw reads the whole site"
        ),
        "threshold": "outcomes.json, scheme plain: 90% sensitivity on a PTB-XL calibration half",
        "sites": {corpus: site(corpus, outcomes) for corpus in SCORES},
        "provenance": provenance_block(PRODUCERS),
    }
    result["seconds"] = round(time.time() - start, 1)
    path = RESULTS_DIR / "infarction_sites.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
