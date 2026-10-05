"""Infarction at the three sites: how well the model separates, and what each site's count supports.

For PTB-XL fold 10, Shandong and Chongqing, from the committed scores: the
AUROC for infarction with a 95% bootstrap interval, and the Wilson interval
each site's own infarction count supports around the sensitivity
``results/outcomes.json`` reports for the threshold set to 90% at PTB-XL.

Writes ``results/infarction_sites.json``.  A few seconds.

Usage: uv run python scripts/infarction_sites.py
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from ecs.config import REPO_ROOT, RESULTS_DIR
from ecs.metrics import wilson_interval
from ecs.provenance import provenance_block
from ecs.transfer import auroc_row

SCORES = {
    "ptbxl": "results/baseline/scores.npz",
    "sph": "results/external/sph.npz",
    "acs": "results/external/acs.npz",
}
PRODUCERS = [
    "scripts/infarction_sites.py",
    "src/ecs/metrics.py",
    "src/ecs/transfer.py",
    "results/outcomes.json",
    *SCORES.values(),
]


def site(corpus: str, outcomes: dict[str, Any]) -> dict[str, Any]:
    with np.load(REPO_ROOT / SCORES[corpus]) as data:
        y = data["labels"].astype(int)
        p = data["probs"][:, 1].astype(np.float64)
    plain = outcomes["by_corpus"][corpus]["schemes"]["plain"]
    sens = plain["1"]["correct"]["mean"]
    spec = plain["0"]["correct"]["mean"]
    n_mi, n_other = int(y.sum()), int((1 - y).sum())
    return {
        "n": int(len(y)),
        "n_mi": n_mi,
        "auroc": auroc_row(y, p),
        "sensitivity": sens,
        "sensitivity_wilson": list(wilson_interval(round(sens * n_mi), n_mi)),
        "specificity": spec,
        "specificity_wilson": list(wilson_interval(round(spec * n_other), n_other)),
    }


def main() -> None:
    start = time.time()
    outcomes = json.loads((RESULTS_DIR / "outcomes.json").read_text())
    result: dict[str, Any] = {
        "question": (
            "How well the infarction model separates at each site, and the interval each "
            "site's own count supports around the sensitivity of the threshold set at PTB-XL."
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
