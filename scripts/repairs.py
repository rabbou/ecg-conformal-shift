"""Three repairs on the same scores, judged on net benefit at named thresholds.

For every pair in ``score_pairs`` whose target is another population than the
source, ``ecs.repairs.repair_cell`` reads the model as delivered, the label-free
prevalence correction, the recalibration on 100 site labels and the per-label
abstention on one half of the target, cut by patient.  Treating everyone and
treating no one are read beside them.

The thresholds are the probabilities of disease at which a clinician would act.
At 10%, sending a patient to echocardiography is judged worth nine normal
studies for one abnormal one found; 5% and 20% bracket it.  That weighing is a
stated assumption, not a measured preference.

Writes ``results/repairs.json``.  It needs the patient tables of PTB-XL,
Shandong and Chongqing on disk, and EchoNext with its stored scores; without
EchoNext the family is left out and the file says so.

Usage: .venv/bin/python scripts/repairs.py
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from score_pairs import Pair, echonext, echonext_available, infarction, rotation, rounded

from ecs.config import RESULTS_DIR
from ecs.provenance import provenance_block
from ecs.repairs import DRAWS, LOCAL_LABELS, THRESHOLDS, repair_cell

NAMED_THRESHOLD = 0.10
DIGITS = 4  # net benefit per patient to four significant digits: 0.1 per 1,000 at worst
CURVE = np.round(np.arange(0.01, 0.51, 0.01), 2)
# A cell enters the summary when its evaluation half holds this many ill
# patients; below it, the net benefit of every rule is within noise of zero.
MIN_ILL_EVAL = 20
RULES = ("as_delivered", "prior", "recalibrated", "abstention_cleared", "abstention_referred")
CURVE_CELLS = {
    ("echonext", "outpatient", "shd_moderate_or_greater_flag"),
    ("echonext", "emergency", "shd_moderate_or_greater_flag"),
    ("infarction", "sph", "MI"),
    ("infarction", "acs", "MI"),
}

PRODUCERS = [
    "scripts/repairs.py",
    "scripts/score_pairs.py",
    "src/ecs/repairs.py",
    "src/ecs/calibration.py",
    "src/ecs/conformal.py",
    "src/ecs/transfer.py",
    "src/ecs/splits.py",
]


def measure(pair: Pair) -> dict[str, Any]:
    assert pair.target_patients is not None
    head = {
        "family": pair.family,
        "model": pair.model,
        "source": pair.source,
        "target": pair.target,
        "label": pair.label,
    }
    if pair.y_source.min() == pair.y_source.max() or pair.y_target.max() == 0:
        return head | {"summarised": False, "skipped": "one class only in the source or target"}
    curve = CURVE if (pair.family, pair.target, pair.label) in CURVE_CELLS else None
    cell = repair_cell(
        pair.p_source,
        pair.y_source,
        pair.p_target,
        pair.y_target,
        pair.target_patients.reset_index(drop=True),
        curve=curve,
    )
    return head | {
        "summarised": cell["n_eval_ill"] >= MIN_ILL_EVAL,
        **cell,
    }


def _best(row: dict[str, Any]) -> str:
    """The rule with the highest net benefit, treat-all and treat-none included."""
    options = {name: row[name] for name in (*RULES, "treat_all", "treat_none")}
    return max(options, key=lambda name: options[name])


def summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Per threshold: how often each rule wins, and its mean gain over the model as delivered.

    Gains are in net true positives per 1,000 patients.
    """
    kept = [c for c in cells if c["summarised"]]
    out: dict[str, Any] = {"cells": len(kept), "cells_not_summarised": len(cells) - len(kept)}
    if not kept:
        return out
    for i, t in enumerate(THRESHOLDS):
        rows = [c["net_benefit"][i] for c in kept]
        wins = [_best(r) for r in rows]
        block: dict[str, Any] = {
            "wins": {name: wins.count(name) for name in (*RULES, "treat_all", "treat_none")},
            "mean_gain_per_1000": {},
            "share_below_treat_none": {},
            "share_below_treat_all": {},
        }
        for name in RULES:
            gain = np.array([r[name] - r["as_delivered"] for r in rows]) * 1000
            block["mean_gain_per_1000"][name] = float(gain.mean())
            block["share_below_treat_none"][name] = float(np.mean([r[name] < 0 for r in rows]))
            block["share_below_treat_all"][name] = float(
                np.mean([r[name] < r["treat_all"] for r in rows])
            )
        out[f"{t:.2f}"] = block
    out["mean_abs_intercept"] = {
        name: float(np.mean([abs(c["calibration"][name]["intercept"] or 0.0) for c in kept]))
        for name in ("as_delivered", "prior", "recalibrated")
    }
    return out


def write(result: dict[str, Any]) -> None:
    """One cell per line, so the file stays under the repository's 512 kB bar and a
    rerun's diff names the cells that moved."""
    head = {k: v for k, v in result.items() if k != "cells"}
    cells = ",\n".join(
        "  " + json.dumps(rounded(c, DIGITS), separators=(",", ":")) for c in result["cells"]
    )
    text = json.dumps(head, indent=1)[:-2] + ',\n "cells": [\n' + cells + "\n ]\n}\n"
    (RESULTS_DIR / "repairs.json").write_text(text)


def main() -> None:
    started = time.time()
    pairs = [*infarction(patients=True), *rotation(patients=True)]
    families = ["infarction", "rotation"]
    if echonext_available():
        pairs += list(echonext(patients=True))
        families.append("echonext")
    cells = [measure(p) for p in pairs if not p.in_distribution]
    result = {
        "thresholds": list(THRESHOLDS),
        "named_threshold": NAMED_THRESHOLD,
        "named_action": (
            "send to echocardiography (EchoNext); the same weighing is applied to the other "
            "diagnoses as a common yardstick"
        ),
        "local_labels": LOCAL_LABELS,
        "draws": DRAWS,
        "families": families,
        "unit": "net benefit: true positives per patient minus false positives per patient "
        "times t / (1 - t)",
        "abstention": (
            "a {0, 1} or empty set goes to a human reader whose decision is not modelled: "
            "'abstention_cleared' counts it as not referred, 'abstention_referred' as referred"
        ),
        "cells": cells,
        "summary": {
            family: summary([c for c in cells if c["family"] == family]) for family in families
        }
        | {"all": summary(cells)},
        "seconds": round(time.time() - started, 1),
        "commit": provenance_block(PRODUCERS)["commit"],
    }
    write(rounded(result))
    print(f"{len(cells)} cells in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
