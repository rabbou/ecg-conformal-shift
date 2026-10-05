"""The rotation with MIMIC-IV-ECG as a sixth corpus: coverage per label with Wilson intervals.

Every source's thresholds are fitted once, on its whole calibration part (2,000
records), and spent unchanged on its own test part and on the test part of each
of the five other corpora: thirty ordered pairs and six readings at home.  Per
pair and diagnosis, three decision rules (``plain``, ``pooled``, ``perlabel``,
as in ``ecs.transfer``) at the 90% level, and for each the coverage of the
tracings carrying the diagnosis and of those without it, each with its 95%
Wilson interval.

A Wilson interval treats the test tracings as independent draws with the
threshold held fixed.  It is the precision of one deployed threshold on this
test part; the spread over re-drawn calibration halves is the published
rotation's ``coverage_sd`` and is not repeated here.

The models are retrained on this machine (``scripts/train_source.py
--rotation-dir results/rotation_six``), because the published run's checkpoints
were not kept.  So the same estimator is run on the published scores too, and
the difference on the twenty-five pairs the two runs share says how far a
retrained model moves a figure.

Writes ``results/mimic_rotation.csv`` (every pair) and ``results/mimic_rotation.json``
(settings, the MIMIC rows at the headline rule, the reproduction check).

Usage: .venv/bin/python scripts/mimic_rotation.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ecs.config import RESULTS_DIR
from ecs.encoders import SAW
from ecs.provenance import provenance_block
from ecs.rotation import EXTENDED_SOURCES, SOURCES, class_keys, usable_classes
from ecs.transfer import ALPHA, METHODS, auroc_row, conformal_sets, coverage_row

ROTATION_SIX = Path(RESULTS_DIR) / "rotation_six"
PUBLISHED = Path(RESULTS_DIR) / "rotation"
OUT_CSV = Path(RESULTS_DIR) / "mimic_rotation.csv"
OUT_JSON = Path(RESULTS_DIR) / "mimic_rotation.json"
HEADLINE = "perlabel"

# The files whose behaviour the table's numbers depend on.
PRODUCERS = [
    "scripts/mimic_rotation.py",
    "src/ecs/conformal.py",
    "src/ecs/encoders.py",
    "src/ecs/metrics.py",
    "src/ecs/rotation.py",
    "src/ecs/small_set.py",
    "src/ecs/transfer.py",
]

NAMES = {
    "ptbxl": "PTB-XL",
    "sph": "Shandong",
    "chapman_ningbo": "Chapman-Shaoxing and Ningbo",
    "georgia": "Georgia",
    "cpsc": "CPSC 2018 and extension",
    "mimic": "MIMIC-IV-ECG (Beth Israel Deaconess)",
}

COLUMNS = (
    "source",
    "target",
    "role",
    "label",
    "method",
    "n",
    "n_pos",
    "prevalence",
    "auroc",
    "coverage_pos",
    "coverage_pos_low",
    "coverage_pos_high",
    "coverage_neg",
    "coverage_neg_low",
    "coverage_neg_high",
    "abstention",
    "n_cal_pos",
)


def load(path: Path) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    with np.load(path) as data:
        classes = [str(c) for c in data["classes"]]
        if classes != class_keys():
            raise ValueError(f"{path}: classes {classes}, expected {class_keys()}")
        return data["y"].astype(int), data["p"].astype(np.float64)


def pair_rows(rotation: Path, source: str, target: str) -> list[dict[str, Any]]:
    """Every label and rule for one source spent on one target."""
    y_cal, p_cal = load(rotation / source / "scores" / f"{source}_cal.npz")
    y_tgt, p_tgt = load(rotation / source / "scores" / f"{target}_test.npz")
    shared = [k for k in usable_classes(source) if k in usable_classes(target)]
    rows = []
    for key in shared:
        j = class_keys().index(key)
        sets = conformal_sets(p_cal[:, j], y_cal[:, j], p_tgt[:, j], ALPHA)
        auc = auroc_row(y_tgt[:, j], p_tgt[:, j])
        for method in METHODS:
            row = coverage_row(sets[method], y_tgt[:, j])
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "role": "home" if source == target else "away",
                    "label": key,
                    "method": method,
                    "n": row["n"],
                    "n_pos": row["n_pos"],
                    "prevalence": round(row["n_pos"] / row["n"], 5) if row["n"] else None,
                    "auroc": auc["auroc"],
                    "coverage_pos": row["coverage_pos"],
                    "coverage_pos_low": row["coverage_pos_low"],
                    "coverage_pos_high": row["coverage_pos_high"],
                    "coverage_neg": row["coverage_neg"],
                    "coverage_neg_low": row["coverage_neg_low"],
                    "coverage_neg_high": row["coverage_neg_high"],
                    "abstention": row["abstention"],
                    "n_cal_pos": int(y_cal[:, j].sum()),
                }
            )
    return rows


def rounded(row: dict[str, Any]) -> dict[str, Any]:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}


def reproduction(six: list[dict[str, Any]]) -> dict[str, Any]:
    """The same estimator on the published scores, pair by pair, at the headline rule."""
    published: list[dict[str, Any]] = []
    for source in SOURCES:
        for target in SOURCES:
            published.extend(pair_rows(PUBLISHED, source, target))

    def key(r: dict[str, Any]) -> tuple[str, str, str, str]:
        return (r["source"], r["target"], r["label"], r["method"])

    retrained = {key(r): r for r in six}
    differences = []
    for row in published:
        if row["method"] != HEADLINE or key(row) not in retrained:
            continue
        other = retrained[key(row)]
        differences.append(
            {
                "source": row["source"],
                "target": row["target"],
                "label": row["label"],
                "published": round(row["coverage_pos"], 4),
                "retrained": round(other["coverage_pos"], 4),
                "difference": round(other["coverage_pos"] - row["coverage_pos"], 4),
                "auroc_published": row["auroc"],
                "auroc_retrained": other["auroc"],
            }
        )
    gaps = np.abs([d["difference"] for d in differences])
    return {
        "what": (
            "coverage of the diagnosis under the per-label rule, the published "
            "five-corpus models against the models retrained here, same records, same "
            "estimator"
        ),
        "n_cells": len(differences),
        "median_absolute_difference": round(float(np.median(gaps)), 4),
        "max_absolute_difference": round(float(gaps.max()), 4),
        "cells": differences,
    }


def main() -> int:
    rows: list[dict[str, Any]] = []
    for source in EXTENDED_SOURCES:
        for target in EXTENDED_SOURCES:
            rows.extend(pair_rows(ROTATION_SIX, source, target))
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rounded(r) for r in rows)

    mimic = [
        rounded(r)
        for r in rows
        if r["method"] == HEADLINE and "mimic" in (r["source"], r["target"])
    ]
    configs = {
        s: json.loads((ROTATION_SIX / s / "config.json").read_text()) for s in EXTENDED_SOURCES
    }
    check = reproduction(rows)
    table: dict[str, Any] = {
        "written_by": "scripts/mimic_rotation.py",
        "provenance": provenance_block(PRODUCERS),
        "settings": {
            "alpha": ALPHA,
            "methods": list(METHODS),
            "headline_method": HEADLINE,
            "interval": "Wilson score interval, 95%, over the target's test tracings",
            "calibration": "the source's whole calibration part, thresholds fitted once",
            "corpus_names": NAMES,
            "classes": class_keys(),
        },
        "label_sources": {
            "mimic": "machine statements (results/mimic_label_map.json)",
            "others": "SNOMED or AHA codes of each release (results/label_map.json)",
        },
        "encoders_that_saw_mimic": sorted(a for a, seen in SAW.items() if "mimic" in seen),
        "sources": {
            s: {
                "split_sizes": c["split_sizes"],
                "standardisation": c["standardisation"],
                "deviations": c["deviations"],
                "machine": c["machine"],
            }
            for s, c in configs.items()
        },
        "grid": {"file": "results/mimic_rotation.csv", "columns": list(COLUMNS)},
        "mimic_pairs": mimic,
        "reproduction": check,
    }
    OUT_JSON.write_text(json.dumps(table, indent=1) + "\n")
    print(f"wrote {OUT_CSV} ({len(rows)} rows) and {OUT_JSON}")
    for r in mimic:
        print(
            f"{r['source']:>15} -> {r['target']:<15} {r['label']:5s} "
            f"cov+ {r['coverage_pos']:.3f} [{r['coverage_pos_low']:.3f}, "
            f"{r['coverage_pos_high']:.3f}]  n+ {r['n_pos']}"
        )
    print(
        f"reproduction: {check['n_cells']} cells, median |diff| "
        f"{check['median_absolute_difference']}, max {check['max_absolute_difference']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
