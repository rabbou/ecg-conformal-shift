"""The source rotation: five hospitals, each calibrating once and spending everywhere.

For every source and every diagnosis, the thresholds are fitted on that source's
calibration part and spent, unchanged, on its own test part and on the test part
of the four other corpora.  Twenty ordered pairs and five readings at home, under
three corrections, two non-conformity scores and three confidence levels, each
figure a mean over two hundred re-draws of the calibration half with its spread.

The question the table answers is Leinonen's, asked of coverage instead of
discrimination: how far below the level it promises does a conformal predictor
land when the hospital changes, and does that gap have the same shape whichever
hospital it started from.  The answer is the bias per label -- measured coverage
of the positive class minus the level asked for -- with its spread across the
five sources.

Nothing here can be tuned on a target: a threshold is a function of the
calibration scores and labels, and, for the weighted correction alone, of the
target's unlabelled predicted-label marginal.  ``test_rotation.py`` holds that
by permuting every target's labels on all twenty pairs and checking no threshold
moves.

Usage: .venv/bin/python scripts/rotation_table.py [--draws 200]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ecs.config import RESULTS_DIR
from ecs.encoders import machine_info
from ecs.report import (
    CORRECTIONS,
    SCORES,
    WEIGHTING_NOTE,
    Source,
    Target,
    frozen_calibration_table,
)
from ecs.rotation import SOURCES, CorpusIndex, class_keys, corpus_index, usable_classes

ALPHAS = (0.20, 0.10, 0.05)
HEADLINE_ALPHA = 0.10
HEADLINE_SCORE = "lac"
POSITIVE = "1"  # the class index, as the table's keys spell it: the diagnosis present

CORPUS_NAMES = {
    "ptbxl": "PTB-XL",
    "sph": "Shandong",
    "chapman_ningbo": "Chapman-Shaoxing and Ningbo",
    "georgia": "Georgia",
    "cpsc": "CPSC 2018 and extension",
}


def _patients_for(index: CorpusIndex, ids: list[str]) -> list[str]:
    """The patient of each scored record, in the order the scores were written."""
    return [str(p) for p in index.frame.loc[ids, "patient"]]


def commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()


class Scores:
    """The probabilities one model gave one part of one corpus."""

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.ids = [str(i) for i in data["ids"]]
            self.y = np.asarray(data["y"], dtype=int)
            self.p = np.asarray(data["p"], dtype=np.float64)
            self.classes = [str(c) for c in data["classes"]]

    def binary(self, label: str) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        """(n, 2) probabilities and (n,) labels for one diagnosis against the rest."""
        column = self.classes.index(label)
        p = self.p[:, column]
        return np.column_stack([1.0 - p, p]), self.y[:, column]


def cell(
    source: str,
    label: str,
    scores: dict[tuple[str, str, str], Scores],
    patients: dict[str, list[str]],
    draws: int,
) -> list[dict[str, Any]]:
    """One (source, diagnosis) block of the rotation: the frozen-calibration table."""
    calibration = scores[(source, source, "cal")]
    probs, labels = calibration.binary(label)
    home = Source(
        name=f"{source}-calibration-holdout",
        probs=probs,
        labels=labels,
        patients=patients[f"{source}-cal"],
    )
    targets = {}
    for corpus in SOURCES:
        if label not in usable_classes(corpus):
            continue
        test = scores[(source, corpus, "test")]
        target_probs, target_labels = test.binary(label)
        targets[corpus] = Target(
            probs=target_probs,
            labels=target_labels,
            n_patients=len(set(patients[f"{corpus}-test"])),
            deviations=(),
        )
    return frozen_calibration_table(home, targets, ALPHAS, n_draws=draws, seed=0, keep_draws=False)


def bias_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage of the positive class minus the level asked for, per label and pair.

    ``away`` gathers the twenty ordered source-to-target pairs, ``home`` the five
    readings a source takes on its own test part under the same threshold.  The
    spread across sources is the quantity the rotation exists to produce: one
    pair is an anecdote, five sources is an estimate.
    """
    out: dict[str, Any] = {}
    for correction in CORRECTIONS:
        for entry in cells:
            label = entry["label"]
            source = entry["source"]
            row = next(
                r
                for r in entry["rows"]
                if r["alpha"] == HEADLINE_ALPHA
                and r["score"] == HEADLINE_SCORE
                and r["correction"] == correction
            )
            block = out.setdefault(label, {}).setdefault(
                correction, {"away": [], "home": [], "by_source": {}}
            )
            away = []
            for corpus, cell_block in row["by_corpus"].items():
                if corpus.endswith("-calibration-holdout"):
                    continue
                covered = cell_block["coverage_by_class"][POSITIVE]["mean"]
                bias = covered - row["target_coverage"]
                # How many records the class-conditional figure rests on. A cell
                # with a handful of positives is noise however many draws it is
                # averaged over, and the count is what says so.
                positives = int(round(cell_block["prevalence"] * cell_block["n_points"]))
                if corpus == source:
                    block["home"].append(
                        {"source": source, "bias": round(bias, 4), "n_positive": positives}
                    )
                else:
                    away.append(
                        {
                            "source": source,
                            "target": corpus,
                            "bias": round(bias, 4),
                            "n_positive": positives,
                        }
                    )
            block["away"].extend(away)
            if away:
                block["by_source"][source] = round(float(np.mean([a["bias"] for a in away])), 4)
    for corrections in out.values():
        for block in corrections.values():
            values = np.asarray([a["bias"] for a in block["away"]], dtype=float)
            by_source = np.asarray(list(block["by_source"].values()), dtype=float)
            homes = np.asarray([h["bias"] for h in block["home"]], dtype=float)
            block["away_bias"] = {
                "mean": round(float(values.mean()), 4) if values.size else None,
                "sd_across_pairs": round(float(values.std(ddof=1)), 4) if values.size > 1 else None,
                "sd_across_sources": round(float(by_source.std(ddof=1)), 4)
                if by_source.size > 1
                else None,
                "n_pairs": int(values.size),
                "n_sources": int(by_source.size),
                "worst": round(float(values.min()), 4) if values.size else None,
            }
            block["fewest_positives_behind_a_pair"] = (
                min(int(a["n_positive"]) for a in block["away"]) if block["away"] else None
            )
            block["home_bias"] = {
                "mean": round(float(homes.mean()), 4) if homes.size else None,
                "sd_across_sources": round(float(homes.std(ddof=1)), 4) if homes.size > 1 else None,
                "n_sources": int(homes.size),
            }
    return out


# One row of results/rotation.csv per (source, diagnosis, corpus, level, score,
# correction). The grid is a table and is written as one: the same numbers
# nested in JSON run to seven megabytes, and a coverage table is read by column.
GRID_COLUMNS = (
    "source",
    "label",
    "corpus",
    "role",
    "alpha",
    "score",
    "correction",
    "n_points",
    "n_patients",
    "prevalence",
    "n_positive",
    "coverage_mean",
    "coverage_sd",
    "coverage_diagnosis_mean",
    "coverage_diagnosis_sd",
    "coverage_no_diagnosis_mean",
    "coverage_no_diagnosis_sd",
    "mean_set_size_mean",
    "mean_set_size_sd",
    "threshold_no_diagnosis_mean",
    "threshold_diagnosis_mean",
    "threshold_diagnosis_n_infinite",
    "calibration_n_mean",
    "calibration_effective_size_mean",
    "n_draws",
)


def grid_rows(cells: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Flatten every (source, diagnosis, corpus, setting) cell into one row."""
    for entry in cells:
        source, label = entry["source"], entry["label"]
        for row in entry["rows"]:
            for corpus, block in row["by_corpus"].items():
                if corpus.endswith("-calibration-holdout"):
                    role = "calibration holdout"
                elif corpus == source:
                    role = "home"
                else:
                    role = "away"
                thresholds = block["threshold_by_class"]
                yield {
                    "source": source,
                    "label": label,
                    "corpus": corpus,
                    "role": role,
                    "alpha": row["alpha"],
                    "score": row["score"],
                    "correction": row["correction"],
                    "n_points": block["n_points"],
                    "n_patients": block["n_patients"],
                    "prevalence": round(block["prevalence"], 5),
                    "n_positive": int(round(block["prevalence"] * block["n_points"])),
                    "coverage_mean": round(block["coverage"]["mean"], 4),
                    "coverage_sd": round(block["coverage"]["sd"], 4),
                    "coverage_diagnosis_mean": round(
                        block["coverage_by_class"][POSITIVE]["mean"], 4
                    ),
                    "coverage_diagnosis_sd": round(block["coverage_by_class"][POSITIVE]["sd"], 4),
                    "coverage_no_diagnosis_mean": round(block["coverage_by_class"]["0"]["mean"], 4),
                    "coverage_no_diagnosis_sd": round(block["coverage_by_class"]["0"]["sd"], 4),
                    "mean_set_size_mean": round(block["mean_set_size"]["mean"], 4),
                    "mean_set_size_sd": round(block["mean_set_size"]["sd"], 4),
                    "threshold_no_diagnosis_mean": round(thresholds["0"]["mean"], 5),
                    "threshold_diagnosis_mean": round(thresholds[POSITIVE]["mean"], 5),
                    "threshold_diagnosis_n_infinite": thresholds[POSITIVE]["n_infinite"],
                    "calibration_n_mean": round(row["calibration"]["n"]["mean"], 1),
                    "calibration_effective_size_mean": round(
                        block["calibration"]["effective_sample_size"]["mean"], 1
                    ),
                    "n_draws": block["coverage"]["n_draws"],
                }


def write_grid(cells: list[dict[str, Any]], path: Path) -> int:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GRID_COLUMNS))
        writer.writeheader()
        written = 0
        for row in grid_rows(cells):
            writer.writerow(row)
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=200)
    args = parser.parse_args()

    results = Path(RESULTS_DIR)
    started = time.time()

    indices = {corpus: corpus_index(corpus) for corpus in SOURCES}
    scores: dict[tuple[str, str, str], Scores] = {}
    for source in SOURCES:
        directory = results / "rotation" / source / "scores"
        scores[(source, source, "cal")] = Scores(directory / f"{source}_cal.npz")
        for corpus in SOURCES:
            scores[(source, corpus, "test")] = Scores(directory / f"{corpus}_test.npz")

    # The ids a part was scored on are the ids the index gives, minus anything the
    # reader had to drop, so the patient vectors follow the scored ids.
    patients: dict[str, list[str]] = {}
    for source in SOURCES:
        patients[f"{source}-cal"] = _patients_for(
            indices[source], scores[(source, source, "cal")].ids
        )
    for corpus in SOURCES:
        patients[f"{corpus}-test"] = _patients_for(
            indices[corpus], scores[(SOURCES[0], corpus, "test")].ids
        )

    cells: list[dict[str, Any]] = []
    for source in SOURCES:
        for label in usable_classes(source):
            cell_started = time.time()
            rows = cell(source, label, scores, patients, args.draws)
            cells.append({"source": source, "label": label, "rows": rows})
            print(
                f"  {source} / {label}: {len(rows)} rows, {time.time() - cell_started:.0f} s",
                flush=True,
            )

    configs = {
        source: json.loads((results / "rotation" / source / "config.json").read_text())
        for source in SOURCES
    }
    out: dict[str, Any] = {
        "written_by": "scripts/rotation_table.py",
        "commit": commit(),
        "machine": machine_info(),
        "settings": {
            "alphas": list(ALPHAS),
            "scores": list(SCORES),
            "corrections": list(CORRECTIONS),
            "n_draws": args.draws,
            "headline": {"alpha": HEADLINE_ALPHA, "score": HEADLINE_SCORE},
            "corpus_names": CORPUS_NAMES,
            "classes": class_keys(),
        },
        "sources": {
            source: {
                "n_train": config["n_train"],
                "n_validation": config["n_validation"],
                "split_sizes": config["split_sizes"],
                "usable_classes": config["usable_classes"],
                "standardisation": config["standardisation"],
                "deviations": config["deviations"],
                "epochs": config["epochs"],
                "val_macro_auroc": json.loads(
                    (results / "rotation" / source / "metrics.json").read_text()
                )["val_macro_auroc"],
            }
            for source, config in configs.items()
        },
        "grid": {
            "file": "results/rotation.csv",
            "columns": list(GRID_COLUMNS),
            "roles": {
                "home": "the source's own test part, scored with its own model and threshold",
                "away": "another corpus's test part, scored with this source's model and "
                "threshold, never re-calibrated on itself",
                "calibration holdout": "the half of the source's calibration part that did "
                "not fit the threshold on that draw",
            },
            "weighting": {correction: note for correction, note in WEIGHTING_NOTE.items()},
        },
        "bias": bias_summary(cells),
        "seconds": round(time.time() - started, 1),
    }
    grid_path = results / "rotation.csv"
    grid: dict[str, Any] = out["grid"]
    grid["n_rows"] = write_grid(cells, grid_path)
    n_rows = grid["n_rows"]
    path = results / "rotation.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"written {path} and {grid_path} ({n_rows:,} rows) in {out['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
