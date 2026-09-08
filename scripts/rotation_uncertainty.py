"""Three things the rotation table cannot say on its own.

**What the interval carries.** Every spread in ``results/rotation.csv`` is taken
over the two hundred calibration draws, on a target cohort that never moves. It
is the variability of the threshold, not of the population, and a coverage read
on Shandong's 23 left bundle-branch blocks is uncertain because there are 23 of
them, whatever the threshold does. Each target cohort is resampled by patient
here, and the percentile interval is reported beside the draw spread rather than
instead of it.

**What the conformal formalism adds.** A rejection rule with one plain empirical
quantile per class -- Chow, 1970 -- is the same construction minus the
finite-sample correction: Mondrian takes the ``ceil((n+1)(1-alpha))``-th
smallest calibration score, Chow takes the ``(1-alpha)`` quantile. If the two
land in the same place, the conformal machinery is a rename of a per-class
rejection rule and the object has to say so. The gap is measured per row.

**Who the coverage holds for.** A figure that holds over a cohort can fail over
half of it. Every row is repeated by sex and by age band wherever the corpus
carries them, with the number of positives the cell rests on, so a cell too thin
to support a reading says so instead of reading as a result.

Written for the headline setting only -- the 90% level on the LAC score --
because that is where the reading is made.

Output: ``results/rotation_uncertainty.csv``, one row per (source, diagnosis,
corpus, correction, subgroup), and a summary in the matching ``.json``.

Usage: .venv/bin/python scripts/rotation_uncertainty.py [--draws 200] [--boot 2000]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ecs.config import RESULTS_DIR
from ecs.conformal import (
    conformal_quantile,
    lac_scores_all,
    mondrian_quantiles,
    predict_sets_per_class,
)
from ecs.encoders import machine_info
from ecs.rotation import AGE_BANDS, SOURCES, class_keys, corpus_index, usable_classes
from ecs.splits import patient_split

ALPHA = 0.10
SCORE = "lac"
POSITIVE = 1
N_CLASSES = 2
CORRECTIONS = ("none", "mondrian")

# A cell resting on fewer positives than this is reported with its count and
# flagged, not hidden and not read as a result: a 95% interval on twenty cases
# is wider than any difference the rotation is looking for.
THIN_BELOW = 25

COLUMNS = (
    "source",
    "label",
    "corpus",
    "role",
    "correction",
    "subgroup_kind",
    "subgroup",
    "n_positive",
    "n_positive_patients",
    "thin",
    "coverage",
    "sd_over_calibration_draws",
    "bootstrap_lo",
    "bootstrap_hi",
    "bootstrap_width",
    "chow_coverage",
    "conformal_minus_chow",
)


def commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def chow_quantiles(
    scores: NDArray[np.float64], labels: NDArray[np.int_], alpha: float
) -> NDArray[np.float64]:
    """One plain empirical quantile per class: the rejection rule, no correction.

    The difference from ``mondrian_quantiles`` is the whole of what split
    conformal adds here -- the ``(n+1)`` term that turns a sample quantile into a
    finite-sample guarantee.
    """
    out = np.empty(N_CLASSES, dtype=np.float64)
    for klass in range(N_CLASSES):
        inside = scores[labels == klass]
        out[klass] = float(np.quantile(inside, 1.0 - alpha)) if inside.size else np.inf
    return out


def bootstrap_by_patient(
    covered: NDArray[np.float64],
    patients: NDArray[np.str_],
    boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """A percentile interval for the mean of ``covered``, resampling patients.

    Resampling records would treat two tracings of one patient as two draws from
    the population, the same error the splits rule out; the unit of resampling is
    the unit of independence.
    """
    if covered.size == 0:
        return (float("nan"), float("nan"))
    keys, index = np.unique(patients, return_inverse=True)
    totals = np.bincount(index, weights=covered, minlength=len(keys))
    counts = np.bincount(index, minlength=len(keys)).astype(np.float64)
    drawn = rng.integers(0, len(keys), size=(boot, len(keys)))
    numerator = totals[drawn].sum(axis=1)
    denominator = counts[drawn].sum(axis=1)
    means = numerator / np.where(denominator == 0, np.nan, denominator)
    return (float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5)))


def subgroup_masks(
    sex: NDArray[np.str_], band: NDArray[np.str_]
) -> list[tuple[str, str, NDArray[np.bool_]]]:
    """Every slice a row is repeated over: the whole cohort, then sex, then age.

    A subgroup the corpus does not record -- an unknown sex, a missing age -- is
    left out rather than gathered into a bucket of its own, and the cohort row
    above it still holds those records.
    """
    out: list[tuple[str, str, NDArray[np.bool_]]] = [("all", "all", np.ones(len(sex), dtype=bool))]
    for value in ("male", "female"):
        out.append(("sex", value, sex == value))
    for name, _low, _high in AGE_BANDS:
        out.append(("age", name, band == name))
    return out


class Scored:
    """One model's probabilities for one part of one corpus."""

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.ids = [str(i) for i in data["ids"]]
            self.y = np.asarray(data["y"], dtype=int)
            self.p = np.asarray(data["p"], dtype=np.float64)
            self.classes = [str(c) for c in data["classes"]]

    def binary(self, label: str) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        column = self.classes.index(label)
        p = self.p[:, column]
        return np.column_stack([1.0 - p, p]), self.y[:, column]


def run(draws: int, boot: int, seed: int) -> list[dict[str, Any]]:
    results = Path(RESULTS_DIR)
    indices = {corpus: corpus_index(corpus) for corpus in SOURCES}
    scores: dict[tuple[str, str, str], Scored] = {}
    for source in SOURCES:
        directory = results / "rotation" / source / "scores"
        scores[(source, source, "cal")] = Scored(directory / f"{source}_cal.npz")
        for corpus in SOURCES:
            scores[(source, corpus, "test")] = Scored(directory / f"{corpus}_test.npz")

    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        calibration = scores[(source, source, "cal")]
        cal_patients = pd.Series(
            [str(p) for p in indices[source].frame.loc[calibration.ids, "patient"]],
            index=range(len(calibration.ids)),
        )
        for label in usable_classes(source):
            started = time.time()
            cal_probs, cal_labels = calibration.binary(label)
            cal_all = lac_scores_all(cal_probs)

            # Everything about a target that does not change between draws.
            targets: dict[str, dict[str, Any]] = {}
            for corpus in SOURCES:
                if label not in usable_classes(corpus):
                    continue
                test = scores[(source, corpus, "test")]
                probs, labels = test.binary(label)
                frame = indices[corpus].frame.loc[test.ids]
                sick = labels == POSITIVE
                targets[corpus] = {
                    "all_scores": lac_scores_all(probs),
                    "sick": sick,
                    "patients": np.array([str(p) for p in frame["patient"]])[sick],
                    "groups": subgroup_masks(
                        np.array([str(v) for v in frame["sex"]])[sick],
                        np.array([str(v) for v in frame["age_band"]])[sick],
                    ),
                }

            keys = [
                (corpus, correction, kind, name)
                for corpus, target in targets.items()
                for correction in CORRECTIONS
                for kind, name, _mask in target["groups"]
            ]
            covered = {key: np.zeros(int(targets[key[0]]["sick"].sum())) for key in keys}
            chow_covered = {key: np.zeros_like(covered[key]) for key in keys}
            per_draw: dict[tuple[str, str, str, str], list[float]] = {k: [] for k in keys}

            for draw in range(draws):
                part = patient_split(
                    cal_patients, {"calibration": 0.5, "test": 0.5}, seed=seed + draw
                )
                is_calibration = (part == "calibration").to_numpy()
                true_scores = cal_all[is_calibration, cal_labels[is_calibration]]
                fitted_labels = cal_labels[is_calibration]
                thresholds = {
                    "none": np.full(
                        N_CLASSES, conformal_quantile(true_scores, ALPHA), dtype=np.float64
                    ),
                    "mondrian": mondrian_quantiles(true_scores, fitted_labels, ALPHA, N_CLASSES),
                }
                chow = {
                    "none": np.full(
                        N_CLASSES, float(np.quantile(true_scores, 1.0 - ALPHA)), dtype=np.float64
                    ),
                    "mondrian": chow_quantiles(true_scores, fitted_labels, ALPHA),
                }
                for corpus, target in targets.items():
                    sick = target["sick"]
                    for correction in CORRECTIONS:
                        hit = predict_sets_per_class(target["all_scores"], thresholds[correction])[
                            sick, POSITIVE
                        ].astype(np.float64)
                        chow_hit = predict_sets_per_class(target["all_scores"], chow[correction])[
                            sick, POSITIVE
                        ].astype(np.float64)
                        for kind, name, mask in target["groups"]:
                            key = (corpus, correction, kind, name)
                            covered[key] += np.where(mask, hit, 0.0)
                            chow_covered[key] += np.where(mask, chow_hit, 0.0)
                            per_draw[key].append(
                                float(hit[mask].mean()) if mask.any() else float("nan")
                            )

            rng = np.random.default_rng(seed)
            for corpus, target in targets.items():
                for correction in CORRECTIONS:
                    for kind, name, mask in target["groups"]:
                        key = (corpus, correction, kind, name)
                        n_positive = int(mask.sum())
                        if n_positive == 0:
                            continue
                        inside = covered[key][mask] / draws
                        chow_inside = chow_covered[key][mask] / draws
                        patients = target["patients"][mask]
                        lo, hi = bootstrap_by_patient(inside, patients, boot, rng)
                        conformal = float(np.nanmean(per_draw[key]))
                        chow_coverage = float(chow_inside.mean())
                        rows.append(
                            {
                                "source": source,
                                "label": label,
                                "corpus": corpus,
                                "role": "home" if corpus == source else "away",
                                "correction": correction,
                                "subgroup_kind": kind,
                                "subgroup": name,
                                "n_positive": n_positive,
                                "n_positive_patients": int(len(set(patients))),
                                "thin": n_positive < THIN_BELOW,
                                "coverage": round(conformal, 4),
                                "sd_over_calibration_draws": round(
                                    float(np.nanstd(per_draw[key], ddof=1)), 4
                                ),
                                "bootstrap_lo": round(lo, 4),
                                "bootstrap_hi": round(hi, 4),
                                "bootstrap_width": round(hi - lo, 4),
                                "chow_coverage": round(chow_coverage, 4),
                                "conformal_minus_chow": round(conformal - chow_coverage, 4),
                            }
                        )
            print(f"  {source} / {label}: {time.time() - started:.0f} s", flush=True)
    return rows


def summarise(rows: list[dict[str, Any]], draws: int, boot: int) -> dict[str, Any]:
    whole = [r for r in rows if r["subgroup_kind"] == "all" and r["role"] == "away"]
    by_sex = [r for r in rows if r["subgroup_kind"] == "sex" and r["role"] == "away"]
    by_age = [r for r in rows if r["subgroup_kind"] == "age" and r["role"] == "away"]
    thin = [r for r in rows if r["thin"]]
    spread = []
    for row in whole:
        pair = [
            r
            for r in by_sex
            if (r["source"], r["label"], r["corpus"], r["correction"])
            == (row["source"], row["label"], row["corpus"], row["correction"])
            and not r["thin"]
        ]
        if len(pair) == 2:
            spread.append(abs(pair[0]["coverage"] - pair[1]["coverage"]))
    return {
        "written_by": "scripts/rotation_uncertainty.py",
        "commit": commit(),
        "machine": machine_info(),
        "settings": {
            "alpha": ALPHA,
            "score": SCORE,
            "n_draws": draws,
            "n_bootstrap": boot,
            "bootstrap_unit": "patient of the target test part",
            "corrections": list(CORRECTIONS),
            "classes": class_keys(),
            "age_bands": [name for name, _low, _high in AGE_BANDS],
            "thin_below": THIN_BELOW,
        },
        "reading": {
            "bootstrap_width_against_draw_spread": {
                "median_bootstrap_width": round(
                    float(np.median([r["bootstrap_width"] for r in whole])), 4
                ),
                "median_draw_spread": round(
                    float(np.median([r["sd_over_calibration_draws"] for r in whole])), 4
                ),
            },
            "conformal_minus_chow": {
                "median": round(
                    float(np.median([abs(r["conformal_minus_chow"]) for r in whole])), 4
                ),
                "max": round(float(np.max([abs(r["conformal_minus_chow"]) for r in whole])), 4),
            },
            "between_the_sexes": {
                "median_absolute_gap": round(float(np.median(spread)), 4) if spread else None,
                "widest_gap": round(float(np.max(spread)), 4) if spread else None,
                "n_pairs_compared": len(spread),
            },
            "n_rows": len(rows),
            "n_rows_by_sex": len(by_sex),
            "n_rows_by_age": len(by_age),
            "n_rows_too_thin_to_read": len(thin),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    started = time.time()
    rows = run(args.draws, args.boot, args.seed)
    results = Path(RESULTS_DIR)
    path = results / "rotation_uncertainty.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarise(rows, args.draws, args.boot)
    summary["seconds"] = round(time.time() - started, 1)
    (results / "rotation_uncertainty.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"written {path} ({len(rows)} rows) in {summary['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
