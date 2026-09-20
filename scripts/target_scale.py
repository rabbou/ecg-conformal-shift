"""How many labelled records of its own a hospital needs before the guarantee holds there.

The break table answers "what is the coverage worth at another hospital".  This
answers the question that follows it, which is the one a hospital actually asks:
if I label some of my own tracings, how many buy the level back?

One pair, the hardest one on the table: PTB-XL to Chongqing, infarction.  Four
rungs -- 0, 100, 500 and 2,000 labelled target records -- and at each rung two
ways of spending them:

``recalibrated``  the threshold is fitted on the target records alone, which is
                  what "recalibrate on site" means;
``pooled``        the target records are added to the source calibration half,
                  which is what a site with too few records would try instead.

Rung zero is the frozen source threshold, so the ladder starts exactly where the
break table left off.

Two things are held fixed so that a difference between rungs is the threshold
and nothing else.  Chongqing is cut once, by patient, into a pool the target
records are drawn from and an evaluation half they are never drawn from; every
rung is measured on that same evaluation half.  And the draws are shared: rung
zero and rung 2,000 see the same two hundred source calibration halves in the
same order, so the comparison between them can be taken inside a draw.

The weighted correction has no meaning on this ladder above rung zero -- once the
threshold is fitted on target records there is no source prior to reweight from --
so it is reported at rung zero only, where it is the break table's own cell.

Usage: .venv/bin/python scripts/target_scale.py [--draws 200]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ecs.config import ACS_DIR, ACS_LABELLED_SPLIT, PTBXL_DIR, RESULTS_DIR
from ecs.conformal import (
    aps_scores_all,
    class_prior,
    label_shift_weights,
    lac_scores_all,
    predict_sets_per_class,
)
from ecs.encoders import machine_info
from ecs.metrics import (
    class_conditional_coverage,
    coverage,
    effective_sample_size,
    mean_set_size,
)
from ecs.report import SCORES, estimated_prior, frozen_threshold, spread
from ecs.splits import patient_split

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]

# The rungs, in labelled target records.  Zero is the frozen source threshold.
RUNGS = (0, 100, 500, 2000)
FAMILIES = ("recalibrated", "pooled")
ALPHAS = (0.20, 0.10, 0.05)
HEADLINE_ALPHA = 0.10
HEADLINE_SCORE = "lac"
N_CLASSES = 2
POOL_SHARE = 0.5  # of Chongqing's patients; the rest is the evaluation half

# Only the two corrections that can be fitted on target records alone.  The
# weighted one needs a source prior to reweight from and is reported at rung
# zero, where the source is still what calibrated the threshold.
LADDER_CORRECTIONS = ("none", "mondrian")
RUNG_ZERO_CORRECTIONS = ("none", "mondrian", "weighted")


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def acs_patients(ids: list[str]) -> list[str]:
    """The patient each Chongqing record belongs to, in the order of ``ids``."""
    table = pd.read_csv(ACS_DIR / ACS_LABELLED_SPLIT)
    by_record = {
        str(f).removesuffix(".dat"): str(p)
        for f, p in zip(table["ecg_row_record"], table["Patient_id"], strict=True)
    }
    return [by_record[i] for i in ids]


def ptbxl_patients(ids: list[str]) -> list[str]:
    database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    return [str(database.loc[int(i), "patient_id"]) for i in ids]


def rows_by_patient(pool: NDArray[np.int_], patients: NDArray[np.str_]) -> list[NDArray[np.int_]]:
    """The pool's rows grouped by patient, computed once for every draw."""
    grouped: dict[str, list[int]] = {}
    for row in pool:
        grouped.setdefault(str(patients[row]), []).append(int(row))
    return [np.array(rows, dtype=int) for _, rows in sorted(grouped.items())]


def draw_target_records(
    groups: list[NDArray[np.int_]], n_wanted: int, rng: np.random.Generator
) -> NDArray[np.int_]:
    """At most ``n_wanted`` records from the pool, taken by whole patients.

    Drawing records rather than patients would put two tracings of one patient
    in a calibration sample that is supposed to hold exchangeable draws from the
    population, the same error C-4 rules out at the split.  A patient whose
    records would overflow the rung is skipped rather than split, so a rung can
    land a record or two short; the calibration size is reported per draw
    instead of being assumed equal to the rung.
    """
    chosen: list[int] = []
    for index in rng.permutation(len(groups)):
        rows = groups[index]
        if len(chosen) + len(rows) > n_wanted:
            continue
        chosen.extend(int(r) for r in rows)
        if len(chosen) == n_wanted:
            break
    return np.array(sorted(chosen), dtype=int)


def _all_scores(probs: Array, score: str, rng: np.random.Generator) -> Array:
    return lac_scores_all(probs) if score == "lac" else aps_scores_all(probs, rng=rng)


class Cell:
    """One (alpha, score, correction, family, rung), gathering a figure per draw."""

    def __init__(self) -> None:
        self.covered: list[float] = []
        self.sick: list[float] = []
        self.well: list[float] = []
        self.sizes: list[float] = []
        self.thresholds: list[list[float]] = []
        self.n_calibration: list[float] = []
        self.n_by_class: list[list[float]] = []
        self.ess: list[float] = []

    def add(
        self,
        sets: NDArray[np.bool_],
        labels: IntArray,
        qhat: Array,
        calibration_labels: IntArray,
        ess: float | None = None,
    ) -> None:
        self.covered.append(coverage(sets, labels))
        by_class = class_conditional_coverage(sets, labels, N_CLASSES)
        self.well.append(by_class[0][0])
        self.sick.append(by_class[1][0])
        self.sizes.append(mean_set_size(sets))
        self.thresholds.append([float(v) for v in qhat])
        self.n_calibration.append(float(len(calibration_labels)))
        self.n_by_class.append([float((calibration_labels == c).sum()) for c in range(N_CLASSES)])
        self.ess.append(float(len(calibration_labels)) if ess is None else ess)

    def as_dict(self) -> dict[str, Any]:
        drawn = np.asarray(self.thresholds, dtype=np.float64)
        return {
            "coverage": spread(self.covered).as_dict(),
            "coverage_by_class": {
                "0": spread(self.well).as_dict(),
                "1": spread(self.sick).as_dict(),
            },
            "mean_set_size": spread(self.sizes).as_dict(),
            "calibration": {
                "n": spread(self.n_calibration).as_dict(),
                "n_by_class": {
                    str(c): spread([row[c] for row in self.n_by_class]).as_dict()
                    for c in range(N_CLASSES)
                },
                "effective_sample_size": spread(self.ess).as_dict(),
            },
            "threshold_by_class": {
                str(c): _threshold_spread(list(drawn[:, c])) for c in range(N_CLASSES)
            },
        }


def _threshold_spread(values: list[float]) -> dict[str, float | int]:
    """Where a threshold sat, with the infinite draws counted rather than averaged.

    An infinite threshold is the honest answer when a class has fewer
    calibration points than the level needs -- ceil(1/alpha) - 1 of them -- and it
    is the reading the low rungs of this ladder are for.  Averaging it into a
    finite mean would hide exactly what the rung is measuring.
    """
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "mean": float(finite.mean()) if finite.size else float("inf"),
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "n_draws": int(array.size),
        "n_infinite": int(array.size - finite.size),
    }


def run(draws: int, seed: int) -> dict[str, Any]:
    started = time.time()
    with np.load(RESULTS_DIR / "baseline/scores.npz", allow_pickle=False) as data:
        source_ids = [str(i) for i in data["ids"]]
        source_labels = np.asarray(data["labels"])
        source_probs = np.asarray(data["probs"], dtype=np.float64)
    with np.load(RESULTS_DIR / "external/acs.npz", allow_pickle=False) as data:
        target_ids = [str(i) for i in data["ids"]]
        target_labels = np.asarray(data["labels"])
        target_probs = np.asarray(data["probs"], dtype=np.float64)

    source_keys = pd.Series(ptbxl_patients(source_ids), index=range(len(source_labels)))
    target_patients = np.array(acs_patients(target_ids))

    # Chongqing is cut once, by patient: a pool the labelled records come from,
    # and an evaluation half no rung ever calibrates on.
    side = patient_split(
        pd.Series(target_patients, index=range(len(target_labels))),
        {"pool": POOL_SHARE, "eval": 1.0 - POOL_SHARE},
        seed=seed,
    )
    pool = np.flatnonzero((side == "pool").to_numpy())
    held_out = np.flatnonzero((side == "eval").to_numpy())
    groups = rows_by_patient(pool, target_patients)

    cells: dict[tuple[float, str, str, str, int], Cell] = {}

    def cell_for(key: tuple[float, str, str, str, int]) -> Cell:
        return cells.setdefault(key, Cell())

    for score in SCORES:
        for draw in range(draws):
            rng = np.random.default_rng(seed + draw)
            source_part = patient_split(
                source_keys, {"calibration": 0.5, "test": 0.5}, seed=seed + draw
            )
            is_calibration = (source_part == "calibration").to_numpy()
            source_all = _all_scores(source_probs, score, rng)
            source_true = source_all[is_calibration, source_labels[is_calibration]]
            source_cal_labels = source_labels[is_calibration]

            target_all = _all_scores(target_probs, score, rng)
            evaluated = target_all[held_out]
            evaluated_labels = target_labels[held_out]
            target_prior_prediction = target_probs[held_out].argmax(axis=1)

            for rung in RUNGS:
                if rung == 0:
                    for correction in RUNG_ZERO_CORRECTIONS:
                        prior = None
                        ess: float | None = None
                        if correction == "weighted":
                            prior = estimated_prior(
                                target_prior_prediction,
                                source_probs[is_calibration].argmax(axis=1),
                                source_cal_labels,
                                N_CLASSES,
                            )
                            if prior is None:
                                # BBSE refused: label shift does not explain this
                                # target, so its estimated prior must not be spent.
                                continue
                            weights = label_shift_weights(
                                class_prior(source_cal_labels, N_CLASSES), prior
                            )
                            ess = effective_sample_size(weights[source_cal_labels])
                        for alpha in ALPHAS:
                            qhat = frozen_threshold(
                                source_true,
                                source_cal_labels,
                                alpha,
                                correction,
                                N_CLASSES,
                                prior,
                            )
                            for family in FAMILIES:
                                cell_for((alpha, score, correction, family, 0)).add(
                                    predict_sets_per_class(evaluated, qhat),
                                    evaluated_labels,
                                    qhat,
                                    source_cal_labels,
                                    ess,
                                )
                    continue

                rows = draw_target_records(groups, rung, rng)
                target_true = target_all[rows, target_labels[rows]]
                for family in FAMILIES:
                    if family == "recalibrated":
                        scores_used = target_true
                        labels_used = target_labels[rows]
                    else:
                        scores_used = np.concatenate([source_true, target_true])
                        labels_used = np.concatenate([source_cal_labels, target_labels[rows]])
                    for correction in LADDER_CORRECTIONS:
                        for alpha in ALPHAS:
                            qhat = frozen_threshold(
                                scores_used, labels_used, alpha, correction, N_CLASSES, None
                            )
                            cell_for((alpha, score, correction, family, rung)).add(
                                predict_sets_per_class(evaluated, qhat),
                                evaluated_labels,
                                qhat,
                                labels_used,
                            )

    rows_out = [
        {
            "alpha": alpha,
            "target_coverage": 1.0 - alpha,
            "score": score,
            "correction": correction,
            "family": family,
            "n_target_records": rung,
            "calibrated_on": (
                "PTB-XL alone"
                if rung == 0
                else (
                    f"{rung} labelled Chongqing records"
                    if family == "recalibrated"
                    else f"the PTB-XL calibration half plus {rung} labelled Chongqing records"
                )
            ),
            **cell.as_dict(),
        }
        for (alpha, score, correction, family, rung), cell in sorted(
            cells.items(), key=lambda item: item[0]
        )
    ]

    return {
        "written_by": "scripts/target_scale.py",
        "commit": git_commit(),
        "machine": machine_info(),
        "pair": {
            "source": "ptbxl",
            "target": "acs",
            "label": "infarction",
            "model": "the PTB-XL supervised baseline, frozen (results/baseline/model.pt)",
        },
        "settings": {
            "rungs": list(RUNGS),
            "families": list(FAMILIES),
            "alphas": list(ALPHAS),
            "scores": list(SCORES),
            "n_draws": draws,
            "seed": seed,
            "headline": {"alpha": HEADLINE_ALPHA, "score": HEADLINE_SCORE},
        },
        "split": {
            "how": "Chongqing cut once by patient; the evaluation half never calibrates",
            "n_pool": int(pool.size),
            "n_eval": int(held_out.size),
            "n_pool_patients": int(len(set(target_patients[pool]))),
            "n_eval_patients": int(len(set(target_patients[held_out]))),
            "eval_prevalence": float(np.mean(target_labels[held_out])),
            "pool_prevalence": float(np.mean(target_labels[pool])),
            "source_prior": [float(v) for v in class_prior(source_labels, N_CLASSES)],
        },
        "rows": rows_out,
        "seconds": round(time.time() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(RESULTS_DIR / "target_scale.json"))
    args = parser.parse_args(argv)
    out = run(args.draws, args.seed)
    path = Path(args.out)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"written {path} in {out['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
