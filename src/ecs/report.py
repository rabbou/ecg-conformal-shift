"""Coverage as a distribution, not a single number.

One calibration/test split gives one coverage figure, and that figure moves by
several points depending on which half the split happened to pick.  Reporting it
alone invites reading noise as an effect, so every coverage number in this
project is a mean over at least a hundred re-draws with the spread beside it
(C-10).  The re-draw is the standard way of evaluating a split conformal
predictor -- Angelopoulos & Bates, arXiv:2107.07511, section 3.

The split is drawn over patients, never over records (C-4): two tracings of one
patient on opposite sides of the boundary are one draw counted twice, and the
coverage they produce is flattered by exactly that much.

*Abstaining* here means returning something other than a single label: either
both labels, which says the tracing is genuinely ambiguous at this confidence,
or none, which says no label is plausible at it.  Both mean the same thing in a
clinic -- this one goes to a human -- and both are counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .conformal import (
    aps_scores_all,
    conformal_quantile,
    lac_scores_all,
    mondrian_quantiles,
    predict_sets,
    predict_sets_per_class,
)
from .metrics import (
    abstention_rate,
    class_conditional_coverage,
    coverage,
    mean_set_size,
    singleton_rate,
)
from .splits import patient_split

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]

__all__ = ["CORRECTIONS", "SCORES", "Spread", "repeated_split_report", "spread"]

SCORES = ("lac", "aps")
CORRECTIONS = ("none", "mondrian")


@dataclass(frozen=True)
class Spread:
    """One quantity across the re-draws: where it sits and how far it moves."""

    mean: float
    sd: float
    n_draws: int

    def as_dict(self) -> dict[str, float | int]:
        return {"mean": self.mean, "sd": self.sd, "n_draws": self.n_draws}


def spread(values: list[float]) -> Spread:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("no draws to summarise")
    return Spread(
        float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0, array.size
    )


def repeated_split_report(
    probs: Array,
    labels: IntArray,
    patients: NDArray[np.str_] | list[str],
    alpha: float,
    score: str = "lac",
    correction: str = "none",
    n_draws: int = 200,
    seed: int = 0,
) -> dict[str, object]:
    """Calibrate on half the patients, measure on the other half, ``n_draws`` times.

    ``correction`` is ``none`` for one threshold shared by both classes, or
    ``mondrian`` for one threshold per class, calibrated inside that class only
    so its miscoverage holds whatever share of the population the class turns
    out to hold.
    """
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}, got {score!r}")
    if correction not in CORRECTIONS:
        raise ValueError(f"correction must be one of {CORRECTIONS}, got {correction!r}")
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    n_classes = probs.shape[1]
    keys = pd.Series(list(patients), index=range(len(labels)))

    covered: list[float] = []
    empty: list[float] = []
    single: list[float] = []
    ambiguous: list[float] = []
    sizes: list[float] = []
    silent: list[float] = []
    per_class: dict[int, list[float]] = {c: [] for c in range(n_classes)}

    for draw in range(n_draws):
        rng = np.random.default_rng(seed + draw)
        part = patient_split(keys, {"calibration": 0.5, "test": 0.5}, seed=seed + draw)
        is_calibration = (part == "calibration").to_numpy()
        all_scores = lac_scores_all(probs) if score == "lac" else aps_scores_all(probs, rng=rng)
        calibration_true = all_scores[is_calibration, labels[is_calibration]]
        test_scores = all_scores[~is_calibration]
        test_labels = labels[~is_calibration]
        if correction == "none":
            sets = predict_sets(test_scores, conformal_quantile(calibration_true, alpha))
        else:
            sets = predict_sets_per_class(
                test_scores,
                mondrian_quantiles(calibration_true, labels[is_calibration], alpha, n_classes),
            )
        counts = sets.sum(axis=1)
        covered.append(coverage(sets, test_labels))
        empty.append(float((counts == 0).mean()))
        single.append(singleton_rate(sets))
        ambiguous.append(float((counts > 1).mean()))
        sizes.append(mean_set_size(sets))
        silent.append(abstention_rate(sets))
        for klass, (value, _support) in class_conditional_coverage(
            sets, test_labels, n_classes
        ).items():
            per_class[klass].append(value)

    return {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "score": score,
        "correction": correction,
        "n_points": int(len(labels)),
        "n_patients": int(keys.nunique()),
        "coverage": spread(covered).as_dict(),
        "empty_rate": spread(empty).as_dict(),
        "one_label_rate": spread(single).as_dict(),
        "two_label_rate": spread(ambiguous).as_dict(),
        "abstention_rate": spread(silent).as_dict(),
        "mean_set_size": spread(sizes).as_dict(),
        "coverage_by_class": {
            str(klass): spread(values).as_dict() for klass, values in per_class.items()
        },
    }
