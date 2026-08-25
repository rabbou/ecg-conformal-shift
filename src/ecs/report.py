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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

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
    effective_sample_size,
    mean_set_size,
    singleton_rate,
)
from .splits import patient_split

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]

__all__ = [
    "CORRECTIONS",
    "SCORES",
    "Source",
    "Spread",
    "Target",
    "frozen_calibration_table",
    "frozen_threshold",
    "repeated_split_report",
    "spread",
]

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


# ---------------------------------------------------------------------------
# The frozen calibration
#
# A coverage guarantee is a promise about the population the calibration sample
# came from.  Measuring what it is worth somewhere else means fitting the
# threshold on the source and spending it, unchanged, on the target -- which is
# also the only honest protocol, because a threshold re-fitted on the target
# would measure nothing but the target's own difficulty (C-20).
#
# The spread still has to come from somewhere, since one calibration half gives
# one number and that number moves by points depending on which half was drawn
# (C-10).  So the re-draw is over the SOURCE alone: each draw halves PTB-XL by
# patient, fits a threshold on one half, and spends it on the other half AND on
# every external corpus at once.  The external corpora are never subsampled;
# each is scored whole, every draw, which is what makes the three panels
# comparable -- same threshold, same draw, different population.
#
# The re-draw itself is the standard evaluation of a split conformal predictor:
# Angelopoulos & Bates, arXiv:2107.07511, section 3.
# ---------------------------------------------------------------------------


def frozen_threshold(
    calibration_scores: Array,
    calibration_labels: IntArray,
    alpha: float,
    correction: str,
    n_classes: int,
) -> Array:
    """The threshold, fitted on the calibration sample and nothing else.

    The signature is the guarantee C-20 asks for: no target corpus is in scope
    here, so the threshold cannot be re-fitted on the corpus it will be spent
    on, whatever anyone later adds to the caller.

    Returns one threshold per candidate class -- the same value repeated when
    the correction shares one across classes -- so that applying it is the same
    operation either way.
    """
    if correction not in CORRECTIONS:
        raise ValueError(f"correction must be one of {CORRECTIONS}, got {correction!r}")
    if correction == "none":
        shared = conformal_quantile(calibration_scores, alpha)
        return np.full(n_classes, shared, dtype=np.float64)
    return mondrian_quantiles(calibration_scores, calibration_labels, alpha, n_classes)


@dataclass(frozen=True)
class Source:
    """PTB-XL: the only corpus a threshold is ever fitted on.

    Each draw halves it by patient; the half that did not calibrate is what the
    in-distribution panel reports, so the source is held to exactly the protocol
    the external corpora are held to.
    """

    name: str
    probs: Array
    labels: IntArray
    patients: Sequence[str]
    deviations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Target:
    """A corpus the frozen threshold is spent on, whole and unchanged.

    ``deviations`` carries what the ingestion and label chains could not make
    identical to the source's (C-14); it travels with the numbers rather than
    being remembered separately.
    """

    probs: Array
    labels: IntArray
    n_patients: int
    deviations: tuple[str, ...] = field(default_factory=tuple)


def _all_scores(probs: Array, score: str, rng: np.random.Generator) -> Array:
    if score == "lac":
        return lac_scores_all(probs)
    return aps_scores_all(probs, rng=rng)


def _threshold_spread(values: list[float]) -> dict[str, float | int]:
    """Where a threshold sat across the draws, and how often it was infinite.

    An infinite threshold is not a large number to be averaged in: it means the
    calibration class was too small to certify the level and every set became
    the full label set.  Averaging it with finite draws would hide that, so the
    infinite draws are counted separately and the spread describes the rest.
    """
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    n_infinite = int(array.size - finite.size)
    if finite.size == 0:
        return {
            "mean": float("inf"),
            "sd": 0.0,
            "n_draws": int(array.size),
            "n_infinite": n_infinite,
        }
    return {
        "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "n_draws": int(array.size),
        "n_infinite": n_infinite,
    }


class _Accumulator:
    """One corpus at one setting, gathering a figure per draw."""

    def __init__(self, n_classes: int) -> None:
        self.covered: list[float] = []
        self.empty: list[float] = []
        self.single: list[float] = []
        self.ambiguous: list[float] = []
        self.sizes: list[float] = []
        self.silent: list[float] = []
        self.per_class: dict[int, list[float]] = {c: [] for c in range(n_classes)}

    def add(self, sets: NDArray[np.bool_], labels: IntArray, n_classes: int) -> None:
        counts = sets.sum(axis=1)
        self.covered.append(coverage(sets, labels))
        self.empty.append(float((counts == 0).mean()))
        self.single.append(singleton_rate(sets))
        self.ambiguous.append(float((counts > 1).mean()))
        self.sizes.append(mean_set_size(sets))
        self.silent.append(abstention_rate(sets))
        for klass, (value, _support) in class_conditional_coverage(sets, labels, n_classes).items():
            self.per_class[klass].append(value)

    def as_dict(self) -> dict[str, object]:
        return {
            "coverage": spread(self.covered).as_dict(),
            "empty_rate": spread(self.empty).as_dict(),
            "one_label_rate": spread(self.single).as_dict(),
            "two_label_rate": spread(self.ambiguous).as_dict(),
            "abstention_rate": spread(self.silent).as_dict(),
            "mean_set_size": spread(self.sizes).as_dict(),
            "coverage_by_class": {
                str(klass): spread(values).as_dict() for klass, values in self.per_class.items()
            },
        }


def frozen_calibration_table(
    source: Source,
    targets: Mapping[str, Target],
    alphas: Sequence[float],
    n_draws: int = 200,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Coverage on every corpus under one PTB-XL threshold, over ``n_draws`` draws.

    One row per (level, score, correction).  Each row carries the threshold that
    was spent, the calibration sample it came from with that sample's effective
    size (C-9), and a block per corpus holding coverage, coverage per class
    (C-11) and the set-size shares -- each a mean over the draws with its
    standard deviation (C-10).
    """
    n_classes = source.probs.shape[1]
    keys = pd.Series(list(source.patients), index=range(len(source.labels)))
    corpora = [source.name, *targets]
    settings = [
        (alpha, score, correction)
        for alpha in alphas
        for score in SCORES
        for correction in CORRECTIONS
    ]
    gathered = {
        setting: {name: _Accumulator(n_classes) for name in corpora} for setting in settings
    }
    thresholds: dict[tuple[float, str, str], list[list[float]]] = {s: [] for s in settings}
    calibration_n: dict[tuple[float, str, str], list[float]] = {s: [] for s in settings}
    calibration_ess: dict[tuple[float, str, str], list[float]] = {s: [] for s in settings}
    calibration_by_class: dict[tuple[float, str, str], list[list[float]]] = {
        s: [] for s in settings
    }

    for score in SCORES:
        for draw in range(n_draws):
            rng = np.random.default_rng(seed + draw)
            part = patient_split(keys, {"calibration": 0.5, "test": 0.5}, seed=seed + draw)
            is_calibration = (part == "calibration").to_numpy()
            source_all = _all_scores(source.probs, score, rng)
            calibration_true = source_all[is_calibration, source.labels[is_calibration]]
            calibration_labels = source.labels[is_calibration]
            evaluated: dict[str, tuple[Array, IntArray]] = {
                source.name: (source_all[~is_calibration], source.labels[~is_calibration]),
                **{
                    name: (_all_scores(target.probs, score, rng), target.labels)
                    for name, target in targets.items()
                },
            }
            per_class_n = [float((calibration_labels == c).sum()) for c in range(n_classes)]
            uniform = np.ones(calibration_true.size, dtype=np.float64)
            for correction in CORRECTIONS:
                for alpha in alphas:
                    setting = (alpha, score, correction)
                    qhat = frozen_threshold(
                        calibration_true, calibration_labels, alpha, correction, n_classes
                    )
                    thresholds[setting].append([float(v) for v in qhat])
                    calibration_n[setting].append(float(calibration_true.size))
                    calibration_ess[setting].append(effective_sample_size(uniform))
                    calibration_by_class[setting].append(per_class_n)
                    for name, (all_scores, labels) in evaluated.items():
                        gathered[setting][name].add(
                            predict_sets_per_class(all_scores, qhat), labels, n_classes
                        )

    supports = {
        source.name: (int(len(source.labels)), int(keys.nunique()), tuple(source.deviations)),
        **{
            name: (int(len(target.labels)), int(target.n_patients), tuple(target.deviations))
            for name, target in targets.items()
        },
    }
    rows: list[dict[str, object]] = []
    for alpha, score, correction in settings:
        setting = (alpha, score, correction)
        drawn = np.asarray(thresholds[setting], dtype=np.float64)
        rows.append(
            {
                "alpha": alpha,
                "target_coverage": 1.0 - alpha,
                "score": score,
                "correction": correction,
                "calibrated_on": source.name,
                "threshold_by_class": {
                    str(c): _threshold_spread(list(drawn[:, c])) for c in range(n_classes)
                },
                "calibration": {
                    "weighting": "uniform: no correction here reweights the calibration sample",
                    "n": spread(calibration_n[setting]).as_dict(),
                    "effective_sample_size": spread(calibration_ess[setting]).as_dict(),
                    "n_by_class": {
                        str(c): spread([row[c] for row in calibration_by_class[setting]]).as_dict()
                        for c in range(n_classes)
                    },
                },
                "by_corpus": {
                    name: {
                        "n_points": supports[name][0],
                        "n_patients": supports[name][1],
                        "prevalence": float(
                            np.mean(source.labels if name == source.name else targets[name].labels)
                        ),
                        "deviations": list(supports[name][2]),
                        **gathered[setting][name].as_dict(),
                    }
                    for name in corpora
                },
            }
        )
    return rows
