"""Coverage metrics: set size, class-conditional coverage, effective sample size.

A coverage figure on its own can mislead in three ways.  A set that always
contains every class covers perfectly and says nothing; a global 90% can come
from covering the healthy majority while missing most of the sick; a weighted
correction can restore coverage while resting on a handful of patients.  Set
size, class-conditional coverage and effective sample size expose those three
cases.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]

__all__ = [
    "abstention_rate",
    "bootstrap_ci",
    "class_conditional_coverage",
    "coverage",
    "effective_sample_size",
    "mean_set_size",
    "singleton_rate",
    "wilson_interval",
]


# Share of bootstrap draws that must hold both classes for the percentile
# interval to be reported at all.
MIN_USABLE_DRAWS = 0.95


def coverage(sets: BoolArray, labels: IntArray) -> float:
    """Fraction of test points whose prediction set contains the true class."""
    _check_sets(sets, labels)
    return float(sets[np.arange(len(labels)), labels].mean())


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because coverage sits near 0.9 and
    the sample of positives can be small -- exactly where the naive interval
    runs past 1 or understates the width.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} outside [0, {n}]")
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    p = successes / n
    denominator = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = (z / denominator) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float(centre - half), float(centre + half)


def bootstrap_ci(
    statistic: Callable[[IntArray, Array], float],
    labels: IntArray,
    scores: Array,
    n_draws: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """The statistic on the sample, and the percentile interval around it.

    Returns ``(point, low, high)``.  The interval comes from resampling the
    test points with replacement ``n_draws`` times, which is what makes a
    claim that one arm beats another checkable (C-18): two point estimates a
    hundredth apart with overlapping intervals are not a difference.

    A draw that happens to contain one class only leaves AUROC undefined and is
    dropped.  Dropping draws biases the percentiles, so it is tolerated only
    while it stays rare: if fewer than ``MIN_USABLE_DRAWS`` of the draws hold
    both classes, the sample is too small for an interval to mean anything and
    that is an error rather than a number computed on what is left.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores):
        raise ValueError(f"{len(labels)} labels for {len(scores)} scores")
    rng = np.random.default_rng(seed)
    drawn = []
    for _ in range(n_draws):
        index = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[index])) < 2:
            continue
        drawn.append(statistic(labels[index], scores[index]))
    if len(drawn) < MIN_USABLE_DRAWS * n_draws:
        raise ValueError(
            f"only {len(drawn)} of {n_draws} draws held both classes; "
            f"the sample of {len(labels)} is too small for an interval"
        )
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(drawn, [100 * tail, 100 * (1 - tail)])
    return float(statistic(labels, scores)), float(low), float(high)


def class_conditional_coverage(
    sets: BoolArray, labels: IntArray, n_classes: int | None = None
) -> dict[int, tuple[float, int]]:
    """Coverage within each true class, with that class's support.

    The clinically load-bearing number.  A model may hold 90% overall while
    covering the infarction class far less often, because the healthy majority
    carries the average.
    """
    _check_sets(sets, labels)
    k = sets.shape[1] if n_classes is None else n_classes
    out: dict[int, tuple[float, int]] = {}
    for c in range(k):
        mask = labels == c
        n = int(mask.sum())
        out[c] = (float(sets[mask, c].mean()) if n else float("nan"), n)
    return out


def mean_set_size(sets: BoolArray) -> float:
    return float(sets.sum(axis=1).mean())


def singleton_rate(sets: BoolArray) -> float:
    """Fraction of test points given exactly one class -- a committed answer."""
    return float((sets.sum(axis=1) == 1).mean())


def abstention_rate(sets: BoolArray) -> float:
    """Fraction of test points the model declines to resolve.

    Counts both the ambiguous case (more than one class kept) and the empty set
    (no class plausible at this level), because clinically both mean the same
    thing: this one goes to a human.
    """
    sizes = sets.sum(axis=1)
    return float((sizes != 1).mean())


def effective_sample_size(weights: Array) -> float:
    """Kish effective sample size, (sum w)^2 / sum w^2.

    The diagnostic that stops a weighted correction from being believed on
    faith.  If 2,000 calibration points collapse to an effective 30, the
    restored coverage rests on 30 patients and the interval around it is wide,
    whatever the point estimate says.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0:
        raise ValueError("weights must be non-empty")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    squared = float(np.sum(weights**2))
    if squared == 0.0:
        return 0.0
    return float(np.sum(weights) ** 2 / squared)


def _check_sets(sets: BoolArray, labels: IntArray) -> None:
    if sets.ndim != 2:
        raise ValueError(f"sets must be 2-D (n, K), got shape {sets.shape}")
    if len(labels) != sets.shape[0]:
        raise ValueError(f"{len(labels)} labels for {sets.shape[0]} rows")
    if labels.size and (labels.min() < 0 or labels.max() >= sets.shape[1]):
        raise ValueError("labels index outside the class axis")
