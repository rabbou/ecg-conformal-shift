"""Reporting a coverage number honestly.

Every function here exists because a coverage figure on its own can hide the
thing that matters.  A set that always contains every class covers perfectly and
says nothing; a global 90% can be bought by covering the healthy majority while
missing most of the sick; a weighted correction can "restore" coverage by
leaning on five patients.  Size, class-conditional coverage and effective sample
size are what keep those three failures visible.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]

__all__ = [
    "abstention_rate",
    "class_conditional_coverage",
    "coverage",
    "effective_sample_size",
    "mean_set_size",
    "singleton_rate",
    "wilson_interval",
]


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
