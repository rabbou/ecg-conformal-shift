"""Split conformal prediction, and the weighted variant for covariate shift.

Two non-conformity scores are provided:

``lac``
    1 - p(true class).  Gives the smallest average set size at a given coverage
    but its coverage is only marginal -- easy cases get singletons, hard cases
    get small or empty sets.  Sadinle, Lei & Wasserman, JASA 2019.

``aps``
    The cumulative probability mass of all classes ranked at or above the true
    class, optionally randomised so the guarantee is exact rather than
    conservative.  Larger sets, better conditional behaviour.
    Romano, Sesia & Candes, NeurIPS 2020.

The calibration step is the same for both: take the conformal quantile of the
calibration scores, then keep every class whose score falls at or below it.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]

__all__ = [
    "aps_scores",
    "aps_scores_all",
    "bbse_target_prior",
    "class_prior",
    "conformal_quantile",
    "label_shift_quantiles",
    "label_shift_weights",
    "lac_scores",
    "lac_scores_all",
    "mondrian_quantiles",
    "predict_sets",
    "predict_sets_per_class",
    "weighted_conformal_quantile",
]


def lac_scores(probs: Array, labels: IntArray) -> Array:
    """Score of the true class: 1 - p(y)."""
    _check_probs(probs)
    return 1.0 - probs[np.arange(len(labels)), labels]


def lac_scores_all(probs: Array) -> Array:
    """Score of every class, shape (n, K)."""
    _check_probs(probs)
    return 1.0 - probs


def aps_scores(
    probs: Array,
    labels: IntArray,
    rng: np.random.Generator | None = None,
    randomised: bool = True,
) -> Array:
    """Adaptive prediction-set score of the true class."""
    all_scores = aps_scores_all(probs, rng=rng, randomised=randomised)
    return all_scores[np.arange(len(labels)), labels]


def aps_scores_all(
    probs: Array,
    rng: np.random.Generator | None = None,
    randomised: bool = True,
) -> Array:
    """Adaptive prediction-set score of every class, shape (n, K).

    For class k the score is the probability mass of all classes ranked strictly
    above k, plus (a uniform fraction of) the mass of k itself.  Randomising that
    last term is what turns the guarantee from ">= 1 - alpha" into "= 1 - alpha"
    up to 1/(n+1).
    """
    _check_probs(probs)
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)
    mass_above = cumulative - sorted_probs
    if randomised:
        generator = np.random.default_rng() if rng is None else rng
        u = generator.uniform(size=probs.shape)
        scores_sorted = mass_above + u * sorted_probs
    else:
        scores_sorted = cumulative
    scores = np.empty_like(scores_sorted)
    np.put_along_axis(scores, order, scores_sorted, axis=1)
    return scores


def conformal_quantile(scores: Array, alpha: float) -> float:
    """The (1 - alpha) conformal quantile with the finite-sample correction.

    Returns the ceil((n + 1)(1 - alpha))-th smallest calibration score.  When
    that rank exceeds n the sample is too small to certify the level and the
    quantile is +inf, which makes every prediction set the full label set
    rather than a silently invalid threshold.
    """
    _check_alpha(alpha)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
    n = scores.size
    if n == 0:
        raise ValueError("cannot calibrate on an empty score vector")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return math.inf
    return float(np.partition(scores, rank - 1)[rank - 1])


def weighted_conformal_quantile(
    scores: Array,
    weights: Array,
    test_weights: Array,
    alpha: float,
) -> Array:
    """Per-test-point conformal quantile under covariate shift.

    Tibshirani, Barber, Candes & Ramdas (NeurIPS 2019).  Each calibration point
    i carries mass w_i / (sum_j w_j + w(x)) and the test point itself carries
    w(x) / (sum_j w_j + w(x)) at +inf; the quantile is the smallest score whose
    cumulative mass reaches 1 - alpha.  Keeping the point mass at +inf is what
    preserves the guarantee: when the weights say the test point is unlike
    anything in calibration, the set widens to everything instead of lying.

    Returns one quantile per test point, shape (m,).
    """
    _check_alpha(alpha)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    test_weights = np.asarray(test_weights, dtype=np.float64)
    if scores.shape != weights.shape:
        raise ValueError(f"scores {scores.shape} and weights {weights.shape} must match")
    if np.any(weights < 0) or np.any(test_weights < 0):
        raise ValueError("weights must be non-negative")

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    total = cumulative[-1]

    denominator = total + test_weights[:, None]
    normalised = cumulative[None, :] / denominator
    reached = normalised >= (1.0 - alpha)
    idx = np.argmax(reached, axis=1)
    quantiles = np.where(reached.any(axis=1), sorted_scores[idx], np.inf)
    return quantiles.astype(np.float64)


def predict_sets(all_scores: Array, qhat: float | Array) -> NDArray[np.bool_]:
    """Boolean membership matrix: keep every class scoring at or below qhat."""
    all_scores = np.asarray(all_scores, dtype=np.float64)
    if all_scores.ndim != 2:
        raise ValueError(f"all_scores must be 2-D, got shape {all_scores.shape}")
    if isinstance(qhat, (int, float)):
        threshold: float | Array = float(qhat)
    else:
        threshold = np.asarray(qhat, dtype=np.float64)[:, None]
    inside: NDArray[np.bool_] = all_scores <= threshold
    return inside


def _check_probs(probs: Array) -> None:
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (n, K), got shape {probs.shape}")
    if probs.shape[1] < 2:
        raise ValueError("need at least two classes")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError("each row of probs must sum to 1")


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha}")


# ---------------------------------------------------------------------------
# Label shift
#
# A change of hospital moves the share of sick patients far more than it moves
# the shape of the signal: between PTB-XL and Shandong the infarction rate goes
# from 25.1% to 1.01%.  Covariate-shift weighting is the wrong instrument for
# that -- it assumes P(Y|X) is fixed and reweights by the covariate likelihood
# ratio.  Under label shift the stable object is P(X|Y) and what moves is P(Y).
#
# Two corrections apply, and the cheaper one is also the stronger:
#
#   mondrian_quantiles   calibrate separately within each true class.  Valid
#                        under any change of class proportions, exactly and in
#                        finite samples, with no ratio to estimate -- label
#                        conditional validity, Vovk, ACML 2012 (PMLR
#                        25:475-490), Proposition 3.
#   label_shift_quantiles  reweight by w(y) = q(y)/p(y) in the Tibshirani form.
#                        Needs the target prior, so the guarantee is only as
#                        good as the estimate of it -- Podkopaev & Ramdas, UAI
#                        2021 (arXiv:2103.03323), adapting the weighted
#                        exchangeability of Tibshirani et al. from covariates
#                        to labels.
# ---------------------------------------------------------------------------


def mondrian_quantiles(
    scores: Array,
    labels: IntArray,
    alpha: float,
    n_classes: int | None = None,
) -> Array:
    """One conformal quantile per class, calibrated within that class only.

    Because each threshold is computed from that class's own calibration points,
    the miscoverage of class y is controlled at alpha whatever share of the test
    population class y turns out to hold.  That is what makes the construction
    immune to a prevalence change -- and why it needs no importance weights.

    The price is that each class must carry enough calibration points on its
    own: a class with fewer than ceil(1/alpha) - 1 gets an infinite threshold
    and is always included, so the failure is visible rather than silent.
    """
    _check_alpha(alpha)
    labels = np.asarray(labels)
    k = int(labels.max()) + 1 if n_classes is None else n_classes
    out = np.empty(k, dtype=np.float64)
    for c in range(k):
        mask = labels == c
        out[c] = conformal_quantile(scores[mask], alpha) if mask.any() else math.inf
    return out


def label_shift_quantiles(
    scores: Array,
    labels: IntArray,
    class_weights: Array,
    alpha: float,
) -> Array:
    """Conformal quantile per candidate class under a known change of prior.

    ``class_weights[y]`` is w(y) = q(y) / p(y), the target prior over the source
    prior.  Each calibration point carries the weight of its own label, and the
    candidate class contributes the point mass at infinity that keeps the
    guarantee finite-sample.  Returns one threshold per candidate class.
    """
    _check_alpha(alpha)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    class_weights = np.asarray(class_weights, dtype=np.float64)
    if np.any(class_weights < 0):
        raise ValueError("class weights must be non-negative")
    point_weights = class_weights[labels]
    return weighted_conformal_quantile(scores, point_weights, class_weights, alpha)


def bbse_target_prior(
    predictions: IntArray,
    cal_predictions: IntArray,
    cal_labels: IntArray,
    n_classes: int,
) -> Array:
    """Estimate the target class prior from unlabelled target predictions.

    Black Box Shift Estimation (Lipton, Wang & Smola, ICML 2018).  Invert the
    source confusion matrix against the distribution of predictions on the
    target: q(y) solves C @ q = mu, where C[i, j] = P(predict i, true j) on the
    source and mu[i] = P(predict i) on the target.  The predictor may be biased
    or badly calibrated -- only invertibility of C is required.

    Raises when the confusion matrix is singular or the solution leaves the
    simplex, which is the signal that label shift alone does not explain the
    data and that the estimate must not be used.
    """
    confusion = np.zeros((n_classes, n_classes), dtype=np.float64)
    for predicted, true in zip(cal_predictions, cal_labels, strict=True):
        confusion[predicted, true] += 1.0
    confusion /= len(cal_labels)

    mu = np.bincount(predictions, minlength=n_classes).astype(np.float64)
    mu /= mu.sum()

    if abs(np.linalg.det(confusion)) < 1e-12:
        raise ValueError("confusion matrix is singular; BBSE cannot identify the target prior")

    # Solving the joint confusion against the target's prediction marginal yields
    # the RATIO q(y)/p(y), not q(y) itself -- the source prior is already baked
    # into the joint.  Multiplying it back out is what turns the weight into a
    # prior; skipping that step silently returns a normalised weight vector that
    # looks like a prior and is not one.
    weights: Array = np.linalg.solve(confusion, mu)
    source_prior = np.bincount(cal_labels, minlength=n_classes).astype(np.float64)
    source_prior /= source_prior.sum()
    prior = weights * source_prior
    if np.any(prior < -1e-8):
        raise ValueError(f"BBSE returned a negative class prior {prior}; label shift is violated")
    prior = np.clip(prior, 0.0, None)
    return prior / prior.sum()


def class_prior(labels: IntArray, n_classes: int) -> Array:
    """The empirical share of each class, as a vector that sums to one."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.float64)
    total = counts.sum()
    if total == 0.0:
        raise ValueError("cannot read a class prior off an empty label vector")
    return counts / total


def label_shift_weights(source_prior: Array, target_prior: Array) -> Array:
    """w(y) = q(y) / p(y), the weight each calibration point carries by its label.

    A class the source never saw cannot be reweighted into existence: its source
    share is zero and the ratio is undefined.  Raising is the right response,
    because the alternative -- an infinite or a silently zeroed weight -- would
    put the whole of the calibration mass on one class without saying so.
    """
    source_prior = np.asarray(source_prior, dtype=np.float64)
    target_prior = np.asarray(target_prior, dtype=np.float64)
    if source_prior.shape != target_prior.shape:
        raise ValueError(f"priors must match: {source_prior.shape} vs {target_prior.shape}")
    if np.any(source_prior <= 0.0):
        raise ValueError(f"a class absent from the source cannot be reweighted: {source_prior}")
    weights: Array = target_prior / source_prior
    return weights


def predict_sets_per_class(all_scores: Array, qhats: Array) -> NDArray[np.bool_]:
    """Membership matrix when the threshold differs by candidate class."""
    all_scores = np.asarray(all_scores, dtype=np.float64)
    qhats = np.asarray(qhats, dtype=np.float64)
    if all_scores.ndim != 2:
        raise ValueError(f"all_scores must be 2-D, got shape {all_scores.shape}")
    if qhats.shape != (all_scores.shape[1],):
        raise ValueError(f"need one quantile per class: {qhats.shape} vs {all_scores.shape[1]}")
    return all_scores <= qhats[None, :]
