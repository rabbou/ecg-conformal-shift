"""Three repairs a receiving site can apply to a model's scores, judged on net benefit.

``prior``        no site label.  The probabilities are first recalibrated on the
                 source's own labels, so that they are posteriors at the source
                 prevalence; the site's prevalence is then the maximum-likelihood
                 estimate that Saerens, Latinne & Decaestecker's EM computes
                 (Neural Computation 2002) from the site's unlabelled
                 probabilities, and every probability is moved to it by Bayes'
                 rule.  The source step is what Alexandari, Kundaje & Shrikumar
                 (ICML 2020) found EM needs; without it the prior the
                 probabilities carry is the training set's, not the source's.
                 Exact when only the prevalence changed.
``recalibrated`` n site labels.  A logistic model of the outcome on logit(p),
                 intercept and slope, fitted on n labelled site ECGs; intercept
                 only when the draw holds fewer than ``MIN_EVENTS`` of the rarer
                 class, as Steyerberg's updating hierarchy advises for small
                 samples; left as delivered when it holds none.
``abstention``   no site label.  Per-label conformal sets fitted on the source:
                 a set {1} refers, {0} clears, and {0, 1} goes to a human
                 reader.  What the reader decides is not modelled, so the rule
                 has two net benefits, one with every abstention cleared and
                 one with every abstention referred.  When no set is empty the
                 second is the plain threshold at 90% source sensitivity, the
                 one the positive predictive value gap is measured at.

Each repair is read on one half of the site, cut by patient; the labels the
recalibration spends come from the other half only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .calibration import (
    brier,
    calibration_in_the_large,
    recalibrated,
    recalibration_coefficients,
    slope_intercept,
)
from .splits import patient_split
from .transfer import ALPHA, conformal_sets

__all__ = [
    "LOCAL_LABELS",
    "THRESHOLDS",
    "abstention_rules",
    "adjust_to_prior",
    "estimate_prior",
    "prior_shift",
    "local_recalibration",
    "net_benefit_of",
    "repair_cell",
]

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]

THRESHOLDS = (0.05, 0.10, 0.20)
LOCAL_LABELS = 100
DRAWS = 200
BOOTSTRAP = 500
MIN_EVENTS = 10
BISECTION_STEPS = 60
PRIOR_FLOOR = 1e-6


def adjust_to_prior(p: Array, prior_from: float, prior_to: float) -> Array:
    """Bayes' rule for a change of prevalence: the posterior odds times the prior odds ratio."""
    p = np.asarray(p, dtype=np.float64)
    up = prior_to / prior_from
    down = (1.0 - prior_to) / (1.0 - prior_from)
    out: Array = up * p / (up * p + down * (1.0 - p))
    return out


def estimate_prior(p: Array, prior_source: float) -> float:
    """The site prevalence that maximises the likelihood of the site's probabilities.

    Saerens et al. reach it by EM, which crawls when the answer is near 0 or 1.
    The log-likelihood sum(log(w p + v (1 - p))), with w and v the two prior
    ratios, is concave in the site prevalence, so its derivative falls
    monotonically and bisection finds the same maximum, or the boundary when the
    derivative keeps one sign over [0, 1].  At the root the prevalence equals the
    mean of the probabilities moved to it, EM's fixed point.
    """
    if not 0.0 < prior_source < 1.0:
        raise ValueError("the source prevalence must lie strictly between 0 and 1")
    p = np.clip(np.asarray(p, dtype=np.float64), PRIOR_FLOOR, 1.0 - PRIOR_FLOOR)
    a, b = p / prior_source, (1.0 - p) / (1.0 - prior_source)

    def slope(q: float) -> float:
        return float(np.sum((a - b) / (q * a + (1.0 - q) * b)))

    if slope(0.0) <= 0.0:
        return 0.0
    if slope(1.0) >= 0.0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(BISECTION_STEPS):
        mid = (low + high) / 2
        low, high = (mid, high) if slope(mid) > 0.0 else (low, mid)
    return (low + high) / 2


def prior_shift(p_cal: Array, y_cal: IntArray, p_tgt: Array) -> tuple[float, Array]:
    """The site prevalence estimated without site labels, and the site probabilities moved to it."""
    y_cal = np.asarray(y_cal, dtype=int)
    prior_source = float(y_cal.mean())
    a, b = recalibration_coefficients(y_cal, p_cal)
    at_source = recalibrated(p_tgt, a, b)
    prior = estimate_prior(at_source, prior_source)
    return prior, adjust_to_prior(at_source, prior_source, prior)


def local_recalibration(p: Array, y: IntArray) -> tuple[float, float, str]:
    """Intercept, slope and which fit was possible on these labelled records."""
    y = np.asarray(y, dtype=int)
    rarer = min(int(y.sum()), int((1 - y).sum()))
    if rarer == 0:
        return 0.0, 1.0, "none"
    if rarer >= MIN_EVENTS:
        try:
            a, b = recalibration_coefficients(y, p)
            return a, b, "intercept_and_slope"
        except ValueError:
            pass
    return calibration_in_the_large(y, p), 1.0, "intercept"


def net_benefit_of(y: IntArray, treated: BoolArray, threshold: float) -> float:
    """True positives per patient minus false positives per patient at the threshold's odds."""
    y = np.asarray(y, dtype=int)
    n = len(y)
    tp = float((treated & (y == 1)).sum())
    fp = float((treated & (y == 0)).sum())
    return tp / n - fp / n * threshold / (1.0 - threshold)


def _nb_draws(y: IntArray, treated: BoolArray, threshold: float, index: IntArray) -> Array:
    """Net benefit on each bootstrap redraw of the records, one row of ``index`` each."""
    yy, tt = y[index], treated[index]
    tp = (tt & (yy == 1)).mean(axis=1)
    fp = (tt & (yy == 0)).mean(axis=1)
    out: Array = tp - fp * threshold / (1.0 - threshold)
    return out


def _calibration(y: IntArray, p: Array) -> dict[str, float | None]:
    if y.min() == y.max():
        return {"brier": brier(y, p), "slope": None, "intercept": None}
    try:
        slope, intercept = slope_intercept(y, p)
    except ValueError:
        return {"brier": brier(y, p), "slope": None, "intercept": None}
    return {"brier": brier(y, p), "slope": slope, "intercept": intercept}


def abstention_rules(p_cal: Array, y_cal: IntArray, p: Array) -> tuple[BoolArray, BoolArray]:
    """Who the per-label sets refer outright, and who they leave to a human.

    A set {1} refers.  A set {0, 1} and an empty set both abstain: the first
    rules nothing out, the second rules out both answers, and neither is a
    decision a clinician can act on.
    """
    sets = conformal_sets(p_cal, y_cal, p, ALPHA)["perlabel"]
    rule_in = sets[:, 1] & ~sets[:, 0]
    return rule_in, sets.sum(axis=1) != 1


def _halves(patients: pd.Series, seed: int) -> tuple[IntArray, IntArray]:
    part = patient_split(patients, {"pool": 0.5, "eval": 0.5}, seed).to_numpy()
    return np.flatnonzero(part == "pool"), np.flatnonzero(part == "eval")


def _recalibration_draws(
    p_pool: Array,
    y_pool: IntArray,
    p_eval: Array,
    y_eval: IntArray,
    n_labels: int,
    draws: int,
    seed: int,
) -> tuple[list[Array], dict[str, Any]]:
    """The evaluation half's probabilities under each draw's recalibration, and
    the calibration those probabilities reach, averaged over the draws."""
    rng = np.random.default_rng(seed)
    probs, kinds = [], {"intercept_and_slope": 0, "intercept": 0, "none": 0}
    briers, intercepts = [], []
    both = 0 < y_eval.sum() < len(y_eval)
    for _ in range(draws):
        drawn = rng.choice(len(p_pool), size=min(n_labels, len(p_pool)), replace=False)
        a, b, kind = local_recalibration(p_pool[drawn], y_pool[drawn])
        kinds[kind] += 1
        q = recalibrated(p_eval, a, b)
        probs.append(q)
        briers.append(brier(y_eval, q))
        if both:
            intercepts.append(calibration_in_the_large(y_eval, q))
    calibration = {
        "brier": float(np.mean(briers)),
        "intercept": float(np.mean(intercepts)) if intercepts else None,
        "abs_intercept": float(np.mean(np.abs(intercepts))) if intercepts else None,
    }
    return probs, {"fits": kinds, "calibration": calibration}


def repair_cell(
    p_cal: Array,
    y_cal: IntArray,
    p_tgt: Array,
    y_tgt: IntArray,
    patients: pd.Series,
    thresholds: tuple[float, ...] = THRESHOLDS,
    curve: Array | None = None,
    n_labels: int = LOCAL_LABELS,
    draws: int = DRAWS,
    seed: int = 0,
) -> dict[str, Any]:
    """Every repair on one source-target pair, read on the target's evaluation half.

    ``curve``, when given, adds the net benefit of each rule at every threshold
    of that grid, for a decision-curve figure.
    """
    p_cal, p_tgt = np.asarray(p_cal, dtype=np.float64), np.asarray(p_tgt, dtype=np.float64)
    y_cal, y_tgt = np.asarray(y_cal, dtype=int), np.asarray(y_tgt, dtype=int)
    pool, held = _halves(patients, seed)
    p_eval, y_eval = p_tgt[held], y_tgt[held]
    prior_source = float(y_cal.mean())
    prior, prior_corrected = prior_shift(p_cal, y_cal, p_eval)
    as_delivered = p_eval
    local, fitted = _recalibration_draws(
        p_tgt[pool], y_tgt[pool], p_eval, y_eval, n_labels, draws, seed
    )
    rule_in, abstained = abstention_rules(p_cal, y_cal, p_eval)
    not_ruled_out = rule_in | abstained
    oracle = (
        recalibrated(p_eval, *recalibration_coefficients(y_eval, p_eval))
        if 0 < y_eval.sum() < len(y_eval)
        else p_eval
    )
    index = np.random.default_rng(seed + 1).integers(0, len(held), size=(BOOTSTRAP, len(held)))

    def at(t: float) -> dict[str, Any]:
        fixed = {
            "as_delivered": as_delivered >= t,
            "prior": prior_corrected >= t,
            "abstention_cleared": rule_in,
            "abstention_referred": not_ruled_out,
            "treat_all": np.ones(len(held), dtype=bool),
            "every_local_label": oracle >= t,
        }
        row: dict[str, Any] = {"threshold": t, "treat_none": 0.0}
        base = _nb_draws(y_eval, fixed["as_delivered"], t, index)
        # 95% bootstrap interval of each rule's net benefit minus the model's.
        versus: dict[str, list[float]] = {}
        for name, treated in fixed.items():
            row[name] = net_benefit_of(y_eval, treated, t)
            if name not in ("as_delivered", "every_local_label"):
                diff = _nb_draws(y_eval, treated, t, index) - base
                versus[name] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        row["minus_as_delivered"] = versus
        local_nb = np.array([net_benefit_of(y_eval, q >= t, t) for q in local])
        row["recalibrated"] = float(local_nb.mean())
        row["recalibrated_p10"] = float(np.percentile(local_nb, 10))
        row["recalibrated_p90"] = float(np.percentile(local_nb, 90))
        return row

    out: dict[str, Any] = {
        "n_pool": int(len(pool)),
        "n_eval": int(len(held)),
        "n_eval_ill": int(y_eval.sum()),
        "prevalence_source": prior_source,
        "prevalence_eval": float(y_eval.mean()),
        "prevalence_estimated": prior,
        "local_labels": n_labels,
        "draws": draws,
        "recalibration_fits": fitted["fits"],
        "abstained": float(abstained.mean()),
        "net_benefit": [at(t) for t in thresholds],
        "calibration": {
            "as_delivered": _calibration(y_eval, as_delivered),
            "prior": _calibration(y_eval, prior_corrected),
            "recalibrated": fitted["calibration"],
        },
    }
    if curve is not None:
        out["curve"] = [
            {
                "threshold": float(t),
                "as_delivered": net_benefit_of(y_eval, as_delivered >= t, float(t)),
                "prior": net_benefit_of(y_eval, prior_corrected >= t, float(t)),
                "recalibrated": float(
                    np.mean([net_benefit_of(y_eval, q >= t, float(t)) for q in local])
                ),
                "abstention_cleared": net_benefit_of(y_eval, rule_in, float(t)),
                "abstention_referred": net_benefit_of(y_eval, not_ruled_out, float(t)),
                "treat_all": net_benefit_of(y_eval, np.ones(len(held), dtype=bool), float(t)),
                "treat_none": 0.0,
            }
            for t in curve
        ]
    return out
