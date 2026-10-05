"""Calibration and clinical utility of a probability, for one label at one site.

Coverage says whether a set holds the truth; this module says whether the
probability itself can be read as a risk, and what acting on it is worth.

``calibration_curve``    observed rate against mean prediction per bin, with a
                         Wilson interval on each observed rate.
``slope_intercept``      the logistic recalibration of Cox (1958): the slope is
                         the coefficient of logit(p) in a logistic model of
                         the outcome (1 when the spread is right); the
                         intercept is calibration-in-the-large, fitted with
                         logit(p) as a fixed offset (0 when the level is right).
``recalibration_coefficients``  intercept and slope of that same logistic
                         model fitted jointly, the pair that maps logit(p) to
                         a recalibrated logit.
``brier``                mean squared error of the probability.
``net_benefit``          Vickers & Elkin (2006): true positives per patient
                         minus false positives per patient weighted by the odds
                         of the threshold.
``ppv_at_prevalence``    Bayes' rule from sensitivity and specificity at a
                         stated prevalence, and ``false_alerts_per_detection``
                         the same quantity as a count a clinician feels.

Every function takes 0/1 outcomes and probabilities in [0, 1].
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metrics import wilson_interval

__all__ = [
    "brier",
    "calibration_curve",
    "calibration_in_the_large",
    "decision_curve",
    "false_alerts_per_detection",
    "net_benefit",
    "net_benefit_treat_all",
    "ppv_at_prevalence",
    "recalibrated",
    "recalibration_coefficients",
    "sensitivity_specificity",
    "slope_intercept",
]

Array = NDArray[np.float64]
Outcomes = NDArray[Any]  # 0/1, of any numeric dtype

# Probabilities are clipped this far from 0 and 1 before the logit, so a model
# that states certainty contributes a large finite value instead of infinity.
LOGIT_EPS = 1e-6
NEWTON_STEPS = 200
NEWTON_TOL = 1e-10


def _check(y: Outcomes, p: Array) -> tuple[Array, Array]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(f"outcomes {y.shape} and probabilities {p.shape} must be matching 1-D")
    if y.size == 0:
        raise ValueError("no records")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("outcomes must be 0 or 1")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return y, p


def _logit(p: Array) -> Array:
    q = np.clip(p, LOGIT_EPS, 1 - LOGIT_EPS)
    out: Array = np.log(q / (1 - q))
    return out


def brier(y: Outcomes, p: Array) -> float:
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2))


def _log_likelihood(design: Array, y: Array, offset: Array, beta: Array) -> float:
    eta = design @ beta + offset
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def _logistic_fit(design: Array, y: Array, offset: Array) -> Array:
    """Unpenalised maximum likelihood by damped Newton steps.

    A step is halved until the likelihood rises, so a start far from the optimum
    (an offset that already saturates every prediction) still converges.  The
    fit ends when the score equations hold, judged on the gradient: a step
    shrunk by the line search is not evidence of an optimum, and near one the
    likelihood moves by less than rounding, which a strict rise would refuse.
    Raises when it does not converge, which is what separable outcomes produce.
    """
    beta = np.zeros(design.shape[1])
    current = _log_likelihood(design, y, offset, beta)
    for _ in range(NEWTON_STEPS):
        mu = 0.5 * (1.0 + np.tanh((design @ beta + offset) / 2.0))
        gradient = design.T @ (y - mu)
        if np.max(np.abs(gradient)) < NEWTON_TOL * len(y):
            return beta
        hessian = design.T @ (design * np.maximum(mu * (1 - mu), 1e-12)[:, None])
        step = np.linalg.solve(hessian, gradient)
        slack = 1e-12 * (1.0 + abs(current))
        for _ in range(60):
            candidate = _log_likelihood(design, y, offset, beta + step)
            if candidate >= current - slack:
                break
            step = step / 2
        beta, current = beta + step, candidate
    raise ValueError("logistic recalibration did not converge (separable outcomes?)")


def calibration_in_the_large(y: Outcomes, p: Array) -> float:
    """The shift of logit(p) that makes the mean prediction match the observed rate."""
    y, p = _check(y, p)
    if y.min() == y.max():
        raise ValueError("calibration-in-the-large needs both outcomes present")
    z = _logit(p)
    return float(_logistic_fit(np.ones((len(z), 1)), y, z)[0])


def slope_intercept(y: Outcomes, p: Array) -> tuple[float, float]:
    """Calibration slope and calibration-in-the-large, in that order."""
    y, p = _check(y, p)
    if y.min() == y.max():
        raise ValueError("slope and intercept need both outcomes present")
    z = _logit(p)
    slope = _logistic_fit(np.column_stack([np.ones_like(z), z]), y, np.zeros_like(z))[1]
    return float(slope), calibration_in_the_large(y, p)


def recalibration_coefficients(y: Outcomes, p: Array) -> tuple[float, float]:
    """Intercept and slope of the joint logistic fit of the outcome on logit(p)."""
    y, p = _check(y, p)
    if y.min() == y.max():
        raise ValueError("recalibration needs both outcomes present")
    z = _logit(p)
    a, b = _logistic_fit(np.column_stack([np.ones_like(z), z]), y, np.zeros_like(z))
    return float(a), float(b)


def recalibrated(p: Array, intercept: float, slope: float) -> Array:
    """The probability expit(intercept + slope * logit(p))."""
    eta = intercept + slope * _logit(np.asarray(p, dtype=float))
    out: Array = 0.5 * (1.0 + np.tanh(eta / 2.0))  # expit, without overflow for large |eta|
    return out


def calibration_curve(y: Outcomes, p: Array, n_bins: int = 10) -> list[dict[str, float]]:
    """One row per bin of equal count: prediction range, mean prediction, observed rate."""
    y, p = _check(y, p)
    order = np.argsort(p, kind="stable")
    rows = []
    for part in np.array_split(order, min(n_bins, len(p))):
        if part.size == 0:
            continue
        events = int(y[part].sum())
        low, high = wilson_interval(events, int(part.size))
        rows.append(
            {
                "p_low": float(p[part].min()),
                "p_high": float(p[part].max()),
                "n": int(part.size),
                "mean_predicted": float(p[part].mean()),
                "observed": events / part.size,
                "observed_low": low,
                "observed_high": high,
            }
        )
    return rows


def sensitivity_specificity(y: Outcomes, flagged: NDArray[np.bool_]) -> tuple[float, float]:
    y = np.asarray(y, dtype=np.float64)
    flagged = np.asarray(flagged, dtype=bool)
    positives, negatives = y == 1, y == 0
    if not positives.any() or not negatives.any():
        raise ValueError("sensitivity and specificity need both outcomes present")
    return float(flagged[positives].mean()), float((~flagged[negatives]).mean())


def net_benefit(y: Outcomes, p: Array, threshold: float) -> float:
    """TP/n - FP/n * t/(1 - t), treating everyone whose probability reaches t."""
    y, p = _check(y, p)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between 0 and 1")
    treated = p >= threshold
    n = len(y)
    tp = float((treated & (y == 1)).sum())
    fp = float((treated & (y == 0)).sum())
    return tp / n - fp / n * threshold / (1.0 - threshold)


def net_benefit_treat_all(prevalence: float, threshold: float) -> float:
    return prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)


def decision_curve(y: Outcomes, p: Array, thresholds: Array) -> list[dict[str, float]]:
    """Net benefit of the model, of treating all and of treating none, per threshold."""
    y, p = _check(y, p)
    prevalence = float(y.mean())
    return [
        {
            "threshold": float(t),
            "model": net_benefit(y, p, float(t)),
            "treat_all": net_benefit_treat_all(prevalence, float(t)),
            "treat_none": 0.0,
        }
        for t in thresholds
    ]


def ppv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Share of flagged patients who are ill, at the stated prevalence."""
    true_flags = sensitivity * prevalence
    false_flags = (1.0 - specificity) * (1.0 - prevalence)
    if true_flags + false_flags == 0.0:
        raise ValueError("nobody is flagged; the positive predictive value is undefined")
    return true_flags / (true_flags + false_flags)


def false_alerts_per_detection(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Healthy patients flagged for each ill patient found."""
    if sensitivity * prevalence == 0.0:
        raise ValueError("no ill patient is found; the ratio is undefined")
    return (1.0 - specificity) * (1.0 - prevalence) / (sensitivity * prevalence)
