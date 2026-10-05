"""The positive predictive value a buyer recomputes, against the one a site observes.

The recipe under test: take the sensitivity and specificity a model reached
where it was validated, and the prevalence of the receiving site, and apply
Bayes' rule.  It is exact when sensitivity and specificity carry over to the
site unchanged, and wrong by however much they move.  Everything here works on
the four cells of a two-by-two table, so a bootstrap is a multinomial draw of
those cells rather than a resampling of records.

``Table``          true and false positives and negatives at one fixed threshold.
``gap_row``        the observed PPV with its Wilson interval, the recomputed
                   one, their difference in points with a bootstrap interval,
                   and how much of that difference the sensitivity and the
                   specificity each account for.
``label_free``     three predictions of the site's PPV that need no site
                   label: the recipe at a prevalence estimated without labels,
                   the model's own probability among the flagged, and the same
                   probability once corrected to that estimated prevalence.
``label_shift_check``  the mean likelihood ratio the source-calibrated
                   probability implies, among the ill and among the healthy, at
                   the source and at the site.  If only the prevalence moved,
                   each class keeps its source value, and the mean among the
                   healthy is 1 because the ratio is a density ratio
                   averaged over the healthy.  A site value far from the source
                   one says the ECGs of that class changed, which no prevalence
                   correction can undo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .calibration import ppv_at_prevalence, recalibrated, recalibration_coefficients
from .metrics import wilson_interval
from .repairs import prior_shift

__all__ = ["GAP_DRAWS", "Table", "gap_row", "label_free", "label_shift_check", "table_of"]

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]

GAP_DRAWS = 2000


@dataclass(frozen=True)
class Table:
    """Counts at one threshold: flagged ill, flagged healthy, missed ill, cleared healthy."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def prevalence(self) -> float:
        return (self.tp + self.fn) / self.n

    @property
    def sensitivity(self) -> float:
        return self.tp / (self.tp + self.fn)

    @property
    def specificity(self) -> float:
        return self.tn / (self.tn + self.fp)

    @property
    def ppv(self) -> float:
        return self.tp / (self.tp + self.fp)


def table_of(y: IntArray, flagged: BoolArray) -> Table:
    y = np.asarray(y, dtype=int)
    flagged = np.asarray(flagged, dtype=bool)
    if y.shape != flagged.shape:
        raise ValueError(f"outcomes {y.shape} and flags {flagged.shape} differ in shape")
    ill = y == 1
    return Table(
        tp=int((flagged & ill).sum()),
        fp=int((flagged & ~ill).sum()),
        fn=int((~flagged & ill).sum()),
        tn=int((~flagged & ~ill).sum()),
    )


def _cells(t: Table) -> IntArray:
    return np.array([t.tp, t.fp, t.fn, t.tn])


def _ppv_vec(sens: Array, spec: Array, prev: Array) -> Array:
    true_flags = sens * prev
    false_flags = (1.0 - spec) * (1.0 - prev)
    with np.errstate(invalid="ignore", divide="ignore"):
        out: Array = true_flags / (true_flags + false_flags)
    return out


def _gap_draws(source: Table, target: Table, draws: int, seed: int) -> Array:
    """Recomputed minus observed PPV over multinomial redraws of both tables.

    A draw that leaves a table without ill patients, without healthy ones or
    without a flag has no gap; it is dropped, and the caller is told how many.
    """
    rng = np.random.default_rng(seed)
    s = rng.multinomial(source.n, _cells(source) / source.n, size=draws).astype(np.float64)
    t = rng.multinomial(target.n, _cells(target) / target.n, size=draws).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = s[:, 0] / (s[:, 0] + s[:, 2])
        spec = s[:, 3] / (s[:, 3] + s[:, 1])
        prev = (t[:, 0] + t[:, 2]) / target.n
        observed = t[:, 0] / (t[:, 0] + t[:, 1])
    gap: Array = _ppv_vec(sens, spec, prev) - observed
    return gap[np.isfinite(gap)]


def gap_row(source: Table, target: Table, draws: int = GAP_DRAWS, seed: int = 0) -> dict[str, Any]:
    """The recipe against the observation, at one threshold fitted on the source.

    The recipe is given the target's true prevalence, which a buyer would only
    estimate: any gap left is owed to sensitivity and specificity alone.  The
    split of the gap holds one of the two at its source value and moves the
    other to its target value; the two parts need not sum to the gap.
    """
    row: dict[str, Any] = {
        "n_source": source.n,
        "n_target": target.n,
        "n_target_ill": target.tp + target.fn,
        "n_flagged": target.tp + target.fp,
        "n_flagged_ill": target.tp,
        "prevalence_source": source.prevalence,
        "prevalence_target": target.prevalence,
    }
    defined = min(source.tp + source.fn, source.fp + source.tn, target.tp + target.fn) > 0
    if not defined or target.tp + target.fp == 0:
        return row | {"defined": False}
    sens_s, spec_s = source.sensitivity, source.specificity
    sens_t, spec_t = target.sensitivity, target.specificity
    prev = target.prevalence
    observed = target.ppv
    low, high = wilson_interval(target.tp, target.tp + target.fp)
    recomputed = ppv_at_prevalence(sens_s, spec_s, prev)
    draws_kept = _gap_draws(source, target, draws, seed)
    return row | {
        "defined": True,
        "sens_source": sens_s,
        "spec_source": spec_s,
        "sens_target": sens_t,
        "spec_target": spec_t,
        "ppv_observed": observed,
        "ppv_observed_low": low,
        "ppv_observed_high": high,
        "ppv_recomputed": recomputed,
        "gap": recomputed - observed,
        "gap_low": float(np.percentile(draws_kept, 2.5)),
        "gap_high": float(np.percentile(draws_kept, 97.5)),
        "gap_draws_kept": int(draws_kept.size),
        "recomputed_inside_interval": bool(low <= recomputed <= high),
        "gap_from_sensitivity": ppv_at_prevalence(sens_t, spec_s, prev) - recomputed,
        "gap_from_specificity": ppv_at_prevalence(sens_s, spec_t, prev) - recomputed,
        "false_alerts_observed": target.fp / target.tp if target.tp else None,
        "false_alerts_recomputed": (1 - spec_s) * (1 - prev) / (sens_s * prev),
    }


def label_free(
    p_cal: Array, y_cal: IntArray, p_target: Array, flagged: BoolArray, sens: float, spec: float
) -> dict[str, float | None]:
    """Three predictions of the site's PPV made from its unlabelled ECGs.

    ``p_cal`` and ``y_cal`` are the source's, which the prevalence correction
    recalibrates on; ``sens`` and ``spec`` are the source's at the threshold
    behind ``flagged``.
    """
    p_target = np.asarray(p_target, dtype=np.float64)
    flagged = np.asarray(flagged, dtype=bool)
    if not flagged.any():
        return dict.fromkeys(
            (
                "prevalence_estimated",
                "ppv_recipe_estimated_prevalence",
                "ppv_mean_probability",
                "ppv_mean_probability_prior_corrected",
            )
        )
    prior, corrected = prior_shift(p_cal, y_cal, p_target)
    recipe = ppv_at_prevalence(sens, spec, prior) if sens * prior + (1 - spec) > 0 else None
    return {
        "prevalence_estimated": prior,
        "ppv_recipe_estimated_prevalence": recipe,
        "ppv_mean_probability": float(p_target[flagged].mean()),
        "ppv_mean_probability_prior_corrected": float(corrected[flagged].mean()),
    }


def label_shift_check(
    p_cal: Array, y_cal: IntArray, p_target: Array, y_target: IntArray
) -> dict[str, float | None]:
    """Mean likelihood ratio per class, source against site, on source-calibrated probabilities."""
    y_cal, y_target = np.asarray(y_cal, dtype=int), np.asarray(y_target, dtype=int)
    prior = float(y_cal.mean())
    a, b = recalibration_coefficients(y_cal, p_cal)

    def ratios(p: Array) -> Array:
        q = np.clip(recalibrated(p, a, b), 1e-6, 1 - 1e-6)
        out: Array = (q / prior) / ((1 - q) / (1 - prior))
        return out

    source, site = ratios(p_cal), ratios(p_target)

    def mean(r: Array, mask: NDArray[np.bool_]) -> float | None:
        return float(r[mask].mean()) if mask.any() else None

    return {
        "likelihood_ratio_healthy_source": mean(source, y_cal == 0),
        "likelihood_ratio_healthy_target": mean(site, y_target == 0),
        "likelihood_ratio_ill_source": mean(source, y_cal == 1),
        "likelihood_ratio_ill_target": mean(site, y_target == 1),
    }
