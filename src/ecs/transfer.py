"""One model, one source, one target: the measurements a transfer report prints.

Every threshold here is fitted on the source calibration records and spent on
the target without refitting, except on the ladder, where it is refitted on a
stated number of target records drawn from a pool the evaluation half never
sees.  Three ways of turning a probability into a decision are compared, with
the names the study already uses:

``plain``     one threshold on the probability, placed so that 90% of the
              calibration positives reach it; a committed yes or no.  It is the
              per-label rule's threshold for the positive class.
``pooled``    split conformal with the LAC score over all calibration records:
              the set {0}, {1}, {0, 1} or the empty set.
``perlabel``  the same with one quantile per true class (Mondrian), which holds
              coverage inside each class whatever the prevalence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from .calibration import (
    brier,
    calibration_curve,
    calibration_in_the_large,
    decision_curve,
    false_alerts_per_detection,
    ppv_at_prevalence,
    sensitivity_specificity,
    slope_intercept,
)
from .conformal import SCHEMES, fit_thresholds
from .metrics import bootstrap_ci, wilson_interval
from .splits import patient_split

__all__ = [
    "AGE_BANDS",
    "METHODS",
    "auroc_row",
    "calibration_row",
    "conformal_sets",
    "coverage_row",
    "decision_rows",
    "ladder_rows",
    "outcomes_from_coverage",
    "ppv_row",
    "subgroup_rows",
]

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]

ALPHA = 0.10
METHODS = SCHEMES
AGE_BANDS = ((18, 50, "18-49"), (50, 65, "50-64"), (65, 80, "65-79"), (80, 200, "80+"))
BOOTSTRAP_DRAWS = 500
NET_BENEFIT_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
LADDER_RUNGS = (0, 25, 50, 100, 200, 400)
LADDER_DRAWS = 200
POOL_SHARE = 0.5


def _two_columns(p: Array) -> Array:
    p = np.asarray(p, dtype=np.float64)
    return np.column_stack([1.0 - p, p])


def conformal_sets(
    p_cal: Array, y_cal: IntArray, p_tgt: Array, alpha: float = ALPHA
) -> dict[str, BoolArray]:
    """The (n, 2) membership matrix of each method, fitted on calibration only."""
    return fit_thresholds(_two_columns(p_cal), y_cal, alpha).sets(_two_columns(p_tgt))


def _share(mask: BoolArray) -> tuple[float | None, float | None, float | None, int]:
    n = int(mask.size)
    if n == 0:
        return None, None, None, 0
    k = int(mask.sum())
    low, high = wilson_interval(k, n)
    return k / n, low, high, n


def coverage_row(sets: BoolArray, y: IntArray) -> dict[str, Any]:
    """Coverage overall and inside each class, and the share sent to a human."""
    y = np.asarray(y, dtype=int)
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(axis=1)
    pos, pos_low, pos_high, n_pos = _share(covered[y == 1])
    neg, neg_low, neg_high, n_neg = _share(covered[y == 0])
    return {
        "n": len(y),
        "n_pos": n_pos,
        "coverage": float(covered.mean()),
        "coverage_pos": pos,
        "coverage_pos_low": pos_low,
        "coverage_pos_high": pos_high,
        "coverage_neg": neg,
        "coverage_neg_low": neg_low,
        "coverage_neg_high": neg_high,
        "abstention": float((size != 1).mean()),
        "both": float((size == 2).mean()),
        "empty": float((size == 0).mean()),
    }


def outcomes_from_coverage(plain: dict[str, Any], perlabel: dict[str, Any]) -> dict[str, Any]:
    """What each patient gets under the per-label rule, read off two coverage rows.

    The plain rule flags a patient exactly when the per-label rule admits the
    disease, since the two share that threshold.  With no empty set, a patient
    is then recognised (the disease alone), referred to a human (both labels) or
    missed (no disease), and the counts follow from the two rows of one cell:

      healthy cleared   = plain coverage of the healthy
      healthy referred  = per-label coverage of the healthy - plain coverage of the healthy
      ill referred      = every referral - the healthy referred
      ill recognised    = coverage of the ill - the ill referred

    Raises when a premise fails: a row with an empty set, two rows that disagree
    on the ill, or a share that is not a whole number of patients.
    """
    if plain["n"] != perlabel["n"] or plain["n_pos"] != perlabel["n_pos"]:
        raise ValueError("the two rows describe different cohorts")
    if perlabel["empty"] != 0 or plain["abstention"] != 0:
        raise ValueError("an empty set or a deferral under the plain rule breaks the identities")
    n, n_ill = int(perlabel["n"]), int(perlabel["n_pos"])
    n_healthy = n - n_ill

    def whole(share: float, of: int) -> int:
        count = share * of
        if abs(count - round(count)) > 1e-6:
            raise ValueError(f"{share} of {of} is not a whole number of patients")
        return int(round(count))

    covered = whole(perlabel["coverage_pos"], n_ill)
    if covered != whole(plain["coverage_pos"], n_ill):
        raise ValueError("the plain and per-label rules disagree on the ill")
    cleared = whole(plain["coverage_neg"], n_healthy)
    healthy_referred = whole(perlabel["coverage_neg"], n_healthy) - cleared
    ill_referred = whole(perlabel["both"], n) - healthy_referred
    counts = {
        "ill": {
            "recognised": covered - ill_referred,
            "referred": ill_referred,
            "missed": n_ill - covered,
        },
        "healthy": {
            "cleared": cleared,
            "referred": healthy_referred,
            "false_alarm": n_healthy - cleared - healthy_referred,
        },
    }
    if min(min(group.values()) for group in counts.values()) < 0:
        raise ValueError(f"a negative count: {counts}")
    totals = {"ill": n_ill, "healthy": n_healthy}
    return {
        group: {
            "n": totals[group],
            **{name: {"count": c, "share": c / totals[group]} for name, c in parts.items()},
        }
        for group, parts in counts.items()
    }


def _age_band(age: float) -> str:
    for low, high, name in AGE_BANDS:
        if low <= age < high:
            return name
    return "unknown"


def subgroup_rows(sets: BoolArray, y: IntArray, meta: pd.DataFrame) -> list[dict[str, Any]]:
    """Coverage of the ill inside each sex, age band and recorded ethnicity."""
    y = np.asarray(y, dtype=int)
    covered = sets[np.arange(len(y)), y]
    groups = {
        "sex": meta["sex"].astype(str).to_numpy(),
        "age": np.array([_age_band(a) for a in meta["age_at_ecg"]]),
        "race_ethnicity": meta["race_ethnicity"].astype(str).to_numpy(),
    }
    rows = []
    for kind, values in groups.items():
        for group in sorted(set(values)):
            mask = (values == group) & (y == 1)
            share, low, high, n = _share(covered[mask])
            rows.append(
                {
                    "kind": kind,
                    "group": group,
                    "n_pos": n,
                    "coverage_pos": share,
                    "low": low,
                    "high": high,
                }
            )
    return rows


def auroc_row(y: IntArray, p: Array, seed: int = 0) -> dict[str, float | None]:
    """AUROC with its bootstrap interval; the interval is None when the sample
    holds too few positives for draws to keep both classes."""
    y = np.asarray(y, dtype=int)
    if y.min() == y.max():
        return {"auroc": None, "low": None, "high": None}

    def statistic(labels: IntArray, scores: Array) -> float:
        return float(roc_auc_score(labels, scores))

    try:
        point, low, high = bootstrap_ci(statistic, y, p, n_draws=BOOTSTRAP_DRAWS, seed=seed)
    except ValueError:
        return {"auroc": statistic(y, p), "low": None, "high": None}
    return {"auroc": point, "low": low, "high": high}


def calibration_row(y: IntArray, p: Array) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    row: dict[str, Any] = {
        "prevalence": float(y.mean()),
        "mean_predicted": float(np.mean(p)),
        "brier": brier(y, p),
        "slope": None,
        "intercept": None,
    }
    if 0 < y.sum() < len(y):
        row["slope"], row["intercept"] = slope_intercept(y, p)
    return row


def decision_rows(y: IntArray, p: Array) -> list[dict[str, float]]:
    return decision_curve(y, p, np.array(NET_BENEFIT_THRESHOLDS))


def curve_rows(y: IntArray, p: Array) -> list[dict[str, float]]:
    return calibration_curve(y, p)


def ppv_row(
    source_flags: BoolArray, y_source: IntArray, target_flags: BoolArray, y_target: IntArray
) -> dict[str, float | None]:
    """At the plain threshold: the PPV observed on the target, and the one a buyer
    would compute from the source's sensitivity and specificity at the target's
    prevalence, with the false alerts each implies per ill patient found."""
    y_target = np.asarray(y_target, dtype=int)
    prevalence = float(y_target.mean())
    out: dict[str, float | None] = {"prevalence": prevalence}
    try:
        src_sens, src_spec = sensitivity_specificity(y_source, source_flags)
        tgt_sens, tgt_spec = sensitivity_specificity(y_target, target_flags)
    except ValueError:
        return out | {
            k: None
            for k in (
                "sens_source",
                "spec_source",
                "sens_target",
                "spec_target",
                "ppv_observed",
                "ppv_from_source",
                "false_alerts_observed",
                "false_alerts_from_source",
            )
        }
    flagged = int(np.sum(target_flags))
    out |= {
        "sens_source": src_sens,
        "spec_source": src_spec,
        "sens_target": tgt_sens,
        "spec_target": tgt_spec,
        "ppv_observed": float(y_target[target_flags].mean()) if flagged else None,
        "ppv_from_source": ppv_at_prevalence(src_sens, src_spec, prevalence),
        "false_alerts_observed": (
            false_alerts_per_detection(tgt_sens, tgt_spec, prevalence) if tgt_sens else None
        ),
        "false_alerts_from_source": false_alerts_per_detection(src_sens, src_spec, prevalence),
    }
    return out


def _intercept_shift(y: IntArray, p: Array) -> float:
    """The calibration-in-the-large a sample implies; zero when it holds one class."""
    if y.min() == y.max():
        return 0.0
    return calibration_in_the_large(y, p)


def _shifted(p: Array, shift: float) -> Array:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    out: Array = 1.0 / (1.0 + np.exp(-(np.log(q / (1 - q)) + shift)))
    return out


def ladder_rows(
    p_cal: Array,
    y_cal: IntArray,
    p_tgt: Array,
    y_tgt: IntArray,
    patients: pd.Series,
    seed: int = 0,
    rungs: tuple[int, ...] = LADDER_RUNGS,
    draws: int = LADDER_DRAWS,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """How many target labels buy the coverage of the ill back, and the calibration level.

    The target is cut once, by patient, into a pool and an evaluation half.  Rung
    zero spends the source thresholds; rung n refits the per-label thresholds and
    an intercept shift on n records drawn from the pool, and every rung is read
    on the same evaluation half.  Coverage of the ill is split, draw by draw, into
    the recognised (the disease label alone) and the referred (both labels), and
    the ill among the records a rung fits on are counted in every draw.
    """
    y_tgt = np.asarray(y_tgt, dtype=int)
    part = patient_split(patients, {"pool": POOL_SHARE, "eval": 1 - POOL_SHARE}, seed).to_numpy()
    pool, held = np.flatnonzero(part == "pool"), np.flatnonzero(part == "eval")
    y_eval, p_eval = y_tgt[held], p_tgt[held]
    rng = np.random.default_rng(seed) if rng is None else rng
    rows = []
    for rung in rungs:
        if rung > len(pool):
            break
        pos, neg, citl, infinite = [], [], [], 0
        recognised, referred, referred_all, fit_ill = [], [], [], []
        for _ in range(draws if rung else 1):
            if rung:
                drawn = rng.choice(pool, size=rung, replace=False)
                fit_p, fit_y = p_tgt[drawn], y_tgt[drawn]
            else:
                fit_p, fit_y = p_cal, np.asarray(y_cal, dtype=int)
            sets = conformal_sets(fit_p, fit_y, p_eval)["perlabel"]
            covered = sets[np.arange(len(y_eval)), y_eval]
            pos.append(covered[y_eval == 1].mean())
            ill = sets[y_eval == 1]
            recognised.append((ill[:, 1] & ~ill[:, 0]).mean())
            referred.append(ill.all(axis=1).mean())
            referred_all.append(sets.all(axis=1).mean())
            fit_ill.append(int(fit_y.sum()))
            neg.append(covered[y_eval == 0].mean())
            infinite += int(sets[:, 1].all())
            shift = _intercept_shift(fit_y, fit_p) if rung else 0.0
            citl.append(abs(_intercept_shift(y_eval, _shifted(p_eval, shift))))
        rows.append(
            {
                "labels": rung,
                "draws": draws if rung else 1,
                "coverage_pos_mean": float(np.mean(pos)),
                "coverage_pos_p10": float(np.percentile(pos, 10)),
                "coverage_pos_p90": float(np.percentile(pos, 90)),
                "coverage_neg_mean": float(np.mean(neg)),
                "recognised_pos_mean": float(np.mean(recognised)),
                "referred_pos_mean": float(np.mean(referred)),
                "referred_all_mean": float(np.mean(referred_all)),
                "fit_ill_mean": float(np.mean(fit_ill)),
                "fit_ill_min": min(fit_ill),
                "fit_ill_max": max(fit_ill),
                "share_all_flagged": infinite / (draws if rung else 1),
                "abs_intercept_mean": float(np.mean(citl)),
                "n_eval": len(held),
                "n_eval_pos": int(y_eval.sum()),
            }
        )
    return rows
