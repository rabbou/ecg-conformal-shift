"""What a fixed threshold does to patients, counted the way a clinic counts them.

Each function takes the decision sets a threshold produced and the true labels,
and returns shares per 100 ill and per 100 healthy patients: given the right
answer alone, sent to a human reader, or given the wrong answer alone.  The
plain threshold sends nobody to a reader, so its ill split into caught and
missed; the per-label pair adds a second threshold, and the patients between
the two go to a reader.

The functions are pure; ``scripts/echonext_clinical.py`` reads the stored
EchoNext scores and writes ``results/echonext_clinical.json`` from them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.linear_model import LogisticRegression

from .metrics import wilson_interval
from .transfer import conformal_sets

__all__ = [
    "holm",
    "outcome_shares",
    "paired_difference",
    "per_thousand",
    "predictive_values",
    "rerandomised_ladder",
    "standardised_coverage",
    "subgroup_tests",
    "two_halves",
]

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]


def _share(k: int, n: int) -> dict[str, float | int | None]:
    if n == 0:
        return {"share": None, "low": None, "high": None, "count": 0, "n": 0}
    low, high = wilson_interval(k, n)
    return {"share": k / n, "low": low, "high": high, "count": k, "n": n}


def outcome_shares(sets: BoolArray, y: IntArray) -> dict[str, dict[str, Any]]:
    """Per class, the share given the right label alone, deferred, and given the wrong one alone.

    A set holding both labels, or neither, is a deferral: no machine answer.
    """
    y = np.asarray(y, dtype=int)
    size = sets.sum(axis=1)
    alone_ill = (size == 1) & sets[:, 1]
    alone_healthy = (size == 1) & sets[:, 0]
    deferred = size != 1
    out: dict[str, dict[str, Any]] = {}
    for name, cls, right, wrong in (
        ("ill", 1, alone_ill, alone_healthy),
        ("healthy", 0, alone_healthy, alone_ill),
    ):
        mask = y == cls
        n = int(mask.sum())
        out[name] = {
            "right_alone": _share(int((right & mask).sum()), n),
            "deferred": _share(int((deferred & mask).sum()), n),
            "wrong_alone": _share(int((wrong & mask).sum()), n),
        }
    return out


def predictive_values(sens: float, spec: float, prevalence: float) -> dict[str, float]:
    """PPV and NPV of a yes-or-no test at a prevalence, by Bayes' rule."""
    flagged = sens * prevalence + (1 - spec) * (1 - prevalence)
    cleared = (1 - sens) * prevalence + spec * (1 - prevalence)
    return {
        "ppv": sens * prevalence / flagged if flagged else math.nan,
        "npv": spec * (1 - prevalence) / cleared if cleared else math.nan,
    }


def per_thousand(sens: float, spec: float, prevalence: float) -> dict[str, float]:
    """Per 1,000 patients: those flagged (each one an echocardiogram), the ill
    found, the ill missed, and the healthy flagged."""
    ill = 1000 * prevalence
    healthy = 1000 - ill
    return {
        "flagged": sens * ill + (1 - spec) * healthy,
        "ill_found": sens * ill,
        "ill_missed": (1 - sens) * ill,
        "healthy_flagged": (1 - spec) * healthy,
    }


def paired_difference(
    a: BoolArray,
    b: BoolArray,
    n_draws: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean of ``a`` minus mean of ``b`` on the same patients, with a 95% percentile
    interval from resampling the patients, both arms read on each resample."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("the two arms must be read on the same patients")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_draws, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return {
        "difference": float(a.mean() - b.mean()),
        "low": float(np.percentile(diffs, 2.5)),
        "high": float(np.percentile(diffs, 97.5)),
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def two_halves(order: IntArray) -> tuple[IntArray, IntArray]:
    """A shuffled cohort cut into the half labelled samples are drawn from and the
    half every figure is read on; no patient is in both."""
    middle = len(order) // 2
    return order[:middle], order[middle:]


def rerandomised_ladder(
    p_cal: Array,
    y_cal: IntArray,
    p_tgt: Array,
    y_tgt: IntArray,
    rungs: Sequence[int],
    draws: int,
    *,
    alpha: float,
    level: float,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Refit the thresholds on n labelled target patients, read on the others.

    In every draw the target is cut anew into two halves, one patient per row:
    the labelled sample is drawn from the first half and every figure is read on
    the second.  Rung zero spends the source thresholds on the same halves.  Each
    draw records how many ill patients the labelled sample held.
    """
    y_tgt = np.asarray(y_tgt, dtype=int)
    rng = np.random.default_rng(seed)
    halves = [rng.permutation(len(y_tgt)) for _ in range(draws)]
    rows = []
    for rung in rungs:
        record: dict[str, list[float]] = {
            k: []
            for k in (
                "ill_in_sample",
                "sensitivity",
                "specificity",
                "ill_flagged_alone",
                "ill_deferred",
                "healthy_cleared_alone",
                "healthy_deferred",
                "deferred_all",
            )
        }
        for order in halves:
            pool, held = two_halves(order)
            if rung:
                drawn = rng.choice(pool, size=rung, replace=False)
                fit_p, fit_y = p_tgt[drawn], y_tgt[drawn]
            else:
                fit_p, fit_y = p_cal, np.asarray(y_cal, dtype=int)
            sets = conformal_sets(fit_p, fit_y, p_tgt[held], alpha)
            y_eval = y_tgt[held]
            flagged = sets["plain"][:, 1]
            shares = outcome_shares(sets["perlabel"], y_eval)
            record["ill_in_sample"].append(float(fit_y.sum()) if rung else math.nan)
            record["sensitivity"].append(float(flagged[y_eval == 1].mean()))
            record["specificity"].append(float((~flagged)[y_eval == 0].mean()))
            record["ill_flagged_alone"].append(shares["ill"]["right_alone"]["share"])
            record["ill_deferred"].append(shares["ill"]["deferred"]["share"])
            record["healthy_cleared_alone"].append(shares["healthy"]["right_alone"]["share"])
            record["healthy_deferred"].append(shares["healthy"]["deferred"]["share"])
            record["deferred_all"].append(float((sets["perlabel"].sum(axis=1) != 1).mean()))
        row: dict[str, Any] = {"labels": rung, "draws": draws}
        for key, values in record.items():
            if key == "ill_in_sample" and not rung:
                continue
            row[key] = _summary(values)
        row["share_of_draws_below_level"] = float(np.mean(np.array(record["sensitivity"]) < level))
        rows.append(row)
    return rows


def _design(frame: pd.DataFrame, columns: Sequence[str]) -> Array:
    return frame[list(columns)].to_numpy(dtype=float)


def standardised_coverage(
    covered: BoolArray,
    features: pd.DataFrame,
    outpatient: BoolArray,
    n_draws: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """How much of the gap between inpatients and outpatients the case mix explains.

    Among the ill of both settings, a logistic model predicts whether the
    threshold caught a patient from the features and the setting.  The
    outpatients' coverage is then predicted at the inpatients' case mix: each
    inpatient's features, read as an outpatient.  The share explained is the
    part of the gap that prediction closes.  Patients are resampled within each
    setting and the model refitted for the interval.
    """
    covered = np.asarray(covered, dtype=int)
    outpatient = np.asarray(outpatient, dtype=bool)
    columns = list(features.columns)

    def estimate(rows: IntArray) -> dict[str, float]:
        frame = features.iloc[rows].reset_index(drop=True)
        x = np.column_stack([_design(frame, columns), outpatient[rows].astype(float)])
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(x, covered[rows])
        inpatients = ~outpatient[rows]
        x_counterfactual = x[inpatients].copy()
        x_counterfactual[:, -1] = 1.0
        at_inpatient_mix = float(model.predict_proba(x_counterfactual)[:, 1].mean())
        observed_in = float(covered[rows][inpatients].mean())
        observed_out = float(covered[rows][~inpatients].mean())
        gap = observed_in - observed_out
        return {
            "observed_inpatient": observed_in,
            "observed_outpatient": observed_out,
            "outpatient_at_inpatient_mix": at_inpatient_mix,
            "share_explained": (at_inpatient_mix - observed_out) / gap if gap else math.nan,
        }

    every = np.arange(len(covered))
    point = estimate(every)
    rng = np.random.default_rng(seed)
    groups = [every[outpatient], every[~outpatient]]
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_draws):
        rows = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        for key, value in estimate(rows).items():
            draws[key].append(value)
    return {
        "features": columns,
        "n_inpatient": int((~outpatient).sum()),
        "n_outpatient": int(outpatient.sum()),
        "bootstrap_draws": n_draws,
        **{
            key: {
                "estimate": value,
                "low": float(np.percentile(draws[key], 2.5)),
                "high": float(np.percentile(draws[key], 97.5)),
            }
            for key, value in point.items()
        },
    }


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm's step-down adjustment, in the order the p-values were given."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def subgroup_tests(
    covered: BoolArray,
    groups: dict[str, NDArray[np.str_]],
) -> list[dict[str, Any]]:
    """One test per subgroup kind: Fisher's exact test for two groups, a chi-square
    test of independence for more.  Each row carries its counts."""
    covered = np.asarray(covered, dtype=bool)
    rows = []
    for kind, values in groups.items():
        names = sorted(set(values.tolist()))
        table = np.array(
            [
                [int((covered & (values == g)).sum()), int((~covered & (values == g)).sum())]
                for g in names
            ]
        )
        two = len(names) == 2
        p = fisher_exact(table)[1] if two else chi2_contingency(table)[1]
        rows.append(
            {
                "kind": kind,
                "groups": {
                    g: {"caught": int(c), "n": int(c + m)}
                    for g, (c, m) in zip(names, table, strict=True)
                },
                "test": "fisher_exact" if two else "chi2_contingency",
                "p": float(p),
            }
        )
    return rows
