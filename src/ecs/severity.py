"""Severity among the ill, and coverage of the ill read at equal severity.

The per-label threshold covers fewer ill outpatients than ill inpatients.  If
outpatients are ill more mildly, and milder disease is harder to read on the
ECG, the drop could come from the severity mix alone.  This module holds what
it takes to check: the echocardiographic grades as ordered numbers, the
severity strata, the coverage inside each stratum, and the outpatients'
coverage reweighted to the inpatients' severity mix, with a bootstrap interval.

``VERDICT_RULE`` is the test the verdict is read from.  It was written down
before any coverage by stratum was computed, and the results file carries it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from .metrics import percentile_interval, wilson_interval

__all__ = [
    "BOOTSTRAP_DRAWS",
    "CONTINUOUS",
    "GRADES",
    "LVEF_BANDS",
    "MIN_STRATUM",
    "MODERATE",
    "VERDICT_RULE",
    "bootstrap_reweighted",
    "compare",
    "findings_count",
    "graded",
    "lvef_band",
    "reweighted",
    "severity_cell",
    "stratum_rows",
    "verdict",
]

BoolArray = NDArray[np.bool_]
StrArray = NDArray[np.str_]

# The ordered grades of each categorical value column.  "presumed none" is read
# as none; a value outside the list raises, so a new grade cannot pass unranked.
_VALVE = {"none": 0, "presumed none": 0, "mild": 1, "moderate": 2, "severe": 3}
GRADES: dict[str, dict[str, int]] = {
    "aortic_stenosis_value": _VALVE,
    "aortic_regurgitation_value": _VALVE,
    "mitral_regurgitation_value": _VALVE,
    "tricuspid_regurgitation_value": _VALVE,
    "pulmonary_regurgitation_value": _VALVE,
    "rv_systolic_function_value": {
        "normal": 0,
        "mildly_reduced": 1,
        "moderately_reduced": 2,
        "severely_reduced": 3,
    },
    "pericardial_effusion_value": {"none": 0, "trace": 1, "small": 2, "moderate": 3, "large": 4},
}
# The first grade each column calls moderate: its finding's threshold.
MODERATE = {
    column: scale["moderately_reduced" if "moderately_reduced" in scale else "moderate"]
    for column, scale in GRADES.items()
}
CONTINUOUS = (
    "lvef_value",
    "pasp_value",
    "tr_max_velocity_value",
    "ivs_measurement",
    "lvpw_measurement",
)

# Upper bounds, inclusive, of the ejection-fraction bands.
LVEF_BANDS = ((35.0, "<=35"), (45.0, "36-45"), (float("inf"), ">45"))

# Below this many ill outpatients a stratum is reported, never judged.
MIN_STRATUM = 30
BOOTSTRAP_DRAWS = 2000
LEVEL = 0.90

VERDICT_RULE = (
    "Fixed before any coverage by stratum was computed, after counting the patients "
    "per cell. Strata: one finding or two and more, crossed with the ejection fraction "
    "at 35 or less, 36 to 45, above 45 (six cells, each holding at least 10 ill "
    "outpatients). For each of the three strongest models, R is the per-label coverage "
    "of the ill outpatients reweighted to the share of the ill calibration inpatients in "
    "each cell, O the observed coverage, and the 95% intervals come from 2,000 bootstrap "
    "draws that resample the ill of both cohorts, thresholds held fixed. Severity "
    "explains none of the drop if the interval of R - O holds 0; all of it if the "
    "interval of R reaches 90%; part of it otherwise, the share being (R - O) / "
    "(90% - O); both at once is too imprecise to judge. The verdict is the one the "
    "three models share; if they differ, each is stated."
)


def findings_count(meta: pd.DataFrame, findings: tuple[str, ...]) -> NDArray[np.int_]:
    """How many of the findings each record carries."""
    out: NDArray[np.int_] = meta[list(findings)].to_numpy(dtype=int).sum(axis=1)
    return out


def graded(values: pd.Series, column: str) -> pd.Series:
    """The column's grades as ordered numbers, missing values kept missing."""
    scale = GRADES[column]
    present = values.dropna()
    unknown = sorted(set(present) - set(scale))
    if unknown:
        raise ValueError(f"{column} holds grades {unknown} outside {sorted(scale)}")
    return values.map(scale).astype(float)


def lvef_band(lvef: NDArray[np.float64]) -> StrArray:
    lvef = np.asarray(lvef, dtype=float)
    if np.isnan(lvef).any():
        raise ValueError("an ejection fraction is missing; its band is undefined")
    names = np.array([name for _, name in LVEF_BANDS])
    bounds = np.array([bound for bound, _ in LVEF_BANDS])
    out: StrArray = names[np.searchsorted(bounds, lvef, side="left")]
    return out


def severity_cell(count: NDArray[np.int_], lvef: NDArray[np.float64]) -> StrArray:
    """The reweighting cell: one finding or two and more, by ejection-fraction band."""
    findings = np.where(np.asarray(count) >= 2, "2+", "1")
    out: StrArray = np.char.add(np.char.add(findings, " finding, LVEF "), lvef_band(lvef))
    return out


def _summary(values: pd.Series) -> dict[str, Any]:
    present = values.dropna().to_numpy(dtype=float)
    if present.size == 0:
        return {"n": 0, "missing": int(values.isna().sum())}
    q1, median, q3 = np.percentile(present, [25, 50, 75])
    return {
        "n": int(present.size),
        "missing": int(values.isna().sum()),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "mean": float(present.mean()),
    }


def compare(inpatient: pd.Series, outpatient: pd.Series) -> dict[str, Any]:
    """Medians and quartiles of both cohorts, and the Mann-Whitney test between them.

    The test is two-sided on ranks, which suits both a measurement and an ordered
    grade with many ties; it is omitted when either side has no value.
    """
    row: dict[str, Any] = {"inpatient": _summary(inpatient), "outpatient": _summary(outpatient)}
    a, b = inpatient.dropna().to_numpy(dtype=float), outpatient.dropna().to_numpy(dtype=float)
    if a.size and b.size:
        test = stats.mannwhitneyu(a, b, alternative="two-sided")
        row["test"] = "Mann-Whitney U, two-sided"
        row["p_value"] = float(test.pvalue)
    return row


def stratum_rows(
    covered: BoolArray, strata: StrArray, order: tuple[str, ...] | None = None
) -> list[dict[str, Any]]:
    """Coverage inside each stratum with its count and Wilson interval."""
    covered = np.asarray(covered, dtype=bool)
    names = order if order is not None else tuple(sorted(set(strata.tolist())))
    rows = []
    for name in names:
        inside = covered[strata == name]
        n = int(inside.size)
        row: dict[str, Any] = {"stratum": name, "n": n, "judged": n >= MIN_STRATUM}
        if n:
            k = int(inside.sum())
            low, high = wilson_interval(k, n)
            row |= {"covered": k, "coverage": k / n, "low": low, "high": high}
        rows.append(row)
    return rows


def reweighted(covered: BoolArray, strata: StrArray, reference: StrArray) -> float:
    """Coverage of the covered cohort if its strata had the reference's shares.

    Raises when a stratum of the reference holds nobody in the cohort, since its
    coverage, and so the reweighted figure, is then undefined.
    """
    covered = np.asarray(covered, dtype=bool)
    names, counts = np.unique(reference, return_counts=True)
    total = 0.0
    for name, count in zip(names.tolist(), counts.tolist(), strict=True):
        inside = covered[strata == name]
        if inside.size == 0:
            raise ValueError(f"stratum {name} holds nobody in the reweighted cohort")
        total += count / len(reference) * float(inside.mean())
    return total


def bootstrap_reweighted(
    covered: BoolArray,
    strata: StrArray,
    reference: StrArray,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 0,
    level: float = LEVEL,
) -> dict[str, float]:
    """Observed and reweighted coverage, their gap and the share of the drop below
    ``level`` that the gap closes, each with its 95% percentile interval.

    Each draw resamples the cohort and the reference independently, with
    replacement; the thresholds that made ``covered`` stay as fitted.
    """
    covered = np.asarray(covered, dtype=bool)
    rng = np.random.default_rng(seed)
    observed, weighted, gap, share = [], [], [], []
    for _ in range(draws):
        a = rng.integers(0, len(covered), len(covered))
        b = rng.integers(0, len(reference), len(reference))
        o = float(covered[a].mean())
        r = reweighted(covered[a], strata[a], reference[b])
        observed.append(o)
        weighted.append(r)
        gap.append(r - o)
        share.append((r - o) / (level - o))
    o, r = float(covered.mean()), reweighted(covered, strata, reference)

    out = {"observed": o, "reweighted": r, "gap": r - o, "share": (r - o) / (level - o)}
    for name, xs in (
        ("observed", observed),
        ("reweighted", weighted),
        ("gap", gap),
        ("share", share),
    ):
        out[f"{name}_low"], out[f"{name}_high"] = percentile_interval(xs)
    return out


def verdict(row: dict[str, float], level: float = LEVEL) -> dict[str, Any]:
    """The verdict ``VERDICT_RULE`` states, from one model's bootstrap row."""
    none = row["gap_low"] <= 0.0 <= row["gap_high"]
    every = row["reweighted_high"] >= level
    if none and every:
        name = "too imprecise"
    elif none:
        name = "none"
    elif every:
        name = "all"
    else:
        name = "part"
    return {"verdict": name, "drop": level - row["observed"]}
