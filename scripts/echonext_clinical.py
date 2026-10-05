"""EchoNext, counted in patients: what the inpatient threshold does to 100 ill outpatients.

For the composite label (moderate or worse structural heart disease) and each of
the four arms, with the thresholds fitted on the 1,903 calibration inpatients
exactly as ``scripts/echonext_transfer.py`` fits them:

  outcomes     per 100 ill and per 100 healthy patients in each care setting of
               the test split, under the sensitivity threshold alone (plain) and
               under the per-label pair: caught alone, deferred, missed;
  bedside      sensitivity, specificity, PPV and NPV at the outpatients'
               prevalence and at 10% and 5%, and per 1,000 patients the flagged
               (each one an echocardiogram) and the ill missed;
  label_free   what a site sees without diagnoses: the share flagged, the share
               deferred, the mean score;
  paired       differences in outpatient sensitivity between arms, read on the
               same patients, and the fall from inpatients to outpatients;
  ladder       thresholds refitted on 25 to 200 labelled outpatients, the halves
               re-drawn in every draw, with the ill each sample held;
  case_mix     the outpatients' sensitivity predicted at the inpatients' case
               mix (findings, ejection fraction, age, sex), against the
               inpatients of the test split;
  subgroups    sex and age among ill outpatients, each test Holm-adjusted;
  histogram    the trained network's score, binned, by setting and class;
  roc          each arm's specificity at every whole-percent sensitivity, by
               setting, so the curve and the threshold's point on it can be drawn.

Reads the stored scores under ``$ECS_ECHONEXT_DERIVED/scores`` and refuses to
run without them.  Writes ``results/echonext_clinical.json``: counts and
aggregates only.  About two minutes on an Apple M5.

Usage: uv run python scripts/echonext_clinical.py
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ecs.clinical import (
    holm,
    outcome_shares,
    paired_difference,
    per_thousand,
    predictive_values,
    rerandomised_ladder,
    standardised_coverage,
    subgroup_tests,
)
from ecs.config import REPO_ROOT, RESULTS_DIR
from ecs.conformal import lac_scores, mondrian_quantiles
from ecs.echonext import (
    COMPOSITE,
    DERIVED_DIR,
    DISTRIBUTION,
    LABELS,
    read_metadata,
    transfer_cohorts,
)
from ecs.transfer import AGE_BANDS, ALPHA, conformal_sets

ARMS = ("resnet", "echonext_mini", "ecgfounder", "random_init")
STRONGEST = ("resnet", "echonext_mini", "ecgfounder")
CONTEXTS = ("inpatient", "emergency", "outpatient")
FINDINGS = LABELS[:-1]
SCORES_DIR = DERIVED_DIR / "scores"
SCREENING_PREVALENCES = (0.10, 0.05)
LADDER_RUNGS = (0, 25, 50, 100, 200)
LADDER_DRAWS = 200
BINS = np.linspace(0.0, 1.0, 21)
SENSITIVITY_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)


def composite_scores(arm: str, meta: pd.DataFrame) -> NDArray[np.float64]:
    """The arm's stored composite probability for every metadata row, NaN where unscored."""
    path = SCORES_DIR / f"{arm}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} is absent; this script reads stored scores, never refits")
    with np.load(path) as data:
        if tuple(data["labels"].tolist()) != LABELS:
            raise ValueError(f"{path} holds labels {data['labels']}, expected {LABELS}")
        where = pd.Index(meta["ecg_key"]).get_indexer(data["ecg_key"])
        if (where < 0).any():
            raise ValueError(f"{path} names ECGs absent from the metadata")
        probs = np.full(len(meta), np.nan)
        probs[where] = data["probs"][:, LABELS.index(COMPOSITE)]
    return probs


def age_band(ages: pd.Series) -> NDArray[np.str_]:
    out = np.full(len(ages), "unknown", dtype=object)
    for low, high, name in AGE_BANDS:
        out[(ages >= low).to_numpy() & (ages < high).to_numpy()] = name
    return out.astype(str)


def case_mix(frame: pd.DataFrame) -> pd.DataFrame:
    """The features the case-mix model reads: the eleven findings, the ejection
    fraction band, the age band and sex."""
    lvef = frame["lvef_value"].to_numpy(dtype=float)
    bands = age_band(frame["age_at_ecg"])
    columns: dict[str, NDArray[np.float64]] = {f: frame[f].to_numpy(dtype=float) for f in FINDINGS}
    columns |= {
        "lvef_le_35": (lvef <= 35).astype(float),
        "lvef_36_45": ((lvef > 35) & (lvef <= 45)).astype(float),
        "lvef_unmeasured": np.isnan(lvef).astype(float),
        "male": (frame["sex"].astype(str) == "male").to_numpy(dtype=float),
    }
    for _, _, name in AGE_BANDS[1:]:
        columns[f"age_{name}"] = (bands == name).astype(float)
    return pd.DataFrame(columns)


def bedside(sens: float, spec: float, prevalence: float) -> dict[str, Any]:
    return {
        "prevalence": prevalence,
        **predictive_values(sens, spec, prevalence),
        "per_thousand": per_thousand(sens, spec, prevalence),
    }


def thresholds(p_cal: NDArray[np.float64], y_cal: NDArray[np.int_]) -> dict[str, float]:
    """The cut-offs on the probability: flag above ``ill``, clear below ``healthy``."""
    two = np.column_stack([1 - p_cal, p_cal])
    q = mondrian_quantiles(lac_scores(two, y_cal), y_cal, ALPHA, n_classes=2)
    return {"ill": float(1 - q[1]), "healthy": float(q[0])}


def measure_arm(
    p: NDArray[np.float64], y: NDArray[np.int_], cal: NDArray[np.int_], targets: dict[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {"thresholds": thresholds(p[cal], y[cal])}
    for context, rows in targets.items():
        sets = conformal_sets(p[cal], y[cal], p[rows], ALPHA)
        flagged = sets["plain"][:, 1]
        yt = y[rows]
        sens = float(flagged[yt == 1].mean())
        spec = float((~flagged)[yt == 0].mean())
        prevalence = float(yt.mean())
        out[context] = {
            "n": int(len(rows)),
            "n_ill": int(yt.sum()),
            "outcomes": {m: outcome_shares(sets[m], yt) for m in ("plain", "perlabel")},
            "sensitivity": sens,
            "specificity": spec,
            "bedside": [bedside(sens, spec, prevalence)]
            + [bedside(sens, spec, q) for q in SCREENING_PREVALENCES],
            "label_free": {
                "share_flagged": float(flagged.mean()),
                "share_deferred": float((sets["perlabel"].sum(axis=1) != 1).mean()),
                "mean_score": float(p[rows].mean()),
            },
        }
    return out


def histogram(
    p: NDArray[np.float64], y: NDArray[np.int_], targets: dict[str, Any]
) -> dict[str, Any]:
    return {
        "bins": BINS.tolist(),
        **{
            context: {
                cls: np.histogram(p[rows][y[rows] == k], bins=BINS)[0].tolist()
                for cls, k in (("ill", 1), ("healthy", 0))
            }
            for context, rows in targets.items()
        },
    }


def roc(p: NDArray[np.float64], y: NDArray[np.int_], targets: dict[str, Any]) -> dict[str, Any]:
    """Specificity at each sensitivity of the grid: the best a cut-off on these
    patients' own labels could do."""
    out: dict[str, Any] = {"sensitivity": SENSITIVITY_GRID.tolist()}
    for context, rows in targets.items():
        ill, healthy = np.sort(p[rows][y[rows] == 1]), p[rows][y[rows] == 0]
        cut = np.quantile(ill, 1 - SENSITIVITY_GRID, method="inverted_cdf")
        out[context] = [float((healthy < c).mean()) for c in cut]
    return out


def measure(meta: pd.DataFrame) -> dict[str, Any]:
    cohorts = transfer_cohorts(meta, "inpatient", CONTEXTS)
    cal, targets = cohorts.calibration, cohorts.targets
    y = meta[COMPOSITE].to_numpy(dtype=int)
    out_rows = targets["outpatient"]
    in_rows = targets["inpatient"]
    probs = {arm: composite_scores(arm, meta) for arm in ARMS}

    result: dict[str, Any] = {
        "distribution": DISTRIBUTION,
        "label": COMPOSITE,
        "alpha": ALPHA,
        "calibration": {"n": int(len(cal)), "n_ill": int(y[cal].sum())},
        "arms": {arm: measure_arm(probs[arm], y, cal, targets) for arm in ARMS},
    }

    flags = {
        arm: conformal_sets(probs[arm][cal], y[cal], probs[arm][out_rows], ALPHA)["plain"][:, 1]
        for arm in ARMS
    }
    ill_out = y[out_rows] == 1
    result["paired"] = {
        "outpatient_sensitivity": {
            f"{a}-{b}": paired_difference(flags[a][ill_out], flags[b][ill_out])
            for i, a in enumerate(STRONGEST)
            for b in STRONGEST[i + 1 :]
        }
    }

    result["ladder"] = {
        arm: rerandomised_ladder(
            probs[arm][cal], y[cal], probs[arm][out_rows], y[out_rows], LADDER_RUNGS, LADDER_DRAWS
        )
        for arm in ARMS
    }

    ill_in_rows, ill_out_rows = in_rows[y[in_rows] == 1], out_rows[y[out_rows] == 1]
    both = np.concatenate([ill_in_rows, ill_out_rows])
    features = case_mix(meta.iloc[both].reset_index(drop=True))
    is_out = np.r_[np.zeros(len(ill_in_rows), bool), np.ones(len(ill_out_rows), bool)]
    result["case_mix"] = {}
    for arm in STRONGEST:
        caught = conformal_sets(probs[arm][cal], y[cal], probs[arm][both], ALPHA)["plain"][:, 1]
        result["case_mix"][arm] = standardised_coverage(caught, features, is_out)

    sub_meta = meta.iloc[ill_out_rows]
    groups = {
        "sex": sub_meta["sex"].astype(str).to_numpy(),
        "age": age_band(sub_meta["age_at_ecg"]),
    }
    tests = []
    for arm in STRONGEST:
        caught = flags[arm][ill_out]
        tests += [{"arm": arm, **row} for row in subgroup_tests(caught, groups)]
    for row, adjusted in zip(tests, holm([t["p"] for t in tests]), strict=True):
        row["p_holm"] = adjusted
    result["subgroups"] = {
        "family": "sex and age, three arms, Holm-adjusted together",
        "tests": tests,
    }

    result["histogram"] = {"arm": "resnet", **histogram(probs["resnet"], y, targets)}
    result["roc"] = {arm: roc(probs[arm], y, targets) for arm in ARMS}
    return result


def main() -> None:
    start = time.time()
    result = measure(read_metadata())
    result["seconds"] = round(time.time() - start, 1)
    path = RESULTS_DIR / "echonext_clinical.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path.relative_to(REPO_ROOT), result["seconds"], "s")


if __name__ == "__main__":
    main()
