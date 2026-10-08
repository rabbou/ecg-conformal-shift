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
  subgroups    sex and age among ill outpatients, each test Holm-adjusted; and
               within each sex and age band, the specificity among healthy
               outpatients and the AUROC; and the sensitivity among the ill of
               each race and ethnicity group, by setting;
  histogram    the trained network's score, binned, by setting and class;
  roc          each arm's specificity at every whole-percent sensitivity, by
               setting, so the curve and the threshold's point on it can be drawn;
  calibration_variants
               the same rule set on other validation patients: every setting
               together, and the validation split's own outpatients, read on
               the test split by setting;
  ladder_validation
               thresholds refitted on 25 to 200 outpatients drawn from the
               validation split's outpatients, read on every test outpatient;
  flow         ECGs and patients by split and setting, and which this study
               uses;
  by_finding   the composite threshold's catch among the ill carrying each
               finding, by setting, and who the missed outpatients are;
  eras         the year each cohort's ECGs were recorded, and the sensitivity
               of the inpatient threshold within each band of years;
  refit_without_margin
               the refit on 100 outpatients with the sample's own 10th
               percentile and no finite-sample margin;
  age_sex      a score on age and sex alone, a logistic regression fitted on the
               training split, thresholded and read like the ECG models;
  threshold_spread
               how far the sensitivity moves when the calibration inpatients
               are redrawn, by bootstrap, and the sensitivity at the plain
               90th-percentile threshold without the (n+1) correction;
  half_means   the mean sensitivity of the refit on 100 over the draws of one
               cut into halves, for 400 random cuts: the spread a refit read
               on one fixed half belongs in;
  decision     net benefit per outpatient at decision thresholds of 5%, 10%
               and 20%, for each way of setting the threshold and for an
               echocardiogram for every outpatient or none;
  unmeasured   the healthy ECGs that carry no echocardiographic measurement,
               by setting, and the specificity, prevalence and AUROC once
               they are set aside.

Reads the stored scores under ``$ECS_ECHONEXT_DERIVED/scores`` and refuses to
run without them.  Writes ``results/echonext_clinical.json``: counts and
aggregates only.  The run's wall time is that file's ``seconds`` field.

Usage: uv run python scripts/echonext_clinical.py
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from ecs.clinical import (
    empirical_refit,
    external_ladder,
    half_means,
    holm,
    net_benefit,
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
from ecs.metrics import wilson_interval
from ecs.transfer import AGE_BANDS, ALPHA, auroc_row, conformal_sets

ARMS = ("resnet", "echonext_mini", "ecgfounder", "random_init")
STRONGEST = ("resnet", "echonext_mini", "ecgfounder")
CONTEXTS = ("inpatient", "emergency", "outpatient")
FINDINGS = LABELS[:-1]
SCORES_DIR = DERIVED_DIR / "scores"
SCREENING_PREVALENCES = (0.10, 0.05)
DECISION_THRESHOLDS = (0.05, 0.10, 0.20)
LADDER_RUNGS = (0, 25, 50, 100, 200)
# 2,000 draws put the Monte Carlo error of a share of draws near one point.
LADDER_DRAWS = 2000
# EchoNext records these finer measurements only for an ECG taken within a year
# before the echocardiogram; an ECG with all of them blank was taken earlier.
MEASUREMENTS = (
    "aortic_stenosis_value",
    "aortic_regurgitation_value",
    "mitral_regurgitation_value",
    "tricuspid_regurgitation_value",
    "pulmonary_regurgitation_value",
    "rv_systolic_function_value",
    "pericardial_effusion_value",
    "ivs_measurement",
    "lvpw_measurement",
    "pasp_value",
    "tr_max_velocity_value",
    "lvef_value",
)
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


VARIANTS = {
    "inpatients": ("inpatient",),
    "every_setting": ("inpatient", "emergency", "outpatient", "procedural"),
    "outpatients": ("outpatient",),
}


def validation_rows(meta: pd.DataFrame, settings: tuple[str, ...]) -> NDArray[np.int_]:
    split = meta["split"].to_numpy()
    context = meta["location_setting"].to_numpy()
    return np.flatnonzero((split == "val") & np.isin(context, settings))


def calibration_variants(
    meta: pd.DataFrame,
    probs: dict[str, NDArray[np.float64]],
    y: NDArray[np.int_],
    targets: dict[str, Any],
) -> dict[str, Any]:
    """The same 90% rule set on three groups of validation patients and read on the
    test split: the inpatients the report uses, every validation patient whatever
    the setting, and the validation split's own outpatients.  Validation and test
    hold different patients, which this checks."""
    patients = meta["patient_key"].to_numpy()
    tested = set(patients[np.concatenate(list(targets.values()))])
    out: dict[str, Any] = {}
    for name, settings in VARIANTS.items():
        pool = validation_rows(meta, settings)
        if tested & set(patients[pool]):
            raise ValueError(f"validation group {name} shares patients with the test split")
        entry: dict[str, Any] = {
            "settings": list(settings),
            "n": int(len(pool)),
            "n_ill": int(y[pool].sum()),
            "arms": {},
        }
        for arm in ARMS:
            p = probs[arm]
            entry["arms"][arm] = {"threshold": thresholds(p[pool], y[pool])["ill"]}
            for context, rows in targets.items():
                plain = outcome_shares(
                    conformal_sets(p[pool], y[pool], p[rows], ALPHA)["plain"], y[rows]
                )
                entry["arms"][arm][context] = {
                    "sensitivity": plain["ill"]["right_alone"],
                    "specificity": plain["healthy"]["right_alone"],
                }
        out[name] = entry
    return out


def age_sex(
    meta: pd.DataFrame, y: NDArray[np.int_], cal: NDArray[np.int_], targets: dict[str, Any]
) -> dict[str, Any]:
    """A comparator that sees no ECG: age and sex in a logistic regression fitted on
    the training split, its threshold set on the calibration inpatients by the same
    rule.  If its sensitivity also fell among outpatients, the fall would belong to the
    population rather than to what the ECG shows."""
    from sklearn.linear_model import LogisticRegression

    x = np.column_stack(
        [
            meta["age_at_ecg"].to_numpy(dtype=float),
            (meta["sex"].astype(str) == "male").to_numpy(dtype=float),
        ]
    )
    train = np.flatnonzero(meta["split"].to_numpy() == "train")
    model = LogisticRegression(max_iter=1000).fit(x[train], y[train])
    p = model.predict_proba(x)[:, 1]
    out: dict[str, Any] = {"n_train": int(len(train))}
    roc_at = roc(p, y, targets)
    at90 = SENSITIVITY_GRID.tolist().index(0.9)
    for context in ("inpatient", "outpatient"):
        rows = targets[context]
        flagged = conformal_sets(p[cal], y[cal], p[rows], ALPHA)["plain"][:, 1]
        yt = y[rows]
        out[context] = {
            "sensitivity": float(flagged[yt == 1].mean()),
            "specificity": float((~flagged)[yt == 0].mean()),
            "auroc": float(roc_auc_score(yt, p[rows])),
            "specificity_at_90": roc_at[context][at90],
        }
    return out


def threshold_spread(
    p: NDArray[np.float64],
    y: NDArray[np.int_],
    cal: NDArray[np.int_],
    targets: dict[str, Any],
    draws: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """The sensitivity in each setting when the calibration inpatients are resampled
    and the threshold set again, and at the uncorrected 90th percentile.

    A Wilson interval holds the threshold fixed and resamples the patients read; a
    clinic that sets the threshold on its own calibration sample also carries that
    sample's chance.  The spread here is that second part.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"draws": draws}
    resampled = [rng.choice(cal, size=len(cal), replace=True) for _ in range(draws)]
    ill_cal = np.sort(p[cal][y[cal] == 1])
    plain_cut = np.quantile(ill_cal, ALPHA, method="inverted_cdf")
    for context in ("inpatient", "outpatient"):
        rows = targets[context]
        ill = rows[y[rows] == 1]
        spread = [
            float(conformal_sets(p[c], y[c], p[ill], ALPHA)["plain"][:, 1].mean())
            for c in resampled
        ]
        out[context] = {
            "sd": float(np.std(spread)),
            "p2_5": float(np.percentile(spread, 2.5)),
            "p97_5": float(np.percentile(spread, 97.5)),
            "uncorrected_sensitivity": float((p[ill] >= plain_cut).mean()),
        }
    return out


def decision(result: dict[str, Any], arm: str) -> dict[str, Any]:
    """Net benefit among the test outpatients of each way of setting the threshold,
    from the sensitivities and specificities measured elsewhere in ``result``."""
    measured = result["arms"][arm]["outpatient"]
    prevalence = measured["n_ill"] / measured["n"]
    ladder = {r["labels"]: r for r in result["ladder"][arm]}
    held = result["calibration_variants"]["outpatients"]["arms"][arm]["outpatient"]
    roc_at90 = result["roc"][arm]["outpatient"][SENSITIVITY_GRID.tolist().index(0.9)]
    options = {
        "inpatient_threshold": (measured["sensitivity"], measured["specificity"]),
        "refit_100": (ladder[100]["sensitivity"]["mean"], ladder[100]["specificity"]["mean"]),
        "refit_200": (ladder[200]["sensitivity"]["mean"], ladder[200]["specificity"]["mean"]),
        "validation_outpatients": (held["sensitivity"]["share"], held["specificity"]["share"]),
        "every_diagnosis_known_90": (0.9, roc_at90),
        "echo_for_all": (1.0, 0.0),
        "echo_for_none": (0.0, 1.0),
    }
    return {
        "prevalence": prevalence,
        "options": {
            name: {
                "sensitivity": sens,
                "specificity": spec,
                "flagged_per_thousand": 1000 * (sens * prevalence + (1 - spec) * (1 - prevalence)),
                "net_benefit": {
                    f"{t:.2f}": net_benefit(sens, spec, prevalence, t) for t in DECISION_THRESHOLDS
                },
            }
            for name, (sens, spec) in options.items()
        },
    }


ERAS = ((2008, 2015), (2016, 2018), (2019, 2022))
SEVERE_GRADES = ("severe", "severely_reduced", "large")
GRADES = MEASUREMENTS[:7]  # the graded findings; the rest are measurements
WALL = "lvwt_gte_13_flag"


def by_finding(
    meta: pd.DataFrame,
    probs: dict[str, NDArray[np.float64]],
    y: NDArray[np.int_],
    cal: NDArray[np.int_],
    targets: dict[str, Any],
) -> dict[str, Any]:
    """The composite threshold, the one the report follows, read among the ill who
    carry each finding; and, among the ill outpatients it misses, how many carry one
    finding only, which, and how many a severe one."""
    severe = meta[list(GRADES)].isin(SEVERE_GRADES).any(axis=1).to_numpy() | (
        meta["lvef_value"].to_numpy(dtype=float) <= 35
    )
    findings = meta[list(FINDINGS)].to_numpy(dtype=int)
    out: dict[str, Any] = {}
    for arm in STRONGEST:
        p = probs[arm]
        entry: dict[str, Any] = {}
        caught_of: dict[str, NDArray[np.bool_]] = {}
        for context in ("inpatient", "outpatient"):
            rows = targets[context]
            flagged = conformal_sets(p[cal], y[cal], p[rows], ALPHA)["plain"][:, 1]
            caught_of[context] = flagged
            entry[context] = {}
            for j, finding in enumerate(FINDINGS):
                ill = findings[rows, j] == 1
                if ill.any():
                    low, high = wilson_interval(int(flagged[ill].sum()), int(ill.sum()))
                    entry[context][finding] = {
                        "n": int(ill.sum()),
                        "caught": int(flagged[ill].sum()),
                        "low": low,
                        "high": high,
                    }
            alone_wall = (findings[rows].sum(axis=1) == 1) & (
                findings[rows, FINDINGS.index(WALL)] == 1
            )
            other_ill = (y[rows] == 1) & ~alone_wall
            entry[context]["without_wall_alone"] = {
                "n": int(other_ill.sum()),
                "caught": int(flagged[other_ill].sum()),
            }
        rows = targets["outpatient"]
        missed = (y[rows] == 1) & ~caught_of["outpatient"]
        single = missed & (findings[rows].sum(axis=1) == 1)
        entry["missed_outpatients"] = {
            "n": int(missed.sum()),
            "one_finding": {
                f: int((single & (findings[rows, j] == 1)).sum())
                for j, f in enumerate(FINDINGS)
                if (single & (findings[rows, j] == 1)).any()
            },
            "severe": int((missed & severe[rows]).sum()),
        }
        out[arm] = entry
    return out


def eras(
    meta: pd.DataFrame,
    probs: dict[str, NDArray[np.float64]],
    y: NDArray[np.int_],
    cal: NDArray[np.int_],
    targets: dict[str, Any],
) -> dict[str, Any]:
    """Recording years by cohort, and the inpatient threshold's sensitivity among the
    ill of each setting within bands of years: whether the fall from inpatients to
    outpatients survives inside one era."""
    year = meta["acquisition_year"].to_numpy(dtype=int)
    cohorts = {"calibration": cal} | dict(targets)
    out: dict[str, Any] = {
        "years": {
            name: {
                "median": float(np.median(year[rows])),
                "min": int(year[rows].min()),
                "max": int(year[rows].max()),
                "share_to_2018": float((year[rows] <= 2018).mean()),
            }
            for name, rows in cohorts.items()
        },
        "bands": [list(b) for b in ERAS],
        "arms": {},
    }
    for arm in STRONGEST:
        p = probs[arm]
        out["arms"][arm] = {}
        for context in ("inpatient", "outpatient"):
            rows = targets[context]
            out["arms"][arm][context] = [
                outcome_shares(
                    conformal_sets(p[cal], y[cal], p[rows][band], ALPHA)["plain"],
                    y[rows][band],
                )["ill"]["right_alone"]
                for band in ((year[rows] >= low) & (year[rows] <= high) for low, high in ERAS)
            ]
    return out


def flow(meta: pd.DataFrame) -> dict[str, Any]:
    """Every ECG of the distribution by split and setting, the patients behind them,
    and the cells this study reads."""
    table = pd.crosstab(meta["split"], meta["location_setting"])
    return {
        "ecgs": {s: {c: int(n) for c, n in row.items()} for s, row in table.iterrows()},
        "patients": {s: int(n) for s, n in meta.groupby("split")["patient_key"].nunique().items()},
        "youngest": {s: int(n) for s, n in meta.groupby("split")["age_at_ecg"].min().items()},
        "used": {
            "train": "every setting, to train the network and fit the probes",
            "val": "inpatients set the thresholds; every setting and the outpatients set the "
            "variants of calibration_variants and ladder_validation",
            "test": "inpatient, emergency and outpatient cohorts; procedural not read",
            "no_split": "not used",
        },
    }


def unmeasured(
    meta: pd.DataFrame,
    probs: dict[str, NDArray[np.float64]],
    y: NDArray[np.int_],
    cal: NDArray[np.int_],
    targets: dict[str, Any],
) -> dict[str, Any]:
    """The healthy ECGs with no echocardiographic measurement, and the figures without them.

    EchoNext labels an ECG ill when it was taken within a year before an
    abnormal echocardiogram, and healthy when it was taken at any time before
    the patient's last normal one.  Its finer measurements exist only within the
    year, so a healthy ECG with every measurement blank was taken earlier.  The
    specificity, prevalence and AUROC are read again on the ECGs that carry a
    measurement, with the same calibration-inpatient threshold.
    """
    blank = meta[list(MEASUREMENTS)].isna().all(axis=1).to_numpy()
    out: dict[str, Any] = {}
    for context, rows in targets.items():
        yt = y[rows]
        kept = rows[~blank[rows]]
        yk = y[kept]
        entry: dict[str, Any] = {
            "n_healthy": int((yt == 0).sum()),
            "n_healthy_unmeasured": int(blank[rows][yt == 0].sum()),
            "n_ill_unmeasured": int(blank[rows][yt == 1].sum()),
            "n_measured": int(len(kept)),
            "prevalence_measured": float(yk.mean()),
            "arms": {},
        }
        for arm in STRONGEST:
            p = probs[arm]
            flagged = conformal_sets(p[cal], y[cal], p[kept], ALPHA)["plain"][:, 1]
            entry["arms"][arm] = {
                "specificity_measured": float((~flagged)[yk == 0].mean()),
                "auroc_measured": float(roc_auc_score(yk, p[kept])),
            }
        out[context] = entry
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
            probs[arm][cal],
            y[cal],
            probs[arm][out_rows],
            y[out_rows],
            LADDER_RUNGS,
            LADDER_DRAWS,
            alpha=ALPHA,
            level=1 - ALPHA,
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
    out_meta = meta.iloc[out_rows]
    out_groups = {
        "sex": out_meta["sex"].astype(str).to_numpy(),
        "age": age_band(out_meta["age_at_ecg"]),
    }
    beside: dict[str, Any] = {}
    for arm in STRONGEST:
        p, yo = probs[arm][out_rows], y[out_rows]
        flagged = conformal_sets(probs[arm][cal], y[cal], p, ALPHA)["plain"][:, 1]
        beside[arm] = {}
        for kind, values in out_groups.items():
            beside[arm][kind] = {}
            for group in sorted(set(values.tolist())):
                g = values == group
                healthy = g & (yo == 0)
                low, high = wilson_interval(int((~flagged[healthy]).sum()), int(healthy.sum()))
                beside[arm][kind][group] = {
                    "healthy": int(healthy.sum()),
                    "specificity": float((~flagged[healthy]).mean()),
                    "specificity_low": low,
                    "specificity_high": high,
                    "auroc": auroc_row(yo[g], p[g]),
                }
    race: dict[str, Any] = {}
    for arm in STRONGEST:
        race[arm] = {}
        for context in ("inpatient", "outpatient"):
            rows = targets[context]
            flagged = conformal_sets(probs[arm][cal], y[cal], probs[arm][rows], ALPHA)["plain"][
                :, 1
            ]
            values = meta.iloc[rows]["race_ethnicity"].astype(str).to_numpy()
            ill = y[rows] == 1
            race[arm][context] = {
                group: {
                    "n": int((ill & (values == group)).sum()),
                    "caught": int(flagged[ill & (values == group)].sum()),
                }
                for group in sorted(set(values.tolist()))
            }
    result["subgroups"] = {
        "family": "sex and age, three arms, Holm-adjusted together",
        "tests": tests,
        "healthy_and_auroc": beside,
        "race_ethnicity": race,
    }

    result["histogram"] = {"arm": "resnet", **histogram(probs["resnet"], y, targets)}
    result["roc"] = {arm: roc(probs[arm], y, targets) for arm in ARMS}
    result["unmeasured"] = unmeasured(meta, probs, y, cal, targets)
    result["calibration_variants"] = calibration_variants(meta, probs, y, targets)
    pool = validation_rows(meta, VARIANTS["outpatients"])
    result["ladder_validation"] = {
        arm: external_ladder(
            probs[arm][pool],
            y[pool],
            probs[arm][out_rows],
            y[out_rows],
            LADDER_RUNGS[1:],
            LADDER_DRAWS,
            alpha=ALPHA,
            level=1 - ALPHA,
        )
        for arm in ARMS
    }
    result["flow"] = flow(meta)
    result["eras"] = eras(meta, probs, y, cal, targets)
    result["by_finding"] = by_finding(meta, probs, y, cal, targets)
    result["refit_without_margin"] = {
        arm: empirical_refit(probs[arm][out_rows], y[out_rows], 100, LADDER_DRAWS, level=1 - ALPHA)
        for arm in ARMS
    }
    result["decision"] = {arm: decision(result, arm) for arm in ARMS}
    result["age_sex"] = age_sex(meta, y, cal, targets)
    result["threshold_spread"] = {
        arm: threshold_spread(probs[arm], y, cal, targets) for arm in STRONGEST
    }
    result["half_means"] = {
        arm: {
            "labels": 100,
            "halves": 400,
            "draws_per_half": 100,
            "means": half_means(probs[arm][out_rows], y[out_rows], 100, 400, 100, alpha=ALPHA),
        }
        for arm in STRONGEST
    }
    return result


PRODUCERS = [
    "scripts/echonext_clinical.py",
    "src/ecs/clinical.py",
    "src/ecs/conformal.py",
    "src/ecs/echonext.py",
    "src/ecs/metrics.py",
    "src/ecs/transfer.py",
]


def inputs() -> dict[str, str]:
    """The SHA-256 of every file the numbers are read from that cannot be committed."""
    from ecs.echonext import ECHONEXT_DIR, METADATA
    from ecs.provenance import digest_of

    files = {f"echonext/{METADATA}": ECHONEXT_DIR / METADATA}
    files |= {f"echonext-derived/scores/{arm}.npz": SCORES_DIR / f"{arm}.npz" for arm in ARMS}
    return {name: digest_of(path) for name, path in files.items()}


def main() -> None:
    from ecs.provenance import provenance_block

    start = time.time()
    result = measure(read_metadata())
    result["inputs"] = inputs()
    result["provenance"] = provenance_block(PRODUCERS)
    result["seconds"] = round(time.time() - start, 1)
    path = RESULTS_DIR / "echonext_clinical.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path.relative_to(REPO_ROOT), result["seconds"], "s")


if __name__ == "__main__":
    main()
