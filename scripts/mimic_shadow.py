"""The shadow run at Beth Israel: thresholds fitted on early tracings, spent on late ones.

A hospital that deploys a model in shadow mode fits its thresholds on what it
has and lets them run, unchanged, on what comes next.  Here the thresholds are
fitted on 2,000 tracings from the earliest third of MIMIC-IV-ECG and spent on
8,000 from the latest third, next to two comparators on the same model:

``patient_split``  fitted on the rotation's MIMIC calibration part and measured
                   on its test part, both drawn by patient across all years;
``same_era``       fitted on the same early 2,000 and measured on 4,000 other
                   early patients, which isolates time from the sample.

Every cohort holds one tracing per patient and no patient the model trained or
stopped on.  Coverage per label carries its 95% Wilson interval.

Which third a tracing belongs to.  MIMIC shifts every patient's dates by its own
offset, and the real year is recoverable only through the credentialed MIMIC-IV
patients table.  ``--eras anchor`` uses that table (``ECS_MIMIC_PATIENTS_CSV``)
and keeps tracings surely in 2008-2011 against 2016-2019.  The default,
``--eras estimated``, uses open data alone: the carts are ordered in time from
within-patient differences (``ecs.mimic.cart_chronology``), each tracing is
placed by its patient's carts, and the terciles of that estimate are the eras.
The estimate is checked against the real year groups of the open MIMIC-IV demo
patients, and the check is written beside the result.

Writes ``results/mimic_shadow.json``, ``results/mimic_shadow.csv`` and the scores
under ``results/mimic_shadow/scores/<model>/<cohort>.npz``.

Usage: .venv/bin/python scripts/mimic_shadow.py [--eras estimated|anchor]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from score_rotation import Scorer  # noqa: E402

from ecs.config import MIMIC_ECG_DIR, MIMIC_PATIENTS_CSV, RESULTS_DIR
from ecs.mimic import (
    SHADOW_SIZES,
    cart_chronology,
    estimated_eras,
    estimated_years,
    real_eras,
    real_year_bounds,
    rotation_frame,
    shadow_cohorts,
)
from ecs.rotation import (
    EXTENDED_SOURCES,
    CorpusIndex,
    class_keys,
    corpus_index,
    load_waveforms,
)
from ecs.transfer import ALPHA, METHODS, auroc_row, conformal_sets, coverage_row

ROTATION_SIX = Path(RESULTS_DIR) / "rotation_six"
SCORES = Path(RESULTS_DIR) / "mimic_shadow" / "scores"
OUT_JSON = Path(RESULTS_DIR) / "mimic_shadow.json"
OUT_CSV = Path(RESULTS_DIR) / "mimic_shadow.csv"
CHUNK = 500
HEADLINE_MODEL = "mimic"
HEADLINE_METHOD = "perlabel"

# Fit cohort -> evaluation cohort, per scheme.
SCHEMES = {
    "patient_split": ("patient_cal", "patient_test"),
    "same_era": ("early_cal", "early_test"),
    "shadow": ("early_cal", "late_test"),
}

# The open demo holds 100 patients and the credentialed table all of MIMIC-IV's.
# Below this, --eras anchor is refused rather than run on a handful of patients.
MIN_PATIENTS_FOR_ANCHOR = 100_000


def commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()


def demo_check(
    measures: pd.DataFrame, estimate: pd.Series, cuts: tuple[float, float]
) -> dict[str, Any]:
    """The open estimate against the real year groups of the open demo patients."""
    if not MIMIC_PATIENTS_CSV.exists():
        return {"available": False, "path": str(MIMIC_PATIENTS_CSV)}
    patients = pd.read_csv(MIMIC_PATIENTS_CSV)
    table = patients.set_index(patients["subject_id"])
    seen = measures[measures["subject_id"].isin(table.index)]
    rows = table.loc[seen["subject_id"]]
    shifted = pd.to_datetime(seen["ecg_time"]).dt.year.to_numpy()
    low, high = real_year_bounds(
        pd.Series(shifted, index=rows.index), rows["anchor_year"], rows["anchor_year_group"]
    )
    middle = (low.to_numpy() + high.to_numpy()) / 2.0
    placed = estimate.loc[seen["study_id"].astype(str)].to_numpy()
    inside = (middle >= 2008) & (middle <= 2019)
    era = np.where(placed <= cuts[0], "early", np.where(placed >= cuts[1], "late", "middle"))
    frame = pd.DataFrame({"era": era[inside], "year": middle[inside]})
    by_era = frame.groupby("era")["year"].agg(["size", "mean", "std"]).round(2)
    rho = pd.Series(placed[inside]).corr(pd.Series(middle[inside]), method="spearman")
    return {
        "available": True,
        "path": str(MIMIC_PATIENTS_CSV),
        "n_patients": int(seen["subject_id"].nunique()),
        "n_tracings": int(inside.sum()),
        "n_tracings_clock_outside_2008_2019": int((~inside).sum()),
        "real_year": (
            "middle of the anchor-group bounds; the group spans three years, so the "
            "reference itself is uncertain by about 1.5 years"
        ),
        "spearman_estimate_vs_real_year": round(float(rho), 3),
        "real_year_by_estimated_era": {
            str(e): {
                "n_tracings": int(r["size"]),
                "mean": float(r["mean"]),
                "sd": float(r["std"]),
            }
            for e, r in by_era.iterrows()
        },
    }


def score_cohort(
    index_frame: pd.DataFrame, name: str, ids: list[str], scorers: dict[str, Scorer]
) -> dict[str, tuple[NDArray[np.int_], NDArray[np.float64], list[str]]]:
    """Every model's probabilities for one cohort, read once, written per model."""
    index = CorpusIndex("mimic", index_frame, [])
    kept: list[str] = []
    probabilities: dict[str, list[NDArray[np.float64]]] = {s: [] for s in scorers}
    for start in range(0, len(ids), CHUNK):
        x, got = load_waveforms(index, ids[start : start + CHUNK])
        kept.extend(got)
        for source, scorer in scorers.items():
            probabilities[source].append(scorer.probabilities(x))
    y = index.labels(kept)
    out = {}
    for source in scorers:
        p = np.concatenate(probabilities[source])
        directory = SCORES / source
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / f"{name}.npz",
            ids=np.asarray(kept),
            y=y,
            p=p.astype(np.float32),
            classes=np.asarray(class_keys()),
            scored_by=source,
            cohort=name,
        )
        out[source] = (y, p, kept)
    print(f"  {name}: {len(kept)} / {len(ids)} read and scored", flush=True)
    return out


def rows_for(
    scores: dict[str, dict[str, tuple[NDArray[np.int_], NDArray[np.float64], list[str]]]],
) -> list[dict[str, Any]]:
    rows = []
    for model in EXTENDED_SOURCES:
        for scheme, (fit, spend) in SCHEMES.items():
            y_cal, p_cal, _ = scores[fit][model]
            y_tgt, p_tgt, _ = scores[spend][model]
            for j, key in enumerate(class_keys()):
                sets = conformal_sets(p_cal[:, j], y_cal[:, j], p_tgt[:, j], ALPHA)
                auc = auroc_row(y_tgt[:, j], p_tgt[:, j])["auroc"]
                for method in METHODS:
                    row = coverage_row(sets[method], y_tgt[:, j])
                    rows.append(
                        {
                            "model": model,
                            "scheme": scheme,
                            "fit_on": fit,
                            "spent_on": spend,
                            "label": key,
                            "method": method,
                            "n": row["n"],
                            "n_pos": row["n_pos"],
                            "n_cal_pos": int(y_cal[:, j].sum()),
                            "prevalence": round(row["n_pos"] / row["n"], 5),
                            "auroc": None if auc is None else round(auc, 4),
                            **{
                                k: None if row[k] is None else round(row[k], 4)
                                for k in (
                                    "coverage_pos",
                                    "coverage_pos_low",
                                    "coverage_pos_high",
                                    "coverage_neg",
                                    "coverage_neg_low",
                                    "coverage_neg_high",
                                    "abstention",
                                )
                            },
                        }
                    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eras", choices=("estimated", "anchor"), default="estimated")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu or mps")
    args = parser.parse_args()
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    started = time.time()

    measures = pd.read_csv(
        MIMIC_ECG_DIR / "machine_measurements.csv",
        usecols=["subject_id", "study_id", "cart_id", "ecg_time", "bandwidth"],
    )
    index = corpus_index("mimic")
    frame = index.frame
    records, _ = rotation_frame(MIMIC_ECG_DIR)
    chronology = cart_chronology(measures)
    estimate = estimated_years(measures, chronology)
    own = estimate.loc[frame.index]
    cuts = (float(own.quantile(1 / 3)), float(own.quantile(2 / 3)))
    if args.eras == "anchor":
        patients = pd.read_csv(MIMIC_PATIENTS_CSV)
        if len(patients) < MIN_PATIENTS_FOR_ANCHOR:
            raise SystemExit(
                f"{MIMIC_PATIENTS_CSV} holds {len(patients)} patients: the anchor eras need "
                "the credentialed MIMIC-IV patients table, not the demo"
            )
        era = real_eras(records, patients)
    else:
        era = estimated_eras(own)
    eligible = frame.index[frame["part"].isin(["cal", "test", "unused"])]
    cohorts = shadow_cohorts(eligible, era)
    cohorts["patient_cal"] = index.ids("cal")
    cohorts["patient_test"] = index.ids("test")

    scorers = {s: Scorer(s, ROTATION_SIX, args.device) for s in EXTENDED_SOURCES}
    scores = {name: score_cohort(frame, name, ids, scorers) for name, ids in cohorts.items()}
    rows = rows_for(scores)
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    bandwidth = measures.set_index(measures["study_id"].astype(str))["bandwidth"]
    composition = {}
    for name, ids in cohorts.items():
        y = scores[name][HEADLINE_MODEL][0]
        kept = scores[name][HEADLINE_MODEL][2]
        composition[name] = {
            "n_requested": len(ids),
            "n_read": len(kept),
            "prevalence": {k: round(float(y[:, j].mean()), 4) for j, k in enumerate(class_keys())},
            "bandwidth_share": {
                str(k): round(float(v), 4)
                for k, v in bandwidth.loc[kept].value_counts(normalize=True).items()
            },
            "estimated_years_mean": round(float(own.loc[kept].mean()), 3),
        }

    headline = [r for r in rows if r["model"] == HEADLINE_MODEL and r["method"] == HEADLINE_METHOD]
    table = {
        "written_by": "scripts/mimic_shadow.py",
        "commit": commit(),
        "eras": args.eras,
        "era_definition": (
            "terciles of each tracing's estimated place in time, from the cart chronology"
            if args.eras == "estimated"
            else "real year surely in 2008-2011 (early) or 2016-2019 (late), from anchor_year_group"
        ),
        "settings": {
            "alpha": ALPHA,
            "methods": list(METHODS),
            "schemes": {k: {"fit_on": f, "spent_on": s} for k, (f, s) in SCHEMES.items()},
            "sizes": SHADOW_SIZES,
            "interval": "Wilson score interval, 95%",
            "headline": {"model": HEADLINE_MODEL, "method": HEADLINE_METHOD},
        },
        "cart_chronology": {
            "n_carts": int(chronology.position.notna().sum()),
            "n_patients_used": chronology.n_patients_used,
            "residual_sd_years": round(chronology.residual_sd_years, 3),
            "span_of_cart_positions_years": round(
                float(chronology.position.max() - chronology.position.min()), 2
            ),
            "tercile_cuts_years": [round(c, 3) for c in cuts],
        },
        "check_against_demo_patients": demo_check(measures, estimate, cuts),
        "cohorts": composition,
        "headline_rows": headline,
        "grid": {"file": "results/mimic_shadow.csv", "columns": list(rows[0])},
        "seconds": round(time.time() - started),
    }
    OUT_JSON.write_text(json.dumps(table, indent=1) + "\n")
    for r in headline:
        print(
            f"{r['scheme']:>13} {r['label']:5s} cov+ {r['coverage_pos']} "
            f"[{r['coverage_pos_low']}, {r['coverage_pos_high']}] n+ {r['n_pos']}"
        )
    print(f"wrote {OUT_JSON} in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
