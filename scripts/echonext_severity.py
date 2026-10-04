"""EchoNext: is the outpatients' lost coverage a matter of milder disease?

Among the ECGs with the composite, the inpatients who calibrate the thresholds
and the outpatients who test them are compared on echocardiographic severity;
then the outpatients' coverage of the ill is read inside severity strata and
reweighted to the inpatients' severity mix, for the three strongest arms, with
the per-label thresholds exactly as ``scripts/echonext_transfer.py`` fits them.

Reads the stored scores under ``$ECS_ECHONEXT_DERIVED/scores`` and refuses to
run without them rather than refit a model.  Writes
``results/echonext_severity.json``: counts and aggregates only.

Usage: .venv/bin/python scripts/echonext_severity.py
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ecs.config import REPO_ROOT, RESULTS_DIR
from ecs.echonext import (
    COMPOSITE,
    DERIVED_DIR,
    DISTRIBUTION,
    LABELS,
    read_metadata,
    transfer_cohorts,
)
from ecs.provenance import head_commit
from ecs.severity import (
    BOOTSTRAP_DRAWS,
    CONTINUOUS,
    GRADES,
    LVEF_BANDS,
    MIN_STRATUM,
    MODERATE,
    VERDICT_RULE,
    bootstrap_reweighted,
    compare,
    findings_count,
    graded,
    lvef_band,
    reweighted,
    severity_cell,
    stratum_rows,
    verdict,
)
from ecs.transfer import ALPHA, conformal_sets

ARMS = ("resnet", "echonext_mini", "ecgfounder")
FINDINGS = LABELS[:-1]
SCORES_DIR = DERIVED_DIR / "scores"
COUNT_ORDER = ("1", "2+")
BAND_ORDER = tuple(name for _, name in LVEF_BANDS)


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


def describe(inpatient: pd.DataFrame, outpatient: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "findings_count": compare(
            pd.Series(findings_count(inpatient, FINDINGS)),
            pd.Series(findings_count(outpatient, FINDINGS)),
        )
    }
    for column in CONTINUOUS:
        out[column] = compare(inpatient[column], outpatient[column])
    for column in GRADES:
        a, b = graded(inpatient[column], column), graded(outpatient[column], column)
        out[column] = compare(a, b) | {
            "grades": GRADES[column],
            "moderate_from_grade": MODERATE[column],
            "share_moderate_or_worse": {
                "inpatient": float((a.dropna() >= MODERATE[column]).mean()),
                "outpatient": float((b.dropna() >= MODERATE[column]).mean()),
            },
        }
    return out


def count_stratum(meta: pd.DataFrame) -> NDArray[np.str_]:
    out: NDArray[np.str_] = np.where(findings_count(meta, FINDINGS) >= 2, "2+", "1")
    return out


def measure(meta: pd.DataFrame) -> dict[str, Any]:
    """Every figure of the results file but the commit, from the metadata and stored scores."""
    cohorts = transfer_cohorts(meta, "inpatient", ("outpatient",))
    cal, out = cohorts.calibration, cohorts.targets["outpatient"]
    y = meta[COMPOSITE].to_numpy(dtype=int)
    ill_cal, ill_out = cal[y[cal] == 1], out[y[out] == 1]
    inpatient, outpatient = meta.iloc[ill_cal], meta.iloc[ill_out]

    cell_in = severity_cell(findings_count(inpatient, FINDINGS), inpatient["lvef_value"].to_numpy())
    cell_out = severity_cell(
        findings_count(outpatient, FINDINGS), outpatient["lvef_value"].to_numpy()
    )
    cell_order = tuple(sorted(set(cell_in.tolist())))
    count_in, count_out = count_stratum(inpatient), count_stratum(outpatient)
    band_in = lvef_band(inpatient["lvef_value"].to_numpy())
    band_out = lvef_band(outpatient["lvef_value"].to_numpy())

    result: dict[str, Any] = {
        "distribution": DISTRIBUTION,
        "alpha": ALPHA,
        "verdict_rule": VERDICT_RULE,
        "min_stratum_to_judge": MIN_STRATUM,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "cohorts": {
            "inpatient": {
                "cohort": "EchoNext validation split, inpatients, composite present",
                "n_cohort": int(len(cal)),
                "n_ill": int(len(ill_cal)),
            },
            "outpatient": {
                "cohort": "EchoNext test split, outpatients, composite present",
                "n_cohort": int(len(out)),
                "n_ill": int(len(ill_out)),
            },
        },
        "severity": describe(inpatient, outpatient),
        "cells": {
            name: {
                "inpatient": int((cell_in == name).sum()),
                "outpatient": int((cell_out == name).sum()),
            }
            for name in cell_order
        },
        "arms": {},
    }
    for arm in ARMS:
        p = composite_scores(arm, meta)
        sets = conformal_sets(p[cal], y[cal], p[np.concatenate([cal, out])], ALPHA)["perlabel"]
        covered_cal, covered_out = sets[: len(cal), 1], sets[len(cal) :, 1]
        ill_in = covered_cal[y[cal] == 1]
        ill = covered_out[y[out] == 1]
        row = bootstrap_reweighted(ill, cell_out, cell_in)
        result["arms"][arm] = {
            "coverage_ill_outpatient": float(ill.mean()),
            "coverage_ill_calibration_in_sample": float(ill_in.mean()),
            "by_findings": {
                "outpatient": stratum_rows(ill, count_out, COUNT_ORDER),
                "inpatient_in_sample": stratum_rows(ill_in, count_in, COUNT_ORDER),
            },
            "by_lvef": {
                "outpatient": stratum_rows(ill, band_out, BAND_ORDER),
                "inpatient_in_sample": stratum_rows(ill_in, band_in, BAND_ORDER),
            },
            "by_cell": {
                "outpatient": stratum_rows(ill, cell_out, cell_order),
                "inpatient_in_sample": stratum_rows(ill_in, cell_in, cell_order),
            },
            "reweighted": row,
            "reweighted_by_findings_only": reweighted(ill, count_out, count_in),
            "reweighted_by_lvef_only": reweighted(ill, band_out, band_in),
            **verdict(row),
        }
    verdicts = {r["verdict"] for r in result["arms"].values()}
    result["verdict"] = verdicts.pop() if len(verdicts) == 1 else "differs between models"
    return result


def main() -> None:
    result = measure(read_metadata())
    result["commit"] = head_commit()
    path = RESULTS_DIR / "echonext_severity.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
