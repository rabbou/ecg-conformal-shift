"""EchoNext inpatients to outpatients: the coverage table, in one command.

Calibration is the inpatient ECGs of the validation split; the targets are the
inpatient, emergency and outpatient ECGs of the test split, read without
refitting.  An arm whose scores are not yet in ``$ECS_ECHONEXT_DERIVED/scores``
is scored first by ``scripts/echonext_scores.py``.

Writes ``results/echonext_transfer.json`` and ``results/echonext_coverage.csv``;
``scripts/echonext_outcomes.py`` reads the first.  The per-record scores stay
outside the repository: they are derived from restricted data.

Usage: PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/echonext_transfer.py
"""

from __future__ import annotations

import json
import subprocess
import sys
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
    SAMPLING_HZ,
    UNIT,
    read_metadata,
    transfer_cohorts,
)
from ecs.provenance import head_commit
from ecs.transfer import (
    ALPHA,
    METHODS,
    auroc_row,
    calibration_row,
    conformal_sets,
    coverage_row,
    curve_rows,
    decision_rows,
    ladder_rows,
    ppv_row,
    subgroup_rows,
)

LVEF = "lvef_lte_45_flag"
TARGETS = ("inpatient", "emergency", "outpatient")
DETAIL_LABELS = (COMPOSITE, LVEF)
SCORES_DIR = DERIVED_DIR / "scores"
CONTEXT_NAMES = {"inpatient": "inpatients", "emergency": "emergency", "outpatient": "outpatients"}

ARMS = {
    "resnet": "the study's ResNet, trained on EchoNext",
    "random_init": "the study's ResNet frozen at random initialisation, with logistic probes",
    "ecgfounder": "ECGFounder frozen, with logistic probes",
    "echonext_mini": "the published EchoNext mini-model",
}


Probs = NDArray[np.float32]


def scores_for(arm: str, meta: pd.DataFrame) -> tuple[Probs, dict[str, Any]]:
    """The arm's (len(meta), 12) probabilities, NaN outside validation and test."""
    path = SCORES_DIR / f"{arm}.npz"
    if not path.exists():
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/echonext_scores.py"), "--arm", arm],
            check=True,
        )
    with np.load(path) as data:
        if tuple(data["labels"].tolist()) != LABELS:
            raise ValueError(f"{path} holds labels {data['labels']}, expected {LABELS}")
        where = pd.Index(meta["ecg_key"]).get_indexer(data["ecg_key"])
        if (where < 0).any():
            raise ValueError(f"{path} names ECGs absent from the metadata")
        probs = np.full((len(meta), len(LABELS)), np.nan, dtype=np.float32)
        probs[where] = data["probs"]
    info = json.loads(path.with_suffix(".json").read_text())
    return probs, {k: v for k, v in info.items() if k != "machine"}


def prevalence(meta: pd.DataFrame, rows: NDArray[np.int_]) -> dict[str, float]:
    return {label: float(meta[label].to_numpy()[rows].mean()) for label in LABELS}


def prevalence_replay(meta: pd.DataFrame) -> dict[str, Any]:
    """Prevalence per care context over validation and test, one ECG per patient."""
    both = meta[meta["split"].isin(["val", "test"])]
    return {
        str(context): {"n": int(len(group)), **{k: float(group[k].mean()) for k in LABELS}}
        for context, group in both.groupby("location_setting")
    }


def measure_arm(
    meta: pd.DataFrame,
    probs: Probs,
    calibration: NDArray[np.int_],
    targets: dict[str, NDArray[np.int_]],
) -> dict[str, Any]:
    labels = meta[list(LABELS)].to_numpy(dtype=int)
    out: dict[str, Any] = {
        "coverage": [],
        "subgroups": [],
        "auroc": {},
        "calibration": {},
        "calibration_curve": {},
        "decision_curve": {},
        "ppv": {},
        "ladder": {},
    }
    test = np.flatnonzero(meta["split"].to_numpy() == "test")
    out["auroc_test_all_contexts"] = {
        COMPOSITE: auroc_row(labels[test, -1], probs[test, -1].astype(np.float64))
    }
    for context, rows in targets.items():
        for key in ("auroc", "calibration", "calibration_curve", "decision_curve", "ppv", "ladder"):
            out[key][context] = {}
        for k, label in enumerate(LABELS):
            p_cal, y_cal = probs[calibration, k].astype(np.float64), labels[calibration, k]
            p_tgt, y_tgt = probs[rows, k].astype(np.float64), labels[rows, k]
            sets = conformal_sets(p_cal, y_cal, p_tgt, ALPHA)
            for method in METHODS:
                out["coverage"].append(
                    {
                        "label": label,
                        "context": context,
                        "method": method,
                        **coverage_row(sets[method], y_tgt),
                    }
                )
            out["auroc"][context][label] = auroc_row(y_tgt, p_tgt)
            out["calibration"][context][label] = calibration_row(y_tgt, p_tgt)
            source_flags = conformal_sets(p_cal, y_cal, p_cal, ALPHA)["plain"][:, 1]
            out["ppv"][context][label] = ppv_row(source_flags, y_cal, sets["plain"][:, 1], y_tgt)
            if label not in DETAIL_LABELS:
                continue
            for method in METHODS:
                for row in subgroup_rows(sets[method], y_tgt, meta.iloc[rows]):
                    out["subgroups"].append(
                        {"label": label, "context": context, "method": method, **row}
                    )
            out["calibration_curve"][context][label] = curve_rows(y_tgt, p_tgt)
            out["decision_curve"][context][label] = decision_rows(y_tgt, p_tgt)
            out["ladder"][context][label] = ladder_rows(
                p_cal, y_cal, p_tgt, y_tgt, meta["patient_key"].iloc[rows]
            )
    return out


def coverage_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"arm": arm, **row}
        for arm, measured in result["arms"].items()
        for row in measured["coverage"]
    ]
    return pd.DataFrame(rows)


def main() -> None:
    meta = read_metadata()
    cohorts = transfer_cohorts(meta, "inpatient", TARGETS)
    patients = meta["patient_key"].to_numpy()
    result: dict[str, Any] = {
        "distribution": DISTRIBUTION,
        "alpha": ALPHA,
        "labels": list(LABELS),
        "input": f"the tracing as distributed, 12 leads at {SAMPLING_HZ} Hz, unit {UNIT}",
        "source": {
            "cohort": "EchoNext validation split, inpatients",
            "n": int(len(cohorts.calibration)),
            "patients": int(len(set(patients[cohorts.calibration]))),
            "prevalence": prevalence(meta, cohorts.calibration),
        },
        "targets": {
            context: {
                "cohort": f"EchoNext test split, {CONTEXT_NAMES[context]}",
                "n": int(len(rows)),
                "patients": int(len(set(patients[rows]))),
                "prevalence": prevalence(meta, rows),
            }
            for context, rows in cohorts.targets.items()
        },
        "training": {"n": int(len(cohorts.training))},
        "patients_shared_between_roles": 0,  # transfer_cohorts raises otherwise
        "prevalence_by_context_val_and_test": prevalence_replay(meta),
        "arms": {},
    }
    for arm, title in ARMS.items():
        probs, info = scores_for(arm, meta)
        measured = measure_arm(meta, probs, cohorts.calibration, cohorts.targets)
        result["arms"][arm] = {
            "title": title,
            "description": info["description"],
            "run": info,
            **measured,
        }
    result["commit"] = head_commit()
    (RESULTS_DIR / "echonext_transfer.json").write_text(json.dumps(result, indent=2) + "\n")
    coverage_table(result).to_csv(RESULTS_DIR / "echonext_coverage.csv", index=False)


if __name__ == "__main__":
    main()
