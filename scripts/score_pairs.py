"""Every source-target pair the study holds scores for, one diagnosis at a time.

Three families.  ``infarction``: the PTB-XL baseline, its fold 10 as source,
Shandong and Chongqing as targets.  ``rotation``: five corpora, each model's
calibration part as source and every corpus's test part as target, its own
included, on five rhythm and conduction diagnoses.  ``echonext``: four models,
the inpatient ECGs of EchoNext's validation split as source and the inpatient,
emergency and outpatient ECGs of its test split as targets, on eleven findings
and their composite.

The first two read only files this repository carries.  The third needs
EchoNext's metadata and the per-record scores kept beside it, under its
licence; when they are absent the family is skipped and the caller is told.
Patients are attached on request, for the repairs that cut a target in halves
by patient: they come from each corpus's own table, on disk beside the corpus.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ecs.config import ACS_DIR, PTBXL_DIR, RESULTS_DIR, SPH_DIR

ROTATION = ("ptbxl", "sph", "chapman_ningbo", "georgia", "cpsc")
# Records of Georgia, CPSC and Chapman-Ningbo carry no patient identifier; the
# study's own index treats each record as its own patient, and so does this.
ONE_RECORD_PER_PATIENT = ("chapman_ningbo", "georgia", "cpsc")


@dataclass(frozen=True)
class Pair:
    family: str
    model: str
    source: str
    target: str
    label: str
    p_source: NDArray[np.float64]
    y_source: NDArray[np.int_]
    p_target: NDArray[np.float64]
    y_target: NDArray[np.int_]
    target_patients: pd.Series | None

    @property
    def in_distribution(self) -> bool:
        """The target is drawn from the source's own population."""
        return self.source == self.target


def _patients(corpus: str, ids: list[str]) -> pd.Series:
    if corpus in ONE_RECORD_PER_PATIENT:
        return pd.Series(ids, index=ids)
    if corpus == "ptbxl":
        table = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
        return pd.Series([str(table.loc[int(i), "patient_id"]) for i in ids], index=ids)
    if corpus == "sph":
        table = pd.read_csv(SPH_DIR / "metadata.csv", index_col="ECG_ID")
        return pd.Series([str(table.loc[i, "Patient_ID"]) for i in ids], index=ids)
    if corpus == "acs":
        table = pd.read_csv(ACS_DIR / "CSV/train.csv")
        key = table["ecg_row_record"].str.removesuffix(".dat")
        by_record = dict(zip(key, table["Patient_id"].astype(str), strict=True))
        return pd.Series([by_record[i] for i in ids], index=ids)
    raise ValueError(f"no patient table for {corpus}")


def infarction(results: Path = RESULTS_DIR, patients: bool = False) -> Iterator[Pair]:
    with np.load(results / "baseline/scores.npz") as source:
        p_s, y_s = source["probs"][:, 1].astype(np.float64), source["labels"].astype(int)
    for corpus in ("sph", "acs"):
        with np.load(results / f"external/{corpus}.npz") as target:
            ids = [str(i) for i in target["ids"]]
            p_t, y_t = target["probs"][:, 1].astype(np.float64), target["labels"].astype(int)
        yield Pair(
            "infarction",
            "ptbxl_baseline",
            "ptbxl",
            corpus,
            "MI",
            p_s,
            y_s,
            p_t,
            y_t,
            _patients(corpus, ids) if patients else None,
        )


def rotation(results: Path = RESULTS_DIR, patients: bool = False) -> Iterator[Pair]:
    cache: dict[str, pd.Series] = {}
    for source in ROTATION:
        directory = results / "rotation" / source / "scores"
        with np.load(directory / f"{source}_cal.npz") as cal:
            classes = [str(c) for c in cal["classes"]]
            p_s, y_s = cal["p"].astype(np.float64), cal["y"].astype(int)
        for target in ROTATION:
            with np.load(directory / f"{target}_test.npz") as test:
                ids = [str(i) for i in test["ids"]]
                p_t, y_t = test["p"].astype(np.float64), test["y"].astype(int)
            if patients and target not in cache:
                cache[target] = _patients(target, ids)
            for k, label in enumerate(classes):
                yield Pair(
                    "rotation",
                    f"resnet_{source}",
                    source,
                    target,
                    label,
                    p_s[:, k],
                    y_s[:, k],
                    p_t[:, k],
                    y_t[:, k],
                    cache[target] if patients else None,
                )


def echonext_available() -> bool:
    from ecs.echonext import DERIVED_DIR, ECHONEXT_DIR, METADATA

    return (ECHONEXT_DIR / METADATA).exists() and (DERIVED_DIR / "scores").is_dir()


def echonext(patients: bool = False) -> Iterator[Pair]:
    from echonext_transfer import ARMS, TARGETS, scores_for

    from ecs.echonext import LABELS, read_metadata, transfer_cohorts

    meta = read_metadata()
    cohorts = transfer_cohorts(meta, "inpatient", TARGETS)
    truth = meta[list(LABELS)].to_numpy(dtype=int)
    keys = meta["patient_key"].astype(str).to_numpy()
    for arm in ARMS:
        probs, _ = scores_for(arm, meta)
        cal = cohorts.calibration
        for context, rows in cohorts.targets.items():
            for k, label in enumerate(LABELS):
                yield Pair(
                    "echonext",
                    arm,
                    "inpatient",
                    context,
                    label,
                    probs[cal, k].astype(np.float64),
                    truth[cal, k],
                    probs[rows, k].astype(np.float64),
                    truth[rows, k],
                    pd.Series(keys[rows]) if patients else None,
                )


def rounded(value: Any, digits: int = 6) -> Any:
    """Significant digits kept: the file is read by people and diffed, not recomputed from."""
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v, digits) for v in value]
    return value
