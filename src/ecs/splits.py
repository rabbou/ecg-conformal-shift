"""Patient-level splits: every record of a patient lands on one side of a boundary.

The coverage guarantee is a statement about exchangeable patients.  Two
tracings of the same patient, one used to calibrate and one used to test,
amount to one draw from the population seen twice, which inflates the measured
coverage.  Splits are therefore drawn over patients, never over records, and
the records follow their patient.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "HALVES",
    "calibration_half",
    "calibration_halves",
    "patient_split",
    "ptbxl_benchmark_split",
    "resample_patients",
]

# A calibration sample and an evaluation sample of equal size, by patient.
HALVES = {"calibration": 0.5, "test": 0.5}


def calibration_half(patients: pd.Series, seed: int) -> NDArray[np.bool_]:
    """One calibration half, drawn by patient: True for the records it holds."""
    return (patient_split(patients, HALVES, seed) == "calibration").to_numpy()


def calibration_halves(
    patients: NDArray[Any], draws: int, seed: int
) -> Iterator[NDArray[np.bool_]]:
    """The ``draws`` calibration halves of one cohort, draw ``d`` seeded ``seed + d``.

    Every table that re-draws a calibration sample reads this sequence, so two
    tables built on the same cohort and seed hold the same draws, one for one.
    """
    keys = pd.Series(np.asarray(patients))
    for draw in range(draws):
        yield calibration_half(keys, seed + draw)


def resample_patients(
    patients: NDArray[Any], rng: np.random.Generator
) -> tuple[NDArray[np.int_], NDArray[np.str_]]:
    """One bootstrap replicate of a cohort, drawn by patient with replacement.

    Returns the rows of the replicate and a patient key for each row.  A patient
    drawn twice gets two keys, one per draw, so a split of the replicate treats
    the two copies as the two patients they now stand for.
    """
    patients = np.asarray(patients)
    unique = np.unique(patients)
    index_of = {p: np.flatnonzero(patients == p) for p in unique}
    picked = rng.choice(unique, size=unique.size, replace=True)
    rows = np.concatenate([index_of[p] for p in picked])
    keys = np.concatenate([np.full(index_of[p].size, str(i)) for i, p in enumerate(picked)])
    return rows, keys


def patient_split(patients: pd.Series, fractions: Mapping[str, float], seed: int) -> pd.Series:
    """The part each record belongs to, drawn by patient.

    ``patients`` holds one patient id per record (its index is the record key).
    ``fractions`` maps part name to its share of patients and must sum to one.
    The distinct patients are sorted, shuffled with ``seed``, and dealt to the
    parts in order, so the same patients and seed always give the same
    assignment whatever the order of the records.
    """
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"fractions sum to {total}, not 1")
    ids = np.sort(patients.unique())
    ids = ids[np.random.default_rng(seed).permutation(len(ids))]
    cuts = np.floor(np.cumsum(list(fractions.values())) * len(ids)).astype(int)
    cuts[-1] = len(ids)
    part_of: dict[object, str] = {}
    start = 0
    for name, stop in zip(fractions, cuts, strict=True):
        part_of.update(dict.fromkeys(ids[start:stop], name))
        start = stop
    return patients.map(part_of).rename("part")


def ptbxl_benchmark_split(database: pd.DataFrame) -> pd.Series:
    """The split PTB-XL ships for benchmarking: ``strat_fold`` 1-8 train, 9
    validation, 10 test (Strodthoff et al. 2020).  The folds were drawn by
    patient at source, which ``test_splits.py`` checks rather than assumes."""
    fold = database["strat_fold"]
    part = pd.Series("train", index=database.index, name="part")
    part[fold == 9] = "validation"
    part[fold == 10] = "test"
    return part
