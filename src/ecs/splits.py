"""Patient-level splits: every record of a patient lands on one side of a boundary.

The coverage guarantee is a statement about exchangeable patients.  Two
tracings of the same patient, one used to calibrate and one used to test, are
not two draws from the population but one draw seen twice, and the guarantee
they produce is flattered by exactly that much.  So a split is never drawn
over records here; it is drawn over patients and the records follow.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

__all__ = ["patient_split", "ptbxl_benchmark_split"]


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
