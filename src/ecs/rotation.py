"""The five corpora of the source rotation, split the same way and read the same way.

Every corpus is cut once, by patient, into four parts that keep their meaning
whichever role the corpus is playing:

``train``   what a model fitted on this corpus learns from;
``val``     what decides when that model stops;
``cal``     what its conformal thresholds are fitted on, used only when this
            corpus is the source;
``test``    where coverage is measured, used whether this corpus is the source
            or one of the four targets.

Holding ``test`` fixed is what makes the rotation readable: a corpus is scored
on exactly the same records at home and away, so a difference between the two is
the model and the threshold, never the sample.

The train part is capped at a common size across corpora, so "which source" is
not read together with "how much data the source had".  Chapman-Shaoxing with
Ningbo has four times the records of Georgia, and without the cap its model
would be the better one for a reason that has nothing to do with the hospital.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .challenge import (
    challenge_small_set_labels,
    join_ptbxl_to_challenge,
    partition_deviations,
    scan_source,
    sph_small_set_labels,
)
from .config import PTBXL_DIR, SAMPLING_RATE_HZ, SPH_DIR, WINDOW_SAMPLES
from .ingest import Record, assemble_corpus, read_or_error, read_sph, read_wfdb
from .small_set import SMALL_SET, refusals
from .splits import patient_split

__all__ = [
    "CAL_CAP",
    "PARTS",
    "SOURCES",
    "TEST_CAP",
    "TRAIN_CAP",
    "VAL_CAP",
    "CorpusIndex",
    "AGE_BANDS",
    "age_band",
    "class_keys",
    "corpus_index",
    "labels_of",
    "load_waveforms",
    "usable_classes",
]

SOURCES: tuple[str, ...] = ("ptbxl", "sph", "chapman_ningbo", "georgia", "cpsc")

# Shares of patients, drawn once per corpus with a fixed seed.  The eval side is
# the larger one: a coverage figure needs test records far more than a model
# needs the last thousand training records.
PARTS: dict[str, float] = {"train": 0.34, "val": 0.06, "cal": 0.20, "test": 0.40}

# Common caps, so no corpus gets a better model or a tighter threshold for being
# bigger.  Every cap is applied by patient, keeping whole patients.  The train
# and calibration caps are set by the smallest corpus, Georgia and CPSC at about
# ten thousand records each; the test cap is looser because test size costs only
# precision, and the same records serve a corpus at home and away.
TRAIN_CAP = 3500
VAL_CAP = 600
CAL_CAP = 2000
TEST_CAP = 8000

SPLIT_SEED = 27  # the task this rotation belongs to


# Age bands, wide enough that a diagnosis still has cases inside one.  PTB-XL
# de-identifies every patient older than 89 by recording the age as 300, so
# those 293 records land in the oldest band and the substitution is named in the
# corpus's deviations rather than passed off as a real age.
AGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<50", -np.inf, 50.0),
    ("50-64", 50.0, 65.0),
    ("65-74", 65.0, 75.0),
    (">=75", 75.0, np.inf),
)
PTBXL_AGE_PLACEHOLDER = 300.0


def age_band(age: float) -> str:
    """Which band an age falls in, or ``unknown`` when the corpus does not say."""
    if age is None or not np.isfinite(age):
        return "unknown"
    for name, low, high in AGE_BANDS:
        if low <= age < high:
            return name
    return "unknown"


def _sex(value: object) -> str:
    """One vocabulary for three conventions: male, female, or unknown.

    PTB-XL codes sex 0 and 1 in its own database; that 0 is male and 1 is female
    is not assumed here but read off the Challenge bundle's own header for the
    same 21,799 records, which agrees on every one of them.
    """
    text = str(value).strip().lower()
    if text in {"0", "m", "male"}:
        return "male"
    if text in {"1", "f", "female"}:
        return "female"
    return "unknown"


@dataclass
class CorpusIndex:
    """One corpus, ready to be read: a row per record with its patient, part and labels."""

    name: str
    frame: pd.DataFrame  # index: record id; columns: patient, part, one per class
    deviations: list[str]

    def ids(self, part: str) -> list[str]:
        return [str(i) for i in self.frame.index[self.frame["part"] == part]]

    def labels(self, ids: Sequence[str]) -> NDArray[np.int_]:
        """(n, 5) of 0/1 in the order of ``class_keys()``."""
        rows = self.frame.loc[list(ids), class_keys()]
        return rows.to_numpy().astype(int)


def class_keys() -> list[str]:
    return [klass.key for klass in SMALL_SET]


def usable_classes(corpus: str) -> list[str]:
    """The classes this corpus can carry, refusals removed.

    A refused class is not a class with no positives: it is a class the mapping
    would have to invent to fill.  It is dropped from the corpus's rows of the
    table rather than reported as zero coverage.
    """
    refused = {key for name, key in refusals() if name == corpus}
    return [key for key in class_keys() if key not in refused]


def _capped(frame: pd.DataFrame, part: str, cap: int, rng: np.random.Generator) -> pd.Series:
    """Keep at most ``cap`` records of this part, dropping whole patients."""
    inside = frame["part"] == part
    if int(inside.sum()) <= cap:
        return inside
    patients = frame.loc[inside, "patient"].drop_duplicates().to_numpy()
    order = rng.permutation(len(patients))
    keep: set[object] = set()
    total = 0
    counts = frame.loc[inside, "patient"].value_counts()
    for index in order:
        patient = patients[index]
        size = int(counts[patient])
        if total + size > cap:
            continue
        keep.add(patient)
        total += size
        if total == cap:
            break
    return inside & frame["patient"].isin(keep)


def _split(frame: pd.DataFrame) -> pd.DataFrame:
    """Draw the four parts by patient, then cap each one."""
    frame = frame.copy()
    frame["part"] = patient_split(frame["patient"], PARTS, SPLIT_SEED)
    rng = np.random.default_rng(SPLIT_SEED)
    caps = {"train": TRAIN_CAP, "val": VAL_CAP, "cal": CAL_CAP, "test": TEST_CAP}
    keep = pd.Series(False, index=frame.index)
    for part, cap in caps.items():
        keep |= _capped(frame, part, cap, rng)
    frame.loc[~keep, "part"] = "unused"
    return frame


def corpus_index(corpus: str) -> CorpusIndex:
    """The index of one corpus: patients, parts, labels, and what it deviates on."""
    if corpus == "ptbxl":
        database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
        challenge = scan_source("ptbxl")
        joined, _ = join_ptbxl_to_challenge(database, challenge)
        labels = challenge_small_set_labels(joined)
        frame = pd.DataFrame(
            {
                "patient": joined["patient_id"].astype(str).to_numpy(),
                "path": joined["filename_hr"].to_numpy(),
            },
            index=[str(i) for i in joined.index],
        )
        ages = database.loc[joined.index, "age"].to_numpy(dtype=float)
        n_placeholder = int((ages == PTBXL_AGE_PLACEHOLDER).sum())
        ages = np.where(ages == PTBXL_AGE_PLACEHOLDER, 90.0, ages)
        frame["age"] = ages
        frame["sex"] = [_sex(v) for v in database.loc[joined.index, "sex"]]
        deviations = [
            "waveforms and patient identifiers from the PTB-XL distribution, SNOMED codes "
            "from the Challenge-2021 bundle's PTB-XL partition"
        ]
        if n_placeholder:
            deviations.append(
                f"{n_placeholder} patients older than 89 are recorded as age 300 by PTB-XL's "
                "de-identification and are read into the oldest band"
            )
    elif corpus == "sph":
        metadata = pd.read_csv(SPH_DIR / "metadata.csv")
        labels = sph_small_set_labels(metadata)
        frame = pd.DataFrame(
            {
                "patient": metadata["Patient_ID"].astype(str).to_numpy(),
                "path": [str(i) for i in metadata["ECG_ID"]],
            },
            index=[str(i) for i in metadata["ECG_ID"]],
        )
        frame["age"] = metadata["Age"].to_numpy(dtype=float)
        frame["sex"] = [_sex(v) for v in metadata["Sex"]]
        deviations = [
            "diagnoses are AHA codes bridged to SNOMED by Leinonen et al.'s table",
            "records run from ten to sixty seconds; the first ten are kept",
        ]
    else:
        table = scan_source(corpus)
        # Georgia ships 52 records of five seconds and CPSC 22 that fall short of
        # ten by as little as one sample. They cannot reach the canonical window,
        # so they leave before the split rather than thinning a part unevenly
        # once it is drawn.
        seconds = WINDOW_SAMPLES / SAMPLING_RATE_HZ
        short = table["n_samples"] < table["sampling_rate_hz"] * seconds
        dropped = int(short.sum())
        table = table[~short]
        labels = challenge_small_set_labels(table)
        frame = pd.DataFrame(
            {"patient": [str(i) for i in table.index], "path": table["path"].to_numpy()},
            index=[str(i) for i in table.index],
        )
        frame["age"] = table["age"].to_numpy(dtype=float)
        frame["sex"] = [_sex(v) for v in table["sex"]]
        deviations = partition_deviations(corpus, table)
        if dropped:
            deviations.append(
                f"{dropped} records shorter than ten seconds, dropped before the split"
            )
    for key in class_keys():
        frame[key] = labels[key].to_numpy().astype(bool)
    frame["age_band"] = [age_band(float(a)) for a in frame["age"]]
    return CorpusIndex(corpus, _split(frame), deviations)


def _read(corpus: str, path: object) -> Record | Exception:
    if corpus == "sph":
        return read_or_error(read_sph, SPH_DIR / "records" / f"{path}.h5")
    if corpus == "ptbxl":
        return read_or_error(read_wfdb, PTBXL_DIR / str(path))
    return read_or_error(read_wfdb, Path(str(path)))


def _records(index: CorpusIndex, ids: Sequence[str]) -> Iterator[tuple[str, Record | Exception]]:
    for record_id in ids:
        yield record_id, _read(index.name, index.frame.loc[record_id, "path"])


def load_waveforms(index: CorpusIndex, ids: Sequence[str]) -> tuple[NDArray[np.float32], list[str]]:
    """Canonical (n, 12, 5000) float32 for these records, and the ids that survived."""
    corpus = assemble_corpus(index.name, _records(index, ids), len(ids))
    return corpus.x, corpus.ids


def labels_of(index: CorpusIndex, ids: Sequence[str]) -> NDArray[np.int_]:
    return index.labels(ids)
