"""EchoNext read as one more corpus: its files, one provenance row per tracing, its cohorts.

EchoNext (PhysioNet, restricted licence) ships 100,000 Columbia ECGs, each paired
with an echocardiogram, eleven echo-derived flags and their composite.  An ECG
with a flag precedes an abnormal echocardiogram by a year at most; an ECG with
none may precede the patient's last normal echocardiogram by any time.
The tracings are stored already processed: 250 Hz, median-filtered, clipped at
the 0.1st and 99.9th percentiles and standardised per lead with a mean and a
standard deviation computed over the whole training set.  Nothing on disk says
how many microvolts a unit is, so the unit of every tracing is ``z-score`` and a
difference in amplitude between this corpus and a millivolt corpus cannot be
read off the files.

The licence forbids sharing the data (clause 3).  Everything this module writes
per record therefore goes under ``ECS_ECHONEXT_DERIVED`` outside the repository;
only counts and aggregate figures reach ``results/``.

Row order is the contract the distribution states: the rows of one split in the
metadata file, in file order, are the rows of that split's arrays.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "COMPOSITE",
    "CONTEXTS",
    "DERIVED_DIR",
    "DISTRIBUTION",
    "ECHONEXT_DIR",
    "LABELS",
    "LEAD_ORDER",
    "PROVENANCE_FIELDS",
    "UNIT",
    "Cohorts",
    "canonical",
    "iter_blocks",
    "provenance_rows",
    "published_digests",
    "quality_flags",
    "read_metadata",
    "read_rows",
    "tracing_digests",
    "transfer_cohorts",
    "verify_files",
]

ECHONEXT_DIR = Path(os.environ.get("ECS_ECHONEXT_DIR", Path.home() / "data/echonext"))
DERIVED_DIR = Path(os.environ.get("ECS_ECHONEXT_DERIVED", Path.home() / "data/echonext-derived"))

DISTRIBUTION = "physionet.org/content/echonext/1.1.1"
METADATA = "echonext_metadata_100k.csv"
DIGESTS = "SHA256SUMS.txt"
SPLITS = ("train", "val", "test", "no_split")
CONTEXTS = ("inpatient", "emergency", "outpatient", "procedural")
UNIT = "z-score"
SAMPLING_HZ = 250
N_SAMPLES = 2500

# The order the authors' parse_xml.py writes the leads in, which is the order
# of the last axis of every waveform array.
LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

# The eleven echo-derived flags, then the composite that is positive when any
# of them is.  This is the column order of the mini-model's output as well.
LABELS = (
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45_flag",
    "tr_max_gte_32_flag",
    "shd_moderate_or_greater_flag",
)
COMPOSITE = LABELS[-1]

REQUIRED_COLUMNS = (
    "ecg_key",
    "patient_key",
    "split",
    "location_setting",
    "sex",
    "age_at_ecg",
    "race_ethnicity",
    *LABELS,
)

# The six provenance fields of every tracing, as eight columns: the source file
# travels with its published digest, and the quality indicators are two.
# A column missing from a table built here is a failure of this module, and
# tests/test_echonext.py holds the table to this declaration.
PROVENANCE_FIELDS: dict[str, str] = {
    "distribution": "str",
    "source_file": "str",
    "source_sha256": "str",
    "copy_group": "str",
    "quality_flat_leads": "int",
    "quality_non_finite": "bool",
    "unit": "str",
    "lead_order": "str",
}

FLAT_LEAD_STD = 1e-6

Float32Array = NDArray[np.float32]


def published_digests(root: Path = ECHONEXT_DIR) -> dict[str, str]:
    """File name to SHA-256, as the distribution's own SHA256SUMS.txt states it."""
    out: dict[str, str] = {}
    for line in (root / DIGESTS).read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(root: Path = ECHONEXT_DIR) -> dict[str, bool]:
    """Whether each file matches its published digest; a missing file is False."""
    return {
        name: (root / name).exists() and _sha256(root / name) == digest
        for name, digest in published_digests(root).items()
    }


def read_metadata(root: Path = ECHONEXT_DIR) -> pd.DataFrame:
    """The metadata, with ``row``: the record's position inside its split's arrays.

    Raises when a column this study reads is absent, naming every one, since a
    care context or a label that is not there cannot be measured around.
    """
    table = pd.read_csv(root / METADATA, index_col=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{root / METADATA} lacks the columns {missing}")
    table["row"] = table.groupby("split").cumcount()
    return table


def waveform_path(root: Path, split: str) -> Path:
    return root / f"EchoNext_{split}_waveforms.npy"


def _layout(path: Path) -> tuple[tuple[int, ...], np.dtype[Any], int]:
    """Shape, dtype and byte offset of the data in a C-ordered .npy file."""
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        if fortran:
            raise ValueError(f"{path} is Fortran-ordered")
        return shape, dtype, handle.tell()


def read_rows(path: Path, rows: NDArray[np.int_]) -> NDArray[np.float64]:
    """The given rows of a .npy file, in the order asked, by sequential reads.

    Through a memory map every page of this corpus arrives by a fault, and a full
    pass ran at ~23 MB/s; reading runs of consecutive rows into one buffer runs at
    the disk's speed and keeps the resident set to the rows asked for.
    """
    shape, dtype, offset = _layout(path)
    rows = np.asarray(rows, dtype=int)
    if rows.size and (rows.min() < 0 or rows.max() >= shape[0]):
        raise IndexError(f"rows outside 0..{shape[0] - 1} of {path.name}")
    row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    order = np.argsort(rows, kind="stable")
    ordered = rows[order]
    out = np.empty((len(rows), *shape[1:]), dtype=dtype)
    flat = out.reshape(len(rows), -1).view(np.uint8)
    with path.open("rb", buffering=0) as handle:
        start = 0
        while start < len(ordered):
            stop = start + 1
            while stop < len(ordered) and ordered[stop] == ordered[stop - 1] + 1:
                stop += 1
            buffer = bytearray((stop - start) * row_bytes)
            handle.seek(offset + int(ordered[start]) * row_bytes)
            if handle.readinto(buffer) != len(buffer):
                raise OSError(f"{path.name}: short read at row {ordered[start]}")
            run = np.frombuffer(buffer, dtype=np.uint8).reshape(stop - start, row_bytes)
            flat[order[start:stop]] = run
            start = stop
    return out


def iter_blocks(
    root: Path, split: str, block: int = 2048
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """The split's waveforms as stored, ``block`` records at a time, never all at once.

    The training file is 17.4 GB of float64; one block at a time keeps the
    resident set to a block whatever the file size.
    """
    path = waveform_path(root, split)
    n = _layout(path)[0][0]
    for start in range(0, n, block):
        yield start, read_rows(path, np.arange(start, min(start + block, n)))


def canonical(stored: NDArray[np.float64]) -> Float32Array:
    """(N, 1, 2500, 12) float64 as stored, to (N, 12, 2500) float32 leads-first."""
    if stored.ndim != 4 or stored.shape[1:] != (1, N_SAMPLES, len(LEAD_ORDER)):
        raise ValueError(f"expected (N, 1, {N_SAMPLES}, 12), got {stored.shape}")
    return np.ascontiguousarray(stored[:, 0].transpose(0, 2, 1), dtype=np.float32)


def tracing_digests(stored: NDArray[np.float64]) -> list[str]:
    """SHA-256 of each record's stored bytes: the key an embedding is filed under."""
    rows = np.ascontiguousarray(stored)
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in rows]


def quality_flags(x: Float32Array) -> tuple[NDArray[np.int_], NDArray[np.bool_]]:
    """Per record: how many leads are flat, and whether any sample is not finite."""
    non_finite = ~np.isfinite(x).all(axis=(1, 2))
    flat = (np.nan_to_num(x).std(axis=2) < FLAT_LEAD_STD).sum(axis=1)
    return flat.astype(int), non_finite


def provenance_rows(
    meta: pd.DataFrame, root: Path = ECHONEXT_DIR, splits: tuple[str, ...] = SPLITS
) -> pd.DataFrame:
    """One row per tracing: the six provenance fields, the tracing digest, its record.

    ``copy_group`` is the smallest ``ecg_key`` among the records whose stored
    bytes are identical, so a record with no copy is its own group and two
    copies can never land on two sides of a split unnoticed.
    """
    digests = published_digests(root)
    parts = []
    for split in splits:
        name = waveform_path(root, split).name
        rows = meta[meta["split"] == split].sort_values("row")
        stored_digests: list[str] = []
        flat: list[NDArray[np.int_]] = []
        non_finite: list[NDArray[np.bool_]] = []
        for _, stored in iter_blocks(root, split):
            stored_digests.extend(tracing_digests(stored))
            f, n = quality_flags(canonical(stored))
            flat.append(f)
            non_finite.append(n)
        if len(stored_digests) != len(rows):
            raise ValueError(f"{name}: {len(stored_digests)} tracings for {len(rows)} rows")
        parts.append(
            pd.DataFrame(
                {
                    "ecg_key": rows["ecg_key"].to_numpy(),
                    "split": split,
                    "row": rows["row"].to_numpy(),
                    "tracing_sha256": stored_digests,
                    "distribution": DISTRIBUTION,
                    "source_file": name,
                    "source_sha256": digests[name],
                    "quality_flat_leads": np.concatenate(flat),
                    "quality_non_finite": np.concatenate(non_finite),
                    "unit": UNIT,
                    "lead_order": ",".join(LEAD_ORDER),
                }
            )
        )
    table = pd.concat(parts, ignore_index=True)
    table["copy_group"] = table.groupby("tracing_sha256")["ecg_key"].transform("min").astype(str)
    return table


@dataclass(frozen=True)
class Cohorts:
    """Row positions into the metadata for each role of one transfer measurement."""

    training: NDArray[np.int_]
    calibration: NDArray[np.int_]
    targets: dict[str, NDArray[np.int_]]


def transfer_cohorts(
    meta: pd.DataFrame,
    source_context: str = "inpatient",
    target_contexts: tuple[str, ...] = ("inpatient", "emergency", "outpatient"),
) -> Cohorts:
    """Training on the train split, calibration on the source context of val,
    evaluation on each target context of test.

    The guarantee is about exchangeable patients, so a patient seen in two roles
    would be one draw counted twice.  The published splits are drawn by patient;
    this checks it rather than trusting it, and raises naming the count, because
    quietly dropping the overlap would change the cohorts a reader compares.
    """
    position = np.arange(len(meta))
    split = meta["split"].to_numpy()
    context = meta["location_setting"].to_numpy()
    training = position[split == "train"]
    calibration = position[(split == "val") & (context == source_context)]
    targets = {c: position[(split == "test") & (context == c)] for c in target_contexts}

    patients = meta["patient_key"].to_numpy()
    roles = {"training": training, "calibration": calibration} | {
        f"target:{c}": rows for c, rows in targets.items()
    }
    names = list(roles)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if a.startswith("target:") and b.startswith("target:"):
                continue  # two target contexts are read side by side, never pooled
            shared = set(patients[roles[a]]) & set(patients[roles[b]])
            if shared:
                raise ValueError(f"{len(shared)} patients are in both {a} and {b}")
    return Cohorts(training=training, calibration=calibration, targets=targets)
