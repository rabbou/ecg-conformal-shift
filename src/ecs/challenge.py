"""The PhysioNet/CinC Challenge-2021 partitions, read where they sit on disk.

Four of the five rotation sources come out of this bundle: PTB-XL,
Chapman-Shaoxing with Ningbo, Georgia, and CPSC 2018 with its extension.  Every
record is a WFDB pair whose header carries the diagnoses as SNOMED CT codes on a
``# Dx:`` line, so the labels need no mapping of mine -- only the union that
``small_set`` defines.

Two things the bundle does not ship, and neither is guessed:

*Patient identifiers.*  A Challenge header has an age and a sex and no patient
key.  Splitting by patient (C-4) therefore treats each record as its own
patient, which is right for Chapman-Shaoxing, Ningbo, Georgia and CPSC -- one
tracing per patient by construction -- and wrong for PTB-XL, whose 21,799
records come from 18,869 patients.  PTB-XL is the one corpus with a published
patient key, so its waveforms and its patients are read from the distribution
itself and only its SNOMED codes come from here.  The deviation is named in
``partition_deviations`` for every corpus that carries it.

*A count that matches the distribution.*  The bundle froze PTB-XL at 21,837
records; the distribution this study reads is v1.0.3, at 21,799.  The join keeps
the intersection and reports what fell out on either side.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CHALLENGE2021_DIR, MAPPINGS_DIR
from .small_set import SMALL_SET, aha_codes_for, refusals, snomed_codes_for

__all__ = [
    "CHALLENGE_PARTITIONS",
    "completeness",
    "distributed_records",
    "PTBXL_RECORD_PREFIX",
    "Header",
    "challenge_small_set_labels",
    "join_ptbxl_to_challenge",
    "partition_deviations",
    "read_header",
    "scan_source",
    "sph_small_set_labels",
]

# The Challenge directory names each rotation source is assembled from.
CHALLENGE_PARTITIONS: dict[str, tuple[str, ...]] = {
    "ptbxl": ("ptb-xl",),
    "chapman_ningbo": ("chapman_shaoxing", "ningbo"),
    "georgia": ("georgia",),
    "cpsc": ("cpsc_2018", "cpsc_2018_extra"),
}

# The bundle names each PTB-XL record HR<ecg_id zero-padded to five digits>.
# `test_challenge.py` checks that against the waveform rather than assuming it.
PTBXL_RECORD_PREFIX = "HR"

NO_PATIENT_KEY = (
    "the Challenge bundle ships no patient identifier, so each record is its own "
    "patient for splitting"
)


@dataclass(frozen=True)
class Header:
    """What one ``.hea`` says: the record, its shape, and its diagnoses."""

    record_id: str
    path: Path
    dx: tuple[str, ...]
    sampling_rate_hz: int
    n_samples: int
    age: float
    sex: str


def read_header(path: Path) -> Header:
    """Parse one Challenge header without touching its samples."""
    lines = path.read_text(errors="replace").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty header")
    fields = lines[0].split()
    if len(fields) < 4:
        raise ValueError(f"{path}: first line is not a WFDB record line: {lines[0]!r}")
    dx: tuple[str, ...] = ()
    age = float("nan")
    sex = "Unknown"
    for line in lines:
        if line.startswith("# Dx:"):
            dx = tuple(c.strip() for c in line.removeprefix("# Dx:").split(",") if c.strip())
        elif line.startswith("# Age:"):
            raw = line.removeprefix("# Age:").strip()
            age = float(raw) if raw.replace(".", "", 1).isdigit() else float("nan")
        elif line.startswith("# Sex:"):
            sex = line.removeprefix("# Sex:").strip()
    return Header(
        record_id=fields[0],
        path=path.with_suffix(""),
        dx=dx,
        sampling_rate_hz=int(fields[2]),
        n_samples=int(fields[3]),
        age=age,
        sex=sex,
    )


def scan_source(source: str, root: Path = CHALLENGE2021_DIR) -> pd.DataFrame:
    """Every record of one rotation source, indexed by record id.

    Columns: ``partition``, ``path`` (without extension), ``dx`` (a tuple of
    SNOMED CT codes), ``sampling_rate_hz``, ``n_samples``, ``age``, ``sex``.
    """
    rows: list[dict[str, object]] = []
    for partition in CHALLENGE_PARTITIONS[source]:
        directory = root / "training" / partition
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} is not on disk")
        for header_path in sorted(directory.rglob("*.hea")):
            header = read_header(header_path)
            rows.append(
                {
                    "record_id": header.record_id,
                    "partition": partition,
                    "path": header.path,
                    "dx": header.dx,
                    "sampling_rate_hz": header.sampling_rate_hz,
                    "n_samples": header.n_samples,
                    "age": header.age,
                    "sex": header.sex,
                }
            )
    table = pd.DataFrame(rows).set_index("record_id")
    if table.index.has_duplicates:
        duplicates = sorted(table.index[table.index.duplicated()].unique())
        raise ValueError(f"{source}: repeated record ids {duplicates[:5]}")
    return table


def distributed_records(partition: str, root: Path = CHALLENGE2021_DIR) -> set[str]:
    """The record ids the bundle's own SHA256SUMS.txt says this partition holds.

    The manifest ships with the bundle and lists every distributed file, so it is
    the reference for whether the copy on disk is complete -- a question a class
    count cannot answer on its own, because a missing record and a mis-parsed
    label look the same in a total.
    """
    manifest = root / "SHA256SUMS.txt"
    prefix = f"training/{partition}/"
    ids = set()
    for line in manifest.read_text().splitlines():
        _, _, path = line.partition(" ")
        path = path.strip()
        if path.startswith(prefix) and path.endswith(".hea"):
            ids.add(Path(path).stem)
    if not ids:
        raise ValueError(f"{manifest} lists no .hea under {prefix}")
    return ids


def completeness(
    source: str, table: pd.DataFrame, root: Path = CHALLENGE2021_DIR
) -> dict[str, list[str] | int]:
    """What this copy of the bundle holds against what the bundle distributes."""
    on_disk = set(table.index)
    listed: set[str] = set()
    for partition in CHALLENGE_PARTITIONS[source]:
        listed |= distributed_records(partition, root)
    missing = sorted(listed - on_disk)
    return {
        "n_distributed": len(listed),
        "n_on_disk": len(on_disk),
        "missing_from_this_copy": missing,
        "not_in_the_manifest": sorted(on_disk - listed),
    }


def partition_deviations(source: str, table: pd.DataFrame) -> list[str]:
    """What the chain could not make identical for this source (C-14)."""
    out = [NO_PATIENT_KEY]
    rates = sorted({int(r) for r in table["sampling_rate_hz"]})
    if rates != [500]:
        out.append(f"sampling rates present: {rates}")
    lengths = sorted({int(n) for n in table["n_samples"]})
    if len(lengths) > 1:
        out.append(
            f"record lengths run from {min(lengths)} to {max(lengths)} samples; "
            "the first ten seconds are kept"
        )
    return out


def challenge_small_set_labels(table: pd.DataFrame, root: Path = MAPPINGS_DIR) -> pd.DataFrame:
    """One boolean column per small-set class, indexed like ``table``.

    A record is a positive of a class when its ``# Dx:`` line carries any of the
    SNOMED codes that class unions.
    """
    codes = {klass.key: snomed_codes_for(klass.key, root) for klass in SMALL_SET}
    columns = {
        key: table["dx"].apply(lambda dx, want=wanted: bool(want & set(dx)))
        for key, wanted in codes.items()
    }
    return pd.DataFrame(columns, index=table.index)


def sph_small_set_labels(metadata: pd.DataFrame, root: Path = MAPPINGS_DIR) -> pd.DataFrame:
    """One boolean column per small-set class for Shandong, indexed like ``metadata``.

    Shandong writes each diagnosis as an AHA code with an optional ``+modifier``;
    the base code is what carries the diagnosis (ambiguity ``aha-modifier-tokens``).
    A class with no AHA bridge is refused: its column is all-``False`` and the
    pair appears in ``small_set.refusals()``, which every reader of this table has
    to consult before using the column.
    """
    refused = {key for corpus, key in refusals() if corpus == "sph"}
    base = (
        metadata["AHA_Code"]
        .astype(str)
        .apply(
            lambda raw: {
                token.strip().split("+", 1)[0] for token in raw.split(";") if token.strip()
            }
        )
    )
    columns = {}
    for klass in SMALL_SET:
        if klass.key in refused:
            columns[klass.key] = pd.Series(False, index=metadata.index)
            continue
        wanted = aha_codes_for(klass.key, root)
        if not wanted:
            raise ValueError(
                f"{klass.key} has no AHA bridge and is not among the declared refusals"
            )
        columns[klass.key] = base.apply(lambda codes, want=wanted: bool(want & codes))
    return pd.DataFrame(columns, index=metadata.index)


def join_ptbxl_to_challenge(
    database: pd.DataFrame, challenge: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """PTB-XL's own rows carrying the Challenge's SNOMED codes.

    Returns the intersection, indexed by ``ecg_id`` and carrying ``patient_id``,
    ``strat_fold``, ``filename_hr`` and ``dx``, together with what fell out of
    each side.
    """
    ecg_id = pd.Series(
        [int(str(r).removeprefix(PTBXL_RECORD_PREFIX)) for r in challenge.index],
        index=challenge.index,
        name="ecg_id",
    )
    dx_by_ecg_id = pd.Series(list(challenge["dx"]), index=np.asarray(ecg_id), name="dx")
    kept = database.index.intersection(dx_by_ecg_id.index)
    joined = database.loc[kept, ["patient_id", "strat_fold", "filename_hr"]].copy()
    joined["dx"] = dx_by_ecg_id.loc[kept]
    fell_out = {
        "in_the_distribution_not_in_the_bundle": [
            str(i) for i in database.index.difference(dx_by_ecg_id.index)
        ],
        "in_the_bundle_not_in_the_distribution": [
            f"{PTBXL_RECORD_PREFIX}{i:05d}" for i in dx_by_ecg_id.index.difference(database.index)
        ],
    }
    return joined, fell_out
