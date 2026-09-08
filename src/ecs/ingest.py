"""Reducing every corpus to one canonical waveform form.

The form is float32, shape (N, 12, 5000): ten seconds at 500 Hz, in millivolts,
leads ordered I, II, III, aVR, aVL, aVF, V1-V6.  It is the convention of
arXiv:2602.17531, not a house style.

The chain that gets a record there is the same for every corpus -- reorder the
leads by the names the record's header gives them, resample to 500 Hz if the
record is not already there, keep the first ten seconds, cast to float32 -- and
nothing else: no filter, no normalisation.  A difference this pipeline produced
would be indistinguishable from one the hospital produced.  Where a corpus
cannot be made to match, the step is named in ``Corpus.deviations`` rather
than absorbed.

Nothing transformed is written to disk.  The corpora stay exactly as
distributed and every transform happens here, at read time.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import wfdb
from numpy.typing import NDArray
from scipy.signal import resample_poly

from .config import ACS_DIR, N_LEADS, PTBXL_DIR, SAMPLING_RATE_HZ, SPH_DIR, WINDOW_SAMPLES

__all__ = [
    "CANONICAL_LEADS",
    "Corpus",
    "Record",
    "assemble_corpus",
    "canonicalise",
    "iter_acs",
    "iter_ptbxl",
    "iter_sph",
    "load_acs",
    "load_ptbxl",
    "load_sph",
    "read_or_error",
    "read_sph",
    "read_wfdb",
]

CANONICAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

# Shandong's HDF5 files carry no header at all, so the lead order and sampling
# rate cannot be read from the record.  Both are taken from the corpus paper:
# "The order of leads is I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6"
# and "The sampling frequency is 500 Hz" -- Liu et al., Sci Data 2022, Data
# Records section (PMC9174207), read 2026-08-23.
SPH_LEADS = CANONICAL_LEADS
SPH_SAMPLING_RATE_HZ = 500
SPH_DEVIATION = (
    "lead order and sampling rate are not in the record (no HDF5 attributes); "
    "taken from Liu et al. 2022, Data Records"
)


@dataclass(frozen=True)
class Record:
    """One tracing as the corpus ships it, in millivolts, with what its header says."""

    signal: NDArray[np.float32]  # (12, T), millivolts, in the header's lead order
    lead_names: tuple[str, ...]
    sampling_rate_hz: int


@dataclass
class Corpus:
    """The canonical array for one corpus, with what had to give to get there."""

    name: str
    x: NDArray[np.float32]  # (N, 12, 5000)
    ids: list[str]  # one per row of x
    excluded: dict[str, str]  # record id -> why it was dropped
    n_resampled: int = 0  # records that were not at 500 Hz
    n_cropped: int = 0  # records longer than ten seconds
    notes: tuple[str, ...] = ()  # deviations known before any record is read

    @property
    def deviations(self) -> list[str]:
        """Every step the chain could not make identical across corpora (C-14)."""
        n = len(self.ids) + len(self.excluded)
        out = list(self.notes)
        if self.n_resampled:
            out.append(f"{self.n_resampled} of {n} records resampled to {SAMPLING_RATE_HZ} Hz")
        if self.n_cropped:
            out.append(
                f"{self.n_cropped} of {n} records longer than ten seconds, cropped to the first"
            )
        return out


# --------------------------------------------------------------------------
# Readers: one record, as shipped, in millivolts
# --------------------------------------------------------------------------


def read_wfdb(path: Path) -> Record:
    """A WFDB record (PTB-XL, Chongqing): ``path`` without its extension.

    ``wfdb`` applies the header's gain and baseline, so the result is in the
    header's physical units.  Those are required to be millivolts: a corpus
    stored in anything else is refused rather than converted on a guess.
    """
    record = wfdb.rdrecord(str(path))
    units = set(record.units)
    if units != {"mV"}:
        raise ValueError(f"{path}: units {sorted(units)} are not millivolts")
    signal = np.ascontiguousarray(record.p_signal.T, dtype=np.float32)
    return Record(signal, tuple(record.sig_name), int(record.fs))


def read_sph(path: Path) -> Record:
    """A Shandong record: ``path`` to the ``.h5`` file.

    The file holds one dataset, ``ecg``, of shape (12, L) in float16, already in
    millivolts ("the unit is mV", Liu et al. 2022, Methods).
    """
    with h5py.File(path, "r") as f:
        signal = f["ecg"][()].astype(np.float32)
    return Record(signal, SPH_LEADS, SPH_SAMPLING_RATE_HZ)


def read_or_error(read: Callable[[Path], Record], path: Path) -> Record | Exception:
    """The record, or the error the reader raised.

    A file the corpus ships but cannot be read (Chongqing has two whose samples
    stop at seven seconds under a ten-second header) is a fact about the
    corpus.  It is reported in ``Corpus.excluded`` with the reader's message,
    not raised halfway through a pass.
    """
    try:
        return read(path)
    except (ValueError, OSError, KeyError) as error:
        return error


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------


def canonicalise(record: Record) -> NDArray[np.float32]:
    """(12, 5000) float32: canonical lead order, 500 Hz, the first ten seconds."""
    signal = record.signal[_lead_order(record.lead_names)]
    if record.sampling_rate_hz != SAMPLING_RATE_HZ:
        divisor = math.gcd(SAMPLING_RATE_HZ, record.sampling_rate_hz)
        signal = resample_poly(
            signal, SAMPLING_RATE_HZ // divisor, record.sampling_rate_hz // divisor, axis=-1
        )
    if signal.shape[-1] < WINDOW_SAMPLES:
        raise ValueError(f"record is shorter than ten seconds: {signal.shape[-1]} samples")
    return np.ascontiguousarray(signal[:, :WINDOW_SAMPLES], dtype=np.float32)


def _lead_order(lead_names: Sequence[str]) -> list[int]:
    """Index into ``lead_names`` that yields the canonical order."""
    upper = [name.upper() for name in lead_names]
    order = []
    for lead in CANONICAL_LEADS:
        if lead.upper() not in upper:
            raise ValueError(f"lead {lead} missing from header {list(lead_names)}")
        order.append(upper.index(lead.upper()))
    return order


def assemble_corpus(
    name: str,
    records: Iterable[tuple[str, Record | Exception]],
    n: int,
    notes: Sequence[str] = (),
) -> Corpus:
    """Run ``n`` records through the chain and stack the survivors.

    A record with a NaN or Inf sample, or one its reader could not load, is
    excluded and its id kept with the reason, so the count is reported rather
    than the record silently imputed.  ``n`` is known before any record is
    read, so the array is allocated once rather than stacked.
    """
    x = np.empty((n, N_LEADS, WINDOW_SAMPLES), dtype=np.float32)
    ids: list[str] = []
    excluded: dict[str, str] = {}
    resampled = 0
    cropped = 0
    for record_id, record in records:
        if isinstance(record, Exception):
            excluded[record_id] = f"unreadable: {record}"
            continue
        if not np.isfinite(record.signal).all():
            excluded[record_id] = "NaN or Inf sample"
            continue
        if record.sampling_rate_hz != SAMPLING_RATE_HZ:
            resampled += 1
        if record.signal.shape[-1] > WINDOW_SAMPLES * record.sampling_rate_hz / SAMPLING_RATE_HZ:
            cropped += 1
        try:
            canonical = canonicalise(record)
        except ValueError as error:
            # A record that cannot reach the canonical form -- too short for the
            # ten-second window, a lead its header does not name -- is excluded
            # and counted, not padded and not allowed to stop the pass.
            excluded[record_id] = str(error)
            continue
        x[len(ids)] = canonical
        ids.append(record_id)
    if len(ids) + len(excluded) != n:
        raise ValueError(f"{name}: expected {n} records, read {len(ids) + len(excluded)}")
    if excluded:
        x = x[: len(ids)].copy()
    return Corpus(name, x, ids, excluded, resampled, cropped, tuple(notes))


# --------------------------------------------------------------------------
# The three corpora, read in place
# --------------------------------------------------------------------------


def iter_ptbxl(
    database: pd.DataFrame, root: Path = PTBXL_DIR, ids: Sequence[int] | None = None
) -> Iterator[tuple[str, Record | Exception]]:
    """PTB-XL records at 500 Hz.  ``database`` is ptbxl_database.csv indexed by
    ecg_id; the path of each record is the database's own ``filename_hr``."""
    rows = database if ids is None else database.loc[list(ids)]
    for ecg_id, filename in rows["filename_hr"].items():
        yield str(ecg_id), read_or_error(read_wfdb, root / str(filename))


def iter_sph(
    metadata: pd.DataFrame, root: Path = SPH_DIR, ids: Sequence[str] | None = None
) -> Iterator[tuple[str, Record | Exception]]:
    """Shandong records.  ``metadata`` is the corpus metadata.csv."""
    wanted = metadata["ECG_ID"] if ids is None else pd.Series(list(ids))
    for ecg_id in wanted:
        yield str(ecg_id), read_or_error(read_sph, root / "records" / f"{ecg_id}.h5")


def iter_acs(
    table: pd.DataFrame, root: Path = ACS_DIR, ids: Sequence[str] | None = None
) -> Iterator[tuple[str, Record | Exception]]:
    """Chongqing records.  ``table`` is train.csv or test.csv, whose
    ``ecg_row_record`` column names the file (``04904.dat``); ``ids`` are those
    names without the extension."""
    names = (
        [str(f).removesuffix(".dat") for f in table["ecg_row_record"]] if ids is None else list(ids)
    )
    for name in names:
        yield name, read_or_error(read_wfdb, root / "row_data" / name)


def load_ptbxl(
    database: pd.DataFrame, root: Path = PTBXL_DIR, ids: Sequence[int] | None = None
) -> Corpus:
    """Canonical PTB-XL: WFDB at 1000 ADC units per millivolt, read by its header."""
    n = len(database) if ids is None else len(ids)
    return assemble_corpus("ptbxl", iter_ptbxl(database, root, ids), n)


def load_sph(
    metadata: pd.DataFrame, root: Path = SPH_DIR, ids: Sequence[str] | None = None
) -> Corpus:
    """Canonical Shandong: HDF5 float16 millivolts, lead order from the paper."""
    n = len(metadata) if ids is None else len(ids)
    return assemble_corpus("sph", iter_sph(metadata, root, ids), n, [SPH_DEVIATION])


def load_acs(table: pd.DataFrame, root: Path = ACS_DIR, ids: Sequence[str] | None = None) -> Corpus:
    """Canonical Chongqing: WFDB at 1000 ADC units per millivolt, read by its header."""
    n = len(table) if ids is None else len(ids)
    return assemble_corpus("acs", iter_acs(table, root, ids), n)
