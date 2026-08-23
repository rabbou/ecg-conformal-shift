"""The ingestion contract (C-13, C-14, C-14b, C-15).

The synthetic tests pin the chain itself: lead reordering by header name,
resampling, cropping, the millivolt scale, NaN exclusion.  The ``data`` tests
pin each corpus to sample values read off its own files on 2026-08-23, so a
change in any loader's scale or lead order fails here before it reaches a
result.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import wfdb

from ecs.config import ACS_DIR, PTBXL_DIR, SPH_DIR
from ecs.ingest import (
    CANONICAL_LEADS,
    Record,
    assemble_corpus,
    canonicalise,
    load_acs,
    load_ptbxl,
    load_sph,
    read_wfdb,
)

LEADS_AS_PTBXL_WRITES_THEM = (
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


def _record(signal: np.ndarray, leads: tuple[str, ...] = CANONICAL_LEADS, fs: int = 500) -> Record:
    return Record(np.asarray(signal, dtype=np.float32), leads, fs)


def _write_wfdb(
    directory: Path, name: str, d_signal: np.ndarray, units: str = "mV", fs: int = 500
) -> Path:
    """A WFDB record in the layout PTB-XL and Chongqing use: format 16, 1000
    ADC units per millivolt, baseline 0."""
    wfdb.wrsamp(
        name,
        fs=fs,
        units=[units] * 12,
        sig_name=list(CANONICAL_LEADS),
        d_signal=d_signal.astype(np.int16),
        fmt=["16"] * 12,
        adc_gain=[1000.0] * 12,
        baseline=[0] * 12,
        write_dir=str(directory),
    )
    return directory / name


class TestChain:
    def test_shape_dtype_and_identity_on_a_canonical_record(self) -> None:
        x = canonicalise(_record(np.arange(12 * 5000).reshape(12, 5000)))
        assert x.shape == (12, 5000)
        assert x.dtype == np.float32
        np.testing.assert_array_equal(x, np.arange(12 * 5000).reshape(12, 5000))

    def test_leads_are_reordered_by_header_name_not_position(self) -> None:
        # Lead k of the stored array is the constant k; the header lists the
        # leads in reverse and in PTB-XL's upper-case spelling.
        stored = np.repeat(np.arange(12.0)[:, None], 5000, axis=1)
        reversed_header = tuple(reversed(LEADS_AS_PTBXL_WRITES_THEM))
        x = canonicalise(_record(stored, reversed_header))
        for position, lead in enumerate(CANONICAL_LEADS):
            stored_index = [name.upper() for name in reversed_header].index(lead.upper())
            assert x[position, 0] == stored_index

    def test_a_missing_lead_is_an_error_not_a_guess(self) -> None:
        eleven = CANONICAL_LEADS[:11]
        with pytest.raises(ValueError, match="V6 missing"):
            canonicalise(_record(np.zeros((11, 5000)), eleven))

    def test_longer_records_keep_their_first_ten_seconds(self) -> None:
        x = canonicalise(_record(np.arange(12 * 6000).reshape(12, 6000)))
        np.testing.assert_array_equal(x, np.arange(12 * 6000).reshape(12, 6000)[:, :5000])

    def test_shorter_records_are_refused(self) -> None:
        with pytest.raises(ValueError, match="shorter than ten seconds"):
            canonicalise(_record(np.zeros((12, 4999))))

    def test_resampling_to_500hz_preserves_a_band_limited_signal(self) -> None:
        # A 5 Hz sine sampled at 250 Hz, resampled, against the same sine
        # sampled at 500 Hz.  Polyphase resampling with anti-aliasing should
        # reproduce it to well under one percent away from the edges.
        t250 = np.arange(2500) / 250.0
        t500 = np.arange(5000) / 500.0
        x = canonicalise(_record(np.tile(np.sin(2 * np.pi * 5 * t250), (12, 1)), fs=250))
        assert x.shape == (12, 5000)
        np.testing.assert_allclose(
            x[0, 100:-100], np.sin(2 * np.pi * 5 * t500)[100:-100], atol=5e-3
        )


class TestAssembly:
    def test_nan_and_inf_records_are_excluded_and_counted(self) -> None:
        bad_nan = np.zeros((12, 5000))
        bad_nan[3, 17] = np.nan
        bad_inf = np.zeros((12, 5000))
        bad_inf[0, 0] = np.inf
        records = [
            ("good-1", _record(np.ones((12, 5000)))),
            ("bad-nan", _record(bad_nan)),
            ("good-2", _record(2 * np.ones((12, 5000)))),
            ("bad-inf", _record(bad_inf)),
        ]
        corpus = assemble_corpus("synthetic", records, n=4)
        assert corpus.x.shape == (2, 12, 5000)
        assert corpus.ids == ["good-1", "good-2"]
        assert corpus.excluded == ["bad-nan", "bad-inf"]
        assert corpus.x[1, 0, 0] == 2.0

    def test_deviations_name_what_the_chain_did_differently(self) -> None:
        records = [
            ("ten-seconds", _record(np.zeros((12, 5000)))),
            ("twelve-seconds", _record(np.zeros((12, 6000)))),
            ("at-250hz", _record(np.zeros((12, 2500)), fs=250)),
        ]
        corpus = assemble_corpus("synthetic", records, n=3, notes=["given up front"])
        assert corpus.deviations == [
            "given up front",
            "1 of 3 records resampled to 500 Hz",
            "1 of 3 records longer than ten seconds, cropped to the first",
        ]

    def test_a_short_read_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="expected 2 records, read 1"):
            assemble_corpus("synthetic", [("only", _record(np.zeros((12, 5000))))], n=2)

    def test_empty_corpus_has_the_canonical_shape(self) -> None:
        assert assemble_corpus("synthetic", [], n=0).x.shape == (0, 12, 5000)


class TestWfdbScale:
    def test_one_thousand_adc_units_per_millivolt(self, tmp_path: Path) -> None:
        d_signal = np.zeros((5000, 12))
        d_signal[0, :] = np.arange(12) * 100  # lead k starts at k/10 mV
        d_signal[1, 0] = -115
        record = read_wfdb(_write_wfdb(tmp_path, "synthetic", d_signal))
        assert record.signal.shape == (12, 5000)
        assert record.lead_names == CANONICAL_LEADS
        assert record.sampling_rate_hz == 500
        np.testing.assert_allclose(record.signal[:, 0], np.arange(12) / 10.0, atol=1e-7)
        assert record.signal[0, 1] == pytest.approx(-0.115)

    def test_units_other_than_millivolts_are_refused(self, tmp_path: Path) -> None:
        path = _write_wfdb(tmp_path, "microvolts", np.zeros((5000, 12)), units="uV")
        with pytest.raises(ValueError, match="not millivolts"):
            read_wfdb(path)


# --------------------------------------------------------------------------
# Against the corpora
# --------------------------------------------------------------------------


def _skip_unless(*paths: Path) -> None:
    for path in paths:
        if not path.exists():
            pytest.skip(f"corpus not on disk: {path}")


@pytest.mark.data
class TestPTBXL:
    def test_first_sample_of_record_1_matches_its_header_in_every_lead(self) -> None:
        """00001_hr.hea lists each lead's initial ADC value; at 1000 units per
        millivolt those are the first canonical samples, in header order."""
        _skip_unless(PTBXL_DIR / "ptbxl_database.csv")
        database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
        corpus = load_ptbxl(database, ids=[1, 2, 3])
        assert corpus.x.shape == (3, 12, 5000)
        assert corpus.x.dtype == np.float32
        assert corpus.ids == ["1", "2", "3"]
        assert corpus.excluded == []
        assert corpus.deviations == []
        header = wfdb.rdheader(str(PTBXL_DIR / str(database.at[1, "filename_hr"])))
        assert [name.upper() for name in header.sig_name] == [n.upper() for n in CANONICAL_LEADS]
        np.testing.assert_allclose(corpus.x[0, :, 0], np.array(header.init_value) / 1000.0)
        np.testing.assert_allclose(
            corpus.x[0, :, 0],
            np.array([-115, -50, 65, 82, -90, 7, -65, -40, -5, -35, -35, -75]) / 1000.0,
        )


@pytest.mark.data
class TestSPH:
    def test_record_a00001_is_read_as_shipped_and_a00002_is_cropped(self) -> None:
        _skip_unless(SPH_DIR / "metadata.csv", SPH_DIR / "records/A00002.h5")
        metadata = pd.read_csv(SPH_DIR / "metadata.csv")
        corpus = load_sph(metadata, ids=["A00001", "A00002"])
        assert corpus.x.shape == (2, 12, 5000)
        assert corpus.x.dtype == np.float32
        assert corpus.excluded == []
        assert corpus.deviations == [
            "lead order and sampling rate are not in the record (no HDF5 attributes); "
            "taken from Liu et al. 2022, Data Records",
            "1 of 2 records longer than ten seconds, cropped to the first",
        ]
        # Values as stored (float16 millivolts), read back independently.
        with h5py.File(SPH_DIR / "records/A00002.h5", "r") as f:
            stored = f["ecg"][()]
        assert stored.dtype == np.float16
        assert stored.shape == (12, 6000)
        np.testing.assert_array_equal(corpus.x[1], stored[:, :5000].astype(np.float32))
        np.testing.assert_array_equal(
            corpus.x[0, 0, :4],
            np.array([0.0216, 0.0216, 0.0208, 0.0184], dtype=np.float16).astype(np.float32),
        )


@pytest.mark.data
class TestACS:
    def test_record_04904_first_samples_and_header_lead_order(self) -> None:
        _skip_unless(ACS_DIR / "CSV/train.csv", ACS_DIR / "row_data/04904.hea")
        train = pd.read_csv(ACS_DIR / "CSV/train.csv")
        assert train.at[0, "ecg_row_record"] == "04904.dat"
        corpus = load_acs(train, ids=["04904", "00001"])
        assert corpus.x.shape == (2, 12, 5000)
        assert corpus.x.dtype == np.float32
        assert corpus.excluded == []
        assert corpus.deviations == []
        header = wfdb.rdheader(str(ACS_DIR / "row_data/04904"))
        assert tuple(header.sig_name) == CANONICAL_LEADS
        assert set(header.adc_gain) == {1000.0} and set(header.baseline) == {0}
        np.testing.assert_allclose(
            corpus.x[0, :, 0],
            np.array([0, 4, 4, -2, -2, 4, 0, -118, 20, 2, 4, 16]) / 1000.0,
        )
