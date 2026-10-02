"""EchoNext ingestion, on a small corpus written by the fixture in the distribution's layout."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecs.echonext import (
    LABELS,
    LEAD_ORDER,
    PROVENANCE_FIELDS,
    UNIT,
    canonical,
    iter_blocks,
    provenance_rows,
    published_digests,
    read_metadata,
    read_rows,
    transfer_cohorts,
    verify_files,
    waveform_path,
)

# split, patient, care context, for each record, in file order within a split.
RECORDS = [
    ("train", "p1", "inpatient"),
    ("train", "p1", "outpatient"),
    ("train", "p2", "emergency"),
    ("train", "p3", "inpatient"),
    ("val", "p4", "inpatient"),
    ("val", "p5", "inpatient"),
    ("val", "p6", "outpatient"),
    ("test", "p7", "inpatient"),
    ("test", "p8", "outpatient"),
    ("test", "p9", "emergency"),
    ("test", "p10", "outpatient"),
]


def write_corpus(root: Path, records: list[tuple[str, str, str]] = RECORDS) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i, (split, patient, context) in enumerate(records):
        row = {
            "ecg_key": 1000 + i,
            "patient_key": patient,
            "split": split,
            "location_setting": context,
            "sex": "male" if i % 2 else "female",
            "age_at_ecg": 40 + 5 * i,
            "race_ethnicity": "unknown",
        }
        row |= {label: int(rng.uniform() < 0.4) for label in LABELS}
        rows.append(row)
    meta = pd.DataFrame(rows)
    meta.to_csv(root / "echonext_metadata_100k.csv")
    for split in ("train", "val", "test"):
        n = int((meta["split"] == split).sum())
        waves = rng.normal(size=(n, 1, 2500, 12))
        if split == "train":
            waves[3] = waves[0]  # the fourth training tracing is a copy of the first
            waves[2, 0, :, 5] = 0.0  # and the third has a flat aVF
        np.save(waveform_path(root, split), waves)
    lines = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()} {p.name}" for p in sorted(root.iterdir())
    ]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    return meta


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    write_corpus(tmp_path)
    return tmp_path


class TestFiles:
    def test_every_file_matches_its_published_digest(self, corpus: Path) -> None:
        assert set(verify_files(corpus).values()) == {True}

    def test_an_altered_file_is_reported(self, corpus: Path) -> None:
        path = waveform_path(corpus, "val")
        data = np.load(path)
        data[0, 0, 0, 0] += 1.0
        np.save(path, data)
        assert verify_files(corpus)[path.name] is False

    def test_a_missing_column_is_named(self, corpus: Path) -> None:
        table = pd.read_csv(corpus / "echonext_metadata_100k.csv", index_col=0)
        table.drop(columns=["location_setting"]).to_csv(corpus / "echonext_metadata_100k.csv")
        with pytest.raises(ValueError, match="location_setting"):
            read_metadata(corpus)


class TestReading:
    def test_rows_read_in_the_order_asked_equal_the_array(self, corpus: Path) -> None:
        path = waveform_path(corpus, "test")
        rows = np.array([3, 0, 1, 2, 1])
        assert np.array_equal(read_rows(path, rows), np.load(path)[rows])

    def test_blocks_cover_the_split_once(self, corpus: Path) -> None:
        stored = np.load(waveform_path(corpus, "train"))
        blocks = list(iter_blocks(corpus, "train", block=3))
        assert [start for start, _ in blocks] == [0, 3]
        assert np.array_equal(np.concatenate([b for _, b in blocks]), stored)

    def test_canonical_puts_leads_first_in_float32(self, corpus: Path) -> None:
        stored = np.load(waveform_path(corpus, "val"))
        x = canonical(stored)
        assert x.shape == (3, 12, 2500) and x.dtype == np.float32
        assert np.allclose(x[1, 4], stored[1, 0, :, 4])

    def test_row_is_the_position_inside_the_split(self, corpus: Path) -> None:
        meta = read_metadata(corpus)
        assert meta.loc[meta["split"] == "test", "row"].tolist() == [0, 1, 2, 3]


class TestProvenance:
    def test_the_table_carries_every_declared_field(self, corpus: Path) -> None:
        table = provenance_rows(read_metadata(corpus), corpus, ("train", "val", "test"))
        missing = [field for field in PROVENANCE_FIELDS if field not in table.columns]
        assert missing == []

    def test_the_schema_check_fails_on_a_dropped_column(self, corpus: Path) -> None:
        table = provenance_rows(read_metadata(corpus), corpus, ("val",)).drop(columns="unit")
        assert [f for f in PROVENANCE_FIELDS if f not in table.columns] == ["unit"]

    def test_every_tracing_is_in_z_score_with_the_lead_order(self, corpus: Path) -> None:
        table = provenance_rows(read_metadata(corpus), corpus, ("train", "val", "test"))
        assert set(table["unit"]) == {UNIT} == {"z-score"}
        assert set(table["lead_order"]) == {",".join(LEAD_ORDER)}
        assert len(table) == len(RECORDS)

    def test_identical_tracings_share_one_copy_group(self, corpus: Path) -> None:
        table = provenance_rows(read_metadata(corpus), corpus, ("train", "val", "test"))
        groups = table.set_index("ecg_key")["copy_group"]
        assert groups[1003] == groups[1000] == "1000"
        assert table["copy_group"].nunique() == len(RECORDS) - 1

    def test_a_flat_lead_is_flagged(self, corpus: Path) -> None:
        table = provenance_rows(read_metadata(corpus), corpus, ("train",)).set_index("ecg_key")
        assert table.loc[1002, "quality_flat_leads"] == 1
        assert table["quality_flat_leads"].drop(1002).eq(0).all()

    def test_the_published_digest_travels_with_each_row(self, corpus: Path) -> None:
        table = provenance_rows(read_metadata(corpus), corpus, ("val",))
        digests = published_digests(corpus)
        assert set(table["source_sha256"]) == {digests["EchoNext_val_waveforms.npy"]}


class TestCohorts:
    def test_roles_follow_split_and_care_context(self, corpus: Path) -> None:
        meta = read_metadata(corpus)
        cohorts = transfer_cohorts(meta)
        keys = meta["ecg_key"].to_numpy()
        assert keys[cohorts.calibration].tolist() == [1004, 1005]
        assert keys[cohorts.targets["outpatient"]].tolist() == [1008, 1010]
        assert keys[cohorts.targets["inpatient"]].tolist() == [1007]
        assert keys[cohorts.training].tolist() == [1000, 1001, 1002, 1003]

    def test_no_patient_is_both_calibration_and_target(self, corpus: Path) -> None:
        meta = read_metadata(corpus)
        cohorts = transfer_cohorts(meta)
        patients = meta["patient_key"].to_numpy()
        calibration = set(patients[cohorts.calibration])
        for rows in cohorts.targets.values():
            assert calibration.isdisjoint(patients[rows])

    def test_a_patient_on_both_sides_is_refused(self, tmp_path: Path) -> None:
        records = [*RECORDS[:-1], ("test", "p4", "outpatient")]  # p4 calibrates in val
        write_corpus(tmp_path, records)
        with pytest.raises(ValueError, match="1 patients are in both calibration and target"):
            transfer_cohorts(read_metadata(tmp_path))

    def test_a_training_patient_in_a_target_is_refused(self, tmp_path: Path) -> None:
        records = [*RECORDS[:-1], ("test", "p2", "emergency")]
        write_corpus(tmp_path, records)
        with pytest.raises(ValueError, match="training and target:emergency"):
            transfer_cohorts(read_metadata(tmp_path))
