"""C-4: no patient on both sides of any boundary, on a synthetic frame and on
each corpus; determinism for a fixed seed; the proportions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecs.config import ACS_DIR, PTBXL_DIR, SPH_DIR
from ecs.splits import patient_split, ptbxl_benchmark_split

HALVES = {"calibration": 0.5, "test": 0.5}


def _synthetic_patients(seed: int = 0, n_patients: int = 300) -> pd.Series:
    """Records over patients with one to six tracings each, in shuffled order."""
    rng = np.random.default_rng(seed)
    per_patient = rng.integers(1, 7, size=n_patients)
    patients = np.repeat([f"P{i:04d}" for i in range(n_patients)], per_patient)
    return pd.Series(rng.permutation(patients), index=[f"rec{i}" for i in range(len(patients))])


def _assert_patient_disjoint(patients: pd.Series, part: pd.Series) -> None:
    assert part.index.equals(patients.index)
    assert not part.isna().any()
    sides_per_patient = pd.DataFrame({"patient": patients, "part": part}).groupby("patient")["part"]
    assert (sides_per_patient.nunique() == 1).all()


class TestSynthetic:
    def test_every_patient_is_on_one_side_only(self) -> None:
        patients = _synthetic_patients()
        _assert_patient_disjoint(patients, patient_split(patients, HALVES, seed=0))

    @pytest.mark.parametrize("seed", range(5))
    def test_holds_for_three_parts_and_any_seed(self, seed: int) -> None:
        patients = _synthetic_patients(seed=seed)
        fractions = {"train": 0.8, "calibration": 0.1, "test": 0.1}
        _assert_patient_disjoint(patients, patient_split(patients, fractions, seed=seed))

    def test_same_seed_same_split_and_record_order_does_not_matter(self) -> None:
        patients = _synthetic_patients()
        first = patient_split(patients, HALVES, seed=7)
        second = patient_split(patients, HALVES, seed=7)
        shuffled = patients.sample(frac=1.0, random_state=3)
        third = patient_split(shuffled, HALVES, seed=7)
        pd.testing.assert_series_equal(first, second)
        pd.testing.assert_series_equal(first, third.loc[first.index])

    def test_different_seeds_give_different_splits(self) -> None:
        patients = _synthetic_patients()
        assert not patient_split(patients, HALVES, seed=0).equals(
            patient_split(patients, HALVES, seed=1)
        )

    def test_proportions_are_met_in_patients_not_records(self) -> None:
        patients = _synthetic_patients()
        fractions = {"train": 0.8, "calibration": 0.1, "test": 0.1}
        part = patient_split(patients, fractions, seed=0)
        by_patient = pd.DataFrame({"patient": patients, "part": part}).drop_duplicates("patient")
        counts = by_patient["part"].value_counts()
        assert counts["train"] == 240
        assert counts["calibration"] == 30
        assert counts["test"] == 30
        # Records follow the patients, so their shares only approximate the fractions.
        record_share = part.value_counts(normalize=True)
        assert record_share["train"] == pytest.approx(0.8, abs=0.05)

    def test_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to"):
            patient_split(_synthetic_patients(), {"a": 0.5, "b": 0.4}, seed=0)


def _skip_unless(*paths: object) -> None:
    for path in paths:
        if not getattr(path, "exists", lambda: False)():
            pytest.skip(f"corpus not on disk: {path}")


@pytest.mark.data
class TestCorpora:
    def test_ptbxl(self) -> None:
        _skip_unless(PTBXL_DIR / "ptbxl_database.csv")
        database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
        _assert_patient_disjoint(
            database["patient_id"], patient_split(database["patient_id"], HALVES, 0)
        )

    def test_ptbxl_benchmark_folds_were_drawn_by_patient(self) -> None:
        _skip_unless(PTBXL_DIR / "ptbxl_database.csv")
        database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
        part = ptbxl_benchmark_split(database)
        _assert_patient_disjoint(database["patient_id"], part)
        assert part.value_counts().to_dict() == {"train": 17418, "validation": 2183, "test": 2198}

    def test_sph(self) -> None:
        _skip_unless(SPH_DIR / "metadata.csv")
        metadata = pd.read_csv(SPH_DIR / "metadata.csv", index_col="ECG_ID")
        _assert_patient_disjoint(
            metadata["Patient_ID"], patient_split(metadata["Patient_ID"], HALVES, 0)
        )

    def test_acs(self) -> None:
        _skip_unless(ACS_DIR / "CSV/train.csv")
        train = pd.read_csv(ACS_DIR / "CSV/train.csv", index_col="ecg_row_record")
        _assert_patient_disjoint(train["Patient_id"], patient_split(train["Patient_id"], HALVES, 0))
