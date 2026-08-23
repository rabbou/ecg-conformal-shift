"""Reference-value tests against the three corpora.

Every number here was read off the corpora's own description files on
2026-08-22 and is the anchor for the whole study: if a loader silently changes
what counts as an infarction, the prevalence gap that the study measures would
move with it and nothing else would notice.

Marked ``data`` -- skipped when the corpora are not on disk.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecs.config import ACS_DIR, PTBXL_DIR, SPH_DIR
from ecs.labels import MILabelSpec, ptbxl_mi_label, sph_mi_label

pytestmark = pytest.mark.data

ACS_CSV = ACS_DIR / "CSV/train.csv"


def _skip_unless(*paths: object) -> None:
    for path in paths:
        if not getattr(path, "exists", lambda: False)():
            pytest.skip(f"corpus not on disk: {path}")


@pytest.fixture(scope="module")
def ptbxl() -> tuple[pd.DataFrame, pd.DataFrame]:
    database = PTBXL_DIR / "ptbxl_database.csv"
    statements = PTBXL_DIR / "scp_statements.csv"
    _skip_unless(database, statements)
    return pd.read_csv(database, index_col="ecg_id"), pd.read_csv(statements, index_col=0)


@pytest.fixture(scope="module")
def sph() -> pd.DataFrame:
    path = SPH_DIR / "metadata.csv"
    _skip_unless(path)
    return pd.read_csv(path)


class TestPTBXL:
    def test_corpus_shape(self, ptbxl: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        database, _ = ptbxl
        assert len(database) == 21799
        assert database["patient_id"].nunique() == 18869

    def test_mi_count_as_shipped(self, ptbxl: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        database, statements = ptbxl
        assert int(ptbxl_mi_label(database, statements).sum()) == 5469

    def test_dropping_subendocardial_injury_moves_it_by_181(
        self, ptbxl: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """The label-definition sensitivity check. It matters because it is the
        first thing a reviewer will suspect is behind the prevalence gap, and
        the answer is that it accounts for 0.8 points of 24."""
        database, statements = ptbxl
        spec = MILabelSpec(include_injury=False)
        assert int(ptbxl_mi_label(database, statements, spec).sum()) == 5288

    def test_confidence_filtering_removes_the_lowest_likelihood_statements(
        self, ptbxl: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        database, statements = ptbxl
        high = ptbxl_mi_label(database, statements, MILabelSpec(min_likelihood=50.0))
        assert int(high.sum()) == 4134


class TestSPH:
    def test_corpus_shape(self, sph: pd.DataFrame) -> None:
        assert len(sph) == 25770
        assert sph["Patient_ID"].nunique() == 24666

    def test_mi_count(self, sph: pd.DataFrame) -> None:
        assert int(sph_mi_label(sph).sum()) == 260

    def test_almost_every_infarction_is_labelled_old(self, sph: pd.DataFrame) -> None:
        """What licenses comparing this corpus to PTB-XL at all: both are
        dominated by chronic infarct patterns rather than acute events."""
        chronic = int(sph_mi_label(sph, MILabelSpec(chronic_only=True)).sum())
        assert chronic == 233
        assert chronic / 260 > 0.85


class TestPrevalenceGap:
    def test_the_gap_the_study_is_built_on(
        self, ptbxl: tuple[pd.DataFrame, pd.DataFrame], sph: pd.DataFrame
    ) -> None:
        database, statements = ptbxl
        source = float(ptbxl_mi_label(database, statements).mean())
        target = float(sph_mi_label(sph).mean())
        assert source == pytest.approx(0.2509, abs=5e-4)
        assert target == pytest.approx(0.0101, abs=5e-4)
        assert source / target > 20.0


class TestACS:
    def test_labelled_split_shape_and_withheld_test_labels(self) -> None:
        _skip_unless(ACS_CSV)
        train = pd.read_csv(ACS_CSV)
        assert train.shape == (17960, 28)
        assert train["Patient_id"].nunique() == 17018
        test = pd.read_csv(ACS_CSV.with_name("test.csv"))
        assert "OMI" not in test.columns, "test labels are withheld by the publishers"

    def test_acute_label_counts(self) -> None:
        _skip_unless(ACS_CSV)
        train = pd.read_csv(ACS_CSV)
        assert int(train["OMI"].sum()) == 1151
        assert int(train["STEMI"].sum()) == 1442
        assert int(train["AMI"].sum()) == 2679
