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
from ecs.labels import MILabelSpec, acs_mi_label, ptbxl_mi_label, sph_mi_label

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


def _read_acs() -> pd.DataFrame:
    _skip_unless(ACS_CSV)
    return pd.read_csv(ACS_CSV)


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

    def test_the_label_the_study_scores_is_the_corpus_own_ami_column(self) -> None:
        train = _read_acs()
        assert int(acs_mi_label(train).sum()) == 2679
        assert float(acs_mi_label(train).mean()) == pytest.approx(0.1492, abs=5e-4)

    def test_ami_is_the_union_of_stemi_and_nstemi_on_all_but_two_records(self) -> None:
        """The internal consistency check on the column the study leans on. Two
        records of 17,960 are marked AMI with neither sub-type set; they stay
        positive, because the corpus's own infarction column is the label."""
        train = _read_acs()
        union = (train["STEMI"] == 1) | (train["NSTEMI"] == 1)
        assert int((acs_mi_label(train) & ~union).sum()) == 2
        assert int((union & ~acs_mi_label(train)).sum()) == 0

    def test_every_occlusion_infarction_is_inside_the_label(self) -> None:
        train = _read_acs()
        assert int(((train["OMI"] == 1) & acs_mi_label(train)).sum()) == 1151

    def test_a_chronic_only_label_is_refused_rather_than_returned_empty(self) -> None:
        """Every Chongqing positive is an acute event, so the spec that makes
        PTB-XL and Shandong comparable has no meaning here and says so."""
        train = _read_acs()
        with pytest.raises(ValueError, match="no chronic infarction"):
            acs_mi_label(train, MILabelSpec(chronic_only=True))

    def test_the_withheld_test_split_is_refused_rather_than_scored_blind(self) -> None:
        _skip_unless(ACS_CSV)
        test = pd.read_csv(ACS_CSV.with_name("test.csv"))
        with pytest.raises(ValueError, match="withholds the labels"):
            acs_mi_label(test)


class TestTheThreeWayPrevalenceGap:
    def test_what_the_frozen_calibration_is_spent_across(
        self, ptbxl: tuple[pd.DataFrame, pd.DataFrame], sph: pd.DataFrame
    ) -> None:
        """The three numbers the break is read against: a quarter of PTB-XL, a
        hundredth of Shandong, a seventh of Chongqing."""
        database, statements = ptbxl
        train = _read_acs()
        assert float(ptbxl_mi_label(database, statements).mean()) == pytest.approx(0.2509, abs=5e-4)
        assert float(sph_mi_label(sph).mean()) == pytest.approx(0.0101, abs=5e-4)
        assert float(acs_mi_label(train).mean()) == pytest.approx(0.1492, abs=5e-4)
