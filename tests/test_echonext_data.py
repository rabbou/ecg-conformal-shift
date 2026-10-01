"""The committed EchoNext figures against the corpus on disk.

These need EchoNext (restricted licence, ``ECS_ECHONEXT_DIR``) and the derived
files the scripts write outside the repository.  They are deselected by
``-m "not data"``, never skipped: run where the corpus is absent, they fail
and say what is missing, so a green run cannot hide a missing corpus.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ecs.config import RESULTS_DIR
from ecs.echonext import DERIVED_DIR, ECHONEXT_DIR, LABELS, read_metadata, transfer_cohorts

pytestmark = pytest.mark.data


def require(path: Any) -> None:
    if not path.exists():
        pytest.fail(f"{path} is absent; these checks need EchoNext and its derived files")


@pytest.fixture(scope="module")
def meta() -> pd.DataFrame:
    require(ECHONEXT_DIR / "echonext_metadata_100k.csv")
    return read_metadata(ECHONEXT_DIR)


def test_the_metadata_carries_care_context_and_the_twelve_labels(meta: pd.DataFrame) -> None:
    assert set(meta["location_setting"]) == {"inpatient", "emergency", "outpatient", "procedural"}
    assert all(label in meta.columns for label in LABELS)
    assert meta[list(LABELS)].isin([0, 1]).all().all()


def test_the_prevalence_replay_matches_the_metadata(meta: pd.DataFrame) -> None:
    replay = json.loads((RESULTS_DIR / "echonext_transfer.json").read_text())
    replay = replay["prevalence_by_context_val_and_test"]
    both = meta[meta["split"].isin(["val", "test"])]
    for context, group in both.groupby("location_setting"):
        assert replay[context]["n"] == len(group)
        for label in LABELS:
            assert replay[context][label] == pytest.approx(group[label].mean())


def test_the_published_splits_share_no_patient(meta: pd.DataFrame) -> None:
    transfer_cohorts(meta, "inpatient", ("inpatient", "emergency", "outpatient"))


def test_every_tracing_in_the_provenance_table_is_in_z_score(meta: pd.DataFrame) -> None:
    require(DERIVED_DIR / "provenance.csv.gz")
    table = pd.read_csv(DERIVED_DIR / "provenance.csv.gz", usecols=["ecg_key", "unit"])
    assert len(table) == len(meta)
    assert table["unit"].unique().tolist() == ["z-score"]


def test_the_scores_cover_every_validation_and_test_ecg(meta: pd.DataFrame) -> None:
    keys = set(meta.loc[meta["split"].isin(["val", "test"]), "ecg_key"])
    for arm in ("resnet", "random_init"):
        path = DERIVED_DIR / f"scores/{arm}.npz"
        require(path)
        with np.load(path) as data:
            assert set(data["ecg_key"].tolist()) == keys
            assert np.isfinite(data["probs"]).all()
