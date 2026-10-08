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
    for arm in ("resnet", "random_init", "ecgfounder", "echonext_mini"):
        path = DERIVED_DIR / f"scores/{arm}.npz"
        require(path)
        with np.load(path) as data:
            assert set(data["ecg_key"].tolist()) == keys
            assert np.isfinite(data["probs"]).all()


def test_both_published_checkpoints_load_whole_without_full_unpickling() -> None:
    """The mini-model under weights_only=True, ECGFounder through the allowlist,
    each into its architecture with no tensor missing and none left over."""
    import torch

    from ecs.echonext_mini import EchoNextMini
    from ecs.encoders import allowlisted_load, ecgfounder_net, verified

    mini = torch.load(verified("echonext_mini/weights.pt"), map_location="cpu", weights_only=True)
    EchoNextMini().load_state_dict(mini["model"], strict=True)
    founder = allowlisted_load(verified("ecgfounder/12_lead_ECGFounder.pth"))
    ecgfounder_net().load_state_dict(founder["state_dict"], strict=True)


def test_the_severity_file_is_what_the_corpus_and_stored_scores_give(meta: pd.DataFrame) -> None:
    """Rebuilt from the metadata and the stored scores, with nothing refitted, the
    severity results equal the committed file in every figure but the commit."""
    from echonext_severity import ARMS, measure

    for arm in ARMS:
        require(DERIVED_DIR / f"scores/{arm}.npz")
    committed = json.loads((RESULTS_DIR / "echonext_severity.json").read_text())
    committed.pop("commit")
    assert json.loads(json.dumps(measure(meta))) == committed


def test_the_clinical_file_is_what_the_corpus_and_stored_scores_give(meta: pd.DataFrame) -> None:
    """The report's outcome counts, ladder and case-mix model come from this file, and the
    scores behind it cannot be committed: rebuilt here from the metadata and the stored
    scores, whose digests the file records, it equals the committed file in every figure
    but its run time and provenance."""
    from echonext_clinical import ARMS, measure

    for arm in ARMS:
        require(DERIVED_DIR / f"scores/{arm}.npz")
    from echonext_clinical import inputs

    committed = json.loads((RESULTS_DIR / "echonext_clinical.json").read_text())
    for key in ("seconds", "provenance"):
        committed.pop(key)
    assert committed.pop("inputs") == inputs(), "the stored scores are not the ones the file read"
    assert json.loads(json.dumps(measure(meta))) == committed
