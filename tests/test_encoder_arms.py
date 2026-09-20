"""The ECG-JEPA arm: the checkpoint loads whole, and the chain is the published one.

The other three pre-trained arms are already held by ``test_extract_embeddings.py``
and by the arm grid's own tests.  This file covers the fourth, and the one claim
that makes it worth adding: it is the only pre-trained arm whose PTB-XL figure is
not partly memory.

The load is marked ``data`` because the weights are 340 MB and live under
``data/weights/``, outside the repository.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecs.encoders import ARMS, JEPA_LEADS, JEPA_SAMPLES, PRETRAINING, SAW
from ecs.ingest import CANONICAL_LEADS


class TestWhatTheArmDeclares:
    def test_ecg_jepa_is_among_the_arms(self) -> None:
        assert "ecg_jepa" in ARMS

    def test_every_arm_says_what_it_was_pre_trained_on(self) -> None:
        assert set(PRETRAINING) == set(ARMS)
        for arm, corpora in PRETRAINING.items():
            assert len(corpora.split()) >= 3, arm

    def test_the_two_contaminated_arms_are_the_ones_that_name_ptb_xl(self) -> None:
        """C-12's point: which arm saw the calibration corpus is on the record."""
        assert {arm for arm, corpora in SAW.items() if "ptbxl" in corpora} == {
            "ecgfm",
            "hubert_ecg",
        }

    def test_what_each_arm_saw_matches_what_its_sources_say(self) -> None:
        """The structured fact and the sentence beside it cannot drift apart."""
        for arm, corpora in SAW.items():
            described = PRETRAINING[arm]
            if "ptbxl" in corpora:
                assert "PTB-XL" in described and "not PTB-XL" not in described, arm
            if "sph" in corpora:
                assert "Shandong" in described, arm
            if "chapman_ningbo" in corpora:
                assert "Chapman-Shaoxing" in described, arm
        assert set(SAW) == set(ARMS)

    def test_ecg_jepa_names_the_corpus_it_did_see(self) -> None:
        """It is clean on PTB-XL and contaminated on Chapman-Shaoxing and Ningbo."""
        described = PRETRAINING["ecg_jepa"]
        assert "Chapman-Shaoxing" in described and "Ningbo" in described
        assert "not PTB-XL" in described
        assert SAW["ecg_jepa"] == ("chapman_ningbo",)

    def test_the_eight_leads_are_the_ones_the_authors_keep(self) -> None:
        kept = [CANONICAL_LEADS[i] for i in JEPA_LEADS]
        assert kept == ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


@pytest.mark.data
class TestTheCheckpointLoadsWhole:
    def test_no_weight_is_missing_and_none_is_left_over(self) -> None:
        _embed, meta = ARMS["ecg_jepa"]()
        assert "missing=0 unexpected=0" in str(meta["notes"])
        assert int(str(meta["n_params"])) > 10_000_000

    def test_a_batch_comes_out_as_one_vector_per_record(self) -> None:
        embed, _meta = ARMS["ecg_jepa"]()
        x = np.random.default_rng(0).normal(size=(2, 12, 5000)).astype(np.float32)
        out = embed(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 768
        assert bool(np.isfinite(out.detach().numpy()).all())

    def test_the_arm_resamples_to_the_length_the_encoder_asks_for(self) -> None:
        """The encoder asserts (batch, 8, 2500); a wrong rate would raise there."""
        assert JEPA_SAMPLES == 2500
        embed, _ = ARMS["ecg_jepa"]()
        x = np.zeros((1, 12, 5000), dtype=np.float32)
        assert embed(x).shape == (1, 768)
