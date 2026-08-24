"""The supervised baseline: the run holds together, and the reference it is
measured against is on record (C-19).

The end-to-end test builds a miniature PTB-XL on disk -- its database table,
its statement table, its WFDB records -- and runs the real script over it for
one epoch.  Nothing here asserts the model is any good on eighty synthetic
tracings; what it asserts is that the four files a result is made of appear,
agree with each other, and hold the fields the plan says they hold.  The
quality claim is C-19's and is checked against results/baseline.json once the
real run finishes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import train_baseline as tb
import wfdb

from ecs.config import RESULTS_DIR
from ecs.ingest import CANONICAL_LEADS

N_RECORDS = 80


def _write_ptbxl(root: Path) -> None:
    """A PTB-XL small enough to train on: eighty records dealt round-robin into
    the ten folds, with MI alternating by block of ten so that every fold --
    the eighth and tenth included -- holds four MI records and four without."""
    rng = np.random.default_rng(0)
    rows = []
    for k in range(N_RECORDS):
        is_mi = (k // 10) % 2 == 0
        name = f"records500/{k:05d}"
        (root / "records500").mkdir(parents=True, exist_ok=True)
        # A different mean amplitude per class, so one epoch has something to learn.
        signal = rng.normal(300 if is_mi else -300, 50, (5000, 12))
        wfdb.wrsamp(
            f"{k:05d}",
            fs=500,
            units=["mV"] * 12,
            sig_name=list(CANONICAL_LEADS),
            d_signal=signal.astype(np.int16),
            fmt=["16"] * 12,
            adc_gain=[1000.0] * 12,
            baseline=[0] * 12,
            write_dir=str(root / "records500"),
        )
        rows.append(
            {
                "ecg_id": k,
                "patient_id": k,
                "filename_hr": name,
                "strat_fold": (k % 10) + 1,
                "scp_codes": "{'IMI': 100.0}" if is_mi else "{'NORM': 100.0}",
            }
        )
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame(
        {
            "Unnamed: 0": ["IMI", "NORM"],
            "diagnostic": [1, 1],
            "diagnostic_class": ["MI", "NORM"],
        }
    ).to_csv(root / "scp_statements.csv", index=False)


class TestOneEpochEndToEnd:
    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory: pytest.TempPathFactory) -> Path:
        root = tmp_path_factory.mktemp("ptbxl")
        out = tmp_path_factory.mktemp("baseline")
        _write_ptbxl(root)
        code = tb.main(
            [
                "--epochs",
                "1",
                "--batch-size",
                "16",
                "--bootstrap-draws",
                "50",
                "--ptbxl-dir",
                str(root),
                "--out",
                str(out),
            ]
        )
        assert code == 0
        return out

    def test_it_writes_the_four_files_a_result_is_made_of(self, run: Path) -> None:
        for name in ("config.json", "train.log", "metrics.json", "scores.npz"):
            assert (run / name).exists(), name
        assert (run / "model.pt").exists()

    def test_the_log_carries_one_line_for_the_one_epoch(self, run: Path) -> None:
        lines = (run / "train.log").read_text().strip().splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("epoch  1")
        assert "validation AUROC" in lines[0]

    def test_the_config_states_everything_that_could_move_the_number(self, run: Path) -> None:
        config = json.loads((run / "config.json").read_text())
        assert set(config) >= {
            "epochs",
            "learning_rate",
            "batch_size",
            "seed",
            "label_spec",
            "split",
            "standardisation",
            "n_train",
            "n_validation",
            "n_test",
            "machine",
        }
        # Folds 1-8 train, 9 validation, 10 test, over eighty records dealt round-robin.
        assert (config["n_train"], config["n_validation"], config["n_test"]) == (64, 8, 8)
        assert config["epochs"] == 1

    def test_the_metrics_carry_an_interval_around_each_number(self, run: Path) -> None:
        metrics = json.loads((run / "metrics.json").read_text())
        assert set(metrics) >= {"auroc", "auroc_ci95", "auprc", "auprc_ci95", "epoch_kept"}
        for name in ("auroc", "auprc"):
            low, high = metrics[f"{name}_ci95"]
            assert 0.0 <= low <= metrics[name] <= high <= 1.0
        assert metrics["n_test"] == 8
        assert metrics["n_test_positive"] == 4
        assert metrics["epoch_kept"] == 1

    def test_the_scores_are_one_probability_row_per_test_record(self, run: Path) -> None:
        with np.load(run / "scores.npz", allow_pickle=False) as scores:
            ids, labels, probs = scores["ids"], scores["labels"], scores["probs"]
        assert len(ids) == len(labels) == len(probs) == 8
        # Fold 10 is every tenth record: ecg_id 9, 19, ... 79.
        assert sorted(int(i) for i in ids) == list(range(9, 80, 10))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        # The label follows the record: MI alternates by block of ten.
        assert labels.tolist() == [1 if (int(i) // 10) % 2 == 0 else 0 for i in ids]
        assert labels.sum() == 4


class TestTheReferenceValue:
    """C-19 compares against a number that is on disk with its provenance, never
    one recalled from memory."""

    def test_results_baseline_json_states_the_value_and_where_it_came_from(self) -> None:
        reference = json.loads((RESULTS_DIR / "baseline.json").read_text())
        assert 0.5 < reference["reference_auroc"] < 1.0
        assert reference["reference_source"]
        assert reference["reference_split"]
        assert reference["reference_read_on"]
