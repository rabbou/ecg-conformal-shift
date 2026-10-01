"""The transfer measurements, on synthetic scores whose answer is known."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ecs import transfer
from ecs.transfer import conformal_sets, coverage_row, ladder_rows, ppv_row, subgroup_rows


def calibrated(n: int, prevalence: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores from two overlapping Beta laws; the label is drawn first, so a change
    of ``prevalence`` is a pure label shift.  The ill are scored less confidently
    than the healthy, so the two classes' scores do not share one law."""
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < prevalence).astype(int)
    p = np.where(y == 1, rng.beta(3, 3, n), rng.beta(1, 4, n))
    return p, y


class TestThresholdsComeFromCalibrationOnly:
    def test_a_target_record_gets_the_same_set_whatever_the_other_targets(self) -> None:
        p_cal, y_cal = calibrated(800, 0.5, 0)
        p_a, _ = calibrated(300, 0.1, 1)
        p_b, _ = calibrated(300, 0.9, 2)
        first = conformal_sets(p_cal, y_cal, np.r_[0.42, p_a])
        second = conformal_sets(p_cal, y_cal, np.r_[0.42, p_b])
        for method in transfer.METHODS:
            assert np.array_equal(first[method][0], second[method][0])

    def test_the_plain_threshold_reaches_ninety_percent_of_calibration_positives(self) -> None:
        p_cal, y_cal = calibrated(1000, 0.4, 3)
        flagged = conformal_sets(p_cal, y_cal, p_cal)["plain"][:, 1]
        sensitivity = flagged[y_cal == 1].mean()
        assert 0.90 <= sensitivity <= 0.90 + 2 / (y_cal == 1).sum()


class TestCoverageUnderLabelShift:
    """Calibrated where the ill are rare, read where they are common: the per-label
    scheme holds coverage in each class; the pooled one, whose quantile the healthy
    majority sets, does not hold it in the ill."""

    p_cal, y_cal = calibrated(4000, 0.1, 4)
    p_tgt, y_tgt = calibrated(20000, 0.5, 5)
    sets = conformal_sets(p_cal, y_cal, p_tgt)

    def test_per_label_holds_coverage_in_both_classes(self) -> None:
        row = coverage_row(self.sets["perlabel"], self.y_tgt)
        assert row["coverage_pos"] == pytest.approx(0.9, abs=0.02)
        assert row["coverage_neg"] == pytest.approx(0.9, abs=0.02)

    def test_pooled_coverage_of_the_ill_falls_below_per_label(self) -> None:
        pooled = coverage_row(self.sets["pooled"], self.y_tgt)["coverage_pos"]
        per = coverage_row(self.sets["perlabel"], self.y_tgt)["coverage_pos"]
        assert pooled < per - 0.05


def test_coverage_row_counts_by_hand() -> None:
    sets = np.array([[True, False], [True, True], [False, False], [False, True]])
    y = np.array([0, 1, 1, 1])
    row = coverage_row(sets, y)
    assert row["n"] == 4 and row["n_pos"] == 3
    assert row["coverage"] == pytest.approx(3 / 4)
    assert row["coverage_pos"] == pytest.approx(2 / 3)
    assert row["coverage_neg"] == pytest.approx(1.0)
    assert row["abstention"] == pytest.approx(2 / 4)
    assert row["both"] == pytest.approx(1 / 4) and row["empty"] == pytest.approx(1 / 4)


def test_too_few_calibration_positives_flag_everyone() -> None:
    # Eight positives cannot certify 90%: ceil(9 * 0.9) = 9 > 8.
    p_cal = np.r_[np.full(8, 0.8), np.full(100, 0.2)]
    y_cal = np.r_[np.ones(8, dtype=int), np.zeros(100, dtype=int)]
    sets = conformal_sets(p_cal, y_cal, np.array([0.01, 0.5, 0.99]))
    assert sets["perlabel"][:, 1].all() and sets["plain"][:, 1].all()
    assert math.isfinite(transfer.conformal_quantile(1 - p_cal[y_cal == 0], 0.1))


def test_subgroups_split_the_ill_by_sex_and_age_band() -> None:
    sets = np.array([[False, True], [True, False], [False, True], [True, False]])
    y = np.array([1, 1, 1, 0])
    meta = pd.DataFrame(
        {
            "sex": ["female", "female", "male", "male"],
            "age_at_ecg": [45, 70, 82, 30],
            "race_ethnicity": ["white"] * 4,
        }
    )
    rows = {(r["kind"], r["group"]): r for r in subgroup_rows(sets, y, meta)}
    assert rows[("sex", "female")]["n_pos"] == 2
    assert rows[("sex", "female")]["coverage_pos"] == pytest.approx(0.5)
    assert rows[("sex", "male")]["coverage_pos"] == pytest.approx(1.0)
    assert rows[("age", "80+")]["n_pos"] == 1 and rows[("age", "18-49")]["n_pos"] == 1


def test_ppv_from_source_uses_the_target_prevalence() -> None:
    y_src = np.array([1, 1, 0, 0])
    src_flags = np.array([True, False, True, False])  # sens 0.5, spec 0.5
    y_tgt = np.array([1, 0, 0, 0])
    tgt_flags = np.array([True, True, False, False])  # sens 1, spec 2/3
    row = ppv_row(src_flags, y_src, tgt_flags, y_tgt)
    assert row["ppv_from_source"] == pytest.approx(0.5 * 0.25 / (0.5 * 0.25 + 0.5 * 0.75))
    assert row["ppv_observed"] == pytest.approx(0.5)
    assert row["false_alerts_observed"] == pytest.approx((1 / 3) * 0.75 / 0.25)


class TestLadder:
    p_cal, y_cal = calibrated(1000, 0.5, 6)
    p_tgt, y_tgt = calibrated(1200, 0.15, 7)
    patients = pd.Series([f"q{i}" for i in range(1200)])

    def rows(self) -> list[dict]:
        return ladder_rows(
            self.p_cal,
            self.y_cal,
            self.p_tgt,
            self.y_tgt,
            self.patients,
            rungs=(0, 100, 400),
            draws=40,
        )

    def test_every_rung_is_read_on_the_same_evaluation_half(self) -> None:
        rows = self.rows()
        assert len({r["n_eval"] for r in rows}) == 1
        assert rows[0]["n_eval"] == 600

    def test_rung_zero_is_the_source_threshold_once(self) -> None:
        zero = self.rows()[0]
        assert zero["draws"] == 1
        expected = conformal_sets(self.p_cal, self.y_cal, self.p_tgt)  # same thresholds
        assert zero["coverage_pos_p10"] == zero["coverage_pos_p90"]
        assert expected["perlabel"].shape == (1200, 2)

    def test_target_labels_shrink_the_calibration_intercept(self) -> None:
        rows = self.rows()
        assert rows[-1]["abs_intercept_mean"] < rows[0]["abs_intercept_mean"]

    def test_draws_come_from_the_pool_only(self) -> None:
        class Recording:
            def __init__(self) -> None:
                self.inner = np.random.default_rng(0)
                self.drawn_from: list[set[int]] = []

            def choice(self, a: np.ndarray, **kw: Any) -> np.ndarray:
                self.drawn_from.append(set(np.asarray(a).tolist()))
                return self.inner.choice(a, **kw)

        recording = Recording()
        ladder_rows(
            self.p_cal,
            self.y_cal,
            self.p_tgt,
            self.y_tgt,
            self.patients,
            rungs=(0, 100),
            draws=5,
            rng=recording,  # type: ignore[arg-type]
        )
        part = transfer.patient_split(
            self.patients, {"pool": transfer.POOL_SHARE, "eval": 1 - transfer.POOL_SHARE}, 0
        ).to_numpy()
        pool = set(np.flatnonzero(part == "pool").tolist())
        assert len(recording.drawn_from) == 5
        assert all(drawn == pool for drawn in recording.drawn_from)
