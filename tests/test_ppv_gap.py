"""The recomputed-against-observed PPV measurement, on tables worked out by hand."""

from __future__ import annotations

import numpy as np
import pytest

from ecs.ppv_gap import Table, gap_row, label_free, label_shift_check, table_of

# Source: 100 ill, 90 flagged; 100 healthy, 40 flagged.  Sensitivity 0.9,
# specificity 0.6.
SOURCE = Table(tp=90, fp=40, fn=10, tn=60)
# Target: 40 ill of 400, 36 flagged; 360 healthy, 60 flagged.  Sensitivity
# 0.9, specificity 300/360, prevalence 0.1, observed PPV 36/96 = 0.375.
TARGET = Table(tp=36, fp=60, fn=4, tn=300)


def test_table_counts_the_four_cells() -> None:
    """A swapped cell would swap sensitivity and specificity in every row of T-067."""
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    flagged = np.array([True, True, False, True, False, False, False])
    assert table_of(y, flagged) == Table(tp=2, fp=1, fn=1, tn=3)


class TestGapRow:
    def test_the_recipe_and_the_observation_by_hand(self) -> None:
        """T-067 SHALL give, per cell, recomputed PPV, observed PPV and the gap in points.

        Recomputed: 0.9 * 0.1 / (0.09 + 0.4 * 0.9) = 0.09 / 0.45 = 0.2.
        Observed: 36 / 96 = 0.375.  Gap: -0.175.
        """
        row = gap_row(SOURCE, TARGET, draws=200)
        assert row["ppv_recomputed"] == pytest.approx(0.2)
        assert row["ppv_observed"] == pytest.approx(0.375)
        assert row["gap"] == pytest.approx(-0.175)
        assert row["recomputed_inside_interval"] is False

    def test_the_gap_is_split_between_sensitivity_and_specificity(self) -> None:
        """Sensitivity did not move, so the whole gap is specificity's.

        With the target's specificity 300/360: 0.09 / (0.09 + (60/360) * 0.9)
        = 0.09 / 0.24 = 0.375, the observed value.
        """
        row = gap_row(SOURCE, TARGET, draws=200)
        assert row["gap_from_sensitivity"] == pytest.approx(0.0)
        assert row["gap_from_specificity"] == pytest.approx(0.175)

    def test_false_alerts_per_detection_by_hand(self) -> None:
        """Observed 60 / 36; recomputed 0.4 * 0.9 / (0.9 * 0.1) = 4."""
        row = gap_row(SOURCE, TARGET, draws=200)
        assert row["false_alerts_observed"] == pytest.approx(60 / 36)
        assert row["false_alerts_recomputed"] == pytest.approx(4.0)

    def test_the_observed_ppv_carries_its_wilson_interval(self) -> None:
        """Wilson for 36 of 96 at 95%: centre 37.92 / 99.84 = 0.3798, half-width
        1.96 * sqrt(0.234375 * 96 + 0.9604) / 99.84 = 0.0951, so 0.2847 to 0.4749."""
        row = gap_row(SOURCE, TARGET, draws=200)
        assert row["ppv_observed_low"] == pytest.approx(0.2847, abs=5e-4)
        assert row["ppv_observed_high"] == pytest.approx(0.4749, abs=5e-4)

    def test_the_bootstrap_interval_brackets_the_gap(self) -> None:
        row = gap_row(SOURCE, TARGET, draws=2000, seed=1)
        assert row["gap_low"] < -0.175 < row["gap_high"]
        assert row["gap_high"] < 0.0

    def test_a_target_that_is_the_source_has_no_gap_and_an_interval_around_zero(self) -> None:
        """The control: sensitivity and specificity carried unchanged make the recipe exact."""
        big = Table(tp=900, fp=400, fn=100, tn=600)
        row = gap_row(big, big, draws=2000, seed=2)
        assert row["gap"] == pytest.approx(0.0, abs=1e-12)
        assert row["gap_low"] < 0.0 < row["gap_high"]

    def test_the_recipe_is_exact_at_any_prevalence_when_nothing_else_moves(self) -> None:
        """The invariant behind the controls: the target's own sensitivity and
        specificity, carried to its own prevalence, give back its observed PPV."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            tp, fp, fn, tn = (int(v) for v in rng.integers(1, 500, size=4))
            target = Table(tp=tp, fp=fp, fn=fn, tn=tn)
            row = gap_row(target, target, draws=10)
            assert row["ppv_recomputed"] == pytest.approx(tp / (tp + fp))

    def test_a_target_with_no_flag_has_no_ppv(self) -> None:
        row = gap_row(SOURCE, Table(tp=0, fp=0, fn=5, tn=95), draws=10)
        assert row["defined"] is False
        assert "gap" not in row


def _gaussian(
    n: int, prior: float, healthy_mean: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    y = (rng.random(n) < prior).astype(int)
    x = rng.normal(np.where(y == 1, 1.0, healthy_mean), 1.0)
    return x, y


def _posterior(x: np.ndarray, prior: float) -> np.ndarray:
    """The exact posterior at ``prior`` for unit Gaussians centred at +1 and -1."""
    odds = np.exp(2.0 * x) * prior / (1.0 - prior)
    return odds / (1.0 + odds)


class TestLabelShiftCheck:
    def test_under_label_shift_the_healthy_keep_a_mean_ratio_of_one(self) -> None:
        """The diagnostic must stay at 1 when only the prevalence moved."""
        rng = np.random.default_rng(0)
        xs, ys = _gaussian(20_000, 0.5, -1.0, rng)
        xt, yt = _gaussian(20_000, 0.2, -1.0, rng)
        check = label_shift_check(_posterior(xs, 0.5), ys, _posterior(xt, 0.5), yt)
        assert check["likelihood_ratio_healthy_target"] == pytest.approx(1.0, abs=0.1)
        assert check["likelihood_ratio_healthy_source"] == pytest.approx(1.0, abs=0.1)

    def test_healthier_healthy_patients_pull_the_ratio_below_one(self) -> None:
        """The ratio is exp(2x), whose mean over N(m, 1) is exp(2m + 2): 1 for the
        source healthy at m = -1, e^-1 = 0.37 for target healthy at m = -1.5."""
        # A source at prevalence 0.3, so the prior odds inside the ratio matter.
        rng = np.random.default_rng(1)
        xs, ys = _gaussian(40_000, 0.3, -1.0, rng)
        xt, yt = _gaussian(20_000, 0.2, -1.5, rng)
        check = label_shift_check(_posterior(xs, 0.3), ys, _posterior(xt, 0.3), yt)
        assert check["likelihood_ratio_healthy_source"] == pytest.approx(1.0, abs=0.1)
        assert check["likelihood_ratio_healthy_target"] == pytest.approx(np.exp(-1.0), abs=0.05)


def test_label_free_predictions_by_hand() -> None:
    """A calibrated source at prevalence 0.5 and a target whose probabilities
    are 0.8 for three in four records and 0.2 for the rest: the estimated
    prevalence is 11/12 (worked in test_repairs), and the mean probability of
    the flagged is that of the 0.8 group."""
    p_cal = np.r_[np.full(80, 0.8), np.full(20, 0.8), np.full(20, 0.2), np.full(80, 0.2)]
    y_cal = np.r_[np.ones(80), np.zeros(20), np.ones(20), np.zeros(80)].astype(int)
    p_target = np.r_[np.full(300, 0.8), np.full(100, 0.2)]
    flagged = p_target > 0.5
    out = label_free(p_cal, y_cal, p_target, flagged, sens=0.8, spec=0.8)
    assert out["prevalence_estimated"] == pytest.approx(11 / 12, abs=1e-6)
    assert out["ppv_mean_probability"] == pytest.approx(0.8)
    # The recipe at 11/12: 0.8 * 11/12 / (0.8 * 11/12 + 0.2 * 1/12) = 8.8 / 9.0
    assert out["ppv_recipe_estimated_prevalence"] == pytest.approx(8.8 / 9.0, abs=1e-6)
