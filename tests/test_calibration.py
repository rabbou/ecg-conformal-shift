"""The calibration module against values worked out by hand in each test."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ecs.calibration import (
    brier,
    calibration_curve,
    calibration_in_the_large,
    decision_curve,
    false_alerts_per_detection,
    net_benefit,
    net_benefit_treat_all,
    ppv_at_prevalence,
    sensitivity_specificity,
    slope_intercept,
)


def logit(x: float) -> float:
    return math.log(x / (1 - x))


def two_groups(
    p1: float, r1: float, p2: float, r2: float, n: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """n records predicted p1 of which a share r1 are ill, and n predicted p2 with r2 ill."""
    y = np.r_[
        np.ones(round(n * r1)),
        np.zeros(n - round(n * r1)),
        np.ones(round(n * r2)),
        np.zeros(n - round(n * r2)),
    ]
    p = np.r_[np.full(n, p1), np.full(n, p2)]
    return y, p


class TestSlopeAndIntercept:
    def test_two_groups_give_the_line_through_both_observed_logits(self) -> None:
        # With two distinct predictions the logistic fit passes through both
        # observed rates exactly, so the slope is the ratio of logit differences.
        y, p = two_groups(0.2, 0.1, 0.6, 0.5)
        slope, _ = slope_intercept(y, p)
        expected = (logit(0.5) - logit(0.1)) / (logit(0.6) - logit(0.2))
        assert slope == pytest.approx(expected, abs=1e-8)

    def test_a_constant_prediction_gives_the_logit_gap_as_intercept(self) -> None:
        y = np.r_[np.ones(30), np.zeros(70)]
        p = np.full(100, 0.5)
        assert calibration_in_the_large(y, p) == pytest.approx(logit(0.3) - logit(0.5), abs=1e-8)

    def test_a_calibrated_two_group_model_has_slope_one_and_intercept_zero(self) -> None:
        y, p = two_groups(0.2, 0.2, 0.7, 0.7)
        slope, intercept = slope_intercept(y, p)
        assert slope == pytest.approx(1.0, abs=1e-8)
        assert intercept == pytest.approx(0.0, abs=1e-8)

    def test_a_saturating_offset_still_converges(self) -> None:
        # Predictions near zero for everyone while a third are ill: the first
        # undamped Newton step overshoots; the answer is still the logit gap.
        y = np.r_[np.ones(10), np.zeros(20)]
        p = np.full(30, 1e-5)
        assert calibration_in_the_large(y, p) == pytest.approx(logit(1 / 3) - logit(1e-5), abs=1e-6)

    def test_one_outcome_only_is_refused(self) -> None:
        with pytest.raises(ValueError, match="both outcomes"):
            slope_intercept(np.ones(5), np.full(5, 0.5))


def test_brier_is_the_mean_squared_error() -> None:
    # (0.9-1)^2 + (0.2-0)^2 + (0.6-1)^2 + (0.5-0)^2 = 0.01 + 0.04 + 0.16 + 0.25
    assert brier(np.array([1, 0, 1, 0]), np.array([0.9, 0.2, 0.6, 0.5])) == pytest.approx(0.115)


class TestNetBenefit:
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.3, 0.6, 0.1, 0.05])

    def test_counts_true_and_false_positives_at_the_threshold(self) -> None:
        # At t = 0.25: treated are 0.9, 0.3, 0.6 -> TP 2, FP 1, n 5, odds 1/3.
        assert net_benefit(self.y, self.p, 0.25) == pytest.approx(2 / 5 - 1 / 5 * (1 / 3))

    def test_treat_all_at_prevalence(self) -> None:
        # prevalence 0.4, t = 0.2: 0.4 - 0.6 * 0.25
        assert net_benefit_treat_all(0.4, 0.2) == pytest.approx(0.25)

    def test_decision_curve_puts_the_three_strategies_side_by_side(self) -> None:
        rows = decision_curve(self.y, self.p, np.array([0.25]))
        assert rows == [
            {
                "threshold": 0.25,
                "model": pytest.approx(1 / 3),
                "treat_all": pytest.approx(0.2),
                "treat_none": 0.0,
            }
        ]


class TestPredictiveValue:
    def test_ppv_by_bayes(self) -> None:
        # sens 0.8, spec 0.9, prevalence 0.1: 0.08 / (0.08 + 0.09)
        assert ppv_at_prevalence(0.8, 0.9, 0.1) == pytest.approx(0.08 / 0.17)

    def test_false_alerts_per_detection(self) -> None:
        # 0.09 healthy flagged per 0.08 ill found
        assert false_alerts_per_detection(0.8, 0.9, 0.1) == pytest.approx(0.09 / 0.08)

    def test_sensitivity_and_specificity(self) -> None:
        y = np.array([1, 1, 1, 0, 0])
        flagged = np.array([True, True, False, True, False])
        assert sensitivity_specificity(y, flagged) == (pytest.approx(2 / 3), pytest.approx(0.5))


def test_calibration_curve_bins_by_count_with_wilson_intervals() -> None:
    y = np.array([0, 0, 1, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    rows = calibration_curve(y, p, n_bins=2)
    assert [r["n"] for r in rows] == [3, 3]
    assert rows[0]["mean_predicted"] == pytest.approx(0.2)
    assert rows[0]["observed"] == pytest.approx(1 / 3)
    assert rows[1]["observed"] == pytest.approx(2 / 3)
    assert rows[0]["observed_low"] < 1 / 3 < rows[0]["observed_high"]
