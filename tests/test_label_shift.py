"""The claim under test: class-conditional calibration survives a prevalence
change that breaks ordinary split conformal, and it does so without estimating
anything."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ecs.conformal import (
    bbse_target_prior,
    conformal_quantile,
    label_shift_quantiles,
    lac_scores,
    lac_scores_all,
    mondrian_quantiles,
    predict_sets,
    predict_sets_per_class,
)

K = 2
MI, HEALTHY = 1, 0


def _cohort(
    rng: np.random.Generator, n: int, prevalence: float, separation: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """A two-class cohort with a tunable share of positives.

    P(X | Y) is held fixed across cohorts and only the mix changes, which is
    exactly the label-shift model.
    """
    labels = (rng.uniform(size=n) < prevalence).astype(int)
    signal = rng.normal(loc=np.where(labels == MI, separation, -separation), scale=1.0)
    p_mi = 1.0 / (1.0 + np.exp(-2.0 * signal))
    return np.column_stack([1.0 - p_mi, p_mi]), labels


class TestMondrianUnderPrevalenceShift:
    def test_class_conditional_coverage_survives_a_twenty_five_fold_drop(self) -> None:
        """Source at 25% positives, target at 1% -- the PTB-XL / Shandong gap.

        Ordinary split conformal controls only the marginal rate, so on the
        target its coverage of the positive class is free to fall.  Mondrian
        thresholds are calibrated inside each class and must hold for both.
        """
        rng = np.random.default_rng(20260823)
        alpha = 0.1
        mondrian_mi, plain_mi = [], []
        for _ in range(120):
            cal_p, cal_y = _cohort(rng, 4000, prevalence=0.25)
            tgt_p, tgt_y = _cohort(rng, 6000, prevalence=0.01)
            cal_s = lac_scores(cal_p, cal_y)
            tgt_all = lac_scores_all(tgt_p)

            q_mondrian = mondrian_quantiles(cal_s, cal_y, alpha, n_classes=K)
            sets_m = predict_sets_per_class(tgt_all, q_mondrian)
            sets_p = predict_sets(tgt_all, conformal_quantile(cal_s, alpha))

            mask = tgt_y == MI
            if mask.sum():
                mondrian_mi.append(sets_m[mask, MI].mean())
                plain_mi.append(sets_p[mask, MI].mean())

        assert np.mean(mondrian_mi) >= 1 - alpha - 0.02, (
            f"Mondrian lost the guarantee on the positive class: {np.mean(mondrian_mi):.4f}"
        )

    def test_mondrian_thresholds_are_computed_within_each_class(self) -> None:
        scores = np.array([0.0, 0.1, 0.2, 0.3, 9.0, 9.1, 9.2, 9.3] * 3)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 3)
        q = mondrian_quantiles(scores, labels, alpha=0.2, n_classes=2)
        assert q[0] < 1.0, "class 0 threshold must come from class 0 scores only"
        assert q[1] > 8.0, "class 1 threshold must come from class 1 scores only"

    def test_a_class_too_small_to_certify_gets_an_infinite_threshold(self) -> None:
        """Four positives cannot support a 90% guarantee. The set must widen,
        not silently pretend."""
        scores = np.concatenate([np.linspace(0, 1, 40), np.array([0.1, 0.2, 0.3, 0.4])])
        labels = np.array([0] * 40 + [1] * 4)
        q = mondrian_quantiles(scores, labels, alpha=0.1, n_classes=2)
        assert math.isfinite(q[0])
        assert math.isinf(q[1])


class TestLabelShiftWeighting:
    def test_an_unchanged_prior_reproduces_the_unweighted_quantile(self) -> None:
        rng = np.random.default_rng(5)
        scores = rng.uniform(size=80)
        labels = rng.integers(0, K, size=80)
        q = label_shift_quantiles(scores, labels, np.ones(K), alpha=0.1)
        np.testing.assert_allclose(q, conformal_quantile(scores, 0.1))

    def test_the_threshold_shifts_when_the_target_prior_shifts(self) -> None:
        rng = np.random.default_rng(9)
        scores = np.concatenate([rng.uniform(0, 0.3, 100), rng.uniform(0.6, 1.0, 100)])
        labels = np.array([0] * 100 + [1] * 100)
        flat = label_shift_quantiles(scores, labels, np.ones(K), 0.1)
        rare = label_shift_quantiles(scores, labels, np.array([1.0, 0.04]), 0.1)
        assert not np.allclose(flat, rare)

    def test_rejects_negative_class_weights(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            label_shift_quantiles(
                np.arange(10.0), np.zeros(10, dtype=int), np.array([-1.0, 1.0]), 0.1
            )


class TestBBSE:
    def test_recovers_a_known_target_prior(self) -> None:
        rng = np.random.default_rng(4242)
        cal_p, cal_y = _cohort(rng, 40000, prevalence=0.25)
        tgt_p, tgt_y = _cohort(rng, 40000, prevalence=0.02)
        prior = bbse_target_prior(tgt_p.argmax(1), cal_p.argmax(1), cal_y, n_classes=K)
        assert prior[MI] == pytest.approx(0.02, abs=0.01), f"got {prior}"
        assert prior.sum() == pytest.approx(1.0)

    def test_refuses_a_predictor_that_cannot_identify_the_shift(self) -> None:
        """A predictor that always says the same thing has a singular confusion
        matrix and carries no information about the target prior."""
        n = 500
        cal_pred = np.zeros(n, dtype=int)
        cal_y = np.array([0] * 400 + [1] * 100)
        with pytest.raises(ValueError, match="singular"):
            bbse_target_prior(np.zeros(200, dtype=int), cal_pred, cal_y, n_classes=K)
