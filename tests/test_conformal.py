"""The coverage guarantee is a theorem, so it can be asserted, not eyeballed."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ecs.conformal import (
    aps_scores,
    aps_scores_all,
    conformal_quantile,
    lac_scores,
    lac_scores_all,
    predict_sets,
    weighted_conformal_quantile,
)


def _exchangeable_sample(
    rng: np.random.Generator, n: int, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Draw (probs, label) pairs from one joint distribution, so any split of
    the result is exchangeable -- the only assumption conformal needs."""
    logits = rng.normal(size=(n, k)) * 1.6
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(k, p=p) for p in probs])
    return probs, labels


class TestQuantile:
    def test_rank_is_the_finite_sample_corrected_order_statistic(self) -> None:
        # n=19, alpha=0.1 -> ceil(20 * 0.9) = 18 -> the 18th smallest score.
        scores = np.arange(1.0, 20.0)
        assert conformal_quantile(scores, 0.1) == 18.0

    def test_returns_infinity_when_the_sample_cannot_certify_the_level(self) -> None:
        # n=5, alpha=0.1 -> ceil(6 * 0.9) = 6 > 5. No finite threshold is valid.
        assert conformal_quantile(np.arange(5.0), 0.1) == math.inf

    def test_smallest_sample_that_can_certify_ninety_percent_is_nine(self) -> None:
        assert math.isinf(conformal_quantile(np.arange(8.0), 0.1))
        assert math.isfinite(conformal_quantile(np.arange(9.0), 0.1))

    @pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.5])
    def test_rejects_alpha_outside_the_open_unit_interval(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            conformal_quantile(np.arange(10.0), alpha)

    def test_rejects_empty_calibration(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            conformal_quantile(np.array([]), 0.1)


class TestScores:
    def test_lac_score_is_one_minus_the_true_class_probability(self) -> None:
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
        labels = np.array([0, 2])
        np.testing.assert_allclose(lac_scores(probs, labels), [0.3, 0.2])

    def test_unrandomised_aps_score_of_the_last_ranked_class_is_one(self) -> None:
        probs = np.array([[0.5, 0.3, 0.2]])
        scores = aps_scores_all(probs, randomised=False)
        assert scores[0, 2] == pytest.approx(1.0)
        assert scores[0, 0] == pytest.approx(0.5)
        assert scores[0, 1] == pytest.approx(0.8)

    def test_randomised_aps_score_stays_within_its_rank_band(self) -> None:
        rng = np.random.default_rng(0)
        probs = np.array([[0.5, 0.3, 0.2]])
        scores = aps_scores_all(probs, rng=rng, randomised=True)
        assert 0.0 <= scores[0, 0] <= 0.5
        assert 0.5 <= scores[0, 1] <= 0.8
        assert 0.8 <= scores[0, 2] <= 1.0

    def test_rejects_rows_that_are_not_distributions(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            lac_scores_all(np.array([[0.5, 0.2, 0.1]]))


class TestCoverageGuarantee:
    """Split conformal must cover at 1 - alpha under exchangeability, for any
    underlying model.  Averaged over many splits the empirical rate should sit
    in [1 - alpha, 1 - alpha + 1/(n_cal + 1)]."""

    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
    @pytest.mark.parametrize("score", ["lac", "aps"])
    def test_marginal_coverage_matches_the_nominal_level(self, alpha: float, score: str) -> None:
        rng = np.random.default_rng(20260822)
        n_cal, n_test, trials = 300, 400, 250
        covered = 0
        total = 0
        for _ in range(trials):
            probs, labels = _exchangeable_sample(rng, n_cal + n_test)
            cal_p, test_p = probs[:n_cal], probs[n_cal:]
            cal_y, test_y = labels[:n_cal], labels[n_cal:]
            if score == "lac":
                cal_s = lac_scores(cal_p, cal_y)
                test_all = lac_scores_all(test_p)
            else:
                cal_s = aps_scores(cal_p, cal_y, rng=rng)
                test_all = aps_scores_all(test_p, rng=rng)
            qhat = conformal_quantile(cal_s, alpha)
            sets = predict_sets(test_all, qhat)
            covered += int(sets[np.arange(n_test), test_y].sum())
            total += n_test
        empirical = covered / total
        assert empirical >= (1 - alpha) - 0.012, f"under-covered: {empirical:.4f}"
        assert empirical <= (1 - alpha) + 0.025, f"over-covered: {empirical:.4f}"

    def test_a_useless_model_still_covers_by_widening_its_sets(self) -> None:
        """Coverage does not require accuracy.  A model that outputs noise must
        still reach 1 - alpha, and must pay for it in set size."""
        rng = np.random.default_rng(7)
        n_cal, n_test, k = 500, 2000, 5
        probs = np.full((n_cal + n_test, k), 1.0 / k)
        labels = rng.integers(0, k, size=n_cal + n_test)
        cal_s = lac_scores(probs[:n_cal], labels[:n_cal])
        qhat = conformal_quantile(cal_s, 0.1)
        sets = predict_sets(lac_scores_all(probs[n_cal:]), qhat)
        assert sets[np.arange(n_test), labels[n_cal:]].mean() == 1.0
        assert sets.sum(axis=1).mean() == float(k)


class TestWeightedQuantile:
    def test_uniform_weights_reproduce_the_unweighted_quantile(self) -> None:
        """With w_i = w(x) = 1 the weighted rank collapses to
        ceil((n + 1)(1 - alpha)), so the two calibrations must agree exactly."""
        rng = np.random.default_rng(3)
        scores = rng.uniform(size=40)
        plain = conformal_quantile(scores, 0.1)
        weighted = weighted_conformal_quantile(scores, np.ones(40), np.ones(6), alpha=0.1)
        np.testing.assert_allclose(weighted, plain)

    def test_scaling_every_weight_leaves_the_quantile_unchanged(self) -> None:
        rng = np.random.default_rng(11)
        scores = rng.uniform(size=60)
        w = rng.uniform(0.5, 2.0, size=60)
        a = weighted_conformal_quantile(scores, w, np.array([1.3]), 0.1)
        b = weighted_conformal_quantile(scores, 10 * w, np.array([13.0]), 0.1)
        np.testing.assert_allclose(a, b)

    def test_an_overwhelming_test_weight_forces_the_infinite_quantile(self) -> None:
        """When the likelihood ratio says the test point is unlike anything in
        calibration, the threshold must be +inf -- predict everything."""
        scores = np.linspace(0, 1, 50)
        q = weighted_conformal_quantile(scores, np.ones(50), np.array([1e9]), 0.1)
        assert math.isinf(q[0])

    def test_one_quantile_is_returned_per_test_point(self) -> None:
        scores = np.linspace(0, 1, 30)
        q = weighted_conformal_quantile(scores, np.ones(30), np.array([0.5, 1.0, 4.0]), 0.1)
        assert q.shape == (3,)
        assert np.all(np.diff(q) >= 0), "a heavier test weight cannot lower the threshold"

    def test_rejects_negative_weights(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            weighted_conformal_quantile(
                np.arange(5.0), np.array([1.0, -1.0, 1.0, 1.0, 1.0]), np.ones(1), 0.1
            )
