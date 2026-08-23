from __future__ import annotations

import numpy as np
import pytest

from ecs.metrics import (
    abstention_rate,
    class_conditional_coverage,
    coverage,
    effective_sample_size,
    mean_set_size,
    singleton_rate,
    wilson_interval,
)


class TestCoverage:
    def test_counts_only_the_true_class_column(self) -> None:
        sets = np.array([[True, False], [False, True], [True, True], [False, False]])
        labels = np.array([0, 0, 1, 1])
        assert coverage(sets, labels) == 0.5

    def test_the_clinically_dangerous_case_is_visible_in_the_split(self) -> None:
        """Perfect coverage of a 90% healthy majority, zero coverage of the sick
        minority: the marginal number reads 0.9 and hides it entirely."""
        n_healthy, n_sick = 900, 100
        sets = np.zeros((n_healthy + n_sick, 2), dtype=bool)
        sets[:n_healthy, 0] = True
        labels = np.array([0] * n_healthy + [1] * n_sick)
        assert coverage(sets, labels) == pytest.approx(0.9)
        per_class = class_conditional_coverage(sets, labels)
        assert per_class[0] == (1.0, 900)
        assert per_class[1] == (0.0, 100)


class TestWilsonInterval:
    def test_endpoints_satisfy_the_score_equation_that_defines_them(self) -> None:
        """Stronger than asserting a remembered number: each endpoint p must
        satisfy |p_hat - p| = z * sqrt(p(1-p)/n)."""
        from scipy import stats

        successes, n = 90, 100
        lo, hi = wilson_interval(successes, n, confidence=0.95)
        z = float(stats.norm.ppf(0.975))
        p_hat = successes / n
        for endpoint in (lo, hi):
            expected = z * np.sqrt(endpoint * (1 - endpoint) / n)
            assert abs(p_hat - endpoint) == pytest.approx(expected, rel=1e-9)

    def test_brackets_the_point_estimate(self) -> None:
        lo, hi = wilson_interval(260, 1000)
        assert lo < 0.26 < hi

    def test_stays_inside_the_unit_interval_at_the_boundary(self) -> None:
        lo, hi = wilson_interval(0, 30)
        assert lo == pytest.approx(0.0, abs=1e-12)
        assert 0.0 < hi < 1.0

    def test_narrows_as_the_sample_grows(self) -> None:
        small = wilson_interval(90, 100)
        large = wilson_interval(9000, 10000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_rejects_impossible_counts(self) -> None:
        with pytest.raises(ValueError):
            wilson_interval(11, 10)


class TestSetShape:
    def test_size_singleton_and_abstention_are_consistent(self) -> None:
        sets = np.array(
            [
                [True, False, False],  # singleton
                [True, True, False],  # ambiguous
                [False, False, False],  # empty
                [True, True, True],  # everything
            ]
        )
        assert mean_set_size(sets) == pytest.approx((1 + 2 + 0 + 3) / 4)
        assert singleton_rate(sets) == 0.25
        assert abstention_rate(sets) == 0.75


class TestEffectiveSampleSize:
    def test_equal_weights_give_back_the_sample_size(self) -> None:
        assert effective_sample_size(np.ones(250)) == pytest.approx(250.0)

    def test_one_dominant_weight_collapses_it_to_about_one(self) -> None:
        w = np.concatenate([[1e6], np.ones(999)])
        assert effective_sample_size(w) < 1.01

    def test_never_exceeds_the_sample_size(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(20):
            w = rng.exponential(size=100)
            assert effective_sample_size(w) <= 100.0 + 1e-9

    def test_rejects_negative_weights(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            effective_sample_size(np.array([1.0, -0.5]))
