"""The linear probe and the paired comparison, on data whose answer is known."""

from __future__ import annotations

import numpy as np
import pytest

from ecs.arms import (
    REGULARISATION_GRID,
    auprc,
    auroc,
    fit_probe,
    fitting_statistics,
    paired_difference,
    standardise,
)
from ecs.metrics import bootstrap_ci


def separable(n: int, dim: int, gap: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` rows in two Gaussian clouds ``gap`` apart on the first dimension."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    x = rng.normal(size=(len(y), dim))
    x[:, 0] += y * gap
    return x, y


class TestTheStandardisation:
    """Statistics come from the fitting fold, and nothing dies on the way."""

    def test_a_constant_dimension_does_not_become_a_nan(self) -> None:
        x = np.column_stack([np.arange(10.0), np.full(10, 3.0)])
        centre, scale = fitting_statistics(x)
        assert scale[1] == 1.0
        z = standardise(x, centre, scale)
        assert np.isfinite(z).all()
        assert (z[:, 1] == 0.0).all()

    def test_the_fitting_fold_is_centred_on_itself(self) -> None:
        x, _ = separable(200, 5, 2.0, seed=0)
        centre, scale = fitting_statistics(x)
        z = standardise(x, centre, scale)
        assert np.allclose(z.mean(axis=0), 0.0, atol=1e-12)
        assert np.allclose(z.std(axis=0), 1.0, atol=1e-12)

    def test_another_fold_is_scaled_by_the_fitting_fold_and_not_by_itself(self) -> None:
        """The property that keeps a target from being re-calibrated on itself."""
        train, _ = separable(200, 4, 2.0, seed=0)
        centre, scale = fitting_statistics(train)
        shifted = train + 7.0
        z = standardise(shifted, centre, scale)
        assert np.allclose(z.mean(axis=0), 7.0 / scale, atol=1e-9)


class TestTheProbe:
    """A linear head on a frozen representation."""

    def test_it_separates_a_separable_representation(self) -> None:
        train_x, train_y = separable(2000, 8, 3.0, seed=0)
        validation_x, validation_y = separable(600, 8, 3.0, seed=1)
        test_x, test_y = separable(600, 8, 3.0, seed=2)
        probe = fit_probe(train_x, train_y, validation_x, validation_y)
        scores = probe.head.probabilities(test_x)[:, 1]
        assert auroc(test_y, scores) > 0.95

    def test_it_finds_nothing_in_a_representation_holding_nothing(self) -> None:
        train_x, train_y = separable(2000, 8, 0.0, seed=0)
        validation_x, validation_y = separable(600, 8, 0.0, seed=1)
        test_x, test_y = separable(2000, 8, 0.0, seed=2)
        probe = fit_probe(train_x, train_y, validation_x, validation_y)
        scores = probe.head.probabilities(test_x)[:, 1]
        assert 0.4 < auroc(test_y, scores) < 0.6

    def test_the_strength_it_keeps_is_the_one_the_validation_fold_scored_highest(self) -> None:
        train_x, train_y = separable(400, 6, 1.0, seed=0)
        validation_x, validation_y = separable(400, 6, 1.0, seed=1)
        probe = fit_probe(train_x, train_y, validation_x, validation_y)
        assert set(probe.validation_auroc) == set(REGULARISATION_GRID)
        best = max(probe.validation_auroc, key=lambda c: probe.validation_auroc[c])
        assert probe.chosen == best

    def test_the_head_carries_the_standardisation_it_was_fitted_under(self) -> None:
        train_x, train_y = separable(400, 6, 1.5, seed=0)
        validation_x, validation_y = separable(400, 6, 1.5, seed=1)
        probe = fit_probe(train_x, train_y, validation_x, validation_y)
        centre, scale = fitting_statistics(train_x)
        assert np.allclose(probe.head.centre, centre)
        assert np.allclose(probe.head.scale, scale)

    def test_probabilities_are_one_row_per_record_and_sum_to_one(self) -> None:
        train_x, train_y = separable(400, 6, 1.5, seed=0)
        validation_x, validation_y = separable(400, 6, 1.5, seed=1)
        probe = fit_probe(train_x, train_y, validation_x, validation_y)
        probs = probe.head.probabilities(separable(120, 6, 1.5, seed=2)[0])
        assert probs.shape == (120, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)


class TestPairedDifference:
    """Comparing two arms that scored the same records."""

    @staticmethod
    def two_arms(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Labels, a strong arm and a weak arm, both scoring the same records.

        The weak arm is the strong one plus noise, which is the shape a real
        pair of arms takes: they agree about most records and differ about a
        few.  The gap is deliberately small enough that each arm's own interval
        swallows the other's, so a comparison that ignores the pairing cannot
        see it.
        """
        rng = np.random.default_rng(seed)
        labels = np.repeat([0, 1], 1000)
        strong = rng.normal(loc=labels * 1.6, scale=1.0)
        weak = strong + rng.normal(scale=0.5, size=len(labels))
        return labels, strong, weak

    def test_an_arm_compared_with_itself_is_a_difference_of_exactly_zero(self) -> None:
        labels, strong, _ = self.two_arms()
        result = paired_difference(auroc, labels, strong, strong)
        assert result["difference"] == 0.0
        assert result["ci95_low"] == 0.0 and result["ci95_high"] == 0.0
        assert result["separated"] is False

    def test_a_better_arm_is_separated_from_a_worse_one(self) -> None:
        labels, strong, weak = self.two_arms()
        result = paired_difference(auroc, labels, strong, weak)
        assert result["difference"] > 0.0
        assert result["ci95_low"] > 0.0
        assert result["separated"] is True

    def test_the_difference_reverses_when_the_arms_change_places(self) -> None:
        labels, strong, weak = self.two_arms()
        forward = paired_difference(auroc, labels, strong, weak)
        backward = paired_difference(auroc, labels, weak, strong)
        assert forward["difference"] == pytest.approx(-backward["difference"])
        assert forward["ci95_low"] == pytest.approx(-backward["ci95_high"], abs=1e-9)

    def test_pairing_is_tighter_than_two_separate_intervals(self) -> None:
        """Why the comparison is paired at all.

        The arms scored the same records, so a draw that happens to pick easy
        records flatters both.  Pairing removes that shared movement; comparing
        two independently drawn intervals leaves it in and would call two arms
        indistinguishable that are in fact cleanly separated.
        """
        labels, strong, weak = self.two_arms()
        paired = paired_difference(auroc, labels, strong, weak)
        _, strong_low, strong_high = bootstrap_ci(auroc, labels, strong)
        _, weak_low, weak_high = bootstrap_ci(auroc, labels, weak)
        paired_width = paired["ci95_high"] - paired["ci95_low"]
        separate_width = (strong_high - strong_low) + (weak_high - weak_low)
        assert paired_width < separate_width
        assert strong_low < weak_high  # the two separate intervals overlap
        assert paired["separated"] is True  # the paired one does not

    def test_it_compares_auprc_on_the_same_footing_as_auroc(self) -> None:
        labels, strong, weak = self.two_arms()
        result = paired_difference(auprc, labels, strong, weak)
        assert result["difference"] > 0.0
        assert result["ci95_low"] > 0.0

    def test_two_arms_that_did_not_score_the_same_records_cannot_be_paired(self) -> None:
        labels, strong, weak = self.two_arms()
        with pytest.raises(ValueError, match="one label per score on both arms"):
            paired_difference(auroc, labels, strong, weak[:-1])

    def test_a_sample_too_small_for_an_interval_is_refused_rather_than_estimated(self) -> None:
        labels = np.array([0] + [1] * 40)
        scores = np.linspace(0.0, 1.0, len(labels))
        with pytest.raises(ValueError, match="too small for a paired interval"):
            paired_difference(auroc, labels, scores, scores[::-1])

    def test_the_share_of_draws_agreeing_matches_the_interval(self) -> None:
        labels, strong, weak = self.two_arms()
        separated = paired_difference(auroc, labels, strong, weak)
        assert separated["share_of_draws_on_the_same_side"] > 0.975
        same = paired_difference(auroc, labels, strong, strong + 1e-12)
        assert same["separated"] is False
