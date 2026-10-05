"""The counts the report prints per 100 patients come from these functions; a wrong one
misstates what the threshold does to patients.

Every expected value below is worked out by hand from the rule, not computed by the
code under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecs.clinical import (
    holm,
    outcome_shares,
    paired_difference,
    per_thousand,
    predictive_values,
    rerandomised_ladder,
    standardised_coverage,
    subgroup_tests,
    two_halves,
)

# Columns: healthy admitted, ill admitted.
ILL_ALONE, HEALTHY_ALONE, BOTH, EMPTY = [False, True], [True, False], [True, True], [False, False]


class TestOutcomeShares:
    def test_a_deferred_patient_is_neither_caught_nor_missed(self) -> None:
        """Counting a deferral as caught is the error referee point 2 named."""
        sets = np.array([ILL_ALONE, BOTH, HEALTHY_ALONE, EMPTY, HEALTHY_ALONE, BOTH, ILL_ALONE])
        y = np.array([1, 1, 1, 1, 0, 0, 0])
        shares = outcome_shares(sets, y)
        ill, healthy = shares["ill"], shares["healthy"]
        assert ill["right_alone"]["count"] == 1
        assert ill["deferred"]["count"] == 2
        assert ill["wrong_alone"]["count"] == 1
        assert ill["right_alone"]["share"] == pytest.approx(0.25)
        assert healthy["right_alone"]["count"] == 1
        assert healthy["deferred"]["count"] == 1
        assert healthy["wrong_alone"]["count"] == 1
        assert healthy["wrong_alone"]["share"] == pytest.approx(1 / 3)

    def test_a_yes_or_no_threshold_defers_no_one(self) -> None:
        sets = np.array([ILL_ALONE, HEALTHY_ALONE, ILL_ALONE, HEALTHY_ALONE])
        shares = outcome_shares(sets, np.array([1, 1, 0, 0]))
        assert shares["ill"]["deferred"]["count"] == 0
        assert shares["ill"]["right_alone"]["share"] == pytest.approx(0.5)
        assert shares["healthy"]["wrong_alone"]["share"] == pytest.approx(0.5)


def test_predictive_values_by_bayes_rule() -> None:
    """Sensitivity 0.9, specificity 0.8, prevalence 0.1: of 1,000, 90 true and 180 false
    positives, 10 false and 720 true negatives."""
    pv = predictive_values(0.9, 0.8, 0.1)
    assert pv["ppv"] == pytest.approx(90 / 270)
    assert pv["npv"] == pytest.approx(720 / 730)


def test_counts_per_thousand_patients() -> None:
    """Prevalence 0.25, sensitivity 0.8, specificity 0.6: 250 ill, 200 found, 50 missed,
    300 of the 750 healthy flagged, 500 echocardiograms."""
    k = per_thousand(0.8, 0.6, 0.25)
    assert k == pytest.approx(
        {"flagged": 500.0, "ill_found": 200.0, "ill_missed": 50.0, "healthy_flagged": 300.0}
    )


class TestPairedDifference:
    def test_the_same_arm_twice_differs_by_nothing_on_every_draw(self) -> None:
        a = np.array([True, False, True, True])
        assert paired_difference(a, a) == {"difference": 0.0, "low": 0.0, "high": 0.0}

    def test_an_arm_that_catches_one_more_patient_in_four(self) -> None:
        a = np.array([True, True, True, False])
        b = np.array([True, True, False, False])
        d = paired_difference(a, b, n_draws=4000, seed=1)
        assert d["difference"] == pytest.approx(0.25)
        assert d["low"] >= 0.0 and d["high"] <= 1.0 and d["high"] > d["low"]

    def test_two_cohorts_of_different_size_are_refused(self) -> None:
        with pytest.raises(ValueError, match="same patients"):
            paired_difference(np.array([True]), np.array([True, False]))


def test_holm_steps_down_and_stays_monotone() -> None:
    """p = 0.01, 0.04, 0.03: sorted, 0.01 x 3 = 0.03, 0.03 x 2 = 0.06, 0.04 x 1 = 0.04,
    raised to 0.06 so the order holds."""
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_subgroup_tests_use_fisher_for_two_groups_and_chi_square_for_more() -> None:
    """Fisher's exact test on [[3, 1], [1, 3]] gives p = 34/70, about 0.486."""
    covered = np.array([True, True, True, False, True, False, False, False])
    sex = np.array(["female"] * 4 + ["male"] * 4)
    age = np.array(["a", "a", "b", "b", "c", "c", "a", "b"])
    rows = subgroup_tests(covered, {"sex": sex, "age": age})
    assert rows[0]["test"] == "fisher_exact"
    assert rows[0]["p"] == pytest.approx(34 / 70)
    assert rows[0]["groups"]["female"] == {"caught": 3, "n": 4}
    assert rows[1]["test"] == "chi2_contingency"


class TestLadder:
    def test_separable_scores_are_caught_and_cleared_by_the_source_threshold(self) -> None:
        y = np.array([1] * 40 + [0] * 60)
        p = np.where(y == 1, 0.9, 0.1).astype(float)
        (row,) = rerandomised_ladder(p, y, p, y, rungs=(0,), draws=5, alpha=0.10, level=0.90)
        assert row["sensitivity"]["mean"] == 1.0
        assert row["specificity"]["mean"] == 1.0

    def test_a_sample_with_too_few_ill_to_certify_flags_everyone(self) -> None:
        """Eight ill in the whole cohort: with n ill the threshold uses the
        ceil(0.9(n+1))-th lowest score, which for n of 8 or fewer is beyond the sample, so
        the refitted threshold flags every patient, ill and healthy."""
        y = np.array([1] * 8 + [0] * 92)
        p = np.where(y == 1, 0.9, 0.1).astype(float)
        (row,) = rerandomised_ladder(
            p, y, p, y, rungs=(10,), draws=20, seed=1, alpha=0.10, level=0.90
        )
        assert row["specificity"]["max"] == 0.0

    def test_each_draw_counts_the_ill_its_sample_held(self) -> None:
        """40 ill in 100: a sample of 20 holds 8 on average, and never more than 20."""
        rng = np.random.default_rng(0)
        y = np.array([1] * 40 + [0] * 60)
        p = np.clip(y * 0.3 + rng.uniform(0, 0.7, size=100), 0, 1)
        (row,) = rerandomised_ladder(
            p, y, p, y, rungs=(20,), draws=400, seed=3, alpha=0.10, level=0.90
        )
        assert row["ill_in_sample"]["mean"] == pytest.approx(8.0, abs=0.5)
        assert row["ill_in_sample"]["min"] >= 0 and row["ill_in_sample"]["max"] <= 20

    def test_the_sample_is_never_drawn_from_the_half_it_is_read_on(self) -> None:
        """The refit is read on patients its threshold never saw: the two halves of
        every cut are disjoint and together hold the whole cohort."""
        for n in (1, 2, 7, 1059):
            order = np.random.default_rng(n).permutation(n)
            pool, held = two_halves(order)
            assert not set(pool.tolist()) & set(held.tolist())
            assert sorted(pool.tolist() + held.tolist()) == list(range(n))
            assert len(held) - len(pool) in (0, 1)

    def test_the_halves_are_redrawn_so_rung_zero_varies(self) -> None:
        """A fixed half would give one sensitivity at rung zero; redrawn halves give a spread."""
        rng = np.random.default_rng(1)
        y = np.array([1] * 50 + [0] * 50)
        p = np.clip(y * 0.2 + rng.uniform(0, 0.8, size=100), 0, 1)
        (row,) = rerandomised_ladder(
            p, y, p, y, rungs=(0,), draws=50, seed=2, alpha=0.10, level=0.90
        )
        assert row["sensitivity"]["max"] > row["sensitivity"]["min"]


class TestCaseMix:
    def test_a_gap_that_is_all_case_mix_is_explained_whole(self) -> None:
        """Caught depends on severity only, and outpatients are mostly mild: the
        outpatients read at the inpatients' severity close the whole gap."""
        rng = np.random.default_rng(0)
        severe_in = rng.uniform(size=600) < 0.8
        severe_out = rng.uniform(size=600) < 0.2
        severe = np.r_[severe_in, severe_out]
        caught = rng.uniform(size=1200) < np.where(severe, 0.95, 0.55)
        outpatient = np.r_[np.zeros(600, bool), np.ones(600, bool)]
        result = standardised_coverage(
            caught, pd.DataFrame({"severe": severe.astype(float)}), outpatient, n_draws=50
        )
        assert result["share_explained"]["estimate"] == pytest.approx(1.0, abs=0.15)

    def test_a_gap_that_is_all_setting_is_explained_not_at_all(self) -> None:
        rng = np.random.default_rng(1)
        severe = rng.uniform(size=1200) < 0.5
        outpatient = np.r_[np.zeros(600, bool), np.ones(600, bool)]
        caught = rng.uniform(size=1200) < np.where(outpatient, 0.6, 0.9)
        result = standardised_coverage(
            caught, pd.DataFrame({"severe": severe.astype(float)}), outpatient, n_draws=50
        )
        assert result["share_explained"]["estimate"] == pytest.approx(0.0, abs=0.1)
