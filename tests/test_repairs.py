"""The three repairs and their net benefit, on values worked out by hand."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ecs.repairs import (
    abstention_rules,
    adjust_to_prior,
    estimate_prior,
    local_recalibration,
    net_benefit_of,
    prior_shift,
    repair_cell,
)


def logit(x: float) -> float:
    return math.log(x / (1 - x))


def gaussian(n: int, prior: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y = (rng.random(n) < prior).astype(int)
    return rng.normal(np.where(y == 1, 1.0, -1.0), 1.0), y


def posterior(x: np.ndarray, prior: float) -> np.ndarray:
    odds = np.exp(2.0 * x) * prior / (1.0 - prior)
    return odds / (1.0 + odds)


class TestAdjustToPrior:
    def test_even_odds_move_to_the_new_prevalence(self) -> None:
        """A probability equal to the old prevalence becomes the new one."""
        assert adjust_to_prior(np.array([0.5]), 0.5, 0.2)[0] == pytest.approx(0.2)

    def test_odds_are_multiplied_by_the_prior_odds_ratio(self) -> None:
        """0.8 is odds 4; from 0.5 to 0.2 multiplies odds by 1/4, giving odds 1, so 0.5."""
        assert adjust_to_prior(np.array([0.8]), 0.5, 0.2)[0] == pytest.approx(0.5)


class TestEstimatePrior:
    def test_three_in_four_at_point_eight(self) -> None:
        """At source prevalence 0.5, 300 records at 0.8 and 100 at 0.2.  The score
        equation 0.75 * 1.2 / (1.2q + 0.4) = 0.25 * 1.2 / (1.6 - 1.2q) gives
        3 (1.6 - 1.2q) = 1.2q + 0.4, so q = 4.4 / 4.8 = 11/12."""
        p = np.r_[np.full(300, 0.8), np.full(100, 0.2)]
        assert estimate_prior(p, 0.5) == pytest.approx(11 / 12, abs=1e-9)

    def test_a_symmetric_site_keeps_the_source_prevalence(self) -> None:
        p = np.r_[np.full(200, 0.8), np.full(200, 0.2)]
        assert estimate_prior(p, 0.5) == pytest.approx(0.5, abs=1e-9)

    def test_a_site_that_looks_healthy_everywhere_is_estimated_at_zero(self) -> None:
        """Every likelihood ratio below 1: the likelihood rises all the way to 0,
        the boundary EM only crawls toward."""
        assert estimate_prior(np.full(50, 0.2), 0.5) == 0.0

    def test_a_site_that_looks_ill_everywhere_is_estimated_at_one(self) -> None:
        assert estimate_prior(np.full(50, 0.9), 0.5) == 1.0

    def test_recovers_a_pure_prevalence_change(self) -> None:
        """Exact posteriors at 0.5, a site at 0.2: the estimate lands on 0.2."""
        rng = np.random.default_rng(0)
        x, _ = gaussian(40_000, 0.2, rng)
        assert estimate_prior(posterior(x, 0.5), 0.5) == pytest.approx(0.2, abs=0.01)


def test_prior_shift_corrects_a_model_fitted_at_another_prevalence() -> None:
    """The danger the source recalibration removes: probabilities that are
    posteriors at a training prevalence of 0.3, read at a source of 0.5.  Fed
    straight to EM with the source prevalence, the site at 0.2 comes out near
    0.08; recalibrated on the source first, it comes out at 0.2."""
    rng = np.random.default_rng(3)
    xs, ys = gaussian(40_000, 0.5, rng)
    xt, _ = gaussian(40_000, 0.2, rng)
    p_source, p_target = posterior(xs, 0.3), posterior(xt, 0.3)
    assert estimate_prior(p_target, 0.5) < 0.12
    prior, _ = prior_shift(p_source, ys, p_target)
    assert prior == pytest.approx(0.2, abs=0.01)


class TestLocalRecalibration:
    def test_no_ill_patient_leaves_the_model_as_delivered(self) -> None:
        assert local_recalibration(np.full(100, 0.3), np.zeros(100, dtype=int)) == (
            0.0,
            1.0,
            "none",
        )

    def test_few_events_fit_the_intercept_alone(self) -> None:
        """3 ill of 100 at a constant 0.5: the shift is logit(0.03) - logit(0.5)."""
        y = np.r_[np.ones(3), np.zeros(97)].astype(int)
        a, b, kind = local_recalibration(np.full(100, 0.5), y)
        assert kind == "intercept"
        assert b == 1.0
        assert a == pytest.approx(logit(0.03), abs=1e-8)

    def test_three_events_never_fit_a_slope_even_when_one_could_be_fitted(self) -> None:
        """Two distinct predictions would identify a slope; three events must not buy one."""
        y = np.r_[np.ones(1), np.zeros(49), np.ones(2), np.zeros(48)].astype(int)
        p = np.r_[np.full(50, 0.2), np.full(50, 0.6)]
        _, b, kind = local_recalibration(p, y)
        assert (kind, b) == ("intercept", 1.0)

    def test_enough_events_fit_intercept_and_slope(self) -> None:
        """Two groups, 0.2 predicted with 10% ill and 0.6 with 50%: the line
        through both observed logits."""
        y = np.r_[np.ones(10), np.zeros(90), np.ones(50), np.zeros(50)].astype(int)
        p = np.r_[np.full(100, 0.2), np.full(100, 0.6)]
        a, b, kind = local_recalibration(p, y)
        slope = (logit(0.5) - logit(0.1)) / (logit(0.6) - logit(0.2))
        assert kind == "intercept_and_slope"
        assert b == pytest.approx(slope, abs=1e-8)
        assert a == pytest.approx(logit(0.1) - slope * logit(0.2), abs=1e-8)


def test_net_benefit_of_a_decision_by_hand() -> None:
    """Treated: two ill and one healthy of five, at t = 0.25: 2/5 - 1/5 * 1/3."""
    y = np.array([1, 1, 0, 0, 0])
    treated = np.array([True, True, True, False, False])
    assert net_benefit_of(y, treated, 0.25) == pytest.approx(2 / 5 - 1 / 15)


def test_abstention_rules_by_hand() -> None:
    """Nine ill at 0.2 to 0.95 and nine healthy at 0.01 to 0.5.  At 90% with
    nine per class the conformal rank is ceil(10 * 0.9) = 9, the largest score:
    disease stays in the set from p = 0.2 (the lowest ill probability), and
    health stays in it up to p = 0.5 (the highest healthy probability)."""
    p_cal = np.r_[[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], np.linspace(0.01, 0.5, 9)]
    y_cal = np.r_[np.ones(9), np.zeros(9)].astype(int)
    p = np.array([0.1, 0.3, 0.6])
    rule_in, abstained = abstention_rules(p_cal, y_cal, p)
    assert rule_in.tolist() == [False, False, True]
    assert abstained.tolist() == [False, True, False]


def test_repair_cell_reads_every_rule_on_the_evaluation_half() -> None:
    """Treat-all on the evaluation half is prevalence - (1 - prevalence) t / (1 - t),
    and the recalibration was fitted once per draw."""
    rng = np.random.default_rng(5)
    xs, ys = gaussian(2_000, 0.5, rng)
    xt, yt = gaussian(2_000, 0.2, rng)
    patients = pd.Series([f"p{i // 2}" for i in range(2_000)])
    cell = repair_cell(posterior(xs, 0.5), ys, posterior(xt, 0.5), yt, patients, draws=20)
    assert cell["n_pool"] + cell["n_eval"] == 2_000
    assert sum(cell["recalibration_fits"].values()) == 20
    prev = cell["prevalence_eval"]
    for row in cell["net_benefit"]:
        t = row["threshold"]
        assert row["treat_all"] == pytest.approx(prev - (1 - prev) * t / (1 - t))
    assert cell["prevalence_estimated"] == pytest.approx(prev, abs=0.04)
