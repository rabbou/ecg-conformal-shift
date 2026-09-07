"""The target-scale ladder: what labelled target records buy, and what they cost.

C-29.  Four rungs on one pair, PTB-XL to Chongqing, infarction.  The claims worth
holding are the ones that make the ladder readable rather than the numbers on it:
rung zero is the frozen source threshold and nothing else; the rungs above it
calibrate on target records and only on target records; every rung is measured on
the same held-out half; and a rung whose class is too small for the level says so
with an infinite threshold instead of a finite one it cannot support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from target_scale import draw_target_records, rows_by_patient

from ecs.config import RESULTS_DIR

PATH = Path(RESULTS_DIR) / "target_scale.json"


@pytest.fixture(scope="module")
def ladder() -> dict[str, Any]:
    if not PATH.exists():
        pytest.skip(f"{PATH} is not built; run scripts/target_scale.py")
    return json.loads(PATH.read_text())


def rows(ladder: dict[str, Any], **wanted: Any) -> list[dict[str, Any]]:
    return [r for r in ladder["rows"] if all(r[k] == v for k, v in wanted.items())]


class TestTheLadderHasItsFourRungs:
    def test_the_rungs_are_the_ones_the_card_asks_for(self, ladder: dict[str, Any]) -> None:
        assert ladder["settings"]["rungs"] == [0, 100, 500, 2000]

    def test_every_rung_is_measured_under_both_families(self, ladder: dict[str, Any]) -> None:
        for family in ("recalibrated", "pooled"):
            for rung in (0, 100, 500, 2000):
                found = rows(ladder, family=family, n_target_records=rung)
                assert found, (family, rung)

    def test_every_figure_is_a_mean_over_two_hundred_draws_with_its_spread(
        self, ladder: dict[str, Any]
    ) -> None:
        assert ladder["settings"]["n_draws"] >= 200
        for row in ladder["rows"]:
            for block in (row["coverage"], *row["coverage_by_class"].values()):
                assert block["n_draws"] >= 200
                assert "sd" in block

    def test_the_pair_is_the_one_the_break_table_left_off_at(self, ladder: dict[str, Any]) -> None:
        assert ladder["pair"]["source"] == "ptbxl"
        assert ladder["pair"]["target"] == "acs"
        assert ladder["pair"]["label"] == "infarction"


class TestWhatEachRungWasCalibratedOn:
    def test_rung_zero_is_the_frozen_source_threshold_under_both_families(
        self, ladder: dict[str, Any]
    ) -> None:
        """The ladder starts where the break table stopped, so the two agree exactly."""
        for correction in ("none", "mondrian"):
            for alpha in ladder["settings"]["alphas"]:
                pair = rows(
                    ladder, n_target_records=0, correction=correction, alpha=alpha, score="lac"
                )
                assert len(pair) == 2, (correction, alpha)
                assert pair[0]["threshold_by_class"] == pair[1]["threshold_by_class"]
                assert pair[0]["coverage"] == pair[1]["coverage"]
                assert all(r["calibrated_on"] == "PTB-XL alone" for r in pair)

    def test_a_recalibrated_rung_holds_exactly_that_many_target_records(
        self, ladder: dict[str, Any]
    ) -> None:
        for rung in (100, 500, 2000):
            for row in rows(ladder, family="recalibrated", n_target_records=rung):
                assert row["calibration"]["n"]["mean"] == float(rung)
                assert row["calibration"]["n"]["sd"] == 0.0

    def test_a_pooled_rung_holds_the_source_half_as_well(self, ladder: dict[str, Any]) -> None:
        source_only = rows(ladder, family="pooled", n_target_records=0)[0]
        for rung in (100, 500, 2000):
            for row in rows(ladder, family="pooled", n_target_records=rung):
                grew = row["calibration"]["n"]["mean"] - source_only["calibration"]["n"]["mean"]
                assert abs(grew - rung) < 1.0, (rung, grew)

    def test_the_weighted_correction_appears_at_rung_zero_only(
        self, ladder: dict[str, Any]
    ) -> None:
        """Above rung zero there is no source prior left to reweight from."""
        weighted = rows(ladder, correction="weighted")
        assert weighted
        assert {row["n_target_records"] for row in weighted} == {0}

    def test_the_evaluation_half_is_never_calibrated_on(self, ladder: dict[str, Any]) -> None:
        split = ladder["split"]
        assert split["n_pool"] > 0 and split["n_eval"] > 0
        # results/external/acs.npz holds 17,955 scored records; the split accounts
        # for every one of them and for every patient behind them.
        assert split["n_pool"] + split["n_eval"] == 17955
        assert split["n_pool_patients"] + split["n_eval_patients"] == 17013


class TestWhatTheLadderShows:
    def test_a_thin_rung_says_so_with_an_infinite_threshold(self, ladder: dict[str, Any]) -> None:
        """At 100 records the sick class carries about fifteen; some draws cannot
        certify the level and return +inf, which is the reading, not a failure."""
        counted = [
            row["threshold_by_class"]["1"]["n_infinite"]
            for row in rows(ladder, family="recalibrated")
            if row["n_target_records"] == 100
        ]
        assert any(n > 0 for n in counted), "no rung of 100 records ever ran out of class"

    def test_the_spread_narrows_as_the_rung_grows(self, ladder: dict[str, Any]) -> None:
        sd = {
            row["n_target_records"]: row["coverage"]["sd"]
            for row in rows(
                ladder, family="recalibrated", correction="none", alpha=0.10, score="lac"
            )
            if row["n_target_records"] > 0
        }
        assert sd[100] > sd[2000]

    def test_the_headline_setting_is_on_the_file(self, ladder: dict[str, Any]) -> None:
        headline = ladder["settings"]["headline"]
        assert headline["alpha"] == 0.10
        assert headline["score"] == "lac"


class TestTheDrawIsByPatient:
    def test_a_drawn_sample_keeps_whole_patients_and_stays_inside_the_pool(self) -> None:
        patients = np.array([f"P{i // 3:03d}" for i in range(300)])
        pool = np.arange(0, 200)
        groups = rows_by_patient(pool, patients)
        drawn = draw_target_records(groups, 60, np.random.default_rng(0))
        # At most the rung, never over it: a patient that would overflow is
        # skipped rather than split across the boundary.
        assert 0 < len(drawn) <= 60
        assert set(drawn) <= set(pool.tolist())
        for patient in {patients[r] for r in drawn}:
            whole = {int(r) for r in pool if patients[r] == patient}
            assert whole <= set(drawn.tolist()), patient
