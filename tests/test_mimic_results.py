"""The committed MIMIC rotation and shadow results, recomputed from the committed scores.

``results/mimic_rotation.json`` and ``results/mimic_shadow.json`` are written by
``scripts/mimic_rotation.py`` and ``scripts/mimic_shadow.py`` from the score
files beside them.  These tests redo the coverage arithmetic from those score
files, so a table that drifted from its scores, or a Wilson interval that does
not hold its own point, fails here without any corpus on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ecs.config import RESULTS_DIR
from ecs.rotation import EXTENDED_SOURCES, class_keys, usable_classes
from ecs.transfer import ALPHA, conformal_sets, coverage_row

ROTATION = Path(RESULTS_DIR) / "mimic_rotation.json"
SHADOW = Path(RESULTS_DIR) / "mimic_shadow.json"
SCORES_SIX = Path(RESULTS_DIR) / "rotation_six"
SCORES_SHADOW = Path(RESULTS_DIR) / "mimic_shadow" / "scores"


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        pytest.skip(f"{path} is not built")
    return dict(json.loads(path.read_text()))


@pytest.fixture(scope="module")
def rotation() -> dict[str, Any]:
    return _read(ROTATION)


@pytest.fixture(scope="module")
def shadow() -> dict[str, Any]:
    return _read(SHADOW)


def _scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["y"].astype(int), data["p"].astype(np.float64)


def _coverage(cal: Path, target: Path, label: str) -> dict[str, Any]:
    y_cal, p_cal = _scores(cal)
    y_tgt, p_tgt = _scores(target)
    j = class_keys().index(label)
    sets = conformal_sets(p_cal[:, j], y_cal[:, j], p_tgt[:, j], ALPHA)
    return coverage_row(sets["perlabel"], y_tgt[:, j])


class TestTheRotationWithMimic:
    def test_every_pair_with_mimic_has_a_row_per_shared_label(
        self, rotation: dict[str, Any]
    ) -> None:
        rows = {(r["source"], r["target"], r["label"]) for r in rotation["mimic_pairs"]}
        expected = {
            (s, t, k)
            for s in EXTENDED_SOURCES
            for t in EXTENDED_SOURCES
            if "mimic" in (s, t)
            for k in usable_classes(s)
            if k in usable_classes(t)
        }
        assert rows == expected

    def test_each_wilson_interval_holds_its_point(self, rotation: dict[str, Any]) -> None:
        for r in rotation["mimic_pairs"]:
            assert r["coverage_pos_low"] <= r["coverage_pos"] <= r["coverage_pos_high"], r
            assert r["coverage_neg_low"] <= r["coverage_neg"] <= r["coverage_neg_high"], r

    def test_the_mimic_rows_come_back_from_the_committed_scores(
        self, rotation: dict[str, Any]
    ) -> None:
        for r in rotation["mimic_pairs"]:
            scores = SCORES_SIX / r["source"] / "scores"
            row = _coverage(
                scores / f"{r['source']}_cal.npz", scores / f"{r['target']}_test.npz", r["label"]
            )
            assert row["n_pos"] == r["n_pos"]
            assert round(row["coverage_pos"], 4) == r["coverage_pos"], r
            assert round(row["coverage_pos_low"], 4) == r["coverage_pos_low"], r

    def test_the_seen_target_grid_marks_the_two_encoders_that_saw_mimic(
        self, rotation: dict[str, Any]
    ) -> None:
        assert rotation["encoders_that_saw_mimic"] == ["ecgfm", "hubert_ecg"]

    def test_the_reproduction_summary_is_the_cells_it_summarises(
        self, rotation: dict[str, Any]
    ) -> None:
        check = rotation["reproduction"]
        gaps = [abs(c["difference"]) for c in check["cells"]]
        assert check["n_cells"] == len(gaps) == 116
        assert check["median_absolute_difference"] == round(float(np.median(gaps)), 4)
        assert check["max_absolute_difference"] == round(max(gaps), 4)


class TestTheShadowRun:
    def test_the_three_schemes_are_measured_for_every_label(self, shadow: dict[str, Any]) -> None:
        cells = {(r["scheme"], r["label"]) for r in shadow["headline_rows"]}
        assert cells == {
            (s, k) for s in ("patient_split", "same_era", "shadow") for k in class_keys()
        }

    def test_the_headline_comes_back_from_the_committed_scores(
        self, shadow: dict[str, Any]
    ) -> None:
        model = SCORES_SHADOW / shadow["settings"]["headline"]["model"]
        for r in shadow["headline_rows"]:
            row = _coverage(
                model / f"{r['fit_on']}.npz", model / f"{r['spent_on']}.npz", r["label"]
            )
            assert round(row["coverage_pos"], 4) == r["coverage_pos"], r
            assert r["coverage_pos_low"] <= r["coverage_pos"] <= r["coverage_pos_high"]

    def test_the_cohorts_hold_no_more_than_was_asked(self, shadow: dict[str, Any]) -> None:
        for name, cohort in shadow["cohorts"].items():
            assert 0.98 * cohort["n_requested"] <= cohort["n_read"] <= cohort["n_requested"], name

    def test_the_open_eras_follow_the_real_years_of_the_demo_patients(
        self, shadow: dict[str, Any]
    ) -> None:
        """What the shadow run calls early and late is about four real years apart."""
        if shadow["eras"] != "estimated":
            pytest.skip("anchor eras are real years already")
        check = shadow["check_against_demo_patients"]
        assert check["available"]
        by_era = check["real_year_by_estimated_era"]
        assert by_era["early"]["mean"] < by_era["middle"]["mean"] < by_era["late"]["mean"]
        assert by_era["late"]["mean"] - by_era["early"]["mean"] >= 3.5
        assert check["spearman_estimate_vs_real_year"] >= 0.7
