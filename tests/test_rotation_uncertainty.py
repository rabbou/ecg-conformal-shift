"""The two things the coverage table cannot say on its own.

C-31, the interval: every spread in ``results/rotation.csv`` is taken over the
calibration draws on a target cohort that never moves, so it carries none of the
uncertainty that comes from the cohort being a sample.  The bootstrap resamples
that cohort by patient and is reported beside the draw spread, not instead of it.

C-32, the comparator: Chow's rule is one plain empirical quantile per class,
which is Mondrian without the finite-sample ``(n+1)`` correction.  If the two
agree everywhere then the conformal formalism is a rename, and the file has to
show that rather than the reader having to assume otherwise.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rotation_uncertainty import bootstrap_by_patient, chow_quantiles

from ecs.config import RESULTS_DIR
from ecs.rotation import SOURCES, usable_classes

GRID = Path(RESULTS_DIR) / "rotation_uncertainty.csv"
SUMMARY = Path(RESULTS_DIR) / "rotation_uncertainty.json"


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    if not GRID.exists():
        pytest.skip(f"{GRID} is not built; run scripts/rotation_uncertainty.py")
    with GRID.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    if not SUMMARY.exists():
        pytest.skip(f"{SUMMARY} is not built; run scripts/rotation_uncertainty.py")
    return json.loads(SUMMARY.read_text())


class TestTheBootstrapItself:
    def test_it_resamples_patients_and_not_records(self) -> None:
        """Two tracings of one patient move together or not at all. Resampling
        records would treat them as two draws and shrink the interval."""
        rng = np.random.default_rng(0)
        # Two hundred records. Grouped, they come from a hundred patients with
        # two tracings each and the two always agree; ungrouped, from two
        # hundred patients. The same numbers, half the independent units.
        per_patient = [float(i % 2) for i in range(100)]
        covered = np.repeat(per_patient, 2)
        together = np.repeat([f"P{i:03d}" for i in range(100)], 2)
        apart = np.array([f"P{i:03d}" for i in range(200)])
        lo_together, hi_together = bootstrap_by_patient(covered, together, 4000, rng)
        lo_apart, hi_apart = bootstrap_by_patient(covered, apart, 4000, rng)
        assert (hi_together - lo_together) > (hi_apart - lo_apart)

    def test_an_empty_cohort_gives_no_interval_rather_than_a_number(self) -> None:
        lo, hi = bootstrap_by_patient(
            np.array([]), np.array([], dtype=str), 100, np.random.default_rng(0)
        )
        assert np.isnan(lo) and np.isnan(hi)

    def test_the_interval_brackets_the_mean_it_is_taken_around(self) -> None:
        rng = np.random.default_rng(1)
        covered = (rng.random(400) < 0.9).astype(float)
        patients = np.array([f"P{i:03d}" for i in range(400)])
        lo, hi = bootstrap_by_patient(covered, patients, 2000, rng)
        assert lo < covered.mean() < hi


class TestChowIsMondrianWithoutTheCorrection:
    def test_the_two_differ_by_the_finite_sample_term(self) -> None:
        """Both take a per-class quantile; conformal takes the
        ceil((n+1)(1-alpha))-th smallest, Chow the plain (1-alpha) quantile, so
        the conformal one is never the looser of the two."""
        from ecs.conformal import mondrian_quantiles  # noqa: PLC0415

        rng = np.random.default_rng(0)
        labels = (rng.random(500) < 0.3).astype(int)
        scores = rng.random(500)
        conformal = mondrian_quantiles(scores, labels, 0.10, 2)
        chow = chow_quantiles(scores, labels, 0.10)
        for klass in range(2):
            assert conformal[klass] >= chow[klass] - 1e-12, klass

    def test_a_class_too_thin_to_certify_gets_an_infinite_conformal_threshold(self) -> None:
        """Where the conformal rule refuses, Chow still returns a number. That
        refusal is the whole of what the formalism buys on a starved class."""
        from ecs.conformal import mondrian_quantiles  # noqa: PLC0415

        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        labels = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        conformal = mondrian_quantiles(scores, labels, 0.10, 2)
        chow = chow_quantiles(scores, labels, 0.10)
        assert np.isinf(conformal[1])
        assert np.isfinite(chow[1])


class TestTheCommittedUncertainty:
    def test_every_source_and_diagnosis_it_carries_is_measured(
        self, rows: list[dict[str, str]]
    ) -> None:
        present = {(r["source"], r["label"]) for r in rows}
        assert present == {
            (source, label) for source in SOURCES for label in usable_classes(source)
        }

    def test_every_row_carries_both_uncertainties(self, rows: list[dict[str, str]]) -> None:
        """C-31. The draw spread and the cohort interval are reported together;
        neither replaces the other."""
        for row in rows:
            coverage = float(row["coverage"])
            assert float(row["sd_over_calibration_draws"]) >= 0.0
            assert float(row["bootstrap_lo"]) <= coverage + 0.02, row
            assert float(row["bootstrap_hi"]) >= coverage - 0.02, row
            width = float(row["bootstrap_width"])
            assert width >= 0.0
            if width == 0.0:
                # Every positive covered on every resample: the threshold was
                # infinite, so the set held both labels and nothing could vary.
                assert coverage in (0.0, 1.0), row

    def test_the_interval_widens_as_the_class_thins(self, rows: list[dict[str, str]]) -> None:
        """The point of resampling the cohort: 23 positives cannot support the
        same precision as 800, whatever the threshold does."""
        away = [r for r in rows if r["role"] == "away" and r["correction"] == "mondrian"]
        thin = [r for r in away if int(r["n_positive"]) < 120]
        thick = [r for r in away if int(r["n_positive"]) > 400]
        assert thin and thick
        assert np.median([float(r["bootstrap_width"]) for r in thin]) > np.median(
            [float(r["bootstrap_width"]) for r in thick]
        )

    def test_the_comparator_is_reported_for_every_pair(self, rows: list[dict[str, str]]) -> None:
        """C-32. What the conformal formalism adds is a number on every row."""
        for row in rows:
            gap = float(row["conformal_minus_chow"])
            assert gap == pytest.approx(
                float(row["coverage"]) - float(row["chow_coverage"]), abs=1e-3
            )

    def test_the_two_rules_agree_except_where_the_class_is_starved(
        self, rows: list[dict[str, str]]
    ) -> None:
        """The finding this file exists for: on a populated class the conformal
        rule and a plain per-class rejection rule land in the same place, and the
        gap opens only where the conformal rule refuses."""
        away = [r for r in rows if r["role"] == "away" and r["correction"] == "mondrian"]
        gaps = sorted(away, key=lambda r: -abs(float(r["conformal_minus_chow"])))
        assert abs(float(gaps[0]["conformal_minus_chow"])) > 0.2
        assert {r["label"] for r in gaps[:5]} == {"LBBB"}
        median_gap = float(np.median([abs(float(r["conformal_minus_chow"])) for r in away]))
        assert median_gap < 0.05

    def test_the_summary_says_which_uncertainty_dominates(self, summary: dict[str, Any]) -> None:
        reading = summary["reading"]["bootstrap_width_against_draw_spread"]
        assert reading["median_bootstrap_width"] > 0
        assert reading["median_draw_spread"] > 0
        assert summary["settings"]["n_bootstrap"] >= 2000
        assert summary["settings"]["bootstrap_unit"].startswith("patient")
