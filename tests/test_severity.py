"""Severity strata and reweighted coverage, on hand-built cohorts whose answer is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecs.severity import (
    MIN_STRATUM,
    MODERATE,
    bootstrap_reweighted,
    compare,
    findings_count,
    graded,
    lvef_band,
    reweighted,
    severity_cell,
    stratum_rows,
    verdict,
)


def test_findings_are_counted_per_record() -> None:
    meta = pd.DataFrame({"a": [1, 0, 1], "b": [1, 0, 0]})
    assert findings_count(meta, ("a", "b")).tolist() == [2, 0, 1]


def test_grades_are_ordered_and_presumed_none_is_none() -> None:
    values = pd.Series(["none", "presumed none", "mild", "severe", None])
    out = graded(values, "aortic_stenosis_value")
    assert out.iloc[:4].tolist() == [0, 0, 1, 3]
    assert np.isnan(out.iloc[4])


def test_an_unknown_grade_raises() -> None:
    with pytest.raises(ValueError, match="moderate-severe"):
        graded(pd.Series(["mild", "moderate-severe"]), "mitral_regurgitation_value")


def test_moderate_is_the_grade_each_finding_starts_at() -> None:
    assert MODERATE["aortic_stenosis_value"] == 2
    assert MODERATE["rv_systolic_function_value"] == 2
    assert MODERATE["pericardial_effusion_value"] == 3


def test_ejection_fraction_bands_are_inclusive_at_their_upper_edge() -> None:
    assert lvef_band(np.array([10.0, 35.0, 35.1, 45.0, 45.1])).tolist() == [
        "<=35",
        "<=35",
        "36-45",
        "36-45",
        ">45",
    ]
    with pytest.raises(ValueError, match="missing"):
        lvef_band(np.array([np.nan]))


def test_cells_cross_the_finding_count_with_the_band() -> None:
    cells = severity_cell(np.array([1, 3]), np.array([60.0, 30.0]))
    assert cells.tolist() == ["1 finding, LVEF >45", "2+ finding, LVEF <=35"]


def test_compare_reports_quartiles_and_a_test_only_when_both_sides_have_values() -> None:
    row = compare(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), pd.Series([np.nan, 9.0, 9.0]))
    assert row["inpatient"]["median"] == 3.0
    assert (row["inpatient"]["q1"], row["inpatient"]["q3"]) == (2.0, 4.0)
    assert row["outpatient"]["missing"] == 1
    assert 0 < row["p_value"] < 1
    assert "p_value" not in compare(pd.Series([1.0]), pd.Series([np.nan]))


def test_a_stratum_below_the_minimum_is_not_judged() -> None:
    covered = np.array([True] * MIN_STRATUM + [True, False])
    strata = np.array(["a"] * MIN_STRATUM + ["b", "b"])
    rows = stratum_rows(covered, strata, ("a", "b", "c"))
    assert [(r["n"], r["judged"]) for r in rows] == [(MIN_STRATUM, True), (2, False), (0, False)]
    assert rows[1]["coverage"] == 0.5 and rows[1]["low"] < 0.5 < rows[1]["high"]


def test_reweighting_to_the_same_mix_returns_the_observed_coverage() -> None:
    covered = np.array([True, False, True, True])
    strata = np.array(["mild", "mild", "severe", "severe"])
    assert reweighted(covered, strata, strata) == pytest.approx(covered.mean())


def test_reweighting_moves_coverage_toward_the_reference_mix() -> None:
    # Mild covered 50%, severe 100%; the reference is all severe.
    covered = np.array([True, False, True, True])
    strata = np.array(["mild", "mild", "severe", "severe"])
    assert reweighted(covered, strata, np.array(["severe"] * 3)) == 1.0
    with pytest.raises(ValueError, match="nobody"):
        reweighted(covered, strata, np.array(["other"]))


def test_when_severity_explains_everything_the_gap_closes_the_drop() -> None:
    # Within each stratum the cohort is covered as the reference would be.
    rng = np.random.default_rng(1)
    strata = np.array(["mild"] * 600 + ["severe"] * 200)
    covered = np.concatenate([rng.random(600) < 0.6, rng.random(200) < 0.98])
    reference = np.array(["mild"] * 100 + ["severe"] * 900)
    row = bootstrap_reweighted(covered, strata, reference, draws=300)
    assert row["reweighted_low"] < row["reweighted"] < row["reweighted_high"]
    assert row["share"] == pytest.approx(row["gap"] / (0.9 - row["observed"]))
    assert verdict(row)["verdict"] == "all"


def test_the_verdict_follows_the_stated_rule() -> None:
    base = {"observed": 0.72, "reweighted": 0.77, "gap": 0.05}
    part = base | {"gap_low": 0.02, "gap_high": 0.08, "reweighted_low": 0.72}
    assert verdict(part | {"reweighted_high": 0.82})["verdict"] == "part"
    assert verdict(part | {"reweighted_high": 0.91})["verdict"] == "all"
    none = base | {"gap_low": -0.01, "gap_high": 0.08, "reweighted_low": 0.72}
    assert verdict(none | {"reweighted_high": 0.82})["verdict"] == "none"
    assert verdict(none | {"reweighted_high": 0.95})["verdict"] == "too imprecise"
    assert verdict(part | {"reweighted_high": 0.82})["drop"] == pytest.approx(0.18)
