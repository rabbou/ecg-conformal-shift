"""The committed EchoNext results, the pages rendered from them, the prose quoting them.

EchoNext cannot be committed, so nothing here recomputes a number from the
tracings.  What is held instead: the coverage CSV is the JSON's grid row for
row, and the arms, cohorts and provenance carry what the report says of them.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
from report_text import pct

from ecs.config import RESULTS_DIR
from ecs.echonext import LABELS, PROVENANCE_FIELDS
from ecs.transfer import METHODS

TRANSFER = RESULTS_DIR / "echonext_transfer.json"
PROVENANCE = RESULTS_DIR / "echonext_provenance.json"
COVERAGE = RESULTS_DIR / "echonext_coverage.csv"
CONTEXTS = ("inpatient", "emergency", "outpatient")
COMPOSITE = "shd_moderate_or_greater_flag"
LVEF = "lvef_lte_45_flag"
RSS_CEILING = 12 * 1024**3


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(TRANSFER.read_text())
    return loaded


@pytest.fixture(scope="module")
def provenance() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(PROVENANCE.read_text())
    return loaded


def cell(result: dict[str, Any], arm: str, label: str, context: str, method: str) -> dict:
    for row in result["arms"][arm]["coverage"]:
        if (row["label"], row["context"], row["method"]) == (label, context, method):
            return dict(row)
    raise KeyError((arm, label, context, method))


class TestCoverageGrid:
    def test_every_label_context_and_method_is_measured_for_each_arm(
        self, result: dict[str, Any]
    ) -> None:
        for arm, measured in result["arms"].items():
            seen = {(r["label"], r["context"], r["method"]) for r in measured["coverage"]}
            expected = {(lab, c, m) for lab in LABELS for c in CONTEXTS for m in METHODS}
            assert seen == expected, arm

    def test_the_csv_is_the_json_grid_row_for_row(self, result: dict[str, Any]) -> None:
        table = pd.read_csv(COVERAGE)
        rows = [{"arm": a, **r} for a, m in result["arms"].items() for r in m["coverage"]]
        assert len(table) == len(rows)
        expected = pd.DataFrame(rows)
        for column in ("coverage", "coverage_pos", "abstention", "n", "n_pos"):
            assert table[column].fillna(-1).tolist() == pytest.approx(
                expected[column].fillna(-1).tolist()
            ), column

    def test_the_four_arms_of_the_protocol_ran(self, result: dict[str, Any]) -> None:
        assert list(result["arms"]) == ["resnet", "random_init", "ecgfounder", "echonext_mini"]
        assert "not_run" not in result


# Composite figures per arm, as measured on 2 October 2026: AUROC on the whole
# test split, then on outpatients the per-label coverage of the ill and of the
# healthy, the share sent to a human, and the ill covered after 100 local labels.
PINNED = {
    "resnet": ("0.834", "71.6%", "98.1%", "29.4%", "93.4%"),
    "random_init": ("0.792", "78.6%", "97.2%", "44.2%", "88.8%"),
    "ecgfounder": ("0.824", "71.6%", "97.7%", "32.2%", "95.3%"),
    "echonext_mini": ("0.820", "72.7%", "98.2%", "32.3%", "97.6%"),
}
PUBLISHED_MINI_AUROC = 0.820


class TestArms:
    def test_each_arm_keeps_its_composite_figures(self, result: dict[str, Any]) -> None:
        for arm, pinned in PINNED.items():
            f = arm_figures(result, arm)
            measured = (
                f["auroc_test"],
                f["ill_covered"],
                f["healthy_covered"],
                f["to_a_human"],
                f["ladder_100"],
            )
            assert measured == pinned, arm

    def test_the_mini_model_replays_its_published_auroc(self, result: dict[str, Any]) -> None:
        """82.0% on the test split in the EchoNext paper; a gap over half a point
        would mean the architecture or the label order was rebuilt wrongly."""
        auroc = result["arms"]["echonext_mini"]["auroc_test_all_contexts"][COMPOSITE]["auroc"]
        assert auroc == pytest.approx(PUBLISHED_MINI_AUROC, abs=0.005)

    def test_the_frozen_encoder_clears_the_card_kill_line(self, result: dict[str, Any]) -> None:
        """T-065: the minimal version dies if frozen encoders stay under 0.75 AUROC."""
        auroc = result["arms"]["ecgfounder"]["auroc_test_all_contexts"][COMPOSITE]["auroc"]
        assert auroc > 0.75

    def test_the_new_arms_say_how_their_weights_were_read(self, result: dict[str, Any]) -> None:
        assert "weights_only=True" in result["arms"]["echonext_mini"]["run"]["weights"]
        assert "z-score" in result["arms"]["ecgfounder"]["run"]["input"]


# The ladder at 100 local labels, as measured on 4 October 2026: the ill outpatients
# recognised and referred on the evaluation half, and the ill among the 100 ECGs a
# draw refits on (mean, fewest, most over the 200 draws).
PINNED_LADDER = {
    "resnet": ("55.4%", "38.0%", "24.5", 15, 36),
    "random_init": ("41.9%", "46.9%", "24.5", 15, 36),
    "ecgfounder": ("52.8%", "42.5%", "24.5", 15, 36),
    "echonext_mini": ("52.3%", "45.2%", "24.5", 15, 36),
}


class TestLadderSplit:
    def test_each_arm_keeps_its_split_and_its_count_of_the_ill(
        self, result: dict[str, Any]
    ) -> None:
        for arm, pinned in PINNED_LADDER.items():
            row = ladder_at(result, arm, 100)
            measured = (
                pct(row["recognised_pos_mean"]),
                pct(row["referred_pos_mean"]),
                f"{row['fit_ill_mean']:.1f}",
                row["fit_ill_min"],
                row["fit_ill_max"],
            )
            assert measured == pinned, arm

    def test_the_recognised_and_the_referred_are_the_coverage_of_the_ill(
        self, result: dict[str, Any]
    ) -> None:
        for arm, measured in result["arms"].items():
            for row in measured["ladder"]["outpatient"][COMPOSITE]:
                split = row["recognised_pos_mean"] + row["referred_pos_mean"]
                assert split == pytest.approx(row["coverage_pos_mean"]), (arm, row["labels"])

    def test_rung_zero_recognises_what_the_outcome_table_recognises_on_all_outpatients(
        self, result: dict[str, Any]
    ) -> None:
        """Rung zero reads half the outpatients; it stays within its own sampling error."""
        outcomes = json.loads((RESULTS_DIR / "echonext_outcomes.json").read_text())
        for arm in result["arms"]:
            whole = outcomes["arms"][arm]["outpatient"]["perlabel"]["ill"]["recognised"]["share"]
            half = ladder_at(result, arm, 0)["recognised_pos_mean"]
            assert abs(whole - half) < 0.05, arm


def ladder_at(result: dict[str, Any], arm: str, rung: int) -> dict[str, Any]:
    rows = result["arms"][arm]["ladder"]["outpatient"][COMPOSITE]
    return dict(next(r for r in rows if r["labels"] == rung))


class TestCohorts:
    def test_one_ecg_per_patient_in_calibration_and_targets(self, result: dict[str, Any]) -> None:
        assert result["source"]["n"] == result["source"]["patients"]
        for target in result["targets"].values():
            assert target["n"] == target["patients"]

    def test_lvef_prevalence_replays_the_card(self, result: dict[str, Any]) -> None:
        # T-065: 24.5% of inpatients and 7.7% of outpatients, validation and test.
        replay = result["prevalence_by_context_val_and_test"]
        assert round(100 * replay["inpatient"][LVEF], 1) == pytest.approx(24.5, abs=0.051)
        assert round(100 * replay["outpatient"][LVEF], 1) == pytest.approx(7.7, abs=0.051)
        assert round(100 * replay["inpatient"][COMPOSITE], 1) == 52.8
        assert round(100 * replay["outpatient"][COMPOSITE], 1) == 26.7


class TestProvenanceFile:
    def test_every_tracing_is_in_z_score(self, provenance: dict[str, Any]) -> None:
        assert provenance["units"] == ["z-score"]
        assert provenance["records"] == provenance["metadata_rows"] == 100_000

    def test_every_file_matches_its_published_digest(self, provenance: dict[str, Any]) -> None:
        assert set(provenance["files_match_published_sha256"].values()) == {True}

    def test_the_table_carried_every_declared_field(self, provenance: dict[str, Any]) -> None:
        assert provenance["columns_present"] == list(PROVENANCE_FIELDS)

    def test_the_pass_stayed_under_twelve_gigabytes(self, provenance: dict[str, Any]) -> None:
        assert provenance["peak_rss_bytes"] < RSS_CEILING


def arm_figures(result: dict[str, Any], arm: str) -> dict[str, str]:
    """One model's composite figures on outpatients, as the report prints them."""
    measured = result["arms"][arm]
    per = cell(result, arm, COMPOSITE, "outpatient", "perlabel")
    ladder = {row["labels"]: row for row in measured["ladder"]["outpatient"][COMPOSITE]}
    return {
        "ill_covered": pct(per["coverage_pos"]),
        "healthy_covered": pct(per["coverage_neg"]),
        "to_a_human": pct(per["abstention"]),
        "ladder_100": pct(ladder[100]["coverage_pos_mean"]),
        "auroc_test": f"{measured['auroc_test_all_contexts'][COMPOSITE]['auroc']:.3f}",
    }
