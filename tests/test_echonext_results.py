"""The committed EchoNext results, the pages rendered from them, the prose quoting them.

EchoNext cannot be committed, so nothing here recomputes a number from the
tracings.  What is held instead: every page under ``reports/transfer`` is
exactly what the renderer makes of the committed JSON, the coverage CSV is the
JSON's grid row for row, and every figure ECHONEXT.md prints is read back out of
the JSON it came from.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import pytest

from ecs.config import REPO_ROOT, RESULTS_DIR
from ecs.echonext import LABELS, PROVENANCE_FIELDS
from ecs.transfer import METHODS
from ecs.transfer_report import pct, render

TRANSFER = RESULTS_DIR / "echonext_transfer.json"
PROVENANCE = RESULTS_DIR / "echonext_provenance.json"
COVERAGE = RESULTS_DIR / "echonext_coverage.csv"
PIECE = REPO_ROOT / "ECHONEXT.md"
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


class TestPages:
    def test_each_page_is_what_the_renderer_makes_of_the_json(self, result: dict[str, Any]) -> None:
        for arm in result["arms"]:
            page = REPO_ROOT / f"reports/transfer/{arm}_inpatient_to_outpatient.md"
            assert page.read_text() == render(result, arm, "outpatient"), (
                f"{page.name} differs from its result file; rerun scripts/echonext_transfer.py"
            )

    def test_the_pages_name_the_cells_other_tasks_fill(self, result: dict[str, Any]) -> None:
        for arm in result["arms"]:
            page = (REPO_ROOT / f"reports/transfer/{arm}_inpatient_to_outpatient.md").read_text()
            for filler in ("T-068", "T-067", "T-065, ambitious version"):
                assert filler in page


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

    def test_the_two_pre_trained_arms_are_listed_as_not_run(self, result: dict[str, Any]) -> None:
        assert set(result["not_run"]) == {"echonext_mini", "ecgfounder"}


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


def subgroup(result: dict[str, Any], arm: str, kind: str, group: str) -> dict:
    for row in result["arms"][arm]["subgroups"]:
        key = (row["label"], row["context"], row["method"], row["kind"], row["group"])
        if key == (COMPOSITE, "outpatient", "perlabel", kind, group):
            return dict(row)
    raise KeyError((arm, kind, group))


def arm_figures(result: dict[str, Any], arm: str) -> dict[str, str]:
    """The figures the piece may quote for one arm, by name, as the page formats them."""
    measured = result["arms"][arm]
    per = cell(result, arm, COMPOSITE, "outpatient", "perlabel")
    ladder = {row["labels"]: row for row in measured["ladder"]["outpatient"][COMPOSITE]}
    return {
        "ill_covered": pct(per["coverage_pos"]),
        "healthy_covered": pct(per["coverage_neg"]),
        "to_a_human": pct(per["abstention"]),
        "pooled_ill_covered": pct(
            cell(result, arm, COMPOSITE, "outpatient", "pooled")["coverage_pos"]
        ),
        "lvef_ill_covered": pct(cell(result, arm, LVEF, "outpatient", "perlabel")["coverage_pos"]),
        "emergency_ill_covered": pct(
            cell(result, arm, COMPOSITE, "emergency", "perlabel")["coverage_pos"]
        ),
        "women": pct(subgroup(result, arm, "sex", "female")["coverage_pos"]),
        "men": pct(subgroup(result, arm, "sex", "male")["coverage_pos"]),
        "ladder_0": pct(ladder[0]["coverage_pos_mean"]),
        "ladder_100": pct(ladder[100]["coverage_pos_mean"]),
        "ladder_100_healthy": pct(ladder[100]["coverage_neg_mean"]),
        "auroc_outpatient": f"{measured['auroc']['outpatient'][COMPOSITE]['auroc']:.3f}",
        "auroc_test": f"{measured['auroc_test_all_contexts'][COMPOSITE]['auroc']:.3f}",
    }


def shared_figures(result: dict[str, Any], provenance: dict[str, Any]) -> dict[str, str]:
    replay = result["prevalence_by_context_val_and_test"]
    return {
        "lvef_inpatient": pct(replay["inpatient"][LVEF]),
        "lvef_outpatient": pct(replay["outpatient"][LVEF]),
        "composite_inpatient": pct(replay["inpatient"][COMPOSITE]),
        "composite_outpatient": pct(replay["outpatient"][COMPOSITE]),
        "calibration_ecgs": f"{result['source']['n']:,}",
        "outpatient_ecgs": f"{result['targets']['outpatient']['n']:,}",
        "tracings": f"{provenance['records']:,}",
    }


class TestPiece:
    """ECHONEXT.md quotes the results: its headline figures are found in the text, and
    every percentage it prints is one the result files hold."""

    def test_the_headline_figures_are_in_the_piece(
        self, result: dict[str, Any], provenance: dict[str, Any]
    ) -> None:
        text = PIECE.read_text()
        wanted = shared_figures(result, provenance) | {
            f"resnet:{k}": v for k, v in arm_figures(result, "resnet").items()
        }
        missing = {name: figure for name, figure in wanted.items() if figure not in text}
        assert missing == {}

    def test_the_piece_prints_no_percentage_the_results_do_not_hold(
        self, result: dict[str, Any], provenance: dict[str, Any]
    ) -> None:
        allowed = set(shared_figures(result, provenance).values())
        for arm in result["arms"]:
            allowed |= set(arm_figures(result, arm).values())
        # The level asked for, the LVEF threshold of the label, the published mini-model AUROC.
        allowed |= {"90%", "45%", "82.0%"}
        printed = set(re.findall(r"\d+(?:\.\d)?%", PIECE.read_text()))
        assert printed - allowed == set()

    def test_the_piece_states_the_unit(self) -> None:
        assert "z-score" in PIECE.read_text()
