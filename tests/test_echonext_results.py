"""The committed EchoNext results, the pages rendered from them, the prose quoting them.

EchoNext cannot be committed, so nothing here recomputes a number from the
tracings.  What is held instead: the coverage CSV is the JSON's grid row for
row, and the arms, cohorts and provenance carry what the report says of them.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import pytest
from report_text import pct
from test_echonext_severity_results import SEVERITY, severity_figures

from ecs.config import REPO_ROOT, RESULTS_DIR
from ecs.echonext import LABELS, PROVENANCE_FIELDS
from ecs.transfer import METHODS

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
    source = result["source"]["prevalence"]
    outpatient = result["targets"]["outpatient"]["prevalence"]
    return {
        "lvef_inpatient": pct(source[LVEF]),
        "lvef_outpatient": pct(outpatient[LVEF]),
        "composite_inpatient": pct(source[COMPOSITE]),
        "composite_outpatient": pct(outpatient[COMPOSITE]),
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
        # The severity section, which tests/test_echonext_severity_results.py holds figure
        # by figure, and the level of its intervals.
        severity = json.loads(SEVERITY.read_text())
        allowed |= set(re.findall(r"\d+(?:\.\d)?%", " ".join(severity_figures(severity).values())))
        allowed |= {"95%"}
        printed = set(re.findall(r"\d+(?:\.\d)?%", PIECE.read_text()))
        assert printed - allowed == set()

    def test_the_piece_states_the_unit(self) -> None:
        assert "z-score" in PIECE.read_text()


README = REPO_ROOT / "README.md"
README_SECTION = "## Structural heart disease, from inpatients to outpatients"
# The README's table, model by model, in the order it prints them.
README_MODELS = {
    "resnet": "Residual network, trained on EchoNext",
    "echonext_mini": "EchoNext mini-model, published weights",
    "ecgfounder": "ECGFounder, frozen, logistic regressions",
    "random_init": "Random initialisation, frozen, logistic regressions",
}
STRONGEST = {"resnet", "echonext_mini", "ecgfounder"}
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


def readme_section() -> str:
    text = README.read_text()
    start = text.index(README_SECTION)
    return text[start : text.index("\n## ", start + 1)]


def readme_opening_echonext() -> str:
    """The EchoNext sentences of the README's opening paragraph."""
    text = README.read_text()
    start = text.index("At Columbia, ")
    return text[start : text.index("A site adopting", start)]


def strongest(result: dict[str, Any]) -> list[str]:
    def auroc(arm: str) -> float:
        return float(result["arms"][arm]["auroc_test_all_contexts"][COMPOSITE]["auroc"])

    return sorted(result["arms"], key=auroc, reverse=True)[:3]


def readme_figures(result: dict[str, Any], provenance: dict[str, Any]) -> dict[str, str]:
    """Every figure the README's EchoNext text may print, by name, read from the results."""
    top = strongest(result)
    floor = arm_figures(result, "random_init")
    ill = sorted(cell(result, a, COMPOSITE, "outpatient", "perlabel")["coverage_pos"] for a in top)
    local = [
        row
        for a in top
        for row in result["arms"][a]["ladder"]["outpatient"][COMPOSITE]
        if row["labels"] == 100
    ]
    shared = shared_figures(result, provenance)
    severity = json.loads(SEVERITY.read_text())
    reweighted = [severity["arms"][a]["reweighted"]["reweighted"] for a in top]
    return {
        "tracings": shared["tracings"],
        "calibration_ecgs": shared["calibration_ecgs"],
        "outpatient_ecgs": shared["outpatient_ecgs"],
        "composite_inpatient": shared["composite_inpatient"],
        "composite_outpatient": shared["composite_outpatient"],
        "level": f"{1 - result['alpha']:.0%}",
        "strongest_low": pct(ill[0]),
        "strongest_high": pct(ill[-1]),
        "strongest_about": f"{round(100 * sum(ill) / len(ill))}%",
        "floor_ill_covered": floor["ill_covered"],
        "floor_to_a_human": floor["to_a_human"],
        "local_labels": str(local[0]["labels"]),
        "local_draws": str(local[0]["draws"]),
        "local_low": min(arm_figures(result, a)["ladder_100"] for a in top),
        "local_high": max(arm_figures(result, a)["ladder_100"] for a in top),
        "reweighted_low": pct(min(reweighted)),
        "reweighted_high": pct(max(reweighted)),
    }


class TestReadme:
    """The README's EchoNext text and table are read back out of the result files: the
    table row for row, and every number the prose prints by name."""

    def test_the_three_strongest_are_the_three_the_readme_names(
        self, result: dict[str, Any]
    ) -> None:
        assert set(strongest(result)) == STRONGEST

    def test_the_table_is_the_results_row_for_row(self, result: dict[str, Any]) -> None:
        printed = [
            line
            for line in readme_section().splitlines()
            if line.startswith("| ") and "---" not in line
        ][1:]
        expected = []
        for arm, name in README_MODELS.items():
            f = arm_figures(result, arm)
            expected.append(
                f"| {name} | {f['auroc_test']} | {f['ill_covered']} | {f['to_a_human']} |"
            )
        assert printed == expected

    def test_the_section_prints_the_headline_figures(
        self, result: dict[str, Any], provenance: dict[str, Any]
    ) -> None:
        section = readme_section()
        figures = readme_figures(result, provenance)
        wanted = {k: v for k, v in figures.items() if k != "strongest_about"}
        assert {k: v for k, v in wanted.items() if v not in section} == {}
        # The composite is the last label; the eleven findings are the others.
        assert LABELS[-1] == COMPOSITE and len(LABELS) - 1 == 11
        assert "eleven findings" in section

    def test_the_section_prints_no_number_the_results_do_not_hold(
        self, result: dict[str, Any], provenance: dict[str, Any]
    ) -> None:
        prose = "\n".join(
            line for line in readme_section().splitlines() if not line.startswith("|")
        )
        allowed = set(readme_figures(result, provenance).values())
        assert set(NUMBER.findall(prose)) - allowed == set()

    def test_the_opening_prints_only_the_results_figures(
        self, result: dict[str, Any], provenance: dict[str, Any]
    ) -> None:
        opening = readme_opening_echonext()
        figures = readme_figures(result, provenance)
        wanted = ["calibration_ecgs", "strongest_about", "local_labels", "local_draws"]
        wanted += ["local_low", "local_high"]
        assert {k: figures[k] for k in wanted if figures[k] not in opening} == {}
        assert set(NUMBER.findall(opening)) - set(figures.values()) == set()

    def test_the_severity_sentence_is_the_severity_results(
        self, result: dict[str, Any], provenance: dict[str, Any]
    ) -> None:
        """'About a third' is the mean share of the lost coverage that severity explains."""
        severity = json.loads(SEVERITY.read_text())
        shares = [severity["arms"][a]["reweighted"]["share"] for a in strongest(result)]
        assert abs(sum(shares) / len(shares) - 1 / 3) < 0.03
        figures = readme_figures(result, provenance)
        section = readme_section()
        assert "explains about a third of that loss" in section
        assert f"reaches {figures['reweighted_low']} to {figures['reweighted_high']}," in section

    def test_about_holds_for_each_of_the_three(self, result: dict[str, Any]) -> None:
        """'About 72%' is the mean of the three; each lies within a point and a half of it."""
        ill = [
            cell(result, a, COMPOSITE, "outpatient", "perlabel")["coverage_pos"] for a in STRONGEST
        ]
        mean = sum(ill) / len(ill)
        assert all(abs(x - mean) < 0.015 for x in ill)
