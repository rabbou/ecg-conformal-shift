"""The committed severity results and the SUPPLEMENT.md section that quotes them.

Every figure the section prints is rebuilt here from ``results/echonext_severity.json``
and looked for in the section's text, and the section may print no number the
file does not hold; the verdict is replayed from the rule the file states.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from ecs.config import REPO_ROOT, RESULTS_DIR
from ecs.severity import GRADES, VERDICT_RULE, verdict
from ecs.transfer_report import pct

SEVERITY = RESULTS_DIR / "echonext_severity.json"
TRANSFER = RESULTS_DIR / "echonext_transfer.json"
PIECE = REPO_ROOT / "SUPPLEMENT.md"
HEADING = "### S1.7 Disease severity among cases by care setting"
COMPOSITE = "shd_moderate_or_greater_flag"
ARMS = ("resnet", "echonext_mini", "ecgfounder")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
# Read in the text but not measured: the level asked for, the interval level,
# the band edges, the bootstrap draws and the smallest stratum judged.
CONSTANTS = {"90%", "95%", "35", "36", "45", "45,", "45%", "2,000", "30", "11"}
NAMES = {
    "rv_systolic_function_value": (
        "normal",
        "mildly reduced",
        "moderately reduced",
        "severely reduced",
    ),
    "pericardial_effusion_value": ("none", "trace", "small", "moderate", "large"),
}
VALVE_NAMES = ("none", "mild", "moderate", "severe")


@pytest.fixture(scope="module")
def severity() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(SEVERITY.read_text())
    return loaded


def section() -> str:
    text = PIECE.read_text()
    start = text.index("\n", text.index(HEADING))
    return text[start : text.index("\n#", start)]


def interval(row: dict[str, Any], point: str, low: str, high: str) -> str:
    return f"{pct(row[point])} ({pct(row[low])} to {pct(row[high])})"


def whole(x: float) -> str:
    return f"{round(100 * x)}%"


def grade(column: str, value: float) -> str:
    if value != int(value):
        raise ValueError(f"{column}: quartile {value} falls between two grades")
    return NAMES.get(column, VALVE_NAMES)[int(value)]


def severity_figures(severity: dict[str, Any]) -> dict[str, str]:
    """Every figure the section prints, by name, formatted as the section prints it."""
    sev = severity["severity"]
    out = {
        "n_ill_inpatient": f"{severity['cohorts']['inpatient']['n_ill']:,}",
        "n_ill_outpatient": f"{severity['cohorts']['outpatient']['n_ill']:,}",
        "draws": f"{severity['bootstrap_draws']:,}",
    }
    for side in ("inpatient", "outpatient"):
        f = sev["findings_count"][side]
        out[f"findings:{side}"] = f"{f['median']:.0f} ({f['q1']:.0f} to {f['q3']:.0f})"
        for column in ("lvef_value", "pasp_value", "tr_max_velocity_value"):
            c = sev[column][side]
            out[f"{column}:{side}"] = f"{c['median']:.1f} ({c['q1']:.1f} to {c['q3']:.1f})"
        for column in ("ivs_measurement", "lvpw_measurement"):
            c = sev[column][side]
            out[f"{column}:{side}"] = f"| {c['median']:.1f} ({c['q1']:.1f} to {c['q3']:.1f}) |"
        for column in ("pasp_value", "tr_max_velocity_value"):
            out[f"{column}:{side}:n"] = f"{sev[column][side]['n']} measured"
        for column in GRADES:
            g = sev[column][side]
            out[f"{column}:{side}"] = (
                f"{grade(column, g['median'])} ({grade(column, g['q1'])} to "
                f"{grade(column, g['q3'])}), {pct(sev[column]['share_moderate_or_worse'][side])}"
            )
    for column, row in sev.items():
        p = row["p_value"]
        out[f"{column}:p"] = "< 0.001" if p < 0.001 else f"{p:.3f}"
    for arm in ARMS:
        a = severity["arms"][arm]
        w = a["reweighted"]
        out[f"{arm}:observed"] = pct(a["coverage_ill_outpatient"])
        out[f"{arm}:reweighted"] = interval(w, "reweighted", "reweighted_low", "reweighted_high")
        out[f"{arm}:share"] = f"{whole(w['share'])} ({whole(w['share_low'])} to " + (
            f"{whole(w['share_high'])})"
        )
        for stratum in ("by_findings", "by_lvef"):
            for row in a[stratum]["outpatient"]:
                out[f"{arm}:{stratum}:{row['stratum']}"] = interval(row, "coverage", "low", "high")
                out[f"{stratum}:{row['stratum']}:n"] = f", {row['n']} |"
    arms = [severity["arms"][a] for a in ARMS]
    shares = [a["reweighted"]["share"] for a in arms]
    reweighted = [a["reweighted"]["reweighted"] for a in arms]
    out["share_range"] = f"between {whole(min(shares))} and\n{whole(max(shares))} of it"
    out["reweighted_range"] = f"{pct(min(reweighted))} to {pct(max(reweighted))}"
    above = [next(r for r in a["by_lvef"]["outpatient"] if r["stratum"] == ">45") for a in arms]
    inside = [
        next(r for r in a["by_lvef"]["inpatient_in_sample"] if r["stratum"] == ">45") for a in arms
    ]
    out["above_45_n"] = f"{above[0]['n']}\noutpatients with SHD"
    out["above_45_range"] = (
        f"{pct(min(r['coverage'] for r in above))} to {pct(max(r['coverage'] for r in above))}"
    )
    out["above_45_inpatient_range"] = (
        f"{pct(min(r['coverage'] for r in inside))} to\n{pct(max(r['coverage'] for r in inside))}"
    )
    small = sum(not r["judged"] for r in severity["arms"]["resnet"]["by_cell"]["outpatient"])
    out["cells_unjudged"] = f"{['None', 'One', 'Two', 'Three', 'Four', 'Five', 'Six'][small]}\nof"
    return out


class TestSection:
    def test_every_figure_is_in_the_section(self, severity: dict[str, Any]) -> None:
        text = section()
        missing = {k: v for k, v in severity_figures(severity).items() if v not in text}
        assert missing == {}

    def test_the_section_prints_no_number_the_results_do_not_hold(
        self, severity: dict[str, Any]
    ) -> None:
        allowed = set(NUMBER.findall(" ".join(severity_figures(severity).values()))) | CONSTANTS
        assert set(NUMBER.findall(section())) - allowed == set()

    def test_the_verdict_the_section_states_is_the_files(self, severity: dict[str, Any]) -> None:
        assert severity["verdict"] == "part"
        assert "Severity explains part of the drop for each of the three arms" in section()


class TestFile:
    def test_the_rule_was_recorded_with_the_figures(self, severity: dict[str, Any]) -> None:
        assert severity["verdict_rule"] == VERDICT_RULE
        assert re.fullmatch(r"[0-9a-f]{40}", severity["commit"])

    def test_each_arms_verdict_follows_from_its_bootstrap_row(
        self, severity: dict[str, Any]
    ) -> None:
        for arm in ARMS:
            row = severity["arms"][arm]
            assert row["verdict"] == verdict(row["reweighted"])["verdict"], arm

    def test_the_observed_coverage_is_the_published_one(self, severity: dict[str, Any]) -> None:
        """Same thresholds as the transfer table: the ill outpatients' coverage matches it."""
        transfer = json.loads(TRANSFER.read_text())
        for arm in ARMS:
            published = next(
                r
                for r in transfer["arms"][arm]["coverage"]
                if (r["label"], r["context"], r["method"]) == (COMPOSITE, "outpatient", "perlabel")
            )
            assert severity["arms"][arm]["coverage_ill_outpatient"] == pytest.approx(
                published["coverage_pos"]
            )
            assert severity["cohorts"]["outpatient"]["n_ill"] == published["n_pos"]

    def test_the_cohorts_are_the_published_ones(self, severity: dict[str, Any]) -> None:
        transfer = json.loads(TRANSFER.read_text())
        assert severity["cohorts"]["inpatient"]["n_cohort"] == transfer["source"]["n"]
        assert (
            severity["cohorts"]["outpatient"]["n_cohort"]
            == (transfer["targets"]["outpatient"]["n"])
        )

    def test_the_cells_hold_every_ill_patient_and_ten_outpatients_each(
        self, severity: dict[str, Any]
    ) -> None:
        cells = severity["cells"].values()
        assert sum(c["inpatient"] for c in cells) == severity["cohorts"]["inpatient"]["n_ill"]
        assert sum(c["outpatient"] for c in cells) == severity["cohorts"]["outpatient"]["n_ill"]
        assert min(c["outpatient"] for c in cells) >= 10
