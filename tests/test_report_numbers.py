"""Every number REPORT.md and README.md print is a number a file under ``results/`` holds.

Three checks, each on the text as committed:

  every table row is rebuilt from its results file and must appear verbatim;
  every figure the prose quotes is rebuilt with the words around it and must
  appear verbatim, so a figure that moved in its file fails here;
  every number left in the prose must be one of those, a figure of a cited
  source, or a constant of the design named below with its reason, so a
  number typed by hand fails here.

The rotation appendix is held figure by figure by ``tests/test_rotation_report.py``;
its figures join the allowed set here.
"""

from __future__ import annotations

import re
from functools import cache

import pytest
import report_figures as f
from report_text import NUMBER, README, REPORT
from test_rotation_report import GROUPS, _cases

# Figures quoted from the cited papers, by the reference that holds them.
LITERATURE = {
    "[1] Poterucha 2025, Table 2": {"84.3%", "84.1%"},
    "[2] Hughes 2026": {"82.0%", "70.1%", "77.9%", "70%", "100,000", "0.1", "99.9"},
    "[3] Holste 2025, Figure 2": {"0.92", "0.98", "0.97", "0.99", "0.80", "0.94"},
    "[4] Attia 2021": {"4,277", "0.82", "0.256", "26.9%"},
    "[5] Carter 2026": {"84.5%", "83.6%", "0.45"},
    "[6] Otabor 2026": {"0.790", "0.820", "0.796", "0.773"},
    "[7] Poterucha 2026": {"65", "85"},
    "[11] de Vries 2023": {"13.0%"},
    "[13] Millar 2026": {"85.8%", "77.5%", "3", "1.5"},
    "[17] El Allam 2026": {"3.06", "0.12"},
    "[18] Li 2024": set(),
}

# Constants of the design, not results: each is named with what it is.
CONSTANTS = {
    "90%": "the sensitivity and the coverage the thresholds are set for",
    "0.9": "the same level, in the rank formula",
    "1": "the (n+1) of the conformal rank",
    "95%": "the level of every interval",
    "80%": "the target of the encoder comparison",
    "100": "the rung of the ladder the report reads",
    "25": "the lowest rung of the ladder",
    "50": "a rung of the ladder; also the lower edge of an age band",
    "200": "the calibration draws, and a rung of the ladder",
    "400": "the top rung of the EchoNext ladder",
    "500": "a rung of the Chongqing ladder, and the AUROC bootstrap draws",
    "2,000": "the severity bootstrap draws and the top rung of the Chongqing ladder",
    "30": "the smallest severity cell judged on its own",
    "10": "the length of an EchoNext tracing in seconds; ten ill patients, the floor of appendix D",
    "250": "EchoNext's sampling rate in Hz",
    "500 Hz": "ECGFounder's sampling rate",
    "12": "the leads, and the outputs of the trained network",
    "45%": "the ejection-fraction threshold of the EchoNext label",
    "35%": "a severity band edge of ejection fraction",
    "36%": "a severity band edge of ejection fraction",
    "35": "a severity band edge",
    "36": "a severity band edge",
    "45": "a severity band edge and the pulmonary pressure threshold of the label",
    "1.3": "the wall-thickness threshold of the label, cm",
    "3.2": "the tricuspid velocity threshold of the label, m/s",
    "8": "the folds of PTB-XL used for training (1 to 8)",
    "9": "PTB-XL's validation fold",
    "2018": "part of a corpus name, CPSC 2018",
    "2021": "part of a corpus name, the Challenge-2021 collection",
    "8700": "the CPU model the timings were taken on, i7-8700",
    "0": "seed 0",
    "15233e93": "the IntroECG commit the weights come from",
    "4,082,306": "the parameter count of the infarction network, results/timing.json",
    "19,955": "Chongqing's released tracings, results/ingest_report.json",
    "1,995": "Chongqing's unlabelled tracings",
    "17,955": "Chongqing's tracings read",
    "2017": "Wang et al. 2017, the architecture's source",
    "1989–96": "PTB-XL's recording years",
    "2019–20": "Shandong's recording years",
    "2015–24": "Chongqing's recording years",
    "2026": "the year of the report and of several sources",
    "4": "the day of the report's date",
    "2024-256-01": "the Chongqing ethics approval number",
    "2020-1": "the PTB-XL ethics approval number, PTB-2020-1",
    "1.1.1": "the EchoNext release",
    "10⁻³": "the learning rate",
    "10⁻²": "the weight decay",
    "64": "the batch size",
    "21,799": "PTB-XL's released records",
    "25,770": "Shandong's released records",
    "45,152": "Chapman-Shaoxing and Ningbo's released records",
    "10,344": "Georgia's released records",
    "10,330": "CPSC's released records",
    "6.6": "the size of ptbxl_database.csv in MB",
    "2": "a timing in the README",
    "5": "a timing in the README",
    "20": "a timing in the README",
}


@cache
def figures() -> dict[str, str]:
    out = f.echonext_figures() | f.extra_figures()
    out |= {f"inf.{k}": v for k, v in f.infarction_figures().items()}
    out |= {f"pool.{k}": v for k, v in f.pooled_figures().items()}
    out |= {f"sev.{k}": v.replace("\n", " ") for k, v in f.severity_strings().items()}
    out |= {f"sev.{k}": v for k, v in f.severity_counts().items()}
    return out


TABLES = (
    "cohort_rows",
    "operating_rows",
    "split_rows",
    "severity_rows",
    "ladder_rows",
    "site_rows",
    "severity_compare_rows",
    "finding_rows",
    "perturbation_rows",
    "subgroup_rows",
)


def prose(text: str) -> str:
    """The text a number is read from: no reference list, link, code or citation mark."""
    for first, last in (("## References", "## Data and code"), ("## Licence and citation", None)):
        if first in text:
            start = text.index(first)
            text = text[:start] + (text[text.index(last) :] if last else "")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", text)
    text = re.sub(r"doi:\S+|arXiv:\S+|PMID \d+|\b10\.\d{4,5}/\S+", " ", text)
    text = re.sub(r"^#+ .*$", " ", text, flags=re.M)
    text = re.sub(r"^\d+\. ", " ", text, flags=re.M)
    text = re.sub(r"(Table|Figure|figure|table|section|appendix|Appendix) [A-H]?\d+", " ", text)
    return text


def allowed() -> set[str]:
    values = set(figures().values())
    for name in TABLES:
        values |= set(getattr(f, name)())
    for group in GROUPS:
        values |= {expected for _, expected in _cases(group)}
    numbers = set(NUMBER.findall(" ".join(values)))
    for cited in LITERATURE.values():
        numbers |= set(NUMBER.findall(" ".join(cited)))
    return numbers | set(NUMBER.findall(" ".join(CONSTANTS)))


# The prose's figures with the words around them, rebuilt from the files.
def report_fragments() -> dict[str, str]:
    g = figures()
    return {
        "abstract cohorts": f"{g['records']} ECGs from Columbia",
        "abstract fit": f"fitted on {g['calibration_ecgs']} inpatient ECGs and applied unchanged "
        f"to {g['outpatient_ecgs']} outpatient ECGs",
        "abstract sensitivity": f"recognised {g['strongest_low']} to {g['strongest_high']} of the "
        f"{g['ill_out_n']} outpatients",
        "abstract specificity": f"from {g['spec_in_low']}–{g['spec_in_high']} among test "
        f"inpatients to {g['spec_out_low']}–{g['spec_out_high']} among outpatients",
        "abstract auroc": f"from {g['resnet_auroc_in']} to {g['resnet_auroc_out']}",
        "abstract recognised": f"alone to {g['recognised_low']} to {g['recognised_high']} of ill "
        f"outpatients and referred {g['referred_low']} to {g['referred_high']}",
        "abstract severity": f"accounted for {g['share_low']} to {g['share_high']} of the lost",
        "abstract ladder": f"coverage of the ill to {g['ladder_low']} to {g['ladder_high']}",
        "abstract infarction": f"covered {g['inf.acs_per_mi']} of infarctions at Chongqing",
        "abstract shandong": f"and {g['inf.sph_per_mi']} at Shandong",
        "training split": f"The training split ({g['training_ecgs']} ECGs)",
        "mini auroc": f"test-split AUROC of {g['mini_auroc_test']} replays",
        "ladder halves": f"evaluation half of {g['ladder_eval']} ECGs, {g['ladder_eval_ill']} of "
        "them ill",
        "ladder draws": f"drawn from the pool, {g['ladder_draws']} times",
        "inpatient sensitivity": f"it recognised {g['sens_in_low']} to {g['sens_in_high']}.",
        "ppv": f"was right for {g['resnet_ppv_out']} of outpatients flagged",
        "ppv from inpatients": f"would have been {g['resnet_ppv_from_in']}",
        "prevalence": f"At the outpatients' prevalence of {g['prevalence_out']}",
        "emergency": f"recognised {g['emergency_sens']} of the ill",
        "lvef": f"the finding the model reads best, {g['lvef_sens_out']}",
        "spread": f"within {g['spread_points']} points of one another",
        "floor": f"recognised more of the ill, {g['floor_sens_out']}",
        "floor specificity": f"specificity among outpatients was {g['floor_spec_out']}",
        "floor referred": f"it referred {g['floor_referred_out']} of outpatients",
        "false alarms": f"false alarms fell to {g['false_alarm_low']} to {g['false_alarm_high']}",
        "referred all": f"referring {g['referred_all_low']} to {g['referred_all_high']} of all",
        "no alarm": f"fell from {g['resnet_referred_in']} among test inpatients to "
        f"{g['resnet_referred_out']} among outpatients",
        "no alarm coverage": f"fell from {g['resnet_sens_in']} to {g['resnet_sens_out']}",
        "one finding": f"recognised {g['sev.resnet:by_findings:1']} of ill outpatients with one",
        "two findings": f"and {g['sev.resnet:by_findings:2+']} of those with two or more",
        "reweighted": f"rose to {g['reweighted_low']} to {g['reweighted_high']} across",
        "share": f"severity explained {g['share_low']} to {g['share_high']} of the drop",
        "above 45": f"Of the {g['sev.above_45_n']} whose ejection fraction is above 45%, "
        f"{g['sev.above_45_range']} were covered, against {g['sev.above_45_inpatient_range']}",
        "ladder 100": f"covered {g['ladder_low']} to {g['ladder_high']} of the ill outpatients",
        "ill in 100": f"hold about {g['ladder_ill_in_100']} ill patients",
        "healthy cost": f"fell from {g['healthy_before']} to {g['healthy_after']}",
        "ladder 25": f"infinite in {g['ladder_25_flagged']} of draws",
        "ptbxl": f"covered {g['inf.ptbxl_per_mi']} of infarctions on held-out PTB-XL",
        "shandong": f"{g['inf.sph_per_mi']} at Shandong ({g['inf.sph_wilson']} on its "
        f"{g['inf.sph_mi_n']} infarctions)",
        "chongqing": f"{g['inf.acs_per_mi']} at Chongqing ({g['inf.acs_wilson']} on "
        f"{g['inf.acs_mi_n']})",
        "chongqing other": f"non-infarction class fell to {g['inf.acs_per_non']}",
        "auroc chongqing": f"fell from {g['inf.baseline_auroc']} on PTB-XL to {g['acs_auroc']}",
        "chongqing ladder": f"from {g['inf.ladder_0']} to {g['inf.ladder_100']} on the tracings",
        "chongqing ladder tail": f"leaving it at {g['inf.ladder_500']} and {g['inf.ladder_2000']}",
        "chongqing infarctions": f"hold {g['inf.ladder_100_mi']} infarctions on average",
        "about": f"recognised about {g['strongest_about']} of outpatients with it",
        "women": f"recognised {g['women']} of ill women ({g['women_ci']})",
        "men": f"and {g['men']} of ill men ({g['men_ci']})",
        "pooled ptbxl": f"covered {g['pool.ptbxl_all']} of all tracings and {g['pool.ptbxl_mi']} "
        f"of infarctions, giving the non-infarction label alone to {g['pool.ptbxl_mi_wrong']}",
        "pooled targets": f"covered {g['pool.sph_mi']} of infarctions at Shandong and "
        f"{g['pool.acs_mi']} at Chongqing",
        "pooled echonext": f"covered {g['pool.resnet_pooled_out']} of ill outpatients",
        "baseline": f"its AUROC is {g['inf.baseline_auroc']} ({g['inf.baseline_low']} to "
        f"{g['inf.baseline_high']})",
        "fold 10": f"({g['inf.fold10']} tracings, {g['inf.fold10_mi']} infarctions)",
        "correction": f"is worth {g['inf.correction_points']} of a coverage point",
        "provenance": f"records for all {g['records']}",
    }


def readme_fragments() -> dict[str, str]:
    g = figures()
    return {
        "auroc": f"{g['resnet_auroc_in']} to {g['resnet_auroc_out']} for the network trained here",
        "no alarm": f"fell from {g['resnet_referred_in']} to {g['resnet_referred_out']} while",
        "sensitivity": f"recognised {g['strongest_low']} to {g['strongest_high']} of outpatients",
        "ladder": f"about {g['ladder_ill_in_100']} of them ill, brought coverage of the ill to "
        f"{g['ladder_low']} to {g['ladder_high']}",
        "ladder 25": f"infinite in {g['ladder_25_flagged']} of draws refitted on 25",
        "recognised": f"alone to {g['recognised_low']} to {g['recognised_high']} of the ill and "
        f"referred {g['referred_low']} to {g['referred_high']}",
        "records": f"EchoNext holds {g['records']} ECGs",
        "calibration": f"fitted on {g['calibration_ecgs']} inpatient ECGs",
        "releases": f"| EchoNext | United States | {g['records']} |",
    }


class TestReport:
    @pytest.mark.parametrize("table", TABLES)
    def test_every_row_of_the_table_is_its_results_file(self, table: str) -> None:
        lines = set(REPORT.read_text().splitlines())
        rows = getattr(f, table)()
        assert rows, table
        assert [row for row in rows if row not in lines] == []

    def test_every_figure_the_prose_quotes_is_its_results_file(self) -> None:
        text = REPORT.read_text()
        assert {k: v for k, v in report_fragments().items() if v not in text} == {}

    def test_the_text_prints_no_number_nothing_accounts_for(self) -> None:
        printed = set(NUMBER.findall(prose(REPORT.read_text())))
        assert printed - allowed() == set()

    def test_about_a_third_is_the_mean_share_severity_explains(self) -> None:
        """'About a third' in the report and the README is the severity results' mean share."""
        severity = f.read("echonext_severity.json")["arms"]
        shares = [severity[a]["reweighted"]["share"] for a in f.STRONGEST]
        assert abs(sum(shares) / len(shares) - 1 / 3) < 0.03
        assert "explains about a third" in README.read_text()
        assert "Milder disease explains about a third" in REPORT.read_text()

    def test_about_72_holds_for_each_of_the_three(self) -> None:
        """'About 72%' is the mean of the three; each lies within a point and a half of it."""
        ill = [f.cell(a, "outpatient", "plain")["coverage_pos"] for a in f.STRONGEST]
        mean = sum(ill) / len(ill)
        assert all(abs(x - mean) < 0.015 for x in ill)

    def test_the_per_class_rule_and_the_sensitivity_threshold_flag_the_same_ill(self) -> None:
        """The report says the rule's threshold for the ill is the sensitivity threshold."""
        for arm in f.ARMS:
            for context in f.CONTEXTS:
                plain = f.cell(arm, context, "plain")["coverage_pos"]
                assert f.cell(arm, context, "perlabel")["coverage_pos"] == plain, (arm, context)

    def test_recognised_never_counts_a_referred_patient(self) -> None:
        """Recognised plus referred is the coverage of the ill, never recognised alone."""
        for arm in f.ARMS:
            ill = f.outcomes(arm, "outpatient")["perlabel"]["ill"]
            covered = f.cell(arm, "outpatient", "perlabel")["coverage_pos"]
            assert ill["recognised"]["share"] + ill["referred"]["share"] == pytest.approx(covered)
            assert ill["recognised"]["share"] < covered


class TestReadme:
    def test_every_figure_the_page_quotes_is_its_results_file(self) -> None:
        text = README.read_text()
        assert {k: v for k, v in readme_fragments().items() if v not in text} == {}

    def test_the_page_prints_no_number_nothing_accounts_for(self) -> None:
        printed = set(NUMBER.findall(prose(README.read_text())))
        assert printed - allowed() == set()

    def test_the_page_points_at_the_one_report(self) -> None:
        text = README.read_text()
        assert "](REPORT.md)" in text
        assert "ECHONEXT.md" not in text


def test_a_number_typed_by_hand_would_be_caught() -> None:
    """The refusal bites: a figure no file holds is reported, one a file holds is not."""
    typed = prose("The threshold recognised 71.9% of them.")
    held = prose(f"The threshold recognised {figures()['strongest_low']} of them.")
    assert set(NUMBER.findall(typed)) - allowed() == {"71.9%"}
    assert set(NUMBER.findall(held)) - allowed() == set()
