"""A number in the published report that no longer matches its results file fails here.

REPORT.md, SUPPLEMENT.md, README.md and CITATION.cff are rendered from
``docs/templates/``, where each computed number is ``{{name}}`` and resolves
through ``tests/paper_values.py`` to a file under ``results/``.  Three checks
hold the text to the files:

  the committed text is its template rendered from the committed results, so a
  number edited by hand fails even when it equals another figure, and a results
  file that moved fails until the documents are rendered again;
  the report and the README type no number of their own beyond a cited source's
  figure or a constant of the design named below with its reason;
  each claim the prose makes about its figures ("the same share", "about a
  quarter", "one draw in four") is asserted on the files below.

Rendering: ``uv run python tests/paper_render.py --write``.
"""

from __future__ import annotations

import re
from pathlib import Path

import paper_values as v
import pytest
from paper_render import (
    CITATION,
    PLACEHOLDER,
    README,
    REPORT,
    SUPPLEMENT,
    TEMPLATES,
    render,
    rendered,
)

# A number as prose prints it: digits with thousands commas, an optional decimal
# part, an optional per cent sign.
NUMBER = re.compile(r"(?<![\w.])[-−]?\d+(?:,\d{3})*(?:\.\d+)?%?")

# Figures quoted from cited papers, by the reference that holds them.
LITERATURE = {
    "Poterucha 2025, Table 2: AUROC by care setting, the cut-off and the F1 scores": {
        "0.843",
        "0.841",
        "0.5",
        "0.557",
        "0.728",
    },
    "Otabor 2026: the mini-model on MIMIC-IV, overall and by care setting": {
        "0.790",
        "0.820",
        "0.796",
        "0.773",
    },
    "Poterucha 2026, PREVUE-VALVE: ages and AUROCs": {"65", "85", "0.71", "0.83"},
    "Dhingra 2025, PRESENT-SHD: sensitivity at four hospitals and in ELSA-Brasil": {
        "92.5%",
        "96.0%",
        "87.5%",
    },
    "Attia 2021: the label, the AUROC and the sensitivity at the derivation cut-off": {
        "35%",
        "0.82",
        "26.9%",
    },
    "Carter 2026: patients, label, sensitivity and specificity": {
        "13,960",
        "40%",
        "84.5%",
        "83.6%",
    },
    "Wagner 2020: PTB-XL's recording years": {"1989", "1996"},
    "de Vries 2023: the local screens the mammography cut-off was reset on": {"16,204"},
}

# Constants of the design, not results.
CONSTANTS = {
    "90%": "the sensitivity each threshold is set for, and the level of the refit",
    "90": "90 of 100 ill patients, the same target counted in patients; the 90th percentile",
    "1": "the +1 of the rank correction",
    "95%": "the level of every interval",
    "100": "outcomes are counted per 100 patients; the rung of the ladder the text reads",
    "1,000": "the clinic of 1,000 outpatients the counts are scaled to",
    "100,000": "EchoNext's ECGs",
    "10%": "a screening prevalence, and a decision threshold of net benefit",
    "5%": "a screening prevalence, and a decision threshold of net benefit",
    "20%": "a decision threshold of net benefit",
    "80%": "the middle 80% of the repetitions a band of Figure 3 spans",
    "45%": "EchoNext's ejection-fraction threshold",
    "45": "EchoNext's pulmonary pressure threshold, in mmHg",
    "1.3": "EchoNext's wall-thickness threshold, in cm",
    "3.2": "EchoNext's tricuspid velocity threshold, in m/s",
    "18": "the lower edge of the youngest age band",
    "49": "the upper edge of the youngest age band",
    "80": "the lower edge of the oldest age band",
    "25": "a rung of the ladder",
    "200": "the draws of every ladder, and a rung",
    "23": "the download date of the public corpora, 23 August 2026",
    "2026": "the year of the study and of the download",
    "2020": "part of PTB-XL's ethics reference, PTB-2020-1",
    "2024": "part of Chongqing's approval number, 2024-256-01",
    "256": "part of Chongqing's approval number, 2024-256-01",
    "01": "part of Chongqing's approval number, 2024-256-01",
    "6.6": "the size of ptbxl_database.csv in MB",
    "8700": "the CPU the timings were taken on, an i7-8700",
    "28": "the minutes the full suite took on a cold clone",
    "2012": "Vovk's ACML year in the README's sources",
    "475": "Vovk's first page in the README's sources",
    "490": "Vovk's last page in the README's sources",
    "2021": "the Challenge-2021 collection, and its year in the sources",
    "2022": "the SPH paper's year in the sources",
    "2025": "the EchoNext and ECGFounder papers' year in the sources",
    "2023": "Angelopoulos and Bates' year in the README's sources",
    "5": "the day of the literature search, 5 October 2026",
}

SECTION = re.compile(
    r"\b(?:Tables?|Figures?|figures?|tables?|[Ss]ections?) S?\d+(?:\.\d+)?"
    r"(?:(?:,| to| and) S?\d+(?:\.\d+)?)*"
)


def prose(text: str) -> str:
    """The text a number is read from: no reference list, link, code or citation mark."""
    for first, last in (("## References", None), ("## Licence and citation", "## Sources")):
        if first in text:
            start = text.index(first)
            text = text[:start] + (text[text.index(last) :] if last else "")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", text)
    text = re.sub(r"doi:\S+|arXiv:\S+|\b10\.\d{4,5}/\S+", " ", text)
    text = re.sub(r"^#+ .*$", " ", text, flags=re.M)
    return SECTION.sub(" ", text)


def typed(template: str) -> set[str]:
    """The numbers a template writes itself rather than reading them from a file."""
    return set(NUMBER.findall(prose(PLACEHOLDER.sub(" ", template))))


def allowed() -> set[str]:
    cited = {n for figures in LITERATURE.values() for n in figures}
    return cited | set(CONSTANTS)


@pytest.mark.parametrize("target", [REPORT, SUPPLEMENT, README, CITATION], ids=lambda p: p.name)
def test_the_committed_text_is_its_template_rendered_from_results(target: Path) -> None:
    """A figure edited by hand, or a results file regenerated without re-rendering, fails."""
    assert target.read_text() == rendered(target), (
        f"{target.name} differs from its rendering: uv run python tests/paper_render.py --write"
    )


def checked(target: Path) -> str:
    """The template text whose numbers are held: CITATION.cff's abstract, not its metadata."""
    text = (TEMPLATES / target.name).read_text()
    if target == CITATION:
        return text[text.index("abstract:") : text.index("keywords:")]
    return text


@pytest.mark.parametrize("target", [REPORT, README, CITATION], ids=lambda p: p.name)
def test_the_template_types_no_number_a_file_should_give(target: Path) -> None:
    """A number typed into the report or README is a number no file holds."""
    assert typed(checked(target)) - allowed() == set()


def test_every_constant_and_cited_figure_is_still_typed() -> None:
    """A constant nobody types any more is a door left open for a typed number."""
    used = set().union(*(typed(checked(t)) for t in (REPORT, README, CITATION)))
    assert allowed() - used == set()


def test_a_number_typed_by_hand_is_refused() -> None:
    assert typed("The threshold caught 71.9% of them.") - allowed() == {"71.9%"}
    assert typed("The threshold caught {{sens_resnet_out}} of them.") - allowed() == set()


def test_a_figure_swapped_for_another_figure_of_the_same_file_is_caught() -> None:
    """The specificity among inpatients written where the outpatient one belongs fails."""
    figures = v.values()
    text = REPORT.read_text()
    right = f"It also flagged fewer healthy patients: {figures['flagged_resnet_out']} of 100"
    assert right in text
    assert figures["flagged_resnet_out"] != figures["flagged_resnet_in"]
    swapped = text.replace(
        right, f"It also flagged fewer healthy patients: {figures['flagged_resnet_in']} of 100", 1
    )
    assert swapped != rendered(REPORT)


def test_a_name_no_file_defines_is_refused() -> None:
    with pytest.raises(KeyError, match="no_such_figure"):
        render("caught {{no_such_figure}} of them")


class TestTheReportsShape:
    def test_the_report_ends_at_its_references(self) -> None:
        """Ruben, 5/10: no appendix; what only a statistician needs is in SUPPLEMENT.md."""
        text = REPORT.read_text()
        headings = re.findall(r"^## (.+)$", text, flags=re.M)
        assert headings[-1] == "References"
        assert not re.search(r"appendix", text, flags=re.I)

    def test_the_supplement_is_linked_once_from_methods(self) -> None:
        text = REPORT.read_text()
        assert text.count("](SUPPLEMENT.md)") == 1
        methods = text[text.index("## 2. Methods") : text.index("## 3. Results")]
        assert "](SUPPLEMENT.md)" in methods

    def test_the_three_documents_carry_one_title(self) -> None:
        title = REPORT.read_text().splitlines()[0].removeprefix("# ")
        assert README.read_text().splitlines()[0] == f"# {title}"
        assert f'title: "{title}"' in CITATION.read_text()
        assert "conformal" not in title.lower()
        assert not re.search(r"\d", title)

    def test_the_seven_conditions_are_named_in_the_report(self) -> None:
        """Ruben, 5/10: name all seven conditions."""
        text = REPORT.read_text().lower()
        for condition in (
            "structural heart disease",
            "infarction",
            "sinus rhythm",
            "atrial fibrillation",
            "left bundle-branch block",
            "right bundle-branch block",
            "first-degree atrioventricular block",
        ):
            assert condition in text, condition

    def test_the_two_phrases_ruben_could_not_follow_are_gone(self) -> None:
        """Ruben, 5/10: 'how well the model ranks' and 'three ways of deciding'."""
        for target in (REPORT, README, CITATION):
            text = target.read_text().lower()
            assert "how well the model ranks" not in text
            assert "three ways of deciding" not in text

    def test_the_readme_points_at_the_report_and_the_supplement(self) -> None:
        text = README.read_text()
        assert "](REPORT.md)" in text and "](SUPPLEMENT.md)" in text
        assert not Path(REPORT.parent / "ECHONEXT.md").exists()


class TestClaims:
    """Each sentence that reads a figure in words is held to the figure."""

    def test_the_three_models_catch_the_same_share(self) -> None:
        """'Three models lose the same share': each within a point of the others, and the
        paired intervals inside the bound the text prints."""
        clinical = v.read("echonext_clinical.json")
        sens = [clinical["arms"][a]["outpatient"]["sensitivity"] for a in v.STRONGEST]
        assert max(sens) - min(sens) < 0.015
        for d in clinical["paired"]["outpatient_sensitivity"].values():
            assert d["low"] < 0 < d["high"]

    def test_the_ill_threshold_of_the_pair_is_the_sensitivity_threshold(self) -> None:
        """S1.1: under the pair, the ill whose set holds 'ill' are the plain threshold's."""
        for row in v.coverage_rows():
            if row["method"] == "perlabel":
                plain = v.cell(row["arm"], row["context"], "plain", row["label"])
                assert row["coverage_pos"] == plain["coverage_pos"], row

    def test_the_second_threshold_misses_no_one_new_and_clears_no_one_new(self) -> None:
        """'The second threshold did not change who was missed.'"""
        clinical = v.read("echonext_clinical.json")
        for arm, measured in clinical["arms"].items():
            for context in v.CONTEXT:
                o = measured[context]["outcomes"]
                assert o["perlabel"]["ill"]["wrong_alone"] == o["plain"]["ill"]["wrong_alone"], arm
                assert (
                    o["perlabel"]["healthy"]["right_alone"] == o["plain"]["healthy"]["right_alone"]
                )

    def test_it_reads_half_or_more_of_the_flagged_ill_and_nearly_all_flagged_healthy(self) -> None:
        clinical = v.read("echonext_clinical.json")
        for arm in v.STRONGEST:
            o = clinical["arms"][arm]["outpatient"]["outcomes"]
            caught = o["plain"]["ill"]["right_alone"]["share"]
            flagged = o["plain"]["healthy"]["wrong_alone"]["share"]
            assert o["perlabel"]["ill"]["deferred"]["share"] / caught >= 0.5, arm
            assert o["perlabel"]["healthy"]["deferred"]["share"] / flagged >= 0.9, arm

    def test_the_curve_held_and_both_groups_scored_lower(self) -> None:
        """'Separated almost as well': the AUROC moves by under two hundredths, while
        sensitivity falls and specificity rises from inpatients to outpatients."""
        transfer = v.read("echonext_transfer.json")
        clinical = v.read("echonext_clinical.json")
        for arm in v.STRONGEST:
            a = transfer["arms"][arm]["auroc"]
            assert (
                abs(a["inpatient"][v.COMPOSITE]["auroc"] - a["outpatient"][v.COMPOSITE]["auroc"])
                < 0.02
            )
            inside, outside = (
                clinical["arms"][arm]["inpatient"],
                clinical["arms"][arm]["outpatient"],
            )
            assert outside["sensitivity"] < inside["sensitivity"] - 0.1
            assert outside["specificity"] > inside["specificity"] + 0.2

    def test_ninety_percent_costs_most_of_the_healthy_outpatients(self) -> None:
        """'Restoring 90 caught costs most of the healthy outpatients a flag.'"""
        roc = v.read("echonext_clinical.json")["roc"]
        at90 = roc["resnet"]["sensitivity"].index(0.9)
        for arm in v.STRONGEST:
            assert roc[arm]["outpatient"][at90] < 0.5, arm

    def test_about_a_quarter_is_the_mean_share_the_case_mix_explains(self) -> None:
        mix = v.read("echonext_clinical.json")["case_mix"]
        shares = [mix[a]["share_explained"]["estimate"] for a in v.STRONGEST]
        assert 0.2 <= sum(shares) / len(shares) <= 0.3
        assert all(mix[a]["share_explained"]["high"] < 0.5 for a in v.STRONGEST)
        assert "about a quarter" in README.read_text()
        assert "about a quarter of the fall" in REPORT.read_text()

    def test_the_refit_reaches_ninety_on_average_and_one_repetition_in_three_falls_short(
        self,
    ) -> None:
        """'About one repetition in three', for each of the three models, on enough draws
        that the Monte Carlo error of the share is near one point; and 200 labels do not
        lower the share."""
        ladder = v.read("echonext_clinical.json")["ladder"]
        for arm in v.STRONGEST:
            rows = {r["labels"]: r for r in ladder[arm]}
            assert rows[100]["draws"] >= 2000
            assert rows[100]["sensitivity"]["mean"] >= 0.9, arm
            assert rows[100]["specificity"]["mean"] < 0.4, arm
            assert 0.27 <= rows[100]["share_of_draws_below_level"] <= 0.39, arm
            share100 = rows[100]["share_of_draws_below_level"]
            assert rows[200]["share_of_draws_below_level"] >= share100 - 0.02, arm
            assert rows[200]["sensitivity"]["p90"] - rows[200]["sensitivity"]["p10"] < (
                rows[100]["sensitivity"]["p90"] - rows[100]["sensitivity"]["p10"]
            )
        assert "about one repetition in three" in REPORT.read_text()
        for target in (REPORT, README):
            assert "one repetition in four" not in target.read_text()

    def test_the_fixed_half_sits_high_among_the_means_of_random_halves(self) -> None:
        """S1.9: the fixed half's mean, set among the means of other halves, is above
        most of them for each model and above almost all for the mini-model."""
        transfer = v.read("echonext_transfer.json")
        clinical = v.read("echonext_clinical.json")
        rank = {}
        for arm in v.STRONGEST:
            (fixed,) = [
                r
                for r in transfer["arms"][arm]["ladder"]["outpatient"][v.COMPOSITE]
                if r["labels"] == 100
            ]
            means = clinical["half_means"][arm]["means"]
            rank[arm] = sum(m < fixed["coverage_pos_mean"] for m in means) / len(means)
        assert min(rank.values()) > 0.6 and rank["echonext_mini"] > 0.99
        assert "one of the more favourable ones" not in SUPPLEMENT.read_text()

    def test_the_one_page_reports_say_their_ladder_is_superseded(self) -> None:
        """README links each model's page; the page says its ladder reads one fixed half,
        points to the report, and gives the specificity beside the healthy covered."""
        pages = sorted((v.ROOT / "reports/transfer").glob("*_inpatient_to_outpatient.md"))
        assert len(pages) == 4
        for page in pages:
            text = page.read_text()
            assert "those figures supersede this ladder's" in text, page.name
            assert "of those without it unflagged" in text, page.name

    def test_the_fall_holds_within_each_band_of_recording_years(self) -> None:
        """Section 2.1: the outpatients' ECGs are older, and in every band of years the
        outpatients' sensitivity stays below every inpatient band's, for each model."""
        eras = v.read("echonext_clinical.json")["eras"]
        assert eras["years"]["outpatient"]["median"] < eras["years"]["inpatient"]["median"]
        for arm in v.STRONGEST:
            outside = max(r["share"] for r in eras["arms"][arm]["outpatient"])
            inside = min(r["share"] for r in eras["arms"][arm]["inpatient"])
            assert outside < inside - 0.05, arm
        text = REPORT.read_text()
        assert "nothing else separates the cohorts" not in text
        assert "Each patient contributes one ECG and belongs to one group only" not in text

    def test_more_healthy_outpatients_than_inpatients_carry_no_measurement(self) -> None:
        """Section 2.1 and the limitations: the healthy without a measurement are a larger
        share among outpatients, and no ill ECG lacks one."""
        u = v.read("echonext_clinical.json")["unmeasured"]
        share = {c: u[c]["n_healthy_unmeasured"] / u[c]["n_healthy"] for c in u}
        assert share["outpatient"] > 2 * share["inpatient"]
        assert all(u[c]["n_ill_unmeasured"] == 0 for c in u)
        assert "within a year before a transthoracic" not in REPORT.read_text()

    def test_set_on_separate_outpatients_the_threshold_falls_short_of_ninety(self) -> None:
        """Section 3.6 and the abstract: on outpatients it never saw, the threshold set on
        the validation outpatients catches under 90% for each model, the trained network's
        interval excludes 90%, and it flags more than half of the healthy."""
        variants = v.read("echonext_clinical.json")["calibration_variants"]
        for arm in v.STRONGEST:
            held = variants["outpatients"]["arms"][arm]["outpatient"]
            assert held["sensitivity"]["share"] < 0.9, arm
            assert held["specificity"]["share"] < 0.5, arm
        trained = variants["outpatients"]["arms"]["resnet"]["outpatient"]["sensitivity"]
        assert trained["high"] < 0.9
        text = REPORT.read_text() + README.read_text() + CITATION.read_text()
        assert "restore" not in text.lower()
        assert "short of 90% for all three" in REPORT.read_text()

    def test_net_benefit_orders_the_options_as_the_discussion_says(self) -> None:
        """At 5% an echocardiogram for everyone nets the most, at 20% the inpatient
        threshold does, and at 10% the refit beats the inpatient threshold."""
        for arm in v.STRONGEST:
            o = v.read("echonext_clinical.json")["decision"][arm]["options"]
            nb = {k: x["net_benefit"] for k, x in o.items()}
            assert max(nb, key=lambda k: nb[k]["0.05"]) == "echo_for_all", arm
            assert max(nb, key=lambda k: nb[k]["0.20"]) == "inpatient_threshold", arm
            assert nb["refit_100"]["0.10"] > nb["inpatient_threshold"]["0.10"], arm
        assert "which this study does not measure" not in REPORT.read_text()
        assert "Table S7d" in REPORT.read_text()

    def test_the_recalibration_lowered_net_benefit_for_echonext_outpatients(self) -> None:
        for cell in v.read("repairs.json")["cells"]:
            if (cell["family"], cell["target"], cell["label"]) != (
                "echonext",
                "outpatient",
                v.COMPOSITE,
            ) or cell["model"] not in v.STRONGEST:
                continue
            for row in cell["net_benefit"]:
                if row["threshold"] in (0.05, 0.1):
                    assert row["recalibrated"] < row["as_delivered"], cell["model"]
        assert "it lowered net benefit at decision thresholds of 5% and 10%" in README.read_text()

    def test_set_on_every_validation_patient_the_fall_shrinks_and_remains(self) -> None:
        variants = v.read("echonext_clinical.json")["calibration_variants"]
        for arm in v.STRONGEST:
            paper = variants["inpatients"]["arms"][arm]["outpatient"]["sensitivity"]["share"]
            every = variants["every_setting"]["arms"][arm]["outpatient"]["sensitivity"]
            assert paper < every["share"] and every["high"] < 0.9, arm

    def test_what_a_clinic_sees_without_labels_falls_below_what_prevalence_allows(self) -> None:
        """Section 3.5: no prevalence takes the share flagged, or the share sent to a reader,
        below the floor the inpatients' rates set, and the outpatients' shares lie below it."""
        for arm in v.STRONGEST:
            measured = v.read("echonext_clinical.json")["arms"][arm]
            floor, expected = v.label_free_bounds(measured)
            seen = measured["outpatient"]["label_free"]
            assert seen["share_flagged"] < floor["flagged"] <= expected["flagged"], arm
            assert seen["share_deferred"] < floor["deferred"] <= expected["deferred"], arm
            assert seen["share_flagged"] < measured["inpatient"]["label_free"]["share_flagged"]

    def test_the_label_free_floor_is_the_smaller_rate_at_any_prevalence(self) -> None:
        arm = {
            "inpatient": {
                "sensitivity": 0.9,
                "specificity": 0.4,
                "outcomes": {
                    "perlabel": {
                        "ill": {"deferred": {"share": 0.37}},
                        "healthy": {"deferred": {"share": 0.49}},
                    }
                },
            },
            "outpatient": {"n": 100, "n_ill": 25},
        }
        floor, expected = v.label_free_bounds(arm)
        assert floor == {"flagged": 0.6, "deferred": 0.37}
        assert expected["flagged"] == pytest.approx(0.25 * 0.9 + 0.75 * 0.6)
        assert "same fall" not in REPORT.read_text()
        assert "Nothing visible without diagnoses" not in REPORT.read_text()

    def test_the_findings_that_kept_and_lost_their_sensitivity(self) -> None:
        """Section 3.1, under the composite threshold: a thick wall loses the most for each
        model, an ejection fraction of 45% or less and aortic stenosis far less."""
        for arm in v.STRONGEST:
            found = v.read("echonext_clinical.json")["by_finding"][arm]
            share = {
                (c, label): found[c][label]["caught"] / found[c][label]["n"]
                for c in ("inpatient", "outpatient")
                for label in (
                    "lvef_lte_45_flag",
                    "aortic_stenosis_moderate_or_greater_flag",
                    "lvwt_gte_13_flag",
                )
            }
            wall = share["inpatient", "lvwt_gte_13_flag"] - share["outpatient", "lvwt_gte_13_flag"]
            assert wall > 0.15, arm
            for label in ("lvef_lte_45_flag", "aortic_stenosis_moderate_or_greater_flag"):
                loss = share["inpatient", label] - share["outpatient", label]
                assert loss < wall - 0.1, arm
                if arm == "resnet":  # the figures section 3.1 prints
                    assert loss < 0.05
        assert "A threshold set the same way for each finding on its own" not in REPORT.read_text()
        assert "ill more mildly" not in REPORT.read_text()

    def test_women_and_the_young_are_caught_less_by_each_model(self) -> None:
        for t in v.read("echonext_clinical.json")["subgroups"]["tests"]:
            g = {name: c["caught"] / c["n"] for name, c in t["groups"].items()}
            if t["kind"] == "sex":
                assert g["female"] < g["male"], t["arm"]
            else:
                assert g["18-49"] < g["80+"], t["arm"]

    def test_shandong_caught_more_than_promised_and_chongqing_less(self) -> None:
        sites = v.read("infarction_sites.json")["sites"]
        assert sites["sph"]["sensitivity_interval"][0] > 0.9
        assert sites["acs"]["sensitivity_interval"][1] < 0.9

    def test_the_ptbxl_interval_is_the_spread_of_the_draws_each_read_on_half_the_fold(self) -> None:
        """Each PTB-XL draw reads about half of fold 10, so its interval is the spread
        between draws, at least as wide as 1.96 of their standard deviations."""
        sites = v.read("infarction_sites.json")["sites"]
        draws = v.read("outcomes.json")["by_corpus"]["ptbxl"]["schemes"]["plain"]["1"]["correct"]
        low, high = sites["ptbxl"]["sensitivity_interval"]
        assert high - low >= 2 * 1.95 * draws["sd"]
        assert sites["ptbxl"]["n_mi_read_per_draw"] < 0.6 * sites["ptbxl"]["n_mi"]
        assert "{{mi_pos_read_ptbxl}}" in (v.ROOT / "docs/templates/REPORT.md").read_text()
        assert sites["acs"]["auroc"]["high"] < sites["ptbxl"]["auroc"]["low"]


def test_table_s7_brackets_hold_the_percentiles_its_legend_names() -> None:
    """Table S7's legend promises the 10th to 90th percentile of the draws for every
    bracket, the count of ill patients in the sample included."""
    text = SUPPLEMENT.read_text()
    assert "with the 10th to 90th percentile of the draws in brackets" in text
    for arm in v.STRONGEST:
        for row in v.read("echonext_clinical.json")["ladder"][arm]:
            ill = row.get("ill_in_sample")
            if ill is None:
                continue
            cell = f"| {v.ARM_NAMES[arm]} | {row['labels']} | {ill['mean']:.1f} "
            cell += f"({ill['p10']:.0f} to {ill['p90']:.0f}) |"
            assert cell in text, cell


def test_every_reference_of_the_report_is_cited_in_its_text() -> None:
    """A reference no sentence cites is a claim the reader cannot place."""
    text = REPORT.read_text()
    body, references = text.split("## References")
    listed = {int(n) for n in re.findall(r"^(\d+)\. ", references, flags=re.M)}
    cited = {
        int(n) for group in re.findall(r"\[(\d+(?:,\s*\d+)*)\]", body) for n in group.split(",")
    }
    assert listed == cited
