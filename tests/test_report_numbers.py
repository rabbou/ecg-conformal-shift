"""Every number REPORT.md and README.md print is a named figure read from ``results/``.

The two documents are rendered from ``docs/templates/``, where each computed
number is written as ``{{name}}`` and resolved by ``tests/report_values.py``.
Three checks hold the text to the files:

  the committed text is the template rendered from the committed results, so a
  number changed by hand fails here even when it equals another figure a file
  holds, and a results file that moved fails here until the text is rendered;
  every number the template still types is a figure of a cited source or a
  constant of the design named below with its reason;
  every claim the prose makes about its figures is asserted where the figure is
  built, in ``report_infarction.py`` and ``report_values.py``.

Rendering: ``uv run python tests/report_render.py --write``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import report_figures as f
from report_render import PLACEHOLDER, TEMPLATES, render, rendered
from report_text import NUMBER, README, REPORT

# Figures quoted from the cited papers, by the reference that holds them.
LITERATURE = {
    "[7] Dzikowicz 2025": {"0.923", "0.932", "0.943", "84.8%", "86.7%"},
    "[8] Folgado 2025": {"62", "20", "0.53", "0.67", "0.62", "-0.20"},
    "[9] Wagner 2020, the privacy age and the recording years": {"89", "300", "1989", "1996"},
    "[14] Kinalioglu 2026": {"90.0%", "7.5%", "82.5%"},
    "[15] El Allam 2026": {"36.4%", "91.64%", "88.59%", "90.65%", "90.53%", "3.06", "0.12"},
    "[16] Chow 1970": {"1970"},
    "[17] Poterucha 2025, Table 2": {"84.3%", "84.1%"},
    "[18] Hughes 2026": {"82.0%", "0.1", "99.9"},
    "[19] Attia 2021": {"4,277", "0.82", "26.9%"},
    "[20] Carter 2026": {"84.5%", "83.6%"},
    "[21] Otabor 2026": {"0.790", "0.820"},
    "[22] Poterucha 2026": {"65", "85"},
    "[23] Millar 2026": {"85.8%", "77.5%", "3", "1.5"},
}

# Figures of the tagged release v1.0.0 that the text quotes as such.
EARLIER = {"v1.0.0, section 3.4, the pooled sex gap's interval": {"−0.2", "22.6"}}

# Constants of the design, not results: each is named with what it is.
CONSTANTS = {
    "0": "seed 0, and the records a corpus loses at read time where it loses none",
    "1": "the (n+1) of the conformal rank, and the first PTB-XL fold",
    "2": "a stride of the network; the 2 September literature search",
    "4": "the 4 September search; the 4·10⁻³ of the determinism probe",
    "5": "the day of the draft",
    "7": "the kernel width of the network",
    "8": "the PTB-XL training folds (1 to 8); the 8 September search",
    "9": "PTB-XL's validation fold",
    "10": "PTB-XL's test fold; the 10% that the 90% sensitivity leaves out",
    "12": "the leads",
    "15": "the stem convolution's width",
    "23": "the download date of the public corpora, 23 August 2026",
    "25": "the lowest rung of the EchoNext ladder",
    "30": "the smallest severity cell judged on its own",
    "35": "a severity band edge of ejection fraction",
    "36": "a severity band edge of ejection fraction",
    "45": "the ejection-fraction threshold of the EchoNext label and a severity band edge",
    "50": "a rung of the ladder and the lower edge of an age band",
    "64": "the stem's channels and the batch size; the upper edge of an age band",
    "75": "the lower edge of the oldest age band",
    "90": "the 90 cases in 100 of a 90% target; the 90th percentile",
    "100": "the rung of the ladders the report reads; per 100 patients",
    "128": "a stage width of the network",
    "200": "the calibration draws, and a rung of the EchoNext ladder",
    "250": "EchoNext's sampling rate in Hz",
    "256": "a stage width of the network",
    "400": "the top rung of the EchoNext ladder",
    "500": "PTB-XL's sampling rate in Hz; the AUROC bootstrap draws; a Chongqing rung",
    "2,000": "the bootstrap replicates, and the top rung of the Chongqing ladder",
    "5,000": "the samples of a ten-second tracing at 500 Hz",
    "0.10": "the α of the 90% target",
    "0.3": "the frequency of the baseline wander fault, Hz",
    "0.5": "the amplitude of the baseline wander fault, mV",
    "0.8": "the gain of the first scaling fault",
    "1.25": "the gain of the second scaling fault",
    "10%": "the calibration MI cases below the 90%-sensitivity point",
    "45%": "the ejection-fraction threshold of the EchoNext label",
    "80%": "the second confidence level the results files carry",
    "90%": "the sensitivity and the coverage the thresholds are set for",
    "95%": "the level of every interval",
    "1989–96": "PTB-XL's recording years",
    "2019–20": "Shandong's recording years",
    "2015–24": "Chongqing's recording years",
    "2017": "Wang et al. 2017, the architecture's source",
    "2018": "part of a corpus name, CPSC 2018",
    "2021": "part of a corpus name, the Challenge-2021 collection",
    "2026": "the year of the draft and of the download",
    "2024-256-01": "the Chongqing ethics approval number",
    "2020-1": "the PTB-XL ethics approval number, PTB-2020-1",
    "2.2.2": "the PyTorch release the runs used",
    "3.11.15": "the Python release the runs used",
    "8700": "the CPU model the timings were taken on, i7-8700",
    "15233e93": "the IntroECG commit the weights come from",
    "4,082,306": "the parameter count of the infarction network, results/timing.json",
    "21,799": "PTB-XL's released records",
    "25,770": "Shandong's released records",
    "19,955": "Chongqing's released records",
    "45,152": "Chapman-Shaoxing and Ningbo's released records",
    "10,344": "Georgia's released records",
    "10,330": "CPSC's released records",
    "6.6": "the size of ptbxl_database.csv in MB",
    "28": "the minutes the full suite took on a cold clone",
}

SECTION = re.compile(
    r"\b(?:Tables?|Figures?|figures?|tables?|[Ss]ections?) [A-H]?\d+(?:\.\d+)?"
    r"(?:(?:,| to| and) \d+(?:\.\d+)?)*"
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
    return SECTION.sub(" ", text)


def typed(template: str) -> set[str]:
    """The numbers a template writes itself rather than reading them from a file."""
    return set(NUMBER.findall(prose(PLACEHOLDER.sub(" ", template))))


def allowed() -> set[str]:
    cited = " ".join(" ".join(v) for v in (LITERATURE | EARLIER).values())
    return set(NUMBER.findall(cited)) | set(NUMBER.findall(" ".join(CONSTANTS)))


@pytest.mark.parametrize("target", [REPORT, README], ids=lambda p: p.name)
class TestRendering:
    def test_the_committed_text_is_its_template_rendered_from_results(self, target: Path) -> None:
        assert target.read_text() == rendered(target), (
            f"{target.name} differs from its rendering: "
            "uv run python tests/report_render.py --write"
        )

    def test_the_template_types_no_number_a_file_should_give(self, target: Path) -> None:
        template = (TEMPLATES / target.name).read_text()
        assert typed(template) - allowed() == set()


def test_every_constant_and_cited_figure_is_still_typed() -> None:
    """A constant nobody types any more is a door left open for a typed number."""
    used = typed((TEMPLATES / "REPORT.md").read_text()) | typed(
        (TEMPLATES / "README.md").read_text()
    )
    named = set(NUMBER.findall(" ".join(CONSTANTS)))
    cited = {n for v in (LITERATURE | EARLIER).values() for n in NUMBER.findall(" ".join(v))}
    assert (named | cited) - used == set()


def test_a_figure_swapped_for_another_figure_of_the_same_file_is_caught() -> None:
    """The text rendered from results is the only text that passes, whatever the number."""
    figures = f.echonext_figures()
    specificity = f"from {figures['spec_in_low']}–"
    other = f"from {figures['spec_out_low']}–"
    text = REPORT.read_text()
    assert specificity in text and other != specificity
    assert text.replace(specificity, other, 1) != rendered(REPORT)


def test_a_name_no_file_defines_is_refused() -> None:
    with pytest.raises(KeyError, match="no_such_figure"):
        render("covered {{no_such_figure}} of them")


def test_a_number_typed_by_hand_in_the_template_is_refused() -> None:
    assert typed("The threshold recognised 71.9% of them.") - allowed() == {"71.9%"}
    assert typed("The threshold recognised {{strongest_low}} of them.") - allowed() == set()


class TestClaims:
    def test_about_a_third_is_the_mean_share_severity_explains(self) -> None:
        """'About a third' in the report and the README is the severity results' mean share."""
        severity = f.read("echonext_severity.json")["arms"]
        shares = [severity[a]["reweighted"]["share"] for a in f.STRONGEST]
        assert abs(sum(shares) / len(shares) - 1 / 3) < 0.03
        assert "explains about a third" in README.read_text()
        assert "Severity explains about a third" in REPORT.read_text()

    def test_about_72_holds_for_each_of_the_three(self) -> None:
        """'About 72%' is the mean of the three; each lies within a point and a half of it."""
        ill = [f.cell(a, "outpatient", "plain")["coverage_pos"] for a in f.STRONGEST]
        mean = sum(ill) / len(ill)
        assert all(abs(x - mean) < 0.015 for x in ill)

    def test_the_per_label_scheme_and_the_plain_threshold_flag_the_same_ill(self) -> None:
        """The report says the per-label threshold for the ill is the plain threshold."""
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

    def test_the_sex_gap_is_not_called_established(self) -> None:
        """One of eighteen uncorrected differences, whose sign flipped with the code path."""
        text = REPORT.read_text()
        assert "neither the gap nor its repair is established" in text
        assert "cannot settle either claim" in text
        assert "excludes zero by under a point" not in text
        assert "excludes it by under a point" not in text

    def test_age_gradients_are_read_within_the_label(self) -> None:
        """Class-conditional coverage at a fixed threshold cannot see prevalence."""
        text = REPORT.read_text()
        assert "prevalence gradient" not in text
        assert "depends only on how that label's tracings score" in text

    def test_the_rotation_chow_gap_is_read_as_the_correction(self) -> None:
        """The rotation's Chow rule is section 4's construction minus the correction."""
        text = REPORT.read_text()
        assert "the conformal apparatus still adds only the correction" in text
        assert "does not carry to other corpora" not in text
        assert "not about the rotation" not in text

    def test_a_hundred_labels_come_within_a_point_and_a_tenth_of_draws_fall_short(
        self,
    ) -> None:
        """The conclusion's reading of the two ladders at 100 labels."""
        rungs = {
            r["n_target_records"]: r["coverage_by_class"]["1"]["mean"]
            for r in f.read("target_scale.json")["rows"]
            if (r["alpha"], r["score"], r["correction"], r["family"])
            == (0.1, "lac", "none", "recalibrated")
        }
        assert 0.885 <= rungs[100] < 0.9
        at_100 = [f.ladder(arm)[100] for arm in f.STRONGEST]
        assert all(s["coverage_pos_mean"] >= 0.9 for s in at_100)
        assert any(s["coverage_pos_p10"] < 0.9 for s in at_100)
        assert "one draw in ten still fell below 90% on EchoNext" in REPORT.read_text()
        assert "were enough here" not in REPORT.read_text()


class TestReadme:
    def test_the_page_points_at_the_one_report(self) -> None:
        text = README.read_text()
        assert "](REPORT.md)" in text
        assert "ECHONEXT.md" not in text
        assert "reports/transfer" not in text

    def test_the_page_states_the_shared_training_split(self) -> None:
        """Three arms fitted to one split are not three independent tests of the cause."""
        text = README.read_text()
        assert "the cause" not in text
        assert text.count("fitted to the same EchoNext training split") == 2
