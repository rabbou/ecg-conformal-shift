"""REPORT.fr.md carries every number and figure of REPORT.md, and no other.

The French text is translated by hand, not rendered, so a re-render of
REPORT.md that moves a figure fails here until the translation follows.
French prose writes 5,3 and 1 903 (a narrow no-break space between groups)
where English writes 5.3 and 1,903; the reference list stays in English in
both, so it is read with the English rule.
"""

from __future__ import annotations

import re
from collections import Counter

from paper_render import REPORT

REPORT_FR = REPORT.with_name("REPORT.fr.md")

# Digits, then thousands groups, then any run of decimal or dotted parts:
# 1,903 · 5.3 · v1.1.1 · [2,3] · 1 903 · 5,3.
EN_NUMBER = re.compile(r"\d+(?:,\d{3}(?!\d))*(?:[.,]\d+)*")
FR_NUMBER = re.compile(r"\d+(?: \d{3}(?!\d))*(?:[.,]\d+)*")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def english_numbers(text: str) -> Counter[str]:
    found = (re.sub(r",(?=\d{3}(?!\d))", "", n) for n in EN_NUMBER.findall(text))
    return Counter(n.replace(",", ".") for n in found)


def french_numbers(text: str) -> Counter[str]:
    body, references = text.split("## Références", 1)
    found = (n.replace(" ", "").replace(",", ".") for n in FR_NUMBER.findall(body))
    return Counter(found) + english_numbers(references)


def test_the_french_report_carries_the_same_numbers() -> None:
    english = english_numbers(REPORT.read_text())
    french = french_numbers(REPORT_FR.read_text())
    assert french - english == Counter() and english - french == Counter(), (
        f"only in REPORT.fr.md: {dict(french - english)}; "
        f"only in REPORT.md: {dict(english - french)}"
    )


def test_the_french_report_shows_the_same_figures_in_the_same_order() -> None:
    assert IMAGE.findall(REPORT_FR.read_text()) == IMAGE.findall(REPORT.read_text())


def test_the_french_report_keeps_the_sections() -> None:
    def sections(text: str) -> list[tuple[str, str]]:
        return re.findall(r"^(#{2,3}) (\d[\d.]*)?", text, flags=re.M)

    assert sections(REPORT_FR.read_text()) == sections(REPORT.read_text())


def test_the_rules_tell_french_and_english_numbers_apart() -> None:
    assert english_numbers("1,903 at 5.3 [2,3] v1.1.1") == Counter(
        {"1903": 1, "5.3": 1, "2.3": 1, "1.1.1": 1}
    )
    assert french_numbers("1 903 à 5,3 [2,3] v1.1.1\n## Références\n19,955") == Counter(
        {"1903": 1, "5.3": 1, "2.3": 1, "1.1.1": 1, "19955": 1}
    )
