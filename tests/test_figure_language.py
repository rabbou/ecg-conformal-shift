"""The French figures carry the English figures' numbers, in French words.

Figures 1, 5, 6, 7 and 8 draw in both languages from one function each.  The checks
below hold what that promises: the marks (bars, histograms, lines, limits and
where every label sits) are identical in the two languages, every number printed
in a text is the same number once the decimal mark and thousands separator are
read, the words themselves differ, and the committed ``*_fr.png`` files are
what the script draws.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ecs.config import RESULTS_DIR

FRENCH = (
    "fig3_coverage_fr.png",
    "fig1_thresholds_fr.png",
    "fig2_outcomes_fr.png",
    "fig7_rotation_fr.png",
    "fig8_target_scale_fr.png",
)


def _builders() -> dict[str, Callable[[str], Any]]:
    import bilingual_figures as bf

    shift = json.loads((RESULTS_DIR / "shift.json").read_text())
    outcomes = json.loads((RESULTS_DIR / "outcomes.json").read_text())
    scores = dict(np.load(RESULTS_DIR / "baseline/scores.npz"))
    return {
        "coverage": lambda lang: bf.coverage_figure(shift, lang),
        "thresholds": lambda lang: bf.thresholds_figure(outcomes, scores, lang),
        "outcomes": lambda lang: bf.outcomes_figure(outcomes, lang),
    }


def _marks(figure: Any) -> list[np.ndarray]:
    """Every drawn quantity of a figure, without its words."""
    marks: list[np.ndarray] = []
    for axes in figure.axes:
        for patch in axes.patches:
            path = patch.get_path().transformed(patch.get_patch_transform())
            marks.append(np.asarray(path.vertices))
        for line in axes.lines:
            marks.append(np.asarray(line.get_xydata()))
        for collection in axes.collections:
            marks.extend(np.asarray(s) for s in getattr(collection, "get_segments", list)())
        marks.append(np.array([*axes.get_xlim(), *axes.get_ylim()]))
        marks.append(np.array([t.get_position() for t in axes.texts]).reshape(-1))
    return marks


def _texts(figure: Any) -> list[str]:
    """Every string on the figure, in drawing order, tick labels included."""
    figure.canvas.draw()
    found = [t.get_text() for t in figure.texts]
    if figure._suptitle is not None:
        found.append(figure._suptitle.get_text())
    for axes in figure.axes:
        found += [axes.get_title("left"), axes.get_title(), axes.get_xlabel(), axes.get_ylabel()]
        found += [t.get_text() for t in axes.texts]
        found += [t.get_text() for t in axes.get_xticklabels() + axes.get_yticklabels()]
        legend = axes.get_legend()
        if legend is not None:
            found += [t.get_text() for t in legend.get_texts()]
    for legend in figure.legends:
        found += [t.get_text() for t in legend.get_texts()]
    return [text for text in found if text]


def _numbers(texts: list[str], lang: str) -> list[float]:
    """The numbers a reader of that language would read off the strings."""
    values = []
    for text in texts:
        if lang == "fr":
            text = re.sub(r"(?<=\d) (?=\d{3})", "", text)
            text = re.sub(r"(?<=\d),(?=\d)", ".", text)
        else:
            text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
        text = text.replace("−", "-")
        values += [float(v) for v in re.findall(r"\d+(?:\.\d+)?", text)]
    return values


@pytest.fixture(scope="module")
def drawn() -> Iterator[dict[str, dict[str, Any]]]:
    import matplotlib.pyplot as plt

    figures = {
        name: {lang: build(lang) for lang in ("en", "fr")} for name, build in _builders().items()
    }
    yield figures
    plt.close("all")


class TestOneDataPathTwoLanguages:
    @pytest.mark.parametrize("name", ["coverage", "thresholds", "outcomes"])
    def test_both_languages_draw_the_same_marks(self, drawn: dict, name: str) -> None:
        english, french = (_marks(drawn[name][lang]) for lang in ("en", "fr"))
        assert len(english) == len(french)
        for a, b in zip(english, french, strict=True):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("name", ["coverage", "thresholds", "outcomes"])
    def test_every_printed_number_is_the_same_number(self, drawn: dict, name: str) -> None:
        english = _numbers(_texts(drawn[name]["en"]), "en")
        french = _numbers(_texts(drawn[name]["fr"]), "fr")
        assert english, name
        assert sorted(english) == sorted(french)

    @pytest.mark.parametrize("name", ["coverage", "thresholds", "outcomes"])
    def test_the_words_are_french(self, drawn: dict, name: str) -> None:
        french = " ".join(_texts(drawn[name]["fr"]))
        for english_word in ("infarction", "deferred", "share", "calibration draws", "label"):
            assert english_word not in french, (name, english_word)
        assert "\u00a0%" in french, "a French percentage takes a space before its sign"
        if re.search(r"\d\.\d", " ".join(_texts(drawn[name]["en"]))):
            assert re.search(r"\d,\d", french), "a French decimal is printed with a comma"
            assert not re.search(r"\d\.\d", french), "no English decimal is left"

    def test_the_catalogues_hold_the_same_keys(self) -> None:
        from figure_text import WORDS

        assert set(WORDS["en"]) == set(WORDS["fr"])

    def test_numbers_are_printed_as_each_language_prints_them(self) -> None:
        from figure_text import count, number, percent

        assert (percent(0.1, 0, "en"), percent(0.1, 0, "fr")) == ("10%", "10 %")
        assert (number(0.566, 2, "en"), number(0.566, 2, "fr")) == ("0.57", "0,57")
        assert (count(1648, "en"), count(1648, "fr")) == ("1,648", "1 648")

    def test_an_unknown_language_is_refused(self) -> None:
        from figure_text import words

        with pytest.raises(ValueError, match="no figure text"):
            words("de")


class TestTheCleanTitles:
    """No figure carries a double hyphen for a dash, or a footer naming the script."""

    def test_no_figure_in_either_language_carries_a_leftover(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import bilingual_figures
        import figures
        import matplotlib.pyplot as plt

        kept: list[Any] = []

        def keep(figure: Any, out: Path) -> Path:
            kept.append(figure)
            return out

        monkeypatch.setattr(figures, "_save", keep)
        monkeypatch.setattr(bilingual_figures, "_save", keep)
        assert figures.main(["--out", str(tmp_path)]) == 0
        assert figures.main(["--lang", "fr", "--out", str(tmp_path)]) == 0
        assert len(kept) == 7 + 5
        for figure in kept:
            for text in _texts(figure):
                assert "--" not in text, text
                assert "drawn by" not in text and "figures.py" not in text, text
        plt.close("all")


class TestTheFrenchSet:
    def test_the_committed_french_set_redraws_pixel_for_pixel(self, tmp_path: Path) -> None:
        import figures
        from PIL import Image

        assert figures.main(["--lang", "fr", "--out", str(tmp_path)]) == 0
        assert sorted(p.name for p in tmp_path.glob("*.png")) == sorted(FRENCH)
        for name in FRENCH:
            with Image.open(RESULTS_DIR / "figures" / name) as a, Image.open(tmp_path / name) as b:
                before = np.asarray(a.convert("RGBA"))
                after = np.asarray(b.convert("RGBA"))
            assert before.shape == after.shape, f"{name} changed size"
            assert int(np.count_nonzero(np.any(before != after, axis=-1))) == 0, name

    def test_a_figure_without_french_words_is_refused_in_french(self, tmp_path: Path) -> None:
        import figures

        with pytest.raises(SystemExit):
            figures.main(["--lang", "fr", "--figure", "2", "--out", str(tmp_path)])
        assert not list(tmp_path.glob("*.png"))
