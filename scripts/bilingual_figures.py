"""The three figures the report and the site show first, drawn in any language.

Fig. 1 of the report (thresholds), fig. 2 (outcomes) and fig. 3 (coverage per
hospital) each come from one function that takes a language: the same file, the
same arithmetic and the same layout, with the words of ``figure_text.py``.  The
English and the French set therefore carry the same numbers by construction,
and a test compares what the two draw.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from figure_text import count, number, percent, words  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

# Source first, then the two corpora the threshold is spent on unchanged.
CORPUS_ORDER = ("ptbxl", "sph", "acs")
CORPUS_NAMES = {corpus: words("en")[f"corpus.{corpus}"] for corpus in CORPUS_ORDER}


def _rows(table: dict[str, Any], score: str, correction: str) -> list[dict[str, Any]]:
    chosen = [r for r in table["rows"] if r["score"] == score and r["correction"] == correction]
    return sorted(chosen, key=lambda r: -float(r["alpha"]))


def corpora_on(table: dict[str, Any]) -> list[str]:
    """The corpora on the table, source first."""
    present = set(table["rows"][0]["by_corpus"])
    return [name for name in CORPUS_ORDER if name in present]


def _panel_title(table: dict[str, Any], corpus: str, lang: str = "en") -> str:
    block = table["rows"][0]["by_corpus"][corpus]
    text = words(lang)
    return text["coverage.panel"].format(
        name=text[f"corpus.{corpus}"],
        n=count(block["n_points"], lang),
        prevalence=percent(block["prevalence"], 1, lang),
    )


CORRECTION_ORDER = ("none", "mondrian", "weighted")
CORRECTION_NAMES = {c: words("en")[f"correction.{c}"] for c in CORRECTION_ORDER}
# The three readings inside a panel: the marginal figure, then the class it is
# bought from and the class it is bought for.
GROUP_ORDER = ("overall", "0", "1")
GROUP_COLOUR = {"overall": "#adb5bd", "0": "#219ebc", "1": "#bf4342"}


def corrections_on(table: dict[str, Any]) -> list[str]:
    """The corrections the table holds, in the order the argument runs."""
    present = {row["correction"] for row in table["rows"]}
    return [name for name in CORRECTION_ORDER if name in present]


def _save(figure: plt.Figure, out: Path) -> Path:
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


class _DecimalComma(ScalarFormatter):
    """Matplotlib's own tick labels with a decimal comma, as French prints them."""

    def __call__(self, x: float, pos: int | None = None) -> str:
        return str(super().__call__(x, pos)).replace(".", ",")


def _localise_ticks(figure: plt.Figure, lang: str) -> None:
    """Give every automatic numeric tick the language's decimal mark.

    Tick labels the figure sets itself (the levels asked for, the scheme names)
    are already written in the language and are left alone.
    """
    if lang != "fr":
        return
    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis):
            if type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(_DecimalComma())


def coverage_figure(table: dict[str, Any], lang: str = "en") -> plt.Figure:
    """Coverage against the level asked for, at each hospital and under each correction.

    One row per corpus, one column per correction, all from the same PTB-XL
    calibration: the first row is the population the threshold was fitted on, the
    two below it are populations it was merely spent on.  Reading a row left to
    right is reading what each repair does to that hospital; reading the red bar
    down a column is reading whether the sick are covered at all.
    """
    text = words(lang)
    corpora = corpora_on(table)
    corrections = corrections_on(table)
    figure, axes = plt.subplots(
        len(corpora),
        len(corrections),
        figsize=(4.4 * len(corrections), 3.7 * len(corpora)),
        sharey=True,
        sharex=True,
        squeeze=False,
    )

    for row_index, corpus in enumerate(corpora):
        for column, correction in enumerate(corrections):
            axis = axes[row_index][column]
            rows = _rows(table, "lac", correction)
            width = 0.26
            for offset, group in zip((-width, 0.0, width), GROUP_ORDER, strict=True):
                pairs = []
                for row in rows:
                    block = row["by_corpus"][corpus]
                    cell = (
                        block["coverage"]
                        if group == "overall"
                        else block["coverage_by_class"][group]
                    )
                    pairs.append((cell["mean"], cell["sd"]))
                axis.bar(
                    np.arange(len(rows)) + offset,
                    [value for value, _ in pairs],
                    width,
                    yerr=[sd for _, sd in pairs],
                    capsize=3,
                    label=text[f"group.{group}"],
                    color=GROUP_COLOUR[group],
                    edgecolor="white",
                )
            for index, row in enumerate(rows):
                axis.hlines(
                    row["target_coverage"],
                    index - 0.5,
                    index + 0.5,
                    colors="#333333",
                    linestyles="--",
                )
            axis.set_xticks(range(len(rows)))
            axis.set_xticklabels([percent(1 - float(r["alpha"]), 0, lang) for r in rows])
            axis.set_ylim(0.0, 1.05)
            if row_index == 0:
                axis.set_title(text[f"correction.{correction}"], fontsize=9)
            if row_index == len(corpora) - 1:
                axis.set_xlabel(text["coverage.xlabel"])
        axes[row_index][0].set_ylabel(
            f"{_panel_title(table, corpus, lang)}\n{text['coverage.ylabel']}", fontsize=8
        )
    handles, names = axes[0][-1].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=len(names), fontsize=9, frameon=False)
    figure.suptitle(_coverage_title(table, corrections, lang), fontsize=9, y=0.995, va="top")
    _localise_ticks(figure, lang)
    figure.tight_layout(rect=(0, 0.03, 1, 0.99))
    return figure


def _coverage_title(table: dict[str, Any], corrections: list[str], lang: str) -> str:
    """The title block: the headline infarction coverage, the spread, the estimates."""
    text = words(lang)
    headline = _rows(table, "lac", "none")[1]["by_corpus"]
    quoted = " · ".join(
        f"{text[f'correction.{c}'].splitlines()[0]} "
        + percent(
            _rows(table, "lac", c)[1]["by_corpus"]["ptbxl"]["coverage_by_class"]["1"]["mean"],
            0,
            lang,
        )
        for c in corrections
    )
    estimated = ("\u00a0; " if lang == "fr" else "; ").join(
        text["coverage.estimate"].format(
            corpus=text[f"corpus.{corpus}"].split(" (")[0],
            estimated=percent(block["calibration"]["estimated_prevalence"]["mean"], 1, lang),
            true=percent(headline[corpus]["prevalence"], 1, lang),
        )
        for corpus, block in _rows(table, "lac", "weighted")[1]["by_corpus"].items()
    )
    return "\n".join(
        (
            text["coverage.title"],
            text["coverage.quoted"].format(quoted=quoted),
            text["coverage.spread"].format(draws=table["n_draws"]),
            text["coverage.weighted"].format(estimated=estimated),
        )
    )


def figure_1_coverage(table: dict[str, Any], out: Path, lang: str = "en") -> Path:
    return _save(coverage_figure(table, lang), out)


# The three schemes of the outcome table, in the order the report reads them.
SCHEME_ORDER = ("plain", "pooled", "perlabel")
# Correct, deferred, wrong: the three things that can happen to a case.
OUTCOME_ORDER = ("correct", "deferred", "wrong")
OUTCOME_COLOURS = {"correct": "#4a7c59", "deferred": "#e0a458", "wrong": "#b4423a"}
CLASS_COLOURS = {0: "#7d8a93", 1: "#b4423a"}


def _deferred_share(source_block: dict[str, Any], scheme: str) -> float:
    """The share of all tracings a scheme defers, from its two per-label shares."""
    prevalence = source_block["prevalence"]
    sick = source_block["schemes"][scheme]["1"]["deferred"]["mean"]
    healthy = source_block["schemes"][scheme]["0"]["deferred"]["mean"]
    return float(prevalence * sick + (1 - prevalence) * healthy)


def _threshold_panel(
    axis: plt.Axes,
    scheme: str,
    cuts: tuple[float, ...],
    source_block: dict[str, Any],
    lang: str,
) -> None:
    """One scheme's boundaries on the score axis, and what they cost per label."""
    text = words(lang)
    low, high = cuts[0], cuts[-1]
    if high > low:
        axis.axvspan(low, high, color="#e0a458", alpha=0.30, zorder=0)
        axis.text(
            (low + high) / 2,
            52,
            text["thresholds.deferred"],
            ha="center",
            fontsize=9.5,
            color="#8a5b1c",
        )
    for cut in dict.fromkeys(cuts):
        axis.axvline(cut, color="black", lw=1.2, ls="--")
        axis.text(cut + 0.009, 44, number(cut, 2, lang), fontsize=8.5, color="#333", va="top")
    axis.text(
        low / 2, 52, text["thresholds.labelled.0"], ha="center", fontsize=9.5, color="#3d474d"
    )
    axis.text(
        (high + 1) / 2,
        52,
        text["thresholds.labelled.1"],
        ha="center",
        fontsize=9.5,
        color="#7a2f2a",
    )
    sick = source_block["schemes"][scheme]["1"]
    healthy = source_block["schemes"][scheme]["0"]
    axis.text(
        0.985,
        0.72,
        text["thresholds.box"].format(
            missed=percent(sick["wrong"]["mean"], 0, lang),
            alarms=percent(healthy["wrong"]["mean"], 0, lang),
            deferred=percent(_deferred_share(source_block, scheme), 0, lang),
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.5", "fc": "#f6f4ef", "ec": "#cfc9bd", "lw": 0.8},
    )
    axis.set_ylim(0, 58)
    axis.set_title(
        text[f"scheme.{scheme}"].replace("\n", " \u2013 "), fontsize=11.5, loc="left", pad=6
    )
    axis.set_ylabel(text["thresholds.ylabel"], fontsize=9.5)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def thresholds_figure(
    outcomes: dict[str, Any], scores: dict[str, Any], lang: str = "en"
) -> plt.Figure:
    """Where each scheme puts its thresholds, and what that costs per label.

    The score axis is the same in all three panels; only the thresholds move.
    Each panel carries the consequence beside the placement, so the figure does
    not need the table to be read.
    """
    text = words(lang)
    probs = scores["probs"][:, 1]
    labels = scores["labels"]
    source_block = outcomes["by_corpus"]["ptbxl"]
    thresholds = outcomes["thresholds"]
    bands: dict[str, tuple[float, ...]] = {
        "plain": (thresholds["plain:single"]["mean"],),
        "pooled": (thresholds["pooled:lower"]["mean"], thresholds["pooled:upper"]["mean"]),
        "perlabel": (thresholds["perlabel:lower"]["mean"], thresholds["perlabel:upper"]["mean"]),
    }
    bins = np.linspace(0.0, 1.0, 41)
    figure, axes = plt.subplots(3, 1, figsize=(8.8, 8.4), sharex=True)
    for axis, scheme in zip(axes, SCHEME_ORDER, strict=True):
        for klass in (0, 1):
            of_class = labels == klass
            axis.hist(
                probs[of_class],
                bins=bins,
                weights=np.full(int(of_class.sum()), 100.0 / int(of_class.sum())),
                color=CLASS_COLOURS[klass],
                alpha=0.7,
                label=text["thresholds.legend"].format(
                    name=text[f"class.{klass}"], n=count(int(of_class.sum()), lang)
                ),
            )
        _threshold_panel(axis, scheme, bands[scheme], source_block, lang)
    axes[0].legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(0.26, 0.86))
    axes[-1].set_xlabel(text["thresholds.xlabel"], fontsize=10)
    axes[-1].set_xlim(0, 1)
    for height, key in (
        (0.045, "thresholds.note.rates"),
        (0.030, "thresholds.note.deferred"),
        (0.015, "thresholds.note.halves"),
    ):
        figure.text(0.5, height, text[key], ha="center", fontsize=8.5, color="#555")
    _localise_ticks(figure, lang)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


def figure_5_thresholds(
    outcomes: dict[str, Any], scores: dict[str, Any], out: Path, lang: str = "en"
) -> Path:
    return _save(thresholds_figure(outcomes, scores, lang), out)


def outcomes_figure(outcomes: dict[str, Any], lang: str = "en") -> plt.Figure:
    """What a case of each label gets, under each of the three schemes."""
    text = words(lang)
    source_block = outcomes["by_corpus"]["ptbxl"]
    # The halving is by patient, so the test half's size varies by draw; the
    # counts are the means the table recorded, not a nominal half.
    scored = source_block["n_scored"]["mean"]
    positive = source_block["n_positive"]["mean"]
    panels = (
        ("1", text["outcomes.panel.1"].format(n=count(positive, lang))),
        ("0", text["outcomes.panel.0"].format(n=count(scored - positive, lang))),
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharex=True)
    positions = np.arange(len(SCHEME_ORDER))[::-1]
    for axis, (klass, title) in zip(axes, panels, strict=True):
        for position, scheme in zip(positions, SCHEME_ORDER, strict=True):
            cell = source_block["schemes"][scheme][klass]
            left = 0.0
            for name in OUTCOME_ORDER:
                share = cell[name]["mean"]
                width = share * 100
                axis.barh(position, width, left=left, color=OUTCOME_COLOURS[name], height=0.5)
                if width > 5:
                    axis.text(
                        left + width / 2,
                        position,
                        percent(share, 0, lang),
                        ha="center",
                        va="center",
                        color="#3a2c10" if name == "deferred" else "white",
                        fontsize=10.5 if width > 14 else 8,
                    )
                left += width
        axis.set_yticks(positions)
        axis.set_yticklabels([text[f"scheme.{s}"] for s in SCHEME_ORDER], fontsize=9)
        axis.set_xlim(0, 100)
        axis.set_title(title, fontsize=11.5, pad=10)
        axis.set_xlabel(text["outcomes.xlabel"], fontsize=9.5)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLOURS[n]) for n in OUTCOME_ORDER]
    figure.legend(
        handles,
        [text[f"outcome.{n}"] for n in OUTCOME_ORDER],
        fontsize=10,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=3,
        frameon=False,
    )
    schemes = source_block["schemes"]
    figure.text(
        0.5,
        0.01,
        text["outcomes.note"].format(
            draws=outcomes["n_draws"],
            matched=percent(schemes["plain"]["1"]["wrong"]["mean"], 0, lang),
            pooled=percent(schemes["pooled"]["1"]["wrong"]["mean"], 0, lang),
        ),
        ha="center",
        fontsize=8.5,
        color="#555",
        va="bottom",
    )
    _localise_ticks(figure, lang)
    figure.tight_layout(rect=(0, 0.19, 1, 0.97))
    return figure


def figure_6_outcomes(outcomes: dict[str, Any], out: Path, lang: str = "en") -> Path:
    return _save(outcomes_figure(outcomes, lang), out)
