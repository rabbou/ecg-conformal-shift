"""Draw the figures the report names, each from a committed file.

Each one is drawn from a file under ``results/`` and redraws from it pixel for
pixel, so a figure always comes from the numbers rather than from memory, and a
surprising result cannot be quietly re-cut.

| --figure | File | What it shows |
|---|---|---|
| 1 | fig3_coverage.png | coverage against the level asked for, per hospital and correction |
| 2 | fig2_set_sizes.png | how often the model answers with one label, both, or neither |
| 3 | fig3_arms.png | how far coverage falls per encoder arm |
| 4 | fig4_discrimination.png | baseline AUROC and AUPRC against the published reference |
| 5 | fig1_thresholds.png | where each scheme places its thresholds on the MI score |
| 6 | fig2_outcomes.png | what a case of each label gets under each scheme |
| 7 | fig7_rotation.png | coverage of five diagnoses across the source rotation |
| 8 | fig8_target_scale.png | coverage at Chongqing against labelled target records |

Figures 1, 5, 6, 7 and 8 are the five the report shows, and they also draw in
French: ``--lang fr`` runs the same functions on the same files
with the words of ``figure_text.py`` and writes ``*_fr.png`` beside the English
set.

The report numbers its own figures in its own text; the images carry titles,
not numbers, so the two cannot drift apart.

Usage: .venv/bin/python scripts/figures.py [--figure 1 ... 8] [--lang en|fr]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from bilingual_figures import (  # noqa: E402
    CORPUS_NAMES,
    CORPUS_ORDER,
    CORRECTION_ORDER,
    _localise_ticks,
    _panel_title,
    _rows,
    _save,
    corpora_on,
    figure_1_coverage,
    figure_5_thresholds,
    figure_6_outcomes,
)
from figure_text import LANGUAGES, SUFFIX, count, percent, words  # noqa: E402

from ecs.config import RESULTS_DIR  # noqa: E402

# Levels loosest first, so every figure reads left to right as confidence rising.
LEVELS = (0.20, 0.10, 0.05)
# The arms in contamination order: the control, then the arms that saw no public
# corpus, then the one that saw the calibration corpus, then the one that saw a
# target too.  Reading the figure left to right is reading that order.
ARM_ORDER = ("random_init", "ecgfounder", "ecg_jepa", "ecgfm", "hubert_ecg")
ARM_NAMES = {
    "random_init": "random init,\nfrozen",
    "ecgfounder": "ECGFounder",
    "ecgfm": "ECG-FM",
    "ecg_jepa": "ECG-JEPA",
    "hubert_ecg": "HuBERT-ECG",
}
# Grey for the arms that saw neither corpus, warm for the ones that saw PTB-XL,
# hottest for the one that saw a target as well.
ARM_COLOUR = {
    "random_init": "#adb5bd",
    "ecgfounder": "#6c757d",
    "ecgfm": "#fb8500",
    "ecg_jepa": "#8d99ae",
    "hubert_ecg": "#bf4342",
}

# One hue per level, dark enough to survive greyscale printing.
LEVEL_COLOUR = {0.20: "#8ecae6", 0.10: "#219ebc", 0.05: "#023047"}
SET_COLOUR = {"empty_rate": "#bf4342", "one_label_rate": "#5f8d4e", "two_label_rate": "#e9c46a"}
SET_NAME = {"empty_rate": "no label", "one_label_rate": "one label", "two_label_rate": "both"}


def figure_2_set_sizes(table: dict[str, Any], out: Path) -> Path:
    """How often the model answers with one label, both, or neither, per hospital."""
    corpora = corpora_on(table)
    figure, axes = plt.subplots(
        2, len(corpora), figsize=(3.6 * len(corpora), 8.0), sharey=True, squeeze=False
    )
    for row_index, score in enumerate(("lac", "aps")):
        for column, corpus in enumerate(corpora):
            axis = axes[row_index][column]
            rows = _rows(table, score, "none")
            bottom = np.zeros(len(rows))
            for key in ("one_label_rate", "two_label_rate", "empty_rate"):
                values = np.array([r["by_corpus"][corpus][key]["mean"] for r in rows])
                axis.bar(
                    range(len(rows)),
                    values,
                    0.55,
                    bottom=bottom,
                    label=SET_NAME[key],
                    color=SET_COLOUR[key],
                    edgecolor="white",
                )
                for index, (value, base) in enumerate(zip(values, bottom, strict=True)):
                    if value > 0.06:
                        axis.text(
                            index,
                            base + value / 2,
                            f"{value:.0%}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white",
                        )
                bottom = bottom + values
            axis.set_xticks(range(len(rows)))
            axis.set_xticklabels([f"{1 - float(r['alpha']):.0%}" for r in rows])
            axis.set_ylim(0, 1)
            if row_index == 0:
                axis.set_title(_panel_title(table, corpus), fontsize=9)
            else:
                axis.set_xlabel("confidence asked for")
        axes[row_index][0].set_ylabel(
            ("smallest sets (LAC)" if score == "lac" else "adaptive sets (APS)")
            + "\nshare of the corpus"
        )
    axes[0][-1].legend(loc="lower left", fontsize=8, framealpha=0.9)
    figure.suptitle(
        "Output composition against requested confidence\n"
        "One threshold, fitted on PTB-XL and applied to all three corpora. Tracings with "
        "both labels or no label go to a human reader.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.94))
    return _save(figure, out)


def _headline_row(arms: dict[str, Any], arm: str) -> dict[str, Any]:
    """The arm's coverage row at the level the grid quotes its headline at."""
    wanted = arms["headline"]
    for row in arms["coverage"][arm]:
        if all(row[key] == value for key, value in wanted.items()):
            return row
    raise KeyError(f"{arm} has no row for {wanted}")


def _saw(arms: dict[str, Any], arm: str, corpus: str) -> bool:
    """Whether this arm saw ``corpus`` at pre-training, read off the fact.

    Not off the sentence beside it: ECG-JEPA's pre-training is described as
    "not PTB-XL, not Shandong, not Chongqing", which contains the name of every
    corpus it did not see.
    """
    return corpus in arms["arms"][arm].get("saw", [])


def figure_3_arms(arms: dict[str, Any], out: Path) -> Path:
    """The five encoder arms on the same change of hospital.

    One panel per target: how far each arm's coverage falls between PTB-XL and
    that hospital, as a signed gap, so a bar above zero is coverage lost at that
    hospital and a bar below zero is coverage gained there.  The last
    panel is what each arm is worth at home, because a small gap earned by an
    arm that discriminates nothing is not the same achievement as a small gap
    earned by one that does; the panels have to be read together.

    The two target panels keep their own vertical scales.  Forcing one scale
    would flatten Shandong into a line beside Chongqing, and the comparison the
    figure is for is between arms inside a panel, not between panels.
    """
    targets = [c for c in CORPUS_ORDER if c != "ptbxl"]
    figure, axes = plt.subplots(1, len(targets) + 1, figsize=(14.0, 5.8), squeeze=False)
    positions = np.arange(len(ARM_ORDER))
    ticks = [ARM_NAMES[a] for a in ARM_ORDER]
    colours = [ARM_COLOUR[a] for a in ARM_ORDER]

    for axis, corpus in zip(axes[0], targets, strict=False):
        means = [_headline_row(arms, a)["coverage_gap"][corpus]["mean"] for a in ARM_ORDER]
        sds = [_headline_row(arms, a)["coverage_gap"][corpus]["sd"] for a in ARM_ORDER]
        axis.bar(positions, means, 0.6, yerr=sds, capsize=4, color=colours, edgecolor="white")
        axis.axhline(0.0, color="#333333", linewidth=1)
        low = min([*means, 0.0]) - max(sds)
        high = max([*means, 0.0]) + max(sds)
        pad = 0.22 * (high - low)
        axis.set_ylim(low - pad, high + pad)
        for index, (value, sd) in enumerate(zip(means, sds, strict=True)):
            above = value >= 0
            axis.text(
                index,
                value + (sd + 0.04 * (high - low)) * (1 if above else -1),
                f"{value:+.3f} ± {sd:.3f}",
                ha="center",
                va="bottom" if above else "top",
                fontsize=8,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(ticks, fontsize=8)
        axis.set_title(CORPUS_NAMES[corpus], fontsize=10)
        axis.set_ylabel("coverage at home minus coverage here", fontsize=9)

    home_axis = axes[0][-1]
    home = [arms["discrimination"][a]["ptbxl"]["auroc"] for a in ARM_ORDER]
    low_ci = [arms["discrimination"][a]["ptbxl"]["auroc_ci95"][0] for a in ARM_ORDER]
    high_ci = [arms["discrimination"][a]["ptbxl"]["auroc_ci95"][1] for a in ARM_ORDER]
    home_axis.bar(
        positions,
        home,
        0.6,
        yerr=np.array([np.subtract(home, low_ci), np.subtract(high_ci, home)]),
        capsize=4,
        color=colours,
        edgecolor="white",
    )
    for index, (value, lo, hi) in enumerate(zip(home, low_ci, high_ci, strict=True)):
        home_axis.text(
            index, hi + 0.03, f"{value:.3f}\n[{lo:.3f}, {hi:.3f}]", ha="center", fontsize=8
        )
    home_axis.axhline(0.5, color="#333333", linestyle=":", linewidth=1)
    home_axis.text(-0.45, 0.515, "chance", fontsize=7, color="#333333")
    home_axis.set_ylim(0.0, 1.18)
    home_axis.set_xticks(positions)
    home_axis.set_xticklabels(ticks, fontsize=8)
    home_axis.set_title("what the arm is worth at home", fontsize=10)
    home_axis.set_ylabel("PTB-XL fold 10 AUROC, infarction against the rest", fontsize=9)

    plain = {a: ARM_NAMES[a].replace("\n", " ") for a in ARM_ORDER}
    saw_source = [plain[a] for a in ARM_ORDER if _saw(arms, a, "ptbxl")]
    worst = max(ARM_ORDER, key=lambda a: _headline_row(arms, a)["coverage_gap"]["acs"]["mean"])
    best = min(ARM_ORDER, key=lambda a: _headline_row(arms, a)["coverage_gap"]["acs"]["mean"])
    figure.suptitle(
        "Coverage gap per encoder arm\n"
        f"Gap to Chongqing runs from {plain[best]} "
        f"{_headline_row(arms, best)['coverage_gap']['acs']['mean']:+.3f} to {plain[worst]} "
        f"{_headline_row(arms, worst)['coverage_gap']['acs']['mean']:+.3f}.\n"
        f"Warm bars saw PTB-XL in pre-training ({', '.join(saw_source)}); grey bars saw no "
        f"public corpus. Gap bars: mean over {arms['n_draws']} calibration draws, whiskers\n"
        "one standard deviation. AUROC whiskers are 95% bootstrap intervals over records; "
        "paired arm-versus-arm differences are in the results file.",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.925))
    return _save(figure, out)


def figure_4_discrimination(metrics: dict[str, Any], reference: dict[str, Any], out: Path) -> Path:
    """The baseline against the published figure it has to reproduce."""
    figure, axis = plt.subplots(figsize=(6.6, 4.6))
    names = ["AUROC", "AUPRC"]
    values = [metrics["auroc"], metrics["auprc"]]
    lows = [metrics["auroc_ci95"][0], metrics["auprc_ci95"][0]]
    highs = [metrics["auroc_ci95"][1], metrics["auprc_ci95"][1]]
    errors = np.array([np.subtract(values, lows), np.subtract(highs, values)])
    positions = np.arange(len(names))
    axis.bar(
        positions,
        values,
        0.4,
        yerr=errors,
        capsize=6,
        color=["#023047", "#8ecae6"],
        edgecolor="white",
    )
    for index, (value, low, high) in enumerate(zip(values, lows, highs, strict=True)):
        axis.text(
            index,
            high + 0.025,
            f"{value:.3f}\n[{low:.3f}, {high:.3f}]",
            ha="center",
            fontsize=9,
        )
    axis.axhline(reference["reference_auroc"], color="#bf4342", linestyle="--")
    axis.text(
        positions[-1] + 0.5,
        reference["reference_auroc"] + 0.012,
        f"published {reference['reference_auroc']:.3f}",
        color="#bf4342",
        ha="right",
        fontsize=8,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(names)
    axis.set_xlim(-0.6, 1.6)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("PTB-XL fold 10, infarction against the rest")
    axis.set_title(
        "Baseline discrimination against the published reference\n"
        f"{metrics['n_test']} tracings, {metrics['n_test_positive']} of them infarction; "
        "whiskers are 95% bootstrap intervals.\n"
        "The published figure averages five diagnostic superclasses, not infarction alone.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1.0))
    return _save(figure, out)


# The rotation's five diagnoses, named as a cardiologist reads them, and the five
# corpora that take turns as the source.
ROTATION_LABEL_NAMES = {
    key: words("en")[f"rotation.label.{key}"] for key in ("NSR", "AF", "LBBB", "RBBB", "IAVB")
}
SOURCE_ORDER = ("ptbxl", "sph", "chapman_ningbo", "georgia", "cpsc")
SOURCE_COLOUR = {
    "ptbxl": "#023047",
    "sph": "#219ebc",
    "chapman_ningbo": "#8ecae6",
    "georgia": "#fb8500",
    "cpsc": "#bf4342",
}
FAMILY_COLOUR = {"recalibrated": "#5f8d4e", "pooled": "#bf4342"}


def _headline(table: dict[str, Any]) -> tuple[float, str]:
    headline = table["settings"]["headline"]
    return float(headline["alpha"]), str(headline["score"])


def figure_7_rotation(table: dict[str, Any], out: Path, lang: str = "en") -> Path:
    """Coverage of each diagnosis under each correction, every source-target pair drawn.

    One panel per correction, one column per diagnosis.  Each away pair is a dot
    coloured by the source its threshold came from, and the source's reading on
    its own held-out records is the hollow marker beside it, so the gap between
    home and away is read down a column rather than across two figures.  The bar
    is the mean over the away pairs with the spread across sources, which is what
    turns two hospitals into an estimate.
    """
    text = words(lang)
    alpha, score = _headline(table)
    target = 1.0 - alpha
    corrections = [c for c in CORRECTION_ORDER if c in table["settings"]["corrections"]]
    labels = [key for key in ROTATION_LABEL_NAMES if key in table["bias"]]
    figure, axes = plt.subplots(
        1, len(corrections), figsize=(4.6 * len(corrections), 5.2), sharey=True
    )
    axes = np.atleast_1d(axes)
    # One offset per (diagnosis, source, target), drawn once and reused in every
    # panel, so the same pair sits at the same place under each correction and
    # can be followed across them.
    rng = np.random.default_rng(0)
    offsets: dict[tuple[str, str, str], float] = {}
    for label in labels:
        for entry in table["bias"][label][corrections[0]]["away"]:
            offsets[(label, entry["source"], entry["target"])] = float(rng.uniform(-0.22, 0.22))

    for index, correction in enumerate(corrections):
        axis = axes[index]
        for column, label in enumerate(labels):
            block = table["bias"][label][correction]
            away = block["away"]
            for entry in away:
                jitter = offsets[(label, entry["source"], entry["target"])]
                # A pair whose per-class threshold ran to infinity in some draws
                # covers by admitting both labels, so its point sits near 1.0 for
                # a reason that is not the scheme working. Ringing it in black is
                # what lets the text say the figure marks them.
                abstains = entry.get("n_draws_threshold_infinite", 0) > 0
                axis.plot(
                    column + jitter,
                    target + entry["bias"],
                    "o",
                    color=SOURCE_COLOUR[entry["source"]],
                    markersize=5 if abstains else 4,
                    alpha=0.75,
                    markeredgecolor="#111111" if abstains else "none",
                    markeredgewidth=1.1 if abstains else 0.0,
                    zorder=3,
                )
            for entry in block["home"]:
                # A home reading can starve on the same label its away readings
                # do, and leaving it unringed made the figure disagree with its
                # own caption's count.
                if entry.get("n_draws_threshold_infinite", 0) > 0:
                    axis.plot(
                        column - 0.34,
                        target + entry["bias"],
                        "o",
                        markerfacecolor="none",
                        markeredgecolor="#111111",
                        markersize=9,
                        markeredgewidth=1.1,
                        zorder=4,
                    )
                axis.plot(
                    column - 0.34,
                    target + entry["bias"],
                    "o",
                    markerfacecolor="none",
                    markeredgecolor=SOURCE_COLOUR[entry["source"]],
                    markersize=6,
                    markeredgewidth=1.2,
                    zorder=4,
                )
            mean = target + float(block["away_bias"]["mean"])
            spread = block["away_bias"]["sd_across_sources"]
            axis.errorbar(
                column,
                mean,
                yerr=0.0 if spread is None else float(spread),
                fmt="_",
                color="#111111",
                markersize=22,
                elinewidth=1.4,
                capsize=5,
                zorder=5,
            )
        axis.axhline(target, color="#111111", linestyle="--", linewidth=1.0, zorder=1)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels([text[f"rotation.label.{key}"] for key in labels], fontsize=8)
        axis.set_title(text[f"correction.{correction}"], fontsize=9)
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="y", color="#eeeeee", zorder=0)
    axes[0].set_ylabel(text["rotation.ylabel"].format(level=percent(target, 0, lang)))

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", color=SOURCE_COLOUR[s], label=text[f"source.{s}"]
        )
        for s in SOURCE_ORDER
        if s in {e["source"] for lab in labels for e in table["bias"][lab][corrections[0]]["away"]}
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor="#111111",
            label=text["rotation.legend.home"],
        )
    )
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="#999999",
            markeredgecolor="#111111",
            markeredgewidth=1.1,
            markersize=6,
            label=text["rotation.legend.infinite"],
        )
    )
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="_",
            linestyle="",
            color="#111111",
            markersize=14,
            label=text["rotation.legend.mean"],
        )
    )
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    # Sinus rhythm is refused on Shandong, so it carries twelve ordered pairs
    # where the other four carry twenty. Stating one number would be wrong for
    # four columns of the figure.
    counted = {table["bias"][label][corrections[0]]["away_bias"]["n_pairs"] for label in labels}
    pairs = (
        text["rotation.pairs.range"].format(low=min(counted), high=max(counted))
        if len(counted) > 1
        else text["rotation.pairs.one"].format(n=min(counted))
    )
    figure.suptitle(
        text["rotation.title"].format(
            pairs=pairs, draws=table["settings"]["n_draws"], score=score.upper()
        ),
        fontsize=10,
    )
    _localise_ticks(figure, lang)
    figure.tight_layout(rect=(0, 0.10, 1, 0.96))
    return _save(figure, out)


def figure_8_target_scale(ladder: dict[str, Any], out: Path, lang: str = "en") -> Path:
    """What a hospital's own labelled tracings buy, against pooling them with the source.

    One panel per correction the ladder can carry above rung zero.  The x axis is
    how many labelled target records the threshold saw; the two lines are the two
    ways of spending them.  Rung zero is the frozen source threshold and is the
    same point on both lines, which is where the break table left off.
    """
    text = words(lang)
    alpha, score = _headline(ladder)
    target = 1.0 - alpha
    rungs = list(ladder["settings"]["rungs"])
    corrections = [
        c
        for c in CORRECTION_ORDER
        if c in {r["correction"] for r in ladder["rows"] if r["n_target_records"] > 0}
    ]
    figure, axes = plt.subplots(
        1, len(corrections), figsize=(4.4 * len(corrections), 4.4), sharey=True
    )
    axes = np.atleast_1d(axes)
    positions = np.arange(len(rungs))

    for index, correction in enumerate(corrections):
        axis = axes[index]
        for family in ("recalibrated", "pooled"):
            means, spreads = [], []
            for rung in rungs:
                row = next(
                    r
                    for r in ladder["rows"]
                    if r["alpha"] == alpha
                    and r["score"] == score
                    and r["correction"] == correction
                    and r["family"] == family
                    and r["n_target_records"] == rung
                )
                means.append(row["coverage_by_class"]["1"]["mean"])
                spreads.append(row["coverage_by_class"]["1"]["sd"])
            means_array = np.asarray(means)
            spreads_array = np.asarray(spreads)
            axis.plot(
                positions,
                means_array,
                "-o",
                color=FAMILY_COLOUR[family],
                markersize=5,
                label=text[f"family.{family}"],
                zorder=3,
            )
            axis.fill_between(
                positions,
                means_array - spreads_array,
                means_array + spreads_array,
                color=FAMILY_COLOUR[family],
                alpha=0.15,
                zorder=2,
            )
        axis.axhline(target, color="#111111", linestyle="--", linewidth=1.0, zorder=1)
        axis.set_xticks(positions)
        axis.set_xticklabels([count(r, lang) for r in rungs])
        axis.set_xlabel(text["target.xlabel"])
        axis.set_title(text[f"correction.{correction}"], fontsize=9)
        axis.grid(axis="y", color="#eeeeee", zorder=0)
    axes[0].set_ylabel(text["rotation.ylabel"].format(level=percent(target, 0, lang)))
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=2, frameon=False, fontsize=8)
    pair = ladder["pair"]

    def plain(corpus: str) -> str:
        return text[f"corpus.{corpus}"].split(" (")[0] if f"corpus.{corpus}" in text else corpus

    figure.suptitle(
        text["target.title"].format(
            source=plain(pair["source"]),
            target=plain(pair["target"]),
            label=text.get(f"target.label.{pair['label']}", pair["label"]),
            draws=ladder["settings"]["n_draws"],
            score=score.upper(),
        ),
        fontsize=10,
    )
    _localise_ticks(figure, lang)
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    return _save(figure, out)


# The figures that carry words in every language of figure_text; the rest are English.
TRANSLATED = (1, 5, 6, 7, 8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", nargs="+", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--lang", default="en", choices=LANGUAGES)
    parser.add_argument("--out", default=str(RESULTS_DIR / "figures"))
    args = parser.parse_args(argv)
    lang = args.lang
    if args.figure is None:
        args.figure = [1, 2, 3, 4, 5, 6, 7, 8] if lang == "en" else list(TRANSLATED)
    untranslated = sorted(set(args.figure) - set(TRANSLATED)) if lang != "en" else []
    if untranslated:
        parser.error(
            f"figures {untranslated} are drawn in English only; --lang {lang} draws "
            f"{list(TRANSLATED)}"
        )
    suffix = SUFFIX[lang]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shift_path = RESULTS_DIR / "shift.json"
    metrics_path = RESULTS_DIR / "baseline/metrics.json"
    baseline_path = RESULTS_DIR / "baseline.json"
    arms_path = RESULTS_DIR / "arms.json"
    rotation_path = RESULTS_DIR / "rotation.json"
    ladder_path = RESULTS_DIR / "target_scale.json"

    drawn = []
    if 1 in args.figure:
        table = json.loads(shift_path.read_text())
        drawn.append(figure_1_coverage(table, out / f"fig3_coverage{suffix}.png", lang))
    if 2 in args.figure:
        table = json.loads(shift_path.read_text())
        drawn.append(figure_2_set_sizes(table, out / "fig2_set_sizes.png"))
    if 3 in args.figure:
        if arms_path.exists():
            drawn.append(figure_3_arms(json.loads(arms_path.read_text()), out / "fig3_arms.png"))
        else:
            print(
                "figure 3 needs each encoder arm's cached representations turned into scores "
                f"and put through the same frozen calibration; {arms_path} does not exist "
                "yet, so it is not drawn",
                file=sys.stderr,
            )
    if 4 in args.figure:
        drawn.append(
            figure_4_discrimination(
                json.loads(metrics_path.read_text()),
                json.loads(baseline_path.read_text()),
                out / "fig4_discrimination.png",
            )
        )
    outcomes_path = RESULTS_DIR / "outcomes.json"
    if 5 in args.figure or 6 in args.figure:
        outcomes = json.loads(outcomes_path.read_text())
        if 5 in args.figure:
            scores = dict(np.load(RESULTS_DIR / "baseline/scores.npz"))
            drawn.append(
                figure_5_thresholds(outcomes, scores, out / f"fig1_thresholds{suffix}.png", lang)
            )
        if 6 in args.figure:
            drawn.append(figure_6_outcomes(outcomes, out / f"fig2_outcomes{suffix}.png", lang))
    if 7 in args.figure:
        if rotation_path.exists():
            drawn.append(
                figure_7_rotation(
                    json.loads(rotation_path.read_text()),
                    out / f"fig7_rotation{suffix}.png",
                    lang,
                )
            )
        else:
            print(
                "figure 7 needs every corpus scored by every source's model and the "
                f"coverage table built from those scores; {rotation_path} does not exist "
                "yet, so it is not drawn",
                file=sys.stderr,
            )
    if 8 in args.figure:
        if ladder_path.exists():
            drawn.append(
                figure_8_target_scale(
                    json.loads(ladder_path.read_text()),
                    out / f"fig8_target_scale{suffix}.png",
                    lang,
                )
            )
        else:
            print(
                f"figure 8 needs {ladder_path}, which does not exist yet, so it is not drawn",
                file=sys.stderr,
            )
    for path in drawn:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
