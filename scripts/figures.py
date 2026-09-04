"""Draw the four figures listed in PLAN.md.

Each one is drawn from a file under ``results/`` that was committed before it,
so a figure can always be regenerated from the numbers rather than redrawn from
memory, and a surprising result cannot be quietly re-cut.

1. Coverage against target -- does the guarantee hold at each hospital, and does
   it hold for the sick as well as the healthy.
2. Set-size distribution -- how often the model answers with one label, both, or
   neither, at each hospital.
3. Encoder arms on the same break -- does the encoder underneath change how far
   the guarantee falls, and does an encoder that saw the calibration corpus look
   better at home for a reason other than being better.
4. Baseline discrimination -- AUROC and AUPRC against the published reference
   value.

Figures 1 and 2 read ``results/shift.json``, where one PTB-XL threshold was
spent on all three corpora at once, so the panels differ in nothing but the
population they describe.

Figure 3 reads ``results/arms.json``, where each arm's cached representations
went through the same linear probe, the same folds and the same frozen
calibration, so a difference between two arms is a difference between two
pre-trainings.  Until that file exists this script says what it is waiting for
rather than drawing an empty frame.

Usage: .venv/bin/python scripts/figures.py [--figure 1 2 3 4 5 6]
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

from ecs.config import RESULTS_DIR  # noqa: E402

# Levels loosest first, so every figure reads left to right as confidence rising.
LEVELS = (0.20, 0.10, 0.05)
CLASS_NAMES = {"0": "no infarction", "1": "infarction"}
# Source first, then the two corpora the threshold is spent on unchanged.
CORPUS_ORDER = ("ptbxl", "sph", "acs")
CORPUS_NAMES = {
    "ptbxl": "PTB-XL (calibrated here)",
    "sph": "Shandong",
    "acs": "Chongqing",
}
# The arms in contamination order: the control, then the arms that saw no public
# corpus, then the one that saw the calibration corpus, then the one that saw a
# target too.  Reading the figure left to right is reading that order.
ARM_ORDER = ("random_init", "ecgfounder", "ecgfm", "hubert_ecg")
ARM_NAMES = {
    "random_init": "random init,\nfrozen",
    "ecgfounder": "ECGFounder",
    "ecgfm": "ECG-FM",
    "hubert_ecg": "HuBERT-ECG",
}
# Grey for the arms that saw neither corpus, warm for the ones that saw PTB-XL,
# hottest for the one that saw a target as well.
ARM_COLOUR = {
    "random_init": "#adb5bd",
    "ecgfounder": "#6c757d",
    "ecgfm": "#fb8500",
    "hubert_ecg": "#bf4342",
}

# One hue per level, dark enough to survive greyscale printing.
LEVEL_COLOUR = {0.20: "#8ecae6", 0.10: "#219ebc", 0.05: "#023047"}
SET_COLOUR = {"empty_rate": "#bf4342", "one_label_rate": "#5f8d4e", "two_label_rate": "#e9c46a"}
SET_NAME = {"empty_rate": "no label", "one_label_rate": "one label", "two_label_rate": "both"}


def _source_note(figure: plt.Figure, path: Path) -> None:
    figure.text(
        0.01,
        0.01,
        f"drawn by scripts/figures.py from {path.relative_to(RESULTS_DIR.parent)}",
        fontsize=7,
        color="#666666",
    )


def _rows(table: dict[str, Any], score: str, correction: str) -> list[dict[str, Any]]:
    chosen = [r for r in table["rows"] if r["score"] == score and r["correction"] == correction]
    return sorted(chosen, key=lambda r: -float(r["alpha"]))


def corpora_on(table: dict[str, Any]) -> list[str]:
    """The corpora on the table, source first."""
    present = set(table["rows"][0]["by_corpus"])
    return [name for name in CORPUS_ORDER if name in present]


def _panel_title(table: dict[str, Any], corpus: str) -> str:
    block = table["rows"][0]["by_corpus"][corpus]
    return (
        f"{CORPUS_NAMES[corpus]}\n"
        f"{block['n_points']:,} tracings, {block['prevalence']:.1%} infarction"
    )


CORRECTION_ORDER = ("none", "mondrian", "weighted")
CORRECTION_NAMES = {
    "none": "no correction\none threshold for both classes",
    "mondrian": "Mondrian\none threshold per class, exact",
    "weighted": "label-shift weighted\ntarget prior estimated",
}
# The three readings inside a panel: the marginal figure, then the class it is
# bought from and the class it is bought for.
GROUP_ORDER = ("overall", "0", "1")
GROUP_NAMES = {"overall": "every tracing", "0": CLASS_NAMES["0"], "1": CLASS_NAMES["1"]}
GROUP_COLOUR = {"overall": "#adb5bd", "0": "#219ebc", "1": "#bf4342"}


def corrections_on(table: dict[str, Any]) -> list[str]:
    """The corrections the table holds, in the order the argument runs."""
    present = {row["correction"] for row in table["rows"]}
    return [name for name in CORRECTION_ORDER if name in present]


def figure_1_coverage(table: dict[str, Any], out: Path, source: Path) -> Path:
    """Coverage against the level asked for, at each hospital and under each correction.

    One row per corpus, one column per correction, all from the same PTB-XL
    calibration: the first row is the population the threshold was fitted on, the
    two below it are populations it was merely spent on.  Reading a row left to
    right is reading what each repair does to that hospital; reading the red bar
    down a column is reading whether the sick are covered at all.
    """
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
                    label=GROUP_NAMES[group],
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
            axis.set_xticklabels([f"{1 - float(r['alpha']):.0%}" for r in rows])
            axis.set_ylim(0.0, 1.05)
            if row_index == 0:
                axis.set_title(CORRECTION_NAMES[correction], fontsize=9)
            if row_index == len(corpora) - 1:
                axis.set_xlabel("confidence asked for")
        axes[row_index][0].set_ylabel(
            f"{_panel_title(table, corpus)}\nshare whose set holds the true label", fontsize=8
        )
    axes[0][-1].legend(loc="lower right", fontsize=8, framealpha=0.95)

    headline = _rows(table, "lac", "none")[1]["by_corpus"]
    quoted = " · ".join(
        f"{CORRECTION_NAMES[c].splitlines()[0]} "
        f"{_rows(table, 'lac', c)[1]['by_corpus']['ptbxl']['coverage_by_class']['1']['mean']:.0%}"
        for c in corrections
    )
    estimated = "; ".join(
        f"{CORPUS_NAMES[corpus].split(' (')[0]} "
        f"{block['calibration']['estimated_prevalence']['mean']:.1%} estimated for "
        f"{headline[corpus]['prevalence']:.1%} true"
        for corpus, block in _rows(table, "lac", "weighted")[1]["by_corpus"].items()
    )
    figure.suptitle(
        "Figure 1 — coverage per hospital and correction\n"
        f"Share of infarctions inside the 90% set on PTB-XL: {quoted}.\n"
        f"Dashed line: the requested level. Bars: mean over {table['n_draws']} calibration "
        "draws on PTB-XL, whiskers one standard deviation.\n"
        f"The weighted thresholds use each corpus's estimated class mix: {estimated}.",
        fontsize=9,
        y=0.995,
        va="top",
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.945))
    _source_note(figure, source)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


def figure_2_set_sizes(table: dict[str, Any], out: Path, source: Path) -> Path:
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
        "Figure 2 — output composition against requested confidence\n"
        "One threshold, fitted on PTB-XL and applied to all three corpora. Tracings with "
        "both labels or no label go to a human reader.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.94))
    _source_note(figure, source)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


def _headline_row(arms: dict[str, Any], arm: str) -> dict[str, Any]:
    """The arm's coverage row at the level the grid quotes its headline at."""
    wanted = arms["headline"]
    for row in arms["coverage"][arm]:
        if all(row[key] == value for key, value in wanted.items()):
            return row
    raise KeyError(f"{arm} has no row for {wanted}")


def _saw(arms: dict[str, Any], arm: str, corpus_name: str) -> bool:
    """Whether this arm's own sidecar names ``corpus_name`` among its pre-training."""
    return corpus_name.lower() in arms["arms"][arm]["pretraining_corpora"].lower()


def figure_3_arms(arms: dict[str, Any], out: Path, source: Path) -> Path:
    """The four encoder arms on the same break.

    One panel per target: how far each arm's coverage falls between PTB-XL and
    that hospital, as a signed gap, so a bar above zero is a guarantee that
    stopped holding and a bar below zero is one that over-covered.  The last
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
    saw_source = [plain[a] for a in ARM_ORDER if _saw(arms, a, "PTB-XL")]
    worst = max(ARM_ORDER, key=lambda a: _headline_row(arms, a)["coverage_gap"]["acs"]["mean"])
    best = min(ARM_ORDER, key=lambda a: _headline_row(arms, a)["coverage_gap"]["acs"]["mean"])
    figure.suptitle(
        "Figure 3 — coverage gap per encoder arm\n"
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
    _source_note(figure, source)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


def figure_4_discrimination(
    metrics: dict[str, Any], reference: dict[str, Any], out: Path, source: Path
) -> Path:
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
        "Figure 4 — baseline discrimination against the published reference\n"
        f"{metrics['n_test']} tracings, {metrics['n_test_positive']} of them infarction; "
        "whiskers are 95% bootstrap intervals.\n"
        "The published figure averages five diagnostic superclasses, not infarction alone.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1.0))
    _source_note(figure, source)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


# The three schemes of the outcome table, in the order the report reads them.
SCHEME_ORDER = ("plain", "pooled", "perlabel")
SCHEME_NAMES = {
    "plain": "Single tuned threshold\n90% sensitivity, no deferral",
    "pooled": "Pooled conformal calibration\n90% coverage, all cases",
    "perlabel": "Class-conditional calibration\n90% coverage within each label",
}
# Correct, deferred, wrong -- the three things that can happen to a case.
OUTCOME_COLOURS = {"correct": "#4a7c59", "deferred": "#e0a458", "wrong": "#b4423a"}


def figure_5_thresholds(
    outcomes: dict[str, Any], scores: dict[str, Any], out: Path, source: Path
) -> Path:
    """Where each scheme puts its thresholds, and what that costs per label.

    The score axis is the same in all three panels; only the thresholds move.
    Each panel carries the consequence beside the placement, so the figure does
    not need the table to be read.
    """
    probs = scores["probs"][:, 1]
    labels = scores["labels"]
    source_block = outcomes["by_corpus"]["ptbxl"]
    prevalence = source_block["prevalence"]
    thresholds = outcomes["thresholds"]
    bands: dict[str, tuple[float, ...]] = {
        "plain": (thresholds["plain:single"]["mean"],),
        "pooled": (thresholds["pooled:lower"]["mean"], thresholds["pooled:upper"]["mean"]),
        "perlabel": (thresholds["perlabel:lower"]["mean"], thresholds["perlabel:upper"]["mean"]),
    }
    bins = np.linspace(0.0, 1.0, 41)
    figure, axes = plt.subplots(3, 1, figsize=(8.8, 8.4), sharex=True)
    for axis, scheme in zip(axes, SCHEME_ORDER, strict=True):
        for klass, colour, name in ((0, "#7d8a93", "non-MI"), (1, "#b4423a", "MI")):
            of_class = labels == klass
            axis.hist(
                probs[of_class],
                bins=bins,
                weights=np.full(int(of_class.sum()), 100.0 / int(of_class.sum())),
                color=colour,
                alpha=0.7,
                label=f"{name} (n={int(of_class.sum()):,})",
            )
        cuts = bands[scheme]
        low, high = cuts[0], cuts[-1]
        if high > low:
            axis.axvspan(low, high, color="#e0a458", alpha=0.30, zorder=0)
            axis.text((low + high) / 2, 52, "deferred", ha="center", fontsize=9.5, color="#8a5b1c")
        for cut in dict.fromkeys(cuts):
            axis.axvline(cut, color="black", lw=1.2, ls="--")
            axis.text(cut + 0.009, 44, f"{cut:.2f}", fontsize=8.5, color="#333", va="top")
        axis.text(low / 2, 52, "labelled non-MI", ha="center", fontsize=9.5, color="#3d474d")
        axis.text((high + 1) / 2, 52, "labelled MI", ha="center", fontsize=9.5, color="#7a2f2a")
        sick = source_block["schemes"][scheme]["1"]
        healthy = source_block["schemes"][scheme]["0"]
        deferred = (
            prevalence * sick["deferred"]["mean"] + (1 - prevalence) * healthy["deferred"]["mean"]
        )
        axis.text(
            0.985,
            0.72,
            f"MI missed  {sick['wrong']['mean'] * 100:.0f}%\n"
            f"false alarms  {healthy['wrong']['mean'] * 100:.0f}%\n"
            f"deferred  {deferred * 100:.0f}%",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            linespacing=1.55,
            bbox={"boxstyle": "round,pad=0.5", "fc": "#f6f4ef", "ec": "#cfc9bd", "lw": 0.8},
        )
        axis.set_ylim(0, 58)
        axis.set_title(SCHEME_NAMES[scheme].replace("\n", " -- "), fontsize=11.5, loc="left", pad=6)
        axis.set_ylabel("% of that class", fontsize=9.5)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(0.30, 1.02))
    axes[-1].set_xlabel("model score for MI", fontsize=10)
    axes[-1].set_xlim(0, 1)
    figure.text(
        0.5,
        0.055,
        "MI missed: share of MI cases given the non-MI label alone.   "
        "False alarms: share of non-MI cases given the MI label alone.",
        ha="center",
        fontsize=8.5,
        color="#555",
    )
    figure.text(
        0.5,
        0.034,
        "Deferred: share of all tracings returning both labels or neither, sent to a specialist.",
        ha="center",
        fontsize=8.5,
        color="#555",
    )
    figure.tight_layout(rect=(0, 0.072, 1, 1))
    _source_note(figure, source)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


def figure_6_outcomes(outcomes: dict[str, Any], out: Path, source: Path) -> Path:
    """What a case of each label gets, under each of the three schemes."""
    source_block = outcomes["by_corpus"]["ptbxl"]
    panels = (
        ("1", f"{int(round(source_block['n_points'] * source_block['prevalence'])):,} MI cases"),
        (
            "0",
            f"{int(round(source_block['n_points'] * (1 - source_block['prevalence']))):,} "
            "non-MI cases",
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharex=True)
    positions = np.arange(len(SCHEME_ORDER))[::-1]
    for axis, (klass, title) in zip(axes, panels, strict=True):
        for position, scheme in zip(positions, SCHEME_ORDER, strict=True):
            cell = source_block["schemes"][scheme][klass]
            left = 0.0
            for name in ("correct", "deferred", "wrong"):
                width = cell[name]["mean"] * 100
                axis.barh(position, width, left=left, color=OUTCOME_COLOURS[name], height=0.5)
                if width > 5:
                    axis.text(
                        left + width / 2,
                        position,
                        f"{width:.0f}%",
                        ha="center",
                        va="center",
                        color="#3a2c10" if name == "deferred" else "white",
                        fontsize=10.5,
                    )
                left += width
        axis.set_yticks(positions)
        axis.set_yticklabels([SCHEME_NAMES[s] for s in SCHEME_ORDER], fontsize=9)
        axis.set_xlim(0, 100)
        axis.set_title(title, fontsize=11.5, pad=10)
        axis.set_xlabel("share of cases carrying this label (%)", fontsize=9.5)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis="y", length=0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLOURS[name])
        for name in ("correct", "deferred", "wrong")
    ]
    figure.legend(
        handles,
        ["correct label returned", "deferred to a specialist", "incorrect label returned"],
        fontsize=10,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=False,
    )
    figure.text(
        0.5,
        0.012,
        f"PTB-XL fold 10, {outcomes['n_draws']} patient-level calibration draws. "
        "The three schemes sit at a comparable MI miss rate by construction.",
        ha="center",
        fontsize=8.5,
        color="#555",
    )
    figure.tight_layout(rect=(0, 0.15, 1, 0.97))
    _source_note(figure, source)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6],
        choices=[1, 2, 3, 4, 5, 6],
    )
    parser.add_argument("--out", default=str(RESULTS_DIR / "figures"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shift_path = RESULTS_DIR / "shift.json"
    metrics_path = RESULTS_DIR / "baseline/metrics.json"
    baseline_path = RESULTS_DIR / "baseline.json"
    arms_path = RESULTS_DIR / "arms.json"

    drawn = []
    if 1 in args.figure:
        table = json.loads(shift_path.read_text())
        drawn.append(figure_1_coverage(table, out / "fig1_coverage.png", shift_path))
    if 2 in args.figure:
        table = json.loads(shift_path.read_text())
        drawn.append(figure_2_set_sizes(table, out / "fig2_set_sizes.png", shift_path))
    if 3 in args.figure:
        if arms_path.exists():
            drawn.append(
                figure_3_arms(json.loads(arms_path.read_text()), out / "fig3_arms.png", arms_path)
            )
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
                metrics_path,
            )
        )
    outcomes_path = RESULTS_DIR / "outcomes.json"
    if 5 in args.figure or 6 in args.figure:
        outcomes = json.loads(outcomes_path.read_text())
        if 5 in args.figure:
            scores = dict(np.load(RESULTS_DIR / "baseline/scores.npz"))
            drawn.append(
                figure_5_thresholds(outcomes, scores, out / "fig1_thresholds.png", outcomes_path)
            )
        if 6 in args.figure:
            drawn.append(figure_6_outcomes(outcomes, out / "fig2_outcomes.png", outcomes_path))
    for path in drawn:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
