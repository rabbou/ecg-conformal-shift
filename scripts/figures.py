"""The figures the plan fixed before any result existed.

Each one is drawn from a file under ``results/`` that was committed before it,
so a figure can always be regenerated from the numbers rather than redrawn from
memory, and a surprising result cannot be quietly re-cut.

1. Coverage against target -- does the guarantee hold at each hospital, and does
   it hold for the sick as well as the healthy.
2. Set-size distribution -- how often the model answers with one label, both, or
   neither, at each hospital.
4. Baseline discrimination -- the reproduction of known ground that licenses
   everything else.

Figures 1 and 2 read ``results/shift.json``, where one PTB-XL threshold was
spent on all three corpora at once, so the panels differ in nothing but the
population they describe.

Figure 3 compares the encoder arms on that same break; it needs each arm's
cached representations turned into scores, which happens later in the week, and
this script says so rather than drawing an empty frame.

Usage: .venv/bin/python scripts/figures.py [--figure 1 2 4]
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


def figure_1_coverage(table: dict[str, Any], out: Path, source: Path) -> Path:
    """Coverage against the level asked for, at each hospital and inside each class.

    One row per corpus, all under the same PTB-XL threshold: the first row is the
    population the threshold was fitted on, the two below it are populations it
    was merely spent on.
    """
    corpora = corpora_on(table)
    corrections = ["none", "mondrian"]
    labels = {"none": "one threshold for both classes", "mondrian": "one threshold per class"}
    colours = {"none": "#219ebc", "mondrian": "#fb8500"}
    groups = ["overall", "0", "1"]
    titles = {"overall": "every tracing", "0": CLASS_NAMES["0"], "1": CLASS_NAMES["1"]}
    figure, axes = plt.subplots(
        len(corpora), 3, figsize=(12.5, 3.9 * len(corpora)), sharey=True, sharex=True, squeeze=False
    )

    for row_index, corpus in enumerate(corpora):
        for axis, group in zip(axes[row_index], groups, strict=True):
            width = 0.35
            for offset, correction in zip((-width / 2, width / 2), corrections, strict=True):
                rows = _rows(table, "lac", correction)
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
                    label=labels[correction],
                    color=colours[correction],
                    edgecolor="white",
                )
            for index, row in enumerate(_rows(table, "lac", "none")):
                axis.hlines(
                    row["target_coverage"],
                    index - 0.5,
                    index + 0.5,
                    colors="#333333",
                    linestyles="--",
                )
            axis.set_xticks(range(len(LEVELS)))
            axis.set_xticklabels([f"{1 - a:.0%}" for a in LEVELS])
            axis.set_ylim(0.0, 1.05)
            if row_index == 0:
                axis.set_title(titles[group])
            if row_index == len(corpora) - 1:
                axis.set_xlabel("confidence asked for")
        axes[row_index][0].set_ylabel(
            f"{_panel_title(table, corpus)}\nshare whose set holds the true label", fontsize=8
        )
    axes[0][2].legend(loc="lower right", fontsize=8, framealpha=0.95)

    sick = {
        corpus: _rows(table, "lac", "none")[1]["by_corpus"][corpus]["coverage_by_class"]["1"][
            "mean"
        ]
        for corpus in corpora
    }
    reading = " · ".join(f"{CORPUS_NAMES[c].split(' (')[0]} {v:.0%}" for c, v in sick.items())
    figure.suptitle(
        "Figure 1 — one PTB-XL threshold, spent on three hospitals\n"
        f"Share of infarctions inside the 90% set, one shared threshold: {reading}.\n"
        f"Dashed line: the level asked for. Bars: mean over {table['n_draws']} "
        "calibration draws on PTB-XL, whiskers one standard deviation.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.93))
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
        "Figure 2 — what the model returns as the confidence asked for rises\n"
        "One threshold, fitted on PTB-XL and spent on all three. “both” and “no label” are the "
        "same thing in a clinic: the tracing goes to a human.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.94))
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
        "Figure 4 — the baseline reproduces known ground\n"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", nargs="+", type=int, default=[1, 2, 4], choices=[1, 2, 3, 4])
    parser.add_argument("--out", default=str(RESULTS_DIR / "figures"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shift_path = RESULTS_DIR / "shift.json"
    metrics_path = RESULTS_DIR / "baseline/metrics.json"
    baseline_path = RESULTS_DIR / "baseline.json"

    drawn = []
    if 1 in args.figure:
        table = json.loads(shift_path.read_text())
        drawn.append(figure_1_coverage(table, out / "fig1_coverage.png", shift_path))
    if 2 in args.figure:
        table = json.loads(shift_path.read_text())
        drawn.append(figure_2_set_sizes(table, out / "fig2_set_sizes.png", shift_path))
    if 3 in args.figure:
        print(
            "figure 3 needs each encoder arm's cached representations turned into scores "
            "and put through the same frozen calibration; that is later in the week, so it "
            "is not drawn",
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
    for path in drawn:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
