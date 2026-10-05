"""Draw the three figures of REPORT.md from ``results/echonext_clinical.json``.

``fig_curve.png``: for the trained network, the share of the ill caught against
the share of the healthy flagged, inpatients and outpatients, with the point the
inpatient threshold lands on in each setting.

``fig_patients.png``: what the inpatient threshold does to 100 ill and 100
healthy outpatients, for each model, alone and with the second threshold.

``fig_repair.png``: the threshold refitted on labelled outpatients, sensitivity
and specificity against the number labelled, with the spread over draws.

Usage: uv run python scripts/paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ecs.config import RESULTS_DIR  # noqa: E402

FIGURES = RESULTS_DIR / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#eeeeee"
# Fixed slots: a setting or an outcome keeps its colour in every figure, and
# each also carries a dash, a marker or a printed label.
INPATIENT, OUTPATIENT = "#2a78d6", "#eb6834"
RIGHT, DEFERRED, WRONG = "#2a78d6", "#c9c7c1", "#eb6834"
ARM_NAMES = {
    "resnet": "Network trained here",
    "echonext_mini": "EchoNext mini-model",
    "ecgfounder": "ECGFounder",
    "random_init": "Untrained floor",
}


def read() -> dict[str, Any]:
    result: dict[str, Any] = json.loads((RESULTS_DIR / "echonext_clinical.json").read_text())
    return result


def style(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def curve(result: dict[str, Any], path: Path) -> None:
    roc = result["roc"]["resnet"]
    arm = result["arms"]["resnet"]
    fig, axis = plt.subplots(figsize=(5.2, 4.4), dpi=200)
    style(axis)
    for context, colour, dash, name in (
        ("inpatient", INPATIENT, "-", "Inpatients"),
        ("outpatient", OUTPATIENT, "--", "Outpatients"),
    ):
        flagged = [100 * (1 - s) for s in roc[context]]
        axis.plot(
            flagged, [100 * s for s in roc["sensitivity"]], dash, color=colour, lw=2, label=name
        )
        x, y = 100 * (1 - arm[context]["specificity"]), 100 * arm[context]["sensitivity"]
        axis.plot(x, y, "o", ms=9, color=colour, markeredgecolor="white", markeredgewidth=2)
        axis.annotate(
            f"{y:.0f} of 100 ill caught\n{x:.0f} of 100 healthy flagged",
            (x, y),
            xytext=(10, -30) if context == "outpatient" else (-150, 6),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    axis.axhline(90, color=MUTED, lw=1, ls=":")
    axis.text(101, 90, "90", fontsize=7, color=MUTED, va="center")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 102)
    axis.set_xlabel("Healthy patients flagged, per 100", fontsize=9, color=INK)
    axis.set_ylabel("Ill patients caught, per 100", fontsize=9, color=INK)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def patients(result: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=200, sharey=True)
    rows = [(arm, method) for arm in ARM_NAMES for method in ("plain", "perlabel")]
    labels = [
        f"{ARM_NAMES[arm]}, {'one threshold' if method == 'plain' else 'two thresholds'}"
        for arm, method in rows
    ]
    for axis, group, names in (
        (axes[0], "ill", ("caught", "sent to a reader", "missed")),
        (axes[1], "healthy", ("cleared", "sent to a reader", "flagged")),
    ):
        style(axis)
        axis.grid(False)
        for i, (arm, method) in enumerate(rows):
            shares = result["arms"][arm]["outpatient"]["outcomes"][method][group]
            left = 0.0
            for key, colour in (
                ("right_alone", RIGHT),
                ("deferred", DEFERRED),
                ("wrong_alone", WRONG),
            ):
                width = 100 * shares[key]["share"]
                axis.barh(
                    i, width, left=left, color=colour, edgecolor="white", linewidth=2, height=0.75
                )
                if width >= 6:
                    ink = "white" if colour != DEFERRED else INK
                    axis.text(
                        left + width / 2,
                        i,
                        f"{width:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=ink,
                    )
                left += width
        axis.set_xlim(0, 100)
        axis.set_title(f"Per 100 {group} outpatients", fontsize=9, color=INK, loc="left")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (RIGHT, DEFERRED, WRONG)]
        axis.legend(
            handles,
            names,
            frameon=False,
            fontsize=7,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
        )
        axis.set_xticks([])
        axis.spines[["bottom"]].set_visible(False)
    axes[0].set_yticks(range(len(rows)), labels, fontsize=7)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def repair(result: dict[str, Any], path: Path) -> None:
    ladder = result["ladder"]["resnet"]
    fig, axis = plt.subplots(figsize=(5.6, 4.0), dpi=200)
    style(axis)
    x = [row["labels"] for row in ladder]
    for key, colour, dash, name in (
        ("sensitivity", OUTPATIENT, "-", "Ill caught"),
        ("specificity", INPATIENT, "--", "Healthy cleared"),
    ):
        mean = [100 * row[key]["mean"] for row in ladder]
        low = [100 * row[key]["p10"] for row in ladder]
        high = [100 * row[key]["p90"] for row in ladder]
        axis.fill_between(x, low, high, color=colour, alpha=0.15, linewidth=0)
        axis.plot(x, mean, dash, color=colour, lw=2, marker="o", ms=5, label=name)
    axis.axhline(90, color=MUTED, lw=1, ls=":")
    ticks = [
        f"{row['labels']}"
        + (f"\n({row['ill_in_sample']['mean']:.0f} ill)" if row["labels"] else "\n(none)")
        for row in ladder
    ]
    axis.set_xticks(x, ticks, fontsize=7)
    axis.set_ylim(0, 102)
    axis.set_xlabel("Labelled outpatients the threshold is refitted on", fontsize=9, color=INK)
    axis.set_ylabel("Per 100 outpatients of each group", fontsize=9, color=INK)
    axis.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    result = read()
    out = FIGURES
    for draw, path in (
        (curve, out / "fig_curve.png"),
        (patients, out / "fig_patients.png"),
        (repair, out / "fig_repair.png"),
    ):
        draw(result, path)
        print(path.relative_to(RESULTS_DIR.parent))


if __name__ == "__main__":
    main()
