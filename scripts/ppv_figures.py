"""Draw the two figures of the positive predictive value study, in English and French.

``ppv_gap.png``: the PPV recomputed by Bayes' rule against the PPV observed,
per cell, and the gap against the change in specificity, from
``ppv_gap.json`` and ``echonext_ppv_gap.json``.

``decision_curves.png``: the net benefit of the model and of the three repairs
against treating everyone and no one, at three sites, from ``repairs.json``.

Without ``echonext_ppv_gap.json`` the first figure is drawn from the public
families alone, and says so in its title.

Usage: .venv/bin/python scripts/ppv_figures.py [--lang en|fr]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from figure_text import LANGUAGES, SUFFIX, number, percent  # noqa: E402

from ecs.config import RESULTS_DIR  # noqa: E402

FIGURES = RESULTS_DIR / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#eeeeee"
# Categorical slots in fixed order; each also carries a marker or a dash, so no
# identity rests on colour alone.
FAMILY_STYLE = {
    "echonext": ("#2a78d6", "o"),
    "rotation": ("#eb6834", "s"),
    "infarction": ("#1baf7a", "D"),
}
CONTROL_STYLE = ("#a3a29d", "o")
RULE_STYLE = {
    "as_delivered": ("#2a78d6", "-"),
    "prior": ("#eb6834", (0, (5, 2))),
    "recalibrated": ("#1baf7a", "-"),
    "abstention_referred": ("#eda100", (0, (1, 1.5))),
}
PANELS = (
    ("echonext", "resnet", "outpatient", "shd_moderate_or_greater_flag"),
    ("infarction", "ptbxl_baseline", "sph", "MI"),
    ("infarction", "ptbxl_baseline", "acs", "MI"),
)

WORDS: dict[str, dict[str, str]] = {
    "en": {
        "gap.title": "The PPV a buyer recomputes against the PPV observed",
        "gap.public_only": " (public corpora only)",
        "gap.x": "PPV recomputed from the source's sensitivity and specificity",
        "gap.y": "PPV observed at the target",
        "gap.band": "within two points",
        "gap.x2": "Specificity at the target minus at the source",
        "gap.y2": "Recomputed minus observed PPV, points",
        "family.control": "same population (control)",
        "family.echonext": "Columbia, inpatients to emergency and outpatients",
        "family.rotation": "five corpora, each to the other four",
        "family.infarction": "PTB-XL to Shandong and Chongqing",
        "dc.title": "Net benefit of acting on the model, by repair",
        "dc.x": "Probability of disease at which one acts",
        "dc.y": "Net benefit, true positives per 100 patients",
        "dc.named": "10%: send to echocardiography",
        "dc.named_short": "10%",
        "panel.echonext": "Columbia outpatients\nstructural heart disease",
        "panel.sph": "Shandong\ninfarction",
        "panel.acs": "Chongqing\ninfarction",
        "rule.as_delivered": "model as delivered",
        "rule.prior": "prevalence corrected, no label",
        "rule.recalibrated": "recalibrated on 100 local labels",
        "rule.abstention_referred": "per-label sets, abstentions referred",
        "rule.treat_all": "treat everyone",
        "rule.treat_none": "treat no one",
    },
    "fr": {
        "gap.title": "La VPP recalculée par l'acheteur contre la VPP observée",
        "gap.public_only": " (corpus publics seuls)",
        "gap.x": "VPP recalculée à partir de la sensibilité et de la spécificité de la source",
        "gap.y": "VPP observée sur la cible",
        "gap.band": "à deux points près",
        "gap.x2": "Spécificité sur la cible moins sur la source",
        "gap.y2": "VPP recalculée moins observée, en points",
        "family.control": "même population (témoin)",
        "family.echonext": "Columbia, des hospitalisés vers les urgences et les consultations",
        "family.rotation": "cinq corpus, chacun vers les quatre autres",
        "family.infarction": "PTB-XL vers Shandong et Chongqing",
        "dc.title": "Bénéfice net à agir sur le modèle, par réparation",
        "dc.x": "Probabilité de maladie à partir de laquelle on agit",
        "dc.y": "Bénéfice net, vrais positifs pour 100 patients",
        "dc.named": "10 % : envoyer en échographie",
        "dc.named_short": "10 %",
        "panel.echonext": "Consultations de Columbia\ncardiopathie structurelle",
        "panel.sph": "Shandong\ninfarctus",
        "panel.acs": "Chongqing\ninfarctus",
        "rule.as_delivered": "modèle tel que livré",
        "rule.prior": "prévalence ajustée, sans étiquette",
        "rule.recalibrated": "recalibré sur 100 étiquettes locales",
        "rule.abstention_referred": "ensembles par étiquette, abstentions envoyées",
        "rule.treat_all": "traiter tout le monde",
        "rule.treat_none": "ne traiter personne",
    },
}


def _gap_rows() -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = json.loads((RESULTS_DIR / "ppv_gap.json").read_text())["rows"]
    private = RESULTS_DIR / "echonext_ppv_gap.json"
    if private.exists():
        rows = rows + json.loads(private.read_text())["rows"]
    return [r for r in rows if r["summarised"]], private.exists()


def _style(row: dict[str, Any]) -> tuple[str, str]:
    return CONTROL_STYLE if row["in_distribution"] else FAMILY_STYLE[row["family"]]


def figure_gap(out: Path, lang: str) -> Path:
    text = WORDS[lang]
    rows, complete = _gap_rows()
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 5.2))
    left.fill_between([0, 1], [-0.02, 0.98], [0.02, 1.02], color=GRID, zorder=0)
    left.plot([0, 1], [0, 1], color=MUTED, linewidth=1, zorder=1)
    left.text(0.62, 0.53, text["gap.band"], color=MUTED, fontsize=8, rotation=40)
    right.axhline(0.0, color=MUTED, linewidth=1, zorder=1)
    right.axvline(0.0, color=MUTED, linewidth=1, zorder=1)
    order = sorted(rows, key=lambda r: not r["in_distribution"])
    for row in order:
        colour, marker = _style(row)
        kwargs = {
            "color": colour,
            "marker": marker,
            "s": 22,
            "alpha": 0.85,
            "edgecolors": "white",
            "linewidths": 0.6,
            "zorder": 2,
        }
        left.scatter(row["ppv_recomputed"], row["ppv_observed"], **kwargs)
        right.scatter(row["spec_target"] - row["spec_source"], 100 * row["gap"], **kwargs)
    left.set(xlim=(0, 1), ylim=(0, 1), xlabel=text["gap.x"], ylabel=text["gap.y"])
    right.set(xlabel=text["gap.x2"], ylabel=text["gap.y2"])
    for axis in (left, right):
        axis.grid(color=GRID, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    left.xaxis.set_major_formatter(lambda v, _: percent(v, 0, lang))
    left.yaxis.set_major_formatter(lambda v, _: percent(v, 0, lang))
    right.xaxis.set_major_formatter(lambda v, _: number(v, 1, lang))
    families = [k for k in FAMILY_STYLE if any(r["family"] == k for r in rows)]
    handles = [
        plt.Line2D([], [], color=c, marker=m, linestyle="", label=text[f"family.{k}"])
        for k, (c, m) in [("control", CONTROL_STYLE)] + [(f, FAMILY_STYLE[f]) for f in families]
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    title = text["gap.title"] + ("" if complete else text["gap.public_only"])
    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0.1, 1, 0.96))
    figure.savefig(out, dpi=150)
    plt.close(figure)
    return out


def _cell(cells: list[dict[str, Any]], family: str, model: str, target: str, label: str) -> dict:
    for c in cells:
        if (c["family"], c["model"], c["target"], c["label"]) == (family, model, target, label):
            return c
    raise KeyError((family, model, target, label))


def figure_decision(out: Path, lang: str) -> Path:
    text = WORDS[lang]
    result = json.loads((RESULTS_DIR / "repairs.json").read_text())
    panels = [p for p in PANELS if p[0] in result["families"]]
    figure, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    named = result["named_threshold"]
    for axis, (family, model, target, label) in zip(axes, panels, strict=True):
        curve = _cell(result["cells"], family, model, target, label)["curve"]
        t = [row["threshold"] for row in curve]
        top = max(max(row[k] for k in (*RULE_STYLE, "treat_all")) for row in curve)
        axis.plot(
            t,
            [100 * r["treat_all"] for r in curve],
            color=MUTED,
            linestyle="--",
            linewidth=1.2,
            label=text["rule.treat_all"],
        )
        axis.axhline(0.0, color=INK, linewidth=1, label=text["rule.treat_none"])
        for rule, (colour, dash) in RULE_STYLE.items():
            axis.plot(
                t,
                [100 * r[rule] for r in curve],
                color=colour,
                linestyle=dash,
                linewidth=2,
                label=text[f"rule.{rule}"],
            )
        axis.axvline(named, color=MUTED, linewidth=1, linestyle=":")
        marker = text["dc.named"] if family == "echonext" else text["dc.named_short"]
        axis.text(named + 0.01, 100 * top * 1.02, marker, fontsize=7.5, color=MUTED)
        axis.set_ylim(-0.3 * 100 * top, 1.15 * 100 * top)
        axis.set_xlim(0, 0.5)
        key = "echonext" if family == "echonext" else target
        axis.set_title(text[f"panel.{key}"], fontsize=9.5)
        axis.set_xlabel(text["dc.x"], fontsize=8.5)
        axis.xaxis.set_major_formatter(lambda v, _: percent(v, 0, lang))
        digits = 0 if top > 0.05 else 1  # bound now: a lambda would read the last panel's top
        axis.yaxis.set_major_formatter(lambda v, _, d=digits: number(v, d, lang))
        axis.grid(color=GRID)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(text["dc.y"], fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    figure.suptitle(text["dc.title"], fontsize=11)
    figure.tight_layout(rect=(0, 0.12, 1, 0.95))
    figure.savefig(out, dpi=150)
    plt.close(figure)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="en", choices=LANGUAGES)
    lang = parser.parse_args(argv).lang
    suffix = SUFFIX[lang]
    out = FIGURES
    out.mkdir(parents=True, exist_ok=True)
    for path in (
        figure_gap(out / f"ppv_gap{suffix}.png", lang),
        figure_decision(out / f"decision_curves{suffix}.png", lang),
    ):
        print(path.relative_to(RESULTS_DIR.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
