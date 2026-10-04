"""Every figure the report and the README print, by the name their templates use.

``docs/templates/REPORT.md`` and ``docs/templates/README.md`` hold the text with
each computed number written as ``{{name}}``; ``tests/report_render.py`` fills
the names from ``values()`` and the tables from ``rows()``.  Infarction figures
come from ``report_infarction``, the EchoNext and rotation figures from here.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from functools import cache
from typing import Any

import report_figures as f
from report_infarction import infarction_values, word
from report_text import pct
from test_rotation_report import GROUPS, _cases, _finite, _grid, _mean_pct, _read, starved

# The EchoNext arms as the report names them, in the order its tables print them.
ARM_NAMES = {
    "resnet": "Study ResNet, trained",
    "echonext_mini": "EchoNext mini-model, published",
    "ecgfounder": "ECGFounder, frozen, probes",
    "random_init": "Random initialisation, frozen, probes",
}
LADDER_RUNGS = (0, 25, 50, 100, 200, 400)


def base_figures() -> dict[str, str]:
    """The EchoNext and infarction figures ``report_figures`` builds, by their names there."""
    out = f.echonext_figures() | f.extra_figures()
    out |= {f"inf.{k}": v for k, v in f.infarction_figures().items()}
    out |= {f"pool.{k}": v for k, v in f.pooled_figures().items()}
    out |= {f"sev.{k}": v.replace("\n", " ") for k, v in f.severity_strings().items()}
    out |= {f"sev.{k}": v for k, v in f.severity_counts().items()}
    return out


def echonext_values() -> dict[str, str]:
    t = f.read("echonext_transfer.json")
    out: dict[str, str] = {}
    for arm in f.ARMS:
        for context in f.CONTEXTS:
            for method in ("plain", "pooled", "perlabel"):
                c = f.cell(arm, context, method)
                stem = f"e.{arm}.{context}.{method}"
                out[f"{stem}.pos"] = pct(c["coverage_pos"])
                out[f"{stem}.neg"] = pct(c["coverage_neg"])
                out[f"{stem}.abst"] = pct(c["abstention"])
            a = f.arm_auroc(arm, context)
            out[f"ea.{arm}.{context}"] = f.auroc(a["auroc"])
        out[f"ea.{arm}.test"] = f.auroc(
            t["arms"][arm]["auroc_test_all_contexts"][f.COMPOSITE]["auroc"]
        )
        inside, outside = f.arm_auroc(arm, "inpatient"), f.arm_auroc(arm, "outpatient")
        assert inside["low"] < outside["high"] and outside["low"] < inside["high"], (
            f"section 3.7 says the two AUROC intervals overlap for every arm: {arm}"
        )
        for rung, r in f.ladder(arm).items():
            stem = f"el.{arm}.{rung}"
            out[f"{stem}.cov"] = pct(r["coverage_pos_mean"])
            out[f"{stem}.neg"] = pct(r["coverage_neg_mean"])
            out[f"{stem}.recognised"] = pct(r["recognised_pos_mean"])
            out[f"{stem}.referred"] = pct(r["referred_pos_mean"])
    # The ill a rung refits on are drawn from one pool with one seed, so every arm
    # counts the same ill in the same draws; the text quotes one count.
    counts = {(r["fit_ill_mean"], r["fit_ill_min"], r["fit_ill_max"]) for r in _rungs(100)}
    assert len(counts) == 1, counts
    at_25 = _rungs(25)[0]
    mean_100 = _rungs(100)[0]["fit_ill_mean"]
    out |= {
        "el.fit_ill_mean": f"{mean_100:.1f}",
        "el.fit_ill_min": str(_rungs(100)[0]["fit_ill_min"]),
        "el.fit_ill_max": str(_rungs(100)[0]["fit_ill_max"]),
        "el.fit_ill_25_mean": f"{at_25['fit_ill_mean']:.1f}",
        "el.fit_ill_25_min": str(at_25["fit_ill_min"]),
        "d.ill_in_100_whole": str(round(mean_100)),
    }
    zero_rung, hundred = f.ladder("resnet")[0], f.ladder("resnet")[100]
    recognised = hundred["recognised_pos_mean"] - zero_rung["recognised_pos_mean"]
    referred = hundred["referred_pos_mean"] - zero_rung["referred_pos_mean"]
    assert recognised > referred, "section 3.7: the labels buy back recognition first"
    out["d.el.recognised_gain"] = f"{100 * recognised:.1f}"
    out["d.el.referred_gain"] = f"{100 * referred:.1f}"
    # Section 3.7 quotes one figure for the ResNet and ECGFounder, and section 2.2
    # puts the mini-model within a tenth of a point of its published 82.0%.
    pos = {a: out[f"e.{a}.outpatient.perlabel.pos"] for a in ("resnet", "ecgfounder")}
    assert pos["resnet"] == pos["ecgfounder"], pos
    mini = t["arms"]["echonext_mini"]["auroc_test_all_contexts"][f.COMPOSITE]["auroc"]
    assert abs(mini - 0.820) < 0.001, mini
    source, targets = t["source"], t["targets"]
    out |= {
        "d.echo_cohorts": f.count(source["n"] + sum(x["n"] for x in targets.values())),
        "d.lvef_in": pct(source["prevalence"][f.LVEF]),
        "d.lvef_out": pct(targets["outpatient"]["prevalence"][f.LVEF]),
    }
    for context in ("inpatient", "emergency"):
        target = targets[context]
        out[f"d.test_{context}_n"] = f.count(target["n"])
        out[f"d.test_{context}_prev"] = pct(target["prevalence"][f.COMPOSITE])
        out[f"d.test_{context}_lvef"] = pct(target["prevalence"][f.LVEF])
    runs = {arm: t["arms"][arm]["run"] for arm in f.ARMS}
    founder = runs["ecgfounder"]
    out |= {
        "e.run.mini_seconds": f"{runs['echonext_mini']['seconds']:.1f}",
        "e.run.founder_features": f.count(int(founder["features"].split("-d", 1)[0])),
        "e.run.store_size": f.count(founder["store_size"]),
        "e.run.founder_seconds": str(round(founder["seconds"])),
    }
    zero = {
        label
        for label in t["labels"]
        if f.cell("resnet", "outpatient", "pooled", label)["n_pos"]
        and f.cell("resnet", "outpatient", "pooled", label)["coverage_pos"] == 0.0
    }
    assert zero == {
        "aortic_stenosis_moderate_or_greater_flag",
        "aortic_regurgitation_moderate_or_greater_flag",
        "pericardial_effusion_moderate_large_flag",
    }, zero
    out["w.pool.zero_findings"] = word(len(zero))
    return out


def _rungs(rung: int) -> list[dict[str, Any]]:
    return [f.ladder(arm)[rung] for arm in f.ARMS]


def rotation_values() -> dict[str, str]:
    """Each fragment ``test_rotation_report`` rebuilds, and the abstract's figures."""
    out: dict[str, str] = {}
    for group in GROUPS:
        for what, expected in _cases(group):
            slug = what.replace(" ", "_")
            if group == "arms" and what.endswith(" auroc"):
                slug = "random_init_auroc" if what.startswith("random_init") else "best_auroc"
            out[f"rot.{group}.{slug}"] = expected
    home = {c: [r for r in _grid(c) if r["role"] == "home"] for c in ("none", "mondrian")}
    away = [r for r in _grid("mondrian") if r["role"] == "away"]
    grid = _grid("none")
    rotation = _read("rotation.json")
    sds = [b["mondrian"]["away_bias"]["sd_across_sources"] for b in rotation["bias"].values()]
    nsr = rotation["bias"]["NSR"]["mondrian"]["away_bias"]
    assert max(sds) > abs(nsr["mean"]), "the spread across corpora exceeds the mean"
    chow = _read("rotation_uncertainty.json")["reading"]["conformal_minus_chow"]
    median = chow["by_correction"]["mondrian"]["median"]
    correction = f.read("auxiliary.json")["correction"]["correction_worth_points"] / 100
    marginal = statistics.mean(float(r["coverage_mean"]) for r in home["none"])
    diagnosis = statistics.mean(float(r["coverage_diagnosis_mean"]) for r in home["none"])
    starving = {
        c: len(
            {
                (r["source"], r["label"])
                for r in _grid(c)
                if int(r["threshold_diagnosis_n_infinite"]) > 0
            }
        )
        for c in ("mondrian", "weighted")
    }
    out |= {
        "rot.none.home.all": _mean_pct(home["none"], "coverage_mean"),
        "rot.none.home.diagnosis": _mean_pct(home["none"], "coverage_diagnosis_mean"),
        "rot.mondrian.home.finite": _mean_pct(_finite(home["mondrian"]), "coverage_diagnosis_mean"),
        "rot.mondrian.away.finite": _mean_pct(_finite(away), "coverage_diagnosis_mean"),
        "rot.mondrian.home.diagnosis": _mean_pct(home["mondrian"], "coverage_diagnosis_mean"),
        "rot.mondrian.away.diagnosis": _mean_pct(away, "coverage_diagnosis_mean"),
        "rot.sd_points": str(round(100 * max(sds))),
        "w.rot.starved": word(len(starved())),
        "w.rot.home_gap": word(round(100 * (marginal - diagnosis))),
        "w.rot.chow_times": word(round(median / correction)),
        "rot.cells.away_n": str(sum(r["role"] == "away" for r in grid)),
        "rot.cells.home_n": str(sum(r["role"] == "home" for r in grid)),
        "rot.cells.holdout_n": str(sum(r["role"] not in ("home", "away") for r in grid)),
        "rot.discussion.chow": f"differ by a median of {median:.3f} in coverage",
        "rot.limitations.mondrian_pairs": f"{word(starving['mondrian'])} source-diagnosis pairs "
        "of section 3.6",
        "rot.limitations.weighted_pairs": f"reweights, {word(starving['weighted'])} do",
    }
    return out


@cache
def values() -> dict[str, str]:
    return base_figures() | echonext_values() | rotation_values() | infarction_values()


def split_rows_named() -> list[str]:
    """Table 5: what the per-label scheme gives each outpatient."""
    return [
        row.replace(f"| {f.ARMS[arm]} |", f"| {ARM_NAMES[arm]} |", 1)
        for arm, row in zip(f.ARMS, f.split_rows(), strict=True)
    ]


def ladder_rows_split() -> list[str]:
    """Table 6: the ladder, with the recognised and the referred at 100 labels."""
    rows = []
    for arm, name in ARM_NAMES.items():
        steps = f.ladder(arm)
        cells = [pct(steps[0]["coverage_pos_mean"])]
        for rung in (50, 100, 200, 400):
            s = steps[rung]
            cells.append(f"{pct(s['coverage_pos_mean'])} ({pct(s['coverage_pos_p10'])})")
        cells += [pct(steps[100]["recognised_pos_mean"]), pct(steps[100]["referred_pos_mean"])]
        cells.append(
            f"{pct(steps[0]['coverage_neg_mean'])} to {pct(steps[100]['coverage_neg_mean'])}"
        )
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return rows


def severity_rows_named() -> list[str]:
    """Table 8: the ill outpatients covered by stratum of severity, and reweighted."""
    return [
        row.replace(f"| {f.ARMS[arm]} |", f"| {ARM_NAMES[arm]} |", 1)
        for arm, row in zip(f.STRONGEST, f.severity_rows(), strict=True)
    ]


ROWS: dict[str, Callable[[], list[str]]] = {
    "subgroup_rows": f.subgroup_rows,
    "severity_compare_rows": f.severity_compare_rows,
    "split_rows_named": split_rows_named,
    "ladder_rows_split": ladder_rows_split,
    "severity_rows_named": severity_rows_named,
}
