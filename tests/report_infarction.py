"""The infarction figures the report prints, by name, each read from its file under ``results/``.

Every key is a string formatted as the text prints it.  A sentence that makes a
claim about the figures (a rate that doubles, an interval that excludes zero)
asserts the claim here, so a regenerated file that breaks the claim fails
before the text is rendered.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from report_figures import SITES, auroc, count, interval, read, scheme
from report_text import pct
from sklearn.metrics import roc_auc_score

from ecs.config import RESULTS_DIR

CLASSES = {"mi": "1", "non": "0"}
SCHEMES = ("plain", "pooled", "perlabel")
SUBGROUPS = {"young": "age:0-49", "old": "age:75+", "men": "sex:male", "women": "sex:female"}

_UNITS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def word(n: int) -> str:
    """A whole number below a hundred as the prose spells it."""
    if not 0 <= n < 100:
        raise ValueError(n)
    if n < 20:
        return _UNITS[n]
    tens, unit = divmod(n, 10)
    return _TENS[tens] + (f"-{_UNITS[unit]}" if unit else "")


def points(x: float, digits: int = 1) -> str:
    """A share in percentage points without the sign: ``0.0974`` is ``9.7``."""
    return f"{100 * x:.{digits}f}"


def signed_interval(low: float, high: float) -> str:
    """An interval in points, a negative end printed with a hyphen-minus."""
    return f"{points(low)} to {points(high)}"


def outcome_figures() -> dict[str, str]:
    out: dict[str, str] = {}
    for site in SITES:
        for name in SCHEMES:
            for klass, code in CLASSES.items():
                s = scheme(site, name, code)
                s["cov"] = s["correct"] + s["deferred"]
                s["noflag"] = s["wrong"] + s["deferred"]
                for what, x in s.items():
                    out[f"o.{site}.{name}.{klass}.{what}"] = pct(x)
                    out[f"o2.{site}.{name}.{klass}.{what}"] = pct(x, 2)
                    out[f"h.{site}.{name}.{klass}.{what}"] = str(round(100 * x))
    plain, pooled, per = (scheme("ptbxl", n, "0") for n in SCHEMES)
    per_mi = scheme("ptbxl", "perlabel", "1")
    saved = plain["wrong"] - per["wrong"]
    # Section 3.2: the plain threshold is the per-label threshold of the ill, so
    # the non-MI cases given the right label alone are the same under both.
    assert pct(plain["correct"]) == pct(per["correct"])
    # The abstract: against pooled CP the per-label scheme doubles the false alarms.
    assert 1.75 < per["wrong"] / pooled["wrong"] < 2.5
    # Calibrating within each label brings coverage to the same figure in each.
    assert pct(per_mi["correct"] + per_mi["deferred"]) == pct(per["correct"] + per["deferred"])
    out |= {
        "d.perlabel_within_each": pct(per_mi["correct"] + per_mi["deferred"]),
        "w.pooled_mi_deferred": word(round(100 * scheme("ptbxl", "pooled", "1")["deferred"])),
        "w.false_alarms_saved": word(round(100 * saved)),
        "d.false_alarms_removed": points(saved),
        "d.deferrals_share_of_gain": pct(per["deferred"] / saved),
        "d.mi_miss_whole": f"{round(100 * scheme('ptbxl', 'plain', '1')['wrong'])}%",
    }
    acs_plain_mi, acs_plain_non = scheme("acs", "plain", "1"), scheme("acs", "plain", "0")
    src_mi = scheme("ptbxl", "plain", "1")
    out["w.acs_sensitivity_lost"] = word(round(100 * (src_mi["correct"] - acs_plain_mi["correct"])))
    out["w.acs_specificity_lost"] = word(round(100 * (plain["correct"] - acs_plain_non["correct"])))
    calibration_mi = read("outcomes.json")["by_corpus"]["ptbxl"]["n_positive"]["mean"]
    out["d.minority_calibration_n"] = str(round(calibration_mi))
    t = read("outcomes.json")["thresholds"]
    q0, q1 = t["quantile:0"]["mean"], t["quantile:1"]["mean"]
    assert q0 + q1 > 1, "the report says no set comes back empty at this level"
    out |= {
        "thr.pooled_upper": f"{t['pooled:upper']['mean']:.3f}",
        "thr.q0": f"{q0:.3f}",
        "thr.q1": f"{q1:.3f}",
        "d.thr_sum": f"{q0 + q1:.2f}",
    }
    return out


def _shift_rows() -> dict[str, dict[str, Any]]:
    return {
        r["correction"]: r
        for r in read("shift.json")["rows"]
        if (r["alpha"], r["score"]) == (0.1, "lac")
    }


def shift_figures() -> dict[str, str]:
    rows = _shift_rows()
    out: dict[str, str] = {}
    for correction, row in rows.items():
        for site, c in row["by_corpus"].items():
            by_class = c["coverage_by_class"]
            figures = {
                "all": c["coverage"]["mean"],
                "mi": by_class["1"]["mean"],
                "non": by_class["0"]["mean"],
                "abst": c["abstention_rate"]["mean"],
                "empty": c["empty_rate"]["mean"],
            }
            for what, x in figures.items():
                out[f"s.{correction}.{site}.{what}"] = pct(x)
            out[f"s.{correction}.{site}.sd_mi_points"] = points(by_class["1"]["sd"], 2)
    corpora = rows["none"]["by_corpus"]
    out["s.sph_patients"] = count(corpora["sph"]["n_patients"])
    out["s.acs_patients"] = count(corpora["acs"]["n_patients"])
    out["d.acs_shortfall"] = points(0.9 - corpora["acs"]["coverage_by_class"]["1"]["mean"])
    out["d.inf_scored"] = count(sum(c["n_points"] for c in corpora.values()))
    out["d.ptbxl_prevalence_whole"] = f"{round(100 * corpora['ptbxl']['prevalence'])}%"
    weighted = rows["weighted"]["by_corpus"]
    paired = read("shift.json")["reading"]["paired"]
    out |= {
        "wt.acs.estimate": pct(weighted["acs"]["calibration"]["estimated_prevalence"]["mean"]),
        "wt.sph.estimate": pct(weighted["sph"]["calibration"]["estimated_prevalence"]["mean"]),
        "wt.calibration_n": count(round(rows["weighted"]["calibration"]["n"]["mean"])),
        "wt.acs.ess": count(round(weighted["acs"]["calibration"]["effective_sample_size"]["mean"])),
        "wt.sph.ess": count(round(weighted["sph"]["calibration"]["effective_sample_size"]["mean"])),
        "wt.ptbxl.gain": points(paired["ptbxl"]["sick coverage"]["weighted - none"]["mean"]),
        "wt.ptbxl.mondrian_gain": points(
            paired["ptbxl"]["sick coverage"]["mondrian - none"]["mean"]
        ),
        "wt.sph.loss": points(-paired["sph"]["sick coverage"]["weighted - none"]["mean"]),
    }
    return out


def record_figures() -> dict[str, str]:
    """The release counts of section 2.1 and the PTB-XL model's settings."""
    ingest = read("ingest_report.json")
    acs = ingest["acs"]
    scored = _shift_rows()["none"]["by_corpus"]["acs"]["n_points"]
    unlabelled = acs["n_kept"] - scored
    reasons = list(acs["excluded"].values())
    cropped = next(d for d in ingest["sph"]["deviations"] if "cropped" in d)
    deviation = next(d for d in read("external/sph.json")["deviations"] if "'old' modifier" in d)
    found = re.search(r"(\d+) of its (\d+) positives", deviation)
    assert found, deviation
    old, positives = map(int, found.groups())
    config, metrics = read("baseline/config.json"), read("baseline/metrics.json")
    minutes = round(metrics["seconds"] / 60)
    hours, rest = divmod(minutes, 60)
    out = {
        **{f"ing.{c}.records": count(ingest[c]["n_records"]) for c in ("ptbxl", "sph", "acs")},
        "ing.acs.excluded": count(acs["n_excluded"]),
        "ing.sph.cropped": count(int(cropped.split()[0])),
        "d.acs_unlabelled": count(unlabelled),
        "d.acs_labelled": count(acs["n_records"] - unlabelled),
        "w.acs_nonfinite": word(sum("NaN" in r for r in reasons)),
        "w.acs_unreadable": word(sum("unreadable" in r for r in reasons)),
        "w.acs_excluded": word(len(reasons)),
        "d.sph_old": pct(old / positives),
        "d.sph_not_old": pct(1 - old / positives),
        "base.n_train": count(config["n_train"]),
        "base.n_validation": count(config["n_validation"]),
        "base.anchor": f"{read('baseline.json')['reference_auroc']:.3f}",
        "base.centre": f"{config['standardisation']['centre_mv']:.5f}",
        "base.scale": f"{config['standardisation']['scale_mv']:.5f}",
        "base.epochs": str(config["epochs"]),
        "base.patience": str(config["patience"]),
        "base.epoch_kept": str(metrics["epoch_kept"]),
        "d.epoch_stopped": str(metrics["epoch_kept"] + config["patience"]),
        "base.seconds": count(round(metrics["seconds"])),
        "w.base_duration": f"{word(hours)} hours {word(rest)}",
        "base.auprc": auroc(metrics["auprc"]),
        "base.auprc_low": auroc(metrics["auprc_ci95"][0]),
        "base.auprc_high": auroc(metrics["auprc_ci95"][1]),
        "d.fold10_non_mi": count(metrics["n_test"] - metrics["n_test_positive"]),
    }
    assert len(reasons) == acs["n_excluded"]
    return out


def subgroup_figures() -> dict[str, str]:
    table = read("subgroups.json")
    cov, diff, counts = table["coverage"], table["differences"], table["counts"]
    out: dict[str, str] = {}
    for name in SCHEMES:
        for short, key in SUBGROUPS.items():
            for klass, suffix in (("all", "all"), ("mi", "mi"), ("non", "non_mi")):
                out[f"sg.{name}.{short}.{klass}"] = pct(cov[name][f"{key}:{suffix}"]["mean"])

    def flipped(name: str) -> tuple[float, tuple[float, float]]:
        """A difference read the other way round: the file holds old minus young."""
        d = diff[name]
        return -d["mean"], (-d["ci95"][1], -d["ci95"][0])

    age_all, age_all_ci = flipped("perlabel:age:75+-0-49:all")
    age_non, age_non_ci = flipped("perlabel:age:75+-0-49:non_mi")
    age_mi = diff["perlabel:age:75+-0-49:mi"]
    sex_pooled, sex_pooled_ci = flipped("pooled:sex:female-male:mi")
    sex_per, sex_per_ci = flipped("perlabel:sex:female-male:mi")
    # The claims of section 3.4 and of the subgroup limitation.
    assert age_all_ci[0] > 0 and age_non_ci[0] > 0
    assert age_mi["ci95"][0] < 0 < age_mi["ci95"][1]
    assert 0 < sex_pooled_ci[0] < 0.01, "the pooled sex gap excludes zero by under a point"
    assert sex_per_ci[0] < 0 < sex_per_ci[1]
    young, old = counts[SUBGROUPS["young"]], counts[SUBGROUPS["old"]]
    out |= {
        "sgd.age.all": points(age_all),
        "sgd.age.all_ci": signed_interval(*age_all_ci),
        "sgd.age.mi": points(age_mi["mean"]),
        "sgd.age.mi_ci_comma": signed_interval(*age_mi["ci95"]),
        "sgd.age.non": points(age_non),
        "sgd.age.non_ci": signed_interval(*age_non_ci),
        "sgd.sex.pooled": points(sex_pooled),
        "sgd.sex.pooled_ci_comma": signed_interval(*sex_pooled_ci),
        "sgd.sex.perlabel": points(sex_per),
        "sgd.sex.perlabel_ci_comma": signed_interval(*sex_per_ci),
        "sg.young_mi_n": str(young["n_mi"]),
        "w.sg.young_mi_per_draw": word(round(young["n_mi"] / 2)),
        "sg.young_prevalence": pct(young["n_mi"] / young["n"]),
        "sg.old_prevalence": pct(old["n_mi"] / old["n"]),
        "sg.old_flagged": pct(1 - cov["perlabel"]["age:75+:non_mi"]["mean"]),
        "sg.young_flagged": pct(1 - cov["perlabel"]["age:0-49:non_mi"]["mean"]),
        "sg.women_mi_n": str(counts[SUBGROUPS["women"]]["n_mi"]),
        "sg.old_mi_n": str(old["n_mi"]),
        "w.subgroup_cells": word(sum(len(cells) for cells in cov.values())),
    }
    return out


def perturbation_figures() -> dict[str, str]:
    table = read("perturbations.json")
    out: dict[str, str] = {}
    for name, c in table["conditions"].items():
        s = c["schemes"]["perlabel"]
        out |= {
            f"p.{name}.auroc": auroc(c["auroc"]),
            f"p.{name}.mi": points(s["coverage_mi"]["mean"]),
            f"p.{name}.non": points(s["coverage_non_mi"]["mean"]),
            f"p.{name}.all": points(s["coverage_all"]["mean"]),
            f"p.{name}.deferred": points(s["deferred_all"]["mean"]),
        }
    clean, gain, limb = (table["conditions"][k] for k in ("clean", "gain_080", "limb_reversal"))

    def per(c: dict[str, Any], what: str) -> float:
        return float(c["schemes"]["perlabel"][what]["mean"])

    scores = np.load(RESULTS_DIR / "perturbations.npz")
    labels = scores["labels"]
    p_clean, p_gain = scores["probs_clean"][:, 1], scores["probs_gain_080"][:, 1]
    p_wander = scores["probs_baseline_wander"][:, 1]
    t = read("outcomes.json")["thresholds"]
    low, high = t["perlabel:lower"]["mean"], t["perlabel:upper"]["mean"]

    def between(p: np.ndarray) -> float:
        return float(((p >= low) & (p <= high)).mean())

    ill, well = labels == 1, labels == 0
    order_clean = np.sign(p_clean[ill][:, None] - p_clean[well][None, :])
    order_gain = np.sign(p_gain[ill][:, None] - p_gain[well][None, :])
    flipped = float((order_clean != order_gain).mean())
    auroc_shift = abs(roc_auc_score(labels, p_gain) - roc_auc_score(labels, p_clean))
    # Section 3.5: the gain fault leaves the AUROC unchanged to four decimals.
    assert round(gain["auroc"], 4) == round(clean["auroc"], 4)
    diff = table["clean_matches_published_scores"]["largest_absolute_difference"]
    out |= {
        "p.clean_agreement": f"{diff / 1e-7:.1f}",
        "d.limb_deferral_move": points(per(limb, "deferred_all") - per(clean, "deferred_all")),
        "p.wander.sd_clean": f"{p_clean.std():.2f}",
        "p.wander.sd": f"{p_wander.std():.2f}",
        "p.wander.median_clean": f"{np.median(p_clean):.2f}",
        "p.wander.median": f"{np.median(p_wander):.2f}",
        "p.wander.between_clean": pct(between(p_clean)),
        "p.wander.between": pct(between(p_wander)),
        "p.gain.moved": count(gain["score_shift"]["records_whose_rank_moved"]),
        "p.gain.tau": f"{gain['score_shift']['kendall_tau_vs_clean']:.3f}",
        "p.gain.pairs_flipped": pct(flipped),
        "p.gain.auroc_fifth": str(round(auroc_shift * 1e5)),
        "p.gain.median_clean": f"{gain['score_shift']['median_score_clean']:.3f}",
        "p.gain.median": f"{gain['score_shift']['median_score']:.3f}",
        "d.gain_non_mi_fall": points(per(clean, "coverage_non_mi") - per(gain, "coverage_non_mi")),
    }
    return out


def auxiliary_figures() -> dict[str, str]:
    aux = read("auxiliary.json")
    chow = aux["chow"]["by_corpus"]
    out: dict[str, str] = {
        "aux.correction_points": f"{aux['correction']['correction_worth_points']:.2f}",
        "aux.between_draw_sd": f"{aux['correction']['between_draw_sd_points']:.2f}",
        "aux.mi_cases_needed": count(aux["sample_size"]["mi_cases_needed"]),
        "aux.tracings.acs": count(aux["sample_size"]["tracings_needed"]["acs"]),
        "aux.tracings.sph": count(aux["sample_size"]["tracings_needed"]["sph"]),
        "d.acs_tracings_rounded": count(round(aux["sample_size"]["tracings_needed"]["acs"], -2)),
        "d.sph_tracings_rounded": count(round(aux["sample_size"]["tracings_needed"]["sph"], -3)),
    }
    for site, c in chow.items():
        conformal, per_class = c["conformal_perlabel"], c["chow_per_class"]
        # The discussion: the reject rule refuses less, by under a point, at every site.
        assert 0 < conformal["deferred"] - per_class["deferred"] < 0.01, site
        out[f"chow.{site}.mi"] = f"{c['conformal_minus_chow_per_class_mi_points']:.2f}"
        out[f"chow.{site}.non"] = f"{abs(conformal['non_mi'] - per_class['non_mi']) * 100:.2f}"
        out[f"chow.{site}.sym_mi"] = pct(c["chow_symmetric"]["mi"])
        out[f"chow.{site}.sym_non"] = pct(c["chow_symmetric"]["non_mi"])
        out[f"chow.{site}.conf_mi"] = pct(conformal["mi"])
        out[f"chow.{site}.conf_non"] = pct(conformal["non_mi"])
    d = aux["differences"]["by_corpus"]
    pooled_ci, per_ci = d["ptbxl"]["pooled"]["ci95"], d["ptbxl"]["perlabel"]["ci95"]
    assert pooled_ci[1] < per_ci[0], "the two levels remain separated"
    out |= {
        "diff.ptbxl": points(d["ptbxl"]["difference"]["mean"]),
        "diff.ptbxl_ci": signed_interval(*d["ptbxl"]["difference"]["ci95"]),
        "diff.acs": points(d["acs"]["difference"]["mean"]),
        "diff.acs_ci": signed_interval(*d["acs"]["difference"]["ci95"]),
        "diff.ptbxl_pooled_ci": interval(*pooled_ci),
        "diff.ptbxl_perlabel_ci": interval(*per_ci),
    }
    sph = aux["site_intervals"]["by_corpus"]["sph"]["pooled"]
    low, high = sph["wilson_ci95"]
    assert low < 0.9 < high, "the 90% requested sits inside Shandong's interval"
    out |= {
        "inf.sph_pooled_wilson": interval(low, high),
        "w.sph_wilson_below": word(round(100 * (sph["coverage"] - low))),
        "w.sph_wilson_above": word(round(100 * (high - sph["coverage"]))),
    }
    return out


def abstention_figures() -> dict[str, str]:
    rows = {(r["alpha"], r["score"], r["correction"]): r for r in read("abstention.json")["rows"]}
    eighty = rows[(0.2, "lac", "none")]
    assert eighty["two_label_rate"]["mean"] == 0, "at 80% every deferral is an empty set"
    return {
        "ab.aps_mi": pct(rows[(0.1, "aps", "none")]["coverage_by_class"]["1"]["mean"]),
        "ab.lac_mi": pct(rows[(0.1, "lac", "none")]["coverage_by_class"]["1"]["mean"]),
        "ab.empty_80": pct(eighty["empty_rate"]["mean"]),
        "d.pooled_deferred_per_100": str(
            round(100 * rows[(0.1, "lac", "none")]["abstention_rate"]["mean"])
        ),
        "d.perlabel_deferred_per_100": str(
            round(100 * rows[(0.1, "lac", "mondrian")]["abstention_rate"]["mean"])
        ),
    }


def target_scale_figures() -> dict[str, str]:
    table = read("target_scale.json")
    out = {"ts.n_eval": count(table["split"]["n_eval"])}
    for r in table["rows"]:
        if (r["alpha"], r["score"]) == (0.1, "lac"):
            key = f"ts.{r['correction']}.{r['family']}.{r['n_target_records']}"
            out[key] = pct(r["coverage_by_class"]["1"]["mean"])
    return out


def infarction_values() -> dict[str, str]:
    return (
        outcome_figures()
        | shift_figures()
        | record_figures()
        | subgroup_figures()
        | perturbation_figures()
        | auxiliary_figures()
        | abstention_figures()
        | target_scale_figures()
    )
