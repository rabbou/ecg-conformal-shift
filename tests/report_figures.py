"""Every figure REPORT.md and README.md print, rebuilt from its file under ``results/``.

Each builder reads one results file and returns what the text prints, formatted
as the text prints it: a table as its rows, a sentence's figures by name.
``tests/test_report_numbers.py`` looks for each one in the text, and refuses a
number in the text that no builder, constant or cited source accounts for.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

from report_text import pct

from ecs.config import RESULTS_DIR

COMPOSITE = "shd_moderate_or_greater_flag"
LVEF = "lvef_lte_45_flag"
CONTEXTS = ("inpatient", "emergency", "outpatient")
# The four EchoNext models, in the order every table prints them.
ARMS = {
    "resnet": "Residual network, trained here",
    "echonext_mini": "EchoNext mini-model, published",
    "ecgfounder": "ECGFounder, frozen",
    "random_init": "Random initialisation, frozen",
}
STRONGEST = ("resnet", "echonext_mini", "ecgfounder")
SITES = {"ptbxl": "PTB-XL", "sph": "Shandong", "acs": "Chongqing"}


@cache
def read(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((RESULTS_DIR / name).read_text())
    return loaded


def count(n: int) -> str:
    return f"{n:,}"


def auroc(x: float) -> str:
    return f"{x:.3f}"


def interval(low: float, high: float) -> str:
    return f"{pct(low)} to {pct(high)}"


# ---------------------------------------------------------------------------
# EchoNext
# ---------------------------------------------------------------------------


def cell(arm: str, context: str, method: str, label: str = COMPOSITE) -> dict[str, Any]:
    rows = read("echonext_transfer.json")["arms"][arm]["coverage"]
    return dict(
        next(r for r in rows if (r["label"], r["context"], r["method"]) == (label, context, method))
    )


def arm_auroc(arm: str, context: str) -> dict[str, float]:
    return dict(read("echonext_transfer.json")["arms"][arm]["auroc"][context][COMPOSITE])


def outcomes(arm: str, context: str) -> dict[str, Any]:
    return dict(read("echonext_outcomes.json")["arms"][arm][context])


def ladder(arm: str) -> dict[int, dict[str, Any]]:
    rows = read("echonext_transfer.json")["arms"][arm]["ladder"]["outpatient"][COMPOSITE]
    return {int(r["labels"]): r for r in rows}


def cohort_rows() -> list[str]:
    """Table 1: the four EchoNext cohorts."""
    t = read("echonext_transfer.json")
    rows = [
        f"| Calibration: validation split, inpatients | {count(t['source']['n'])} | "
        f"{pct(t['source']['prevalence'][COMPOSITE])} | {pct(t['source']['prevalence'][LVEF])} |"
    ]
    names = {"inpatient": "inpatients", "emergency": "emergency", "outpatient": "outpatients"}
    for context in CONTEXTS:
        target = t["targets"][context]
        rows.append(
            f"| Test split, {names[context]} | {count(target['n'])} | "
            f"{pct(target['prevalence'][COMPOSITE])} | {pct(target['prevalence'][LVEF])} |"
        )
    return rows


def operating_rows() -> list[str]:
    """Table 3: AUROC and the sensitivity threshold's operating point, by care setting."""
    rows = []
    for arm, name in ARMS.items():
        cells = []
        for context in ("inpatient", "outpatient"):
            a = arm_auroc(arm, context)
            cells.append(f"{auroc(a['auroc'])} ({auroc(a['low'])} to {auroc(a['high'])})")
        for context in ("inpatient", "outpatient"):
            c = cell(arm, context, "plain")
            low, high = c["coverage_pos_low"], c["coverage_pos_high"]
            cells.append(f"{pct(c['coverage_pos'])} ({interval(low, high)})")
        for context in ("inpatient", "outpatient"):
            cells.append(pct(cell(arm, context, "plain")["coverage_neg"]))
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return rows


def split_rows() -> list[str]:
    """Table 4: what the per-class rule gives each outpatient."""
    rows = []
    for arm, name in ARMS.items():
        o = outcomes(arm, "outpatient")["perlabel"]
        ill, healthy = o["ill"], o["healthy"]
        rows.append(
            f"| {name} | {pct(ill['recognised']['share'])} | {pct(ill['referred']['share'])} | "
            f"{pct(ill['missed']['share'])} | {pct(healthy['cleared']['share'])} | "
            f"{pct(healthy['referred']['share'])} | {pct(healthy['false_alarm']['share'])} | "
            f"{pct(o['referred_all'])} |"
        )
    return rows


def ladder_rows() -> list[str]:
    """Table 6: the ill outpatients covered as local labels are added."""
    rows = []
    for arm, name in ARMS.items():
        steps = ladder(arm)
        cells = [pct(steps[0]["coverage_pos_mean"])]
        for rung in (50, 100, 200, 400):
            s = steps[rung]
            cells.append(f"{pct(s['coverage_pos_mean'])} ({pct(s['coverage_pos_p10'])})")
        cells.append(
            f"{pct(steps[0]['coverage_neg_mean'])} to {pct(steps[100]['coverage_neg_mean'])}"
        )
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return rows


def finding_rows() -> list[str]:
    """Appendix D: each of the eleven findings, the trained network, outpatients."""
    names = {
        "lvef_lte_45_flag": "Ejection fraction 45% or less",
        "lvwt_gte_13_flag": "Left ventricular wall 1.3 cm or more",
        "aortic_stenosis_moderate_or_greater_flag": "Aortic stenosis",
        "aortic_regurgitation_moderate_or_greater_flag": "Aortic regurgitation",
        "mitral_regurgitation_moderate_or_greater_flag": "Mitral regurgitation",
        "tricuspid_regurgitation_moderate_or_greater_flag": "Tricuspid regurgitation",
        "pulmonary_regurgitation_moderate_or_greater_flag": "Pulmonary regurgitation",
        "rv_systolic_dysfunction_moderate_or_greater_flag": "Right ventricular dysfunction",
        "pericardial_effusion_moderate_large_flag": "Pericardial effusion",
        "pasp_gte_45_flag": "Pulmonary artery systolic pressure 45 mmHg or more",
        "tr_max_gte_32_flag": "Tricuspid regurgitation velocity 3.2 m/s or more",
    }
    t = read("echonext_transfer.json")
    rows = []
    for label, name in names.items():
        plain = cell("resnet", "outpatient", "plain", label)
        pooled = cell("resnet", "outpatient", "pooled", label)
        n_ill = plain["n_pos"]
        if n_ill < 10:
            continue
        rows.append(
            f"| {name} | {pct(t['source']['prevalence'][label])} | "
            f"{pct(t['targets']['outpatient']['prevalence'][label])} | {n_ill} | "
            f"{pct(plain['coverage_pos'])} | {pct(pooled['coverage_pos'])} | "
            f"{auroc(t['arms']['resnet']['auroc']['outpatient'][label]['auroc'])} |"
        )
    return rows


def echonext_figures() -> dict[str, str]:
    """The EchoNext figures the prose prints, by name."""
    t = read("echonext_transfer.json")
    severity = read("echonext_severity.json")
    strongest_ill = [cell(a, "outpatient", "plain")["coverage_pos"] for a in STRONGEST]
    strongest_spec_in = [cell(a, "inpatient", "plain")["coverage_neg"] for a in STRONGEST]
    strongest_spec_out = [cell(a, "outpatient", "plain")["coverage_neg"] for a in STRONGEST]
    reweighted = [severity["arms"][a]["reweighted"]["reweighted"] for a in STRONGEST]
    shares = [severity["arms"][a]["reweighted"]["share"] for a in STRONGEST]
    recognised = [
        outcomes(a, "outpatient")["perlabel"]["ill"]["recognised"]["share"] for a in STRONGEST
    ]
    referred = [
        outcomes(a, "outpatient")["perlabel"]["ill"]["referred"]["share"] for a in STRONGEST
    ]
    at_100 = [ladder(a)[100] for a in STRONGEST]
    resnet = "resnet"
    women = next(
        r
        for r in t["arms"][resnet]["subgroups"]
        if (r["label"], r["context"], r["method"], r["kind"], r["group"])
        == (COMPOSITE, "outpatient", "plain", "sex", "female")
    )
    men = next(
        r
        for r in t["arms"][resnet]["subgroups"]
        if (r["label"], r["context"], r["method"], r["kind"], r["group"])
        == (COMPOSITE, "outpatient", "plain", "sex", "male")
    )
    return {
        "records": count(read("echonext_provenance.json")["records"]),
        "training_ecgs": count(t["training"]["n"]),
        "calibration_ecgs": count(t["source"]["n"]),
        "outpatient_ecgs": count(t["targets"]["outpatient"]["n"]),
        "prevalence_in": pct(t["source"]["prevalence"][COMPOSITE]),
        "prevalence_out": pct(t["targets"]["outpatient"]["prevalence"][COMPOSITE]),
        "ill_out_n": count(cell(resnet, "outpatient", "plain")["n_pos"]),
        "strongest_low": pct(min(strongest_ill)),
        "strongest_high": pct(max(strongest_ill)),
        "spec_in_low": pct(min(strongest_spec_in)),
        "spec_in_high": pct(max(strongest_spec_in)),
        "spec_out_low": pct(min(strongest_spec_out)),
        "spec_out_high": pct(max(strongest_spec_out)),
        "resnet_auroc_in": auroc(arm_auroc(resnet, "inpatient")["auroc"]),
        "resnet_auroc_out": auroc(arm_auroc(resnet, "outpatient")["auroc"]),
        "resnet_sens_in": pct(cell(resnet, "inpatient", "plain")["coverage_pos"]),
        "resnet_sens_out": pct(cell(resnet, "outpatient", "plain")["coverage_pos"]),
        "resnet_ppv_out": pct(t["arms"][resnet]["ppv"]["outpatient"][COMPOSITE]["ppv_observed"]),
        "resnet_ppv_from_in": pct(
            t["arms"][resnet]["ppv"]["outpatient"][COMPOSITE]["ppv_from_source"]
        ),
        "recognised_low": pct(min(recognised)),
        "recognised_high": pct(max(recognised)),
        "referred_low": pct(min(referred)),
        "referred_high": pct(max(referred)),
        "resnet_referred_in": pct(cell(resnet, "inpatient", "perlabel")["abstention"]),
        "resnet_referred_out": pct(cell(resnet, "outpatient", "perlabel")["abstention"]),
        "floor_sens_out": pct(cell("random_init", "outpatient", "plain")["coverage_pos"]),
        "floor_referred_out": pct(cell("random_init", "outpatient", "perlabel")["abstention"]),
        "reweighted_low": pct(min(reweighted)),
        "reweighted_high": pct(max(reweighted)),
        "share_low": f"{round(100 * min(shares))}%",
        "share_high": f"{round(100 * max(shares))}%",
        "ladder_low": pct(min(r["coverage_pos_mean"] for r in at_100)),
        "ladder_high": pct(max(r["coverage_pos_mean"] for r in at_100)),
        "ladder_eval": count(at_100[0]["n_eval"]),
        "ladder_eval_ill": count(at_100[0]["n_eval_pos"]),
        "ladder_draws": count(at_100[0]["draws"]),
        "ladder_25_flagged": pct(ladder(resnet)[25]["share_all_flagged"]),
        "emergency_sens": pct(cell(resnet, "emergency", "plain")["coverage_pos"]),
        "lvef_sens_out": pct(cell(resnet, "outpatient", "plain", LVEF)["coverage_pos"]),
        "women": pct(women["coverage_pos"]),
        "women_ci": interval(women["low"], women["high"]),
        "men": pct(men["coverage_pos"]),
        "men_ci": interval(men["low"], men["high"]),
        "mini_auroc_test": auroc(
            t["arms"]["echonext_mini"]["auroc_test_all_contexts"][COMPOSITE]["auroc"]
        ),
        "pooled_resnet_out": pct(cell(resnet, "outpatient", "pooled")["coverage_pos"]),
    }


# ---------------------------------------------------------------------------
# Infarction
# ---------------------------------------------------------------------------


def scheme(site: str, name: str, klass: str) -> dict[str, float]:
    block = read("outcomes.json")["by_corpus"][site]["schemes"][name][klass]
    return {k: float(v["mean"]) for k, v in block.items()}


def site_rows() -> list[str]:
    """Table 7: the infarction model at three sites, sensitivity threshold and per-class rule."""
    aux = read("auxiliary.json")
    rows = []
    for site, name in SITES.items():
        a = aux["discrimination"]["by_corpus"][site]
        plain_mi, plain_non = scheme(site, "plain", "1"), scheme(site, "plain", "0")
        per_mi, per_non = scheme(site, "perlabel", "1"), scheme(site, "perlabel", "0")
        rows.append(
            f"| {name} | {auroc(a['auroc'])} | {pct(plain_mi['correct'])} | "
            f"{pct(plain_non['correct'])} | {pct(per_mi['correct'])} | {pct(per_mi['deferred'])} | "
            f"{pct(per_mi['wrong'])} | {pct(per_non['deferred'])} | {pct(per_non['wrong'])} |"
        )
    return rows


def infarction_figures() -> dict[str, str]:
    aux = read("auxiliary.json")
    metrics = read("baseline/metrics.json")
    shift = read("shift.json")
    corpora = next(
        r
        for r in shift["rows"]
        if (r["alpha"], r["score"], r["correction"]) == (0.1, "lac", "mondrian")
    )["by_corpus"]
    intervals = aux["site_intervals"]["by_corpus"]
    out = {
        "baseline_auroc": auroc(metrics["auroc"]),
        "baseline_low": auroc(metrics["auroc_ci95"][0]),
        "baseline_high": auroc(metrics["auroc_ci95"][1]),
        "fold10": count(metrics["n_test"]),
        "fold10_mi": count(metrics["n_test_positive"]),
        "correction_points": f"{aux['correction']['correction_worth_points']:.2f}",
    }
    for site in SITES:
        out[f"{site}_n"] = count(corpora[site]["n_points"])
        out[f"{site}_prevalence"] = pct(corpora[site]["prevalence"])
        out[f"{site}_per_mi"] = pct(
            scheme(site, "perlabel", "1")["correct"] + scheme(site, "perlabel", "1")["deferred"]
        )
        out[f"{site}_per_non"] = pct(
            scheme(site, "perlabel", "0")["correct"] + scheme(site, "perlabel", "0")["deferred"]
        )
        out[f"{site}_ppv"] = pct(
            aux["predictive_value"]["by_corpus"][site]["positive_predictive_value"]
        )
    for site in ("sph", "acs"):
        w = intervals[site]["perlabel"]["wilson_ci95"]
        out[f"{site}_wilson"] = interval(*w)
        out[f"{site}_mi_n"] = count(intervals[site]["n_mi_cases"])
    ladder_rows = [
        r
        for r in read("target_scale.json")["rows"]
        if (r["alpha"], r["score"], r["family"], r["correction"])
        == (0.1, "lac", "recalibrated", "mondrian")
    ]
    by_rung = {int(r["n_target_records"]): r for r in ladder_rows}
    for rung in sorted(by_rung):
        out[f"ladder_{rung}"] = pct(by_rung[rung]["coverage_by_class"]["1"]["mean"])
    out["ladder_100_mi"] = f"{by_rung[100]['calibration']['n_by_class']['1']['mean']:.1f}"
    return out


def shift_row(correction: str) -> dict[str, Any]:
    rows = read("shift.json")["rows"]
    return dict(
        next(
            r for r in rows if (r["alpha"], r["score"], r["correction"]) == (0.1, "lac", correction)
        )
    )


def pooled_figures() -> dict[str, str]:
    """Appendix A: the shared threshold in both studies."""
    none = shift_row("none")["by_corpus"]
    pooled_mi = scheme("ptbxl", "pooled", "1")
    out = {
        "ptbxl_all": pct(none["ptbxl"]["coverage"]["mean"]),
        "ptbxl_mi": pct(none["ptbxl"]["coverage_by_class"]["1"]["mean"]),
        "ptbxl_mi_wrong": pct(pooled_mi["wrong"]),
        "ptbxl_non_wrong": pct(scheme("ptbxl", "pooled", "0")["wrong"]),
        "sph_mi": pct(none["sph"]["coverage_by_class"]["1"]["mean"]),
        "acs_mi": pct(none["acs"]["coverage_by_class"]["1"]["mean"]),
        "resnet_pooled_out": pct(cell("resnet", "outpatient", "pooled")["coverage_pos"]),
    }
    zero = [
        label
        for label in read("echonext_transfer.json")["labels"]
        if cell("resnet", "outpatient", "pooled", label)["n_pos"]
        and cell("resnet", "outpatient", "pooled", label)["coverage_pos"] == 0.0
    ]
    out["zero_findings"] = str(len(zero))
    return out


def perturbation_rows() -> list[str]:
    """Appendix E: the per-label rule under acquisition faults."""
    names = {
        "clean": "As recorded",
        "limb_reversal": "Arm electrodes exchanged",
        "gain_080": "Every lead × 0.8",
        "gain_125": "Every lead × 1.25",
        "baseline_wander": "Baseline wander added",
        "polarity": "Polarity inverted",
    }
    rows = []
    for key, name in names.items():
        c = read("perturbations.json")["conditions"][key]
        s = c["schemes"]["perlabel"]
        rows.append(
            f"| {name} | {auroc(c['auroc'])} | {pct(s['coverage_mi']['mean'])} | "
            f"{pct(s['coverage_non_mi']['mean'])} | {pct(s['deferred_all']['mean'])} |"
        )
    return rows


def subgroup_rows() -> list[str]:
    """Appendix G: coverage inside sex and age under the per-label rule, PTB-XL fold 10."""
    table = read("subgroups.json")
    names = {
        "age:0-49": "Under 50",
        "age:50-64": "50 to 64",
        "age:65-74": "65 to 74",
        "age:75+": "75 and over",
        "sex:male": "Men",
        "sex:female": "Women",
    }
    rows = []
    for key, name in names.items():
        n = table["counts"][key]
        cells = []
        for label in ("mi", "non_mi"):
            c = table["coverage"]["perlabel"][f"{key}:{label}"]
            low, high = c["ci95"]
            cells.append(f"{100 * c['mean']:.1f} [{100 * low:.1f}, {100 * high:.1f}]")
        rows.append(f"| {name} | {count(n['n'])} ({n['n_mi']}) | " + " | ".join(cells) + " |")
    return rows


def severity_strings() -> dict[str, str]:
    from test_echonext_severity_results import severity_figures

    return severity_figures(read("echonext_severity.json"))


def severity_rows() -> list[str]:
    """Table 5: coverage of the ill outpatients by stratum of severity, and reweighted."""
    s = severity_strings()
    rows = []
    for arm in STRONGEST:
        strata = [f"by_findings:{k}" for k in ("1", "2+")] + [
            f"by_lvef:{k}" for k in ("<=35", "36-45", ">45")
        ]
        cells = [s[f"{arm}:observed"], *(s[f"{arm}:{k}"] for k in strata)]
        cells += [s[f"{arm}:reweighted"], s[f"{arm}:share"]]
        rows.append(f"| {ARMS[arm]} | " + " | ".join(cells) + " |")
    return rows


def severity_counts() -> dict[str, str]:
    arm = read("echonext_severity.json")["arms"]["resnet"]
    n = {
        r["stratum"]: r["n"]
        for r in arm["by_findings"]["outpatient"] + arm["by_lvef"]["outpatient"]
    }
    return {
        "n_one": str(n["1"]),
        "n_two": str(n["2+"]),
        "n_le35": str(n["<=35"]),
        "n_36": str(n["36-45"]),
        "n_gt45": str(n[">45"]),
    }


SEVERITY_NAMES = {
    "findings_count": "Findings present",
    "lvef_value": "Ejection fraction, %",
    "pasp_value": "Pulmonary artery systolic pressure, mmHg",
    "tr_max_velocity_value": "Tricuspid regurgitation peak velocity, m/s",
    "ivs_measurement": "Septal thickness, cm",
    "lvpw_measurement": "Posterior wall thickness, cm",
    "aortic_stenosis_value": "Aortic stenosis, moderate or worse",
    "aortic_regurgitation_value": "Aortic regurgitation, moderate or worse",
    "mitral_regurgitation_value": "Mitral regurgitation, moderate or worse",
    "tricuspid_regurgitation_value": "Tricuspid regurgitation, moderate or worse",
    "pulmonary_regurgitation_value": "Pulmonary regurgitation, moderate or worse",
    "rv_systolic_function_value": "Right ventricular dysfunction, moderate or worse",
    "pericardial_effusion_value": "Pericardial effusion, moderate or large",
}


def severity_compare_rows() -> list[str]:
    """Appendix C: the severity of the ill, inpatients against outpatients."""
    s = severity_strings()
    rows = []
    for column, name in SEVERITY_NAMES.items():
        key = "findings" if column == "findings_count" else column
        sides = []
        for side in ("inpatient", "outpatient"):
            value = s[f"{key}:{side}"].strip("| ")
            measured = s.get(f"{column}:{side}:n")
            sides.append(f"{value}, {measured}" if measured else value)
        rows.append(f"| {name} | {sides[0]} | {sides[1]} | {s[f'{column}:p']} |")
    return rows


def extra_figures() -> dict[str, str]:
    """EchoNext figures the prose prints beyond ``echonext_figures``."""
    t = read("echonext_transfer.json")
    steps = ladder("resnet")
    pool = t["targets"]["outpatient"]["n"] - steps[100]["n_eval"]
    pool_ill = cell("resnet", "outpatient", "plain")["n_pos"] - steps[100]["n_eval_pos"]
    sens_in = [cell(a, "inpatient", "plain")["coverage_pos"] for a in STRONGEST]
    sens_out = [cell(a, "outpatient", "plain")["coverage_pos"] for a in STRONGEST]
    false_alarm = [
        outcomes(a, "outpatient")["perlabel"]["healthy"]["false_alarm"]["share"] for a in STRONGEST
    ]
    referred_all = [outcomes(a, "outpatient")["perlabel"]["referred_all"] for a in STRONGEST]
    return {
        "ladder_ill_in_100": str(round(100 * pool_ill / pool)),
        "sens_in_low": pct(min(sens_in)),
        "sens_in_high": pct(max(sens_in)),
        "spread_points": f"{100 * (max(sens_out) - min(sens_out)):.1f}",
        "floor_spec_out": pct(cell("random_init", "outpatient", "plain")["coverage_neg"]),
        "false_alarm_low": pct(min(false_alarm)),
        "false_alarm_high": pct(max(false_alarm)),
        "referred_all_low": pct(min(referred_all)),
        "referred_all_high": pct(max(referred_all)),
        "healthy_before": pct(steps[0]["coverage_neg_mean"]),
        "healthy_after": pct(steps[100]["coverage_neg_mean"]),
        "acs_auroc": auroc(read("auxiliary.json")["discrimination"]["by_corpus"]["acs"]["auroc"]),
        "strongest_about": strongest_about(),
    }


def strongest_about() -> str:
    """'About 72%': the mean of the three strongest models' sensitivity among outpatients."""
    ill = [cell(a, "outpatient", "plain")["coverage_pos"] for a in STRONGEST]
    return f"{round(100 * sum(ill) / len(ill))}%"
