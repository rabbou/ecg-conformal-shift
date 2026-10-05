"""Every figure the rendered documents print, by name, read from a file under ``results/``.

A name maps to the string the text prints.  Nothing here is typed: each value is
read from the results file it comes from and formatted the way the prose prints
it, so a regenerated file changes the rendered text and the rendering test fails
until the documents are rendered again.

The claims the prose makes about these figures ("the same share", "about a
quarter", "within a point") are asserted in ``test_paper_numbers.py``.
"""

from __future__ import annotations

import csv
import json
import math
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STRONGEST = ("resnet", "echonext_mini", "ecgfounder")
SHORT = {"resnet": "resnet", "echonext_mini": "mini", "ecgfounder": "ecgf", "random_init": "floor"}
CONTEXT = {"inpatient": "in", "emergency": "em", "outpatient": "out"}
COMPOSITE = "shd_moderate_or_greater_flag"


@cache
def read(name: str) -> Any:
    return json.loads((RESULTS / name).read_text())


@cache
def coverage_rows() -> tuple[dict[str, str], ...]:
    with (RESULTS / "echonext_coverage.csv").open() as handle:
        return tuple(csv.DictReader(handle))


def cell(arm: str, context: str, method: str, label: str = COMPOSITE) -> dict[str, str]:
    (row,) = [
        r
        for r in coverage_rows()
        if (r["arm"], r["label"], r["context"], r["method"]) == (arm, label, context, method)
    ]
    return row


def pct(x: float, digits: int = 1) -> str:
    """A share as the text prints it: ``0.716`` is ``71.6%``."""
    return f"{100 * x:.{digits}f}%"


def per100(x: float) -> str:
    """A share as a count of 100 patients: ``0.716`` is ``72``."""
    return f"{round(100 * x)}"


def count(n: float) -> str:
    return f"{round(n):,}"


def interval(low: float, high: float) -> str:
    return f"{pct(low)} to {pct(high)}"


def p_value(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def echonext() -> dict[str, str]:
    transfer = read("echonext_transfer.json")
    clinical = read("echonext_clinical.json")
    severity = read("echonext_severity.json")
    out: dict[str, str] = {}
    source, targets = transfer["source"], transfer["targets"]
    out["cal_n"] = count(source["n"])
    out["cal_prev"] = pct(source["prevalence"][COMPOSITE])
    out["train_n"] = count(transfer["training"]["n"])
    for context, short in CONTEXT.items():
        out[f"{short}_n"] = count(targets[context]["n"])
        out[f"{short}_prev"] = pct(targets[context]["prevalence"][COMPOSITE])
    out["out_ill"] = count(clinical["arms"]["resnet"]["outpatient"]["n_ill"])
    out["out_healthy"] = count(
        clinical["arms"]["resnet"]["outpatient"]["n"]
        - clinical["arms"]["resnet"]["outpatient"]["n_ill"]
    )

    for arm, short in SHORT.items():
        auroc = transfer["arms"][arm]["auroc"]
        for context, c in CONTEXT.items():
            out[f"auroc_{short}_{c}"] = f"{auroc[context][COMPOSITE]['auroc']:.3f}"
            plain = cell(arm, context, "plain")
            out[f"sens_{short}_{c}"] = pct(float(plain["coverage_pos"]))
            out[f"sens_{short}_{c}_ci"] = interval(
                float(plain["coverage_pos_low"]), float(plain["coverage_pos_high"])
            )
            measured = clinical["arms"][arm][context]
            plain_o = measured["outcomes"]["plain"]
            perlabel = measured["outcomes"]["perlabel"]
            out[f"caught_{short}_{c}"] = per100(plain_o["ill"]["right_alone"]["share"])
            out[f"missed_{short}_{c}"] = per100(plain_o["ill"]["wrong_alone"]["share"])
            out[f"flagged_{short}_{c}"] = per100(plain_o["healthy"]["wrong_alone"]["share"])
            out[f"pl_alone_{short}_{c}"] = per100(perlabel["ill"]["right_alone"]["share"])
            out[f"pl_def_{short}_{c}"] = per100(perlabel["ill"]["deferred"]["share"])
            out[f"pl_hdef_{short}_{c}"] = per100(perlabel["healthy"]["deferred"]["share"])
            out[f"pl_hflag_{short}_{c}"] = per100(perlabel["healthy"]["wrong_alone"]["share"])
            out[f"def_{short}_{c}"] = pct(measured["label_free"]["share_deferred"])
            out[f"lf_flag_{short}_{c}"] = pct(measured["label_free"]["share_flagged"])
        out[f"auroc_{short}_all"] = (
            f"{transfer['arms'][arm]['auroc_test_all_contexts'][COMPOSITE]['auroc']:.3f}"
        )
    out["lf_flag_in"], out["lf_flag_out"] = out["lf_flag_resnet_in"], out["lf_flag_resnet_out"]
    for arm in STRONGEST:
        floor, expected = label_free_bounds(clinical["arms"][arm])
        out[f"lf_floor_{SHORT[arm]}"] = pct(floor["flagged"])
        out[f"lf_expect_{SHORT[arm]}"] = pct(expected["flagged"])
        out[f"def_floor_{SHORT[arm]}"] = pct(floor["deferred"])
        out[f"def_expect_{SHORT[arm]}"] = pct(expected["deferred"])
    (prior,) = [
        c
        for c in read("repairs.json")["cells"]
        if (c["family"], c["model"], c["target"], c["label"])
        == ("echonext", "resnet", "outpatient", COMPOSITE)
    ]
    out["prior_prev_estimated"] = pct(prior["prevalence_estimated"])
    out["prior_prev_eval"] = pct(prior["prevalence_eval"], 0)
    caught = sorted({out[f"caught_{SHORT[a]}_out"] for a in STRONGEST}, key=int)
    out["caught_range"] = caught[0] if len(caught) == 1 else f"{caught[0]} to {caught[-1]}"
    for label, key in (
        ("lvef_lte_45_flag", "lvef"),
        ("lvwt_gte_13_flag", "lvwt"),
        ("aortic_stenosis_moderate_or_greater_flag", "as"),
    ):
        for context, c in (("inpatient", "in"), ("outpatient", "out")):
            out[f"sens_{key}_{c}"] = pct(
                float(cell("resnet", context, "plain", label)["coverage_pos"])
            )

    resnet_out = clinical["arms"]["resnet"]["outpatient"]
    bedside = {round(b["prevalence"], 3): b for b in resnet_out["bedside"]}
    own = resnet_out["bedside"][0]
    out["ppv_resnet_out"], out["npv_resnet_out"] = pct(own["ppv"]), pct(own["npv"])
    out["ppv_resnet_10"], out["npv_resnet_10"] = pct(bedside[0.1]["ppv"]), pct(bedside[0.1]["npv"])
    out["ppv_resnet_5"], out["npv_resnet_5"] = pct(bedside[0.05]["ppv"]), pct(bedside[0.05]["npv"])
    per_k = own["per_thousand"]
    out["k_flagged_out"] = count(per_k["flagged"])
    out["k_found_out"] = count(per_k["ill_found"])
    out["k_missed_out"] = count(per_k["ill_missed"])
    out["k_ill_out"] = count(per_k["ill_found"] + per_k["ill_missed"])

    roc = clinical["roc"]
    at90 = roc["resnet"]["sensitivity"].index(0.9)
    for arm in STRONGEST:
        for context, c in (("inpatient", "in"), ("outpatient", "out")):
            out[f"oracle_spec_{SHORT[arm]}_{c}"] = pct(roc[arm][context][at90])
            out[f"oracle_flag_{SHORT[arm]}_{c}"] = per100(1 - roc[arm][context][at90])
    spec_out = resnet_out["specificity"]
    out["extra_missed"] = per100(0.9 - resnet_out["sensitivity"])
    out["extra_cleared"] = per100(spec_out - roc["resnet"]["outpatient"][at90])

    paired = clinical["paired"]["outpatient_sensitivity"]
    out["diff_resnet_mini"] = (
        f"{abs(100 * paired['resnet-echonext_mini']['difference']):.1f} points"
    )
    bound = max(max(abs(d["low"]), abs(d["high"])) for d in paired.values())
    out["diff_bound"] = f"{100 * bound:.1f} points"

    for arm in ARMS_WITH_LADDER:
        rows = {r["labels"]: r for r in clinical["ladder"][arm]}
        s = SHORT[arm]
        for rung in (0, 100, 200):
            out[f"lad{rung}_sens_{s}"] = pct(rows[rung]["sensitivity"]["mean"])
            out[f"lad{rung}_spec_{s}"] = pct(rows[rung]["specificity"]["mean"])
        out[f"lad100_ill_{s}"] = f"{round(rows[100]['ill_in_sample']['mean'])}"
        out[f"lad100_hflag_{s}"] = per100(1 - rows[100]["specificity"]["mean"])
        out[f"lad200_hflag_{s}"] = per100(1 - rows[200]["specificity"]["mean"])
        out[f"lad100_missed_{s}"] = per100(1 - rows[100]["sensitivity"]["mean"])
        out[f"lad100_below_{s}"] = pct(rows[100]["share_of_draws_below_level"])
        out[f"lad200_below_{s}"] = pct(rows[200]["share_of_draws_below_level"])
        out[f"lad200_ill_{s}"] = f"{round(rows[200]['ill_in_sample']['mean'])}"
        for rung in (100, 200):
            out[f"lad{rung}_p10_{s}"] = pct(rows[rung]["sensitivity"]["p10"])
            out[f"lad{rung}_p90_{s}"] = pct(rows[rung]["sensitivity"]["p90"])
    out["lad_draws"] = count(clinical["ladder"]["resnet"][0]["draws"])
    for rung in (100, 200):
        shares = sorted(
            r["share_of_draws_below_level"]
            for arm in STRONGEST
            for r in clinical["ladder"][arm]
            if r["labels"] == rung
        )
        out[f"lad{rung}_below_range"] = f"{pct(shares[0])} to {pct(shares[-1])}"
    rows = {r["labels"]: r for r in clinical["ladder"]["resnet"]}
    out["lad100_ill_min"] = f"{round(rows[100]['ill_in_sample']['min'])}"
    out["lad100_ill_max"] = f"{round(rows[100]['ill_in_sample']['max'])}"
    prevalence = own["prevalence"]
    sens100, spec100 = rows[100]["sensitivity"]["mean"], rows[100]["specificity"]["mean"]
    out["k_flagged_lad100"] = count(
        1000 * (sens100 * prevalence + (1 - spec100) * (1 - prevalence))
    )
    out["k_missed_lad100"] = count(1000 * (1 - sens100) * prevalence)

    for arm in STRONGEST:
        mix = clinical["case_mix"][arm]
        s = SHORT[arm]
        out[f"mix_share_{s}"] = pct(mix["share_explained"]["estimate"], 0)
        out[f"mix_share_{s}_ci"] = (
            f"{pct(mix['share_explained']['low'], 0)} to {pct(mix['share_explained']['high'], 0)}"
        )
        out[f"mix_cov_{s}"] = pct(mix["outpatient_at_inpatient_mix"]["estimate"])
        observed_in = mix["observed_inpatient"]["estimate"]
        observed_out = mix["observed_outpatient"]["estimate"]
        at_mix = mix["outpatient_at_inpatient_mix"]["estimate"]
        out[f"mix_gap_pts_{s}"] = f"{100 * (observed_in - observed_out):.0f}"
        out[f"mix_pts_{s}"] = f"{100 * (at_mix - observed_out):.0f}"

    tests = clinical["subgroups"]["tests"]
    by = {(t["arm"], t["kind"]): t for t in tests}
    sex, age = by[("resnet", "sex")], by[("resnet", "age")]
    for key, group in (("women", "female"), ("men", "male")):
        g = sex["groups"][group]
        out[f"sub_{key}"] = pct(g["caught"] / g["n"])
        out[f"sub_{key}_n"] = f"{g['caught']} of {g['n']}"
    for key, group in (("young", "18-49"), ("old", "80+")):
        g = age["groups"][group]
        out[f"sub_{key}"] = pct(g["caught"] / g["n"])
        out[f"sub_{key}_n"] = f"{g['caught']} of {g['n']}"
    out["p_sex_resnet"] = p_value(sex["p_holm"])
    out["p_age_resnet"] = p_value(age["p_holm"])
    out["p_max_others"] = p_value(max(t["p_holm"] for t in tests if t["arm"] != "resnet"))

    drops = [
        100
        * (
            float(cell("resnet", "inpatient", "plain", label)["coverage_pos"])
            - float(cell("resnet", "outpatient", "plain", label)["coverage_pos"])
        )
        for label in (
            "mitral_regurgitation_moderate_or_greater_flag",
            "tricuspid_regurgitation_moderate_or_greater_flag",
            "rv_systolic_dysfunction_moderate_or_greater_flag",
        )
    ]
    out["drop_other_range"] = f"{min(drops):.0f} to {max(drops):.0f}"
    out["pooled_resnet_out"] = pct(float(cell("resnet", "outpatient", "pooled")["coverage_pos"]))
    cuts = clinical["arms"]["resnet"]["thresholds"]
    out["cut_ill"], out["cut_healthy"] = f"{cuts['ill']:.2f}", f"{cuts['healthy']:.2f}"
    mix = clinical["case_mix"]["resnet"]
    out["mix_n_in"], out["mix_n_out"] = count(mix["n_inpatient"]), count(mix["n_outpatient"])
    for arm in STRONGEST:
        fixed = transfer["arms"][arm]["ladder"]["outpatient"][COMPOSITE]
        (at100,) = [r for r in fixed if r["labels"] == 100]
        out[f"fixed_lad100_{SHORT[arm]}"] = pct(at100["coverage_pos_mean"])

    for context, c in CONTEXT.items():
        u = clinical["unmeasured"][context]
        out[f"unmeas_{c}"] = pct(u["n_healthy_unmeasured"] / u["n_healthy"])
        out[f"unmeas_n_{c}"] = f"{u['n_healthy_unmeasured']:,} of {u['n_healthy']:,}"
        out[f"unmeas_prev_{c}"] = pct(u["prevalence_measured"])
        for arm in STRONGEST:
            a = u["arms"][arm]
            out[f"unmeas_flag_{SHORT[arm]}_{c}"] = per100(1 - a["specificity_measured"])
            out[f"unmeas_auroc_{SHORT[arm]}_{c}"] = f"{a['auroc_measured']:.3f}"

    variants = clinical["calibration_variants"]
    every, outside = variants["every_setting"], variants["outpatients"]
    out["va_n"], out["va_ill"] = count(every["n"]), count(every["n_ill"])
    out["vo_n"], out["vo_ill"] = count(outside["n"]), count(outside["n_ill"])
    ecgs = clinical["flow"]["ecgs"]["val"]
    out["val_in_share"] = pct(ecgs["inpatient"] / sum(ecgs.values()), 0)
    for arm in STRONGEST:
        s = SHORT[arm]
        for context, c in (("inpatient", "in"), ("outpatient", "out")):
            sens = every["arms"][arm][context]["sensitivity"]
            out[f"va_sens_{s}_{c}"] = pct(sens["share"])
            out[f"va_sens_{s}_{c}_ci"] = interval(sens["low"], sens["high"])
        out[f"va_caught_{s}_out"] = per100(every["arms"][arm]["outpatient"]["sensitivity"]["share"])
        held = outside["arms"][arm]["outpatient"]
        out[f"vo_sens_{s}"] = pct(held["sensitivity"]["share"])
        out[f"vo_sens_{s}_ci"] = interval(held["sensitivity"]["low"], held["sensitivity"]["high"])
        out[f"vo_hflag_{s}"] = per100(1 - held["specificity"]["share"])
        drawn = {r["labels"]: r for r in clinical["ladder_validation"][arm]}
        for rung in (100, 200):
            out[f"vl{rung}_sens_{s}"] = pct(drawn[rung]["sensitivity"]["mean"])
            out[f"vl{rung}_below_{s}"] = pct(drawn[rung]["share_of_draws_below_level"])
        out[f"vl100_hflag_{s}"] = per100(1 - drawn[100]["specificity"]["mean"])
    for rung in (100, 200):
        shares = sorted(
            {r["labels"]: r for r in clinical["ladder_validation"][arm]}[rung][
                "share_of_draws_below_level"
            ]
            for arm in STRONGEST
        )
        out[f"vl{rung}_below_range"] = f"{pct(shares[0])} to {pct(shares[-1])}"
    flags = sorted(1 - outside["arms"][a]["outpatient"]["specificity"]["share"] for a in STRONGEST)
    out["vo_hflag_range"] = f"{per100(flags[0])} to {per100(flags[-1])}"
    choice = clinical["decision"]["resnet"]
    options, prevalence = choice["options"], choice["prevalence"]
    found = {k: 1000 * o["sensitivity"] * prevalence for k, o in options.items()}
    extra_echo = (
        options["refit_100"]["flagged_per_thousand"]
        - options["inpatient_threshold"]["flagged_per_thousand"]
    )
    out["echo_per_extra_ill"] = (
        f"{extra_echo / (found['refit_100'] - found['inpatient_threshold']):.0f}"
    )
    option_keys = {
        "inpatient_threshold": "in",
        "refit_100": "refit",
        "validation_outpatients": "vo",
        "every_diagnosis_known_90": "oracle",
        "echo_for_all": "all",
    }
    for name, key in option_keys.items():
        for t, value in options[name]["net_benefit"].items():
            out[f"nb{round(100 * float(t))}_{key}"] = count(1000 * value)
    plain = clinical["refit_without_margin"]["resnet"]
    out["plain_sens"] = pct(plain["sensitivity"]["mean"])
    out["plain_hflag"] = per100(1 - plain["specificity"]["mean"])
    out["plain_below"] = pct(plain["share_of_draws_below_level"])
    floor_rows = {r["labels"]: r for r in clinical["ladder"]["random_init"]}
    out["lad100_sens_floor"] = pct(floor_rows[100]["sensitivity"]["mean"])
    out["lad100_hflag_floor"] = per100(1 - floor_rows[100]["specificity"]["mean"])
    gaps = [
        {r["labels"]: r for r in clinical["ladder"][a]}[100]["specificity"]["mean"]
        - floor_rows[100]["specificity"]["mean"]
        for a in STRONGEST
    ]
    out["floor_gap"] = f"{per100(min(gaps))} to {per100(max(gaps))}"
    floor_out = outside["arms"]["random_init"]["outpatient"]
    out["vo_sens_floor"] = pct(floor_out["sensitivity"]["share"])
    out["vo_hflag_floor"] = per100(1 - floor_out["specificity"]["share"])

    sev = severity["severity"]
    out["sev_findings_in"] = f"{sev['findings_count']['inpatient']['median']:g}"
    out["sev_findings_out"] = f"{sev['findings_count']['outpatient']['median']:g}"
    out["sev_lvef_in"] = f"{sev['lvef_value']['inpatient']['median']:.1f}%"
    out["sev_lvef_out"] = f"{sev['lvef_value']['outpatient']['median']:.1f}%"
    return out


ARMS_WITH_LADDER = STRONGEST


def label_free_bounds(arm: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """What fewer ill patients alone could do to the shares a clinic sees without diagnoses.

    With the inpatients' rates held, the share flagged is ``p * sens + (1 - p) * (1 - spec)``
    at prevalence ``p``: a weighted mean of the two rates, so it never falls below the
    smaller one whatever ``p`` is.  The same holds for the share sent to a reader, with
    the ill and healthy inpatients' deferral rates.  Returns that floor, and the shares
    expected at the outpatients' own prevalence.
    """
    inside, outside = arm["inpatient"], arm["outpatient"]
    perlabel = inside["outcomes"]["perlabel"]
    rates = {
        "flagged": (inside["sensitivity"], 1 - inside["specificity"]),
        "deferred": (
            perlabel["ill"]["deferred"]["share"],
            perlabel["healthy"]["deferred"]["share"],
        ),
    }
    p = outside["n_ill"] / outside["n"]
    floor = {k: min(ill, healthy) for k, (ill, healthy) in rates.items()}
    expected = {k: p * ill + (1 - p) * healthy for k, (ill, healthy) in rates.items()}
    return floor, expected


def infarction() -> dict[str, str]:
    sites = read("infarction_sites.json")["sites"]
    outcomes = read("outcomes.json")["by_corpus"]
    out: dict[str, str] = {}
    for corpus, s in sites.items():
        out[f"mi_n_{corpus}"] = count(s["n"])
        out[f"mi_pos_{corpus}"] = count(s["n_mi"])
        out[f"mi_n_read_{corpus}"] = count(s["n_read_per_draw"])
        out[f"mi_pos_read_{corpus}"] = count(s["n_mi_read_per_draw"])
        out[f"mi_auroc_{corpus}"] = f"{s['auroc']['auroc']:.3f}"
        out[f"mi_sens_{corpus}"] = pct(s["sensitivity"])
        out[f"mi_sens_{corpus}_ci"] = interval(*s["sensitivity_interval"])
        out[f"mi_spec_{corpus}"] = pct(s["specificity"])
        plain = outcomes[corpus]["schemes"]["plain"]
        out[f"mi_caught_{corpus}"] = per100(plain["1"]["correct"]["mean"])
        out[f"mi_flagged_{corpus}"] = per100(plain["0"]["wrong"]["mean"])
    return out


def ppv() -> dict[str, str]:
    every = read("echonext_ppv_gap.json")["all_families"]
    transfer, control = every["transfer"], every["in_distribution"]
    return {
        "ppv_gap_median": f"{100 * transfer['median_abs_gap_points']:.1f}",
        "ppv_gap_cells": f"{transfer['cells']}",
        "ppv_control_median": f"{100 * control['median_abs_gap_points']:.1f}",
        "ppv_control_cells": f"{control['cells']}",
    }


@cache
def values() -> dict[str, str]:
    figures = echonext() | infarction() | ppv()
    assert all(isinstance(v, str) and v and "nan" not in v for v in figures.values()), [
        k for k, v in figures.items() if not v or "nan" in v
    ]
    assert not any(math.isnan(float(v)) for v in figures.values() if v.replace(".", "").isdigit())
    return figures


ARM_NAMES = {
    "resnet": "Network trained here",
    "echonext_mini": "EchoNext mini-model",
    "ecgfounder": "ECGFounder",
    "random_init": "Untrained floor",
}
CONTEXT_NAMES = {"inpatient": "Inpatients", "emergency": "Emergency", "outpatient": "Outpatients"}
FINDING_NAMES = {
    "lvef_lte_45_flag": "Ejection fraction 45% or less",
    "lvwt_gte_13_flag": "Wall thickness 1.3 cm or more",
    "aortic_stenosis_moderate_or_greater_flag": "Aortic stenosis",
    "aortic_regurgitation_moderate_or_greater_flag": "Aortic regurgitation",
    "mitral_regurgitation_moderate_or_greater_flag": "Mitral regurgitation",
    "tricuspid_regurgitation_moderate_or_greater_flag": "Tricuspid regurgitation",
    "pulmonary_regurgitation_moderate_or_greater_flag": "Pulmonary regurgitation",
    "rv_systolic_dysfunction_moderate_or_greater_flag": "Right ventricular dysfunction",
    "pericardial_effusion_moderate_large_flag": "Pericardial effusion",
    "pasp_gte_45_flag": "Pulmonary artery pressure 45 mmHg or more",
    "tr_max_gte_32_flag": "Tricuspid peak velocity 3.2 m/s or more",
    COMPOSITE: "Any of the eleven",
}


def _ci(row: dict[str, str], key: str) -> str:
    return (
        f"{pct(float(row[key]))} ({interval(float(row[key + '_low']), float(row[key + '_high']))})"
    )


def rows_settings() -> list[str]:
    transfer = read("echonext_transfer.json")
    out = []
    for arm, name in ARM_NAMES.items():
        for context, cname in CONTEXT_NAMES.items():
            plain, perlabel = cell(arm, context, "plain"), cell(arm, context, "perlabel")
            a = transfer["arms"][arm]["auroc"][context][COMPOSITE]
            out.append(
                f"| {name} | {cname} | {a['auroc']:.3f} ({a['low']:.3f} to {a['high']:.3f}) | "
                f"{_ci(plain, 'coverage_pos')} | {_ci(plain, 'coverage_neg')} | "
                f"{pct(float(perlabel['abstention']))} |"
            )
    return out


def rows_findings() -> list[str]:
    out = []
    for label, name in FINDING_NAMES.items():
        inside, outside = (
            cell("resnet", "inpatient", "plain", label),
            cell("resnet", "outpatient", "plain", label),
        )
        if int(outside["n_pos"]) == 0:
            out.append(
                f"| {name} | {int(inside['n_pos']):,} | "
                f"{pct(float(inside['coverage_pos']))} | 0 | none ill |"
            )
            continue
        out.append(
            f"| {name} | {int(inside['n_pos']):,} | {pct(float(inside['coverage_pos']))} | "
            f"{int(outside['n_pos']):,} | {_ci(outside, 'coverage_pos')} |"
        )
    return out


def rows_ladder() -> list[str]:
    clinical = read("echonext_clinical.json")
    out = []
    for arm in STRONGEST:
        for row in clinical["ladder"][arm]:
            ill = row.get("ill_in_sample")
            held = (
                "none"
                if ill is None
                else f"{ill['mean']:.1f} ({ill['p10']:.0f} to {ill['p90']:.0f})"
            )
            sens, spec = row["sensitivity"], row["specificity"]
            out.append(
                f"| {ARM_NAMES[arm]} | {row['labels']} | {held} | "
                f"{pct(sens['mean'])} ({pct(sens['p10'])} to {pct(sens['p90'])}) | "
                f"{pct(row['share_of_draws_below_level'])} | "
                f"{pct(spec['mean'])} ({pct(spec['p10'])} to {pct(spec['p90'])}) | "
                f"{pct(row['deferred_all']['mean'])} |"
            )
    return out


def rows_subgroup_tests() -> list[str]:
    out = []
    for t in read("echonext_clinical.json")["subgroups"]["tests"]:
        groups = "; ".join(
            f"{g} {v['caught']} of {v['n']} ({pct(v['caught'] / v['n'])})"
            for g, v in t["groups"].items()
        )
        test = "Fisher's exact" if t["test"] == "fisher_exact" else "chi-square"
        out.append(
            f"| {ARM_NAMES[t['arm']]} | {t['kind']} | {groups} | {test} | "
            f"{p_value(t['p'])} | {p_value(t['p_holm'])} |"
        )
    return out


def rows_bedside() -> list[str]:
    out = []
    for arm in STRONGEST:
        for b in read("echonext_clinical.json")["arms"][arm]["outpatient"]["bedside"]:
            k = b["per_thousand"]
            out.append(
                f"| {ARM_NAMES[arm]} | {pct(b['prevalence'])} | "
                f"{pct(b['ppv'])} | {pct(b['npv'])} | "
                f"{count(k['flagged'])} | {count(k['ill_found'])} | {count(k['ill_missed'])} |"
            )
    return out


def rows_case_mix() -> list[str]:
    out = []
    for arm in STRONGEST:
        m = read("echonext_clinical.json")["case_mix"][arm]

        def show(key: str, m: dict[str, Any] = m) -> str:
            v = m[key]
            return f"{pct(v['estimate'])} ({interval(v['low'], v['high'])})"

        out.append(
            f"| {ARM_NAMES[arm]} | {show('observed_inpatient')} | {show('observed_outpatient')} | "
            f"{show('outpatient_at_inpatient_mix')} | {show('share_explained')} |"
        )
    return out


def rows_unmeasured() -> list[str]:
    clinical = read("echonext_clinical.json")
    out = []
    for context, cname in CONTEXT_NAMES.items():
        u = clinical["unmeasured"][context]
        measured = clinical["arms"]["resnet"][context]
        for arm in STRONGEST:
            a, every = u["arms"][arm], clinical["arms"][arm][context]
            auroc = read("echonext_transfer.json")["arms"][arm]["auroc"][context][COMPOSITE]
            out.append(
                f"| {ARM_NAMES[arm]} | {cname} | "
                f"{u['n_healthy_unmeasured']:,} of {u['n_healthy']:,} | "
                f"{pct(measured['n_ill'] / measured['n'])} to {pct(u['prevalence_measured'])} | "
                f"{pct(every['specificity'])} to {pct(a['specificity_measured'])} | "
                f"{auroc['auroc']:.3f} to {a['auroc_measured']:.3f} |"
            )
    return out


VARIANT_NAMES = {
    "inpatients": "Validation inpatients (the report's threshold)",
    "every_setting": "Every validation patient",
    "outpatients": "Validation outpatients",
}


def rows_variants() -> list[str]:
    out = []
    for name, entry in read("echonext_clinical.json")["calibration_variants"].items():
        for arm in ARM_NAMES:
            a = entry["arms"][arm]
            sens, spec = a["outpatient"]["sensitivity"], a["outpatient"]["specificity"]
            out.append(
                f"| {VARIANT_NAMES[name]}, {entry['n']:,} ({entry['n_ill']:,} ill) | "
                f"{ARM_NAMES[arm]} | "
                f"{pct(sens['share'])} ({interval(sens['low'], sens['high'])}) | "
                f"{pct(spec['share'])} ({interval(spec['low'], spec['high'])}) | "
                f"{pct(a['inpatient']['sensitivity']['share'])} |"
            )
    return out


def rows_ladder_validation() -> list[str]:
    out = []
    for arm in STRONGEST:
        for row in read("echonext_clinical.json")["ladder_validation"][arm]:
            ill, sens, spec = row["ill_in_sample"], row["sensitivity"], row["specificity"]
            out.append(
                f"| {ARM_NAMES[arm]} | {row['labels']} | "
                f"{ill['mean']:.1f} ({ill['p10']:.0f} to {ill['p90']:.0f}) | "
                f"{pct(sens['mean'])} ({pct(sens['p10'])} to {pct(sens['p90'])}) | "
                f"{pct(row['share_of_draws_below_level'])} | "
                f"{pct(spec['mean'])} ({pct(spec['p10'])} to {pct(spec['p90'])}) |"
            )
    return out


OPTION_NAMES = {
    "inpatient_threshold": "Inpatient threshold",
    "refit_100": "Set again on 100 outpatients",
    "refit_200": "Set again on 200 outpatients",
    "validation_outpatients": "Set on the validation outpatients",
    "every_diagnosis_known_90": "90% with every diagnosis known",
    "echo_for_all": "Echocardiogram for every outpatient",
    "echo_for_none": "Echocardiogram for none",
}


def rows_decision() -> list[str]:
    out = []
    for arm in STRONGEST:
        for name, o in read("echonext_clinical.json")["decision"][arm]["options"].items():
            nb = " | ".join(count(1000 * o["net_benefit"][t]) for t in ("0.05", "0.10", "0.20"))
            out.append(
                f"| {ARM_NAMES[arm]} | {OPTION_NAMES[name]} | {pct(o['sensitivity'])} | "
                f"{count(o['flagged_per_thousand'])} | {nb} |"
            )
    return out


def rows_paired() -> list[str]:
    out = []
    for pair, d in read("echonext_clinical.json")["paired"]["outpatient_sensitivity"].items():
        a, b = pair.split("-")
        out.append(
            f"| {ARM_NAMES[a]} minus {ARM_NAMES[b]} | {100 * d['difference']:+.1f} | "
            f"{100 * d['low']:+.1f} to {100 * d['high']:+.1f} |"
        )
    return out


ROWS = {
    "settings": rows_settings,
    "findings": rows_findings,
    "ladder": rows_ladder,
    "subgroup_tests": rows_subgroup_tests,
    "bedside": rows_bedside,
    "case_mix": rows_case_mix,
    "paired": rows_paired,
    "unmeasured": rows_unmeasured,
    "variants": rows_variants,
    "ladder_validation": rows_ladder_validation,
    "decision": rows_decision,
}
