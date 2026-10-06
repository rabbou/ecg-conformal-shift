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


def largest_remainder(shares: list[float]) -> list[int]:
    """Shares that sum to one, as whole counts per 100 that sum to 100: each is rounded
    down, and the points left go to the largest remainders.  Table 1's rows then add up
    to the totals they split."""
    exact = [100 * s for s in shares]
    counts = [int(x) for x in exact]
    order = sorted(range(len(exact)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in order[: 100 - sum(counts)]:
        counts[i] += 1
    return counts


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
    out["cal_ill"] = count(clinical["calibration"]["n_ill"])
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
            out[f"spec_{short}_{c}"] = pct(float(plain["coverage_neg"]))
            out[f"spec_{short}_{c}_ci"] = interval(
                float(plain["coverage_neg_low"]), float(plain["coverage_neg_high"])
            )
            measured = clinical["arms"][arm][context]
            plain_o = measured["outcomes"]["plain"]
            perlabel = measured["outcomes"]["perlabel"]
            out[f"caught_{short}_{c}"] = per100(plain_o["ill"]["right_alone"]["share"])
            out[f"missed_{short}_{c}"] = per100(plain_o["ill"]["wrong_alone"]["share"])
            out[f"flagged_{short}_{c}"] = per100(plain_o["healthy"]["wrong_alone"]["share"])
            keys = ("right_alone", "deferred", "wrong_alone")
            ill = largest_remainder([perlabel["ill"][k]["share"] for k in keys])
            healthy = largest_remainder([perlabel["healthy"][k]["share"] for k in keys])
            out[f"pl_alone_{short}_{c}"], out[f"pl_def_{short}_{c}"] = f"{ill[0]}", f"{ill[1]}"
            out[f"pl_hdef_{short}_{c}"] = f"{healthy[1]}"
            out[f"pl_hflag_{short}_{c}"] = f"{healthy[2]}"
            out[f"n_ill_{short}_{c}"] = count(measured["n_ill"])
            out[f"n_healthy_{short}_{c}"] = count(measured["n"] - measured["n_ill"])
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
    flags90 = sorted(1 - roc[arm]["outpatient"][at90] for arm in STRONGEST)
    out["oracle_flag_range_out"] = f"{per100(flags90[0])} to {per100(flags90[-1])}"
    demo = clinical["age_sex"]
    out["as_sens_in"] = pct(demo["inpatient"]["sensitivity"])
    out["as_sens_out"] = pct(demo["outpatient"]["sensitivity"])
    out["as_flag_out"] = per100(1 - demo["outpatient"]["specificity"])
    out["as_auroc_out"] = f"{demo['outpatient']['auroc']:.3f}"
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
        out[f"mix_share_{s}"] = pct(mix["share_explained"]["estimate"])
        out[f"mix_share_{s}_ci"] = (
            f"{pct(mix['share_explained']['low'], 0)} to {pct(mix['share_explained']['high'], 0)}"
        )
        out[f"mix_cov_{s}"] = pct(mix["outpatient_at_inpatient_mix"]["estimate"])
        g = mix["share_explained_predicted"]
        out[f"mix_g_{s}"] = pct(g["estimate"])
        out[f"mix_g_{s}_ci"] = f"{pct(g['low'], 0)} to {pct(g['high'], 0)}"
        observed_in = mix["observed_inpatient"]["estimate"]
        observed_out = mix["observed_outpatient"]["estimate"]
        at_mix = mix["outpatient_at_inpatient_mix"]["estimate"]
        out[f"mix_gap_pts_{s}"] = f"{100 * (observed_in - observed_out):.1f}"
        out[f"mix_pts_{s}"] = f"{100 * (at_mix - observed_out):.1f}"

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
    beside = clinical["subgroups"]["healthy_and_auroc"]["resnet"]
    for key, kind, group in (
        ("women", "sex", "female"),
        ("men", "sex", "male"),
        ("young", "age", "18-49"),
        ("old", "age", "80+"),
    ):
        g = beside[kind][group]
        out[f"sp_{key}"] = pct(g["specificity"])
        out[f"au_{key}"] = f"{g['auroc']['auroc']:.3f}"
        out[f"au_{key}_ci"] = f"{g['auroc']['low']:.3f} to {g['auroc']['high']:.3f}"
    race = clinical["subgroups"]["race_ethnicity"]["resnet"]
    big = [g for g, c in race["outpatient"].items() if c["n"] >= 20]
    for context, c in (("outpatient", "out"), ("inpatient", "in")):
        shares = [race[context][g]["caught"] / race[context][g]["n"] for g in big]
        out[f"race_{c}_range"] = f"{pct(min(shares))} to {pct(max(shares))}"
    out["race_groups"] = count(len(big))
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
        means = clinical["half_means"][arm]["means"]
        below = sum(m < at100["coverage_pos_mean"] for m in means) / len(means)
        out[f"fixed_rank_{SHORT[arm]}"] = pct(below)
        out[f"half_low_{SHORT[arm]}"] = pct(means[round(0.025 * (len(means) - 1))])
        out[f"half_high_{SHORT[arm]}"] = pct(means[round(0.975 * (len(means) - 1))])
    out["half_cuts"] = count(clinical["half_means"]["resnet"]["halves"])

    for context, c in CONTEXT.items():
        u = clinical["unmeasured"][context]
        out[f"unmeas_{c}"] = pct(u["n_healthy_unmeasured"] / u["n_healthy"])
        out[f"unmeas_n_{c}"] = f"{u['n_healthy_unmeasured']:,} of {u['n_healthy']:,}"
        out[f"unmeas_prev_{c}"] = pct(u["prevalence_measured"])
        for arm in STRONGEST:
            a = u["arms"][arm]
            out[f"unmeas_flag_{SHORT[arm]}_{c}"] = per100(1 - a["specificity_measured"])
            out[f"unmeas_auroc_{SHORT[arm]}_{c}"] = f"{a['auroc_measured']:.3f}"

    found = clinical["by_finding"]["resnet"]
    for label, key in (
        ("lvef_lte_45_flag", "lvef"),
        ("lvwt_gte_13_flag", "lvwt"),
        ("aortic_stenosis_moderate_or_greater_flag", "as"),
        ("mitral_regurgitation_moderate_or_greater_flag", "mr"),
        ("tricuspid_regurgitation_moderate_or_greater_flag", "tr"),
        ("pasp_gte_45_flag", "pasp"),
    ):
        for context, c in (("inpatient", "in"), ("outpatient", "out")):
            f = found[context][label]
            out[f"bf_{key}_{c}"] = pct(f["caught"] / f["n"])
            out[f"bf_{key}_{c}_n"] = f"{f['caught']} of {f['n']}"
    for context, c in (("inpatient", "in"), ("outpatient", "out")):
        w = found[context]["without_wall_alone"]
        out[f"nowall_{c}"] = pct(w["caught"] / w["n"])
    missed = found["missed_outpatients"]
    out["missed_n"] = count(missed["n"])
    out["missed_wall_alone"] = count(missed["one_finding"]["lvwt_gte_13_flag"])
    out["missed_one_finding"] = count(sum(missed["one_finding"].values()))
    out["missed_severe"] = count(missed["severe"])

    flow = clinical["flow"]
    out["flow_nosplit"] = count(sum(flow["ecgs"]["no_split"].values()))
    out["flow_val"] = count(sum(flow["ecgs"]["val"].values()))
    out["flow_test"] = count(sum(flow["ecgs"]["test"].values()))
    out["train_patients"] = count(flow["patients"]["train"])
    for split in ("val", "test"):
        out[f"flow_{split}_proc"] = count(flow["ecgs"][split]["procedural"])
        out[f"flow_{split}_em"] = count(flow["ecgs"][split]["emergency"])
        out[f"flow_{split}_out"] = count(flow["ecgs"][split]["outpatient"])
    out["age_min"] = f"{min(flow['youngest'][s] for s in ('train', 'val', 'test'))}"
    years = clinical["eras"]["years"]
    out["year_min"] = f"{min(y['min'] for y in years.values())}"
    out["year_max"] = f"{max(y['max'] for y in years.values())}"
    out["year_out"] = f"{years['outpatient']['median']:.0f}"
    out["year_in"] = f"{years['inpatient']['median']:.0f}"
    band_sens = [
        r["share"] for arm in STRONGEST for r in clinical["eras"]["arms"][arm]["outpatient"]
    ]
    out["era_out_range"] = f"{pct(min(band_sens))} to {pct(max(band_sens))}"
    band_in = [r["share"] for arm in STRONGEST for r in clinical["eras"]["arms"][arm]["inpatient"]]
    out["era_in_range"] = f"{pct(min(band_in))} to {pct(max(band_in))}"

    for arm in STRONGEST:
        spread = clinical["threshold_spread"][arm]
        for context, c in (("inpatient", "in"), ("outpatient", "out")):
            measured = clinical["arms"][arm][context]
            sens, n = measured["sensitivity"], measured["n_ill"]
            half = 1.959964 * math.sqrt(spread[context]["sd"] ** 2 + sens * (1 - sens) / n)
            out[f"ts_ci_{SHORT[arm]}_{c}"] = interval(sens - half, sens + half)
        out[f"unc_sens_{SHORT[arm]}_out"] = pct(spread["outpatient"]["uncorrected_sensitivity"])
    at72 = roc["resnet"]["sensitivity"].index(0.72)
    out["roc72_in"] = per100(1 - roc["resnet"]["inpatient"][at72])
    out["roc72_out"] = per100(1 - roc["resnet"]["outpatient"][at72])
    for arm in STRONGEST:
        for context, c in CONTEXT.items():
            a = transfer["arms"][arm]["auroc"][context][COMPOSITE]
            out[f"auroc_{SHORT[arm]}_{c}_ci"] = f"{a['low']:.3f} to {a['high']:.3f}"

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


PPV_BLOCKS = {
    "control": "Controls, same population",
    "transfer": "All transfers",
    "columbia_emergency": "Columbia, inpatients to emergency",
    "columbia_outpatient": "Columbia, inpatients to outpatients",
    "rotation": "Five corpora, each to the other four",
}
SHARES = (
    "share_within_two_points",
    "share_outside_observed_interval",
    "share_ratio_off_by_a_quarter_or_more",
    "share_recipe_too_high",
)


def ppv_rows() -> list[dict[str, Any]]:
    return [*read("ppv_gap.json")["rows"], *read("echonext_ppv_gap.json")["rows"]]


def ppv_row(**key: str) -> dict[str, Any]:
    (row,) = [r for r in ppv_rows() if all(r[k] == v for k, v in key.items())]
    return row


def worst_transfer() -> dict[str, Any]:
    kept = [r for r in ppv_rows() if r["summarised"] and not r["in_distribution"]]
    return max(kept, key=lambda r: abs(r["gap"]))


def share_ci(s: dict[str, float]) -> str:
    """A share across transfers, with its cluster-bootstrap interval."""
    return f"{pct(s['estimate'], 0)} ({pct(s['low'], 0)} to {pct(s['high'], 0)})"


def points_ci(s: dict[str, float]) -> str:
    return f"{100 * s['estimate']:.1f} ({100 * s['low']:.1f} to {100 * s['high']:.1f})"


def wilson_ci(w: dict[str, float], digits: int = 1) -> str:
    return f"{pct(w['share'], digits)} ({pct(w['low'], digits)} to {pct(w['high'], digits)})"


def observed_ci(row: dict[str, Any], digits: int = 1) -> str:
    return (
        f"{pct(row['ppv_observed'], digits)} "
        f"({pct(row['ppv_observed_low'], digits)} to {pct(row['ppv_observed_high'], digits)})"
    )


def ppv() -> dict[str, str]:
    every = read("echonext_ppv_gap.json")["all_families"]
    echo = read("echonext_ppv_gap.json")["summary"]["echonext"]["by_target"]
    blocks = read("ppv_intervals.json")["across_transfers"]
    columbia = read("ppv_intervals.json")["columbia"]
    transfer, control = blocks["transfer"], blocks["control"]
    assert transfer["cells"] == every["transfer"]["cells"]
    out = {
        "ppv_gap_cells": f"{transfer['cells']}",
        "ppv_control_cells": f"{control['cells']}",
        "ppv_clusters": f"{transfer['clusters']}",
        "ppv_control_clusters": f"{control['clusters']}",
        "ppv_gap_median": f"{100 * transfer['median_abs_gap_points']['estimate']:.1f}",
        "ppv_gap_median_ci": points_ci(transfer["median_abs_gap_points"]),
        "ppv_control_median": f"{100 * control['median_abs_gap_points']['estimate']:.1f}",
        "ppv_control_median_ci": points_ci(control["median_abs_gap_points"]),
        "ppv_not_summarised": f"{every['transfer']['cells_not_summarised']}",
        "ppv_rotation_cells": f"{blocks['rotation']['cells']}",
        "ppv_infarction_cells": f"{blocks['infarction']['cells']}",
        "ppv_columbia_cells": (
            f"{blocks['columbia_emergency']['cells'] + blocks['columbia_outpatient']['cells']}"
        ),
        "ppv_outside_share": pct(transfer["share_outside_observed_interval"]["estimate"], 0),
    }
    for key, short in zip(SHARES, ("within2", "outside", "quarter", "high"), strict=True):
        out[f"ppv_{short}"] = share_ci(transfer[key])
        out[f"ppv_control_{short}"] = share_ci(control[key])
        out[f"ppv_{short}_out"] = share_ci(blocks["columbia_outpatient"][key])
    for target, c in (("inpatient", "in"), ("emergency", "em"), ("outpatient", "out")):
        ratio = echo[target]["median_likelihood_ratio_healthy_target"]
        out[f"lr_healthy_{c}"] = f"{ratio:.2f}"

    shd = ppv_row(family="echonext", model="resnet", target="outpatient", label=COMPOSITE)
    lvef = ppv_row(family="echonext", model="resnet", target="outpatient", label="lvef_lte_45_flag")
    bedside = read("echonext_clinical.json")["arms"]["resnet"]["outpatient"]["bedside"][0]
    assert abs(shd["ppv_observed"] - bedside["ppv"]) < 1e-5
    out["col_ppv_obs_ci"] = observed_ci(shd)
    out["col_ppv_rec"] = pct(shd["ppv_recomputed"])
    out["col_gap"] = f"{abs(100 * shd['gap']):.1f}"
    out["col_gap_ci"] = f"{abs(100 * shd['gap_high']):.1f} to {abs(100 * shd['gap_low']):.1f}"
    out["col_sens_src"], out["col_sens_tgt"] = pct(shd["sens_source"]), pct(shd["sens_target"])
    out["col_spec_src"], out["col_spec_tgt"] = pct(shd["spec_source"]), pct(shd["spec_target"])
    out["col_fa_obs"] = f"{shd['false_alerts_observed']:.1f}"
    out["col_fa_rec"] = f"{shd['false_alerts_recomputed']:.1f}"
    out["lv_ppv_rec"] = pct(lvef["ppv_recomputed"])
    out["lv_ppv_obs_ci"] = observed_ci(lvef)
    out["lv_fa_obs"] = f"{lvef['false_alerts_observed']:.1f}"
    out["lv_fa_rec"] = f"{lvef['false_alerts_recomputed']:.1f}"
    for context, c in (("inpatient", "in"), ("outpatient", "out")):
        w = columbia[context]
        out[f"col_prev_{c}_ci"] = wilson_ci(w["prevalence"])
        out[f"lf_flag_{c}_ci"] = wilson_ci(w["share_flagged"])
        out[f"col_npv_{c}_ci"] = wilson_ci(w["npv"])

    for corpus in ("acs", "sph"):
        r = ppv_row(family="infarction", target=corpus)
        digits = 1 if r["ppv_observed"] < 0.1 else 0
        out[f"{corpus}_ppv_rec"] = pct(r["ppv_recomputed"], digits)
        out[f"{corpus}_ppv_obs_ci"] = observed_ci(r, digits)
        out[f"{corpus}_spec_src"] = pct(r["spec_source"], 0)
        out[f"{corpus}_spec_tgt"] = pct(r["spec_target"], 0)
        out[f"{corpus}_fa_obs"] = f"{r['false_alerts_observed']:.0f}"
        out[f"{corpus}_fa_rec"] = f"{r['false_alerts_recomputed']:.0f}"
    worst = worst_transfer()
    out["worst_rec"] = pct(worst["ppv_recomputed"], 0)
    out["worst_obs_ci"] = observed_ci(worst, 0)

    free = every["transfer"]["label_free_predictors"]
    out["free_recipe_est"] = f"{100 * free['recipe_estimated_prevalence']['median_abs_points']:.1f}"
    out["free_mean_prob"] = f"{100 * free['mean_probability']['median_abs_points']:.1f}"
    out["free_mean_prob_corr"] = (
        f"{100 * free['mean_probability_prior_corrected']['median_abs_points']:.1f}"
    )

    repairs = read("repairs.json")
    summary = repairs["summary"]["all"]
    out["rep_cells"] = f"{summary['cells']}"
    out["rep_draws"] = f"{repairs['draws']}"
    for t in ("0.05", "0.10", "0.20"):
        gain = summary[t]["mean_gain_per_1000"]["recalibrated"]
        out[f"rep_gain{round(100 * float(t))}"] = f"{gain:.1f}"
    cells = {(c["family"], c["model"], c["target"], c["label"]): c for c in repairs["cells"]}
    col = cells[("echonext", "resnet", "outpatient", COMPOSITE)]
    at10 = {r["threshold"]: r for r in col["net_benefit"]}[0.1]
    out["rep_col_eval"] = count(col["n_eval"])
    out["rep_col_model"] = count(1000 * at10["as_delivered"])
    out["rep_col_all"] = count(1000 * at10["treat_all"])
    sph = cells[("infarction", "ptbxl_baseline", "sph", "MI")]
    out["rep_sph_ill_per100"] = f"{round(100 * sph['prevalence_eval'])}"
    return out


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


def rows_subgroup_healthy() -> list[str]:
    out = []
    beside = read("echonext_clinical.json")["subgroups"]["healthy_and_auroc"]
    for arm in STRONGEST:
        for kind, groups in beside[arm].items():
            for group, g in groups.items():
                a = g["auroc"]
                out.append(
                    f"| {ARM_NAMES[arm]} | {kind} {group} | {g['healthy']:,} | "
                    f"{pct(g['specificity'])} "
                    f"({interval(g['specificity_low'], g['specificity_high'])}) | "
                    f"{a['auroc']:.3f} ({a['low']:.3f} to {a['high']:.3f}) |"
                )
    return out


def rows_race() -> list[str]:
    out = []
    race = read("echonext_clinical.json")["subgroups"]["race_ethnicity"]
    for arm in STRONGEST:
        for group in race[arm]["outpatient"]:
            cells = " | ".join(
                f"{race[arm][c][group]['caught']} of {race[arm][c][group]['n']}, "
                f"{pct(race[arm][c][group]['caught'] / race[arm][c][group]['n'])}"
                for c in ("inpatient", "outpatient")
            )
            out.append(f"| {ARM_NAMES[arm]} | {group} | {cells} |")
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
            f"{show('outpatient_at_inpatient_mix')} | {show('share_explained')} | "
            f"{show('share_explained_predicted')} |"
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


def rows_flow() -> list[str]:
    flow = read("echonext_clinical.json")["flow"]
    out = []
    for split in ("train", "val", "test", "no_split"):
        e = flow["ecgs"][split]
        out.append(
            f"| {split.replace('_', ' ')} | {sum(e.values()):,} | {flow['patients'][split]:,} | "
            f"{e['inpatient']:,} | {e['emergency']:,} | {e['outpatient']:,} | "
            f"{e['procedural']:,} | {flow['used'][split]} |"
        )
    return out


def rows_eras() -> list[str]:
    eras = read("echonext_clinical.json")["eras"]
    out = []
    for arm in STRONGEST:
        for context, cname in (("inpatient", "Inpatients"), ("outpatient", "Outpatients")):
            cells = " | ".join(
                f"{r['count']} of {r['n']}, {pct(r['share'])}" for r in eras["arms"][arm][context]
            )
            out.append(f"| {ARM_NAMES[arm]} | {cname} | {cells} |")
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


def rows_ppv_blocks() -> list[str]:
    blocks = read("ppv_intervals.json")["across_transfers"]
    out = []
    for key, name in PPV_BLOCKS.items():
        b = blocks[key]
        shares = " | ".join(share_ci(b[k]) for k in SHARES)
        out.append(
            f"| {name} | {b['cells']} | {b['clusters']} | "
            f"{points_ci(b['median_abs_gap_points'])} | {shares} |"
        )
    return out


REPAIR_NAMES = {
    "prior": "Prevalence corrected, no label",
    "recalibrated": "Logistic recalibration on 100 local labels",
    "abstention_cleared": "Per-label sets, patients sent to a reader counted as cleared",
    "abstention_referred": "Per-label sets, patients sent to a reader counted as referred",
    "as_delivered": "Model as delivered",
}


def rows_repairs() -> list[str]:
    summary = read("repairs.json")["summary"]["all"]
    out = []
    for key, name in REPAIR_NAMES.items():
        gains = " | ".join(
            "0"
            if key == "as_delivered"
            else f"{summary[t]['mean_gain_per_1000'][key]:+.1f}".replace("-", "−")
            for t in ("0.05", "0.10", "0.20")
        )
        out.append(f"| {name} | {gains} | {summary['0.10']['wins'][key]} |")
    return out


ROWS = {
    "ppv_blocks": rows_ppv_blocks,
    "repairs": rows_repairs,
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
    "flow": rows_flow,
    "subgroup_healthy": rows_subgroup_healthy,
    "race": rows_race,
    "eras": rows_eras,
}
