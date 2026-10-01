"""The one-page transfer report, rendered from ``results/echonext_transfer.json`` alone.

Every figure on the page is read from the result file at render time, so the
page cannot say a number the file does not hold, and re-rendering it is how a
test holds the committed page to the file.  Cells that later tasks fill are
printed empty, with the task that fills them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CONTEXT_NAMES", "LABEL_NAMES", "pct", "render"]

LABEL_NAMES = {
    "lvef_lte_45_flag": "LVEF ≤45%",
    "lvwt_gte_13_flag": "LV wall ≥1.3 cm",
    "aortic_stenosis_moderate_or_greater_flag": "Aortic stenosis, moderate+",
    "aortic_regurgitation_moderate_or_greater_flag": "Aortic regurgitation, moderate+",
    "mitral_regurgitation_moderate_or_greater_flag": "Mitral regurgitation, moderate+",
    "tricuspid_regurgitation_moderate_or_greater_flag": "Tricuspid regurgitation, moderate+",
    "pulmonary_regurgitation_moderate_or_greater_flag": "Pulmonary regurgitation, moderate+",
    "rv_systolic_dysfunction_moderate_or_greater_flag": "RV dysfunction, moderate+",
    "pericardial_effusion_moderate_large_flag": "Pericardial effusion, moderate+",
    "pasp_gte_45_flag": "PASP ≥45 mmHg",
    "tr_max_gte_32_flag": "TR velocity ≥3.2 m/s",
    "shd_moderate_or_greater_flag": "Composite (any of the above)",
}
CONTEXT_NAMES = {"inpatient": "inpatients", "emergency": "emergency", "outpatient": "outpatients"}
COMPOSITE = "shd_moderate_or_greater_flag"
LVEF = "lvef_lte_45_flag"


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def num(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _interval(low: float | None, high: float | None) -> str:
    return "" if low is None or high is None else f" [{pct(low)}, {pct(high)}]"


def _cell(rows: list[dict[str, Any]], label: str, context: str, method: str) -> dict[str, Any]:
    for row in rows:
        if (row["label"], row["context"], row["method"]) == (label, context, method):
            return row
    raise KeyError((label, context, method))


def _coverage_section(result: dict[str, Any], arm: dict[str, Any], target: str) -> list[str]:
    rows = arm["coverage"]
    lines = [
        f"## Coverage per label at the {pct(1 - result['alpha'], 0)} level",
        "",
        "Ill covered is the share of patients with the finding whose decision includes it: "
        "the sensitivity for the plain threshold, the coverage of the positive class for the "
        "two conformal schemes. A per-label threshold that the calibration positives cannot "
        "certify flags everyone, and the row then shows the whole target flagged. The plain "
        "threshold and the per-label threshold of the ill are the same calibration quantile, "
        "so their columns agree; the per-label scheme adds a threshold for the healthy, and "
        "with it the share sent to a human.",
        "",
        "| Label | Prevalence, source | Prevalence, target | Ill in target "
        "| Ill covered, plain | Ill covered, pooled | Ill covered, per-label "
        "| Healthy covered, per-label | Sent to a human, per-label | AUROC, target |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label in result["labels"]:
        plain = _cell(rows, label, target, "plain")
        pooled = _cell(rows, label, target, "pooled")
        per = _cell(rows, label, target, "perlabel")
        auroc = arm["auroc"][target][label]
        lines.append(
            f"| {LABEL_NAMES[label]} | {pct(result['source']['prevalence'][label])} "
            f"| {pct(result['targets'][target]['prevalence'][label])} | {per['n_pos']} "
            f"| {pct(plain['coverage_pos'])} | {pct(pooled['coverage_pos'])} "
            f"| {pct(per['coverage_pos'])}"
            f"{_interval(per['coverage_pos_low'], per['coverage_pos_high'])} "
            f"| {pct(per['coverage_neg'])} | {pct(per['abstention'])} "
            f"| {num(auroc['auroc'], 3)} |"
        )
    return lines


def _subgroup_section(arm: dict[str, Any], target: str) -> list[str]:
    lines = [
        "## Coverage of the ill inside groups, composite, per-label thresholds",
        "",
        "| Group | | Ill | Ill covered [95% CI] |",
        "|---|---|---|---|",
    ]
    for row in arm["subgroups"]:
        if (row["label"], row["context"], row["method"]) != (COMPOSITE, target, "perlabel"):
            continue
        lines.append(
            f"| {row['kind']} | {row['group']} | {row['n_pos']} "
            f"| {pct(row['coverage_pos'])}{_interval(row['low'], row['high'])} |"
        )
    return lines


def _calibration_section(arm: dict[str, Any], target: str) -> list[str]:
    lines = [
        "## Calibration of the composite probability",
        "",
        "Slope 1 and intercept 0 are perfect calibration; the intercept is "
        "calibration-in-the-large on the logit scale.",
        "",
        "| Test split, context | Prevalence | Mean predicted | Slope | Intercept | Brier |",
        "|---|---|---|---|---|---|",
    ]
    for context, by_label in arm["calibration"].items():
        row = by_label[COMPOSITE]
        lines.append(
            f"| {CONTEXT_NAMES[context]} | {pct(row['prevalence'])} | {pct(row['mean_predicted'])} "
            f"| {num(row['slope'])} | {num(row['intercept'])} | {num(row['brier'], 3)} |"
        )
    lines += [
        "",
        f"Calibration curve on {CONTEXT_NAMES[target]}, ten bins of equal count:",
        "",
        "| Predicted range | ECGs | Mean predicted | Observed [95% CI] |",
        "|---|---|---|---|",
    ]
    for row in arm["calibration_curve"][target][COMPOSITE]:
        lines.append(
            f"| {pct(row['p_low'])} to {pct(row['p_high'])} | {row['n']} "
            f"| {pct(row['mean_predicted'])} "
            f"| {pct(row['observed'])}{_interval(row['observed_low'], row['observed_high'])} |"
        )
    return lines


def _decision_section(arm: dict[str, Any], target: str) -> list[str]:
    lines = [
        f"## Net benefit of sending {CONTEXT_NAMES[target]} to echocardiography, composite",
        "",
        "Net benefit is true positives per patient minus false positives per patient weighted "
        "by the odds of the threshold.",
        "",
        "| Threshold | Model | Echo for all | Echo for none |",
        "|---|---|---|---|",
    ]
    for row in arm["decision_curve"][target][COMPOSITE]:
        lines.append(
            f"| {pct(row['threshold'], 0)} | {num(row['model'], 3)} | {num(row['treat_all'], 3)} "
            f"| {num(row['treat_none'], 3)} |"
        )
    return lines


def _ppv_section(arm: dict[str, Any], target: str) -> list[str]:
    lines = [
        "## Positive predictive value at the plain threshold",
        "",
        "The source column is what a buyer computes from the sensitivity and specificity "
        "measured at the source, at the target's prevalence; the target column is observed.",
        "",
        "| Label | Prevalence, target | Sensitivity, source / target "
        "| Specificity, source / target | PPV from source | PPV observed "
        "| False alerts per ill patient found, from source / observed |",
        "|---|---|---|---|---|---|---|",
    ]
    for label in (COMPOSITE, LVEF):
        row = arm["ppv"][target][label]
        lines.append(
            f"| {LABEL_NAMES[label]} | {pct(row['prevalence'])} "
            f"| {pct(row['sens_source'])} / {pct(row['sens_target'])} "
            f"| {pct(row['spec_source'])} / {pct(row['spec_target'])} "
            f"| {pct(row['ppv_from_source'])} | {pct(row['ppv_observed'])} "
            f"| {num(row['false_alerts_from_source'], 1)} "
            f"/ {num(row['false_alerts_observed'], 1)} |"
        )
    return lines


def _ladder_section(arm: dict[str, Any], target: str) -> list[str]:
    lines = [
        "## Target labels that repair the threshold and the calibration",
        "",
        "The target is cut once by patient into a pool and an evaluation half. Rung 0 spends "
        "the source thresholds; rung n refits the per-label thresholds and an intercept shift "
        "on n ECGs drawn from the pool, over repeated draws, all read on the same evaluation "
        "half.",
        "",
        "| Label | Target labels | Ill covered, mean [10th, 90th centile] | Healthy covered "
        "| Draws that flag everyone | Absolute intercept after shift |",
        "|---|---|---|---|---|---|",
    ]
    for label in (COMPOSITE, LVEF):
        for row in arm["ladder"][target][label]:
            lines.append(
                f"| {LABEL_NAMES[label]} | {row['labels']} | {pct(row['coverage_pos_mean'])} "
                f"[{pct(row['coverage_pos_p10'])}, {pct(row['coverage_pos_p90'])}] "
                f"| {pct(row['coverage_neg_mean'])} | {pct(row['share_all_flagged'])} "
                f"| {num(row['abs_intercept_mean'])} |"
            )
    return lines


def render(result: dict[str, Any], arm_name: str, target: str = "outpatient") -> str:
    """The markdown page for one arm, from the Columbia inpatients to one target context."""
    arm = result["arms"][arm_name]
    source, tgt = result["source"], result["targets"][target]
    per = _cell(arm["coverage"], COMPOSITE, target, "perlabel")
    lines = [
        f"# Transfer report: {arm['title']}, "
        f"Columbia inpatients to Columbia {CONTEXT_NAMES[target]}",
        "",
        f"Calibrated on {source['n']} inpatient ECGs and applied unchanged to {tgt['n']} "
        f"{CONTEXT_NAMES[target]}, the per-label thresholds cover "
        f"{pct(per['coverage_pos'])} of {CONTEXT_NAMES[target]} with structural heart "
        f"disease (composite) where {pct(1 - result['alpha'], 0)} was asked, cover "
        f"{pct(per['coverage_neg'])} of those without it, and send "
        f"{pct(per['abstention'])} to a human. The composite's prevalence falls from "
        f"{pct(source['prevalence'][COMPOSITE])} to {pct(tgt['prevalence'][COMPOSITE])}.",
        "",
        "| | Source | Target |",
        "|---|---|---|",
        f"| Cohort | {source['cohort']} | {tgt['cohort']} |",
        f"| ECGs, one per patient | {source['n']} | {tgt['n']} |",
        f"| Composite prevalence | {pct(source['prevalence'][COMPOSITE])} "
        f"| {pct(tgt['prevalence'][COMPOSITE])} |",
        f"| LVEF ≤45% prevalence | {pct(source['prevalence'][LVEF])} "
        f"| {pct(tgt['prevalence'][LVEF])} |",
        "",
        f"Model: {arm['description']}. Input: {result['input']}. Nothing is refitted on the "
        "target except on the ladder at the end of the page.",
        "",
    ]
    for section in (
        _coverage_section(result, arm, target),
        _subgroup_section(arm, target),
        _calibration_section(arm, target),
        _decision_section(arm, target),
        _ppv_section(arm, target),
        _ladder_section(arm, target),
    ):
        lines += [*section, ""]
    lines += ["## Cells filled by other tasks", ""]
    for cell, filler in result["empty_cells"].items():
        lines.append(f"- {cell}: empty, filled by {filler}.")
    return "\n".join(lines) + "\n"
