"""What the coverage guarantee is worth at another hospital.

One threshold is fitted on PTB-XL and spent, unchanged, on three populations at
once: the PTB-XL patients it was not fitted on, all of Shandong, and all of
Chongqing.  Nothing is tuned on the two external corpora and neither is ever
re-calibrated on itself (C-20); they are read once, scored once, and reported
whatever the number says.

The spread comes from re-drawing the calibration alone.  Each of the draws
halves PTB-XL fold 10 by patient (C-4), fits the threshold on one half, and
measures on the other half and on both external corpora with that same
threshold -- so the three panels differ in nothing but the population they
describe.

The result file names, per corpus, what the ingestion and label chains could not
make identical (C-14), and per row the calibration sample behind the threshold
with its effective size (C-9).

Usage: .venv/bin/python scripts/shift_table.py [--draws 200]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ecs.config import ACS_DIR, ACS_LABELLED_SPLIT, PTBXL_DIR, RESULTS_DIR, SPH_DIR
from ecs.report import CORRECTIONS, Source, Target, frozen_calibration_table

# Confidence levels from the plan, loosest first.
ALPHAS = (0.20, 0.10, 0.05)

# The setting the reading is written at: the 90% set on the score that gives the
# smallest sets, which is the one a clinic would be offered.
HEADLINE_ALPHA = 0.10
HEADLINE_SCORE = "lac"
CORPUS_NAMES = {"ptbxl": "PTB-XL", "sph": "Shandong", "acs": "Chongqing"}

# How far below the level asked for a class-conditional figure may land before
# the correction is called a failure there.  Two points: wider than the spread of
# any cell on this table, narrower than every gap the reading calls a break.
RESTORED_WITHIN = 0.02

# PTB-XL ships no ingestion deviation -- it is the corpus the canonical form was
# written around -- but its label is a choice, and the choice is what the two
# other corpora are held against.
PTBXL_DEVIATIONS = (
    "the label is the SCP-ECG MI superclass as shipped, subendocardial-injury "
    "statements included and no likelihood floor; dropping the injury statements "
    "moves the positive count from 5,469 to 5,288",
    "the infarct patterns PTB-XL records are undated, so the class is read as a "
    "chronic pattern rather than an acute event",
)


def patients_of(ids: list[str], root: Path) -> list[str]:
    """The patient each fold-10 record belongs to, in the order of ``ids``."""
    database = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    return [str(database.loc[int(i), "patient_id"]) for i in ids]


def n_patients(corpus: str, sph_dir: Path, acs_dir: Path) -> int:
    """How many distinct patients a corpus's records come from.

    Descriptive here rather than load-bearing: the external corpora are never
    split, so no patient straddles a calibration boundary.  It is reported
    because a corpus with several tracings per patient carries fewer independent
    draws than its record count suggests.
    """
    if corpus == "sph":
        return int(pd.read_csv(sph_dir / "metadata.csv")["Patient_ID"].nunique())
    return int(pd.read_csv(acs_dir / ACS_LABELLED_SPLIT)["Patient_id"].nunique())


def paired(rows: list[dict[str, Any]], corpus: str, key: str, klass: str = "1") -> dict[str, Any]:
    """One correction against another on the draw they shared.

    The draws are the same calibration halves in the same order for every row of
    the table, so a difference can be taken inside a draw.  Two means subtracted
    would throw that away and report a spread wider than the comparison actually
    carries.
    """
    series = {
        row["correction"]: (
            row["by_corpus"][corpus]["coverage_by_class_by_draw"][klass]
            if key == "coverage_by_class"
            else row["by_corpus"][corpus][f"{key}_by_draw"]
        )
        for row in rows
    }
    out: dict[str, Any] = {}
    for left, right in (("mondrian", "none"), ("weighted", "none"), ("mondrian", "weighted")):
        difference = np.asarray(series[left]) - np.asarray(series[right])
        out[f"{left} - {right}"] = {
            "mean": round(float(difference.mean()), 4),
            "sd": round(float(difference.std(ddof=1)), 4),
            "n_draws": int(difference.size),
        }
    return out


def reading(rows: list[dict[str, Any]], corpora: list[str]) -> dict[str, Any]:
    """The answer the day was for, with every figure read back off the rows above.

    Nothing here is typed in: each sentence is assembled from the cells it names,
    so a table that moves moves the answer with it rather than leaving a
    conclusion behind that the numbers no longer support.
    """
    chosen = [r for r in rows if r["alpha"] == HEADLINE_ALPHA and r["score"] == HEADLINE_SCORE]
    by_correction = {row["correction"]: row for row in chosen}
    target = 1.0 - HEADLINE_ALPHA
    external = [c for c in corpora if c != "ptbxl"]

    def sick(correction: str, corpus: str) -> dict[str, float]:
        return by_correction[correction]["by_corpus"][corpus]["coverage_by_class"]["1"]

    def size(correction: str, corpus: str) -> float:
        return float(by_correction[correction]["by_corpus"][corpus]["mean_set_size"]["mean"])

    def holds(correction: str, corpus: str) -> bool:
        return sick(correction, corpus)["mean"] >= target - RESTORED_WITHIN

    def weighted_block(corpus: str) -> dict[str, Any]:
        return by_correction["weighted"]["by_corpus"][corpus]

    repaired = {
        correction: [c for c in external if holds(correction, c)] for correction in CORRECTIONS
    }
    unrepaired = [c for c in external if not any(holds(k, c) for k in CORRECTIONS)]
    calibration = by_correction["mondrian"]["calibration"]
    minority = min(calibration["n_by_class"], key=lambda c: calibration["n_by_class"][c]["mean"])
    sick_paired = {c: paired(chosen, c, "coverage_by_class") for c in corpora}
    names = {"none": "the uncorrected threshold", "mondrian": "Mondrian", "weighted": "weighted"}

    def phrase(chosen_corpora: list[str]) -> str:
        return (
            " and ".join(CORPUS_NAMES[c] for c in chosen_corpora) if chosen_corpora else "neither"
        )

    def against_none(correction: str) -> str:
        return "; ".join(
            f"{CORPUS_NAMES[c]} {sick_paired[c][f'{correction} - none']['mean']:+.4f} "
            f"(sd {sick_paired[c][f'{correction} - none']['sd']:.4f})"
            for c in corpora
        )

    abroad = "; ".join(
        f"{CORPUS_NAMES[c]} "
        + ", ".join(
            f"{names[k]} {sick(k, c)['mean']:.3f} (sd {sick(k, c)['sd']:.3f})" for k in CORRECTIONS
        )
        for c in external
    )
    cost_lines = "; ".join(
        f"{CORPUS_NAMES[c]} {size('none', c):.3f} uncorrected, {size('mondrian', c):.3f} "
        f"Mondrian, {size('weighted', c):.3f} weighted"
        for c in corpora
    )
    estimate = "; ".join(
        f"{CORPUS_NAMES[c]} true {weighted_block(c)['prevalence']:.4f} against estimated "
        f"{weighted_block(c)['calibration']['estimated_prevalence']['mean']:.4f} "
        f"(sd {weighted_block(c)['calibration']['estimated_prevalence']['sd']:.4f}), "
        f"effective sample size "
        f"{weighted_block(c)['calibration']['effective_sample_size']['mean']:.0f} of "
        f"{calibration['n']['mean']:.0f}"
        for c in corpora
    )
    stubborn = (
        ", ".join(f"{names[k]} {sick(k, unrepaired[0])['mean']:.3f}" for k in CORRECTIONS)
        if unrepaired
        else ""
    )
    everywhere = repaired["none"] == repaired["mondrian"] == repaired["weighted"]
    easy = (
        f" -- and there it holds under every correction only because "
        f"{1 - by_correction['none']['by_corpus'][repaired['none'][0]]['prevalence']:.0%} of that "
        f"corpus is healthy and the healthy are easy"
        if everywhere and repaired["none"]
        else ""
    )
    reached = (
        f"Within {RESTORED_WITHIN} of the level asked for, the sick are reached on "
        f"{phrase(repaired['none'])}{easy}."
        if everywhere
        else (
            f"Within {RESTORED_WITHIN} of the level asked for, the sick are reached under "
            f"Mondrian on {phrase(repaired['mondrian'])}, under weighting on "
            f"{phrase(repaired['weighted'])}, and uncorrected on {phrase(repaired['none'])}."
        )
    )
    lost = [c for c in corpora if weighted_block(c)["calibration"]["n_unidentified"]]
    unidentified = (
        " Third, in draws where the estimate does not exist at all: "
        + "; ".join(
            f"{CORPUS_NAMES[c]} on "
            f"{weighted_block(c)['calibration']['n_unidentified']} of "
            f"{weighted_block(c)['coverage']['n_draws']} draws the estimated prior left the "
            f"simplex, which is BBSE saying a change of class mix does not explain that corpus, "
            f"and the sets widened to every label"
            for c in lost
        )
        + "."
        if lost
        else ""
    )
    emptied = [c for c in corpora if weighted_block(c)["empty_rate"]["mean"] > 0.005]
    empties = "; ".join(
        f"{CORPUS_NAMES[c]} {weighted_block(c)['empty_rate']['mean']:.1%} of sets empty under "
        f"weighting against {by_correction['none']['by_corpus'][c]['empty_rate']['mean']:.1%} "
        f"uncorrected"
        for c in emptied
    )
    return {
        "asked_at": (
            f"the {target:.0%} set on the {HEADLINE_SCORE} score, over "
            f"{by_correction['none']['by_corpus']['ptbxl']['coverage']['n_draws']} draws"
        ),
        "restored_within": RESTORED_WITHIN,
        "which_correction_this_shift_needed": (
            f"Mondrian, and the reason is visible before any hospital changes. Where the "
            f"threshold was fitted, infarctions fall inside the {target:.0%} set only "
            f"{sick('none', 'ptbxl')['mean']:.3f} of the time (sd "
            f"{sick('none', 'ptbxl')['sd']:.3f}) while the overall figure lands on "
            f"{by_correction['none']['by_corpus']['ptbxl']['coverage']['mean']:.3f}: the "
            f"guarantee was marginal all along, and it was being paid for by the healthy "
            f"majority. Changing hospital does not create that gap, it only changes how much "
            f"the healthy majority hides it. Away from home: {abroad}. Calibrating inside the "
            f"class closes it on the draw it is measured on -- Mondrian minus uncorrected, "
            f"paired: {against_none('mondrian')}. Estimating the target prior instead does not: "
            f"weighted minus uncorrected, paired: {against_none('weighted')}, which on Shandong "
            f"is a loss. {reached}"
        ),
        "what_each_costs": (
            f"Mondrian pays in set size and in the points the minority class is left with: "
            f"class {minority} gets {calibration['n_by_class'][minority]['mean']:.0f} of "
            f"{calibration['n']['mean']:.0f} calibration points to fit its own threshold on, "
            f"and the sets widen -- mean set size {cost_lines}. Nothing is estimated, so that "
            f"is the whole bill. The weighted correction pays twice. First in the estimate: "
            f"{estimate}. Second in refusals it does not announce as set size -- {empties}: "
            f"a mean set size below one is empty sets, not tight ones, and an empty set is the "
            f"model declining to answer rather than answering well.{unidentified}"
        ),
        "what_neither_repairs": (
            (
                f"On {phrase(unrepaired)} the sick stay below the level under every correction "
                f"({stubborn}). "
                f"Both corrections assume P(X|Y) is fixed and only P(Y) moves; there it is not "
                f"the mix that moved. The deviations this file carries for that corpus name the "
                f"reason -- the label is a different event -- and no reweighting of a "
                f"calibration set can repair a target whose positives are not the same positives."
            )
            if unrepaired
            else "Every corpus on this table reaches the level under at least one correction."
        ),
        "what_cannot_be_claimed": (
            "That class-conditional calibration 'works' in deployment. Its guarantee is "
            "conditional on the patient's true class, which is unknown at the "
            "bedside: the set is computable without the label, but the clinician holding one "
            "cannot know which of the two promises applies to them, and pays the wider set "
            "either way. Nor that the weighted correction was given a fair estimate here and "
            "failed on its merits -- BBSE was handed the same predictor whose coverage is being "
            "measured, and the priors above are what it returned."
        ),
        "paired": {
            corpus: {
                "sick coverage": paired(chosen, corpus, "coverage_by_class"),
                "mean set size": paired(chosen, corpus, "mean_set_size"),
            }
            for corpus in corpora
        },
    }


def git_commit(path: Path | None = None) -> str:
    """The current commit, or the last one that touched ``path``."""
    command = (
        ["git", "rev-parse", "HEAD"]
        if path is None
        else ["git", "log", "-1", "--format=%H", "--", str(path)]
    )
    out = subprocess.run(command, capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scores", default=str(RESULTS_DIR / "baseline/scores.npz"))
    parser.add_argument("--external", default=str(RESULTS_DIR / "external"))
    parser.add_argument("--ptbxl-dir", default=str(PTBXL_DIR))
    parser.add_argument("--sph-dir", default=str(SPH_DIR))
    parser.add_argument("--acs-dir", default=str(ACS_DIR))
    parser.add_argument("--out", default=str(RESULTS_DIR / "shift.json"))
    args = parser.parse_args(argv)

    with np.load(args.scores, allow_pickle=False) as data:
        ids, labels, probs = list(data["ids"]), data["labels"], data["probs"]
    source = Source(
        "ptbxl",
        probs,
        labels,
        patients_of([str(i) for i in ids], Path(args.ptbxl_dir)),
        deviations=PTBXL_DEVIATIONS,
    )

    external = Path(args.external)
    provenance: dict[str, dict[str, object]] = {}
    targets: dict[str, Target] = {}
    for corpus in ("sph", "acs"):
        sidecar = json.loads((external / f"{corpus}.json").read_text())
        with np.load(external / f"{corpus}.npz", allow_pickle=False) as data:
            targets[corpus] = Target(
                data["probs"],
                data["labels"],
                n_patients(corpus, Path(args.sph_dir), Path(args.acs_dir)),
                tuple(sidecar["deviations"]),
            )
        provenance[corpus] = {
            key: sidecar[key]
            for key in ("title", "checkpoint", "git_commit", "seed", "n_scored", "n_excluded")
        }
    provenance["ptbxl"] = {
        "title": "PTB-XL fold 10, the half not used to calibrate at each draw",
        "checkpoint": str(Path(args.scores).relative_to(RESULTS_DIR.parent)),
        "git_commit": git_commit(Path(args.scores)),
        "seed": 0,
        "n_scored": int(len(labels)),
        "n_excluded": 0,
    }

    started = time.perf_counter()
    rows = frozen_calibration_table(
        source, targets, ALPHAS, n_draws=args.draws, seed=args.seed, keep_draws=True
    )
    report = {
        "question": (
            "what a coverage guarantee calibrated on PTB-XL is worth at two other hospitals"
        ),
        "protocol": (
            "the threshold is fitted on half the PTB-XL fold-10 patients and spent unchanged on "
            "the other half, on all of Shandong and on all of Chongqing; the external corpora are "
            "scored once, never re-calibrated on themselves, and nothing is tuned on them (C-20). "
            "The spread is over the calibration draw alone, since the targets are fixed"
        ),
        "calibrated_on": "ptbxl",
        "corrections": {
            "none": "one threshold shared by both classes",
            "mondrian": (
                "one threshold per class, each fitted inside that class on PTB-XL; exact in "
                "finite samples under any change of class proportions, with nothing estimated "
                "(label conditional validity, Vovk, ACML 2012, PMLR 25:475-490, Prop. 3)"
            ),
            "weighted": (
                "each PTB-XL calibration point reweighted by w(y) = q(y)/p(y) in the "
                "weighted-exchangeability form of Tibshirani et al. (NeurIPS 2019); q is the "
                "target's class mix, which is not known and is estimated from that corpus's "
                "unlabelled predictions by BBSE (Lipton, Wang & Smola, ICML 2018), so the "
                "guarantee is worth exactly what the estimate is worth"
            ),
        },
        "what_the_target_supplied": (
            "under the two unweighted corrections, nothing: one PTB-XL threshold is spent "
            "unchanged on all three corpora. Under the weighted correction, each corpus's "
            "unlabelled predicted-label marginal and nothing else -- no target label and no "
            "target score enters a threshold, which is what keeps C-20 intact while the "
            "threshold is allowed to differ by corpus"
        ),
        "classes": {"0": "no infarction", "1": "infarction"},
        "n_draws": args.draws,
        "seed": args.seed,
        "git_commit": git_commit(),
        "corpora": provenance,
        "seconds": round(time.perf_counter() - started, 1),
        "reading": reading(rows, [source.name, *targets]),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    header = f"{'level':>6} {'score':>5} {'correction':>10} {'corpus':>6} {'sick':>7}"
    print(f"{header} {'coverage':>17} {'covered sick':>17} {'size':>6} {'ess':>7}")
    for row in rows:
        for corpus, block in cast(dict[str, dict[str, Any]], row["by_corpus"]).items():
            sick = block["coverage_by_class"]["1"]
            print(
                f"{1 - float(str(row['alpha'])):>5.0%} {row['score']:>5} "
                f"{row['correction']:>10} {corpus:>6} {block['prevalence']:>7.4f} "
                f"{block['coverage']['mean']:>9.4f} ± {block['coverage']['sd']:.4f} "
                f"{sick['mean']:>9.4f} ± {sick['sd']:.4f} "
                f"{block['mean_set_size']['mean']:>6.3f} "
                f"{block['calibration']['effective_sample_size']['mean']:>7.0f}"
            )
    answer = cast(dict[str, Any], report["reading"])
    for key in (
        "which_correction_this_shift_needed",
        "what_each_costs",
        "what_neither_repairs",
        "what_cannot_be_claimed",
    ):
        print(f"\n{key}\n  {answer[key]}")
    print(f"\n{out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
