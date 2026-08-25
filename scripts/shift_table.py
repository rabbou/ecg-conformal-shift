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
from ecs.report import Source, Target, frozen_calibration_table

# The levels the plan names, loosest first, so the table reads as confidence rising.
ALPHAS = (0.20, 0.10, 0.05)

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


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
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
        "git_commit": json.loads((Path(args.scores).parent / "config.json").read_text()).get(
            "git_commit", "recorded in results/baseline/config.json"
        ),
        "seed": 0,
        "n_scored": int(len(labels)),
        "n_excluded": 0,
    }

    started = time.perf_counter()
    rows = frozen_calibration_table(source, targets, ALPHAS, n_draws=args.draws, seed=args.seed)
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
            "mondrian": "one threshold per class, each fitted inside that class on PTB-XL",
        },
        "classes": {"0": "no infarction", "1": "infarction"},
        "n_draws": args.draws,
        "seed": args.seed,
        "git_commit": git_commit(),
        "corpora": provenance,
        "seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    header = f"{'level':>6} {'score':>5} {'correction':>10} {'corpus':>6} {'sick':>7}"
    print(f"{header} {'coverage':>17} {'covered sick':>17}")
    for row in rows:
        for corpus, block in cast(dict[str, dict[str, Any]], row["by_corpus"]).items():
            sick = block["coverage_by_class"]["1"]
            print(
                f"{1 - float(str(row['alpha'])):>5.0%} {row['score']:>5} "
                f"{row['correction']:>10} {corpus:>6} {block['prevalence']:>7.4f} "
                f"{block['coverage']['mean']:>9.4f} ± {block['coverage']['sd']:.4f} "
                f"{sick['mean']:>9.4f} ± {sick['sd']:.4f}"
            )
    print(f"\n{out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
