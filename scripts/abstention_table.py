"""How often the model declines to answer, and at what confidence.

Reads the fold-10 scores the baseline left behind and turns them into the table
the plan calls the abstention table: for each confidence level, how often the
prediction set holds one label, both, or neither, and how often it holds the
true one -- each as a mean over at least a hundred calibration/test re-draws
with its spread (C-10), and separately for infarction and not (C-11).

The re-draw is over patients, not records (C-4), so the patient identifiers come
from PTB-XL's own database table rather than from the score file.

Usage: .venv/bin/python scripts/abstention_table.py [--draws 200]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecs.config import PTBXL_DIR, RESULTS_DIR
from ecs.report import REDRAW_CORRECTIONS, SCORES, repeated_split_report

# The levels the plan names, loosest first, so the table reads as confidence rising.
ALPHAS = (0.20, 0.10, 0.05)


def patients_of(ids: list[str], root: Path) -> list[str]:
    """The patient each fold-10 record belongs to, in the order of ``ids``."""
    database = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    return [str(database.loc[int(i), "patient_id"]) for i in ids]


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scores", default=str(RESULTS_DIR / "baseline/scores.npz"))
    parser.add_argument("--ptbxl-dir", default=str(PTBXL_DIR))
    parser.add_argument("--out", default=str(RESULTS_DIR / "abstention.json"))
    args = parser.parse_args(argv)

    with np.load(args.scores, allow_pickle=False) as data:
        ids, labels, probs = list(data["ids"]), data["labels"], data["probs"]
    patients = patients_of([str(i) for i in ids], Path(args.ptbxl_dir))

    started = time.perf_counter()
    rows = [
        repeated_split_report(
            probs, labels, patients, alpha, score, correction, args.draws, args.seed
        )
        for alpha in ALPHAS
        for score in SCORES
        for correction in REDRAW_CORRECTIONS
    ]
    report = {
        "corpus": "ptbxl",
        "split": "PTB-XL fold 10, halved by patient at every draw",
        "source": str(Path(args.scores).relative_to(RESULTS_DIR.parent)),
        "classes": {"0": "no infarction", "1": "infarction"},
        "n_records": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_draws": args.draws,
        "seed": args.seed,
        "git_commit": git_commit(),
        "seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"{'level':>6} {'score':>5} {'correction':>10} {'covered':>17} {'1':>6} {'2':>6} {'0':>6}"
    )
    for row in rows:
        cell = {
            key: row[key] for key in ("coverage", "one_label_rate", "two_label_rate", "empty_rate")
        }
        got = {key: value["mean"] for key, value in cell.items() if isinstance(value, dict)}
        sd = cell["coverage"]["sd"] if isinstance(cell["coverage"], dict) else 0.0
        print(
            f"{1 - float(str(row['alpha'])):>5.0%} {row['score']:>5} {row['correction']:>10} "
            f"{got['coverage']:>9.4f} ± {sd:.4f} {got['one_label_rate']:>6.3f} "
            f"{got['two_label_rate']:>6.3f} {got['empty_rate']:>6.3f}"
        )
    print(f"\n{out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
