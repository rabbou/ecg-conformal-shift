"""Seconds per hundred PTB-XL records through each encoder arm, on this machine.

The number sizes the week: one forward pass over ~65,000 records on a CPU is
the cost of adding an arm, and no training run starts before it is known
(PLAN.md, day 1).  One row per arm goes to a JSON report; an arm that cannot be
loaded gets ``seconds_per_100: null`` and a note saying exactly why, so the next
session does not rediscover the blocker.

The arms themselves, and the chain each one demands, live in ``ecs.encoders``:
this script only times them.

Usage: .venv/bin/python scripts/timing_probe.py [--arm NAME ...] [--n 100]
                                                [--out results/timing.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecs.config import PTBXL_DIR, RESULTS_DIR
from ecs.encoders import ARMS, PRETRAINING, WEIGHTS, machine_info
from ecs.ingest import load_ptbxl

BATCH = 20


def records(n: int) -> np.ndarray:
    database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    return load_ptbxl(database, ids=list(database.index[:n])).x


def time_arm(name: str, x: np.ndarray) -> dict[str, object]:
    row: dict[str, object] = {
        "arm": name,
        "seconds_per_100": None,
        "n_params": None,
        "weights_source": None,
        "pretraining_corpora": PRETRAINING[name],
        "notes": "",
        "n_records": len(x),
        "batch_size": BATCH,
        "torch_threads": torch.get_num_threads(),
        "load_average_1min": round(os.getloadavg()[0], 1),  # other work on the machine
    }
    try:
        embed, meta = ARMS[name]()
        row.update(meta)
        with torch.no_grad():
            embed(x[:BATCH])  # warm-up, not timed
            started = time.perf_counter()
            dims = {tuple(embed(x[i : i + BATCH]).shape[1:]) for i in range(0, len(x), BATCH)}
            seconds = time.perf_counter() - started
        row["seconds_per_100"] = round(seconds * 100 / len(x), 2)
        row["embedding_shape"] = sorted(dims)[0]
    except Exception as error:  # noqa: BLE001 -- the blocker IS the result
        row["notes"] = f"{row['notes']} | not timed: {type(error).__name__}: {error}".strip(" |")
        traceback.print_exc()
    if name == "ecgfm" and row["n_params"] is None:
        row["n_params"] = _ecgfm_param_count()
    return row


def _ecgfm_param_count() -> int | None:
    """Parameters in the ECG-FM checkpoint, countable without fairseq."""
    path = WEIGHTS / "ecgfm/mimic_iv_ecg_physionet_pretrained.pt"
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return int(sum(v.numel() for v in checkpoint["model"].values()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", nargs="*", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default=str(RESULTS_DIR / "timing.json"))
    args = parser.parse_args()

    out = Path(args.out)
    report = json.loads(out.read_text()) if out.exists() else {"machine": {}, "arms": []}
    report["machine"] = machine_info()
    x = records(args.n)
    print(f"{len(x)} PTB-XL records in canonical form", flush=True)
    for name in args.arm:
        row = time_arm(name, x)
        report["arms"] = [r for r in report["arms"] if r["arm"] != name] + [row]
        report["arms"].sort(key=lambda r: list(ARMS).index(str(r["arm"])))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
