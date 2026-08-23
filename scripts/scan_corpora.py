"""Run every record of every corpus through the ingestion chain and write the
counts to results/ingest.json: records read, records excluded for a NaN or Inf
sample (C-15), and the deviations the chain had to name (C-14).

Reads in chunks of a thousand records so that no corpus is ever held in memory
whole; nothing transformed is written anywhere (C-14b).

Usage: .venv/bin/python scripts/scan_corpora.py
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from ecs.config import ACS_DIR, N_LEADS, PTBXL_DIR, RESULTS_DIR, SPH_DIR, WINDOW_SAMPLES
from ecs.ingest import Corpus, load_acs, load_ptbxl, load_sph

CHUNK = 1000


def scan(
    name: str, ids: Sequence[str], load: Callable[[Sequence[str]], Corpus]
) -> dict[str, object]:
    started = time.time()
    total = Corpus(name, np.empty((0, N_LEADS, WINDOW_SAMPLES), np.float32), [], [])
    for start in range(0, len(ids), CHUNK):
        chunk = load(ids[start : start + CHUNK])
        total.ids.extend(chunk.ids)
        total.excluded.extend(chunk.excluded)
        total.n_resampled += chunk.n_resampled
        total.n_cropped += chunk.n_cropped
        total.notes = chunk.notes
        done = min(start + CHUNK, len(ids))
        print(f"  {name}: {done:>6} / {len(ids)}  {time.time() - started:6.0f} s", flush=True)
    assert len(total.ids) + len(total.excluded) == len(ids)
    return {
        "n_records": len(ids),
        "n_kept": len(total.ids),
        "n_excluded": len(total.excluded),
        "excluded_ids": total.excluded,
        "deviations": total.deviations,
        "seconds": round(time.time() - started, 1),
    }


def main() -> int:
    database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    metadata = pd.read_csv(SPH_DIR / "metadata.csv")
    train = pd.read_csv(ACS_DIR / "CSV/train.csv")
    test = pd.read_csv(ACS_DIR / "CSV/test.csv")
    acs_ids = [str(f).removesuffix(".dat") for f in pd.concat([train, test])["ecg_row_record"]]
    out = {
        "ptbxl": scan(
            "ptbxl",
            [str(i) for i in database.index],
            lambda ids: load_ptbxl(database, ids=[int(i) for i in ids]),
        ),
        "sph": scan("sph", list(metadata["ECG_ID"]), lambda ids: load_sph(metadata, ids=ids)),
        "acs": scan("acs", acs_ids, lambda ids: load_acs(train, ids=ids)),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "ingest.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    for name, counts in out.items():
        print(name, {k: v for k, v in counts.items() if k != "excluded_ids"})
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
