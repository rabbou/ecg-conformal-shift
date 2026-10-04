"""Read every EchoNext tracing once: check the files, file one provenance row per tracing.

The per-tracing table is derived restricted data and goes to
``$ECS_ECHONEXT_DERIVED/provenance.csv.gz``.  What reaches the repository is
``results/echonext_provenance.json``: whether each file matches its published
digest, the count per split, the units found, the copy groups, the quality
flags and the peak resident memory of the pass.

Usage: .venv/bin/python scripts/echonext_provenance.py
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import time

from ecs.config import RESULTS_DIR
from ecs.echonext import (
    DERIVED_DIR,
    DISTRIBUTION,
    ECHONEXT_DIR,
    LEAD_ORDER,
    PROVENANCE_FIELDS,
    provenance_rows,
    read_metadata,
    verify_files,
)
from ecs.encoders import machine_info

OUT = RESULTS_DIR / "echonext_provenance.json"


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # Linux reports KiB


def main() -> None:
    started = time.time()
    files = verify_files(ECHONEXT_DIR)
    meta = read_metadata(ECHONEXT_DIR)
    table = provenance_rows(meta)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(DERIVED_DIR / "provenance.csv.gz", index=False)

    sizes = table.groupby("copy_group").size()
    result = {
        "distribution": DISTRIBUTION,
        "files_match_published_sha256": files,
        "records": int(len(table)),
        "records_per_split": {k: int(v) for k, v in table["split"].value_counts().items()},
        "metadata_rows": int(len(meta)),
        "fields": list(PROVENANCE_FIELDS),
        "columns_present": [c for c in PROVENANCE_FIELDS if c in table.columns],
        "units": sorted(table["unit"].unique().tolist()),
        "lead_order": list(LEAD_ORDER),
        "copy_groups": int(sizes.size),
        "copy_groups_with_more_than_one_record": int((sizes > 1).sum()),
        "records_in_shared_copy_groups": int(sizes[sizes > 1].sum()),
        "copy_groups_across_splits": int(
            (table.groupby("copy_group")["split"].nunique() > 1).sum()
        ),
        "records_with_a_flat_lead": int((table["quality_flat_leads"] > 0).sum()),
        "records_with_non_finite_samples": int(table["quality_non_finite"].sum()),
        "peak_rss_bytes": peak_rss_bytes(),
        "seconds": round(time.time() - started, 1),
        "machine": machine_info(),
        "commit": git_commit(),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
