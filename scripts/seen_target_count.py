"""How much infarction the PhysioNet 2021 bundle holds outside PTB-XL.

The grid measures contamination on the *calibration* side: two arms saw PTB-XL
in pre-training and two did not.  A target corpus the same arms had also seen
would measure it on the other side, and the Challenge-2021 bundle on the box is
the obvious candidate because ECG-FM and HuBERT-ECG both name it.

Two conditions decide whether that cell is worth building, and both are settled
on the files rather than argued:

*The target may not contain the calibration corpus.*  The bundle ships PTB-XL,
PTB and St Petersburg INCART alongside the partitions of interest.  A target
holding PTB-XL would put calibration records inside the population the frozen
threshold is spent on, which is not a shift measurement at all.  So the
partitions are an allow-list here, and asking for a forbidden one is an error
rather than a filtered-out no-op.

*The target must actually carry infarction.*  Coverage is reported per class
(C-11), so a target whose infarction class is a handful of records measures
nothing: the class-conditional figure would be noise however many draws it is
averaged over.  This counts the records carrying an infarction code, per
partition, and the count is what the decision rests on.

The count is checkable against the Challenge's own published tally: the
organisers' ``dx_mapping_scored.csv`` and ``dx_mapping_unscored.csv`` give, per
SNOMED code, how many records of each partition carry it.  Reproducing those
columns from the headers is the oracle -- a parse that disagrees with the
published table is wrong, whatever it reports.

Usage: .venv/bin/python scripts/seen_target_count.py --bundle ~/data/challenge2021
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from ecs.config import CHALLENGE2021_DIR, RESULTS_DIR
from ecs.seen_target import (
    ALLOWED_PARTITIONS,
    INFARCTION_CODES,
    ISCHAEMIA_CODES,
    PARTITIONS_HOLDING_CALIBRATION,
    SNOMED_NAMES,
    PartitionCount,
    assess,
    count_partition,
    dx_codes,
)

__all__ = ["main"]


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def iter_headers(root: Path) -> Iterator[Path]:
    yield from sorted(root.rglob("*.hea"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        default=str(CHALLENGE2021_DIR),
        help="the Challenge-2021 root, the directory holding training/",
    )
    parser.add_argument("--out", default=str(RESULTS_DIR / "seen_target.json"))
    args = parser.parse_args(argv)

    training = Path(args.bundle).expanduser() / "training"
    counts: dict[str, PartitionCount] = {}
    per_code: dict[str, Counter[str]] = {}
    for name in ALLOWED_PARTITIONS:
        directory = training / name
        if not directory.is_dir():
            raise SystemExit(f"{directory} is not there; is --bundle the Challenge-2021 root?")
        records = [dx_codes(path) for path in iter_headers(directory)]
        counts[name] = count_partition(records)
        tally: Counter[str] = Counter()
        for codes in records:
            tally.update(codes)
        per_code[name] = tally
        print(f"{name}: {counts[name].n_records} records, {counts[name].n_infarction} infarction")

    named = [p for p in ALLOWED_PARTITIONS if p != "cpsc_2018_extra"]
    result: dict[str, Any] = {
        "bundle": str(Path(args.bundle).expanduser()),
        "git_commit": git_commit(),
        "partitions_read": list(ALLOWED_PARTITIONS),
        "partitions_refused": {
            name: "holds the calibration corpus or a corpus derived from it"
            for name in PARTITIONS_HOLDING_CALIBRATION
        },
        "infarction_codes": {code: SNOMED_NAMES[code] for code in sorted(INFARCTION_CODES)},
        "ischaemia_codes": {code: SNOMED_NAMES[code] for code in sorted(ISCHAEMIA_CODES)},
        "by_partition": {name: count.as_dict() for name, count in counts.items()},
        "by_partition_by_code": {
            name: {code: n for code, n in sorted(tally.items()) if code in SNOMED_NAMES}
            for name, tally in per_code.items()
        },
        "readings": {
            "four_partitions_named": assess(counts, named),
            "with_cpsc_extra": assess(counts, ALLOWED_PARTITIONS),
        },
    }
    result["decision"] = {
        "seen_target_cell": "kept"
        if any(r["holds"] for r in result["readings"].values())
        else "dropped",
        "rule": (
            "a seen target has to carry at least as much infarction as the thinnest target "
            "already in the study, and no single partition may supply more than half of it; "
            "both conditions and the numbers that decided them are in readings"
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result["readings"], "decision": result["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
