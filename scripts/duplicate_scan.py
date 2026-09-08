"""Which tracings a corpus holds twice, so no split can put one on both sides.

Reads every record of the five rotation corpora through the common ingestion
chain, digests the canonical window (``ecs.duplicates.signal_digest``), and
writes the groups of records that share a digest to
``results/duplicate_groups.json``.

Only the groups are written, not the digests. Every corpus here is mostly
singletons, so the groups are a few hundred records against a hundred thousand,
and the splitter needs nothing else: a record no group names is its own unit.

Two questions, not one. A tracing repeated *inside* a corpus defeats that
corpus's split, which is what the widened key fixes. A tracing shared *between*
two corpora would defeat the rotation itself -- the source would be scored on a
recording the target had trained on -- and no per-corpus split can see it. Both
are counted here.

Where ``ecg-data-chain`` has already published a digest for the same record,
this compares the two and records how many agree. That repository computes the
digest from its own reader; a disagreement would mean one of the two ingestion
chains is not producing the window it says it is.

Usage: .venv/bin/python scripts/duplicate_scan.py [--compare PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ecs.config import RESULTS_DIR
from ecs.duplicates import MICROVOLTS_PER_STEP, duplicate_groups, signal_digest
from ecs.rotation import SOURCES, corpus_index, load_waveforms

CHUNK = 500

# Where the same record is named in ecg-data-chain's cache, so the two digests
# can be compared. Its identifiers carry the distribution as a prefix.
COMPARABLE = {
    "ptbxl": ("physionet/ptb-xl", lambda record: f"{int(record):05d}_hr"),
    "chapman_ningbo": ("challenge-2021/chapman_shaoxing", lambda record: record),
    "georgia": ("challenge-2021/georgia", lambda record: record),
    "cpsc": ("challenge-2021/cpsc_2018", lambda record: record),
}


def commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def digests_of(corpus: str) -> dict[str, str]:
    """Every record of one corpus, digested from the canonical window."""
    index = corpus_index(corpus)
    wanted = [str(i) for i in index.frame.index]
    out: dict[str, str] = {}
    started = time.time()
    for start in range(0, len(wanted), CHUNK):
        x, ids = load_waveforms(index, wanted[start : start + CHUNK])
        for record, window in zip(ids, x, strict=True):
            out[record] = signal_digest(window)
        print(
            f"  {corpus}: {len(out):>6} / {len(wanted)}  {time.time() - started:6.0f} s",
            flush=True,
        )
    return out


def against_the_delivery_corpus(
    corpus: str, digests: dict[str, str], cache: dict[str, Any] | None
) -> dict[str, Any]:
    """How many digests agree with the ones ecg-data-chain published."""
    if cache is None or corpus not in COMPARABLE:
        return {"compared": 0, "agree": 0, "note": "no published digest to compare against"}
    distribution, to_their_id = COMPARABLE[corpus]
    theirs = {key.split(":", 1)[1]: value for key, value in cache.get(distribution, {}).items()}
    compared = agree = 0
    for record, digest in digests.items():
        try:
            key = to_their_id(record)
        except (TypeError, ValueError):
            continue
        if key in theirs:
            compared += 1
            agree += int(theirs[key] == digest)
    return {"compared": compared, "agree": agree, "distribution": distribution}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare",
        default="/home/ruben/Developer/ecg-data-chain/results/cache/signal_digests.json",
        help="a published digest cache to check this one against, when present",
    )
    parser.add_argument("--out", default=str(RESULTS_DIR / "duplicate_groups.json"))
    args = parser.parse_args(argv)

    published = Path(args.compare)
    cache = json.loads(published.read_text()) if published.exists() else None

    started = time.time()
    every: dict[str, dict[str, str]] = {}
    out: dict[str, Any] = {
        "written_by": "scripts/duplicate_scan.py",
        "commit": commit(),
        "screen": (
            "the first ten seconds of the twelve canonical leads, quantised to "
            f"{MICROVOLTS_PER_STEP:.0f} microvolts, SHA-256; the screen of "
            "ecg-data-chain 4bff859"
        ),
        "corpora": {},
    }
    for corpus in SOURCES:
        digests = digests_of(corpus)
        every[corpus] = digests
        groups = duplicate_groups(digests)
        out["corpora"][corpus] = {
            "n_records": len(digests),
            "n_distinct_tracings": len(set(digests.values())),
            "n_groups": len(groups),
            "n_records_in_a_group": sum(len(m) for m in groups.values()),
            "groups": [members for _digest, members in sorted(groups.items())],
            "against_the_delivery_corpus": against_the_delivery_corpus(corpus, digests, cache),
        }
        block = out["corpora"][corpus]
        print(
            f"{corpus}: {block['n_records']} records, {block['n_distinct_tracings']} distinct, "
            f"{block['n_groups']} groups over {block['n_records_in_a_group']} records",
            flush=True,
        )

    # A digest carried by two corpora is one recording in both of them.
    where: dict[str, list[str]] = {}
    for corpus, digests in every.items():
        for record, digest in digests.items():
            where.setdefault(digest, []).append(f"{corpus}:{record}")
    across = [
        sorted(members)
        for members in where.values()
        if len({member.split(":", 1)[0] for member in members}) > 1
    ]
    out["across_corpora"] = {
        "n_groups": len(across),
        "groups": across[:200],
        "reading": (
            "a group here would be a tracing two corpora both hold, which no split "
            "inside one corpus can separate; the rotation would score a source on a "
            "recording its target had trained on"
        ),
    }
    print(f"groups spanning two corpora: {len(across)}", flush=True)

    out["seconds"] = round(time.time() - started, 1)
    path = Path(args.out)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"written {path} in {out['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
