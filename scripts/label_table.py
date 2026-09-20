"""The label table: what each of the five corpora calls each of the five diagnoses.

Writes ``results/label_map.json``, the piece the rotation stands on.  It carries
the mapping itself (class to SNOMED codes to AHA codes to SCP acronyms), the
count of every class in every corpus, the same count as the Challenge's own
table publishes it, the difference between the two, every ambiguity the mapping
could not resolve, and every (corpus, class) cell the mapping refuses to fill.

Nothing in it is typed by hand.  The SNOMED codes come out of
``mappings/dx_mapping_scored.csv``, the AHA codes out of Leinonen's table, the
counts out of the corpora themselves, and the reference counts out of the
Challenge's published columns -- so a count that drifts shows up as a difference
rather than as a table that agrees with itself.

Usage: .venv/bin/python scripts/label_table.py
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ecs.challenge import (
    CHALLENGE_PARTITIONS,
    challenge_small_set_labels,
    completeness,
    join_ptbxl_to_challenge,
    partition_deviations,
    scan_source,
    sph_small_set_labels,
)
from ecs.config import MAPPINGS_DIR, PTBXL_DIR, RESULTS_DIR, SPH_DIR
from ecs.small_set import (
    AMBIGUITIES,
    SMALL_SET,
    aha_codes_for,
    aha_to_snomed,
    challenge_published_counts,
    refusals,
    scored_diagnoses,
    snomed_codes_for,
)

SPH_DEVIATIONS = (
    "diagnoses are AHA codes bridged to SNOMED by Leinonen et al.'s table, not the "
    "Challenge's own SNOMED annotation",
    "records run from ten to sixty seconds; the first ten are kept",
)


def commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()


def class_rows() -> list[dict[str, Any]]:
    """The mapping itself, one row per class."""
    rows = []
    for klass in SMALL_SET:
        rows.append(
            {
                "key": klass.key,
                "description": klass.description,
                "challenge_abbreviations": list(klass.abbreviations),
                "snomed": sorted(snomed_codes_for(klass.key)),
                "aha_base_codes": sorted(aha_codes_for(klass.key)),
                "scp_acronyms": list(klass.scp_acronyms),
                "fusion": klass.fusion,
            }
        )
    return rows


def counts_of(labels: pd.DataFrame) -> dict[str, int]:
    return {key: int(labels[key].sum()) for key in labels.columns}


def double_counted(table: pd.DataFrame) -> dict[str, int]:
    """Records carrying more than one code of a fused class.

    The Challenge's published count is a sum over rows of its table, so a record
    holding both halves of a fused pair is counted twice there and once here.
    This is the whole of the difference, and it is reported rather than absorbed.
    """
    out = {}
    for klass in SMALL_SET:
        if len(klass.abbreviations) < 2:
            continue
        wanted = snomed_codes_for(klass.key)
        out[klass.key] = int(sum(len(wanted & set(dx)) > 1 for dx in table["dx"]))
    return out


def challenge_corpus(source: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One Challenge source, scanned, counted and checked against the published table."""
    started = time.time()
    table = scan_source(source)
    labels = challenge_small_set_labels(table)
    counts = counts_of(labels)
    published = {klass.key: challenge_published_counts(source, klass.key) for klass in SMALL_SET}
    doubles = double_counted(table)
    complete = completeness(source, table)
    missing = complete["missing_from_this_copy"]
    n_missing = len(missing) if isinstance(missing, list) else 0
    unexplained = {key: published[key] - counts[key] - doubles.get(key, 0) for key in counts}
    block: dict[str, Any] = {
        "route": "SNOMED CT codes on the Challenge-2021 '# Dx:' header line",
        "partitions": list(CHALLENGE_PARTITIONS[source]),
        "n_records": int(len(table)),
        "n_patients": int(len(table)),
        "counts": counts,
        "published_counts": published,
        "published_minus_measured": {k: published[k] - counts[k] for k in counts},
        "records_carrying_both_codes_of_a_fused_class": doubles,
        "unexplained_by_double_counting": unexplained,
        "completeness": complete,
        "the_published_count_closes": all(0 <= gap <= n_missing for gap in unexplained.values()),
        "deviations": partition_deviations(source, table),
        "seconds": round(time.time() - started, 1),
    }
    return table, block


def ptbxl_block(challenge: pd.DataFrame) -> dict[str, Any]:
    """PTB-XL as the study reads it: its own waveforms and patients, the bundle's codes."""
    database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    joined, fell_out = join_ptbxl_to_challenge(database, challenge)
    labels = challenge_small_set_labels(joined)
    scp = database.loc[joined.index, "scp_codes"].apply(ast.literal_eval)
    agreement = {}
    for klass in SMALL_SET:
        by_snomed = labels[klass.key]
        by_scp = scp.apply(lambda entry, a=klass.scp_acronyms: any(k in entry for k in a))
        agreement[klass.key] = {
            "scp_acronyms": list(klass.scp_acronyms),
            "both": int((by_snomed & by_scp).sum()),
            "snomed_only": int((by_snomed & ~by_scp).sum()),
            "scp_only": int((~by_snomed & by_scp).sum()),
            "neither": int((~by_snomed & ~by_scp).sum()),
        }
    return {
        "n_records": int(len(joined)),
        "n_patients": int(joined["patient_id"].nunique()),
        "counts": counts_of(labels),
        "join": {
            "n_in_the_distribution": int(len(database)),
            "n_in_the_bundle": int(len(challenge)),
            "n_kept": int(len(joined)),
            "in_the_distribution_not_in_the_bundle": fell_out[
                "in_the_distribution_not_in_the_bundle"
            ],
            "in_the_bundle_not_in_the_distribution": fell_out[
                "in_the_bundle_not_in_the_distribution"
            ],
        },
        "snomed_against_scp": agreement,
    }


def sph_block() -> dict[str, Any]:
    """Shandong: AHA codes through Leinonen's table, with the modifier choice measured."""
    metadata = pd.read_csv(SPH_DIR / "metadata.csv")
    labels = sph_small_set_labels(metadata)
    exact_map = aha_to_snomed()
    tokens = (
        metadata["AHA_Code"]
        .astype(str)
        .apply(lambda raw: {t.strip() for t in raw.split(";") if t.strip()})
    )
    exact_counts = {}
    for klass in SMALL_SET:
        wanted_snomed = snomed_codes_for(klass.key)
        listed = set(exact_map.loc[exact_map["SNOMEDCTCode"].isin(wanted_snomed), "AHA_Code"])
        exact_counts[klass.key] = int(tokens.apply(lambda t, w=listed: bool(w & t)).sum())
    return {
        "route": "AHA codes bridged to SNOMED by Leinonen et al.'s published table",
        "n_records": int(len(metadata)),
        "n_patients": int(metadata["Patient_ID"].nunique()),
        "counts": counts_of(labels),
        "counts_matching_the_full_modifier_token": exact_counts,
        "deviations": list(SPH_DEVIATIONS),
    }


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    corpora: dict[str, Any] = {}
    challenge_tables: dict[str, pd.DataFrame] = {}
    for source in CHALLENGE_PARTITIONS:
        table, block = challenge_corpus(source)
        challenge_tables[source] = table
        corpora[source] = block
        print(f"  {source}: {block['n_records']} records, {block['seconds']} s", flush=True)
    corpora["ptbxl"]["as_read_by_this_study"] = ptbxl_block(challenge_tables["ptbxl"])
    corpora["sph"] = sph_block()

    out = {
        "written_by": "scripts/label_table.py",
        "commit": commit(),
        "mappings": {
            "directory": str(MAPPINGS_DIR.name),
            "scored_diagnoses_rows": int(len(scored_diagnoses())),
            "aha_rows": int(len(aha_to_snomed())),
        },
        "classes": class_rows(),
        "ambiguities": [asdict(a) for a in AMBIGUITIES],
        "refused_cells": [
            {"corpus": corpus, "class": key, "ambiguity": ambiguity.key}
            for (corpus, key), ambiguity in sorted(refusals().items())
        ],
        "corpora": corpora,
    }
    path = Path(RESULTS_DIR) / "label_map.json"
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
