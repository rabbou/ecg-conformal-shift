"""The MIMIC-IV-ECG label table: machine statements to the rotation's diagnoses.

Reads ``record_list.csv`` and ``machine_measurements.csv`` from the release and
writes ``results/mimic_label_map.json``:

- the count of tracings and patients per class, under the positive rule and
  under the strict rule (definite statements only);
- every join that could go the other way, with the number of tracings it moves;
- every distinct statement that touches a class, with its count and status;
- the cart's own PR and QRS durations per status, which is the evidence two of
  the decisions rest on;
- the digests of the two files, checked against the release's SHA256SUMS.txt.

Usage: .venv/bin/python scripts/mimic_label_table.py   (about two minutes)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecs.config import MIMIC_ECG_DIR, RESULTS_DIR
from ecs.mimic import (
    AMBIGUITIES,
    CLASSES,
    POSITIVE_STATUSES,
    REPORT_COLUMNS,
    RULES,
    STATUSES,
    matching_rules,
    normalise,
    read_tables,
    record_labels,
    record_statuses,
    statement_counts,
)

OUT = Path(RESULTS_DIR) / "mimic_label_map.json"
SOURCE_FILES = ("record_list.csv", "machine_measurements.csv")

# The cart writes 29999 where it could not measure an interval.
MISSING_INTERVAL = 29999

# Statements that say the cart could not read the tracing, counted so a reader
# knows how many machine labels rest on a recording the machine itself doubted.
QUALITY_PATTERNS = {
    "data_quality_warning": r"data quality may affect interpretation",
    "unsuitable_for_analysis": r"unsuitable for analysis",
    "lead_reversal_suspected": r"lead reversal",
    "paced_no_further_analysis": r"pacemaker rhythm - no further analysis",
}


def commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()


def digests(root: Path) -> dict[str, dict[str, str | bool]]:
    published: dict[str, str] = {}
    sums = root / "SHA256SUMS.txt"
    if sums.exists():
        for line in sums.read_text().splitlines():
            digest, _, name = line.partition(" ")
            if name.strip() in SOURCE_FILES:
                published[name.strip()] = digest
    out: dict[str, dict[str, str | bool]] = {}
    for name in SOURCE_FILES:
        measured = hashlib.sha256((root / name).read_bytes()).hexdigest()
        out[name] = {
            "sha256": measured,
            "matches_release_sha256sums": published.get(name) == measured,
        }
    return out


def intervals(frame: pd.DataFrame) -> pd.DataFrame:
    def column(name: str) -> pd.Series:
        values = pd.to_numeric(frame[name], errors="coerce")
        return values.where(values < MISSING_INTERVAL)

    return pd.DataFrame(
        {
            "pr": column("qrs_onset") - column("p_onset"),
            "qrs": column("qrs_end") - column("qrs_onset"),
        }
    )


def summary(values: pd.Series, threshold: float) -> dict[str, Any]:
    clean = values.dropna()
    if clean.empty:
        return {"n_measured": 0}
    return {
        "n_measured": int(clean.size),
        "median": float(clean.median()),
        "q10": float(clean.quantile(0.10)),
        "q90": float(clean.quantile(0.90)),
        f"share_above_{int(threshold)}": round(float((clean > threshold).mean()), 4),
        f"share_at_least_{int(threshold)}": round(float((clean >= threshold).mean()), 4),
    }


# The interval a decision about each class rests on, and the textbook bound:
# first-degree block is a PR above 200 ms, a complete bundle-branch block a QRS
# of 120 ms or more.
CLASS_INTERVALS = {"iavb": ("pr", 200.0), "bbb": ("qrs", 120.0)}


def ambiguity_intervals(key: str, records: set[str], measured: pd.DataFrame) -> dict[str, Any]:
    prefix = key.split("-")[0]
    if prefix not in CLASS_INTERVALS:
        return {}
    interval, threshold = CLASS_INTERVALS[prefix]
    values = measured.loc[sorted(records), interval]
    return {"measured_intervals_ms": {"interval": interval, **summary(values, threshold)}}


def main() -> int:
    started = time.time()
    root = Path(MIMIC_ECG_DIR)
    frame = read_tables(root)
    statuses = record_statuses(frame)
    labels = record_labels(statuses)
    strict = record_labels(statuses, {k: frozenset({"definite"}) for k in CLASSES})
    measured = intervals(frame)
    patients = frame["subject_id"]
    n = len(frame)

    classes: dict[str, Any] = {}
    for klass in CLASSES:
        table = statuses[klass]
        positive = labels[klass]
        by_status = {s: int(table[s].sum()) for s in STATUSES}
        hedged_only = table["hedged"] & ~positive
        interval = "pr" if klass == "IAVB" else "qrs"
        threshold = 200.0 if klass == "IAVB" else 120.0
        classes[klass] = {
            "positive_statuses": sorted(POSITIVE_STATUSES[klass]),
            "n_records": int(positive.sum()),
            "n_patients": int(patients[positive].nunique()),
            "prevalence": round(float(positive.mean()), 5),
            "strict_n_records": int(strict[klass].sum()),
            "records_carrying_status": by_status,
            "records_hedged_and_not_positive": int(hedged_only.sum()),
            "measured_intervals_ms": {
                "interval": interval,
                "by_status": {
                    s: summary(measured.loc[table[s], interval], threshold)
                    for s in STATUSES
                    if by_status[s]
                },
                "no_statement_for_this_class": summary(
                    measured.loc[~table.any(axis=1), interval], threshold
                ),
            },
        }

    # Per ambiguity, the tracings carrying at least one statement its rules decide.
    counts = statement_counts(frame)
    statement_rows = []
    by_ambiguity: dict[str, set[str]] = {a.key: set() for a in AMBIGUITIES}
    long = frame[list(REPORT_COLUMNS)].stack().map(normalise)
    records_of = long.reset_index(level=1, drop=True).groupby(level=0).agg(set)
    rules_of = {text: matching_rules(text) for text in counts.index}
    for record, texts in records_of.items():
        for text in texts:
            for rule in rules_of.get(text, {}).values():
                if rule.ambiguity:
                    by_ambiguity[rule.ambiguity].add(str(record))
    for text, count in counts.items():
        rules = rules_of[text]
        if rules:
            statement_rows.append(
                {
                    "statement": text,
                    "n_records": int(count),
                    "status": {k: r.status for k, r in rules.items()},
                }
            )

    texts = frame[list(REPORT_COLUMNS)].fillna("").astype(str).agg(" | ".join, axis=1)
    texts = texts.str.lower()
    quality = {k: int(texts.str.contains(p, regex=True).sum()) for k, p in QUALITY_PATTERNS.items()}

    output = {
        "written_by": "scripts/mimic_label_table.py",
        "commit": commit(),
        "release": "MIMIC-IV-ECG v1.0 (PhysioNet, ODbL 1.0), doi:10.13026/4nqg-sb35",
        "label_source": (
            "machine statements report_0..report_17, written by the cart's own "
            "interpretation software; no cardiologist label"
        ),
        "files": digests(root),
        "n_records": n,
        "n_patients": int(patients.nunique()),
        "n_distinct_statements": int(counts.size),
        "n_distinct_statements_touching_a_class": len(statement_rows),
        "classes": classes,
        "ambiguities": [
            {
                "key": a.key,
                "what": a.what,
                "decision": a.decision,
                "why": a.why,
                "evidence": a.evidence,
                "n_records": len(by_ambiguity[a.key]),
                **ambiguity_intervals(a.key, by_ambiguity[a.key], measured),
            }
            for a in AMBIGUITIES
        ],
        "quality_statements": quality,
        "rules": [
            {"class": r.klass, "status": r.status, "pattern": r.pattern, "ambiguity": r.ambiguity}
            for r in RULES
        ],
        "statements": statement_rows,
        "seconds": round(time.time() - started, 1),
    }
    OUT.write_text(json.dumps(output, indent=1, ensure_ascii=False) + "\n")
    for klass in CLASSES:
        c = classes[klass]
        print(
            f"{klass:7s} {c['n_records']:>7} records ({c['prevalence']:.3%}), "
            f"strict {c['strict_n_records']:>7}, "
            f"hedged only {c['records_hedged_and_not_positive']:>6}"
        )
    print(f"wrote {OUT} in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
