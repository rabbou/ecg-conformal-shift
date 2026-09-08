"""The same tracing under two record identifiers, and what it does to a split.

A patient-level split protects against one patient's two tracings landing on
opposite sides of a boundary.  It does not protect against the same *tracing*
appearing twice under different record identifiers, because nothing in the
corpus says the two are one patient.  Three of the Challenge-2021 partitions
repeat tracings inside themselves; CPSC 2018 with its extension repeats about
one record in eight.  Where a repeat straddles the train and test boundary, the
model is scored on a recording it was fitted on.

The screen is the one the delivery corpus uses (``ecg-data-chain``, commit
4bff859, ``src/ecgchain/quality.py``): quantise the first ten seconds of the
twelve canonical leads to ten microvolts and take a SHA-256 of the result, so
two packagings of one recording at different ADC gains give one digest.  The
function is copied rather than imported because this repository has to build
from its own checkout; ``test_duplicates.py`` holds the copy to the digests that
repository published, so the two cannot drift apart silently.

What this does *not* claim: that two records with one digest are one patient.
That is a different question.  What it establishes is narrower and enough --
they are one recording, so they belong on one side of any boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MICROVOLTS_PER_STEP",
    "duplicate_groups",
    "grouping_key",
    "signal_digest",
    "straddling_groups",
]

# The digest's quantum, from ecg-data-chain's quality.py.  Ten microvolts is
# coarser than any corpus's own resolution here (PTB-XL stores one microvolt per
# step, Chongqing two), so a re-gained copy of one recording still lands on one
# digest.
MICROVOLTS_PER_STEP = 10.0


def signal_digest(window: NDArray[np.float32]) -> str:
    """A digest of a canonical (12, 5000) window, quantised to ten microvolts.

    A sample that is not finite becomes the smallest representable step rather
    than zero: a gap and a flat line at zero millivolts are different things and
    the digest must not confuse them.

    Copied from ecg-data-chain 4bff859, ``ecgchain.quality.signal_digest``.
    """
    scaled = window * (1000.0 / MICROVOLTS_PER_STEP)
    steps = np.where(np.isfinite(scaled), np.rint(scaled), np.iinfo(np.int32).min).astype(np.int32)
    return hashlib.sha256(steps.tobytes()).hexdigest()


def duplicate_groups(digests: Mapping[str, str]) -> dict[str, list[str]]:
    """Digest to the records carrying it, for the digests carried more than once."""
    groups: dict[str, list[str]] = {}
    for record, digest in digests.items():
        groups.setdefault(digest, []).append(record)
    return {digest: sorted(members) for digest, members in groups.items() if len(members) > 1}


def grouping_key(patients: Mapping[str, str], groups: Iterable[Sequence[str]]) -> dict[str, str]:
    """The unit a split may not cut: a patient, widened to swallow repeats.

    Two records of one group are one recording and have to share a key.  When
    they are already filed under different patients, both patients join the same
    unit, because splitting them would put the recording on both sides whatever
    the patient key said.  The result is the transitive closure over "same
    patient" and "same recording", which is the coarser of the two and the only
    one that holds.

    A group naming a record this corpus does not hold -- one the ingestion chain
    dropped -- contributes the members that are left, and a group reduced to one
    member joins nothing.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for patient in patients.values():
        find(f"patient:{patient}")
    for members in groups:
        present = [record for record in members if record in patients]
        for other in present[1:]:
            union(f"patient:{patients[present[0]]}", f"patient:{patients[other]}")
    return {record: find(f"patient:{patient}") for record, patient in patients.items()}


def straddling_groups(
    parts: Mapping[str, str], groups: Iterable[Sequence[str]], ignore: Iterable[str] = ()
) -> list[list[str]]:
    """The groups of identical tracings whose members sit in two different parts.

    ``ignore`` names parts that do not count as a side -- the records no part
    uses -- so the reading is about boundaries a figure is actually read across.
    """
    skipped = set(ignore)
    out: list[list[str]] = []
    for members in groups:
        sides = {parts[m] for m in members if m in parts} - skipped
        if len(sides) > 1:
            out.append(list(members))
    return out
