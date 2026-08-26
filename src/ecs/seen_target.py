"""Reading infarction out of the PhysioNet Challenge-2021 headers.

The bundle annotates each record with SNOMED CT concept identifiers on a
``# Dx:`` line of its WFDB header.  Which of those identifiers count as
infarction is the only judgement here, and it is made once, in
:data:`INFARCTION_CODES`.

SNOMED separates *infarction* -- tissue that died -- from *ischaemia* --
tissue deprived of blood but not yet dead.  The three corpora
already in this study label infarction (PTB-XL's MI superclass, Shandong's AHA
category M, Chongqing's angiographic AMI column), so infarction is the class a
seen target would have to carry, and the ischaemia codes are counted separately
rather than folded in to make the total look healthier.

The names beside the codes are the Challenge's own, from the organisers'
``dx_mapping_scored.csv`` and ``dx_mapping_unscored.csv``
(github.com/physionetchallenges/evaluation-2021, read 2026-08-26).  Those two
files also publish, per code, how many records of each partition carry it,
which is what ``test_seen_target.py`` holds this parse to.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ALLOWED_PARTITIONS",
    "INFARCTION_CODES",
    "ISCHAEMIA_CODES",
    "PARTITIONS_HOLDING_CALIBRATION",
    "SNOMED_NAMES",
    "PartitionCount",
    "assess",
    "count_partition",
    "dx_codes",
    "parse_dx_line",
]

# The partitions a target may be assembled from: every partition of the bundle
# that is not PTB-XL and is not a corpus PTB-XL was drawn from or into.
ALLOWED_PARTITIONS = ("chapman_shaoxing", "cpsc_2018", "cpsc_2018_extra", "georgia", "ningbo")

# The partitions that are refused outright.  ptb-xl IS the calibration corpus;
# ptb is the earlier Physikalisch-Technische Bundesanstalt corpus PTB-XL
# supersedes and shares its provenance with; st_petersburg_incart is neither,
# but at 74 records it cannot carry a class-conditional figure and is left out
# for that reason rather than for contamination.
PARTITIONS_HOLDING_CALIBRATION = ("ptb-xl", "ptb")

# Infarction: tissue that died.
INFARCTION_CODES = frozenset(
    {
        "164865005",  # myocardial infarction
        "164867002",  # old myocardial infarction
        "57054005",  # acute myocardial infarction
        "54329005",  # anterior myocardial infarction
    }
)

# Ischaemia: tissue deprived of blood.  Counted, never added to the infarction class.
ISCHAEMIA_CODES = frozenset(
    {
        "164861001",  # myocardial ischemia
        "413444003",  # acute myocardial ischemia
        "413844008",  # chronic myocardial ischemia
        "425419005",  # inferior ischaemia
        "425623009",  # lateral ischaemia
        "426434006",  # anterior ischemia
    }
)

SNOMED_NAMES = {
    "164865005": "myocardial infarction",
    "164867002": "old myocardial infarction",
    "57054005": "acute myocardial infarction",
    "54329005": "anterior myocardial infarction",
    "164861001": "myocardial ischemia",
    "413444003": "acute myocardial ischemia",
    "413844008": "chronic myocardial ischemia",
    "425419005": "inferior ischaemia",
    "425623009": "lateral ischaemia",
    "426434006": "anterior ischemia",
}


# The two conditions a seen target has to meet, as numbers rather than as a
# judgement made after the counts were seen.
#
# ENOUGH_INFARCTION -- coverage is reported per class (C-11), so the infarction
# class is the binding support, and a new target that carries less of it than
# the thinnest target already in the study measures less than that target does.
# Shandong is that floor: 260 infarction records of 25,770 (PLAN.md, C-2, the
# count `test_labels.py::TestSPH` holds the loader to).
#
# MAX_SHARE_FROM_ONE_PARTITION -- a target glued from partitions of very
# different prevalence has the prevalence of whichever partition was glued on,
# not the prevalence of a population.  A coverage gap measured against it would
# be a fact about the assembly, which is the one thing this study exists not to
# produce.  If one partition supplies more than half the infarction, the target
# is that partition wearing a larger record count.
ENOUGH_INFARCTION = 260
MAX_SHARE_FROM_ONE_PARTITION = 0.5


@dataclass(frozen=True)
class PartitionCount:
    """One partition: how many records, and how many carry each family."""

    n_records: int
    n_infarction: int
    n_ischaemia: int
    n_infarction_or_ischaemia: int

    @property
    def infarction_prevalence(self) -> float:
        return self.n_infarction / self.n_records if self.n_records else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_records": self.n_records,
            "n_infarction": self.n_infarction,
            "n_ischaemia": self.n_ischaemia,
            "n_infarction_or_ischaemia": self.n_infarction_or_ischaemia,
            "infarction_prevalence": self.infarction_prevalence,
        }


def parse_dx_line(text: str) -> list[str]:
    """The SNOMED identifiers on a header's ``# Dx:`` line, in the order given.

    Returns an empty list for a header with no such line: the bundle ships a
    handful, and a record with no diagnosis is a record with no infarction, not
    an error.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        body = stripped.lstrip("#").strip()
        key, separator, value = body.partition(":")
        if separator and key.strip().lower() == "dx":
            return [code.strip() for code in value.split(",") if code.strip()]
    return []


def dx_codes(header: Path) -> list[str]:
    return parse_dx_line(header.read_text(errors="replace"))


def assess(counts: Mapping[str, PartitionCount], partitions: Sequence[str]) -> dict[str, Any]:
    """Whether a target assembled from ``partitions`` could carry the cell.

    Returns the totals, both conditions with the numbers that decided them, and
    the verdict.  Nothing here is a judgement typed after the fact: the two
    thresholds are fixed above, and the answer falls out of the counts.
    """
    chosen = {name: counts[name] for name in partitions}
    n_records = sum(c.n_records for c in chosen.values())
    n_infarction = sum(c.n_infarction for c in chosen.values())
    largest = max(chosen, key=lambda name: chosen[name].n_infarction)
    share = chosen[largest].n_infarction / n_infarction if n_infarction else 0.0
    enough = n_infarction >= ENOUGH_INFARCTION
    from_a_population = share <= MAX_SHARE_FROM_ONE_PARTITION
    return {
        "partitions": list(partitions),
        "n_records": n_records,
        "n_infarction": n_infarction,
        "n_ischaemia": sum(c.n_ischaemia for c in chosen.values()),
        "n_infarction_or_ischaemia": sum(c.n_infarction_or_ischaemia for c in chosen.values()),
        "infarction_prevalence": n_infarction / n_records if n_records else 0.0,
        "enough_infarction": {
            "holds": enough,
            "n_infarction": n_infarction,
            "floor": ENOUGH_INFARCTION,
            "floor_is": "Shandong's infarction count, the thinnest target already in the study",
        },
        "prevalence_is_a_population_not_an_assembly": {
            "holds": from_a_population,
            "largest_contributor": largest,
            "share_of_infarction": share,
            "ceiling": MAX_SHARE_FROM_ONE_PARTITION,
        },
        "holds": enough and from_a_population,
    }


def count_partition(records: Iterable[Sequence[str]]) -> PartitionCount:
    """One partition's records, each as its list of codes.

    A record is counted once however many codes of a family it carries: the
    question is how many *tracings* a target would offer the infarction class,
    not how many annotations they hold between them.
    """
    n_records = n_infarction = n_ischaemia = n_either = 0
    for codes in records:
        held = set(codes)
        infarction = bool(held & INFARCTION_CODES)
        ischaemia = bool(held & ISCHAEMIA_CODES)
        n_records += 1
        n_infarction += infarction
        n_ischaemia += ischaemia
        n_either += infarction or ischaemia
    return PartitionCount(n_records, n_infarction, n_ischaemia, n_either)
