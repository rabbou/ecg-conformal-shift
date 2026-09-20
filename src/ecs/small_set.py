"""The five diagnoses the source rotation is measured on, and what each corpus calls them.

The object's first measurement is one disease (infarction) on two hospitals.
Rotating the calibration source over five corpora needs a label set that all
five carry, that a cardiologist reads at a glance, and that survives Leinonen's
rule for a comparable class -- present in at least four sources with at least
fifty cases.  What is left is normal sinus rhythm, atrial fibrillation, left and
right bundle-branch block, and first-degree atrioventricular block.

Five corpora, three vocabularies.  Four of them (PTB-XL, Chapman-Shaoxing and
Ningbo, Georgia, CPSC 2018 and its extension) ship in the PhysioNet/CinC
Challenge-2021 bundle, whose headers carry SNOMED CT codes, so they share one
vocabulary already.  Shandong annotates in the AHA scheme and is bridged by
Leinonen et al.'s AHA-to-SNOMED table.  Both tables are in ``mappings/``, with
their provenance in ``mappings/NOTICE.md``, and every SNOMED code below is read
out of them rather than typed here: this module names classes, never numbers.

Five joins could not be made cleanly, and none of them is decided in silence.
``AMBIGUITIES`` carries all five with what was done and why, and
``refusals()`` says which corpus loses which class outright.  A cell the
mapping cannot fill is refused with its reason rather than filled with a class
that means something else there.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd

from .config import MAPPINGS_DIR

__all__ = [
    "AMBIGUITIES",
    "CHALLENGE_PARTITION_COLUMNS",
    "SMALL_SET",
    "Ambiguity",
    "SmallSetClass",
    "aha_codes_for",
    "aha_to_snomed",
    "challenge_published_counts",
    "class_of_key",
    "ptbxl_plus_statements",
    "refusals",
    "scored_diagnoses",
    "snomed_codes_for",
]


@dataclass(frozen=True)
class SmallSetClass:
    """One class of the small set, named by the Challenge abbreviations it unions."""

    key: str
    description: str
    abbreviations: tuple[str, ...]
    fusion: str | None = None  # why more than one abbreviation, when there is more than one
    scp_acronyms: tuple[str, ...] = ()  # PTB-XL's own SCP-ECG statements for the same finding


@dataclass(frozen=True)
class Ambiguity:
    """A join that could not be made cleanly, and what was done about it."""

    key: str
    what: str
    corpora: tuple[str, ...]
    decision: str
    why: str
    evidence: str
    refuses: tuple[tuple[str, str], ...] = ()  # (corpus, class key) pairs this empties


# The five classes.  Each names the rows of ``dx_mapping_scored.csv`` it unions;
# the SNOMED codes come from that file.
SMALL_SET: tuple[SmallSetClass, ...] = (
    SmallSetClass(
        "NSR",
        "sinus rhythm: the rhythm the sinus node drives, whatever else the tracing shows",
        ("NSR",),
        scp_acronyms=("SR",),
    ),
    SmallSetClass(
        "AF",
        "atrial fibrillation",
        ("AF",),
        scp_acronyms=("AFIB",),
    ),
    SmallSetClass(
        "LBBB",
        "left bundle-branch block",
        ("LBBB", "CLBBB"),
        fusion=(
            "the Challenge scores 164909002 and 733534002 as one diagnosis, in the Notes "
            "column of dx_mapping_scored.csv; unfused, the complete form exists in Ningbo "
            "alone and the class is unusable"
        ),
        scp_acronyms=("CLBBB",),
    ),
    SmallSetClass(
        "RBBB",
        "right bundle-branch block, complete or unqualified; the incomplete form is a "
        "separate Challenge class and stays out",
        ("RBBB", "CRBBB"),
        fusion=(
            "the Challenge scores 59118001 and 713427006 as one diagnosis, in the Notes "
            "column of dx_mapping_scored.csv; unfused, PTB-XL carries no record of the "
            "unqualified form and CPSC none of the complete one"
        ),
        scp_acronyms=("CRBBB",),
    ),
    SmallSetClass(
        "IAVB",
        "first-degree atrioventricular block, that is a PR interval longer than normal "
        "with every beat conducted",
        ("IAVB", "LPR"),
        fusion=(
            "not a fusion the Challenge declares: see the ambiguity 'iavb-lpr'. The two "
            "are one finding under two names, and Shandong has one code for both"
        ),
        scp_acronyms=("1AVB", "LPR"),
    ),
)

# The per-partition count columns of dx_mapping_scored.csv, and the corpus each
# belongs to here.  St Petersburg and PTB are in the file and not in this study.
CHALLENGE_PARTITION_COLUMNS: dict[str, tuple[str, ...]] = {
    "ptbxl": ("PTB_XL",),
    "chapman_ningbo": ("Chapman_Shaoxing", "Ningbo"),
    "georgia": ("Georgia",),
    "cpsc": ("CPSC", "CPSC_Extra"),
}


AMBIGUITIES: tuple[Ambiguity, ...] = (
    Ambiguity(
        key="sph-normal-versus-sinus",
        what=(
            "Shandong's AHA code 1 is 'Normal ECG', a statement about the whole tracing. "
            "The Challenge's NSR class is SNOMED 426783006 'sinus rhythm', a statement "
            "about the rhythm alone, which sits on abnormal tracings too: 18,092 of "
            "PTB-XL's 21,799 records carry it. Leinonen's AHA-to-SNOMED table has no row "
            "for code 1, so it offers no bridge either."
        ),
        corpora=("sph",),
        decision=(
            "NSR is refused on Shandong. Reading code 1 as sinus rhythm would put a "
            "narrower class on one corpus than on the other four, and a difference in "
            "what the label means is indistinguishable, downstream, from the difference "
            "between hospitals this study measures."
        ),
        why=(
            "13,907 of Shandong's 25,770 records carry code 1 (54%) against 83% NSR in "
            "PTB-XL; the gap is the definition, not the population."
        ),
        evidence=(
            "data/sph/code.csv row 'A,1,Normal ECG'; dx_mapping_scored.csv row 'sinus "
            "rhythm,426783006,NSR'; AHA_SNOMED_mapping.csv has no row with AHA_Code 1"
        ),
        refuses=(("sph", "NSR"),),
    ),
    Ambiguity(
        key="iavb-lpr",
        what=(
            "The Challenge scores 'prolonged pr interval' (164947007) and '1st degree av "
            "block' (270492004) as two classes and does not declare them equivalent. "
            "Shandong has one code for both, AHA 82 'Prolonged PR interval', which "
            "Leinonen's table maps to 270492004."
        ),
        corpora=("ptbxl", "chapman_ningbo", "georgia", "cpsc", "sph"),
        decision=(
            "The two Challenge classes are unioned into one IAVB class on all five "
            "corpora, so that the class denotes the same finding everywhere."
        ),
        why=(
            "A PR interval longer than 200 ms with every beat conducted is what "
            "first-degree atrioventricular block is; the two names are one finding. "
            "Keeping them apart would make the class mean 'prolonged PR' on Shandong and "
            "'prolonged PR minus whatever the annotator called by the other name' on the "
            "four Challenge corpora. The count each side contributes is reported, so the "
            "union can be undone by a reader who disagrees."
        ),
        evidence=(
            "dx_mapping_scored.csv rows 'prolonged pr interval,164947007,LPR' and '1st "
            "degree av block,270492004,IAVB', neither carrying an equivalence note; "
            "AHA_SNOMED_mapping.csv row '1st degree av block,270492004,82'; "
            "data/sph/code.csv row 'H,82,Prolonged PR interval'"
        ),
    ),
    Ambiguity(
        key="ptbxl-plus-is-omop",
        what=(
            "PTB-XL+'s SCP-to-SNOMED file maps each SCP-ECG acronym to OMOP concept "
            "identifiers, not to SNOMED CT codes. Its own apply_snomed_mapping.py joins "
            "them against Athena's CONCEPT.csv, which is not redistributable, so the file "
            "alone does not put PTB-XL's statements in the Challenge's vocabulary."
        ),
        corpora=("ptbxl",),
        decision=(
            "PTB-XL's SNOMED labels are read from the Challenge's own PTB-XL partition "
            "headers, the same route as the other three Challenge corpora. PTB-XL+ is "
            "used to name the clinical concept behind each SCP acronym, and the agreement "
            "between the two routes is reported per class rather than assumed."
        ),
        why=(
            "The alternative -- deriving SNOMED from SCP by hand -- would put PTB-XL on a "
            "mapping of my own while the other four corpora keep the publishers'."
        ),
        evidence=(
            "mappings/ptbxlToSNOMED.csv header 'Acronym,Dx Statement,id1,name1,...' with "
            "id values such as 4065279 for NORM; "
            "physionet.org/files/ptb-xl-plus/1.0.1/labels/mapping/apply_snomed_mapping.py "
            "reads CONCEPT.csv and CONCEPT_RELATIONSHIP.csv"
        ),
    ),
    Ambiguity(
        key="ptbxl-snomed-versus-scp",
        what=(
            "PTB-XL can be read two ways: the SNOMED codes the Challenge put on its "
            "records, and PTB-XL's own SCP-ECG statements. On the sinus-rhythm class the "
            "two disagree on a block of records that carry the SCP statement NORM without "
            "the statement SR, and whose rhythm statements are sinus arrhythmia, sinus "
            "bradycardia or sinus tachycardia -- sinus rhythms all. The Challenge's route "
            "counts them; a route keyed on the SR statement alone does not."
        ),
        corpora=("ptbxl",),
        decision=(
            "The SNOMED route is used, the same route as the other three Challenge "
            "corpora. The agreement between the two is measured per class and written to "
            "results/label_map.json under 'snomed_against_scp', so the size of the "
            "disagreement is on the record rather than asserted."
        ),
        why=(
            "Switching route for one corpus would put PTB-XL on a narrower rhythm class "
            "than its four neighbours, which is the same failure as reading Shandong's "
            "'Normal ECG' as sinus rhythm, in the other direction."
        ),
        evidence=(
            "results/label_map.json, corpora.ptbxl.as_read_by_this_study.snomed_against_scp; "
            "the disagreeing records all carry the SCP statement NORM"
        ),
    ),
    Ambiguity(
        key="aha-modifier-tokens",
        what=(
            "Shandong writes a diagnosis as a code with an optional modifier, '50+346' for "
            "atrial fibrillation with a rapid ventricular response. Leinonen's table lists "
            "the modifier variants its authors met ('50', '50+346', '50+347') and cannot "
            "list the ones they did not."
        ),
        corpora=("sph",),
        decision=(
            "A Shandong code is matched on its base code, before the '+', so a modifier "
            "the table does not list still counts as the diagnosis it modifies."
        ),
        why=(
            "The modifiers in Shandong's own table are rate, frequency and timing "
            "qualifiers, not different diagnoses. The count under exact matching is "
            "reported beside the count under base matching, so the choice is visible."
        ),
        evidence=(
            "data/sph/code.csv rows 330-367 are all listed under Category 'Modifier'; "
            "AHA_SNOMED_mapping.csv rows 'atrial fibrillation,164889003,50', '...,50+346', "
            "'...,50+347'"
        ),
    ),
)


@cache
def scored_diagnoses(root: Path = MAPPINGS_DIR) -> pd.DataFrame:
    """The Challenge-2021 scored-diagnosis table, indexed by abbreviation."""
    table = pd.read_csv(root / "dx_mapping_scored.csv", dtype={"SNOMEDCTCode": str})
    return table.set_index("Abbreviation")


@cache
def aha_to_snomed(root: Path = MAPPINGS_DIR) -> pd.DataFrame:
    """Leinonen's AHA-to-SNOMED table, as published."""
    return pd.read_csv(
        root / "AHA_SNOMED_mapping.csv", dtype={"SNOMEDCTCode": str, "AHA_Code": str}
    )


@cache
def ptbxl_plus_statements(root: Path = MAPPINGS_DIR) -> dict[str, str]:
    """PTB-XL+'s SCP acronym to the clinical statement it stands for.

    Read with the csv module rather than pandas: the file's rows are ragged --
    a statement mapped to four concepts carries more fields than the header
    declares -- and only the first two columns are wanted here.
    """
    out: dict[str, str] = {}
    with (root / "ptbxlToSNOMED.csv").open(encoding="utf-8-sig", newline="") as handle:
        for i, row in enumerate(csv.reader(handle)):
            if i == 0 or len(row) < 2:
                continue
            out[row[0].strip()] = row[1].strip()
    return out


def class_of_key(key: str) -> SmallSetClass:
    """The small-set class with this key."""
    for klass in SMALL_SET:
        if klass.key == key:
            return klass
    raise KeyError(f"{key} is not one of {[c.key for c in SMALL_SET]}")


def snomed_codes_for(key: str, root: Path = MAPPINGS_DIR) -> frozenset[str]:
    """The SNOMED CT codes a record may carry to be a positive of this class.

    Read out of ``dx_mapping_scored.csv`` by abbreviation, so the numbers live in
    the published table and not in this file.
    """
    table = scored_diagnoses(root)
    return frozenset(str(table.loc[a, "SNOMEDCTCode"]) for a in class_of_key(key).abbreviations)


def aha_codes_for(key: str, root: Path = MAPPINGS_DIR) -> frozenset[str]:
    """The Shandong AHA base codes that map to this class, via Leinonen's table.

    Empty when the class has no bridge, which is the case the caller has to
    refuse rather than fill -- see ``refusals()``.
    """
    wanted = snomed_codes_for(key, root)
    rows = aha_to_snomed(root)
    hit = rows[rows["SNOMEDCTCode"].isin(wanted)]
    return frozenset(str(code).split("+", 1)[0] for code in hit["AHA_Code"])


def refusals() -> dict[tuple[str, str], Ambiguity]:
    """(corpus, class) pairs no mapping can fill, each with the ambiguity that says why."""
    return {pair: a for a in AMBIGUITIES for pair in a.refuses}


def challenge_published_counts(corpus: str, key: str, root: Path = MAPPINGS_DIR) -> int:
    """What the Challenge's own table says this corpus holds of this class.

    The reference the label table is checked against (C-23).  A record carrying
    two of a fused pair would be counted twice by this sum; the check reports
    the double-counted records rather than hiding the difference.
    """
    table = scored_diagnoses(root)
    columns = CHALLENGE_PARTITION_COLUMNS[corpus]
    total = 0
    for abbreviation in class_of_key(key).abbreviations:
        for column in columns:
            total += int(str(table.loc[abbreviation, column]))
    return total


def positives(codes: Iterable[str], key: str, root: Path = MAPPINGS_DIR) -> bool:
    """Whether a record carrying these SNOMED codes is a positive of this class."""
    return bool(snomed_codes_for(key, root) & {str(c) for c in codes})
