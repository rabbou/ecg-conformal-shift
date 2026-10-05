"""MIMIC-IV-ECG read as a sixth corpus: machine statements to labels, patients to parts.

MIMIC-IV-ECG v1.0 holds 800,035 ten-second ECGs from Beth Israel Deaconess,
Boston, 2008-2019, for 161,352 patients.  It carries no cardiologist label.
What it carries is the cart's own interpretation, up to eighteen free-text
statements per tracing in ``report_0`` to ``report_17`` of
``machine_measurements.csv``, written by machines from three manufacturers
(Burdick/Spacelabs, Philips, General Electric).  Every label read here is
therefore a machine label, and is named as one wherever it is used.

The statements are mapped onto the rotation's five diagnoses, and onto old
infarction, by the rules in :data:`RULES`.  A rule gives a statement one status
per class:

``definite``          the cart states the finding;
``borderline``        the cart states it with its own "borderline" grade;
``age_undetermined``  an infarct the cart states without dating it;
``hedged``            "possible", "probable", "consider", "cannot rule out";
``excluded``          a neighbouring finding that is not this class.

Which statuses make a positive label is :data:`POSITIVE_STATUSES`, and every
choice there that a reader could make the other way is in :data:`AMBIGUITIES`
with its count, so the alternative can be rebuilt.

Patients: ``subject_id`` is the patient key.  MIMIC holds five ECGs per patient
on average against about one in the other five corpora, so the rotation index
keeps one ECG per patient, drawn with a fixed seed.

Dates: every ``ecg_time`` is shifted by a per-patient offset into the twenty-
second century, and the cart's clock was not synchronised with anything.  The
real year can only be recovered with the credentialed MIMIC-IV patients table
(:func:`real_year_bounds`).  :func:`cart_chronology` orders the 156 carts in time
from open data alone, which is what the open shadow run stands on.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from numpy.typing import NDArray
from scipy.sparse.linalg import lsqr

from .small_set import Ambiguity

__all__ = [
    "AMBIGUITIES",
    "CLASSES",
    "POSITIVE_STATUSES",
    "REPORT_COLUMNS",
    "RULES",
    "STATUSES",
    "Rule",
    "cart_chronology",
    "estimated_eras",
    "estimated_years",
    "real_eras",
    "shadow_cohorts",
    "classify",
    "matching_rules",
    "normalise",
    "one_per_patient",
    "real_year_bounds",
    "record_labels",
    "read_tables",
    "record_statuses",
    "statement_counts",
]

REPORT_COLUMNS = tuple(f"report_{i}" for i in range(18))

# The rotation's five classes in their order, then old infarction, which is in
# the table for the discharge-diagnosis comparison and not in the rotation.
CLASSES = ("NSR", "AF", "LBBB", "RBBB", "IAVB", "MI_OLD")

STATUSES = ("definite", "borderline", "age_undetermined", "hedged", "excluded")

# The statuses a positive label is made of, per class.  IAVB takes the cart's
# borderline grade because the cart's own PR interval is above 200 ms on 97.5%
# of those tracings (results/mimic_label_map.json, measured_intervals_ms); MI_OLD
# takes an undated infarct because the cart has no third option between "old"
# and "acute" and the acute statements are excluded.
POSITIVE_STATUSES: dict[str, frozenset[str]] = {
    "NSR": frozenset({"definite"}),
    "AF": frozenset({"definite"}),
    "LBBB": frozenset({"definite"}),
    "RBBB": frozenset({"definite"}),
    "IAVB": frozenset({"definite", "borderline"}),
    "MI_OLD": frozenset({"definite", "age_undetermined"}),
}

_SPACES = re.compile(r"\s+")


def normalise(statement: str) -> str:
    """Lower case, one space between words, no trailing full stop.

    The three manufacturers write the same statement in upper case, in sentence
    case and with or without a final full stop; nothing else is rewritten.
    """
    text = _SPACES.sub(" ", str(statement)).strip().lower()
    return text.rstrip(". ").strip()


@dataclass(frozen=True)
class Rule:
    """One pattern and the status it gives a statement for one class."""

    klass: str
    status: str
    pattern: str
    ambiguity: str | None = None  # the AMBIGUITIES key the rule belongs to

    def matches(self, text: str) -> bool:
        return re.search(self.pattern, text) is not None


_HEDGE = r"(?:possibl[ey]|probabl[ey]|consider|cannot rule out|questionable|suggests?|\?)"

# Per class, the first rule that matches decides.  Order is the logic: a hedge
# or a neighbouring finding is tested before the bare name, so "incomplete
# left bundle branch block" never reaches the rule for "left bundle branch block".
RULES: tuple[Rule, ...] = (
    # Sinus rhythm.  The Challenge's NSR is SNOMED 426783006 "sinus rhythm"; sinus
    # bradycardia, tachycardia and arrhythmia are other SNOMED classes.
    Rule("NSR", "hedged", rf"{_HEDGE}\s+(?:normal\s+)?sinus rhythm", "nsr-hedged"),
    Rule("NSR", "excluded", r"sinus or ectopic", "nsr-sinus-or-ectopic"),
    Rule("NSR", "definite", r"\b(?:normal\s+)?sinus rhythm\b"),
    Rule(
        "NSR",
        "excluded",
        r"sinus (?:bradycardia|tachycardia|arrhythmia)",
        "nsr-rate-variants",
    ),
    # Atrial fibrillation.
    Rule("AF", "excluded", r"flutter\s*/\s*fib|fib\w*\s*/\s*flutter", "af-or-flutter"),
    Rule("AF", "hedged", rf"{_HEDGE}\s+(?:atrial fibrillation|afib|a-fib)", "af-hedged"),
    Rule("AF", "definite", r"atrial fibrillation|\bafib\b|\ba-fib\b"),
    # Left bundle-branch block.
    Rule(
        "LBBB",
        "excluded",
        r"incomplete\s+(?:lbbb|left bundle)|\bilbbb\b",
        "bbb-incomplete",
    ),
    Rule(
        "LBBB",
        "excluded",
        r"(?:lbbb|left bundle branch block)\s+type|atypical\s+lbbb",
        "bbb-type-conduction-defect",
    ),
    Rule("LBBB", "definite", r"\blbbb\b|left bundle branch block"),
    # Right bundle-branch block.
    Rule(
        "RBBB",
        "excluded",
        r"incomplete\s+(?:rbbb|right bundle)|\birbbb\b",
        "bbb-incomplete",
    ),
    Rule(
        "RBBB",
        "excluded",
        r"(?:rbbb|right bundle branch block)\s+type|atypical\s+rbbb",
        "bbb-type-conduction-defect",
    ),
    Rule("RBBB", "definite", r"\brbbb\b|right bundle branch block"),
    # First-degree atrioventricular block, unioned with prolonged PR as in the
    # rotation (small_set.AMBIGUITIES, "iavb-lpr").
    Rule(
        "IAVB",
        "borderline",
        r"borderline\s+(?:(?:1st|first) degree a-?v block|prolonged pr)",
        "iavb-borderline",
    ),
    Rule("IAVB", "definite", r"(?:1st|first) degree a-?v block|prolonged pr\b"),
    # Old infarction.  Acute and injury statements are a different finding.
    Rule(
        "MI_OLD",
        "excluded",
        r"infarct.*(?:acute|recent)|(?:acute|recent).*(?:infarct|\bmi\b)|\bstemi\b",
        "mi-acute",
    ),
    Rule(
        "MI_OLD",
        "hedged",
        rf"{_HEDGE}.*infarct|infarct.*{_HEDGE}|may be due.*infarct",
        "mi-hedged",
    ),
    Rule("MI_OLD", "definite", r"infarct\w*.*\bold\b|\bold\b.*infarct"),
    Rule(
        "MI_OLD",
        "age_undetermined",
        r"infarct\w*.*age (?:undetermined|indeterminate)",
        "mi-age-undetermined",
    ),
)


def matching_rules(statement: str) -> dict[str, Rule]:
    """The rule that decides each class this statement touches."""
    text = normalise(statement)
    out: dict[str, Rule] = {}
    for rule in RULES:
        if rule.klass not in out and rule.matches(text):
            out[rule.klass] = rule
    return out


def classify(statement: str) -> dict[str, str]:
    """The status of one statement for each class it touches."""
    return {klass: rule.status for klass, rule in matching_rules(statement).items()}


def statement_counts(frame: pd.DataFrame) -> pd.Series:
    """How many tracings carry each normalised statement, counted once per tracing."""
    long = cast(pd.Series, frame[list(REPORT_COLUMNS)].stack()).map(normalise)
    pairs = long[long != ""].reset_index(level=1, drop=True).reset_index()
    pairs.columns = pd.Index(["record", "statement"])
    return pairs.drop_duplicates()["statement"].value_counts()


def record_statuses(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per class, a (records x status) boolean frame: which statuses each record carries."""
    texts = cast(pd.Series, frame[list(REPORT_COLUMNS)].stack()).map(normalise)
    lookup = {text: classify(text) for text in pd.unique(texts.to_numpy())}
    rows = texts.index.get_level_values(0)
    out: dict[str, pd.DataFrame] = {}
    for klass in CLASSES:
        status = texts.map({t: v[klass] for t, v in lookup.items() if klass in v})
        table = pd.DataFrame(False, index=frame.index, columns=list(STATUSES))
        for name in STATUSES:
            hit = rows[(status == name).to_numpy()]
            table.loc[pd.Index(hit).unique(), name] = True
        out[klass] = table
    return out


def record_labels(
    statuses: Mapping[str, pd.DataFrame],
    positive: Mapping[str, frozenset[str]] = POSITIVE_STATUSES,
) -> pd.DataFrame:
    """One boolean column per class: does any statement give a positive status."""
    return pd.DataFrame(
        {k: statuses[k][sorted(positive[k])].any(axis=1) for k in CLASSES},
        index=next(iter(statuses.values())).index,
    )


def one_per_patient(patients: pd.Series, seed: int) -> pd.Index:
    """One record per patient, drawn uniformly with ``seed``, whatever the row order."""
    patients = patients.sort_index()
    order = np.random.default_rng(seed).permutation(len(patients))
    shuffled = patients.iloc[order]
    keep = ~shuffled.duplicated(keep="first")
    return shuffled.index[keep.to_numpy()].sort_values()


def real_year_bounds(
    shifted_year: pd.Series, anchor_year: pd.Series, anchor_year_group: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """The earliest and latest real year a tracing can fall in.

    MIMIC-IV shifts each patient by one offset and publishes, per patient, the
    shifted ``anchor_year`` and the three-year ``anchor_year_group`` the real
    anchor year lies in ("2008 - 2010" ... "2017 - 2019").  A tracing taken
    ``d = shifted_year - anchor_year`` years after the anchor therefore lies in
    [group_start + d, group_end + d].  Bounds outside 2008-2019 mean the cart's
    clock was wrong; they are returned as they are and the caller drops them.
    """
    bounds = anchor_year_group.str.extract(r"(\d{4})\s*-\s*(\d{4})").astype(int)
    delta = shifted_year.to_numpy(dtype=int) - anchor_year.to_numpy(dtype=int)
    index = anchor_year_group.index
    return (
        pd.Series(bounds[0].to_numpy() + delta, index=index),
        pd.Series(bounds[1].to_numpy() + delta, index=index),
    )


@dataclass(frozen=True)
class CartChronology:
    """Each cart's mean position in time, in years, relative to the median cart."""

    position: pd.Series  # index cart_id
    n_ecgs: pd.Series  # index cart_id
    n_patients_used: int
    residual_sd_years: float


# A patient whose tracings span more than the twelve years the release covers
# carries at least one cart clock that was wrong; such patients would pull a
# cart's position by decades and are left out of the fit.
MAX_PLAUSIBLE_SPAN_YEARS = 12.0


def _years(times: pd.Series) -> NDArray[np.float64]:
    stamps = pd.to_datetime(times)
    return (stamps.dt.year + (stamps.dt.dayofyear - 1) / 365.25).to_numpy(dtype=float)


def cart_chronology(frame: pd.DataFrame) -> CartChronology:
    """Order the carts in time from within-patient differences alone.

    The date shift is one offset per patient, so the difference between two of
    one patient's tracings is real.  If cart ``c`` was in use around time
    ``mu_c``, a patient's tracing on ``c`` sits at ``mu_c`` plus that patient's
    offset plus noise; centring each patient's times removes the offset, and a
    least-squares fit of the centred times on centred cart indicators gives
    every ``mu_c`` up to one constant.  The median cart is set to zero.

    ``frame`` needs ``subject_id``, ``cart_id`` and ``ecg_time``.
    """
    data = pd.DataFrame(
        {
            "subject": frame["subject_id"].to_numpy(),
            "cart": frame["cart_id"].to_numpy(),
            "t": _years(frame["ecg_time"]),
        }
    )
    grouped = data.groupby("subject")["t"]
    span = grouped.transform("max") - grouped.transform("min")
    size = grouped.transform("size")
    used = data[(size >= 2) & (span <= MAX_PLAUSIBLE_SPAN_YEARS)].reset_index(drop=True)
    carts = np.sort(data["cart"].unique())
    cart_index = pd.Series(np.arange(len(carts)), index=carts)
    patient = used["subject"].astype("category").cat.codes.to_numpy()
    n = len(used)
    onehot = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), cart_index[used["cart"]].to_numpy())),
        shape=(n, len(carts)),
    )
    member = sparse.csr_matrix((np.ones(n), (patient, np.arange(n))))
    counts = np.asarray(member.sum(axis=1)).ravel()
    means = sparse.diags(1.0 / counts) @ member @ onehot
    design = (onehot - means[patient]).tocsr()
    centred = used["t"].to_numpy() - used.groupby("subject")["t"].transform("mean").to_numpy()
    solution = lsqr(design, centred, damp=1e-6, atol=1e-10, btol=1e-10)[0]
    residual = centred - design @ solution
    in_fit = np.asarray(np.abs(design).sum(axis=0)).ravel() > 0
    position = pd.Series(np.where(in_fit, solution, np.nan), index=carts)
    position -= float(np.nanmedian(position.to_numpy()))
    return CartChronology(
        position=position,
        n_ecgs=data["cart"].value_counts().reindex(carts).fillna(0).astype(int),
        n_patients_used=int(used["subject"].nunique()),
        residual_sd_years=float(residual.std()),
    )


def read_tables(root: Path) -> pd.DataFrame:
    """``record_list.csv`` joined to ``machine_measurements.csv`` on ``study_id``."""
    records = pd.read_csv(root / "record_list.csv")
    measures = pd.read_csv(root / "machine_measurements.csv", low_memory=False)
    joined = records.merge(
        measures.drop(columns=["subject_id", "ecg_time"]), on="study_id", validate="1:1"
    )
    return joined.set_index(joined["study_id"].astype(str))


AMBIGUITIES: tuple[Ambiguity, ...] = (
    Ambiguity(
        key="nsr-rate-variants",
        what=(
            "The carts write 'Sinus bradycardia', 'Sinus tachycardia' and 'Sinus "
            "arrhythmia' as rhythm statements of their own, without 'sinus rhythm'."
        ),
        corpora=("mimic",),
        decision="Not NSR.",
        why=(
            "The rotation's NSR is SNOMED 426783006 'sinus rhythm'; bradycardia, "
            "tachycardia and arrhythmia of the sinus node are the separate SNOMED "
            "classes SB, STach and SA in the same Challenge table, and the four "
            "Challenge corpora label them apart."
        ),
        evidence="mappings/dx_mapping_scored.csv rows NSR, SB, STach, SA",
    ),
    Ambiguity(
        key="nsr-sinus-or-ectopic",
        what="'Sinus or ectopic atrial rhythm' and its rate variants.",
        corpora=("mimic",),
        decision="Not NSR.",
        why="The cart does not decide between a sinus and an atrial focus.",
        evidence="machine_measurements.csv statements",
    ),
    Ambiguity(
        key="nsr-hedged",
        what="'consider sinus rhythm with 2:1 S-A block' and other hedged sinus statements.",
        corpora=("mimic",),
        decision="Not NSR.",
        why="A hedged rhythm is not a stated rhythm.",
        evidence="machine_measurements.csv statements",
    ),
    Ambiguity(
        key="af-or-flutter",
        what="'Atrial flutter/fibrillation', 'Afib/flutter', 'A-flutter/fibrillation'.",
        corpora=("mimic",),
        decision="Not AF.",
        why=(
            "The cart cannot tell flutter from fibrillation, and flutter is the "
            "separate Challenge class AFL."
        ),
        evidence="machine_measurements.csv statements; dx_mapping_scored.csv rows AF, AFL",
    ),
    Ambiguity(
        key="af-hedged",
        what="'Probable atrial fibrillation', 'Possible atrial fibrillation'.",
        corpora=("mimic",),
        decision="Not AF.",
        why=(
            "The positive label stays what the cart asserts. The count is given "
            "so a reader can rebuild the class with the hedged tracings in."
        ),
        evidence="machine_measurements.csv statements",
    ),
    Ambiguity(
        key="bbb-incomplete",
        what="'Incomplete RBBB', 'Incomplete left bundle branch block', 'ILBBB'.",
        corpora=("mimic",),
        decision="Not LBBB or RBBB.",
        why=(
            "The rotation's classes are the complete or unqualified blocks; the "
            "incomplete forms are separate Challenge classes (IRBBB) and stay out "
            "on the other five corpora too."
        ),
        evidence="src/ecs/small_set.py, SMALL_SET RBBB description",
    ),
    Ambiguity(
        key="bbb-type-conduction-defect",
        what=(
            "'Conduction defect of LBBB type', 'IVCD, consider atypical RBBB' and the "
            "upper-case 'I.V. conduction defect of ... bundle branch block type'."
        ),
        corpora=("mimic",),
        decision="Not LBBB or RBBB.",
        why=(
            "The cart measures a wide QRS (120 ms or more on 96% of these tracings, "
            "median 140 ms) and declines to call it a bundle-branch block; the Challenge "
            "class for that is IVCD (nonspecific intraventricular conduction "
            "disorder), not the block."
        ),
        evidence="results/mimic_label_map.json, measured_intervals_ms",
    ),
    Ambiguity(
        key="iavb-borderline",
        what=(
            "'Sinus rhythm with borderline 1st degree A-V block' and 'Borderline "
            "prolonged PR interval', the cart's own grade below 'first degree'."
        ),
        corpora=("mimic",),
        decision="IAVB.",
        why=(
            "First-degree block is a PR interval above 200 ms with every beat "
            "conducted. The cart's own PR measurement on these tracings has a median "
            "of 208 ms and is above 200 ms on 97.5% of them, so they meet the "
            "definition the rotation's IAVB class states. The class roughly doubles "
            "with them; the strict count is given beside it."
        ),
        evidence="results/mimic_label_map.json, measured_intervals_ms and strict_counts",
    ),
    Ambiguity(
        key="mi-age-undetermined",
        what=(
            "'Inferior infarct - age undetermined' and the like: the cart's commonest "
            "infarct statement, with no age."
        ),
        corpora=("mimic",),
        decision="MI_OLD.",
        why=(
            "The cart's infarct statements are old, undated or acute, and the acute "
            "ones are excluded; an undated infarct pattern on a resting ECG is the "
            "Q-wave pattern that ICD-10 I25.2 (old myocardial infarction) describes. "
            "The count with old statements only is given beside it."
        ),
        evidence="machine_measurements.csv statements; results/mimic_label_map.json",
    ),
    Ambiguity(
        key="mi-acute",
        what="'CONSIDER ACUTE ST ELEVATION MI', 'INFERIOR INFARCT - POSSIBLY ACUTE'.",
        corpora=("mimic",),
        decision="Not MI_OLD.",
        why="Acute injury is a different finding from an old infarct.",
        evidence="machine_measurements.csv statements",
    ),
    Ambiguity(
        key="mi-hedged",
        what="'Possible anterior infarct - age undetermined', 'Cannot rule out septal infarct'.",
        corpora=("mimic",),
        decision="Not MI_OLD.",
        why=(
            "The positive label stays what the cart asserts; 'possible' infarct "
            "statements outnumber the definite ones and are counted apart."
        ),
        evidence="machine_measurements.csv statements",
    ),
)


# The seed the one-per-patient draw uses: the rotation's own, so the MIMIC rows
# of the rotation are fixed by the same number as every other corpus's split.
ONE_PER_PATIENT_SEED = 27

ROTATION_DEVIATIONS = (
    "labels are the cart's machine statements mapped by src/ecs/mimic.py, not a "
    "cardiologist's reading; results/mimic_label_map.json names every join",
    "one ECG per patient, drawn with a fixed seed from the patient's 1 to 260 tracings",
    "age and sex are in the credentialed MIMIC-IV patients table, not in the open "
    "release, so every MIMIC record is in the 'unknown' age band and sex",
    "lead order I, II, III, aVR, aVF, aVL, V1-V6 in the release, reordered by header name",
    "200 ADC units per mV: 5 microvolt resolution against PTB-XL's 1",
)


def rotation_frame(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One ECG per patient, with its path, cart and labels.

    Returns the record frame (index ``study_id`` as text; columns ``patient``,
    ``path``, ``age``, ``sex``, ``cart_id``, ``ecg_time``, ``bandwidth``) and the
    label frame (one boolean column per class of :data:`CLASSES`).
    """
    records = pd.read_csv(root / "record_list.csv")
    records.index = pd.Index(records["study_id"].astype(str))
    kept = one_per_patient(records["subject_id"], ONE_PER_PATIENT_SEED)
    records = records.loc[kept]
    measures = pd.read_csv(root / "machine_measurements.csv", low_memory=False)
    measures.index = pd.Index(measures["study_id"].astype(str))
    measures = measures.loc[kept]
    labels = record_labels(record_statuses(measures))
    frame = pd.DataFrame(
        {
            "patient": records["subject_id"].astype(str),
            "path": [str(root / p) for p in records["path"]],
            "age": np.nan,
            "sex": "unknown",
            "cart_id": measures["cart_id"],
            "ecg_time": records["ecg_time"],
            "bandwidth": measures["bandwidth"],
        },
        index=records.index,
    )
    return frame, labels


# The shadow run: thresholds fitted on the earliest third of the tracings and
# spent on the latest third, against the same thresholds spent on held-out
# patients of the earliest third.  Sizes match the rotation's calibration and
# test caps so the two read on the same precision.
SHADOW_SIZES = {"early_cal": 2000, "early_test": 4000, "late_test": 8000}
SHADOW_SEED = 66  # the task this run belongs to


def estimated_years(measures: pd.DataFrame, chronology: CartChronology) -> pd.Series:
    """Each tracing's place in time, in years relative to the median cart.

    A patient's offset is estimated as the median, over all of that patient's
    tracings, of shifted time minus cart position; the tracing's time is its
    shifted time minus that offset.  A patient seen on several carts is placed by
    all of them, which is what makes this tighter than the cart position alone.
    ``measures`` needs ``study_id``, ``subject_id``, ``cart_id`` and ``ecg_time``,
    for all of the release's tracings, not only the ones being placed.
    """
    shifted = pd.Series(_years(measures["ecg_time"]), index=measures.index)
    offset = shifted - measures["cart_id"].map(chronology.position)
    offset = offset.groupby(measures["subject_id"]).transform("median")
    estimate = shifted - offset
    estimate.index = pd.Index(measures["study_id"].astype(str))
    return estimate


def estimated_eras(estimate: pd.Series) -> pd.Series:
    """``early``, ``middle`` or ``late``: the terciles of ``estimate`` over its own records."""
    low, high = np.nanquantile(estimate.to_numpy(dtype=float), [1 / 3, 2 / 3])
    era = np.where(estimate <= low, "early", np.where(estimate >= high, "late", "middle"))
    return pd.Series(np.where(estimate.isna(), "unknown", era), index=estimate.index)


def real_eras(
    frame: pd.DataFrame, patients: pd.DataFrame, early_last: int = 2011, late_first: int = 2016
) -> pd.Series:
    """``early`` when a tracing surely lies in 2008..early_last, ``late`` in late_first..2019.

    Needs the MIMIC-IV patients table (``subject_id``, ``anchor_year``,
    ``anchor_year_group``); a patient absent from it is ``unknown``.  A tracing
    whose bounds leave 2008-2019 has a wrong cart clock and is ``unknown`` too.
    """
    table = patients.set_index(patients["subject_id"].astype(str))
    subject = frame["patient"].astype(str)
    known = subject.isin(table.index)
    era = pd.Series("unknown", index=frame.index)
    if not known.any():
        return era
    rows = table.loc[subject[known]]
    shifted = pd.to_datetime(frame.loc[known, "ecg_time"]).dt.year
    low, high = real_year_bounds(
        pd.Series(shifted.to_numpy(), index=rows.index),
        rows["anchor_year"],
        rows["anchor_year_group"],
    )
    low_v, high_v = low.to_numpy(), high.to_numpy()
    plausible = (low_v >= 2006) & (high_v <= 2021)
    values = np.where(
        plausible & (low_v >= 2008) & (high_v <= early_last),
        "early",
        np.where(plausible & (low_v >= late_first) & (high_v <= 2019), "late", "unknown"),
    )
    era.loc[known] = values
    return era


def shadow_cohorts(eligible: pd.Index, era: pd.Series) -> dict[str, list[str]]:
    """The three shadow cohorts, disjoint, one record per patient, drawn with a seed.

    ``eligible`` is the records whose patients the model never trained or
    stopped on; ``era`` labels them.
    """
    rng = np.random.default_rng(SHADOW_SEED)
    eras = era.loc[eligible]
    early = np.sort(eras.index[eras == "early"].to_numpy())
    late = np.sort(eras.index[eras == "late"].to_numpy())
    early = early[rng.permutation(early.size)]
    late = late[rng.permutation(late.size)]
    n_cal, n_test = SHADOW_SIZES["early_cal"], SHADOW_SIZES["early_test"]
    return {
        "early_cal": [str(i) for i in early[:n_cal]],
        "early_test": [str(i) for i in early[n_cal : n_cal + n_test]],
        "late_test": [str(i) for i in late[: SHADOW_SIZES["late_test"]]],
    }
