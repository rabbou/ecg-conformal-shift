"""MIMIC-IV-ECG as a sixth corpus: the statement rules, the table they build, the eras.

The rule tests are an oracle written by hand from the statements as the carts
print them, not from the patterns.  The table tests read the committed
``results/mimic_label_map.json``; rebuild it with

    .venv/bin/python scripts/mimic_label_table.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ecs.config import MIMIC_ECG_DIR, RESULTS_DIR
from ecs.mimic import (
    AMBIGUITIES,
    CLASSES,
    POSITIVE_STATUSES,
    RULES,
    STATUSES,
    cart_chronology,
    classify,
    estimated_eras,
    estimated_years,
    normalise,
    one_per_patient,
    real_eras,
    real_year_bounds,
    record_labels,
    record_statuses,
    shadow_cohorts,
)

PATH = Path(RESULTS_DIR) / "mimic_label_map.json"

# Statement as printed by a cart -> the status it must get, per class.
ORACLE: dict[str, dict[str, str]] = {
    "Sinus rhythm": {"NSR": "definite"},
    "SINUS RHYTHM": {"NSR": "definite"},
    "Sinus rhythm with borderline 1st degree A-V block.": {
        "NSR": "definite",
        "IAVB": "borderline",
    },
    "Sinus bradycardia with 1st degree A-V block": {"NSR": "excluded", "IAVB": "definite"},
    "Sinus tachycardia": {"NSR": "excluded"},
    "Sinus or ectopic atrial rhythm": {"NSR": "excluded"},
    "Sinus bradycardia - consider sinus rhythm with 2:1 S-A block.": {"NSR": "hedged"},
    "Atrial fibrillation with rapid ventricular response": {"AF": "definite"},
    "ATRIAL FIBRILLATION": {"AF": "definite"},
    "Probable atrial fibrillation with PVC(s)": {"AF": "hedged"},
    "Atrial flutter/fibrillation": {"AF": "excluded"},
    "Afib/flutter and ventricular-paced rhythm": {"AF": "excluded"},
    "Atrial flutter with 4:1 A-V block": {},
    "Left bundle branch block": {"LBBB": "definite"},
    "Incomplete LBBB": {"LBBB": "excluded"},
    "Conduction defect of LBBB type": {"LBBB": "excluded"},
    "Anterior Q waves, possibly due to ILBBB": {"LBBB": "excluded"},
    "RBBB with left anterior fascicular block": {"RBBB": "definite"},
    "RBBB with RAD - possible left posterior fascicular block": {"RBBB": "definite"},
    "Incomplete right bundle branch block": {"RBBB": "excluded"},
    "IVCD, consider atypical RBBB": {"RBBB": "excluded"},
    "I.V. CONDUCTION DEFECT OF RIGHT BUNDLE BRANCH BLOCK TYPE": {"RBBB": "excluded"},
    "Prolonged PR interval": {"IAVB": "definite"},
    "Borderline prolonged PR interval": {"IAVB": "borderline"},
    "- first degree A-V block": {"IAVB": "definite"},
    "Short PR interval": {},
    "Second degree AV block, Mobitz II": {},
    "Inferior infarct, old": {"MI_OLD": "definite"},
    "Old inferior infarct": {"MI_OLD": "definite"},
    "Inferior infarct - age undetermined": {"MI_OLD": "age_undetermined"},
    "Anteroseptal infarct, age indeterminate": {"MI_OLD": "age_undetermined"},
    "Possible anterior infarct - age undetermined": {"MI_OLD": "hedged"},
    "Possible old inferior infarct": {"MI_OLD": "hedged"},
    "QRS changes V3/V4 may be due to LVH but cannot rule out anterior infarct": {
        "MI_OLD": "hedged"
    },
    "*** CONSIDER ACUTE ST ELEVATION MI ***": {"MI_OLD": "excluded"},
    "*** INFERIOR INFARCT - POSSIBLY ACUTE ***": {"MI_OLD": "excluded"},
    "Abnormal ECG": {},
}


class TestTheRules:
    @pytest.mark.parametrize("statement", sorted(ORACLE))
    def test_each_statement_gets_the_status_a_reader_would_give_it(self, statement: str) -> None:
        assert classify(statement) == ORACLE[statement]

    def test_normalising_ignores_case_spacing_and_a_final_full_stop(self) -> None:
        assert normalise("  Sinus  rhythm. ") == normalise("SINUS RHYTHM") == "sinus rhythm"

    def test_every_rule_names_a_known_class_status_and_ambiguity(self) -> None:
        keys = {a.key for a in AMBIGUITIES}
        for rule in RULES:
            assert rule.klass in CLASSES and rule.status in STATUSES
            assert rule.ambiguity is None or rule.ambiguity in keys, rule

    def test_every_ambiguity_is_decided_by_at_least_one_rule(self) -> None:
        assert {a.key for a in AMBIGUITIES} == {r.ambiguity for r in RULES if r.ambiguity}

    def test_a_record_is_positive_when_any_statement_is(self) -> None:
        frame = pd.DataFrame(
            {
                "report_0": ["Sinus rhythm", "Probable atrial fibrillation", "Sinus tachycardia"],
                "report_1": [
                    "with borderline 1st degree A-V block",
                    "Inferior infarct - age undetermined",
                    None,
                ],
            },
            index=["a", "b", "c"],
        )
        for i in range(2, 18):
            frame[f"report_{i}"] = None
        labels = record_labels(record_statuses(frame))
        assert list(labels.loc["a", ["NSR", "IAVB"]]) == [True, True]
        assert list(labels.loc["b"]) == [False, False, False, False, False, True]
        assert not labels.loc["c"].any()
        definite_only = {k: frozenset({"definite"}) for k in CLASSES}
        strict = record_labels(record_statuses(frame), definite_only)
        assert not strict.loc["a", "IAVB"] and not strict.loc["b", "MI_OLD"]

    def test_positive_statuses_are_statuses(self) -> None:
        for klass, statuses in POSITIVE_STATUSES.items():
            assert klass in CLASSES and statuses <= set(STATUSES)


class TestPatientsAndYears:
    def test_one_record_per_patient_whatever_the_row_order(self) -> None:
        patients = pd.Series(["p1", "p1", "p2", "p3", "p3", "p3"], index=list("abcdef"))
        kept = one_per_patient(patients, seed=27)
        assert sorted(patients[kept]) == ["p1", "p2", "p3"]
        shuffled = patients.iloc[np.array([5, 2, 0, 4, 1, 3])]
        again = one_per_patient(shuffled, seed=27)
        assert list(kept) == list(again)

    def test_real_year_bounds_carry_the_anchor_group_and_the_shift(self) -> None:
        # Anchor 2150 lies in real 2011-2013; a tracing three shifted years later
        # lies in real 2014-2016.
        low, high = real_year_bounds(
            pd.Series([2153, 2150]), pd.Series([2150, 2150]), pd.Series(["2011 - 2013"] * 2)
        )
        assert low.tolist() == [2014, 2011] and high.tolist() == [2016, 2013]

    def test_real_eras_keep_only_tracings_surely_inside_a_window(self) -> None:
        frame = pd.DataFrame(
            {
                "patient": ["1", "2", "3", "4", "5"],
                "ecg_time": ["2150-01-01", "2160-06-01", "2150-01-01", "2150-01-01", "2199-01-01"],
            },
            index=list("abcde"),
        )
        patients = pd.DataFrame(
            {
                "subject_id": [1, 2, 3, 5],
                "anchor_year": [2150, 2155, 2150, 2150],
                "anchor_year_group": ["2008 - 2010", "2011 - 2013", "2011 - 2013", "2008 - 2010"],
            }
        )
        era = real_eras(frame, patients)
        # a: 2008-2010 early; b: 2016-2018 late; c: 2011-2013 straddles 2011;
        # d: not in the table; e: 2057-2059, a wrong clock.
        assert era.tolist() == ["early", "late", "unknown", "unknown", "unknown"]


def _synthetic_release(seed: int = 0) -> tuple[pd.DataFrame, dict[int, float]]:
    """Twelve carts at known times, patients shifted by large random offsets."""
    rng = np.random.default_rng(seed)
    carts = {100 + c: 2008.0 + c for c in range(12)}
    rows = []
    study = 0
    for patient in range(3000):
        offset = rng.uniform(100, 200)
        for _ in range(rng.integers(1, 5)):
            cart = int(rng.integers(100, 112))
            real = carts[cart] + rng.normal(0, 0.3)
            shifted = real + offset
            year = int(shifted)
            day = int((shifted - year) * 365) + 1
            stamp = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day - 1)
            rows.append(
                {
                    "study_id": study,
                    "subject_id": patient,
                    "cart_id": cart,
                    "ecg_time": stamp.strftime("%Y-%m-%d"),
                    "real": real,
                }
            )
            study += 1
    return pd.DataFrame(rows), carts


class TestTheCartTimeline:
    def test_carts_are_placed_where_they_were_in_time(self) -> None:
        frame, carts = _synthetic_release()
        chronology = cart_chronology(frame)
        truth = pd.Series(carts) - float(np.median(list(carts.values())))
        recovered = chronology.position.loc[truth.index]
        assert np.abs(recovered - truth).max() < 0.15

    def test_a_patient_whose_tracings_span_decades_is_left_out(self) -> None:
        frame, _ = _synthetic_release()
        clean = cart_chronology(frame).position
        broken = frame.copy()
        broken.loc[broken["subject_id"] == 0, "ecg_time"] = "2299-01-01"
        broken.loc[broken.index[0], "ecg_time"] = "2101-01-01"
        assert np.allclose(cart_chronology(broken).position, clean, atol=0.05)

    def test_the_estimate_orders_tracings_by_their_real_time(self) -> None:
        frame, _ = _synthetic_release()
        estimate = estimated_years(frame, cart_chronology(frame))
        real = frame.set_index(frame["study_id"].astype(str))["real"]
        rho = pd.Series(estimate).corr(real.loc[estimate.index], method="spearman")
        assert rho > 0.95

    def test_eras_are_terciles(self) -> None:
        era = estimated_eras(pd.Series(np.arange(9.0), index=list("abcdefghi")))
        assert era.tolist() == ["early"] * 3 + ["middle"] * 3 + ["late"] * 3


class TestTheShadowCohorts:
    def test_cohorts_are_disjoint_eligible_and_of_the_stated_sizes(self) -> None:
        ids = [str(i) for i in range(30000)]
        era = pd.Series(["early", "late", "middle"] * 10000, index=ids)
        eligible = pd.Index(ids[:27000])
        cohorts = shadow_cohorts(eligible, era)
        assert {k: len(v) for k, v in cohorts.items()} == {
            "early_cal": 2000,
            "early_test": 4000,
            "late_test": 8000,
        }
        seen: set[str] = set()
        for name, members in cohorts.items():
            assert not seen & set(members), name
            seen |= set(members)
            assert set(members) <= set(eligible)
        assert set(era[cohorts["late_test"]]) == {"late"}
        assert set(era[cohorts["early_cal"] + cohorts["early_test"]]) == {"early"}
        assert cohorts == shadow_cohorts(eligible, era)


@pytest.fixture(scope="module")
def table() -> dict[str, Any]:
    if not PATH.exists():
        pytest.skip(f"{PATH} is not built; run scripts/mimic_label_table.py")
    return dict(json.loads(PATH.read_text()))


# The counts the committed table holds, pinned so a change of rule that moves
# them has to change this line too.
PINNED_RECORDS = {
    "NSR": 461068,
    "AF": 78069,
    "LBBB": 23451,
    "RBBB": 57360,
    "IAVB": 66689,
    "MI_OLD": 70388,
}


class TestTheCommittedTable:
    def test_it_covers_the_whole_release_read_from_digest_checked_files(
        self, table: dict[str, Any]
    ) -> None:
        assert table["n_records"] == 800035 and table["n_patients"] == 161352
        for name, entry in table["files"].items():
            assert entry["matches_release_sha256sums"], name

    def test_the_count_per_diagnosis_is_the_pinned_one(self, table: dict[str, Any]) -> None:
        counts = {k: table["classes"][k]["n_records"] for k in CLASSES}
        assert counts == PINNED_RECORDS

    def test_the_strict_count_never_exceeds_the_positive_one(self, table: dict[str, Any]) -> None:
        for klass in CLASSES:
            block = table["classes"][klass]
            assert block["strict_n_records"] <= block["n_records"]
            assert block["n_patients"] <= block["n_records"]
            if POSITIVE_STATUSES[klass] == {"definite"}:
                assert block["strict_n_records"] == block["n_records"]

    def test_every_ambiguous_join_is_listed_with_its_count(self, table: dict[str, Any]) -> None:
        listed = {a["key"]: a for a in table["ambiguities"]}
        assert set(listed) == {a.key for a in AMBIGUITIES}
        for key, entry in listed.items():
            assert entry["n_records"] > 0, key
            assert entry["decision"] and entry["why"], key

    def test_the_statement_table_is_what_the_rules_give_today(self, table: dict[str, Any]) -> None:
        for row in table["statements"]:
            assert classify(row["statement"]) == row["status"], row["statement"]
        assert len(table["statements"]) == table["n_distinct_statements_touching_a_class"]

    def test_borderline_first_degree_block_has_a_pr_above_200_ms(
        self, table: dict[str, Any]
    ) -> None:
        """The reason borderline counts as IAVB, held to the cart's own measurement."""
        entry = next(a for a in table["ambiguities"] if a["key"] == "iavb-borderline")
        measured = entry["measured_intervals_ms"]
        assert measured["interval"] == "pr"
        assert measured["median"] == 208.0
        assert measured["share_above_200"] >= 0.975

    def test_a_bundle_branch_conduction_defect_has_a_wide_qrs(self, table: dict[str, Any]) -> None:
        entry = next(a for a in table["ambiguities"] if a["key"] == "bbb-type-conduction-defect")
        measured = entry["measured_intervals_ms"]
        assert measured["interval"] == "qrs" and measured["median"] == 140.0
        assert measured["share_at_least_120"] >= 0.96


@pytest.mark.data
class TestTheTableRebuildsFromTheRelease:
    def test_the_counts_come_back_from_the_csv(self, table: dict[str, Any]) -> None:
        source = Path(MIMIC_ECG_DIR) / "machine_measurements.csv"
        if not source.exists():
            pytest.skip(f"{source} is not on disk")
        frame = pd.read_csv(source, low_memory=False)
        labels = record_labels(record_statuses(frame))
        assert {k: int(labels[k].sum()) for k in CLASSES} == PINNED_RECORDS
