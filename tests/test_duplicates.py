"""The same tracing twice, and the split that has to survive it.

A patient split keeps one patient's tracings together. It cannot keep one
*recording* together when the corpus files it twice under different record
identifiers and names no patient, which is what three of the Challenge-2021
partitions do. These tests hold three things:

the digest is the one ``ecg-data-chain`` published, so the copy here and the
screen there cannot drift apart; the widened key puts every repeat on one side
of every boundary; and the committed scan says so, with the count.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ecs.config import RESULTS_DIR
from ecs.duplicates import (
    MICROVOLTS_PER_STEP,
    duplicate_groups,
    grouping_key,
    signal_digest,
    straddling_groups,
)
from ecs.rotation import SOURCES

GROUPS = Path(RESULTS_DIR) / "duplicate_groups.json"
LEAK = Path(RESULTS_DIR) / "split_leak.json"


class TestTheDigest:
    def test_a_difference_below_the_step_gives_one_digest(self) -> None:
        """The quantum is what lets a repackaging of one recording land on one
        digest. It absorbs a difference smaller than the step, not one that
        crosses a step boundary: a sample sitting halfway between two steps
        rounds either way, and the screen does not pretend otherwise."""
        rng = np.random.default_rng(0)
        steps = rng.integers(-200, 200, size=(12, 5000))
        window = (steps * (MICROVOLTS_PER_STEP / 1000.0)).astype(np.float32)
        nudged = (window + 0.000_002).astype(np.float32)  # two microvolts, a fifth of a step
        assert signal_digest(window) == signal_digest(nudged)

    def test_a_real_difference_gives_a_different_digest(self) -> None:
        rng = np.random.default_rng(0)
        window = rng.normal(scale=0.5, size=(12, 5000)).astype(np.float32)
        moved = window.copy()
        moved[3, 17] += 0.05  # fifty microvolts, five steps
        assert signal_digest(window) != signal_digest(moved)

    def test_a_gap_is_not_a_flat_line_at_zero(self) -> None:
        """A missing sample and a sample of zero millivolts are different facts."""
        window = np.zeros((12, 5000), dtype=np.float32)
        gapped = window.copy()
        gapped[0, 0] = np.nan
        assert signal_digest(window) != signal_digest(gapped)

    def test_the_quantum_is_the_one_the_screen_was_published_with(self) -> None:
        assert MICROVOLTS_PER_STEP == 10.0


class TestTheWidenedKey:
    def test_a_repeated_tracing_joins_one_unit(self) -> None:
        patients = {"a": "1", "b": "2", "c": "3"}
        widened = grouping_key(patients, [["a", "c"]])
        assert widened["a"] == widened["c"]
        assert widened["b"] != widened["a"]

    def test_joining_is_transitive(self) -> None:
        """Two groups sharing a record make one unit, not two."""
        patients = {"a": "1", "b": "2", "c": "3", "d": "4"}
        widened = grouping_key(patients, [["a", "b"], ["b", "c"]])
        assert widened["a"] == widened["b"] == widened["c"]
        assert widened["d"] != widened["a"]

    def test_a_group_naming_a_record_the_corpus_dropped_still_joins_the_rest(self) -> None:
        patients = {"a": "1", "c": "3"}
        widened = grouping_key(patients, [["a", "absent", "c"]])
        assert widened["a"] == widened["c"]

    def test_a_corpus_with_no_repeats_keeps_its_patients_apart(self) -> None:
        patients = {"a": "1", "b": "2"}
        widened = grouping_key(patients, [])
        assert widened["a"] != widened["b"]

    def test_two_records_of_one_patient_were_already_together(self) -> None:
        patients = {"a": "1", "b": "1", "c": "2"}
        widened = grouping_key(patients, [])
        assert widened["a"] == widened["b"] != widened["c"]

    def test_duplicate_groups_are_the_digests_carried_more_than_once(self) -> None:
        groups = duplicate_groups({"a": "x", "b": "x", "c": "y"})
        assert groups == {"x": ["a", "b"]}

    def test_straddling_ignores_the_parts_no_figure_reads(self) -> None:
        parts = {"a": "train", "b": "unused"}
        assert straddling_groups(parts, [["a", "b"]]) == [["a", "b"]]
        assert straddling_groups(parts, [["a", "b"]], ignore=["unused"]) == []


@pytest.mark.data
class TestTheCommittedScan:
    @pytest.fixture(scope="class")
    def scan(self) -> dict[str, Any]:
        if not GROUPS.exists():
            pytest.skip(f"{GROUPS} is not built; run scripts/duplicate_scan.py")
        return json.loads(GROUPS.read_text())

    def test_every_corpus_was_scanned(self, scan: dict[str, Any]) -> None:
        assert set(scan["corpora"]) == set(SOURCES)

    def test_the_digests_agree_with_the_ones_the_delivery_corpus_published(
        self, scan: dict[str, Any]
    ) -> None:
        """Two ingestion chains, one digest per record. A disagreement would mean
        one of them is not producing the window it says it is."""
        compared = 0
        for corpus, block in scan["corpora"].items():
            against = block["against_the_delivery_corpus"]
            if not against.get("compared"):
                continue
            assert against["agree"] == against["compared"], corpus
            compared += against["compared"]
        assert compared > 40_000, f"only {compared} digests were cross-checked"

    def test_ptbxl_holds_no_repeated_tracing(self, scan: dict[str, Any]) -> None:
        """The PhysioNet distribution is clean; the repeats are a packaging fault
        of the Challenge bundle, and this says which side of that line PTB-XL is."""
        assert scan["corpora"]["ptbxl"]["n_groups"] == 0

    def test_the_corpora_that_repeat_a_tracing_are_named_with_their_counts(
        self, scan: dict[str, Any]
    ) -> None:
        repeats = {
            corpus: block["n_groups"]
            for corpus, block in scan["corpora"].items()
            if block["n_groups"]
        }
        assert set(repeats) == {"sph", "chapman_ningbo", "georgia", "cpsc"}
        assert repeats["cpsc"] > repeats["georgia"]
        for corpus, block in scan["corpora"].items():
            distinct = block["n_distinct_tracings"]
            assert distinct <= block["n_records"], corpus
            assert block["n_records"] - distinct == sum(len(g) - 1 for g in block["groups"]), corpus


@pytest.mark.data
class TestTheSplitNoLongerLeaks:
    @pytest.fixture(scope="class")
    def leak(self) -> dict[str, Any]:
        if not LEAK.exists():
            pytest.skip(f"{LEAK} is not built; run scripts/split_leak.py")
        return json.loads(LEAK.read_text())

    def test_no_group_of_identical_tracings_straddles_a_boundary(
        self, leak: dict[str, Any]
    ) -> None:
        """The claim the rotation rests on: a model is never scored on a
        recording it was fitted on."""
        assert leak["n_groups_still_across_two_used_parts"] == 0
        for corpus, block in leak["corpora"].items():
            assert block["after"]["n_groups_across_two_used_parts"] == 0, corpus

    def test_the_leak_it_closed_is_on_the_record(self, leak: dict[str, Any]) -> None:
        """Reporting only the fixed state would leave no way to see what it cost."""
        before = {
            corpus: block["before"]["n_groups_across_two_used_parts"]
            for corpus, block in leak["corpora"].items()
        }
        assert before["cpsc"] > 0 and before["georgia"] > 0 and before["chapman_ningbo"] > 0
        assert sum(before.values()) > 400

    def test_the_corpora_with_a_patient_key_were_never_leaking(self, leak: dict[str, Any]) -> None:
        """Shandong repeats tracings too, but files them under one patient, so
        the patient split already held them together."""
        assert leak["corpora"]["sph"]["before"]["n_groups"] > 0
        assert leak["corpora"]["sph"]["before"]["n_groups_across_two_used_parts"] == 0
        assert leak["corpora"]["ptbxl"]["before"]["n_groups"] == 0
