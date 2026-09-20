"""The committed label table, checked against the Challenge's own published counts.

``results/label_map.json`` is what the rotation stands on: it says how many
records of each of the five diagnoses each of the five corpora holds.  The
Challenge publishes the same counts for four of them, so the table has an
external reference and not only itself.  The one gap between the two is
arithmetic and named: the Challenge's number is a sum over its rows, so a record
carrying both halves of a fused pair lands in it twice.

These tests read the committed file.  Rebuild it with

    .venv/bin/python scripts/label_table.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ecs.config import RESULTS_DIR
from ecs.small_set import (
    AMBIGUITIES,
    CHALLENGE_PARTITION_COLUMNS,
    SMALL_SET,
    refusals,
    snomed_codes_for,
)

PATH = Path(RESULTS_DIR) / "label_map.json"
CHALLENGE_CORPORA = tuple(CHALLENGE_PARTITION_COLUMNS)
ALL_CORPORA = (*CHALLENGE_CORPORA, "sph")


@pytest.fixture(scope="module")
def table() -> dict[str, Any]:
    if not PATH.exists():
        pytest.skip(f"{PATH} is not built; run scripts/label_table.py")
    return json.loads(PATH.read_text())


class TestTheCountsCloseAgainstThePublishedTable:
    def test_every_challenge_corpus_reproduces_the_published_count(
        self, table: dict[str, Any]
    ) -> None:
        """C-23. Published minus measured minus double-counted is zero, everywhere.

        The subtraction is redone here from the three counts rather than read
        off ``unexplained_by_double_counting``. Asserting that field alone would
        pass on a writer that computed it wrongly, which is the one failure this
        check exists to catch.
        """
        for corpus in CHALLENGE_CORPORA:
            block = table["corpora"][corpus]
            doubles = block["records_carrying_both_codes_of_a_fused_class"]
            for klass in SMALL_SET:
                measured = block["counts"][klass.key]
                published = block["published_counts"][klass.key]
                double = doubles.get(klass.key, 0)
                recomputed = published - measured - double
                assert recomputed == 0, (
                    f"{corpus}/{klass.key}: measured {measured}, published {published}, "
                    f"double-counted {double}, unexplained {recomputed}"
                )
                assert block["unexplained_by_double_counting"][klass.key] == recomputed, (
                    f"{corpus}/{klass.key}: the file's own gap disagrees with the arithmetic"
                )
                assert block["published_minus_measured"][klass.key] == published - measured, (
                    f"{corpus}/{klass.key}: the file's own difference disagrees"
                )

    def test_the_only_difference_is_the_fused_classes(self, table: dict[str, Any]) -> None:
        """A class with one code has nothing to double-count, so it matches exactly."""
        for corpus in CHALLENGE_CORPORA:
            block = table["corpora"][corpus]
            for klass in SMALL_SET:
                if len(klass.abbreviations) > 1:
                    continue
                assert block["published_minus_measured"][klass.key] == 0, (corpus, klass.key)

    def test_this_copy_of_the_bundle_is_complete(self, table: dict[str, Any]) -> None:
        """The reference only means anything on a complete copy of the corpus."""
        for corpus in CHALLENGE_CORPORA:
            complete = table["corpora"][corpus]["completeness"]
            assert complete["missing_from_this_copy"] == [], corpus
            assert complete["not_in_the_manifest"] == [], corpus
            assert complete["n_on_disk"] == complete["n_distributed"], corpus


class TestEveryCorpusAndClassHasACell:
    def test_the_five_classes_are_the_ones_the_module_declares(self, table: dict[str, Any]) -> None:
        keys = [row["key"] for row in table["classes"]]
        assert keys == [klass.key for klass in SMALL_SET]

    def test_every_class_carries_the_snomed_codes_the_mapping_reads(
        self, table: dict[str, Any]
    ) -> None:
        for row in table["classes"]:
            assert set(row["snomed"]) == snomed_codes_for(row["key"]), row["key"]

    def test_every_corpus_counts_every_class(self, table: dict[str, Any]) -> None:
        for corpus in ALL_CORPORA:
            counts = table["corpora"][corpus]["counts"]
            assert set(counts) == {klass.key for klass in SMALL_SET}, corpus

    def test_a_refused_cell_is_empty_and_says_which_ambiguity_refused_it(
        self, table: dict[str, Any]
    ) -> None:
        """C-24. Shandong has no sinus-rhythm class, and the file says why."""
        refused = {
            (row["corpus"], row["class"]): row["ambiguity"] for row in table["refused_cells"]
        }
        assert refused, "no refusal is recorded, yet the mapping declares one"
        assert refused == {pair: a.key for pair, a in refusals().items()}
        for (corpus, key), ambiguity in refused.items():
            assert table["corpora"][corpus]["counts"][key] == 0, (corpus, key)
            assert ambiguity in {a["key"] for a in table["ambiguities"]}

    def test_no_other_cell_is_empty(self, table: dict[str, Any]) -> None:
        refused = {(row["corpus"], row["class"]) for row in table["refused_cells"]}
        for corpus in ALL_CORPORA:
            for klass in SMALL_SET:
                if (corpus, klass.key) in refused:
                    continue
                assert table["corpora"][corpus]["counts"][klass.key] > 0, (corpus, klass.key)


class TestTheAmbiguitiesAreOnTheFile:
    def test_every_declared_ambiguity_is_written_out(self, table: dict[str, Any]) -> None:
        assert {a["key"] for a in table["ambiguities"]} == {a.key for a in AMBIGUITIES}

    def test_each_one_carries_what_it_is_and_what_was_done(self, table: dict[str, Any]) -> None:
        for ambiguity in table["ambiguities"]:
            for field in ("what", "decision", "why", "evidence"):
                assert len(str(ambiguity[field]).split()) >= 10, ambiguity["key"]


class TestPtbxlIsReadThroughItsOwnDistribution:
    def test_the_join_keeps_the_records_the_object_already_counts(
        self, table: dict[str, Any]
    ) -> None:
        """C-1's reference values: 21,799 records from 18,869 patients."""
        block = table["corpora"]["ptbxl"]["as_read_by_this_study"]
        assert block["n_records"] == 21799
        assert block["n_patients"] == 18869
        assert block["join"]["in_the_distribution_not_in_the_bundle"] == []

    def test_the_two_routes_into_ptbxl_are_compared_class_by_class(
        self, table: dict[str, Any]
    ) -> None:
        agreement = table["corpora"]["ptbxl"]["as_read_by_this_study"]["snomed_against_scp"]
        assert set(agreement) == {klass.key for klass in SMALL_SET}
        for key, cell in agreement.items():
            total = cell["both"] + cell["snomed_only"] + cell["scp_only"] + cell["neither"]
            assert total == 21799, key

    def test_the_class_where_the_two_routes_part_is_the_one_the_ambiguity_names(
        self, table: dict[str, Any]
    ) -> None:
        """Sinus rhythm is the disagreement; the four others agree to a handful."""
        agreement = table["corpora"]["ptbxl"]["as_read_by_this_study"]["snomed_against_scp"]
        for key, cell in agreement.items():
            disagreement = cell["snomed_only"] + cell["scp_only"]
            if key == "NSR":
                assert disagreement > 100
            else:
                assert disagreement <= 5, (key, cell)
        assert any(a["key"] == "ptbxl-snomed-versus-scp" for a in table["ambiguities"])


class TestShandongGoesThroughLeinonensTable:
    def test_the_modifier_choice_is_measured_rather_than_assumed(
        self, table: dict[str, Any]
    ) -> None:
        """Matching on the base code can only add records, never remove them."""
        block = table["corpora"]["sph"]
        for klass in SMALL_SET:
            base = block["counts"][klass.key]
            exact = block["counts_matching_the_full_modifier_token"][klass.key]
            assert base >= exact, klass.key
