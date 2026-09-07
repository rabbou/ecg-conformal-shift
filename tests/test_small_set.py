"""The five-class label vocabulary, held to the published tables it is read from.

These tests never look at a corpus.  They check that the mapping in
``src/ecs/small_set.py`` says what the three files in ``mappings/`` say, and
that every join the mapping could not make cleanly is declared rather than
absorbed.  A class quietly redefined here fails on the published table, not on
a count that would have moved with it.
"""

from __future__ import annotations

import pandas as pd

from ecs.small_set import (
    AMBIGUITIES,
    CHALLENGE_PARTITION_COLUMNS,
    SMALL_SET,
    aha_codes_for,
    aha_to_snomed,
    challenge_published_counts,
    class_of_key,
    ptbxl_plus_statements,
    refusals,
    scored_diagnoses,
    snomed_codes_for,
)


class TestTheClassesComeOutOfThePublishedTable:
    def test_every_class_names_rows_the_challenge_table_holds(self) -> None:
        table = scored_diagnoses()
        for klass in SMALL_SET:
            for abbreviation in klass.abbreviations:
                assert abbreviation in table.index, f"{klass.key}: {abbreviation}"

    def test_every_class_carries_at_least_one_snomed_code(self) -> None:
        for klass in SMALL_SET:
            assert snomed_codes_for(klass.key), klass.key

    def test_no_two_classes_share_a_snomed_code(self) -> None:
        seen: dict[str, str] = {}
        for klass in SMALL_SET:
            for code in snomed_codes_for(klass.key):
                assert code not in seen, f"{code} in both {seen.get(code)} and {klass.key}"
                seen[code] = klass.key

    def test_the_bundle_branch_fusions_are_the_ones_the_challenge_declares(self) -> None:
        """The two fusions are the file's own, read out of its Notes column."""
        table = scored_diagnoses()
        for key in ("LBBB", "RBBB"):
            codes = snomed_codes_for(key)
            assert len(codes) == 2, key
            for abbreviation in class_of_key(key).abbreviations:
                note = str(table.loc[abbreviation, "Notes"])
                assert "same diagnosis" in note, f"{abbreviation}: {note!r}"
                for code in codes:
                    assert code in note, f"{abbreviation} note does not name {code}: {note!r}"

    def test_the_first_degree_block_union_is_ours_and_is_declared_as_such(self) -> None:
        """IAVB unions two classes the Challenge keeps apart, so it needs an ambiguity."""
        table = scored_diagnoses()
        for abbreviation in ("IAVB", "LPR"):
            note = table.loc[abbreviation, "Notes"]
            assert pd.isna(note), f"{abbreviation} now carries an equivalence note: {note!r}"
        assert any(a.key == "iavb-lpr" for a in AMBIGUITIES)


class TestTheShandongBridge:
    def test_every_class_the_mapping_fills_has_an_aha_code(self) -> None:
        refused = {key for corpus, key in refusals() if corpus == "sph"}
        for klass in SMALL_SET:
            codes = aha_codes_for(klass.key)
            if klass.key in refused:
                assert not codes, f"{klass.key} is refused on Shandong yet has a bridge"
            else:
                assert codes, f"{klass.key} has no AHA bridge and is not refused"

    def test_sinus_rhythm_is_refused_on_shandong_and_says_why(self) -> None:
        refused = refusals()
        assert ("sph", "NSR") in refused
        ambiguity = refused[("sph", "NSR")]
        assert "Normal ECG" in ambiguity.what
        assert ambiguity.decision.strip()
        assert ambiguity.evidence.strip()

    def test_the_bridge_is_leinonens_table_and_not_a_table_of_ours(self) -> None:
        published = aha_to_snomed()
        base = {str(code).split("+", 1)[0] for code in published["AHA_Code"]}
        for klass in SMALL_SET:
            for code in aha_codes_for(klass.key):
                assert code in base, f"{klass.key}: AHA {code} is not in the published table"


class TestTheScpCrossCheck:
    """The SCP acronyms the PTB-XL cross-check uses are the ones PTB-XL+ names."""

    STATEMENTS = {
        "SR": "sinus rhythm",
        "AFIB": "atrial fibrillation",
        "CLBBB": "complete left bundle branch block",
        "CRBBB": "complete right bundle branch block",
        "1AVB": "first degree AV block",
        "LPR": "prolonged PR interval",
    }

    def test_every_acronym_is_a_row_of_the_ptbxl_plus_mapping(self) -> None:
        published = ptbxl_plus_statements()
        for klass in SMALL_SET:
            for acronym in klass.scp_acronyms:
                assert acronym in published, f"{klass.key}: {acronym}"

    def test_each_acronym_carries_the_statement_it_was_chosen_for(self) -> None:
        published = ptbxl_plus_statements()
        for klass in SMALL_SET:
            for acronym in klass.scp_acronyms:
                assert published[acronym] == self.STATEMENTS[acronym], (
                    f"{acronym}: {published[acronym]!r}"
                )


class TestEveryAmbiguityIsWritten:
    def test_none_is_a_stub(self) -> None:
        for ambiguity in AMBIGUITIES:
            for field in (ambiguity.what, ambiguity.decision, ambiguity.why, ambiguity.evidence):
                assert len(field.split()) >= 10, ambiguity.key

    def test_every_refusal_belongs_to_an_ambiguity_that_names_the_corpus(self) -> None:
        for (corpus, _key), ambiguity in refusals().items():
            assert corpus in ambiguity.corpora, ambiguity.key

    def test_the_keys_are_distinct(self) -> None:
        keys = [a.key for a in AMBIGUITIES]
        assert len(keys) == len(set(keys))


class TestThePublishedCountsAreReadable:
    def test_every_corpus_and_class_has_a_reference_count(self) -> None:
        for corpus in CHALLENGE_PARTITION_COLUMNS:
            for klass in SMALL_SET:
                assert challenge_published_counts(corpus, klass.key) > 0, (corpus, klass.key)
