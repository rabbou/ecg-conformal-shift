"""The seen-target count, held to the Challenge's own published tally."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecs.config import CHALLENGE2021_DIR, RESULTS_DIR
from ecs.seen_target import (
    ALLOWED_PARTITIONS,
    ENOUGH_INFARCTION,
    INFARCTION_CODES,
    ISCHAEMIA_CODES,
    MAX_SHARE_FROM_ONE_PARTITION,
    PARTITIONS_HOLDING_CALIBRATION,
    SNOMED_NAMES,
    PartitionCount,
    assess,
    count_partition,
    dx_codes,
    parse_dx_line,
)

# The organisers' own per-partition counts for every infarction and ischaemia
# code, read from dx_mapping_scored.csv and dx_mapping_unscored.csv at
# github.com/physionetchallenges/evaluation-2021 on 2026-08-26.  A count of the
# headers that disagrees with this table is a broken parse, not a new finding.
PUBLISHED = {
    # code:            (chapman_shaoxing, cpsc_2018, cpsc_2018_extra, georgia, ningbo)
    "164865005": (40, 0, 376, 7, 83),
    "164867002": (0, 0, 1168, 0, 0),
    "57054005": (0, 0, 0, 0, 49),
    "54329005": (0, 0, 62, 0, 57),
    "164861001": (0, 0, 384, 0, 0),
    "413444003": (0, 0, 1, 1, 0),
    "413844008": (0, 0, 161, 0, 0),
    "425419005": (0, 0, 0, 451, 0),
    "425623009": (0, 0, 0, 903, 0),
    "426434006": (0, 0, 0, 281, 0),
}
PUBLISHED_ORDER = ("chapman_shaoxing", "cpsc_2018", "cpsc_2018_extra", "georgia", "ningbo")


class TestTheAllowList:
    """No partition holding the calibration corpus can reach a target."""

    def test_the_calibration_corpus_is_not_among_the_partitions_a_target_may_use(self) -> None:
        assert set(ALLOWED_PARTITIONS).isdisjoint(PARTITIONS_HOLDING_CALIBRATION)
        assert "ptb-xl" in PARTITIONS_HOLDING_CALIBRATION

    def test_infarction_and_ischaemia_are_two_classes_and_not_one(self) -> None:
        assert INFARCTION_CODES.isdisjoint(ISCHAEMIA_CODES)
        assert set(SNOMED_NAMES) == INFARCTION_CODES | ISCHAEMIA_CODES


class TestTheHeaderParse:
    """What a ``# Dx:`` line means, on headers whose answer is known."""

    def test_the_codes_come_off_the_dx_line_in_order(self) -> None:
        header = (
            "JS03355 12 500 5000\n# Age: 62\n# Sex: Male\n"
            "# Dx: 426177001,164934002\n# Rx: Unknown\n"
        )
        assert parse_dx_line(header) == ["426177001", "164934002"]

    def test_a_header_without_a_dx_line_carries_no_diagnosis(self) -> None:
        assert parse_dx_line("A1 12 500 5000\n# Age: 40\n") == []

    def test_spacing_around_the_key_and_the_codes_does_not_change_the_answer(self) -> None:
        assert parse_dx_line("#Dx:164865005, 57054005 \n") == ["164865005", "57054005"]

    def test_a_record_is_counted_once_however_many_codes_of_a_family_it_holds(self) -> None:
        both = ["164865005", "164867002", "426177001"]
        count = count_partition([both, ["426177001"], ["164861001"]])
        assert count.n_records == 3
        assert count.n_infarction == 1
        assert count.n_ischaemia == 1
        assert count.n_infarction_or_ischaemia == 2


@pytest.mark.data
class TestThePublishedCounts:
    """The parse against the organisers' table, on the bundle itself."""

    def test_every_code_matches_the_count_the_challenge_publishes(self) -> None:
        training = CHALLENGE2021_DIR / "training"
        if not training.is_dir():
            pytest.skip(f"the Challenge-2021 bundle is not at {CHALLENGE2021_DIR}")
        for column, partition in enumerate(PUBLISHED_ORDER):
            tally: dict[str, int] = {}
            for header in sorted((training / partition).rglob("*.hea")):
                for code in dx_codes(header):
                    tally[code] = tally.get(code, 0) + 1
            for code, expected in PUBLISHED.items():
                assert tally.get(code, 0) == expected[column], (
                    f"{partition} {code} ({SNOMED_NAMES[code]}): "
                    f"counted {tally.get(code, 0)}, the Challenge publishes {expected[column]}"
                )


class TestTheDecisionRule:
    """Both conditions, on assemblies whose answer is known by construction."""

    @staticmethod
    def counts(**kwargs: tuple[int, int]) -> dict[str, PartitionCount]:
        return {
            name: PartitionCount(records, infarction, 0, infarction)
            for name, (records, infarction) in kwargs.items()
        }

    def test_a_target_below_the_infarction_floor_is_refused(self) -> None:
        counts = self.counts(a=(10_000, 100), b=(10_000, 100))
        verdict = assess(counts, ["a", "b"])
        assert verdict["n_infarction"] == 200 < ENOUGH_INFARCTION
        assert verdict["enough_infarction"]["holds"] is False
        assert verdict["holds"] is False

    def test_a_target_one_partition_supplies_the_infarction_for_is_refused(self) -> None:
        counts = self.counts(thin=(60_000, 100), enriched=(3_000, 1_400))
        verdict = assess(counts, ["thin", "enriched"])
        assert verdict["n_infarction"] >= ENOUGH_INFARCTION
        assert verdict["prevalence_is_a_population_not_an_assembly"]["holds"] is False
        assert verdict["prevalence_is_a_population_not_an_assembly"]["largest_contributor"] == (
            "enriched"
        )
        assert verdict["holds"] is False

    def test_a_target_meeting_both_conditions_is_kept(self) -> None:
        counts = self.counts(a=(20_000, 300), b=(20_000, 300))
        verdict = assess(counts, ["a", "b"])
        assert verdict["enough_infarction"]["holds"] is True
        assert verdict["prevalence_is_a_population_not_an_assembly"]["holds"] is True
        assert verdict["prevalence_is_a_population_not_an_assembly"]["share_of_infarction"] == 0.5
        assert verdict["holds"] is True

    def test_the_ceiling_is_a_share_not_a_count(self) -> None:
        assert 0.0 < MAX_SHARE_FROM_ONE_PARTITION <= 1.0


class TestTheCommittedDecision:
    """What ``results/seen_target.json`` has to carry to settle the option."""

    @staticmethod
    def written() -> dict:
        path = RESULTS_DIR / "seen_target.json"
        if not path.exists():
            pytest.skip(f"{path} has not been written yet")
        return json.loads(Path(path).read_text())

    def test_it_names_every_partition_it_read_and_every_one_it_refused(self) -> None:
        result = self.written()
        assert result["partitions_read"] == list(ALLOWED_PARTITIONS)
        assert set(result["partitions_refused"]) == set(PARTITIONS_HOLDING_CALIBRATION)

    def test_the_calibration_corpus_is_in_no_partition_it_read(self) -> None:
        result = self.written()
        read = set(result["partitions_read"])
        assert read.isdisjoint(PARTITIONS_HOLDING_CALIBRATION)

    def test_it_counts_infarction_separately_from_ischaemia(self) -> None:
        result = self.written()
        for partition, count in result["by_partition"].items():
            assert {"n_records", "n_infarction", "n_ischaemia"} <= set(count), partition
            assert count["n_infarction"] <= count["n_infarction_or_ischaemia"]

    def test_both_readings_of_the_cpsc_partitions_are_assessed(self) -> None:
        result = self.written()
        readings = result["readings"]
        assert "cpsc_2018_extra" not in readings["four_partitions_named"]["partitions"]
        assert "cpsc_2018_extra" in readings["with_cpsc_extra"]["partitions"]

    def test_the_verdict_follows_from_the_two_conditions_and_is_not_typed_over_them(self) -> None:
        result = self.written()
        readings = result["readings"]
        for name, reading in readings.items():
            expected = (
                reading["enough_infarction"]["holds"]
                and reading["prevalence_is_a_population_not_an_assembly"]["holds"]
            )
            assert reading["holds"] is expected, name
        kept = any(reading["holds"] for reading in readings.values())
        assert result["decision"]["seen_target_cell"] == ("kept" if kept else "dropped")

    def test_the_per_code_counts_match_what_the_challenge_publishes(self) -> None:
        result = self.written()
        by_code = result["by_partition_by_code"]
        for column, partition in enumerate(PUBLISHED_ORDER):
            for code, expected in PUBLISHED.items():
                assert by_code[partition].get(code, 0) == expected[column], (
                    f"{partition} {code}: the committed count disagrees with the Challenge's"
                )
