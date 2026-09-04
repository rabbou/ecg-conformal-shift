"""What the outcome table promises, held to the file it committed.

The table's whole purpose is to be comparable across schemes that defer and
schemes that do not, and that comparability rests on one arithmetic property:
the three shares of a label sum to one, with deferral counted rather than
dropped from the denominator.  If that breaks, a miss rate under one scheme
stops meaning what it means under another and every reading built on the table
is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from outcomes import OUTCOMES, SCHEMES, _plain_sets, _split_outcomes

from ecs.config import RESULTS_DIR

TABLE = RESULTS_DIR / "outcomes.json"


@pytest.fixture(scope="module")
def table() -> dict[str, Any]:
    if not TABLE.exists():  # pragma: no cover - the file is committed
        pytest.skip(f"{TABLE} not built")
    return json.loads(TABLE.read_text())


class TestTheSplitItself:
    """The statistic, on cases whose answer can be worked out by hand."""

    def test_the_three_outcomes_partition_the_cases_of_a_label(self) -> None:
        # Four MI cases: one labelled MI, one labelled non-MI, one set holding
        # both labels, one holding neither.
        sets = np.array([[False, True], [True, False], [True, True], [False, False]], dtype=bool)
        labels = np.array([1, 1, 1, 1])
        correct, deferred, wrong = _split_outcomes(sets, labels, 1)
        assert (correct, deferred, wrong) == (0.25, 0.5, 0.25)

    def test_the_same_split_read_from_the_other_label(self) -> None:
        sets = np.array([[True, False], [False, True]], dtype=bool)
        labels = np.array([0, 0])
        correct, deferred, wrong = _split_outcomes(sets, labels, 0)
        assert (correct, deferred, wrong) == (0.5, 0.0, 0.5)

    def test_a_single_threshold_defers_nothing(self) -> None:
        probs = np.array([[0.9, 0.1], [0.2, 0.8]])
        sets = _plain_sets(probs, 0.5)
        labels = np.array([0, 1])
        for klass in (0, 1):
            assert _split_outcomes(sets, labels, klass)[1] == 0.0


class TestTheCommittedOutcomeTable:
    """The file the report and its figures are drawn from."""

    def test_every_label_and_scheme_partitions_into_the_three_outcomes(
        self, table: dict[str, Any]
    ) -> None:
        for corpus, block in table["by_corpus"].items():
            for scheme in SCHEMES:
                for klass in ("0", "1"):
                    cell = block["schemes"][scheme][klass]
                    total = sum(cell[name]["mean"] for name in OUTCOMES)
                    assert total == pytest.approx(1.0, abs=2e-3), (corpus, scheme, klass)

    def test_every_figure_is_a_mean_over_at_least_a_hundred_draws_with_its_spread(
        self, table: dict[str, Any]
    ) -> None:
        for block in table["by_corpus"].values():
            for scheme in SCHEMES:
                for klass in ("0", "1"):
                    for name in OUTCOMES:
                        cell = block["schemes"][scheme][klass][name]
                        assert cell["n_draws"] >= 100
                        assert "sd" in cell

    def test_the_single_threshold_scheme_never_defers(self, table: dict[str, Any]) -> None:
        for block in table["by_corpus"].values():
            for klass in ("0", "1"):
                assert block["schemes"]["plain"][klass]["deferred"]["mean"] == 0.0

    def test_pooled_calibration_misses_more_infarctions_than_a_single_threshold(
        self, table: dict[str, Any]
    ) -> None:
        """The reading the report is built on, pinned to its committed values.

        Pooled calibration honours a coverage stated over everyone by serving
        the majority, and on a corpus at 25% prevalence that costs the minority
        more than an ordinary threshold does.  The reference values are the ones
        in the report; a change beyond the tolerance is a changed finding, not a
        rounding drift.
        """
        source = table["by_corpus"]["ptbxl"]["schemes"]
        assert source["pooled"]["1"]["wrong"]["mean"] == pytest.approx(0.268, abs=0.01)
        assert source["plain"]["1"]["wrong"]["mean"] == pytest.approx(0.104, abs=0.01)
        assert source["pooled"]["1"]["wrong"]["mean"] > 2 * source["plain"]["1"]["wrong"]["mean"]

    def test_label_conditional_calibration_holds_the_miss_rate_and_halves_false_alarms(
        self, table: dict[str, Any]
    ) -> None:
        source = table["by_corpus"]["ptbxl"]["schemes"]
        plain, perlabel = source["plain"], source["perlabel"]
        assert perlabel["1"]["wrong"]["mean"] == pytest.approx(
            plain["1"]["wrong"]["mean"], abs=0.02
        )
        assert perlabel["0"]["wrong"]["mean"] < 0.6 * plain["0"]["wrong"]["mean"]

    def test_the_table_says_the_two_targets_are_different_quantities(
        self, table: dict[str, Any]
    ) -> None:
        """A sensitivity and a coverage both written 90% is the table's one trap."""
        assert table["alpha"] == pytest.approx(0.10)
        assert table["plain_sensitivity"] == pytest.approx(0.90)
        assert "sensitivity" in table["matched_operating_point"]
        assert "coverage" in table["matched_operating_point"]

    def test_the_denominator_is_named_in_the_file(self, table: dict[str, Any]) -> None:
        assert "sum to one" in table["outcome_definitions"]["denominator"]

    def test_no_external_label_reached_a_threshold(self, table: dict[str, Any]) -> None:
        assert "No external label reaches any threshold." in table["protocol"]


class TestTheFiguresDrawnFromIt:
    def test_the_two_figures_exist_and_the_table_precedes_them(self) -> None:
        figures = [
            RESULTS_DIR / "figures/fig1_thresholds.png",
            RESULTS_DIR / "figures/fig2_outcomes.png",
        ]
        for figure in figures:
            assert figure.exists(), figure
            assert figure.stat().st_size > 10_000
        assert TABLE.exists()
        assert isinstance(TABLE, Path)
