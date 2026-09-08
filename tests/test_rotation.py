"""The rotation: the splits it draws, the shape it reads, and the thresholds it spends.

Three claims, in order of what they cost to check.

*The split.*  Every corpus is cut by patient (C-4) and every part is capped at
the size the smallest corpus can reach, so "which source" is not read together
with "how much data the source had" (C-25).

*The ingestion.*  A Challenge-2021 partition comes out of the chain in the same
canonical form as PTB-XL and Shandong: float32, (n, 12, 5000), millivolts, leads
in the canonical order (C-13, C-25).

*The thresholds.*  No threshold is a function of a target label (C-26).  The
check permutes every label of the target corpus and re-runs the whole
frozen-calibration machinery on all twenty ordered pairs: if a single threshold
moves, a target label reached one.  The weighted correction is the interesting
case, because it does read the target -- its unlabelled predicted-label
marginal, which a permutation of the labels leaves alone.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ecs.config import RESULTS_DIR
from ecs.ingest import CANONICAL_LEADS
from ecs.report import CORRECTIONS, Source, Target, frozen_calibration_table
from ecs.rotation import (
    CAL_CAP,
    PARTS,
    SOURCES,
    TEST_CAP,
    TRAIN_CAP,
    VAL_CAP,
    class_keys,
    corpus_index,
    load_waveforms,
    usable_classes,
)

CAPS = {"train": TRAIN_CAP, "val": VAL_CAP, "cal": CAL_CAP, "test": TEST_CAP}
ROTATION = Path(RESULTS_DIR) / "rotation.json"
GRID = Path(RESULTS_DIR) / "rotation.csv"


@pytest.fixture(scope="module")
def indices() -> dict[str, Any]:
    return {corpus: corpus_index(corpus) for corpus in SOURCES}


@pytest.mark.data
class TestTheSplit:
    def test_no_patient_lands_in_two_parts(self, indices: dict[str, Any]) -> None:
        for corpus, index in indices.items():
            sides = index.frame.groupby("patient")["part"].nunique()
            offenders = sorted(sides.index[sides > 1])
            assert not offenders, f"{corpus}: patients on two sides {offenders[:5]}"

    def test_every_part_is_within_its_cap(self, indices: dict[str, Any]) -> None:
        for corpus, index in indices.items():
            for part, cap in CAPS.items():
                size = int((index.frame["part"] == part).sum())
                assert 0 < size <= cap, f"{corpus}/{part}: {size} against cap {cap}"

    def test_the_parts_are_the_four_the_module_declares(self, indices: dict[str, Any]) -> None:
        for corpus, index in indices.items():
            present = set(index.frame["part"].unique())
            assert present <= {*PARTS, "unused"}, corpus
            assert set(PARTS) <= present, corpus

    def test_the_split_is_the_same_on_a_second_reading(self) -> None:
        first = corpus_index("georgia").frame["part"]
        second = corpus_index("georgia").frame["part"]
        assert first.equals(second)

    def test_the_corpora_that_ship_short_records_say_how_many_they_dropped(
        self, indices: dict[str, Any]
    ) -> None:
        """Georgia and CPSC are the two; the count leaves with the deviations."""
        dropped = {
            corpus: [d for d in index.deviations if "shorter than ten seconds" in d]
            for corpus, index in indices.items()
        }
        assert dropped["georgia"] == [
            "52 records shorter than ten seconds, dropped before the split"
        ]
        assert dropped["cpsc"] == ["22 records shorter than ten seconds, dropped before the split"]
        assert dropped["ptbxl"] == [] and dropped["chapman_ningbo"] == []

    def test_a_refused_class_is_dropped_rather_than_reported_as_absent(self) -> None:
        assert "NSR" not in usable_classes("sph")
        assert "NSR" in usable_classes("ptbxl")
        for corpus in SOURCES:
            assert usable_classes(corpus), corpus


@pytest.mark.data
class TestTheIngestionContract:
    @pytest.mark.parametrize("corpus", SOURCES)
    def test_a_sample_comes_out_canonical(self, corpus: str) -> None:
        index = corpus_index(corpus)
        ids = index.ids("test")[:8]
        x, kept = load_waveforms(index, ids)
        assert x.dtype == np.float32
        assert x.shape == (len(kept), 12, 5000)
        assert np.isfinite(x).all()
        # Millivolts: a 12-lead ECG has peaks of a few millivolts, never a few
        # thousand, which is what an unconverted analogue-to-digital count looks like.
        assert 0.05 < float(np.percentile(np.abs(x), 99)) < 10.0

    def test_the_canonical_lead_order_is_the_one_the_chain_promises(self) -> None:
        assert CANONICAL_LEADS == (
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        )


def _fixture_pair(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A calibration or target sample: two-class probabilities, labels, patients."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.25).astype(int)
    signal = rng.normal(loc=labels * 1.2, scale=1.0)
    p = 1.0 / (1.0 + np.exp(-signal))
    return np.column_stack([1.0 - p, p]), labels, [f"P{i}" for i in range(n)]


def _thresholds(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], Any]:
    """Every threshold on a table, keyed by where it sits."""
    out = {}
    for row in rows:
        for corpus, block in row["by_corpus"].items():
            for klass, spread in block["threshold_by_class"].items():
                key = (row["alpha"], row["score"], row["correction"], corpus, klass)
                out[key] = (spread["mean"], spread["sd"], spread["n_infinite"])
    return out


class TestNoThresholdReadsATargetLabel:
    """C-26, on synthetic data first: cheap, and it covers every ordered pair."""

    def test_permuting_a_targets_labels_moves_no_threshold(self) -> None:
        probs, labels, patients = _fixture_pair(400, seed=0)
        source = Source(name="source", probs=probs, labels=labels, patients=patients)
        target_probs, target_labels, _ = _fixture_pair(300, seed=1)
        target = Target(probs=target_probs, labels=target_labels, n_patients=300)
        rng = np.random.default_rng(7)
        shuffled = Target(
            probs=target_probs,
            labels=target_labels[rng.permutation(len(target_labels))],
            n_patients=300,
        )
        before = frozen_calibration_table(
            source, {"t": target}, (0.1,), n_draws=5, corrections=CORRECTIONS
        )
        after = frozen_calibration_table(
            source, {"t": shuffled}, (0.1,), n_draws=5, corrections=CORRECTIONS
        )
        assert _thresholds(before) == _thresholds(after)


def _fake_cell(source: str, label: str, by_corpus: dict[str, float]) -> dict[str, Any]:
    """One (source, diagnosis) block with the coverage each corpus is to report."""
    return {
        "source": source,
        "label": label,
        "rows": [
            {
                "alpha": 0.10,
                "target_coverage": 0.90,
                "score": "lac",
                "correction": correction,
                "by_corpus": {
                    f"{source}-calibration-holdout": {
                        "coverage_by_class": {"1": {"mean": 0.9}},
                        "prevalence": 0.2,
                        "n_points": 1000,
                    },
                    **{
                        corpus: {
                            "coverage_by_class": {"1": {"mean": covered}},
                            "prevalence": 0.2,
                            "n_points": 1000,
                        }
                        for corpus, covered in by_corpus.items()
                    },
                },
            }
            for correction in CORRECTIONS
        ],
    }


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    """Two sources, one diagnosis, coverage chosen so the answer is arithmetic."""
    import rotation_table

    cells = [
        _fake_cell("a", "AF", {"a": 0.90, "b": 0.80, "c": 0.80}),
        _fake_cell("b", "AF", {"b": 0.90, "a": 0.70, "c": 0.70}),
    ]
    return rotation_table.bias_summary(cells)


class TestTheBiasSummary:
    """C-28: what the rotation exists to produce, on numbers whose answer is known.

    Two sources, one diagnosis. The first source lands 10 points low on both its
    targets, the second 20 points low on both, and each reads the level exactly at
    home. The away bias is then -0.15, the spread across the four pairs is the
    spread of two -0.10s and two -0.20s, and the spread across the two sources is
    the spread of -0.10 and -0.20 -- a wider number, which is the reason the
    rotation reports it rather than the pair spread alone.
    """

    def test_the_away_pairs_exclude_the_source_reading_at_home(
        self, summary: dict[str, Any]
    ) -> None:
        block = summary["AF"]["none"]
        assert block["away_bias"]["n_pairs"] == 4
        assert {(a["source"], a["target"]) for a in block["away"]} == {
            ("a", "b"),
            ("a", "c"),
            ("b", "a"),
            ("b", "c"),
        }
        assert [h["source"] for h in block["home"]] == ["a", "b"]

    def test_the_home_reading_is_the_source_on_its_own_test_part(
        self, summary: dict[str, Any]
    ) -> None:
        assert summary["AF"]["none"]["home_bias"]["mean"] == 0.0

    def test_the_mean_away_bias_is_the_mean_over_the_pairs(self, summary: dict[str, Any]) -> None:
        assert summary["AF"]["none"]["away_bias"]["mean"] == pytest.approx(-0.15)
        assert summary["AF"]["none"]["away_bias"]["worst"] == pytest.approx(-0.20)

    def test_the_spread_across_sources_is_wider_than_the_spread_across_pairs(
        self, summary: dict[str, Any]
    ) -> None:
        """Averaging inside a source first is what makes the second number the
        one to quote: it asks how much the break depends on where you started."""
        block = summary["AF"]["none"]["away_bias"]
        assert block["sd_across_pairs"] == pytest.approx(0.0577, abs=1e-3)
        assert block["sd_across_sources"] == pytest.approx(0.0707, abs=1e-3)
        assert block["sd_across_sources"] > block["sd_across_pairs"]
        assert block["n_sources"] == 2

    def test_the_thinnest_pair_behind_the_label_is_reported(self, summary: dict[str, Any]) -> None:
        """A class-conditional figure resting on a handful of positives is noise,
        and the count is what lets a reader see it."""
        assert summary["AF"]["none"]["fewest_positives_behind_a_pair"] == 200

    def test_every_correction_gets_its_own_summary(self, summary: dict[str, Any]) -> None:
        assert set(summary["AF"]) == set(CORRECTIONS)


@pytest.mark.data
class TestTheCommittedRotation:
    """The committed table: results/rotation.json for the answer, rotation.csv for
    the grid it was read off. The grid is a table of 2,520 coverage cells and is
    written as one; nesting it in JSON costs seven megabytes for the same numbers."""

    @staticmethod
    def _grid() -> list[dict[str, str]]:
        if not GRID.exists():
            pytest.skip(f"{GRID} is not built; run scripts/rotation_table.py")
        with GRID.open(newline="") as handle:
            return list(csv.DictReader(handle))

    @pytest.fixture(scope="class")
    def table(self) -> dict[str, Any]:
        if not ROTATION.exists():
            pytest.skip(f"{ROTATION} is not built; run scripts/rotation_table.py")
        return json.loads(ROTATION.read_text())

    @pytest.fixture(scope="class")
    def grid(self) -> list[dict[str, str]]:
        return self._grid()

    def test_the_summary_and_the_grid_describe_the_same_run(
        self, table: dict[str, Any], grid: list[dict[str, str]]
    ) -> None:
        assert table["grid"]["file"] == "results/rotation.csv"
        assert table["grid"]["n_rows"] == len(grid)
        assert list(grid[0]) == table["grid"]["columns"]

    def test_every_source_and_every_class_it_can_carry_is_on_the_grid(
        self, grid: list[dict[str, str]]
    ) -> None:
        present = {(row["source"], row["label"]) for row in grid}
        wanted = {(source, label) for source in SOURCES for label in usable_classes(source)}
        assert present == wanted

    def test_every_ordered_pair_is_measured(self, grid: list[dict[str, str]]) -> None:
        """Twenty ordered pairs per diagnosis, twelve where a refusal removes one."""
        pairs = {
            (row["source"], row["corpus"], row["label"]) for row in grid if row["role"] == "away"
        }
        for source, target in itertools.permutations(SOURCES, 2):
            for label in set(usable_classes(source)) & set(usable_classes(target)):
                assert (source, target, label) in pairs, (source, target, label)
        assert len(pairs) == sum(
            len(set(usable_classes(a)) & set(usable_classes(b)))
            for a, b in itertools.permutations(SOURCES, 2)
        )

    def test_every_corpus_reads_at_home_as_well_as_away(self, grid: list[dict[str, str]]) -> None:
        """A source with no home reading could not be compared against itself."""
        for source in SOURCES:
            home = {
                row["label"] for row in grid if row["source"] == source and row["role"] == "home"
            }
            assert home == set(usable_classes(source)), source

    def test_every_figure_is_a_mean_over_two_hundred_draws_with_its_spread(
        self, table: dict[str, Any], grid: list[dict[str, str]]
    ) -> None:
        """C-27, and C-10 carried over to the rotation."""
        assert table["settings"]["n_draws"] >= 200
        for row in grid:
            assert int(row["n_draws"]) >= 200
            assert row["coverage_sd"] != ""
            assert row["coverage_diagnosis_sd"] != ""

    def test_every_cell_reports_the_effective_size_of_what_calibrated_it(
        self, grid: list[dict[str, str]]
    ) -> None:
        """C-9, carried over: a weighting that costs sample size says what it cost."""
        for row in grid:
            effective = float(row["calibration_effective_size_mean"])
            drawn = float(row["calibration_n_mean"])
            assert drawn > 0
            if row["correction"] == "weighted":
                assert effective <= drawn + 1e-6, row
            else:
                assert effective == pytest.approx(drawn, abs=0.5), row

    def test_the_three_corrections_and_three_levels_are_all_reported(
        self, grid: list[dict[str, str]]
    ) -> None:
        assert {row["correction"] for row in grid} == set(CORRECTIONS)
        assert {row["alpha"] for row in grid} == {"0.2", "0.1", "0.05"}

    def test_an_unweighted_threshold_does_not_depend_on_which_corpus_it_is_spent_on(
        self, grid: list[dict[str, str]]
    ) -> None:
        """C-26 on the committed grid: one threshold, spent everywhere unchanged."""
        seen: dict[tuple[str, str, str, str, str], set[str]] = {}
        for row in grid:
            if row["correction"] == "weighted":
                continue
            key = (
                row["source"],
                row["label"],
                row["alpha"],
                row["score"],
                row["correction"],
            )
            thresholds = f"{row['threshold_no_diagnosis_mean']}/{row['threshold_diagnosis_mean']}"
            seen.setdefault(key, set()).add(thresholds)
        for key, values in seen.items():
            assert len(values) == 1, (key, values)

    def test_the_weighted_threshold_is_the_one_that_moves_by_corpus(
        self, grid: list[dict[str, str]]
    ) -> None:
        """It reads the target's unlabelled predicted-label marginal and nothing
        else of it, which is why it is the only family that varies by target."""
        moved = 0
        for source in SOURCES:
            for label in usable_classes(source):
                rows = [
                    r
                    for r in grid
                    if r["source"] == source
                    and r["label"] == label
                    and r["correction"] == "weighted"
                    and r["alpha"] == "0.1"
                    and r["score"] == "lac"
                ]
                if len({r["threshold_diagnosis_mean"] for r in rows}) > 1:
                    moved += 1
        assert moved > 0, "no weighted threshold differed by target; BBSE read nothing"

    def test_the_bias_carries_its_spread_across_sources(self, table: dict[str, Any]) -> None:
        """C-28. One pair is an anecdote; the spread across sources is the estimate."""
        for label, corrections in table["bias"].items():
            assert set(corrections) == set(CORRECTIONS), label
            for correction, block in corrections.items():
                away = block["away_bias"]
                assert away["n_pairs"] > 0, (label, correction)
                assert away["mean"] is not None
                assert away["sd_across_sources"] is not None
                assert block["fewest_positives_behind_a_pair"] > 0

    def test_the_bias_agrees_with_the_grid_it_was_read_off(
        self, table: dict[str, Any], grid: list[dict[str, str]]
    ) -> None:
        """The answer and the table cannot drift apart: the summary is recomputed
        here from the committed rows and has to land on the same number."""
        for label, corrections in table["bias"].items():
            for correction, block in corrections.items():
                away = [
                    float(r["coverage_diagnosis_mean"]) - 0.90
                    for r in grid
                    if r["label"] == label
                    and r["correction"] == correction
                    and r["role"] == "away"
                    and r["alpha"] == "0.1"
                    and r["score"] == "lac"
                ]
                assert len(away) == block["away_bias"]["n_pairs"], (label, correction)
                assert sum(away) / len(away) == pytest.approx(
                    block["away_bias"]["mean"], abs=5e-4
                ), (label, correction)

    def test_the_classes_a_corpus_refuses_never_appear_in_its_rows(
        self, grid: list[dict[str, str]]
    ) -> None:
        for row in grid:
            if row["role"] == "calibration holdout":
                continue
            assert row["label"] in usable_classes(row["corpus"]), (row["corpus"], row["label"])

    def test_every_source_names_what_it_trained_on(self, table: dict[str, Any]) -> None:
        for source in SOURCES:
            block = table["sources"][source]
            assert block["n_train"] <= TRAIN_CAP
            assert block["usable_classes"]
            assert block["deviations"]


@pytest.mark.data
class TestNoTargetLabelReachesAThresholdOnTheRealPairs:
    """C-26 again, on the scores the rotation actually spent, pair by pair."""

    @staticmethod
    def _scores(source: str, corpus: str, part: str) -> dict[str, Any]:
        path = Path(RESULTS_DIR) / "rotation" / source / "scores" / f"{corpus}_{part}.npz"
        if not path.exists():
            pytest.skip(f"{path} is not built; run scripts/score_rotation.py")
        with np.load(path, allow_pickle=False) as data:
            return {
                "ids": [str(i) for i in data["ids"]],
                "y": np.asarray(data["y"], dtype=int),
                "p": np.asarray(data["p"], dtype=float),
                "classes": [str(c) for c in data["classes"]],
            }

    @pytest.mark.parametrize("source,target", list(itertools.permutations(SOURCES, 2)))
    def test_permuting_the_targets_labels_moves_no_threshold(
        self, source: str, target: str
    ) -> None:
        shared = set(usable_classes(source)) & set(usable_classes(target))
        label = sorted(shared)[0]
        column = class_keys().index(label)
        calibration = self._scores(source, source, "cal")
        test = self._scores(source, target, "test")
        index = corpus_index(source)
        patients = [str(p) for p in index.frame.loc[calibration["ids"], "patient"]]
        p_cal = calibration["p"][:, column]
        home = Source(
            name=f"{source}-cal",
            probs=np.column_stack([1.0 - p_cal, p_cal]),
            labels=calibration["y"][:, column],
            patients=patients,
        )
        p_test = test["p"][:, column]
        probs = np.column_stack([1.0 - p_test, p_test])
        labels = test["y"][:, column]
        rng = np.random.default_rng(11)
        pairs = {
            "straight": Target(probs=probs, labels=labels, n_patients=len(labels)),
            "shuffled": Target(
                probs=probs,
                labels=labels[rng.permutation(len(labels))],
                n_patients=len(labels),
            ),
        }
        tables = {
            name: frozen_calibration_table(
                home, {target: t}, (0.1,), n_draws=3, corrections=CORRECTIONS
            )
            for name, t in pairs.items()
        }
        assert _thresholds(tables["straight"]) == _thresholds(tables["shuffled"])
