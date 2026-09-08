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
    @pytest.fixture(scope="class")
    def table(self) -> dict[str, Any]:
        if not ROTATION.exists():
            pytest.skip(f"{ROTATION} is not built; run scripts/rotation_table.py")
        return json.loads(ROTATION.read_text())

    def test_every_source_and_every_class_it_can_carry_has_a_cell(
        self, table: dict[str, Any]
    ) -> None:
        present = {(cell["source"], cell["label"]) for cell in table["cells"]}
        wanted = {(source, label) for source in SOURCES for label in usable_classes(source)}
        assert present == wanted

    def test_every_ordered_pair_is_measured(self, table: dict[str, Any]) -> None:
        """Twenty ordered pairs, minus the ones a refused class removes."""
        pairs = set()
        for cell in table["cells"]:
            for corpus in cell["rows"][0]["by_corpus"]:
                if corpus.endswith("-calibration-holdout") or corpus == cell["source"]:
                    continue
                pairs.add((cell["source"], corpus, cell["label"]))
        for source, target in itertools.permutations(SOURCES, 2):
            shared = set(usable_classes(source)) & set(usable_classes(target))
            for label in shared:
                assert (source, target, label) in pairs, (source, target, label)

    def test_every_figure_is_a_mean_over_two_hundred_draws_with_its_spread(
        self, table: dict[str, Any]
    ) -> None:
        """C-27, and C-10 carried over to the rotation."""
        assert table["settings"]["n_draws"] >= 200
        for cell in table["cells"]:
            for row in cell["rows"]:
                for corpus, block in row["by_corpus"].items():
                    for klass, spread in block["coverage_by_class"].items():
                        assert spread["n_draws"] >= 200, (cell["source"], corpus, klass)
                        assert "sd" in spread

    def test_every_cell_reports_the_effective_size_of_what_calibrated_it(
        self, table: dict[str, Any]
    ) -> None:
        """C-9, carried over: a weighting that costs sample size says what it cost."""
        for cell in table["cells"]:
            for row in cell["rows"]:
                for corpus, block in row["by_corpus"].items():
                    ess = block["calibration"]["effective_sample_size"]
                    assert ess["n_draws"] >= 0
                    if row["correction"] != "weighted":
                        assert ess["mean"] > 0, (cell["source"], corpus)

    def test_the_three_corrections_are_all_reported(self, table: dict[str, Any]) -> None:
        for cell in table["cells"]:
            present = {row["correction"] for row in cell["rows"]}
            assert present == set(CORRECTIONS), cell["source"]

    def test_an_unweighted_threshold_does_not_depend_on_which_corpus_it_is_spent_on(
        self, table: dict[str, Any]
    ) -> None:
        for cell in table["cells"]:
            for row in cell["rows"]:
                if row["correction"] == "weighted":
                    continue
                seen = {
                    json.dumps(block["threshold_by_class"], sort_keys=True)
                    for block in row["by_corpus"].values()
                }
                assert len(seen) == 1, (cell["source"], cell["label"], row["correction"])

    def test_the_bias_carries_its_spread_across_sources(self, table: dict[str, Any]) -> None:
        """C-28. One pair is an anecdote; the spread across five sources is the estimate."""
        for label, corrections in table["bias"].items():
            assert set(corrections) == set(CORRECTIONS), label
            for correction, block in corrections.items():
                away = block["away_bias"]
                assert away["n_pairs"] > 0, (label, correction)
                assert away["mean"] is not None
                if away["n_sources"] > 1:
                    assert away["sd_across_sources"] is not None

    def test_the_classes_a_corpus_refuses_never_appear_in_its_rows(
        self, table: dict[str, Any]
    ) -> None:
        for cell in table["cells"]:
            for row in cell["rows"]:
                for corpus in row["by_corpus"]:
                    if corpus.endswith("-calibration-holdout"):
                        continue
                    assert cell["label"] in usable_classes(corpus), (corpus, cell["label"])

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
