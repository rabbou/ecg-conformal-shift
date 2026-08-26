"""What a coverage number has to say before it can be believed (C-5, C-10, C-11,
C-12, C-17, C-18).

Two kinds of check live here.  The synthetic ones put a known, exchangeable
problem through the re-draw harness and demand the guarantee come out where the
theory says it will -- if split conformal is wired up wrongly, coverage misses
the target and it fails here on data whose answer is known.  The rest read the
committed result files and hold them to what the plan says a reported number
must carry: a spread over enough draws, a figure per class, and a named source
for every encoder arm.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from ecs.config import RESULTS_DIR
from ecs.report import Source, Target, frozen_calibration_table, repeated_split_report, spread

# C-5's band around the target, and C-10's floor on the number of draws.
COVERAGE_BAND = (-0.012, 0.025)
MIN_DRAWS = 100


def _calibrated_problem(n: int = 1200, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A two-class problem whose probabilities are honest: a record whose stated
    probability of class 1 is p really is class 1 with probability p.  One record
    per patient, so the draw is exchangeable by construction."""
    rng = np.random.default_rng(seed)
    p1 = rng.uniform(0.05, 0.95, n)
    labels = (rng.uniform(size=n) < p1).astype(int)
    probs = np.column_stack([1.0 - p1, p1])
    return probs, labels, [f"patient{i}" for i in range(n)]


class TestTheGuaranteeOnDataWhoseAnswerIsKnown:
    """C-5: on an exchangeable sample, coverage lands on the level asked for."""

    @pytest.mark.parametrize("alpha", [0.20, 0.10, 0.05])
    @pytest.mark.parametrize("score", ["lac", "aps"])
    def test_coverage_lands_inside_the_band_around_one_minus_alpha(
        self, alpha: float, score: str
    ) -> None:
        probs, labels, patients = _calibrated_problem()
        report = repeated_split_report(
            probs, labels, patients, alpha, score=score, n_draws=120, seed=3
        )
        assert isinstance(report["coverage"], dict)
        got = report["coverage"]["mean"]
        low, high = 1 - alpha + COVERAGE_BAND[0], 1 - alpha + COVERAGE_BAND[1]
        assert low <= got <= high, f"{score} at alpha={alpha}: coverage {got:.4f}"

    def test_a_tighter_level_never_returns_fewer_labels(self) -> None:
        probs, labels, patients = _calibrated_problem()
        sizes = [
            repeated_split_report(probs, labels, patients, alpha, n_draws=60, seed=3)[
                "mean_set_size"
            ]
            for alpha in (0.20, 0.10, 0.05)
        ]
        means = [s["mean"] for s in sizes if isinstance(s, dict)]
        assert means == sorted(means), means

    def test_mondrian_holds_the_level_inside_each_class(self) -> None:
        probs, labels, patients = _calibrated_problem()
        report = repeated_split_report(
            probs, labels, patients, 0.10, correction="mondrian", n_draws=120, seed=3
        )
        by_class = report["coverage_by_class"]
        assert isinstance(by_class, dict)
        for klass, figures in by_class.items():
            got = figures["mean"]
            assert 0.9 + COVERAGE_BAND[0] <= got <= 0.9 + COVERAGE_BAND[1], (klass, got)


class TestTheHarnessItself:
    def test_every_figure_comes_with_the_number_of_draws_behind_it(self) -> None:
        probs, labels, patients = _calibrated_problem(n=400)
        report = repeated_split_report(probs, labels, patients, 0.10, n_draws=101, seed=1)
        for key in ("coverage", "empty_rate", "one_label_rate", "two_label_rate"):
            figures = report[key]
            assert isinstance(figures, dict)
            assert figures["n_draws"] == 101
            assert figures["sd"] >= 0.0
        # Coverage genuinely moves from draw to draw; a zero there would mean the
        # draws are not being redrawn at all.
        coverage = report["coverage"]
        assert isinstance(coverage, dict)
        assert coverage["sd"] > 0.0

    def test_the_three_set_shapes_account_for_every_test_point(self) -> None:
        probs, labels, patients = _calibrated_problem(n=400)
        report = repeated_split_report(probs, labels, patients, 0.10, n_draws=60, seed=1)
        shares = [report[k]["mean"] for k in ("empty_rate", "one_label_rate", "two_label_rate")]  # type: ignore[index]
        assert sum(shares) == pytest.approx(1.0)
        assert report["abstention_rate"]["mean"] == pytest.approx(shares[0] + shares[2])  # type: ignore[index]

    def test_the_patient_count_is_the_distinct_patients_not_the_records(self) -> None:
        probs, labels, _ = _calibrated_problem(n=400)
        paired = [f"patient{i // 2}" for i in range(400)]  # two records each
        report = repeated_split_report(probs, labels, paired, 0.10, n_draws=20, seed=1)
        assert report["n_points"] == 400
        assert report["n_patients"] == 200

    def test_an_unknown_score_or_correction_is_refused(self) -> None:
        probs, labels, patients = _calibrated_problem(n=100)
        with pytest.raises(ValueError, match="score must be one of"):
            repeated_split_report(probs, labels, patients, 0.1, score="nearest")
        with pytest.raises(ValueError, match="correction must be one of"):
            repeated_split_report(probs, labels, patients, 0.1, correction="bayes")

    def test_a_single_draw_has_no_spread_and_no_draws_is_an_error(self) -> None:
        assert spread([0.9]).sd == 0.0
        with pytest.raises(ValueError, match="no draws"):
            spread([])


class TestTheCommittedAbstentionTable:
    """The table itself, held to C-5, C-10 and C-11."""

    @pytest.fixture(scope="module")
    def table(self) -> dict:
        return json.loads((RESULTS_DIR / "abstention.json").read_text())

    def test_it_covers_the_three_levels_the_plan_names(self, table: dict) -> None:
        assert sorted({row["alpha"] for row in table["rows"]}) == [0.05, 0.10, 0.20]

    def test_every_coverage_lands_inside_the_band(self, table: dict) -> None:
        for row in table["rows"]:
            target = row["target_coverage"]
            got = row["coverage"]["mean"]
            low, high = target + COVERAGE_BAND[0], target + COVERAGE_BAND[1]
            assert low <= got <= high, (row["score"], row["correction"], target, got)

    def test_every_figure_is_a_mean_over_at_least_a_hundred_draws_with_its_spread(
        self, table: dict
    ) -> None:
        assert table["n_draws"] >= MIN_DRAWS
        for row in table["rows"]:
            for key in ("coverage", "empty_rate", "one_label_rate", "two_label_rate"):
                assert row[key]["n_draws"] >= MIN_DRAWS
                assert row[key]["sd"] >= 0.0

    def test_coverage_is_reported_for_infarction_and_for_not(self, table: dict) -> None:
        assert table["classes"] == {"0": "no infarction", "1": "infarction"}
        for row in table["rows"]:
            assert set(row["coverage_by_class"]) == {"0", "1"}
            for figures in row["coverage_by_class"].values():
                assert figures["n_draws"] >= MIN_DRAWS

    def test_the_split_is_drawn_over_patients_not_records(self, table: dict) -> None:
        assert "patient" in table["split"]
        for row in table["rows"]:
            assert row["n_patients"] < row["n_points"]


class TestTheEncoderArmsOnRecord:
    """C-12 and C-17: every cached representation says where its weights came
    from and what they were pre-trained on, and the random arm is one of them."""

    @pytest.fixture(scope="module")
    def sidecars(self) -> list[dict]:
        paths = sorted((RESULTS_DIR / "embeddings").glob("*/*.json"))
        assert paths, "no embedding sidecars committed"
        return [json.loads(p.read_text()) for p in paths]

    def test_every_arm_covers_the_three_corpora(self, sidecars: list[dict]) -> None:
        pairs = {(s["arm"], s["corpus"]) for s in sidecars}
        assert {a for a, _ in pairs} == {"random_init", "ecgfounder", "hubert_ecg", "ecgfm"}
        for arm in {a for a, _ in pairs}:
            assert {c for a, c in pairs if a == arm} == {"ptbxl", "sph", "acs"}

    def test_each_one_names_its_weights_pre_training_chain_and_commit(
        self, sidecars: list[dict]
    ) -> None:
        for sidecar in sidecars:
            for field in ("weights_source", "pretraining_corpora", "preprocessing"):
                assert sidecar[field], (sidecar["arm"], sidecar["corpus"], field)
            assert len(sidecar["git_commit"]) == 40, sidecar["arm"]

    def test_the_frozen_random_arm_is_among_them(self, sidecars: list[dict]) -> None:
        random_arm = [s for s in sidecars if s["arm"] == "random_init"]
        assert len(random_arm) == 3
        assert all("random initialisation" in s["pretraining_corpora"] for s in random_arm)

    def test_the_vector_count_matches_what_ingestion_kept(self, sidecars: list[dict]) -> None:
        ingest = json.loads((RESULTS_DIR / "ingest_report.json").read_text())
        for sidecar in sidecars:
            expected = ingest[sidecar["corpus"]]["n_kept"]
            assert sidecar["n_kept"] == expected, (sidecar["arm"], sidecar["corpus"])


class TestTheFigures:
    """C-20: the report holds exactly the figures the plan named, each one
    redrawn by a script from a results file that is already committed."""

    NAMED = {
        1: ("fig1_coverage.png", RESULTS_DIR / "shift.json"),
        2: ("fig2_set_sizes.png", RESULTS_DIR / "shift.json"),
        3: ("fig3_arms.png", RESULTS_DIR / "arms.json"),
        4: ("fig4_discrimination.png", RESULTS_DIR / "baseline/metrics.json"),
    }

    def test_each_figure_redraws_from_the_committed_numbers(self, tmp_path: Path) -> None:
        import figures

        assert figures.main(["--figure", "1", "2", "3", "4", "--out", str(tmp_path)]) == 0
        for number, (name, _source) in self.NAMED.items():
            drawn = tmp_path / name
            assert drawn.exists(), number
            assert drawn.stat().st_size > 10_000, f"{name} is too small to hold a plot"

    def test_the_file_each_one_is_drawn_from_is_already_committed(self) -> None:
        for number, (_name, source) in self.NAMED.items():
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(source)],
                capture_output=True,
                text=True,
                cwd=RESULTS_DIR.parent,
                check=False,
            )
            assert tracked.returncode == 0, f"figure {number} draws from an uncommitted {source}"

    def test_figure_one_carries_a_panel_per_corpus_and_per_correction(self, tmp_path: Path) -> None:
        """A figure with two of the three hospitals on it, or with a correction
        left off, would read as a result. Both counts are checked against what
        the table holds rather than against a number written here."""
        import figures

        table = json.loads((RESULTS_DIR / "shift.json").read_text())
        assert set(figures.corpora_on(table)) == {"ptbxl", "sph", "acs"}
        assert figures.corrections_on(table) == ["none", "mondrian", "weighted"]
        assert set(figures.corrections_on(table)) == {r["correction"] for r in table["rows"]}
        drawn = figures.figure_1_coverage(table, tmp_path / "fig1.png", RESULTS_DIR / "shift.json")
        assert drawn.stat().st_size > 10_000

    def test_figure_three_carries_every_arm_and_every_target(self, tmp_path: Path) -> None:
        """A figure missing an arm would read as a grid with a row deleted."""
        import figures

        grid = json.loads((RESULTS_DIR / "arms.json").read_text())
        assert set(figures.ARM_ORDER) == set(grid["arms"])
        drawn = figures.figure_3_arms(grid, tmp_path / "fig3.png", RESULTS_DIR / "arms.json")
        assert drawn.stat().st_size > 10_000

    def test_figure_three_says_what_it_is_waiting_for_rather_than_drawing_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The behaviour that keeps an absent result from becoming a blank plot."""
        import figures

        monkeypatch.setattr(figures, "RESULTS_DIR", tmp_path / "empty")
        assert figures.main(["--figure", "3", "--out", str(tmp_path)]) == 0
        assert not list(tmp_path.glob("*.png"))
        assert "does not exist yet" in capsys.readouterr().err


# The shifted corpora are scored once and whole (C-20), so unlike the re-draw
# harness above the test sample never moves: whatever that one sample happens to
# be, re-drawing the calibration cannot average its own sampling error away.  The
# band a fixed target is held to is therefore C-5's band widened by three
# standard errors of a proportion on that target's own size.
FIXED_TARGET_SIGMAS = 3.0

# The synthetic shift, sized to the one the study measures: a quarter of the
# source sick against a fiftieth of the target.
SOURCE_N = 3000
SOURCE_PREVALENCE = 0.25
TARGET_N = 20_000
TARGET_PREVALENCE = 0.02


def _fixed_target_band(alpha: float, n: int) -> tuple[float, float]:
    slack = FIXED_TARGET_SIGMAS * float(np.sqrt(alpha * (1.0 - alpha) / n))
    return COVERAGE_BAND[0] - slack, COVERAGE_BAND[1] + slack


def _pool(n: int = 45_000, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """A large calibrated two-class pool to cut source and target samples from.

    Because both samples are cut from the same pool, P(score | class) is identical
    on either side by construction and the only thing that can differ is the share
    of each class -- which isolates the shift this study measures.
    """
    rng = np.random.default_rng(seed)
    p1 = rng.uniform(0.05, 0.95, n)
    labels = (rng.uniform(size=n) < p1).astype(int)
    return np.column_stack([1.0 - p1, p1]), labels


def _sample_at(
    probs: np.ndarray, labels: np.ndarray, n: int, prevalence: float, taken: set[int], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` records at the asked-for share of class 1, disjoint from ``taken``."""
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for klass, count in ((1, round(n * prevalence)), (0, n - round(n * prevalence))):
        available = [i for i in np.flatnonzero(labels == klass) if i not in taken]
        picked = rng.choice(available, count, replace=False)
        chosen.extend(int(i) for i in picked)
        taken.update(int(i) for i in picked)
    index = np.asarray(sorted(chosen))
    return probs[index], labels[index]


def _source(probs: np.ndarray, labels: np.ndarray, **kwargs: object) -> Source:
    return Source("source", probs, labels, [f"p{i}" for i in range(len(labels))], **kwargs)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def unshifted_table() -> list[dict]:
    """Source and target cut from the same pool at the same prevalence: no shift
    at all, so the guarantee must survive the journey intact."""
    probs, labels = _pool()
    taken: set[int] = set()
    source = _source(*_sample_at(probs, labels, SOURCE_N, SOURCE_PREVALENCE, taken, seed=1))
    target_probs, target_labels = _sample_at(
        probs, labels, TARGET_N, SOURCE_PREVALENCE, taken, seed=2
    )
    targets = {"elsewhere": Target(target_probs, target_labels, n_patients=TARGET_N)}
    return frozen_calibration_table(source, targets, (0.20, 0.10, 0.05), n_draws=120, seed=5)


@pytest.fixture(scope="module")
def shifted_table() -> list[dict]:
    """The same P(score | class) on both sides, a quarter of the source sick
    against a fiftieth of the target."""
    probs, labels = _pool()
    taken: set[int] = set()
    source = _source(*_sample_at(probs, labels, SOURCE_N, SOURCE_PREVALENCE, taken, seed=1))
    target_probs, target_labels = _sample_at(
        probs, labels, TARGET_N, TARGET_PREVALENCE, taken, seed=2
    )
    targets = {"elsewhere": Target(target_probs, target_labels, n_patients=TARGET_N)}
    return frozen_calibration_table(source, targets, (0.10,), n_draws=120, seed=5)


def _row(table: list[dict], score: str, correction: str, alpha: float = 0.10) -> dict:
    found = [
        r
        for r in table
        if r["score"] == score and r["correction"] == correction and r["alpha"] == alpha
    ]
    assert len(found) == 1
    return found[0]


class TestTheFrozenCalibrationOnDataWhoseAnswerIsKnown:
    """The break harness, on data built so that theory says what must come out.

    C-20 is in force throughout: the threshold is fitted on the source sample
    alone and spent unchanged on a target that is never subsampled, never
    re-calibrated, and scored whole every draw.
    """

    def test_without_a_shift_the_frozen_threshold_still_covers_at_the_level_asked_for(
        self, unshifted_table: list[dict]
    ) -> None:
        for row in unshifted_table:
            if row["correction"] != "none":
                continue
            block = row["by_corpus"]["elsewhere"]
            got, target = block["coverage"]["mean"], row["target_coverage"]
            low, high = _fixed_target_band(row["alpha"], block["n_points"])
            assert target + low <= got <= target + high, (row["score"], target, got)

    def test_no_target_label_ever_reaches_a_threshold(self) -> None:
        """C-20's falsifier, and the one the weighted correction has to survive.

        The same source is given a target twice: once as it is, once with every
        label inverted.  A threshold that read a target label -- or a target
        score, which needs the label to be picked out -- could not come back
        bit-identical from both, whatever the correction."""
        probs, labels = _pool(n=8000, seed=3)
        source = _source(probs[:2000], labels[:2000])
        upright: list[dict[str, Any]] = cast(
            "list[dict[str, Any]]",
            frozen_calibration_table(
                source,
                {"t": Target(probs[2000:5000], labels[2000:5000], 3000)},
                (0.10,),
                40,
                seed=7,
            ),
        )
        inverted: list[dict[str, Any]] = cast(
            "list[dict[str, Any]]",
            frozen_calibration_table(
                source,
                {"t": Target(probs[2000:5000], 1 - labels[2000:5000], 3000)},
                (0.10,),
                40,
                seed=7,
            ),
        )
        for here, there in zip(upright, inverted, strict=True):
            for corpus in ("source", "t"):
                assert (
                    here["by_corpus"][corpus]["threshold_by_class"]
                    == there["by_corpus"][corpus]["threshold_by_class"]
                ), (here["score"], here["correction"], corpus)
                assert (
                    here["by_corpus"][corpus]["calibration"]
                    == there["by_corpus"][corpus]["calibration"]
                ), (here["score"], here["correction"], corpus)
            assert here["calibration"] == there["calibration"]

    def test_an_unweighted_threshold_does_not_depend_on_which_corpus_it_is_spent_on(self) -> None:
        """The two corrections that estimate nothing spend one threshold
        everywhere, and the file carries that as three identical blocks rather
        than as a claim in prose. The weighted one is excluded here because it
        reads the target's class mix -- which is the whole difference this day
        measures, and is asserted separately."""
        probs, labels = _pool(n=8000, seed=3)
        source = _source(probs[:2000], labels[:2000])
        table: list[dict[str, Any]] = cast(
            "list[dict[str, Any]]",
            frozen_calibration_table(
                source,
                {"a": Target(probs[2000:5000], labels[2000:5000], 3000)},
                (0.10,),
                40,
                seed=7,
            ),
        )
        for row in table:
            blocks = [block["threshold_by_class"] for block in row["by_corpus"].values()]
            if row["correction"] == "weighted":
                assert row["threshold_depends_on_target"] is True
                assert "unlabelled" in row["target_information_used"]
            else:
                assert row["threshold_depends_on_target"] is False
                assert blocks[0] == blocks[1], (row["score"], row["correction"])

    def test_the_in_distribution_panel_reproduces_the_re_draw_harness(self) -> None:
        """The two harnesses answer the same question about the source and must
        agree there: same halving, same threshold, same held-out half."""
        probs, labels, patients = _calibrated_problem(n=1500, seed=8)
        frozen = _row(
            frozen_calibration_table(_source(probs, labels), {}, (0.10,), 60, seed=4),
            "lac",
            "none",
        )
        redrawn = repeated_split_report(probs, labels, patients, 0.10, n_draws=60, seed=4)
        assert frozen["by_corpus"]["source"]["coverage"]["mean"] == pytest.approx(
            redrawn["coverage"]["mean"]  # type: ignore[index]
        )


class TestWhatAPrevalenceShiftDoesToTheGuarantee:
    """What breaks and what does not when only the share of sick patients moves."""

    @pytest.mark.parametrize("score", ["lac", "aps"])
    def test_one_threshold_per_class_holds_both_classes_through_the_shift(
        self, shifted_table: list[dict], score: str
    ) -> None:
        """Mondrian's guarantee is class-conditional, so a change of class
        proportions cannot touch it -- whatever share of the target is sick."""
        row = _row(shifted_table, score, "mondrian")
        block = row["by_corpus"]["elsewhere"]
        for klass, figures in block["coverage_by_class"].items():
            support = block["n_points"] * (
                block["prevalence"] if klass == "1" else 1 - block["prevalence"]
            )
            low, high = _fixed_target_band(row["alpha"], round(support))
            assert 0.9 + low <= figures["mean"] <= 0.9 + high, (score, klass, figures["mean"])

    @pytest.mark.parametrize("score", ["lac", "aps"])
    def test_the_sick_are_covered_exactly_as_badly_on_either_side_of_the_shift(
        self, shifted_table: list[dict], score: str
    ) -> None:
        """With one shared threshold, what a sick patient gets is fixed by
        P(score | sick), and that is identical on both sides here. The
        class-conditional figure therefore travels unchanged."""
        row = _row(shifted_table, score, "none")
        here = row["by_corpus"]["source"]["coverage_by_class"]["1"]["mean"]
        there = row["by_corpus"]["elsewhere"]["coverage_by_class"]["1"]["mean"]
        assert there == pytest.approx(here, abs=0.04), (score, here, there)

    @pytest.mark.parametrize("score", ["lac", "aps"])
    def test_the_marginal_figure_is_the_source_per_class_figures_at_the_target_mix(
        self, shifted_table: list[dict], score: str
    ) -> None:
        """The quantitative claim the report rests on: under a pure change of
        prior, the target's overall coverage is predictable from numbers measured
        entirely on the source plus the target's share of sick patients. A
        threshold quietly re-fitted on the target, or labels read off the wrong
        rows, would not land here."""
        row = _row(shifted_table, score, "none")
        source_block = row["by_corpus"]["source"]
        target_block = row["by_corpus"]["elsewhere"]
        q1 = target_block["prevalence"]
        predicted = (
            q1 * source_block["coverage_by_class"]["1"]["mean"]
            + (1 - q1) * source_block["coverage_by_class"]["0"]["mean"]
        )
        assert target_block["coverage"]["mean"] == pytest.approx(predicted, abs=0.02), score

    @pytest.mark.parametrize("score", ["lac", "aps"])
    def test_how_far_the_overall_figure_moves_is_the_gap_between_the_classes(
        self, shifted_table: list[dict], score: str
    ) -> None:
        """Why an overall coverage number cannot be carried across hospitals, and
        exactly how far it can be wrong: under a pure change of prior the overall
        figure moves by the change in the share of sick patients times the gap
        between what the sick and the healthy are covered at. A model whose two
        classes are covered alike barely moves; one that abandons the sick moves
        by that whole gap, without a single patient being treated differently."""
        row = _row(shifted_table, score, "none")
        source_block, target_block = row["by_corpus"]["source"], row["by_corpus"]["elsewhere"]
        for klass in ("0", "1"):
            assert target_block["coverage_by_class"][klass]["mean"] == pytest.approx(
                source_block["coverage_by_class"][klass]["mean"], abs=0.04
            )
        gap = (
            source_block["coverage_by_class"]["1"]["mean"]
            - source_block["coverage_by_class"]["0"]["mean"]
        )
        expected = (target_block["prevalence"] - source_block["prevalence"]) * gap
        moved = target_block["coverage"]["mean"] - source_block["coverage"]["mean"]
        assert moved == pytest.approx(expected, abs=0.02), (score, moved, expected)


class TestTheWeightedCorrection:
    """The estimated correction, on the shift built so theory says what comes out.

    Its two claims are different in kind from Mondrian's.  Mondrian is exact and
    needs nothing: the only question is whether the class had enough points.  The
    weighted correction is exact only if the target prior it was handed is right,
    so what has to be shown is where the estimate comes from, what it costs, and
    what happens when it is wrong.
    """

    def test_on_an_unshifted_target_the_weighting_collapses_to_no_correction(self) -> None:
        """When the target prior equals the source's, w(y) = 1 and the weighted
        quantile is the ordinary one. A correction that moved the threshold here
        would be reading something other than the class mix."""
        probs, labels = _pool()
        taken: set[int] = set()
        source = _source(*_sample_at(probs, labels, SOURCE_N, SOURCE_PREVALENCE, taken, seed=1))
        target_probs, target_labels = _sample_at(
            probs, labels, TARGET_N, SOURCE_PREVALENCE, taken, seed=2
        )
        table = frozen_calibration_table(
            source,
            {"elsewhere": Target(target_probs, target_labels, n_patients=TARGET_N)},
            (0.10,),
            n_draws=120,
            seed=5,
        )
        for score in ("lac", "aps"):
            plain = _row(table, score, "none")["by_corpus"]["elsewhere"]
            weighted = _row(table, score, "weighted")["by_corpus"]["elsewhere"]
            assert weighted["threshold_by_class"]["1"]["mean"] == pytest.approx(
                plain["threshold_by_class"]["1"]["mean"], abs=0.01
            ), score
            n = _row(table, score, "weighted")["calibration"]["n"]["mean"]
            assert weighted["calibration"]["effective_sample_size"]["mean"] > 0.95 * n, score

    def test_the_estimated_prior_follows_the_target_and_costs_effective_sample_size(
        self, shifted_table: list[dict]
    ) -> None:
        """C-9's reason for existing. Reweighting 25% sick calibration points to
        look like a 2% sick target puts most of the mass on one class, and the
        effective sample size is what says how few points the restored guarantee
        actually rests on."""
        for score in ("lac", "aps"):
            row = _row(shifted_table, score, "weighted")
            block = row["by_corpus"]["elsewhere"]
            estimated = block["calibration"]["estimated_prevalence"]["mean"]
            assert abs(estimated - TARGET_PREVALENCE) < abs(estimated - SOURCE_PREVALENCE), (
                score,
                estimated,
            )
            effective = block["calibration"]["effective_sample_size"]["mean"]
            assert effective < 0.9 * row["calibration"]["n"]["mean"], (score, effective)
            assert block["calibration"]["weight_by_class"]["1"]["mean"] < 1.0
            assert block["calibration"]["weight_by_class"]["0"]["mean"] > 1.0

    def test_the_source_panel_is_left_alone_because_its_own_mix_did_not_move(
        self, shifted_table: list[dict]
    ) -> None:
        """Every corpus gets its own estimated prior, the source's held-out half
        included. There the estimate reproduces the mix the threshold was fitted
        at, so the weighted threshold must land on the uncorrected one -- the
        control that shows the correction is driven by the shift and not by the
        act of weighting."""
        for score in ("lac", "aps"):
            plain = _row(shifted_table, score, "none")["by_corpus"]["source"]
            weighted = _row(shifted_table, score, "weighted")["by_corpus"]["source"]
            assert weighted["threshold_by_class"]["1"]["mean"] == pytest.approx(
                plain["threshold_by_class"]["1"]["mean"], abs=0.01
            ), score

    def test_a_target_prior_that_cannot_be_identified_widens_every_set(self) -> None:
        """A predictor that says the same thing about everyone carries no
        information about the target's class mix. The honest answer is an
        infinite threshold -- every label kept -- counted on the file, not a
        prior invented to keep the column full."""
        probs, labels = _pool(n=6000, seed=12)
        flat = np.column_stack([np.full(3000, 0.7), np.full(3000, 0.3)])
        source = _source(probs[:3000], labels[:3000])
        table = frozen_calibration_table(
            source, {"flat": Target(flat, labels[3000:], 3000)}, (0.10,), n_draws=20, seed=3
        )
        for score in ("lac", "aps"):
            block = _row(table, score, "weighted")["by_corpus"]["flat"]
            assert block["calibration"]["n_unidentified"] == 20, score
            assert block["threshold_by_class"]["1"]["n_infinite"] == 20, score
            assert block["coverage"]["mean"] == pytest.approx(1.0), score


class TestWhatTheBreakTableCarries:
    """C-9, C-10, C-11 and C-14 on the harness's own output, before any corpus."""

    @pytest.fixture(scope="class")
    @staticmethod
    def table() -> list[dict]:
        probs, labels = _pool(n=6000, seed=4)
        source = _source(probs[:2000], labels[:2000], deviations=("nothing gave here",))
        targets = {
            "elsewhere": Target(
                probs[2000:], labels[2000:], n_patients=4000, deviations=("a named deviation",)
            )
        }
        return frozen_calibration_table(source, targets, (0.20, 0.10, 0.05), n_draws=101, seed=2)

    def test_it_holds_one_row_per_level_score_and_correction(self, table: list[dict]) -> None:
        assert len(table) == 3 * 2 * 3
        assert sorted({r["alpha"] for r in table}) == [0.05, 0.10, 0.20]
        assert {r["correction"] for r in table} == {"none", "mondrian", "weighted"}

    def test_every_figure_on_every_corpus_is_a_mean_over_at_least_a_hundred_draws(
        self, table: list[dict]
    ) -> None:
        for row in table:
            for block in row["by_corpus"].values():
                for key in ("coverage", "empty_rate", "one_label_rate", "two_label_rate"):
                    assert block[key]["n_draws"] >= MIN_DRAWS
                    assert block[key]["sd"] >= 0.0
                for figures in block["coverage_by_class"].values():
                    assert figures["n_draws"] >= MIN_DRAWS

    def test_every_cell_reports_the_effective_size_of_what_calibrated_it(
        self, table: list[dict]
    ) -> None:
        """C-9, on every (level, score, correction, corpus) cell of the table.

        The two unweighted corrections spend every calibration point at weight
        one, so their effective size is the count itself and the file says so
        rather than leaving it to be assumed.  The weighted one reweights, so
        its effective size can only be lower -- that is the price the criterion
        exists to keep visible."""
        for row in table:
            n = row["calibration"]["n"]["mean"]
            for corpus, block in row["by_corpus"].items():
                calibration = block["calibration"]
                effective = calibration["effective_sample_size"]["mean"]
                assert calibration["effective_sample_size"]["n_draws"] >= MIN_DRAWS
                if row["correction"] == "weighted":
                    assert 0 < effective <= n, (corpus, effective, n)
                    assert "BBSE" in calibration["weighting"]
                    assert calibration["n_unidentified"] == 0
                    assert 0.0 <= calibration["estimated_prevalence"]["mean"] <= 1.0
                else:
                    assert effective == pytest.approx(n), (row["correction"], corpus)
                    assert "uniform" in calibration["weighting"]
                    for weight in calibration["weight_by_class"].values():
                        assert weight["mean"] == pytest.approx(1.0)

    def test_the_split_reports_what_each_class_is_calibrated_on(self, table: list[dict]) -> None:
        """C-9's other half: the Mondrian threshold for a class is only as good
        as the points that class was left with, and the minority class is where
        that bites, so the count per class is on the row beside the total."""
        for row in table:
            calibration = row["calibration"]
            assert set(calibration["n_by_class"]) == {"0", "1"}
            counted = sum(figures["mean"] for figures in calibration["n_by_class"].values())
            assert counted == pytest.approx(calibration["n"]["mean"])
            for figures in calibration["n_by_class"].values():
                assert figures["mean"] > 0

    def test_every_corpus_names_what_could_not_be_made_identical(self, table: list[dict]) -> None:
        """C-14: the deviations travel with the numbers, not beside them."""
        for row in table:
            assert row["by_corpus"]["source"]["deviations"] == ["nothing gave here"]
            assert row["by_corpus"]["elsewhere"]["deviations"] == ["a named deviation"]

    def test_the_threshold_it_spent_is_in_every_corpus_block(self, table: list[dict]) -> None:
        """The threshold lives beside the corpus it was spent on, because one of
        the three corrections makes it depend on that corpus."""
        for row in table:
            for corpus, block in row["by_corpus"].items():
                by_class = block["threshold_by_class"]
                assert set(by_class) == {"0", "1"}, corpus
                for figures in by_class.values():
                    assert figures["n_draws"] == 101
                    assert figures["n_infinite"] == 0
                if row["correction"] == "none":
                    assert by_class["0"]["mean"] == pytest.approx(by_class["1"]["mean"])

    def test_a_tighter_level_never_lowers_the_threshold(self, table: list[dict]) -> None:
        for score in ("lac", "aps"):
            for correction in ("none", "mondrian", "weighted"):
                for corpus in ("source", "elsewhere"):
                    rows = sorted(
                        (r for r in table if r["score"] == score and r["correction"] == correction),
                        key=lambda r: -float(r["alpha"]),
                    )
                    means = [
                        r["by_corpus"][corpus]["threshold_by_class"]["1"]["mean"] for r in rows
                    ]
                    assert means == sorted(means), (score, correction, corpus, means)


class TestTheCommittedBreakTable:
    """The three-corpus table itself, held to C-9, C-10, C-11, C-14 and C-20."""

    # Read off the corpora's own description files on 2026-08-22 and asserted in
    # test_labels.py; repeated here because a table whose prevalences drifted is
    # measuring a different shift from the one the study describes.
    PREVALENCE = {"ptbxl": 0.2502, "sph": 0.0101, "acs": 0.1492}

    @pytest.fixture(scope="class")
    @staticmethod
    def table() -> dict:
        return json.loads((RESULTS_DIR / "shift.json").read_text())

    def test_all_three_corpora_are_on_every_row(self, table: dict) -> None:
        for row in table["rows"]:
            assert set(row["by_corpus"]) == {"ptbxl", "sph", "acs"}

    def test_it_covers_the_three_levels_both_scores_and_all_three_corrections(
        self, table: dict
    ) -> None:
        assert sorted({row["alpha"] for row in table["rows"]}) == [0.05, 0.10, 0.20]
        assert {row["score"] for row in table["rows"]} == {"lac", "aps"}
        assert {row["correction"] for row in table["rows"]} == {"none", "mondrian", "weighted"}
        assert set(table["corrections"]) == {"none", "mondrian", "weighted"}

    def test_every_threshold_was_fitted_on_ptbxl_and_nowhere_else(self, table: dict) -> None:
        """C-20 on the committed file: the protocol is named, every row says which
        corpus its threshold came from, and every row says what -- if anything --
        the target supplied.

        Only the weighted correction is allowed to differ by corpus, and the file
        has to say what it read to differ: the target's unlabelled class mix, and
        nothing that would make the corpus a calibration set for itself."""
        assert table["calibrated_on"] == "ptbxl"
        assert "never re-calibrated on themselves" in table["protocol"]
        assert "no target label" in table["what_the_target_supplied"]
        for row in table["rows"]:
            assert row["calibrated_on"] == "ptbxl"
            spent = [block["threshold_by_class"] for block in row["by_corpus"].values()]
            if row["correction"] == "weighted":
                assert row["threshold_depends_on_target"] is True
                assert "unlabelled" in row["target_information_used"]
            else:
                assert row["threshold_depends_on_target"] is False
                assert spent[1:] == spent[:-1], (row["score"], row["correction"])

    def test_every_figure_is_a_mean_over_at_least_a_hundred_draws_with_its_spread(
        self, table: dict
    ) -> None:
        assert table["n_draws"] >= MIN_DRAWS
        for row in table["rows"]:
            for block in row["by_corpus"].values():
                for key in ("coverage", "empty_rate", "one_label_rate", "two_label_rate"):
                    assert block[key]["n_draws"] >= MIN_DRAWS
                    assert block[key]["sd"] >= 0.0

    def test_coverage_is_reported_for_infarction_and_for_not_on_every_corpus(
        self, table: dict
    ) -> None:
        assert table["classes"] == {"0": "no infarction", "1": "infarction"}
        for row in table["rows"]:
            for block in row["by_corpus"].values():
                assert set(block["coverage_by_class"]) == {"0", "1"}
                for figures in block["coverage_by_class"].values():
                    assert figures["n_draws"] >= MIN_DRAWS

    def test_every_weighted_cell_reports_what_the_weighting_cost_it(self, table: dict) -> None:
        """C-9 on the committed file. The two exact corrections spend every point
        at weight one; the weighted one does not, and on the two corpora whose
        class mix actually moved it must show a smaller effective sample than the
        count -- which is the whole reason the criterion exists."""
        for row in table["rows"]:
            n = row["calibration"]["n"]["mean"]
            assert n > 0
            for corpus, block in row["by_corpus"].items():
                effective = block["calibration"]["effective_sample_size"]["mean"]
                if row["correction"] != "weighted":
                    assert effective == pytest.approx(n), (row["correction"], corpus)
                    continue
                lost = block["calibration"]["n_unidentified"]
                assert lost == block["threshold_by_class"]["1"]["n_infinite"], corpus
                assert lost < 0.05 * table["n_draws"], (corpus, lost)
                assert block["calibration"]["effective_sample_size"]["n_draws"] == (
                    table["n_draws"] - lost
                )
                assert 0.0 < block["calibration"]["estimated_prevalence"]["mean"] < 1.0
                if corpus == "ptbxl":
                    assert effective > 0.98 * n, "the source's own mix did not move"
                else:
                    assert effective < 0.95 * n, (corpus, effective, n)

    def test_the_split_reports_what_the_minority_class_is_calibrated_on(self, table: dict) -> None:
        """C-9's other half: a Mondrian threshold is only as good as the points
        its class was left with, so the row carries the count per class."""
        for row in table["rows"]:
            by_class = row["calibration"]["n_by_class"]
            assert set(by_class) == {"0", "1"}
            assert by_class["1"]["mean"] < by_class["0"]["mean"], "MI is the minority at home"
            counted = sum(figures["mean"] for figures in by_class.values())
            assert counted == pytest.approx(row["calibration"]["n"]["mean"])

    def test_the_reading_is_read_back_off_the_rows(self, table: dict) -> None:
        """The answer the day was for cannot drift from the table it sits on: the
        paired differences it quotes are recomputed here from the per-draw series
        the rows carry, and the headline figures it names are matched as text."""
        reading = table["reading"]
        headline = [r for r in table["rows"] if r["alpha"] == 0.10 and r["score"] == "lac"]
        by_correction = {row["correction"]: row for row in headline}
        for corpus, blocks in reading["paired"].items():
            for pair, quoted in blocks["sick coverage"].items():
                left, right = pair.split(" - ")
                difference = np.asarray(
                    by_correction[left]["by_corpus"][corpus]["coverage_by_class_by_draw"]["1"]
                ) - np.asarray(
                    by_correction[right]["by_corpus"][corpus]["coverage_by_class_by_draw"]["1"]
                )
                assert quoted["mean"] == pytest.approx(float(difference.mean()), abs=5e-5)
                assert quoted["sd"] == pytest.approx(float(difference.std(ddof=1)), abs=5e-5)
                assert quoted["n_draws"] == table["n_draws"]
        prose = " ".join(
            reading[key]
            for key in (
                "which_correction_this_shift_needed",
                "what_each_costs",
                "what_neither_repairs",
                "what_cannot_be_claimed",
            )
        )
        for correction in ("none", "mondrian", "weighted"):
            for corpus in ("sph", "acs"):
                sick = by_correction[correction]["by_corpus"][corpus]["coverage_by_class"]["1"]
                assert f"{sick['mean']:.3f}" in prose, (correction, corpus)
        at_home = by_correction["none"]["by_corpus"]["ptbxl"]["coverage_by_class"]["1"]["mean"]
        assert f"{at_home:.3f}" in prose, "the break at home is what the answer turns on"
        weighted = by_correction["weighted"]["by_corpus"]
        for corpus in ("ptbxl", "sph", "acs"):
            estimated = weighted[corpus]["calibration"]["estimated_prevalence"]["mean"]
            assert f"{estimated:.4f}" in prose, corpus
            effective = weighted[corpus]["calibration"]["effective_sample_size"]["mean"]
            assert f"{effective:.0f}" in prose, corpus
        assert "cannot know which of the two promises applies" in prose

    def test_each_corpus_names_what_could_not_be_made_identical(self, table: dict) -> None:
        """C-14: every external corpus states its own ingestion and label deviations,
        and the file would rather carry an awkward one than drop it."""
        for row in table["rows"]:
            for corpus in ("ptbxl", "sph", "acs"):
                assert row["by_corpus"][corpus]["deviations"], corpus
        deviations = " ".join(table["rows"][0]["by_corpus"]["acs"]["deviations"])
        assert "acute" in deviations, "the label mismatch Chongqing carries must be on the file"

    def test_the_prevalences_are_the_ones_the_study_is_built_on(self, table: dict) -> None:
        for corpus, expected in self.PREVALENCE.items():
            got = table["rows"][0]["by_corpus"][corpus]["prevalence"]
            assert got == pytest.approx(expected, abs=5e-4), corpus

    def test_every_corpus_names_the_commit_that_scored_it(self, table: dict) -> None:
        for corpus in ("ptbxl", "sph", "acs"):
            assert len(table["corpora"][corpus]["git_commit"]) == 40, corpus
        assert len(table["git_commit"]) == 40

    def test_every_corpus_was_scored_whole_against_what_ingestion_kept(self, table: dict) -> None:
        """The external corpora are not subsampled: the row count on the table is
        the count the score file wrote, and that is the corpus minus its named
        exclusions."""
        for corpus in ("sph", "acs"):
            sidecar = json.loads((RESULTS_DIR / f"external/{corpus}.json").read_text())
            assert table["rows"][0]["by_corpus"][corpus]["n_points"] == sidecar["n_scored"]
            assert table["corpora"][corpus]["n_scored"] == sidecar["n_scored"]

    def test_where_it_was_calibrated_the_guarantee_still_holds(self, table: dict) -> None:
        """The control that licenses reading the other two panels: on the PTB-XL
        patients the threshold was not fitted on, coverage lands where it was
        asked to. A break there would mean the harness, not the hospital."""
        for row in table["rows"]:
            block = row["by_corpus"]["ptbxl"]
            got, target = block["coverage"]["mean"], row["target_coverage"]
            low, high = target + COVERAGE_BAND[0], target + COVERAGE_BAND[1]
            assert low <= got <= high, (row["score"], row["correction"], target, got)


class TestTheCommittedArmGrid:
    """C-12, C-17 and C-18 on ``results/arms.json``: four arms on one break,
    each with intervals, and every arm-versus-arm claim made pairwise."""

    ARMS = ("random_init", "ecgfounder", "ecgfm", "hubert_ecg")
    CORPORA = ("ptbxl", "sph", "acs")
    TARGETS = ("sph", "acs")

    @pytest.fixture(scope="module")
    def grid(self) -> dict:
        path = RESULTS_DIR / "arms.json"
        if not path.exists():
            pytest.skip(f"{path} has not been written yet")
        return json.loads(path.read_text())

    def test_the_frozen_random_arm_is_reported_beside_the_others(self, grid: dict) -> None:
        """C-17: the control is in the grid, not mentioned in passing."""
        assert set(grid["arms"]) == set(self.ARMS)
        assert "random initialisation" in grid["arms"]["random_init"]["pretraining_corpora"]
        for arm in self.ARMS:
            assert arm in grid["discrimination"], arm
            assert arm in grid["coverage"], arm

    def test_every_arm_names_the_corpora_it_was_pre_trained_on(self, grid: dict) -> None:
        """C-12: the grid's whole claim rests on which arm saw what."""
        for arm in self.ARMS:
            assert grid["arms"][arm]["pretraining_corpora"].strip(), arm

    def test_the_arms_that_saw_the_calibration_corpus_say_so(self, grid: dict) -> None:
        saw = {a for a in self.ARMS if "ptb-xl" in grid["arms"][a]["pretraining_corpora"].lower()}
        assert saw == {"ecgfm", "hubert_ecg"}
        assert "shandong" in grid["arms"]["hubert_ecg"]["pretraining_corpora"].lower()

    def test_every_arm_and_corpus_carries_auroc_and_auprc_with_an_interval(
        self, grid: dict
    ) -> None:
        """C-18, first half: no bare point estimate anywhere on the grid."""
        for arm in self.ARMS:
            for corpus in self.CORPORA:
                block = grid["discrimination"][arm][corpus]
                for metric in ("auroc", "auprc"):
                    low, high = block[f"{metric}_ci95"]
                    assert low <= block[metric] <= high, (arm, corpus, metric)
                    assert high > low, f"{arm} {corpus} {metric} has a zero-width interval"
                assert block["n_positive"] > 0, (arm, corpus)

    def test_every_pair_of_arms_is_compared_paired_on_every_corpus(self, grid: dict) -> None:
        """C-18, second half: an arm-beats-arm claim is never two separate intervals."""
        expected = {
            f"{a} - {b}" for index, a in enumerate(self.ARMS) for b in self.ARMS[index + 1 :]
        }
        for corpus in self.CORPORA:
            assert set(grid["discrimination_paired"][corpus]) == expected, corpus
            for pair, block in grid["discrimination_paired"][corpus].items():
                for metric in ("auroc", "auprc"):
                    cell = block[metric]
                    assert cell["ci95_low"] <= cell["difference"] <= cell["ci95_high"], (
                        corpus,
                        pair,
                        metric,
                    )
                    assert (
                        cell["n_points"]
                        == grid["discrimination"][pair.split(" - ")[0]][corpus]["n_points"]
                    )

    def test_a_paired_difference_is_only_called_separated_when_it_excludes_zero(
        self, grid: dict
    ) -> None:
        for corpus in self.CORPORA:
            for pair, block in grid["discrimination_paired"][corpus].items():
                for metric in ("auroc", "auprc"):
                    cell = block[metric]
                    excludes_zero = cell["ci95_low"] > 0.0 or cell["ci95_high"] < 0.0
                    assert cell["separated"] is excludes_zero, (corpus, pair, metric)

    def test_every_coverage_figure_is_a_mean_over_at_least_a_hundred_draws_with_its_spread(
        self, grid: dict
    ) -> None:
        """C-10, on this table as on the break table."""
        assert grid["n_draws"] >= MIN_DRAWS
        for arm in self.ARMS:
            for row in grid["coverage"][arm]:
                for corpus in self.CORPORA:
                    cell = row["coverage"][corpus]
                    assert cell["n_draws"] >= MIN_DRAWS, (arm, corpus)
                    assert "sd" in cell
                for corpus in self.TARGETS:
                    gap = row["coverage_gap"][corpus]
                    assert gap["n_draws"] >= MIN_DRAWS, (arm, corpus)
                    assert gap["draw_range_2_5"] <= gap["mean"] <= gap["draw_range_97_5"]

    def test_the_gap_is_the_home_coverage_minus_the_target_coverage(self, grid: dict) -> None:
        """Taken inside each draw, so the mean gap is the gap of the means."""
        for arm in self.ARMS:
            for row in grid["coverage"][arm]:
                for corpus in self.TARGETS:
                    expected = row["coverage"]["ptbxl"]["mean"] - row["coverage"][corpus]["mean"]
                    assert row["coverage_gap"][corpus]["mean"] == pytest.approx(expected, abs=1e-9)

    def test_every_arm_covers_every_level_the_break_table_covers(self, grid: dict) -> None:
        """The grid asks what the encoder underneath does to the break, so it holds
        every level and score the break table holds, under the corrections that
        estimate nothing. Which correction repairs the break is the break table's
        question; a grid that answered it too would be measuring two things at
        once. What the grid does report must be a setting the break table also
        reports, so the two files can be read against each other."""
        break_table = json.loads((RESULTS_DIR / "shift.json").read_text())
        settings = {(r["alpha"], r["score"], r["correction"]) for r in break_table["rows"]}
        for arm in self.ARMS:
            reported = {(r["alpha"], r["score"], r["correction"]) for r in grid["coverage"][arm]}
            assert reported <= settings, arm
            assert {(alpha, score) for alpha, score, _ in reported} == {
                (alpha, score) for alpha, score, _ in settings
            }, arm
            assert {correction for _, _, correction in reported} == {"none", "mondrian"}, arm

    def test_the_arms_were_scored_on_the_same_records_as_each_other(self, grid: dict) -> None:
        """A pairwise comparison across different record sets is not a comparison."""
        for corpus in self.CORPORA:
            counts = {grid["discrimination"][arm][corpus]["n_points"] for arm in self.ARMS}
            assert len(counts) == 1, (corpus, counts)

    def test_the_arms_were_scored_on_the_records_the_break_table_used(self, grid: dict) -> None:
        """The grid and day 3's break describe the same populations, or the two
        cannot be read beside each other."""
        break_table = json.loads((RESULTS_DIR / "shift.json").read_text())
        row = _row(break_table["rows"], "lac", "none", 0.20)
        for corpus in self.CORPORA:
            assert (
                grid["discrimination"][self.ARMS[0]][corpus]["n_points"]
                == row["by_corpus"][corpus]["n_points"]
            ), corpus

    def test_each_arm_names_the_regularisation_its_validation_fold_chose(self, grid: dict) -> None:
        """The one hyper-parameter the probe has, on the file with what it scored."""
        for arm in self.ARMS:
            block = grid["arms"][arm]
            chosen = block["regularisation_kept"]
            scored = block["validation_auroc_by_strength"]
            assert str(chosen) in scored, arm
            assert scored[str(chosen)] == max(scored.values()), arm
            assert block["n_train"] > block["n_validation"] > 0, arm

    def test_the_grid_names_the_commit_it_was_computed_at(self, grid: dict) -> None:
        assert len(grid["git_commit"]) == 40
