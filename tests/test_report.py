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

import numpy as np
import pytest

from ecs.config import RESULTS_DIR
from ecs.report import repeated_split_report, spread

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
        1: ("fig1_coverage.png", RESULTS_DIR / "abstention.json"),
        2: ("fig2_set_sizes.png", RESULTS_DIR / "abstention.json"),
        4: ("fig4_discrimination.png", RESULTS_DIR / "baseline/metrics.json"),
    }

    def test_each_figure_redraws_from_the_committed_numbers(self, tmp_path: Path) -> None:
        import figures

        assert figures.main(["--figure", "1", "2", "4", "--out", str(tmp_path)]) == 0
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

    def test_figure_three_says_what_it_is_waiting_for_rather_than_drawing_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import figures

        assert figures.main(["--figure", "3", "--out", str(tmp_path)]) == 0
        assert not list(tmp_path.glob("*.png"))
        assert "Shandong and Chongqing" in capsys.readouterr().err
