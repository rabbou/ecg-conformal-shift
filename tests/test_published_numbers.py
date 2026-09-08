"""Recompute the published numbers from the raw scores and hold the files to them.

Nothing here imports ``ecs``.  The conformal quantile, the three schemes and the
patient-level halving are written out a second time, from the definitions, so
that agreement with ``results/outcomes.json`` and ``results/shift.json`` is
evidence rather than a shared bug.  Breaking the quantile rank, the threshold
placement or the outcome accounting in ``src/`` cannot move these expectations,
because these expectations do not come from ``src/``.

This is the check the suite lacked: shifting the conformal quantile by one rank
used to fail six tests and none of them a coverage test, because no test
recomputed any number that the report prints.

The two result files draw their 200 halves differently -- ``outcomes.py`` keeps
one generator and reshuffles, ``splits.py`` builds a fresh one per draw -- so
both draw orders are reproduced here and each file is checked against its own.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from ecs.config import PTBXL_DIR, RESULTS_DIR

ALPHA = 0.10
DRAWS = 200
SEED = 0

# The recomputation reproduces the committed means to well inside a rounding
# step; this is the tolerance a reader would accept on a printed 4-decimal
# figure, not the agreement actually observed.
TOLERANCE = 5e-4


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score; +inf if that rank exceeds n."""
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    rank = math.ceil((ordered.size + 1) * (1.0 - alpha))
    return math.inf if rank > ordered.size else float(ordered[rank - 1])


def halves_outcomes(patients: np.ndarray, draws: int, seed: int) -> list[np.ndarray]:
    """The calibration masks scripts/outcomes.py draws: one generator, shuffled in place."""
    unique = np.unique(patients)
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(draws):
        rng.shuffle(unique)
        held = set(unique[: len(unique) // 2].tolist())
        masks.append(np.array([p in held for p in patients]))
    return masks


def halves_shift(patients: np.ndarray, draws: int, seed: int) -> list[np.ndarray]:
    """The calibration masks src/ecs/splits.py draws: a fresh generator per draw."""
    ids = np.sort(np.unique(patients))
    masks = []
    for draw in range(draws):
        order = ids[np.random.default_rng(seed + draw).permutation(len(ids))]
        held = set(order[: int(np.floor(0.5 * len(ids)))].tolist())
        masks.append(np.array([p in held for p in patients]))
    return masks


def _sets(probs: np.ndarray, plain_q: float, pooled_q: float, per_q: np.ndarray) -> dict:
    """Membership matrices, one per scheme."""
    scores = 1.0 - probs
    positive = probs[:, 1] >= plain_q
    return {
        "plain": np.column_stack([~positive, positive]),
        "pooled": scores <= pooled_q,
        "perlabel": scores <= per_q[None, :],
    }


def _shares(sets: np.ndarray, labels: np.ndarray, klass: int) -> tuple[float, float, float]:
    """correct-alone, deferred, wrong-alone, as shares of the cases carrying `klass`."""
    keep = labels == klass
    alone_1, alone_0 = sets[:, 1] & ~sets[:, 0], sets[:, 0] & ~sets[:, 1]
    deferred = ~(alone_0 | alone_1)
    correct, wrong = (alone_1, alone_0) if klass == 1 else (alone_0, alone_1)
    return (
        float(correct[keep].mean()),
        float(deferred[keep].mean()),
        float(wrong[keep].mean()),
    )


def _recompute(halves: Callable[..., list[np.ndarray]]) -> dict[tuple, float]:
    corpora = {
        "ptbxl": dict(np.load(RESULTS_DIR / "baseline/scores.npz", allow_pickle=False)),
        "sph": dict(np.load(RESULTS_DIR / "external/sph.npz", allow_pickle=False)),
        "acs": dict(np.load(RESULTS_DIR / "external/acs.npz", allow_pickle=False)),
    }
    database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    patients = np.array([str(database.loc[int(i), "patient_id"]) for i in corpora["ptbxl"]["ids"]])

    tally: dict[tuple, list[float]] = {}
    for is_cal in halves(patients, DRAWS, SEED):
        probs = corpora["ptbxl"]["probs"][is_cal]
        labels = corpora["ptbxl"]["labels"][is_cal]
        true = 1.0 - probs[np.arange(labels.size), labels]
        pooled_q = conformal_quantile(true, ALPHA)
        per_q = np.array([conformal_quantile(true[labels == c], ALPHA) for c in (0, 1)])
        plain_q = float(np.quantile(probs[:, 1][labels == 1], ALPHA))
        for name, bundle in corpora.items():
            probs_t, labels_t = bundle["probs"], bundle["labels"]
            if name == "ptbxl":
                probs_t, labels_t = bundle["probs"][~is_cal], bundle["labels"][~is_cal]
            for scheme, sets in _sets(probs_t, plain_q, pooled_q, per_q).items():
                for klass in (0, 1):
                    outcome = _shares(sets, labels_t, klass)
                    for key, value in zip(("correct", "deferred", "wrong"), outcome, strict=True):
                        tally.setdefault((name, scheme, klass, key), []).append(value)
                    tally.setdefault((name, scheme, klass, "coverage"), []).append(
                        float(sets[labels_t == klass, klass].mean())
                    )
    return {key: float(np.mean(values)) for key, values in tally.items()}


@pytest.fixture(scope="module")
def as_outcomes() -> dict[tuple, float]:
    return _recompute(halves_outcomes)


@pytest.fixture(scope="module")
def as_shift() -> dict[tuple, float]:
    return _recompute(halves_shift)


@pytest.mark.data
class TestTableOne:
    """Every cell of the source-site table, recomputed."""

    @pytest.mark.parametrize("scheme", ["plain", "pooled", "perlabel"])
    @pytest.mark.parametrize("klass", [0, 1])
    @pytest.mark.parametrize("outcome", ["correct", "deferred", "wrong"])
    def test_cell_matches_outcomes_json(
        self, as_outcomes: dict[tuple, float], scheme: str, klass: int, outcome: str
    ) -> None:
        published = json.loads((RESULTS_DIR / "outcomes.json").read_text())
        expected = published["by_corpus"]["ptbxl"]["schemes"][scheme][str(klass)][outcome]["mean"]
        assert as_outcomes[("ptbxl", scheme, klass, outcome)] == pytest.approx(
            expected, abs=TOLERANCE
        )

    def test_the_three_outcomes_of_a_label_exhaust_it(
        self, as_outcomes: dict[tuple, float]
    ) -> None:
        for scheme in ("plain", "pooled", "perlabel"):
            for klass in (0, 1):
                total = sum(
                    as_outcomes[("ptbxl", scheme, klass, key)]
                    for key in ("correct", "deferred", "wrong")
                )
                assert total == pytest.approx(1.0, abs=1e-9), (scheme, klass)


@pytest.mark.data
class TestCoveragePerSite:
    """Coverage per site and per label, recomputed, against shift.json."""

    @pytest.mark.parametrize("corpus", ["ptbxl", "sph", "acs"])
    @pytest.mark.parametrize(
        ("scheme", "correction"), [("pooled", "none"), ("perlabel", "mondrian")]
    )
    @pytest.mark.parametrize("klass", [0, 1])
    def test_coverage_matches_shift_json(
        self,
        as_shift: dict[tuple, float],
        corpus: str,
        scheme: str,
        correction: str,
        klass: int,
    ) -> None:
        published = json.loads((RESULTS_DIR / "shift.json").read_text())
        row = next(
            r
            for r in published["rows"]
            if r["alpha"] == ALPHA and r["score"] == "lac" and r["correction"] == correction
        )
        expected = row["by_corpus"][corpus]["coverage_by_class"][str(klass)]["mean"]
        assert as_shift[(corpus, scheme, klass, "coverage")] == pytest.approx(
            expected, abs=TOLERANCE
        )

    def test_coverage_is_correct_plus_deferred(self, as_shift: dict[tuple, float]) -> None:
        """The identity the report asks the reader to check by hand."""
        for corpus in ("ptbxl", "sph", "acs"):
            for scheme in ("pooled", "perlabel"):
                for klass in (0, 1):
                    parts = (
                        as_shift[(corpus, scheme, klass, "correct")]
                        + as_shift[(corpus, scheme, klass, "deferred")]
                    )
                    assert parts == pytest.approx(
                        as_shift[(corpus, scheme, klass, "coverage")], abs=1e-9
                    ), (corpus, scheme, klass)


def test_no_empty_set_arises_in_the_two_schemes_the_report_discusses() -> None:
    """Section 2.2 tells the reader which deferral each scheme can produce.

    With two labels the probabilities sum to one, so a set is empty only when
    the two thresholds admitting each label sum to less than one.  For the
    pooled scheme both thresholds are the same number, which makes the
    condition "below 0.5"; for the class-conditional pair it is the sum that
    matters, and the two are not interchangeable.  The weighted scheme, which
    the report reports but does not recommend, does produce empty sets at
    Shandong, so this covers only the two schemes section 2.2 speaks for.
    """
    scores = dict(np.load(RESULTS_DIR / "baseline/scores.npz", allow_pickle=False))
    probs, labels = scores["probs"], scores["labels"]
    true = 1.0 - probs[np.arange(labels.size), labels]

    pooled = conformal_quantile(true, ALPHA)
    per_label = np.array([conformal_quantile(true[labels == c], ALPHA) for c in (0, 1)])

    for name, pair in (("pooled", np.array([pooled, pooled])), ("perlabel", per_label)):
        assert pair.sum() >= 1.0, (
            f"{name}: an empty set becomes possible once the pair sums below 1"
        )
        sizes = ((1.0 - probs) <= pair[None, :]).sum(axis=1)
        assert sizes.min() >= 1, name


class TestTheShippedCodeAgreesWithTheDefinition:
    """The recomputation above deliberately does not import ``ecs``, so on its own
    it cannot notice the shipped quantile drifting.  These do: the same inputs
    through both implementations, over sizes and levels where the rank rule and
    its +inf edge both bite.
    """

    @pytest.mark.parametrize("n", [1, 2, 9, 10, 19, 20, 99, 100, 1000])
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.20, 0.5])
    def test_the_quantile_matches_rank_for_rank(self, n: int, alpha: float) -> None:
        from ecs.conformal import conformal_quantile as shipped

        rng = np.random.default_rng(n * 1000 + int(alpha * 100))
        scores = rng.random(n)
        mine = conformal_quantile(scores, alpha)
        theirs = shipped(scores, alpha)
        if math.isinf(mine):
            assert math.isinf(theirs), (n, alpha)
        else:
            assert theirs == pytest.approx(mine, abs=1e-12), (n, alpha)

    def test_shifting_the_rank_by_one_would_be_visible(self) -> None:
        """The guard the suite was missing: if this passed for both the correct
        rank and its neighbour, the check above would be worthless."""
        rng = np.random.default_rng(7)
        scores = np.sort(rng.random(200))
        correct = conformal_quantile(scores, ALPHA)
        rank = math.ceil((scores.size + 1) * (1.0 - ALPHA))
        assert scores[rank - 2] != correct
        assert scores[rank] != correct


def _subgroup_masks(ids: np.ndarray) -> dict[str, np.ndarray]:
    """The sex and age-band membership of each scored record, written out here.

    The band edges and the treatment of PTB-XL's privacy age of 300 are restated
    from the dataset's own changelog rather than imported, so that changing them
    in ``scripts/subgroups.py`` cannot quietly move the expectation.
    """
    database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    rows = database.loc[[int(i) for i in ids]]
    age, sex = rows["age"].to_numpy(), rows["sex"].to_numpy()
    masks = {"sex:male": sex == 0, "sex:female": sex == 1}
    for low, high, name in (
        (0, 50, "0-49"),
        (50, 65, "50-64"),
        (65, 75, "65-74"),
        (75, 301, "75+"),
    ):
        masks[f"age:{name}"] = (age >= low) & (age < high)
    return masks


@pytest.fixture(scope="module")
def recomputed_subgroups() -> dict[str, float]:
    """Coverage in every subgroup cell, over the same 200 halves as outcomes.py."""
    bundle = dict(np.load(RESULTS_DIR / "baseline/scores.npz", allow_pickle=False))
    database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    patients = np.array([str(database.loc[int(i), "patient_id"]) for i in bundle["ids"]])
    masks = _subgroup_masks(bundle["ids"])

    tally: dict[str, list[float]] = {}
    for is_cal in halves_outcomes(patients, DRAWS, SEED):
        probs, labels = bundle["probs"][is_cal], bundle["labels"][is_cal]
        true = 1.0 - probs[np.arange(labels.size), labels]
        pooled_q = conformal_quantile(true, ALPHA)
        per_q = np.array([conformal_quantile(true[labels == c], ALPHA) for c in (0, 1)])
        plain_q = float(np.quantile(probs[:, 1][labels == 1], ALPHA))

        probs_t, labels_t = bundle["probs"][~is_cal], bundle["labels"][~is_cal]
        covered_by = {
            scheme: sets[np.arange(labels_t.size), labels_t]
            for scheme, sets in _sets(probs_t, plain_q, pooled_q, per_q).items()
        }
        for group, mask in masks.items():
            held = mask[~is_cal]
            for klass, name in ((1, "mi"), (0, "non_mi"), (None, "all")):
                cell = held if klass is None else held & (labels_t == klass)
                if not cell.any():
                    continue
                for scheme, covered in covered_by.items():
                    tally.setdefault(f"{scheme}|{group}:{name}", []).append(
                        float(covered[cell].mean())
                    )
    return {key: float(np.mean(values)) for key, values in tally.items()}


@pytest.mark.data
class TestSubgroupCoverage:
    """Every cell of results/subgroups.json, recomputed without importing it."""

    @pytest.fixture(scope="class")
    def published(self) -> dict:
        return json.loads((RESULTS_DIR / "subgroups.json").read_text())

    def test_every_cell_matches_the_file(
        self, published: dict, recomputed_subgroups: dict[str, float]
    ) -> None:
        checked = 0
        for scheme, cells in published["coverage"].items():
            for key, cell in cells.items():
                mine = recomputed_subgroups[f"{scheme}|{key}"]
                assert abs(mine - cell["mean"]) < TOLERANCE, (
                    f"{scheme} {key}: file {cell['mean']}, recomputed {mine:.6f}"
                )
                checked += 1
        assert checked == 54, f"expected 3 schemes by 18 cells, checked {checked}"

    def test_the_counts_are_the_fold_and_not_a_draw(self, published: dict) -> None:
        """The cell sizes quoted beside the coverages are whole-fold counts."""
        bundle = dict(np.load(RESULTS_DIR / "baseline/scores.npz", allow_pickle=False))
        masks = _subgroup_masks(bundle["ids"])
        labels = bundle["labels"]
        for group, count in published["counts"].items():
            assert int(masks[group].sum()) == count["n"]
            assert int((masks[group] & (labels == 1)).sum()) == count["n_mi"]
        assert sum(c["n"] for k, c in published["counts"].items() if k.startswith("age:")) == (
            labels.size
        ), "the age bands must account for every scored record, privacy age included"

    def test_a_cell_the_report_quotes_is_not_flat_across_age(
        self, recomputed_subgroups: dict[str, float]
    ) -> None:
        """The non-MI age gradient under the recommended scheme is real, not rounding."""
        young = recomputed_subgroups["perlabel|age:0-49:non_mi"]
        old = recomputed_subgroups["perlabel|age:75+:non_mi"]
        assert young - old > 0.15, f"gradient collapsed to {young - old:.4f}"
