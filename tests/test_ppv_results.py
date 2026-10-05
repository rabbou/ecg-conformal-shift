"""The committed PPV results, recomputed where the scores are committed, and the prose quoting them.

The infarction rows are recomputed here from the score files without importing
``ecs``: the conformal rank, the counts and Bayes' rule are written out again
from their definitions, so agreement is evidence rather than a shared bug.  The
EchoNext rows cannot be recomputed without restricted data; they are held
against ``echonext_transfer.json``, which reached the same two PPVs through
other code.  Every figure PPV.md and PPV.fr.md print is read back out of the
result files.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from ecs.config import REPO_ROOT, RESULTS_DIR

ALPHA = 0.10
PUBLIC = RESULTS_DIR / "ppv_gap.json"
PRIVATE = RESULTS_DIR / "echonext_ppv_gap.json"
REPAIRS = RESULTS_DIR / "repairs.json"
TRANSFER = RESULTS_DIR / "echonext_transfer.json"
COMPOSITE = "shd_moderate_or_greater_flag"
LVEF = "lvef_lte_45_flag"


def load(path: Any) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


@pytest.fixture(scope="module")
def public() -> dict[str, Any]:
    return load(PUBLIC)


@pytest.fixture(scope="module")
def private() -> dict[str, Any]:
    return load(PRIVATE)


@pytest.fixture(scope="module")
def repairs() -> dict[str, Any]:
    return load(REPAIRS)


def row(rows: list[dict[str, Any]], **key: str) -> dict[str, Any]:
    found = [r for r in rows if all(r[k] == v for k, v in key.items())]
    assert len(found) == 1, key
    return found[0]


def plain_flags(p_source: np.ndarray, y_source: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Flag where 1 - p is within the ceil((n+1)(1-alpha))-th smallest positive score."""
    scores = np.sort(1.0 - p_source[y_source == 1])
    rank = math.ceil((len(scores) + 1) * (1 - ALPHA))
    return (1.0 - p) <= scores[rank - 1]


class TestInfarctionRecomputed:
    @pytest.mark.parametrize("corpus", ["sph", "acs"])
    def test_the_row_is_what_the_scores_give(self, public: dict[str, Any], corpus: str) -> None:
        """The headline Chongqing and Shandong figures, from the raw scores, by hand."""
        with np.load(RESULTS_DIR / "baseline/scores.npz") as source:
            p_s, y_s = source["probs"][:, 1].astype(float), source["labels"]
        with np.load(RESULTS_DIR / f"external/{corpus}.npz") as target:
            p_t, y_t = target["probs"][:, 1].astype(float), target["labels"]
        fs, ft = plain_flags(p_s, y_s, p_s), plain_flags(p_s, y_s, p_t)
        sens = (fs & (y_s == 1)).sum() / (y_s == 1).sum()
        spec = (~fs & (y_s == 0)).sum() / (y_s == 0).sum()
        prev = y_t.mean()
        recomputed = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
        observed = y_t[ft].mean()
        got = row(public["rows"], family="infarction", target=corpus)
        assert got["ppv_recomputed"] == pytest.approx(recomputed, rel=1e-5)
        assert got["ppv_observed"] == pytest.approx(observed, rel=1e-5)
        assert got["n_flagged"] == int(ft.sum())


def test_the_summary_counts_the_rows_it_says(public: dict[str, Any]) -> None:
    """A summary filter that drifted from the rows would print a count no row backs."""
    rows = [r for r in public["rows"] if r["family"] == "rotation"]
    transfers = [r for r in rows if r["summarised"] and not r["in_distribution"]]
    summary = public["summary"]["rotation"]["transfer"]
    assert summary["cells"] == len(transfers)
    gaps = sorted(abs(r["gap"]) for r in transfers)
    assert summary["median_abs_gap_points"] == pytest.approx(float(np.median(gaps)), rel=1e-4)


def test_echonext_rows_agree_with_the_transfer_file(private: dict[str, Any]) -> None:
    """Two code paths, one PPV: the transfer study's ppv block and this study's rows."""
    transfer = load(TRANSFER)
    for arm, measured in transfer["arms"].items():
        for context, labels in measured["ppv"].items():
            for label in (COMPOSITE, LVEF):
                theirs = labels[label]
                mine = row(private["rows"], model=arm, target=context, label=label)
                assert mine["ppv_recomputed"] == pytest.approx(theirs["ppv_from_source"], rel=1e-5)
                assert mine["ppv_observed"] == pytest.approx(theirs["ppv_observed"], rel=1e-5)


def test_echonext_holds_every_arm_context_and_label(private: dict[str, Any]) -> None:
    assert len(private["rows"]) == 4 * 3 * 12


def one(x: float) -> str:
    return f"{x:.1f}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def quoted(public: dict[str, Any], private: dict[str, Any], repairs: dict[str, Any]) -> list[str]:
    """Every figure PPV.md prints, as it prints it, from the result files."""
    every = private["all_families"]
    echo = private["summary"]["echonext"]["by_target"]
    rotation = public["summary"]["rotation"]["transfer"]
    lvef = row(private["rows"], model="resnet", target="outpatient", label=LVEF)
    shd = row(private["rows"], model="resnet", target="outpatient", label=COMPOSITE)
    sph = row(public["rows"], family="infarction", target="sph")
    acs = row(public["rows"], family="infarction", target="acs")
    free = every["transfer"]["label_free_predictors"]
    ten = repairs["summary"]["all"]["0.10"]
    out = [
        f"{every['transfer']['cells']} transfers",
        f"median of {one(100 * every['transfer']['median_abs_gap_points'])} percentage points",
        f"in {pct(every['transfer']['share_outside_observed_interval'])} of the transfers",
        f"On {every['in_distribution']['cells']} control pairs",
        f"missed by {one(100 * every['in_distribution']['median_abs_gap_points'])} points",
        f"In {pct(every['transfer']['share_ratio_off_by_a_quarter_or_more'])} of transfers",
        *(
            f"| {block['cells']} | {one(100 * block['median_abs_gap_points'])} |"
            for block in (echo["outpatient"], echo["emergency"], rotation)
        ),
        f"that {pct(lvef['ppv_recomputed'])} of the outpatients",
        f"The observed figure was {pct(lvef['ppv_observed'])}",
        f"{one(lvef['false_alerts_recomputed'])} false alerts per patient found where there "
        f"were {one(lvef['false_alerts_observed'])}",
        f"it predicted {pct(shd['ppv_recomputed'])} and the observed figure was "
        f"{pct(shd['ppv_observed'])}",
        f"specificity rose from {pct(shd['spec_source'])} to {pct(shd['spec_target'])}",
        f"is {echo['inpatient']['median_likelihood_ratio_healthy_target']:.2f}",
        f"it is {echo['emergency']['median_likelihood_ratio_healthy_target']:.2f}",
        f"outpatients {echo['outpatient']['median_likelihood_ratio_healthy_target']:.2f}",
        f"PPV of {pct(acs['ppv_recomputed'])} for infarction, and {pct(acs['ppv_observed'])}",
        f"from {pct(acs['spec_source'])} to {pct(acs['spec_target'])}",
        f"{100 * sph['ppv_recomputed']:.1f}% against {100 * sph['ppv_observed']:.1f}%",
        f"{sph['false_alerts_recomputed']:.0f} false alerts per infarction found, where there "
        f"were {sph['false_alerts_observed']:.0f}",
        f"median of {one(100 * free['recipe_estimated_prevalence']['median_abs_points'])} points",
        f"missed by {one(100 * free['mean_probability']['median_abs_points'])}",
        f"by {one(100 * free['mean_probability_prior_corrected']['median_abs_points'])}",
        f"over {repairs['summary']['all']['cells']} transfers",
    ]
    for name in ("prior", "recalibrated", "abstention_cleared", "abstention_referred"):
        gain = ten["mean_gain_per_1000"][name]
        out.append(f"| {gain:+.1f} | {ten['wins'][name]} |".replace("-", "−"))
    gains = [
        repairs["summary"]["all"][t]["mean_gain_per_1000"]["recalibrated"]
        for t in ("0.05", "0.10", "0.20")
    ]
    out.append(", ".join(one(g) for g in gains[:2]) + f" and {one(gains[2])} net true positives")
    return out


def test_every_figure_in_ppv_md_is_in_the_results(
    public: dict[str, Any], private: dict[str, Any], repairs: dict[str, Any]
) -> None:
    """The article's numbers are the files' numbers; a rerun that moves one fails here."""
    text = (REPO_ROOT / "PPV.md").read_text()
    missing = [q for q in quoted(public, private, repairs) if q not in text]
    assert missing == []


def test_the_french_article_quotes_the_same_headline(private: dict[str, Any]) -> None:
    every = private["all_families"]
    text = (REPO_ROOT / "PPV.fr.md").read_text()
    median = f"{100 * every['transfer']['median_abs_gap_points']:.1f}".replace(".", ",")
    assert f"{every['transfer']['cells']} transferts" in text
    assert f"de {median} points de pourcentage en médiane" in text


def test_the_readme_quotes_the_headline(private: dict[str, Any]) -> None:
    every = private["all_families"]
    text = (REPO_ROOT / "README.md").read_text()
    transfer, control = every["transfer"], every["in_distribution"]
    assert (
        f"median of {100 * transfer['median_abs_gap_points']:.1f} percentage points over "
        f"{transfer['cells']} transfers" in text
    )
    assert (
        f"{100 * control['median_abs_gap_points']:.1f} points on {control['cells']} controls"
        in text
    )
