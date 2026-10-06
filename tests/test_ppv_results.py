"""The committed PPV results, recomputed where the scores are committed, and the prose quoting them.

The infarction rows are recomputed here from the score files without importing
``ecs``: the conformal rank, the counts and Bayes' rule are written out again
from their definitions, so agreement is evidence rather than a shared bug.  The
EchoNext rows cannot be recomputed without restricted data; they are held
against ``echonext_transfer.json``, which reached the same two PPVs through
other code.  The article prints these figures through ``tests/paper_values.py``,
and ``tests/test_paper_numbers.py`` holds the text to them.
"""

from __future__ import annotations

import json
import math
import re
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


def test_the_cluster_intervals_hold_the_summary_they_bracket(private: dict[str, Any]) -> None:
    """The interval file's estimates are the PPV files' own summary, and each interval
    brackets its estimate."""
    blocks = load(RESULTS_DIR / "ppv_intervals.json")["across_transfers"]
    every = private["all_families"]
    for name, summary in (("transfer", every["transfer"]), ("control", every["in_distribution"])):
        block = blocks[name]
        assert block["cells"] == summary["cells"]
        for key in (
            "median_abs_gap_points",
            "share_within_two_points",
            "share_outside_observed_interval",
            "share_recipe_too_high",
            "share_ratio_off_by_a_quarter_or_more",
        ):
            assert block[key]["estimate"] == pytest.approx(summary[key], abs=1e-5), (name, key)
            assert block[key]["low"] <= block[key]["estimate"] <= block[key]["high"], (name, key)


def test_ppv_md_points_to_the_article() -> None:
    """One text says each thing: PPV.md and its French page point to REPORT.md."""
    for name in ("PPV.md", "PPV.fr.md"):
        text = (REPO_ROOT / name).read_text()
        assert "](REPORT.md)" in text and "](SUPPLEMENT.md)" in text
        assert not re.search(r"\d%|\d points", text), f"{name} prints a result of its own"
        assert len(text.split()) < 120, name
