"""Hold every rotation number the prose prints to the file it was read from.

Section 2.4 and section 3.6 of ``REPORT.md``, the sentences the rotation adds to
the abstract and the discussion, and questions 11 to 13 of ``QUESTIONS.md`` quote
roughly fifty figures.  Each one is recomputed here from a file under
``results/`` and asserted to appear in the prose, so that regenerating a table
and forgetting to reread the paragraph fails the suite rather than shipping.

The check runs in the direction that catches the error: the expected string is
built from the result file and searched for in the text.  A number nobody
updated therefore fails, and so does a number the prose invented, since the
string built from the file will not be found.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ecs.config import RESULTS_DIR

ROOT = RESULTS_DIR.parent
HEADLINE_ALPHA, HEADLINE_SCORE = "0.1", "lac"
STARVED = {("chapman_ningbo", "LBBB"), ("sph", "IAVB"), ("sph", "LBBB")}


def _read(name: str) -> dict[str, Any]:
    return json.loads((RESULTS_DIR / name).read_text())


def _grid(correction: str) -> list[dict[str, str]]:
    with (RESULTS_DIR / "rotation.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return [
        r
        for r in rows
        if r["alpha"] == HEADLINE_ALPHA
        and r["score"] == HEADLINE_SCORE
        and r["correction"] == correction
    ]


def _uncertainty_rows() -> list[dict[str, str]]:
    with (RESULTS_DIR / "rotation_uncertainty.csv").open() as handle:
        return list(csv.DictReader(handle))


def _pct(value: float) -> str:
    """A coverage as the prose prints it: one decimal, with the per cent sign."""
    return f"{value * 100:.1f}%"


def _mean_pct(rows: list[dict[str, str]], column: str) -> str:
    return _pct(statistics.mean(float(r[column]) for r in rows))


@pytest.fixture(scope="module")
def report() -> str:
    return (ROOT / "REPORT.md").read_text()


@pytest.fixture(scope="module")
def questions() -> str:
    return (ROOT / "QUESTIONS.md").read_text()


@pytest.fixture(scope="module")
def rotation_section(report: str) -> str:
    """Section 2.4 and section 3.6, which is where every figure below is printed."""
    method = report.index("### 2.4 The source rotation")
    method_end = report.index("## 3. Results")
    results = report.index("### 3.6 Five corpora in the calibration role")
    results_end = report.index("## 4. Discussion")
    return report[method:method_end] + report[results:results_end]


def _cases(name: str) -> Iterator[tuple[str, str]]:
    """Named (what, expected string) pairs, every one derived from results/."""
    if name == "split":
        leak = _read("split_leak.json")
        dup = _read("duplicate_groups.json")
        before = {
            c: b["before"]["n_groups_across_two_used_parts"] for c, b in leak["corpora"].items()
        }
        yield "leak before", f"{sum(before.values())} groups"
        yield "leak in cpsc", f"{before['cpsc']} of them in CPSC"
        yield "leak in georgia", f"{before['georgia']} in Georgia"
        yield "leak in chapman", f"{before['chapman_ningbo']} in Chapman-Shaoxing"
        assert leak["n_groups_still_across_two_used_parts"] == 0
        yield "sph repeats", f"Shandong repeats {dup['corpora']['sph']['n_groups']} tracings"
        assert dup["corpora"]["ptbxl"]["n_groups"] == 0
        compared = sum(
            b["against_the_delivery_corpus"]["compared"] for b in dup["corpora"].values()
        )
        yield "digests compared", f"{compared:,} records"
        assert dup["across_corpora"]["n_groups"] == 0
        yield "across corpora", "No group of this screen spans two of the five corpora"

    elif name == "label_map":
        table = _read("label_map.json")
        challenge = [
            c for c, b in table["corpora"].items() if "unexplained_by_double_counting" in b
        ]
        for corpus in challenge:
            block = table["corpora"][corpus]
            assert block["the_published_count_closes"], corpus
            assert not any(block["unexplained_by_double_counting"].values()), corpus
        yield "challenge corpora", f"every one of its {_word(len(challenge))} corpora"
        yield "ambiguities", f"{_word(len(table['ambiguities']))} joins could not be made cleanly"

    elif name == "sizes":
        rot = _read("rotation.json")
        caps = {
            tuple(b["split_sizes"][p] for p in ("train", "cal")) for b in rot["sources"].values()
        }
        yield "caps", f"{max(t for t, _ in caps):,} and {max(c for _, c in caps):,} records"
        yield "draws", f"{rot['settings']['n_draws']} calibration draws"
        settings = _read("rotation_uncertainty.json")["settings"]
        yield "bootstrap", f"{settings['n_bootstrap']:,} times"
        yield "thin", f"fewer than {settings['thin_below']} cases"

    elif name == "cells":
        rows = _grid("mondrian")
        away = [r for r in rows if r["role"] == "away"]
        home = [r for r in rows if r["role"] == "home"]
        pairs = {(r["source"], r["corpus"]) for r in away}
        yield "ordered pairs", f"{_word(len(pairs))} ordered source-target pairs"
        yield "away cells", f"{len(away)} away cells"
        yield "home cells", f"{len(home)} home cells"

    elif name == "coverage":
        for correction, role, column, phrase in (
            ("none", "home", "coverage_mean", "cover {} of all cases"),
            ("none", "home", "coverage_diagnosis_mean", "and {} of the cases carrying"),
            ("none", "away", "coverage_mean", "fall together, to {}"),
            ("none", "away", "coverage_diagnosis_mean", "and {}, which is the shift"),
            ("mondrian", "home", "coverage_diagnosis_mean", "covering {} of the cases carrying"),
            ("mondrian", "home", "coverage_mean", "against {} of all cases"),
            ("mondrian", "away", "coverage_diagnosis_mean", "reads {} within the diagnosis"),
        ):
            rows = [r for r in _grid(correction) if r["role"] == role]
            yield f"{correction} {role} {column}", phrase.format(_mean_pct(rows, column))

    elif name == "bias":
        bias = _read("rotation.json")["bias"]
        for label, text in (
            ("AF", "is {mean} for atrial fibrillation over {pairs} pairs"),
            ("RBBB", "{mean} for right bundle-branch block over {pairs}"),
            ("NSR", "and {mean} for sinus rhythm over the {pairs} pairs"),
        ):
            block = bias[label]["mondrian"]["away_bias"]
            yield (
                f"{label} away bias",
                text.format(mean=_signed(block["mean"]), pairs=_word(block["n_pairs"])),
            )
        sds = {label: bias[label]["mondrian"]["away_bias"]["sd_across_sources"] for label in bias}
        yield "sd range", f"runs from {min(sds.values()):.3f} on first-degree"
        yield "sd worst", f"to {max(sds.values()):.3f} on sinus rhythm"
        # Over every diagnosis, not one of them: an earlier draft searched LBBB
        # alone and named a pair seven points short of the actual worst.
        ranked = sorted(
            (
                (entry["bias"], label, entry["source"], entry["target"], entry)
                for label, block in bias.items()
                for entry in block["mondrian"]["away"]
            ),
            key=lambda row: row[0],
        )
        worst, second = ranked[0], ranked[1]
        assert (worst[1], worst[2], worst[3]) == ("NSR", "cpsc", "ptbxl"), worst[:4]
        assert (second[1], second[2], second[3]) == ("LBBB", "ptbxl", "cpsc"), second[:4]
        yield "worst pair", f"loses {abs(worst[0]) * 100:.1f} points"
        yield "second worst", f"next at {abs(second[0]) * 100:.1f}"
        # The prose leans on both being real transfers rather than abstention.
        for row in (worst, second):
            assert row[4]["n_draws_threshold_infinite"] == 0, row[:4]
        homes = {label: bias[label]["mondrian"]["home_bias"]["mean"] for label in bias}
        yield "home nsr", f"within {abs(homes['NSR']) * 100:.1f} points of the target"
        others = [v for k, v in homes.items() if k != "NSR"]
        yield "home others", f"by {min(others) * 100:.1f} to {max(others) * 100:.1f} points"

    elif name == "starved":
        rows = [r for r in _grid("mondrian") if r["role"] == "away"]
        infinite = {
            (r["source"], r["label"]): int(r["threshold_diagnosis_n_infinite"])
            for r in rows
            if int(r["threshold_diagnosis_n_infinite"]) > 0
        }
        assert set(infinite) == STARVED, sorted(infinite)
        yield "sph lbbb", f"infinite in {infinite[('sph', 'LBBB')]} of 200 draws"
        yield "sph iavb", f"atrioventricular block in {infinite[('sph', 'IAVB')]}"
        yield "chapman lbbb", f"left bundle-branch block in {infinite[('chapman_ningbo', 'LBBB')]}"
        sick = [r for r in rows if (r["source"], r["label"]) in STARVED]
        well = [r for r in rows if (r["source"], r["label"]) not in STARVED]
        sizes = [float(r["mean_set_size_mean"]) for r in sick]
        yield "starved sizes", f"average {min(sizes):.2f} to {max(sizes):.2f} labels"
        clean = statistics.mean(float(r["mean_set_size_mean"]) for r in well)
        yield "clean size", f"against {clean:.3f} on the other {_word(len(well))} away pairs"
        covers = [float(r["coverage_diagnosis_mean"]) for r in sick]
        yield "starved coverage", f"reads {min(covers):.3f} to {max(covers):.3f}"
        bias = _read("rotation.json")["bias"]
        for label, text in (
            ("LBBB", "from {a} over {n} pairs to {b} over the {m}"),
            ("IAVB", "block from {a} to {b}"),
        ):
            block = bias[label]["mondrian"]
            finite = block["away_bias_where_the_threshold_was_finite"]
            yield (
                f"{label} finite",
                text.format(
                    a=_signed(block["away_bias"]["mean"]),
                    b=_signed(finite["mean"]),
                    n=_word(block["away_bias"]["n_pairs"]),
                    m=_word(finite["n_pairs"]),
                ),
            )

    elif name == "uncertainty":
        reading = _read("rotation_uncertainty.json")["reading"]
        widths = reading["bootstrap_width_against_draw_spread"]
        yield "rows", f"{reading['n_rows']:,} rows"
        yield "pairs compared", f"{widths['n_rows_compared']} source-target-diagnosis pairs"
        yield "bootstrap width", f"cohort is {widths['median_bootstrap_width']:.3f}"
        yield "draw spread", f"a standard deviation of {widths['median_draw_spread']:.3f}"
        # The two are a sigma and a 95% width. Quoting them side by side without
        # the conversion is what made an earlier draft read the comparison
        # backwards, so the converted figure is asserted with the raw pair.
        yield (
            "draw spread converted",
            f"which is {widths['median_draw_spread_as_a_95_percent_width']:.3f} as a 95% width",
        )
        wider = widths["n_rows_where_the_draw_spread_is_the_wider"]
        assert wider > widths["n_rows_compared"] / 2, (
            "the prose says the calibration draw is the wider of the two on most pairs"
        )
        yield (
            "which is wider",
            f"wider of the two on {wider} of the {widths['n_rows_compared']} pairs",
        )
        yield "thin rows", f"{reading['n_rows_too_thin_to_read']} of the file's"
        # Only the class-conditional column compares like with like: Chow's rule
        # is a per-class quantile, so the pooled column has no counterpart in it.
        chow = reading["conformal_minus_chow"]["by_correction"]["mondrian"]
        yield "chow pairs", f"over its {chow['n_rows']} away pairs"
        yield "chow median", f"median of {chow['median']:.3f} in coverage"
        yield "chow max", f"by as much as {chow['max']:.3f}"
        sexes = reading["between_the_sexes"]
        yield (
            "sex median",
            f"over {sexes['n_pairs_compared']} pairs is "
            f"{sexes['median_absolute_gap'] * 100:.1f} points",
        )
        yield "sex widest", f"widest is {sexes['widest_gap'] * 100:.1f}"

    elif name == "chow_outliers":
        rows = [
            r
            for r in _uncertainty_rows()
            if r["conformal_minus_chow"] and abs(float(r["conformal_minus_chow"])) > 0.35
        ]
        assert rows, "the prose claims departures above 0.35 exist"
        assert {(r["source"], r["label"]) for r in rows} <= STARVED
        yield "chow outliers", "Every departure above 0.35 sits on the three pairs"

    elif name == "ages":
        rows = [
            r
            for r in _uncertainty_rows()
            if r["role"] == "away"
            and r["correction"] == "mondrian"
            and r["subgroup_kind"] == "age"
            and r["thin"] == "False"
        ]
        bands = ["<50", "50-64", "65-74", ">=75"]
        for label, text in (
            ("NSR", "from {0} below 50 to {1}, {2} and {3} in the oldest band"),
            ("RBBB", "bands, from {0} to {1}, {2} and {3}"),
        ):
            means = [
                _pct(
                    statistics.mean(
                        float(r["coverage"])
                        for r in rows
                        if r["label"] == label and r["subgroup"] == band
                    )
                )
                for band in bands
            ]
            yield f"{label} ages", text.format(*means)

    elif name == "target_scale":
        ladder: list[dict[str, Any]] = [
            r
            for r in _read("target_scale.json")["rows"]
            if r["alpha"] == 0.10 and r["score"] == "lac" and r["correction"] == "none"
        ]
        by_family: dict[str, dict[int, float]] = {}
        for rung in ladder:
            by_family.setdefault(rung["family"], {})[rung["n_target_records"]] = rung[
                "coverage_by_class"
            ]["1"]["mean"]
        recal, pooled = by_family["recalibrated"], by_family["pooled"]
        assert recal[0] == pooled[0], "rung zero is the same frozen threshold on both lines"
        yield "rung zero", f"reads {_pct(recal[0])} with no target records"
        yield "recalibrated 100", f"takes it to {_pct(recal[100])}"
        yield "recalibrated tail", f"leave it at {_pct(recal[500])} and {_pct(recal[2000])}"
        yield (
            "pooled",
            (f"reaches {_pct(pooled[100])}, {_pct(pooled[500])} and {_pct(pooled[2000])}"),
        )

    elif name == "arms":
        arms = _read("arms.json")
        by_auroc = sorted(arms["arms"], key=lambda a: arms["discrimination"][a]["ptbxl"]["auroc"])
        worst, best = by_auroc[0], by_auroc[-1]
        assert worst == "random_init", by_auroc
        for arm, text in (
            (worst, "an AUROC of {auroc:.3f} (95% CI {lo:.3f} to {hi:.3f}) for a randomly"),
            (best, "to {auroc:.3f} ({lo:.3f} to {hi:.3f}) for the strongest"),
        ):
            block = arms["discrimination"][arm]["ptbxl"]
            lo, hi = block["auroc_ci95"]
            yield f"{arm} auroc", text.format(auroc=block["auroc"], lo=lo, hi=hi)
        seen = [a for a, b in arms["arms"].items() if b["saw"]]
        yield "contaminated", f"{_word(len(seen))} of the five were pretrained on corpora used here"

        # The headline rung of the arms file runs at a different target from the
        # rest of the report, which is why the paragraph says so.
        head = arms["headline"]
        cover: dict[str, dict[str, Any]] = {
            arm: next(
                rung["coverage"]
                for rung in arm_rows
                if (rung["alpha"], rung["score"], rung["correction"])
                == (head["alpha"], head["score"], head["correction"])
            )
            for arm, arm_rows in arms["coverage"].items()
        }
        target = 1.0 - float(head["alpha"])
        yield "arms target", f"at an {target * 100:.0f}% target rather than the 90%"
        at_home = [float(c["ptbxl"]["mean"]) for c in cover.values()]
        yield "arms at home", f"between {_pct(min(at_home))} and {_pct(max(at_home))}"
        at_acs = [float(c["acs"]["mean"]) for c in cover.values()]
        assert max(at_acs) < target, "the prose says no arm reaches the target at Chongqing"
        yield "arms at chongqing", f"they read between {_pct(min(at_acs))} and {_pct(max(at_acs))}"
        over = [c for c in cover.values() if float(c["sph"]["mean"]) > target]
        yield "arms at shandong", f"{_word(len(over))} of the five over-cover at Shandong"
        yield "best arm transfers", f"reads {_pct(cover[best]['acs']['mean'])} at Chongqing"
        weaker = max((a for a in cover if a != best), key=lambda a: cover[a]["acs"]["mean"])
        assert (
            arms["discrimination"][weaker]["ptbxl"]["auroc"]
            < arms["discrimination"][best]["ptbxl"]["auroc"]
        ), "the prose calls the better transferrer the weaker arm"
        yield "weaker arm transfers", f"a weaker arm reads {_pct(cover[weaker]['acs']['mean'])}"

    else:  # pragma: no cover - the parametrisation below is closed
        raise AssertionError(name)


def _signed(value: float) -> str:
    """A bias as the prose prints it: three decimals with a true minus sign."""
    return f"{value:+.3f}".replace("-", "−")


def _word(n: int) -> str:
    """The prose spells small counts out; the tests have to as well."""
    words = {
        0: "no",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        8: "eight",
        12: "twelve",
        20: "twenty",
        80: "eighty",
    }
    return words.get(n, f"{n:,}")


GROUPS = [
    "split",
    "label_map",
    "sizes",
    "cells",
    "coverage",
    "bias",
    "starved",
    "uncertainty",
    "chow_outliers",
    "ages",
    "target_scale",
    "arms",
]


@pytest.mark.parametrize("group", GROUPS)
def test_the_rotation_section_prints_what_the_results_files_hold(
    group: str, rotation_section: str
) -> None:
    """Every figure in sections 2.4 and 3.6, rebuilt from results/ and searched for."""
    for what, expected in _cases(group):
        assert expected.lower() in rotation_section.lower(), (
            f"{group}/{what}: {expected!r} not in the section"
        )


def test_the_abstract_carries_the_rotation_figures_it_claims(report: str) -> None:
    """The abstract quotes five of the section's numbers and must quote them alike."""
    abstract = report[report.index("## Abstract") : report.index("## 1. Introduction")]
    home_none = [r for r in _grid("none") if r["role"] == "home"]
    home_mondrian = [r for r in _grid("mondrian") if r["role"] == "home"]
    away_mondrian = [r for r in _grid("mondrian") if r["role"] == "away"]
    scale = {
        (r["family"], r["n_target_records"]): r["coverage_by_class"]["1"]["mean"]
        for r in _read("target_scale.json")["rows"]
        if r["alpha"] == 0.10 and r["score"] == "lac" and r["correction"] == "none"
    }
    for what, expected in (
        ("marginal at home", f"cover {_mean_pct(home_none, 'coverage_mean')} of all cases"),
        (
            "diagnosis at home",
            f"and {_mean_pct(home_none, 'coverage_diagnosis_mean')} of the cases",
        ),
        ("repaired at home", f"at {_mean_pct(home_mondrian, 'coverage_diagnosis_mean')}, and not"),
        (
            "not on transfer",
            f"at {_mean_pct(away_mondrian, 'coverage_diagnosis_mean')}, with a spread",
        ),
        ("ladder foot", f"from {_pct(scale[('recalibrated', 0)])}"),
        ("ladder at 100", f"to {_pct(scale[('recalibrated', 100)])}"),
    ):
        assert expected in abstract, f"{what}: {expected!r} not in the abstract"


def test_the_questions_quote_the_same_files(questions: str) -> None:
    """Questions 11 to 13 restate figures from the section; they must not drift from it."""
    tail = questions[questions.index("## 11.") :]
    leak = _read("split_leak.json")
    before = {c: b["before"]["n_groups_across_two_used_parts"] for c, b in leak["corpora"].items()}
    rows = [r for r in _grid("mondrian") if r["role"] == "away"]
    sick = [r for r in rows if (r["source"], r["label"]) in STARVED]
    bias = _read("rotation.json")["bias"]["LBBB"]["mondrian"]
    ambiguities = _read("label_map.json")["ambiguities"]
    covers = [float(r["coverage_diagnosis_mean"]) for r in sick]
    sizes = [float(r["mean_set_size_mean"]) for r in sick]
    for what, expected in (
        ("leak total", f"{sum(before.values())} groups of identical tracings"),
        ("leak by corpus", f"{before['cpsc']} of them in CPSC, {before['georgia']} in Georgia"),
        ("still leaking", "which is zero"),
        ("bought coverage", f"reads between {_pct(min(covers))} and {_pct(max(covers))}"),
        ("set sizes", f"average {min(sizes):.2f} to {max(sizes):.2f} labels"),
        (
            "sign change",
            f"reads {_signed(bias['away_bias']['mean'])} over twenty pairs and "
            f"{_signed(bias['away_bias_where_the_threshold_was_finite']['mean'])}",
        ),
        ("ambiguities", f"{_word(len(ambiguities))} joins could not be made cleanly"),
    ):
        assert expected.lower() in tail.lower(), f"{what}: {expected!r} not in questions 11 to 13"
    assert leak["n_groups_still_across_two_used_parts"] == 0


def test_no_rotation_figure_is_named_without_a_file_behind_it(report: str) -> None:
    """C-20 over the section this branch adds: both its figures are redrawn by a script."""
    shown = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report)
    rotation = [s for s in shown if "fig7" in s or "fig8" in s]
    assert rotation == [
        "results/figures/fig7_rotation.png",
        "results/figures/fig8_target_scale.png",
    ]
    script = (ROOT / "scripts/figures.py").read_text()
    for relative in rotation:
        assert (ROOT / relative).exists(), relative
        assert Path(relative).name in script, relative


def test_no_committed_figure_is_one_no_script_draws() -> None:
    """A renumbering left fig5_rotation.png behind, byte-identical to fig7.

    Nothing referenced it and nothing failed, so it would have shipped. Every
    file in the figure directory has to be a name ``scripts/figures.py`` writes.
    """
    script = (ROOT / "scripts/figures.py").read_text()
    orphans = [
        p.name for p in sorted((RESULTS_DIR / "figures").glob("*.png")) if p.name not in script
    ]
    assert orphans == [], orphans
