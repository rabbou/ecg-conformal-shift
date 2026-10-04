"""Hold every rotation number the prose prints to the file it was read from.

Appendices B and F of ``REPORT.md`` quote some forty figures of the rotation and
the encoder comparison.  Each one is
recomputed here from a file under ``results/`` and asserted to appear in the
prose, so that regenerating a table and forgetting to reread the paragraph
fails the suite rather than shipping.

The check runs in the direction that catches the error: the expected string is
built from the result file and searched for in the text.  A number nobody
updated therefore fails, and so does a number the prose invented, since the
string built from the file will not be found.
"""

from __future__ import annotations

import ast
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
# The grid holds three roles; figure_7_rotation panels two of them, so anything
# the figure's caption counts has to be counted over these and not over the grid.
DRAWN_ROLES = ("home", "away")


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


def starved(correction: str = "mondrian") -> set[tuple[str, str]]:
    """The source-diagnosis pairs whose threshold ran to infinity in some draw.

    Read off the grid over both roles rather than written down here: a
    hard-coded set goes stale the moment a table is rebuilt, and one derived
    from the away rows alone would miss a pair that starves only at home.
    """
    return {
        (r["source"], r["label"])
        for r in _grid(correction)
        if int(r["threshold_diagnosis_n_infinite"]) > 0
    }


def _finite(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The cells whose class-conditional threshold was finite in every draw.

    A pair that covers by admitting both labels satisfies the definition of
    coverage without answering, so every class-conditional mean is quoted with
    and without them.
    """
    excluded = starved()
    return [r for r in rows if (r["source"], r["label"]) not in excluded]


@pytest.fixture(scope="module")
def report() -> str:
    return (ROOT / "REPORT.md").read_text()


@pytest.fixture(scope="module")
def rotation_section(report: str) -> str:
    """Appendices B and F, which is where every figure below is printed."""
    rotation = report.index("## Appendix B.")
    rotation_end = report.index("## Appendix C.")
    arms = report.index("## Appendix F.")
    arms_end = report.index("## Appendix G.")
    return report[rotation:rotation_end] + report[arms:arms_end]


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
        # The pooled figures are quoted once because a pooled threshold is a
        # quantile of the whole calibration sample: the prose says it is finite
        # everywhere, which is a claim about the grid and is checked here.
        pooled_cells = _grid("none")
        assert all(int(r["threshold_diagnosis_n_infinite"]) == 0 for r in pooled_cells), (
            "the prose says no pooled coverage is bought by abstaining"
        )
        # The class-conditional figures are quoted twice, and the second reading
        # is the one the paragraph rests on, so both are rebuilt from the grid.
        kept: dict[str, str] = {}
        for role in ("home", "away"):
            rows = [r for r in _grid("mondrian") if r["role"] == role]
            assert len(_finite(rows)) < len(rows), role
            kept[role] = _mean_pct(_finite(rows), "coverage_diagnosis_mean")
        yield "mondrian finite pair", f"read {kept['home']} and {kept['away']}"

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

    elif name == "abstention_by_correction":
        # The caption counts rings, so it has to count over the roles the figure
        # draws. Counting the whole grid put one pair too many in it: a pair can
        # starve in the calibration-holdout role, which no panel shows.
        drawn = {
            correction: len(
                {
                    (r["source"], r["label"])
                    for r in _grid(correction)
                    if r["role"] in DRAWN_ROLES and int(r["threshold_diagnosis_n_infinite"]) > 0
                }
            )
            for correction in ("none", "mondrian", "weighted")
        }
        assert drawn["none"] == 0, drawn
        yield (
            "caption mondrian",
            f"{_word(drawn['mondrian'])} source-diagnosis pairs under the per-class rule",
        )
        yield "caption weighted", f"and {_word(drawn['weighted'])} under label-shift weighting"

    elif name == "starved":
        rows = [r for r in _grid("mondrian") if r["role"] == "away"]
        infinite = {
            (r["source"], r["label"]): int(r["threshold_diagnosis_n_infinite"])
            for r in rows
            if int(r["threshold_diagnosis_n_infinite"]) > 0
        }
        assert set(infinite) == starved(), sorted(infinite)
        yield "sph lbbb", f"infinite in {infinite[('sph', 'LBBB')]} of 200 draws"
        yield "sph iavb", f"atrioventricular block in {infinite[('sph', 'IAVB')]}"
        yield "chapman lbbb", f"left bundle-branch block in {infinite[('chapman_ningbo', 'LBBB')]}"

    elif name == "arms":
        arms = _read("arms.json")
        by_auroc = sorted(arms["arms"], key=lambda a: arms["discrimination"][a]["ptbxl"]["auroc"])
        worst, best = by_auroc[0], by_auroc[-1]
        assert worst == "random_init", by_auroc
        for arm, text in (
            (worst, "an AUROC of {auroc:.3f} (95% CI {lo:.3f} to {hi:.3f})"),
            (best, "to {auroc:.3f} ({lo:.3f} to {hi:.3f})"),
        ):
            block = arms["discrimination"][arm]["ptbxl"]
            lo, hi = block["auroc_ci95"]
            yield f"{arm} auroc", text.format(auroc=block["auroc"], lo=lo, hi=hi)
        # The prose says both ends of the AUROC range are held-out readings,
        # which is a claim about which arms sit at those ends, not about the
        # numbers: naming the range without it would credit a contaminated arm.
        assert not arms["arms"][worst]["saw"] and not arms["arms"][best]["saw"], (
            "the prose says both ends of the range are arms with no corpus of this study"
        )
        seen = [a for a, b in arms["arms"].items() if b["saw"]]
        yield (
            "contaminated",
            f"{_word(len(seen))} of the five arms had a corpus used here in their pre-training",
        )

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
        # Chongqing is held out for every arm, which is what lets the paragraph
        # read those figures across arms while refusing to order the AUROCs.
        assert all("acs" not in block["saw"] for block in arms["arms"].values())
        yield "chongqing held out", "no arm having been pretrained there"

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
        14: "fourteen",
        15: "fifteen",
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
    "abstention_by_correction",
    "starved",
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


def test_the_figure_reads_the_abstention_field_in_both_role_loops() -> None:
    """The caption counts rings in both roles, so both loops must consult the field.

    Searched inside the function's own syntax tree rather than anywhere in the
    file: a bare ``"n_draws_threshold_infinite" in script`` was satisfied while
    only the away loop read it, which is how the caption came to count a ring
    the figure never drew.
    """
    tree = ast.parse((ROOT / "scripts/figures.py").read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "figure_7_rotation"
    )
    reads = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and node.value == "n_draws_threshold_infinite"
    ]
    assert len(reads) >= 2, f"only {len(reads)} role loop(s) consult the field"


def drawn_names() -> set[str]:
    """The figure file names ``scripts/figures.py`` actually writes.

    Parsed rather than searched for as a substring: ``"fig7_rotation.png" in
    script`` also passes on a file the script only mentions in a comment, and it
    would pass a name that is a prefix of a real one.
    A name written ``f"fig1_thresholds{suffix}.png"`` is one file per language,
    so it expands to every suffix ``figure_text.SUFFIX`` names.
    """
    from figure_text import SUFFIX

    script = (ROOT / "scripts/figures.py").read_text()
    names = set()
    for stem, suffixed in re.findall(r'out / f?"([A-Za-z0-9_.-]+?)(\{suffix\})?\.png"', script):
        names |= {f"{stem}{s}.png" for s in SUFFIX.values()} if suffixed else {f"{stem}.png"}
    return names


def test_no_rotation_figure_is_named_without_a_file_behind_it(report: str) -> None:
    """Both rotation figures are redrawn by a script."""
    shown = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report)
    rotation = [s for s in shown if s.endswith(("fig7_rotation.png", "fig8_target_scale.png"))]
    assert rotation == [
        "results/figures/fig7_rotation.png",
        "results/figures/fig8_target_scale.png",
    ]
    names = drawn_names()
    assert names, "no figure names parsed out of scripts/figures.py"
    for relative in rotation:
        assert (ROOT / relative).exists(), relative
        assert Path(relative).name in names, relative


def test_no_committed_figure_is_one_no_script_draws() -> None:
    """A renumbering left fig5_rotation.png behind, byte-identical to fig7.

    Nothing referenced it and nothing failed, so it would have shipped. Every
    file in the figure directory has to be a name ``scripts/figures.py`` writes.
    The glob is case-blind, since a ``.PNG`` would slip a case-sensitive one.
    """
    names = drawn_names()
    orphans = [
        p.name
        for p in sorted((RESULTS_DIR / "figures").iterdir())
        if p.is_file() and p.name not in names
    ]
    assert orphans == [], orphans
