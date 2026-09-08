"""Coverage inside sex and age, under the same three schemes as the outcome table.

The class conditional guarantee is conditional on the label and says nothing
about a subgroup inside it.  A site that adopted per label reporting on this
report's argument would still not know what the tool delivers to its oldest
patients, so the report has to measure it rather than assume it is flat.

Coverage here is the same event as in `shift_table.py`: the set the scheme
returns holds the true label.  Under the plain threshold every set is a single
label, so coverage within the MI cases is that scheme's sensitivity; the three
schemes stay comparable because the event is defined the same way for all of
them.

Two quantities are reported for every cell, and they are not the same thing:

  mean, sd   the average over 200 draws that halve fold 10 by patient, and the
             spread across those draws.  Identical protocol to `outcomes.py`,
             which is what makes the numbers comparable with table 1.  The
             draws re-partition one fixed cohort, so the sd is the spread of
             the calibration and not the sampling error of the subgroup.
  ci95       a percentile interval from `BOOTSTRAP_DRAWS` replicates, each of
             which redraws the patients of fold 10 with replacement, halves the
             redrawn patients, refits every threshold on one half and evaluates
             on the other.  Because the cohort itself is resampled, this
             interval carries the sampling error of the subgroup as well as the
             calibration draw, which is what a difference between subgroups has
             to be read against.

Differences between two levels of a subgroup are computed inside each replicate
before the percentile is taken, so the interval is on the paired difference and
not on the gap between two separately quoted intervals.

Usage: .venv/bin/python scripts/subgroups.py [--draws 200] [--bootstrap 2000]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ecs.config import PTBXL_DIR, RESULTS_DIR
from ecs.conformal import (
    conformal_quantile,
    lac_scores,
    lac_scores_all,
    mondrian_quantiles,
    predict_sets,
    predict_sets_per_class,
)

ALPHA = 0.10
PLAIN_SENSITIVITY = 1.0 - ALPHA
SCHEMES = ("plain", "pooled", "perlabel")
DRAWS = 200
BOOTSTRAP_DRAWS = 2000

# PTB-XL stores sex as 0 and 1; Wagner et al. 2020 give 0 as male.
SEX_LEVELS = {"male": 0, "female": 1}
# Bands wide enough that the rarer label is present in every one of them on
# fold 10.  PTB-XL stores every age above 89 as 300 for privacy, which the
# v1.0.2 changelog states and which nothing between 89 and 300 contradicts, so
# 300 belongs in the oldest band rather than in a band of its own.
AGE_EDGES = ((0, 50), (50, 65), (65, 75), (75, 301))
AGE_LEVELS = tuple(f"{lo}-{hi - 1}" if hi <= 300 else f"{lo}+" for lo, hi in AGE_EDGES)
# The two contrasts the report quotes: the widest age gap and the sex gap.
CONTRASTS = (("age", AGE_LEVELS[0], AGE_LEVELS[-1]), ("sex", "male", "female"))


def read_metadata(ids: NDArray[Any], root: Path) -> pd.DataFrame:
    """Patient, sex and age band for each scored record, in the order of ``ids``."""
    database = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    rows = database.loc[[int(i) for i in ids]]
    band = pd.Series("", index=rows.index, dtype=object)
    for (low, high), name in zip(AGE_EDGES, AGE_LEVELS, strict=True):
        band[(rows["age"] >= low) & (rows["age"] < high)] = name
    if (band == "").any():
        raise ValueError(f"{int((band == '').sum())} records fall outside every age band")
    return pd.DataFrame(
        {
            "patient": rows["patient_id"].astype(str).to_numpy(),
            "sex": rows["sex"].to_numpy(),
            "age_band": band.to_numpy(),
        }
    )


def _covered(sets: NDArray[np.bool_], labels: NDArray[np.int_]) -> NDArray[np.bool_]:
    """Whether each returned set holds the true label."""
    return sets[np.arange(labels.size), labels]


def _fit(
    probs: NDArray[np.float64], labels: NDArray[np.int_]
) -> tuple[float, float, NDArray[np.float64]]:
    scores = lac_scores(probs, labels)
    return (
        float(np.quantile(probs[:, 1][labels == 1], 1.0 - PLAIN_SENSITIVITY)),
        conformal_quantile(scores, ALPHA),
        mondrian_quantiles(scores, labels, ALPHA, 2),
    )


def _sets(
    probs: NDArray[np.float64], plain_q: float, pooled_q: float, perlabel_q: NDArray[np.float64]
) -> dict[str, NDArray[np.bool_]]:
    positive = probs[:, 1] >= plain_q
    all_scores = lac_scores_all(probs)
    return {
        "plain": np.column_stack([~positive, positive]),
        "pooled": predict_sets(all_scores, pooled_q),
        "perlabel": predict_sets_per_class(all_scores, perlabel_q),
    }


def _cells(
    covered: NDArray[np.bool_],
    labels: NDArray[np.int_],
    meta: pd.DataFrame,
) -> dict[str, float]:
    """Coverage in every subgroup cell, and in the subgroup as a whole."""
    out: dict[str, float] = {}
    groups = {"sex": [(k, meta["sex"].to_numpy() == v) for k, v in SEX_LEVELS.items()]}
    groups["age"] = [(b, meta["age_band"].to_numpy() == b) for b in AGE_LEVELS]
    for name, levels in groups.items():
        for level, mask in levels:
            for klass, label in ((1, "mi"), (0, "non_mi"), (None, "all")):
                cell = mask if klass is None else mask & (labels == klass)
                out[f"{name}:{level}:{label}"] = (
                    float(covered[cell].mean()) if cell.any() else float("nan")
                )
    return out


def _halve(
    patients: NDArray[Any], unique: NDArray[Any], rng: np.random.Generator
) -> NDArray[np.bool_]:
    """One calibration half, drawn by patient.

    ``unique`` is shuffled in place and handed back in across draws, which is
    what ``outcomes.py`` does. Re-sorting it every draw would give a different
    sequence of halves and the numbers here would no longer be comparable with
    table 1 draw for draw.
    """
    rng.shuffle(unique)
    held = set(unique[: len(unique) // 2].tolist())
    return np.array([p in held for p in patients])


def _one(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    meta: pd.DataFrame,
    is_calibration: NDArray[np.bool_],
) -> dict[str, dict[str, float]]:
    plain_q, pooled_q, perlabel_q = _fit(probs[is_calibration], labels[is_calibration])
    test = ~is_calibration
    built = _sets(probs[test], plain_q, pooled_q, perlabel_q)
    return {
        scheme: _cells(_covered(sets, labels[test]), labels[test], meta[test])
        for scheme, sets in built.items()
    }


def collect(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    meta: pd.DataFrame,
    draws: int,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    patients = meta["patient"].to_numpy()

    rng = np.random.default_rng(seed)
    unique = np.unique(patients)
    protocol: dict[str, dict[str, list[float]]] = {s: {} for s in SCHEMES}
    for _ in range(draws):
        for scheme, cells in _one(probs, labels, meta, _halve(patients, unique, rng)).items():
            for key, value in cells.items():
                protocol[scheme].setdefault(key, []).append(value)

    # Each replicate resamples the patients themselves, so a subgroup's own
    # sampling error is inside the interval rather than held fixed.
    rng = np.random.default_rng(seed)
    unique = np.unique(patients)
    index_of = {p: np.flatnonzero(patients == p) for p in unique}
    resampled: dict[str, dict[str, list[float]]] = {s: {} for s in SCHEMES}
    for _ in range(bootstrap):
        picked = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([index_of[p] for p in picked])
        # A patient drawn twice must not straddle the halving, so the replicate's
        # patient identity is its draw position and not its PTB-XL id.
        replica = pd.DataFrame(
            {
                "patient": np.concatenate(
                    [np.full(index_of[p].size, f"{i}") for i, p in enumerate(picked)]
                ),
                "sex": meta["sex"].to_numpy()[rows],
                "age_band": meta["age_band"].to_numpy()[rows],
            }
        )
        replica_patients = replica["patient"].to_numpy()
        held = _halve(replica_patients, np.unique(replica_patients), rng)
        for scheme, cells in _one(probs[rows], labels[rows], replica, held).items():
            for key, value in cells.items():
                resampled[scheme].setdefault(key, []).append(value)

    def summarise(scheme: str, key: str) -> dict[str, Any]:
        drawn = np.asarray(protocol[scheme][key], dtype=np.float64)
        boot = np.asarray(resampled[scheme][key], dtype=np.float64)
        boot = boot[~np.isnan(boot)]
        return {
            "mean": round(float(np.nanmean(drawn)), 4),
            "sd": round(float(np.nanstd(drawn, ddof=1)), 4),
            "ci95": [round(float(np.percentile(boot, 2.5)), 4)]
            + [round(float(np.percentile(boot, 97.5)), 4)],
            "n_draws": int(drawn.size),
            "n_bootstrap": int(boot.size),
        }

    coverage = {
        scheme: {key: summarise(scheme, key) for key in sorted(protocol[scheme])}
        for scheme in SCHEMES
    }

    differences: dict[str, Any] = {}
    for group, first, second in CONTRASTS:
        for label in ("mi", "non_mi", "all"):
            a, b = f"{group}:{first}:{label}", f"{group}:{second}:{label}"
            for scheme in SCHEMES:
                gap = np.asarray(resampled[scheme][b], dtype=np.float64) - np.asarray(
                    resampled[scheme][a], dtype=np.float64
                )
                gap = gap[~np.isnan(gap)]
                point = protocol[scheme][b]
                differences[f"{scheme}:{group}:{second}-{first}:{label}"] = {
                    "mean": round(float(np.nanmean(point) - np.nanmean(protocol[scheme][a])), 4),
                    "ci95": [round(float(np.percentile(gap, 2.5)), 4)]
                    + [round(float(np.percentile(gap, 97.5)), 4)],
                    "n_bootstrap": int(gap.size),
                }
    return {"coverage": coverage, "differences": differences}


def counts(labels: NDArray[np.int_], meta: pd.DataFrame) -> dict[str, dict[str, int]]:
    """How many records of each label sit in each cell, over the whole fold."""
    out: dict[str, dict[str, int]] = {}
    for name, levels in (
        ("sex", [(k, meta["sex"].to_numpy() == v) for k, v in SEX_LEVELS.items()]),
        ("age", [(b, meta["age_band"].to_numpy() == b) for b in AGE_LEVELS]),
    ):
        for level, mask in levels:
            out[f"{name}:{level}"] = {
                "n": int(mask.sum()),
                "n_mi": int((mask & (labels == 1)).sum()),
            }
    return out


def build(draws: int, bootstrap: int, seed: int) -> dict[str, Any]:
    started = time.time()
    bundle = np.load(RESULTS_DIR / "baseline/scores.npz")
    probs = bundle["probs"]
    labels = bundle["labels"]
    meta = read_metadata(bundle["ids"], PTBXL_DIR)
    body = collect(probs, labels, meta, draws, bootstrap, seed)
    return {
        "question": (
            "Inside each label, is coverage flat across sex and age, or does the "
            "scheme this report recommends deliver a different coverage to the "
            "oldest patients and to women than to the label as a whole."
        ),
        "protocol": (
            "Same halving of PTB-XL fold 10 by patient as outcomes.py, same seed, "
            "same three schemes. Coverage is the set holding the true label. "
            "Intervals are percentile intervals over bootstrap replicates that "
            "resample the patients of fold 10 with replacement before halving, so "
            "they carry the subgroup's sampling error and not only the calibration "
            "draw. Differences are taken inside each replicate."
        ),
        "sex_encoding": "PTB-XL sex: 0 male, 1 female (Wagner et al. 2020).",
        "age_note": (
            "PTB-XL stores every age above 89 as 300 for privacy (v1.0.2 changelog); "
            "those 293 records of the release, 34 of fold 10, sit in the oldest band."
        ),
        "alpha": ALPHA,
        "n_draws": draws,
        "n_bootstrap": bootstrap,
        "seed": seed,
        "counts": counts(labels, meta),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "seconds": round(time.time() - started, 1),
        **body,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "subgroups.json")
    args = parser.parse_args()
    table = build(args.draws, args.bootstrap, args.seed)
    args.out.write_text(json.dumps(table, indent=2) + "\n")
    print(f"wrote {args.out}")
    for scheme in SCHEMES:
        for key, cell in table["coverage"][scheme].items():
            if key.endswith(":mi"):
                low, high = cell["ci95"]
                print(f"  {scheme:<9s} {key:<22s} {cell['mean']:.4f}  [{low:.4f}, {high:.4f}]")
    for key, gap in table["differences"].items():
        if key.endswith(":mi"):
            low, high = gap["ci95"]
            print(f"  diff {key:<34s} {gap['mean']:+.4f}  [{low:+.4f}, {high:+.4f}]")


if __name__ == "__main__":
    main()
