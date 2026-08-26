"""Four encoder arms put through the same break, and the distance between them.

Day 3 measured one model's coverage guarantee falling apart between hospitals.
This asks whether the encoder underneath changes that, and in particular whether
an encoder that saw the calibration corpus in pre-training looks better at home
for a reason that has nothing to do with being better.

The grid the arms fill:

    arm            saw PTB-XL (calibration)   saw Shandong (target)
    random_init    no                         no
    ecgfounder     no                         no
    ecgfm          yes                        no
    hubert_ecg     yes                        yes

Everything downstream of the encoder is identical across the four: the same
cached representations protocol, the same PTB-XL folds in the same roles, the
same conformal draws with the same seed, so a difference between two rows is a
difference between two pre-trainings.

Three things come out, into ``results/arms.json``:

*Discrimination.*  AUROC and AUPRC per arm and corpus, each with a 95% bootstrap
interval, and the paired difference between every pair of arms on every corpus
(C-18).  Paired because the arms scored the same records: two intervals that
overlap can still hide a difference that is there in every draw.

*The break.*  Coverage on PTB-XL, Shandong and Chongqing under one threshold
fitted on PTB-XL and spent unchanged, over the same 200 draws day 3 used, and
the coverage gap -- home minus target -- as a mean over those draws with its
spread (C-10).  The gap is taken inside each draw, not between two means,
because the draws are shared and subtracting the summaries would throw that
away.

*The distance between arms.*  For each target and each pair of arms, the
difference of their gaps, draw by draw.  This one is a spread over calibration
draws rather than a bootstrap over records, and it is labelled as such: it says
how much the two arms differ across the calibrations we could have drawn, not
how much they would differ on another sample of patients.

Usage: .venv/bin/python scripts/arms_table.py [--draws 200]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ecs.arms import auprc, auroc, fit_probe, paired_difference
from ecs.config import ACS_DIR, ACS_LABELLED_SPLIT, PTBXL_DIR, RESULTS_DIR, SPH_DIR
from ecs.encoders import PRETRAINING
from ecs.labels import MILabelSpec, acs_mi_label, ptbxl_mi_label, sph_mi_label
from ecs.metrics import bootstrap_ci
from ecs.report import Source, Target, frozen_calibration_table
from ecs.splits import ptbxl_benchmark_split

# The arms, in the order the grid reads: the control first, then the arms that
# saw nothing public, then the ones that saw the calibration corpus, then the
# one that saw a target too.  Reading order is the contamination order.
ARM_ORDER = ("random_init", "ecgfounder", "ecgfm", "hubert_ecg")

# The level the headline is quoted at, and the level day 3's break was quoted
# at, so the two are read side by side.  The file carries every level.
HEADLINE = {"alpha": 0.20, "score": "lac", "correction": "none"}
# The grid asks what the encoder underneath does to the break, so it holds the
# two corrections that need nothing estimated; which correction repairs the
# break is the break table's question, not this one's.
ARM_CORRECTIONS = ("none", "mondrian")
ALPHAS = (0.20, 0.10, 0.05)

TARGETS = ("sph", "acs")
METRICS = {"auroc": auroc, "auprc": auprc}


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def read_embeddings(root: Path, arm: str, corpus: str) -> tuple[list[str], NDArray[np.float64]]:
    """One arm's representations of one corpus, ids and vectors in file order."""
    with np.load(root / arm / f"{corpus}.npz", allow_pickle=False) as data:
        return [str(i) for i in data["ids"]], np.asarray(data["embedding"], dtype=np.float64)


def aligned(
    ids: list[str], vectors: NDArray[np.float64], labels: pd.Series
) -> tuple[list[str], NDArray[np.float64], NDArray[np.int_]]:
    """The records this arm embedded *and* the label table knows, in that order.

    An arm's cache holds every record the ingestion chain could read, which is
    not the same set as the labelled one: Chongqing publishes labels for its
    training split only.  Taking the intersection here, rather than trusting the
    two files to line up, is what keeps a label from being read against the
    wrong vector.
    """
    if labels.index.has_duplicates:
        raise ValueError(
            "the label table indexes one record more than once; a lookup would "
            "silently return several labels for one vector"
        )
    known = set(labels.index.astype(str))
    keep = [i for i, key in enumerate(ids) if key in known]
    kept_ids = [ids[i] for i in keep]
    y = labels.loc[kept_ids].to_numpy().astype(int)
    if len(y) != len(kept_ids):
        raise ValueError(f"{len(kept_ids)} records matched {len(y)} labels")
    return kept_ids, vectors[keep], y


def corpus_labels() -> dict[str, pd.Series]:
    """One boolean infarction label per record, per corpus, indexed by record id."""
    spec = MILabelSpec()
    database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    statements = pd.read_csv(PTBXL_DIR / "scp_statements.csv", index_col=0)
    ptbxl = ptbxl_mi_label(database, statements, spec)
    ptbxl.index = ptbxl.index.astype(str)

    metadata = pd.read_csv(SPH_DIR / "metadata.csv")
    sph = sph_mi_label(metadata, spec)
    sph.index = metadata["ECG_ID"].astype(str)

    acs_table = pd.read_csv(ACS_DIR / ACS_LABELLED_SPLIT)
    acs = acs_mi_label(acs_table, spec)
    acs.index = pd.Index([str(f).removesuffix(".dat") for f in acs_table["ecg_row_record"]])
    return {"ptbxl": ptbxl, "sph": sph, "acs": acs}


def target_patients(corpus: str) -> int:
    """How many distinct patients a target's records come from.

    Descriptive rather than load-bearing -- a target is never split, so no
    patient straddles a calibration boundary -- but it is read from the corpus
    rather than substituted with the record count, because a corpus with several
    tracings per patient carries fewer independent draws than its records
    suggest and writing the larger number would say otherwise.
    """
    if corpus == "sph":
        return int(pd.read_csv(SPH_DIR / "metadata.csv")["Patient_ID"].nunique())
    return int(pd.read_csv(ACS_DIR / ACS_LABELLED_SPLIT)["Patient_id"].nunique())


def ptbxl_parts() -> tuple[pd.Series, pd.Series]:
    """Which PTB-XL fold each record is in, and which patient it belongs to."""
    database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    part = ptbxl_benchmark_split(database)
    part.index = part.index.astype(str)
    patients = database["patient_id"].astype(str)
    patients.index = patients.index.astype(str)
    return part, patients


def discrimination(
    labels: NDArray[np.int_], scores: NDArray[np.float64], draws: int, seed: int
) -> dict[str, object]:
    """AUROC and AUPRC with the interval that makes them arguable (C-18)."""
    out: dict[str, object] = {"n_points": int(len(labels)), "n_positive": int(labels.sum())}
    for name, statistic in METRICS.items():
        point, low, high = bootstrap_ci(statistic, labels, scores, n_draws=draws, seed=seed)
        out[name] = point
        out[f"{name}_ci95"] = [low, high]
    return out


def per_draw_gap(row: dict[str, Any], home: str, target: str) -> list[float]:
    """Home coverage minus target coverage, inside each draw."""
    home_draws = row["by_corpus"][home]["coverage_by_draw"]
    target_draws = row["by_corpus"][target]["coverage_by_draw"]
    return [h - t for h, t in zip(home_draws, target_draws, strict=True)]


def series(values: list[float]) -> dict[str, float | int]:
    """A quantity across the calibration draws: where it sits and how it moves.

    The percentiles are the spread of the draws themselves, not a bootstrap
    interval over patients, and the key names say so rather than borrowing the
    word "confidence" for something that is not one.
    """
    array = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(array, [2.5, 97.5])
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "draw_range_2_5": float(low),
        "draw_range_97_5": float(high),
        "n_draws": int(array.size),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embeddings", default=str(RESULTS_DIR / "embeddings"))
    parser.add_argument("--out", default=str(RESULTS_DIR / "arms.json"))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    root = Path(args.embeddings)

    labels = corpus_labels()
    part, patients = ptbxl_parts()

    scored: dict[str, dict[str, dict[str, Any]]] = {}
    probes: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        ids, vectors = read_embeddings(root, arm, "ptbxl")
        kept, x, y = aligned(ids, vectors, labels["ptbxl"])
        where = part.loc[kept].to_numpy()
        probe = fit_probe(
            x[where == "train"],
            y[where == "train"],
            x[where == "validation"],
            y[where == "validation"],
            seed=args.seed,
        )
        held_out = where == "test"
        scored[arm] = {
            "ptbxl": {
                "ids": [k for k, keep in zip(kept, held_out, strict=True) if keep],
                "labels": y[held_out],
                "probs": probe.head.probabilities(x[held_out]),
            }
        }
        for corpus in TARGETS:
            target_ids, target_vectors = read_embeddings(root, arm, corpus)
            kept_ids, tx, ty = aligned(target_ids, target_vectors, labels[corpus])
            scored[arm][corpus] = {
                "ids": kept_ids,
                "labels": ty,
                "probs": probe.head.probabilities(tx),
            }
        probes[arm] = {
            "regularisation_kept": probe.chosen,
            "validation_auroc_by_strength": {str(c): v for c, v in probe.validation_auroc.items()},
            "embedding_dim": int(x.shape[1]),
            "n_train": int((where == "train").sum()),
            "n_validation": int((where == "validation").sum()),
            "n_test": int(held_out.sum()),
        }
        print(
            f"{arm}: C={probe.chosen}, {x.shape[1]}-d, fold 10 n={int(held_out.sum())}",
            flush=True,
        )

    corpora = ("ptbxl", *TARGETS)
    for arm, other in combinations(ARM_ORDER, 2):
        for corpus in corpora:
            if scored[arm][corpus]["ids"] != scored[other][corpus]["ids"]:
                raise SystemExit(
                    f"{arm} and {other} did not score the same {corpus} records; "
                    "the arms cannot be compared pairwise until they do"
                )

    metrics = {
        arm: {
            corpus: discrimination(
                scored[arm][corpus]["labels"],
                scored[arm][corpus]["probs"][:, 1],
                args.bootstrap_draws,
                args.seed,
            )
            for corpus in corpora
        }
        for arm in ARM_ORDER
    }
    paired_metrics = {
        corpus: {
            f"{arm} - {other}": {
                name: paired_difference(
                    statistic,
                    scored[arm][corpus]["labels"],
                    scored[arm][corpus]["probs"][:, 1],
                    scored[other][corpus]["probs"][:, 1],
                    n_draws=args.bootstrap_draws,
                    seed=args.seed,
                )
                for name, statistic in METRICS.items()
            }
            for arm, other in combinations(ARM_ORDER, 2)
        }
        for corpus in corpora
    }

    tables: dict[str, list[dict[str, Any]]] = {}
    for arm in ARM_ORDER:
        source = Source(
            "ptbxl",
            scored[arm]["ptbxl"]["probs"],
            scored[arm]["ptbxl"]["labels"],
            list(patients.loc[scored[arm]["ptbxl"]["ids"]]),
        )
        targets = {
            corpus: Target(
                scored[arm][corpus]["probs"],
                scored[arm][corpus]["labels"],
                target_patients(corpus),
            )
            for corpus in TARGETS
        }
        tables[arm] = frozen_calibration_table(
            source,
            targets,
            ALPHAS,
            n_draws=args.draws,
            seed=args.seed,
            keep_draws=True,
            corrections=ARM_CORRECTIONS,
        )
        print(f"{arm}: conformal table over {args.draws} draws", flush=True)

    def row_of(arm: str, setting: dict[str, Any]) -> dict[str, Any]:
        for row in tables[arm]:
            if all(row[key] == value for key, value in setting.items()):
                return row
        raise KeyError(f"no row for {setting}")

    settings = [
        {"alpha": alpha, "score": score, "correction": correction}
        for alpha in ALPHAS
        for score in ("lac", "aps")
        for correction in ARM_CORRECTIONS
    ]
    gaps: dict[str, list[dict[str, Any]]] = {}
    for arm in ARM_ORDER:
        gaps[arm] = []
        for setting in settings:
            row = row_of(arm, setting)
            gaps[arm].append(
                {
                    **setting,
                    "target_coverage": row["target_coverage"],
                    "coverage": {
                        corpus: row["by_corpus"][corpus]["coverage"] for corpus in corpora
                    },
                    "coverage_gap": {
                        corpus: series(per_draw_gap(row, "ptbxl", corpus)) for corpus in TARGETS
                    },
                }
            )

    paired_gaps = {}
    for setting in settings:
        key = (
            f"alpha={setting['alpha']} score={setting['score']} correction={setting['correction']}"
        )
        paired_gaps[key] = {
            corpus: {
                f"{arm} - {other}": series(
                    [
                        a - b
                        for a, b in zip(
                            per_draw_gap(row_of(arm, setting), "ptbxl", corpus),
                            per_draw_gap(row_of(other, setting), "ptbxl", corpus),
                            strict=True,
                        )
                    ]
                )
                for arm, other in combinations(ARM_ORDER, 2)
            }
            for corpus in TARGETS
        }

    result = {
        "question": (
            "does the encoder underneath change how far a PTB-XL coverage guarantee "
            "falls at another hospital, and does an encoder that saw PTB-XL look better "
            "at home for a reason other than being better"
        ),
        "protocol": (
            "each arm is frozen; a linear probe is fitted on its PTB-XL fold 1-8 "
            "representations, its regularisation chosen on fold 9, and fold 10 and both "
            "external corpora are scored once with it; the conformal threshold is then "
            "fitted on half the fold-10 patients and spent unchanged on the other half "
            "and on both external corpora, exactly as in results/shift.json"
        ),
        "arms": {
            arm: {"pretraining_corpora": PRETRAINING[arm], **probes[arm]} for arm in ARM_ORDER
        },
        "headline": HEADLINE,
        "n_draws": args.draws,
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "git_commit": git_commit(),
        "discrimination": metrics,
        "discrimination_paired": paired_metrics,
        "coverage": gaps,
        "coverage_gap_paired": paired_gaps,
        "paired_note": (
            "discrimination_paired resamples records, so its interval is a statement "
            "about another sample of patients; coverage_gap_paired is the spread over "
            "the calibration draws themselves and says how much two arms differ across "
            "the calibrations that could have been drawn"
        ),
        "seconds": round(time.perf_counter() - started, 1),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")

    print(f"\nheadline: {HEADLINE}")
    for arm in ARM_ORDER:
        row = [g for g in gaps[arm] if all(g[k] == v for k, v in HEADLINE.items())][0]
        home = row["coverage"]["ptbxl"]["mean"]
        print(
            f"  {arm:<12} home {home:.3f}  "
            + "  ".join(
                f"{c} gap {row['coverage_gap'][c]['mean']:+.3f}±{row['coverage_gap'][c]['sd']:.3f}"
                for c in TARGETS
            )
            + f"  AUROC ptbxl {metrics[arm]['ptbxl']['auroc']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
