"""What four acquisition faults do to coverage, and whether any scheme notices.

A hospital does not decide to change the input distribution.  It swaps two
cables, re-scales a gain, or records over a wandering baseline, and the model
goes on returning sets.  The question this script answers is whether the
deferral rate -- the only signal a deployed conformal scheme emits without
labels -- moves when the input breaks.

Every threshold is fitted on a clean calibration half, as it would be at a site
that calibrated before the fault, and spent on a perturbed test half.  The
draws, the seed and the halving are the ones ``outcomes.py`` uses, so the clean
row of the table is table 1's row and the difference is the fault alone.

The identity condition is scored through the same path as everything else and
checked against the committed ``results/baseline/scores.npz``.  If that check
fails, nothing below is about perturbation.

The transformations, in the canonical lead order I, II, III, aVR, aVL, aVF,
V1-V6:

  limb_reversal   the left and right arm electrodes swapped, the commonest
                  placement error.  From Einthoven's definitions I = LA - RA,
                  II = LL - RA, III = LL - LA and Goldberger's augmented leads,
                  exchanging LA and RA gives I -> -I, II <-> III, aVR <-> aVL,
                  aVF unchanged.  Wilson's central terminal is the mean of the
                  three limb electrodes and does not move, so V1-V6 are
                  untouched.  This is a relabelling of channels, not noise: no
                  information is lost and a reader sees it at once.
  gain_080        every lead scaled by 0.8, and
  gain_125        by 1.25: a calibration pulse set to the wrong standard.
  baseline_wander a 0.3 Hz sinusoid of 0.5 mV added to every lead, phase drawn
                  per record, which is respiration and electrode drift.
  polarity        every lead multiplied by -1: the whole cable set inverted.

Usage: .venv/bin/python scripts/perturbations.py [--draws 200]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from score_external import load_model, probabilities
from sklearn.metrics import roc_auc_score

from ecs.config import PTBXL_DIR, RESULTS_DIR, SAMPLING_RATE_HZ
from ecs.conformal import (
    conformal_quantile,
    lac_scores,
    lac_scores_all,
    mondrian_quantiles,
    predict_sets,
    predict_sets_per_class,
)
from ecs.ingest import load_ptbxl
from ecs.labels import MILabelSpec, ptbxl_mi_label
from ecs.splits import ptbxl_benchmark_split

ALPHA = 0.10
PLAIN_SENSITIVITY = 1.0 - ALPHA
SCHEMES = ("plain", "pooled", "perlabel")
DRAWS = 200
CHUNK = 500

WANDER_HZ = 0.3
WANDER_MV = 0.5

# Positions in the canonical lead order.
LEAD_I, LEAD_II, LEAD_III, LEAD_AVR, LEAD_AVL = 0, 1, 2, 3, 4


def limb_reversal(x: NDArray[np.float32], rng: np.random.Generator) -> NDArray[np.float32]:
    """Left and right arm electrodes exchanged."""
    del rng
    out = x.copy()
    out[:, LEAD_I] = -x[:, LEAD_I]
    out[:, LEAD_II], out[:, LEAD_III] = x[:, LEAD_III], x[:, LEAD_II]
    out[:, LEAD_AVR], out[:, LEAD_AVL] = x[:, LEAD_AVL], x[:, LEAD_AVR]
    return out


def gain(
    factor: float,
) -> Callable[[NDArray[np.float32], np.random.Generator], NDArray[np.float32]]:
    def apply(x: NDArray[np.float32], rng: np.random.Generator) -> NDArray[np.float32]:
        del rng
        return (x * factor).astype(np.float32)

    return apply


def baseline_wander(x: NDArray[np.float32], rng: np.random.Generator) -> NDArray[np.float32]:
    """A slow sinusoid on every lead, phase drawn once per record."""
    t = np.arange(x.shape[2], dtype=np.float32) / SAMPLING_RATE_HZ
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(x.shape[0], 1, 1)).astype(np.float32)
    drift = WANDER_MV * np.sin(2.0 * np.pi * WANDER_HZ * t[None, None, :] + phase)
    return (x + drift).astype(np.float32)


def polarity(x: NDArray[np.float32], rng: np.random.Generator) -> NDArray[np.float32]:
    del rng
    return -x


CONDITIONS: dict[str, Callable[[NDArray[np.float32], np.random.Generator], NDArray[np.float32]]] = {
    "clean": lambda x, rng: x,
    "limb_reversal": limb_reversal,
    "gain_080": gain(0.8),
    "gain_125": gain(1.25),
    "baseline_wander": baseline_wander,
    "polarity": polarity,
}
DESCRIPTIONS = {
    "clean": "the tracings as recorded; the row that must reproduce results/baseline/scores.npz",
    "limb_reversal": "left and right arm electrodes exchanged",
    "gain_080": "every lead scaled by 0.8",
    "gain_125": "every lead scaled by 1.25",
    "baseline_wander": f"a {WANDER_HZ} Hz sinusoid of {WANDER_MV} mV added to every lead",
    "polarity": "every lead multiplied by -1",
}


def read_fold_ten(
    root: Path,
) -> tuple[NDArray[np.float32], NDArray[np.int_], NDArray[Any], list[str]]:
    """Fold 10's tracings, labels, patients and identifiers, in the scored order."""
    database = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    labels = ptbxl_mi_label(database, statements, MILabelSpec())
    wanted = ptbxl_benchmark_split(database).pipe(lambda s: s.index[s == "test"])

    ids: list[str] = []
    blocks: list[NDArray[np.float32]] = []
    for start in range(0, len(wanted), CHUNK):
        chunk = load_ptbxl(database, root=root, ids=[int(i) for i in wanted[start : start + CHUNK]])
        ids.extend(chunk.ids)
        blocks.append(chunk.x)
        print(f"  read {len(ids):>6} / {len(wanted)}", flush=True)
    x = np.concatenate(blocks)
    y = labels.loc[[int(i) for i in ids]].to_numpy().astype(int)
    patients = np.array([str(database.loc[int(i), "patient_id"]) for i in ids])
    return x, y, patients, ids


def _sets(
    probs: NDArray[np.float64], plain_q: float, pooled_q: float, per_q: NDArray[np.float64]
) -> dict[str, NDArray[np.bool_]]:
    positive = probs[:, 1] >= plain_q
    all_scores = lac_scores_all(probs)
    return {
        "plain": np.column_stack([~positive, positive]),
        "pooled": predict_sets(all_scores, pooled_q),
        "perlabel": predict_sets_per_class(all_scores, per_q),
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 4),
        "sd": round(float(array.std(ddof=1)), 4),
        "n_draws": int(array.size),
    }


def measure(
    clean: NDArray[np.float64],
    perturbed: dict[str, NDArray[np.float64]],
    labels: NDArray[np.int_],
    patients: NDArray[Any],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Thresholds from the clean calibration half, spent on the perturbed half."""
    unique = np.unique(patients)
    rng = np.random.default_rng(seed)
    tally: dict[tuple[str, str, str], list[float]] = {}

    for _ in range(draws):
        rng.shuffle(unique)
        held = set(unique[: len(unique) // 2].tolist())
        is_cal = np.array([p in held for p in patients])

        cal_probs, cal_labels = clean[is_cal], labels[is_cal]
        scores = lac_scores(cal_probs, cal_labels)
        pooled_q = conformal_quantile(scores, ALPHA)
        per_q = mondrian_quantiles(scores, cal_labels, ALPHA, 2)
        plain_q = float(np.quantile(cal_probs[:, 1][cal_labels == 1], 1.0 - PLAIN_SENSITIVITY))

        test_labels = labels[~is_cal]
        for name, probs in perturbed.items():
            for scheme, sets in _sets(probs[~is_cal], plain_q, pooled_q, per_q).items():
                covered = sets[np.arange(test_labels.size), test_labels]
                deferred = sets[:, 0] & sets[:, 1] | ~(sets[:, 0] | sets[:, 1])
                for klass, label in ((1, "mi"), (0, "non_mi")):
                    of_class = test_labels == klass
                    tally.setdefault((name, scheme, f"coverage_{label}"), []).append(
                        float(covered[of_class].mean())
                    )
                    tally.setdefault((name, scheme, f"deferred_{label}"), []).append(
                        float(deferred[of_class].mean())
                    )
                tally.setdefault((name, scheme, "coverage_all"), []).append(float(covered.mean()))
                tally.setdefault((name, scheme, "deferred_all"), []).append(float(deferred.mean()))

    return {
        name: {
            "description": DESCRIPTIONS[name],
            "auroc": round(float(roc_auc_score(labels, perturbed[name][:, 1])), 4),
            "schemes": {
                scheme: {
                    key: _summary(values)
                    for (n, s, key), values in tally.items()
                    if n == name and s == scheme
                }
                for scheme in SCHEMES
            },
        }
        for name in perturbed
    }


def build(
    draws: int, seed: int, root: Path, checkpoint: Path, threads: int
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    started = time.time()
    torch.set_num_threads(threads)
    model, centre, scale = load_model(checkpoint)
    x, labels, patients, ids = read_fold_ten(root)

    scored: dict[str, NDArray[np.float64]] = {}
    for name, transform in CONDITIONS.items():
        rng = np.random.default_rng(seed)
        moved = transform(x, rng)
        scored[name] = probabilities(model, moved, centre, scale)
        del moved
        print(f"  scored {name}", flush=True)

    published = np.load(RESULTS_DIR / "baseline/scores.npz")
    if [str(i) for i in published["ids"]] != ids:
        raise ValueError("fold 10 came back in a different order than the committed scores")
    largest = float(np.abs(scored["clean"] - published["probs"].astype(np.float64)).max())

    return {
        "question": (
            "When an acquisition fault changes the input, does coverage fall, and "
            "does the deferral rate -- the only thing a deployed scheme can see "
            "without labels -- move enough to signal it."
        ),
        "protocol": (
            "Every threshold is fitted on a clean PTB-XL calibration half drawn by "
            "patient, the same 200 halves outcomes.py draws under the same seed, and "
            "spent on the perturbed held-out half. AUROC is over the whole perturbed "
            "fold. The clean condition is scored through this path and checked "
            "against results/baseline/scores.npz."
        ),
        "alpha": ALPHA,
        "n_draws": draws,
        "seed": seed,
        "n_records": int(labels.size),
        "n_positive": int((labels == 1).sum()),
        "torch_threads": threads,
        "clean_matches_published_scores": {
            "largest_absolute_difference": largest,
            "tolerance": 1e-6,
            "passed": bool(largest < 1e-6),
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "seconds": round(time.time() - started, 1),
        "conditions": measure(scored["clean"], scored, labels, patients, draws, seed),
    }, {
        "ids": np.array(ids),
        "labels": labels,
        **{f"probs_{name}": probs for name, probs in scored.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ptbxl-dir", type=Path, default=PTBXL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=RESULTS_DIR / "baseline/model.pt")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "perturbations.json")
    parser.add_argument("--scores-out", type=Path, default=RESULTS_DIR / "perturbations.npz")
    args = parser.parse_args()
    table, scores = build(args.draws, args.seed, args.ptbxl_dir, args.checkpoint, args.threads)
    args.out.write_text(json.dumps(table, indent=2) + "\n")
    # The checkpoint and the raw tracings are not in the repository, so the
    # scores go in beside the table: everything below the model is then
    # recomputable from committed files, as it is for the external corpora.
    np.savez_compressed(args.scores_out, **scores)
    print(f"wrote {args.out} and {args.scores_out}")
    check = table["clean_matches_published_scores"]
    print(f"  clean vs published scores: max |diff| {check['largest_absolute_difference']:.2e}")
    for name, block in table["conditions"].items():
        per = block["schemes"]["perlabel"]
        print(
            f"  {name:<16s} AUROC {block['auroc']:.4f}  "
            f"MI cov {per['coverage_mi']['mean']:.4f}  "
            f"non-MI cov {per['coverage_non_mi']['mean']:.4f}  "
            f"deferred {per['deferred_all']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
