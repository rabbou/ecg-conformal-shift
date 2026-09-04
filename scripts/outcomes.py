"""What a patient of each label actually gets, under three ways of thresholding.

`shift_table.py` reports coverage: whether the true label is inside the set.  A
set holding both labels covers the truth and still decides nothing, so coverage
alone cannot say whether a scheme serves a patient or hands the tracing on.
This table splits every case of a label into the three things that can happen to
it -- the correct label alone, a deferral, or the wrong label alone -- which is
what makes a conformal scheme comparable with an ordinary tuned threshold.

The three schemes share the model's scores and differ only in where the
thresholds sit:

  plain     one threshold fitted for 90% sensitivity on the calibration half.
            Every case is labelled; there is no deferral.
  pooled    one threshold pair fitted on the whole calibration half, so the
            coverage asked for holds over all cases at once.
  perlabel  one threshold fitted inside each label, so the coverage asked for
            holds within each -- label conditional validity, Vovk (2012),
            Proposition 3.

The plain scheme's 90% is a sensitivity and the two conformal 90%s are
coverages.  They are matched deliberately: it puts the three schemes at
comparable miss rates on the source, so the remaining differences are the
false-alarm rate and what each scheme defers.

The spread comes from re-drawing the calibration alone, on the same protocol as
`shift_table.py`: each draw halves PTB-XL fold 10 by patient (C-4), fits every
threshold on one half, and spends them on the other half and on both external
corpora unchanged (C-20).

Usage: .venv/bin/python scripts/outcomes.py [--draws 200]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shift_table import patients_of

from ecs.config import PTBXL_DIR, RESULTS_DIR
from ecs.conformal import (
    conformal_quantile,
    lac_scores,
    lac_scores_all,
    mondrian_quantiles,
    predict_sets,
    predict_sets_per_class,
)

# The setting the reading is written at, matching shift_table.py.
ALPHA = 0.10
# The sensitivity the plain threshold is fitted for.  Equal to 1 - ALPHA so the
# schemes meet at a comparable miss rate rather than at an arbitrary pair.
PLAIN_SENSITIVITY = 1.0 - ALPHA
SCHEMES = ("plain", "pooled", "perlabel")
CORPORA = ("ptbxl", "sph", "acs")
CORPUS_NAMES = {"ptbxl": "PTB-XL", "sph": "Shandong", "acs": "Chongqing"}
OUTCOMES = ("correct", "deferred", "wrong")


def _split_outcomes(
    sets: NDArray[np.bool_], labels: NDArray[np.int_], klass: int
) -> tuple[float, float, float]:
    """The three shares for one label: correct alone, deferred, wrong alone.

    Deferral covers both a set holding both labels and a set holding neither:
    from the reader's side they are the same event, no machine answer.  The
    three shares are of every case carrying the label, so they sum to one and a
    scheme with deferrals stays comparable with one without.
    """
    of_class = labels == klass
    positive, negative = sets[:, 1], sets[:, 0]
    deferred = (positive & negative) | (~positive & ~negative)
    correct = (positive & ~negative) if klass == 1 else (negative & ~positive)
    wrong = (negative & ~positive) if klass == 1 else (positive & ~negative)
    return (
        float(np.mean(correct[of_class])),
        float(np.mean(deferred[of_class])),
        float(np.mean(wrong[of_class])),
    )


def _plain_sets(probs: NDArray[np.float64], threshold: float) -> NDArray[np.bool_]:
    """One threshold, every case labelled, nothing deferred."""
    positive = probs[:, 1] >= threshold
    return np.column_stack([~positive, positive])


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 4),
        "sd": round(float(array.std(ddof=1)), 4),
        "n_draws": int(array.size),
    }


def collect(
    scores_by_corpus: dict[str, dict[str, NDArray[Any]]],
    patients: list[str],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Re-draw the calibration half and record every scheme on every corpus."""
    source = scores_by_corpus["ptbxl"]
    unique = np.unique(np.asarray(patients))
    rng = np.random.default_rng(seed)
    tally: dict[tuple[str, str, str, str], list[float]] = {}
    thresholds: dict[tuple[str, str], list[float]] = {}

    for _ in range(draws):
        rng.shuffle(unique)
        held = set(unique[: len(unique) // 2].tolist())
        is_calibration = np.array([p in held for p in patients])
        calibration_probs = source["probs"][is_calibration]
        calibration_labels = source["labels"][is_calibration]

        scores = lac_scores(calibration_probs, calibration_labels)
        pooled_q = conformal_quantile(scores, ALPHA)
        perlabel_q = mondrian_quantiles(scores, calibration_labels, ALPHA, 2)
        # The operating point a sensitivity target buys: the score below which
        # PLAIN_SENSITIVITY of the calibration sick already sit.
        plain_q = float(
            np.quantile(calibration_probs[:, 1][calibration_labels == 1], 1.0 - PLAIN_SENSITIVITY)
        )
        # The quantiles are on the LAC score, 1 - p(true label); the report and
        # its figures read the probability axis.  A label enters the set when
        # 1 - p <= q, so p >= 1 - q: the MI boundary is 1 - q(MI) and the non-MI
        # boundary is q(non-MI).  Recorded as boundaries so nothing downstream
        # has to redo the conversion and get it wrong.
        thresholds.setdefault(("plain", "single"), []).append(plain_q)
        thresholds.setdefault(("pooled", "lower"), []).append(1.0 - pooled_q)
        thresholds.setdefault(("pooled", "upper"), []).append(pooled_q)
        thresholds.setdefault(("perlabel", "lower"), []).append(1.0 - float(perlabel_q[1]))
        thresholds.setdefault(("perlabel", "upper"), []).append(float(perlabel_q[0]))
        for klass, value in enumerate(perlabel_q):
            thresholds.setdefault(("quantile", str(klass)), []).append(float(value))

        for corpus, bundle in scores_by_corpus.items():
            probs = bundle["probs"]
            labels = bundle["labels"]
            if corpus == "ptbxl":
                probs, labels = probs[~is_calibration], labels[~is_calibration]
            all_scores = lac_scores_all(probs)
            built = {
                "plain": _plain_sets(probs, plain_q),
                "pooled": predict_sets(all_scores, pooled_q),
                "perlabel": predict_sets_per_class(all_scores, perlabel_q),
            }
            for scheme, sets in built.items():
                for klass in (0, 1):
                    shares = _split_outcomes(sets, labels, klass)
                    for name, share in zip(OUTCOMES, shares, strict=True):
                        tally.setdefault((corpus, scheme, str(klass), name), []).append(share)

    by_corpus: dict[str, Any] = {}
    for corpus, bundle in scores_by_corpus.items():
        prevalence = float(np.mean(bundle["labels"] == 1))
        n_points = int(bundle["labels"].size)
        if corpus == "ptbxl":
            n_points = int(round(n_points / 2))
        by_corpus[corpus] = {
            "name": CORPUS_NAMES[corpus],
            "n_points": n_points,
            "prevalence": round(prevalence, 4),
            "schemes": {
                scheme: {
                    klass: {
                        name: _summary(tally[(corpus, scheme, klass, name)]) for name in OUTCOMES
                    }
                    for klass in ("0", "1")
                }
                for scheme in SCHEMES
            },
        }
    return {
        "by_corpus": by_corpus,
        "thresholds": {
            f"{scheme}:{which}": _summary(v) for (scheme, which), v in thresholds.items()
        },
        "threshold_note": (
            "Boundaries are on the model's probability axis. A case below the "
            "lower boundary gets the non-MI label alone, above the upper boundary "
            "the MI label alone, and between them both labels: a deferral. The "
            "'quantile:*' entries are the raw LAC quantiles the boundaries come from."
        ),
    }


def build(draws: int, seed: int) -> dict[str, Any]:
    started = time.time()
    scores_by_corpus = {
        "ptbxl": dict(np.load(RESULTS_DIR / "baseline/scores.npz")),
        "sph": dict(np.load(RESULTS_DIR / "external/sph.npz")),
        "acs": dict(np.load(RESULTS_DIR / "external/acs.npz")),
    }
    patients = patients_of([str(i) for i in scores_by_corpus["ptbxl"]["ids"]], PTBXL_DIR)
    body = collect(scores_by_corpus, patients, draws, seed)
    return {
        "question": (
            "For a patient of each label, what does the model return: the correct "
            "label alone, no decision, or the wrong label alone -- under a single "
            "tuned threshold, pooled conformal calibration, and label conditional "
            "conformal calibration."
        ),
        "protocol": (
            "Every threshold is fitted on a PTB-XL calibration half drawn by patient "
            "and spent unchanged on the held-out half and on both external corpora. "
            "No external label reaches any threshold."
        ),
        "alpha": ALPHA,
        "plain_sensitivity": PLAIN_SENSITIVITY,
        "matched_operating_point": (
            "The plain threshold's target is a sensitivity and the two conformal "
            "targets are coverages. They are set to the same number so the schemes "
            "meet at a comparable miss rate on the source corpus."
        ),
        "outcome_definitions": {
            "correct": "the set holds that label alone",
            "deferred": "the set holds both labels or neither; no machine answer",
            "wrong": "the set holds the other label alone",
            "denominator": "every case carrying the label, so the three shares sum to one",
        },
        "n_draws": draws,
        "seed": seed,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "seconds": round(time.time() - started, 1),
        **body,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "outcomes.json")
    args = parser.parse_args()
    table = build(args.draws, args.seed)
    args.out.write_text(json.dumps(table, indent=2) + "\n")
    print(f"wrote {args.out}")
    for corpus in CORPORA:
        block = table["by_corpus"][corpus]
        for scheme in SCHEMES:
            sick = block["schemes"][scheme]["1"]
            healthy = block["schemes"][scheme]["0"]
            print(
                f"  {block['name']:>9s} {scheme:<9s} "
                f"MI wrong {sick['wrong']['mean']:.3f}  "
                f"MI deferred {sick['deferred']['mean']:.3f}  "
                f"non-MI wrong {healthy['wrong']['mean']:.3f}"
            )


if __name__ == "__main__":
    main()
