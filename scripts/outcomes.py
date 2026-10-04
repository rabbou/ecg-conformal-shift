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
`shift_table.py`: each draw halves PTB-XL fold 10 by patient, fits every
threshold on one half, and spends them on the other half and on both external
corpora unchanged.

Usage: .venv/bin/python scripts/outcomes.py [--draws 200]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ecs.config import PTBXL_DIR, RESULTS_DIR
from ecs.conformal import SCHEMES, fit_thresholds
from ecs.ingest import ptbxl_patients
from ecs.metrics import outcome_shares
from ecs.provenance import provenance_block
from ecs.report import draw_summary
from ecs.splits import calibration_halves

# The setting the reading is written at, matching shift_table.py.
ALPHA = 0.10
CORPORA = ("ptbxl", "sph", "acs")
CORPUS_NAMES = {"ptbxl": "PTB-XL", "sph": "Shandong", "acs": "Chongqing"}
OUTCOMES = ("correct", "deferred", "wrong")


def collect(
    scores_by_corpus: dict[str, dict[str, NDArray[Any]]],
    patients: list[str],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Re-draw the calibration half and record every scheme on every corpus."""
    source = scores_by_corpus["ptbxl"]
    tally: dict[tuple[str, str, str, str], list[float]] = {}
    thresholds: dict[tuple[str, str], list[float]] = {}
    # The halving is by patient, so the record counts on each side vary from
    # draw to draw.  Recorded rather than assumed, because the figures print it.
    scored: dict[str, list[float]] = {}
    positives: dict[str, list[float]] = {}

    for is_calibration in calibration_halves(np.asarray(patients), draws, seed):
        fitted = fit_thresholds(
            source["probs"][is_calibration], source["labels"][is_calibration], ALPHA
        )
        # The quantiles are on the LAC score, 1 - p(true label); the report and
        # its figures read the probability axis.  A label enters the set when
        # 1 - p <= q, so p >= 1 - q: the MI boundary is 1 - q(MI) and the non-MI
        # boundary is q(non-MI).  Recorded as boundaries so nothing downstream
        # has to redo the conversion and get it wrong.
        thresholds.setdefault(("plain", "single"), []).append(1.0 - fitted.plain)
        thresholds.setdefault(("pooled", "lower"), []).append(1.0 - fitted.pooled)
        thresholds.setdefault(("pooled", "upper"), []).append(fitted.pooled)
        thresholds.setdefault(("perlabel", "lower"), []).append(1.0 - float(fitted.perlabel[1]))
        thresholds.setdefault(("perlabel", "upper"), []).append(float(fitted.perlabel[0]))
        for klass, value in enumerate(fitted.perlabel):
            thresholds.setdefault(("quantile", str(klass)), []).append(float(value))

        for corpus, bundle in scores_by_corpus.items():
            probs = bundle["probs"]
            labels = bundle["labels"]
            if corpus == "ptbxl":
                probs, labels = probs[~is_calibration], labels[~is_calibration]
            scored.setdefault(corpus, []).append(float(labels.size))
            positives.setdefault(corpus, []).append(float((labels == 1).sum()))
            for scheme, sets in fitted.sets(probs).items():
                for klass in (0, 1):
                    shares = outcome_shares(sets, labels, klass)
                    for name, share in zip(OUTCOMES, shares, strict=True):
                        tally.setdefault((corpus, scheme, str(klass), name), []).append(share)

    by_corpus: dict[str, Any] = {}
    for corpus, bundle in scores_by_corpus.items():
        prevalence = float(np.mean(bundle["labels"] == 1))
        by_corpus[corpus] = {
            "name": CORPUS_NAMES[corpus],
            "n_scored": draw_summary(scored[corpus]),
            "n_positive": draw_summary(positives[corpus]),
            "prevalence": round(prevalence, 4),
            "schemes": {
                scheme: {
                    klass: {
                        name: draw_summary(tally[(corpus, scheme, klass, name)])
                        for name in OUTCOMES
                    }
                    for klass in ("0", "1")
                }
                for scheme in SCHEMES
            },
        }
    return {
        "by_corpus": by_corpus,
        "thresholds": {
            f"{scheme}:{which}": draw_summary(v) for (scheme, which), v in thresholds.items()
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
    patients = ptbxl_patients(scores_by_corpus["ptbxl"]["ids"], PTBXL_DIR)
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
        "plain_sensitivity": 1.0 - ALPHA,
        "matched_operating_point": (
            "The plain threshold is the per-label conformal quantile of the MI class: "
            "the score that 90% of the calibration MI cases clear. The plain and "
            "per-label schemes therefore share their MI threshold and miss the same "
            "MI cases on every draw; per-label adds the non-MI threshold."
        ),
        "outcome_definitions": {
            "correct": "the set holds that label alone",
            "deferred": "the set holds both labels or neither; no machine answer",
            "wrong": "the set holds the other label alone",
            "denominator": "every case carrying the label, so the three shares sum to one",
        },
        "n_draws": draws,
        "seed": seed,
        "provenance": provenance_block(
            [
                "scripts/outcomes.py",
                "src/ecs/conformal.py",
                "src/ecs/ingest.py",
                "src/ecs/metrics.py",
                "src/ecs/report.py",
                "src/ecs/splits.py",
            ]
        ),
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
