"""The infarction quantities the report argues from that are not a coverage table.

Everything here reads the committed score files and nothing else, and every
calibration draw is the one ``outcomes.py`` and ``shift_table.py`` read.

  differences        a per-patient bootstrap on the gap between the pooled and
                     per-label rules.  A replicate resamples the PTB-XL patients
                     with replacement, halves the resampled patients, fits every
                     threshold on one half, and evaluates on the other and on a
                     resampled target cohort.  The target corpora are resampled
                     by record: the committed score files carry no patient
                     identifier, and both corpora hold close to one tracing per
                     patient, 25,770 from 24,666 at Shandong and 17,955 from
                     17,018 at Chongqing.
  chow               a reject rule in Chow's sense (1970), thresholds placed at
                     the empirical 90th percentile of each label's calibration
                     scores, and the symmetric variant that refuses one band
                     around the sensitivity threshold.  Nothing conformal in either.
  correction         what the (n+1) in the conformal quantile is worth, as the
                     coverage difference between the rank it takes and the rank
                     the plain empirical quantile takes, against the spread
                     between draws.
  sample_size        the MI cases needed to pin a per-label coverage near 90% to
                     within two points, and the tracings that implies at each
                     site's prevalence.
  predictive_value   what a positive answer is worth at each site, which
                     coverage does not say.
  site_intervals     each target site's MI coverage with the Wilson interval its
                     own MI cases support.
  discrimination     the baseline model's AUROC on each corpus.

Usage: .venv/bin/python scripts/auxiliary.py [--bootstrap 2000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from ecs.config import PTBXL_DIR, RESULTS_DIR
from ecs.conformal import fit_thresholds, lac_scores, lac_scores_all, predict_sets_per_class
from ecs.ingest import ptbxl_patients
from ecs.metrics import bootstrap_ci, percentile_interval, wilson_interval
from ecs.provenance import provenance_block
from ecs.splits import calibration_half, calibration_halves, resample_patients

ALPHA = 0.10
DRAWS = 200
BOOTSTRAP_DRAWS = 2000
AUROC_DRAWS = 1000
TARGET_HALF_WIDTH = 0.02
CORPUS_NAMES = {"ptbxl": "PTB-XL", "sph": "Shandong", "acs": "Chongqing"}


def _mi_coverage(sets: NDArray[np.bool_], labels: NDArray[np.int_]) -> float:
    return float(sets[labels == 1, 1].mean())


def _non_mi_coverage(sets: NDArray[np.bool_], labels: NDArray[np.int_]) -> float:
    return float(sets[labels == 0, 0].mean())


def _chow_sets(probs: NDArray[Any], mi_cut: float, non_mi_cut: float) -> NDArray[np.bool_]:
    """A two-threshold reject rule, in the column order the rest of the code uses.

    Column 0 admits non-MI, column 1 admits MI. A tracing scoring at or above
    ``mi_cut`` may be MI, one at or below ``non_mi_cut`` may be non-MI, and one
    that is both gets both labels, which is a refusal. Nothing conformal in it.
    """
    score = probs[:, 1]
    return np.column_stack([score <= non_mi_cut, score >= mi_cut])


def _percentile(values: list[float]) -> list[float]:
    return [round(bound, 4) for bound in percentile_interval(values)]


def differences(
    corpora: dict[str, dict[str, NDArray[Any]]],
    patients: NDArray[Any],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Per-patient bootstrap on the gap between the two conformal schemes."""
    rng = np.random.default_rng(seed)
    gaps: dict[str, list[float]] = {}
    levels: dict[str, list[float]] = {}

    for _ in range(bootstrap):
        rows, keys = resample_patients(patients, rng)
        is_cal = calibration_half(pd.Series(keys), int(rng.integers(2**31)))

        source = corpora["ptbxl"]
        probs, labels = source["probs"][rows], source["labels"][rows]
        fitted = fit_thresholds(probs[is_cal], labels[is_cal], ALPHA)

        for name, bundle in corpora.items():
            if name == "ptbxl":
                test_probs, test_labels = probs[~is_cal], labels[~is_cal]
            else:
                draw = rng.integers(0, bundle["labels"].size, bundle["labels"].size)
                test_probs, test_labels = bundle["probs"][draw], bundle["labels"][draw]
            sets = fitted.sets(test_probs)
            pooled = _mi_coverage(sets["pooled"], test_labels)
            per = _mi_coverage(sets["perlabel"], test_labels)
            gaps.setdefault(name, []).append(per - pooled)
            levels.setdefault(f"{name}:pooled", []).append(pooled)
            levels.setdefault(f"{name}:perlabel", []).append(per)

    return {
        "note": (
            "MI coverage under class-conditional calibration minus MI coverage under "
            "pooled calibration, formed inside each replicate. The source cohort is "
            "resampled by patient and the target cohorts by record."
        ),
        "n_bootstrap": bootstrap,
        "by_corpus": {
            name: {
                "name": CORPUS_NAMES[name],
                "difference": {
                    "mean": round(float(np.mean(values)), 4),
                    "ci95": _percentile(values),
                },
                "pooled": {
                    "mean": round(float(np.mean(levels[f"{name}:pooled"])), 4),
                    "ci95": _percentile(levels[f"{name}:pooled"]),
                },
                "perlabel": {
                    "mean": round(float(np.mean(levels[f"{name}:perlabel"])), 4),
                    "ci95": _percentile(levels[f"{name}:perlabel"]),
                },
            }
            for name, values in gaps.items()
        },
    }


def chow(
    corpora: dict[str, dict[str, NDArray[Any]]],
    patients: NDArray[Any],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """A reject rule against the conformal scheme it is said to reproduce."""
    tally: dict[tuple[str, str], list[float]] = {}

    for is_cal in calibration_halves(patients, draws, seed):
        source = corpora["ptbxl"]
        cal_probs, cal_labels = source["probs"][is_cal], source["labels"][is_cal]
        fitted = fit_thresholds(cal_probs, cal_labels, ALPHA)

        # Per-class Chow: the 10th percentile of the MI scores admits MI, the
        # 90th percentile of the non-MI scores admits non-MI. No (n+1).
        mi_cut = float(np.quantile(cal_probs[:, 1][cal_labels == 1], ALPHA))
        non_mi_cut = float(np.quantile(cal_probs[:, 1][cal_labels == 0], 1.0 - ALPHA))
        # Symmetric Chow: one band around the single tuned threshold, widened
        # until it refuses the share the class-conditional scheme refuses on the
        # same calibration half. Comparing at a matched refusal rate is the only
        # way the two rules are comparable at all: any band can be made to look
        # better by refusing more.
        middle = 1.0 - fitted.plain
        conformal_sets = fitted.sets(cal_probs)["perlabel"]
        wanted = float((conformal_sets[:, 0] & conformal_sets[:, 1]).mean())
        distance = np.sort(np.abs(cal_probs[:, 1] - middle))
        rank = min(int(round(wanted * distance.size)), distance.size - 1)
        half = float(distance[rank])

        for name, bundle in corpora.items():
            probs, labels = bundle["probs"], bundle["labels"]
            if name == "ptbxl":
                probs, labels = probs[~is_cal], labels[~is_cal]
            built = {
                "conformal_perlabel": fitted.sets(probs)["perlabel"],
                "chow_per_class": _chow_sets(probs, mi_cut, non_mi_cut),
                "chow_symmetric": _chow_sets(probs, middle - half, middle + half),
            }
            for rule, sets in built.items():
                tally.setdefault((name, f"{rule}:mi"), []).append(_mi_coverage(sets, labels))
                tally.setdefault((name, f"{rule}:non_mi"), []).append(
                    _non_mi_coverage(sets, labels)
                )
                tally.setdefault((name, f"{rule}:deferred"), []).append(
                    float((sets[:, 0] & sets[:, 1]).mean())
                )

    rules = ("conformal_perlabel", "chow_per_class", "chow_symmetric")
    by_corpus: dict[str, Any] = {}
    for name in corpora:
        row: dict[str, Any] = {
            rule: {
                key: round(float(np.mean(tally[(name, f"{rule}:{key}")])), 4)
                for key in ("mi", "non_mi", "deferred")
            }
            for rule in rules
        }
        row["conformal_minus_chow_per_class_mi_points"] = round(
            (row["conformal_perlabel"]["mi"] - row["chow_per_class"]["mi"]) * 100, 2
        )
        by_corpus[name] = {"name": CORPUS_NAMES[name], **row}
    return {
        "note": (
            "MI coverage under class-conditional conformal calibration, under a "
            "per-class reject rule at the same empirical quantiles without the (n+1) "
            "correction, and under a symmetric reject rule around the single tuned "
            "threshold widened to refuse the same share of the calibration half."
        ),
        "n_draws": draws,
        "by_corpus": by_corpus,
    }


def correction(
    source: dict[str, NDArray[Any]], patients: NDArray[Any], draws: int, seed: int
) -> dict[str, Any]:
    """What the (n+1) buys, against the spread it has to be read beside."""
    with_correction: list[float] = []
    without: list[float] = []

    for is_cal in calibration_halves(patients, draws, seed):
        cal_probs, cal_labels = source["probs"][is_cal], source["labels"][is_cal]
        probs, labels = source["probs"][~is_cal], source["labels"][~is_cal]
        scores = lac_scores(cal_probs, cal_labels)
        conformal_q = fit_thresholds(cal_probs, cal_labels, ALPHA).perlabel
        # The plain empirical quantile of each label's scores: the same rule
        # without the finite-sample step the conformal construction adds.
        plain_q = np.array(
            [float(np.quantile(scores[cal_labels == c], 1.0 - ALPHA)) for c in (0, 1)]
        )
        all_scores = lac_scores_all(probs)
        with_correction.append(
            _mi_coverage(predict_sets_per_class(all_scores, conformal_q), labels)
        )
        without.append(_mi_coverage(predict_sets_per_class(all_scores, plain_q), labels))

    gap = np.asarray(with_correction) - np.asarray(without)
    return {
        "note": (
            "MI coverage with the ceil((n+1)(1-alpha)) rank against the plain "
            "empirical quantile of the same calibration scores, on the same draws."
        ),
        "n_draws": draws,
        "coverage_with_correction": round(float(np.mean(with_correction)), 4),
        "coverage_without_correction": round(float(np.mean(without)), 4),
        "correction_worth_points": round(float(gap.mean()) * 100, 2),
        "between_draw_sd_points": round(float(np.std(with_correction, ddof=1)) * 100, 2),
    }


def sample_size(corpora: dict[str, dict[str, NDArray[Any]]]) -> dict[str, Any]:
    """How many cases a site needs before its own coverage estimate says anything."""
    z = float(norm.ppf(0.975))
    n_mi = int(np.ceil(z**2 * 0.9 * 0.1 / TARGET_HALF_WIDTH**2))
    return {
        "note": (
            "Cases of the rarer label needed for a 95% normal interval of "
            f"+/-{TARGET_HALF_WIDTH:.2f} on a coverage near 0.90, and the tracings that "
            "implies at each site's observed MI prevalence."
        ),
        "target_half_width": TARGET_HALF_WIDTH,
        "mi_cases_needed": n_mi,
        "tracings_needed": {
            name: int(np.ceil(n_mi / float(np.mean(bundle["labels"] == 1))))
            for name, bundle in corpora.items()
        },
    }


def predictive_value(
    corpora: dict[str, dict[str, NDArray[Any]]],
    patients: NDArray[Any],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """What a positive answer is worth, which a coverage figure does not say."""
    tally: dict[str, list[float]] = {}
    for is_cal in calibration_halves(patients, draws, seed):
        source = corpora["ptbxl"]
        fitted = fit_thresholds(source["probs"][is_cal], source["labels"][is_cal], ALPHA)
        for name, bundle in corpora.items():
            probs, labels = bundle["probs"], bundle["labels"]
            if name == "ptbxl":
                probs, labels = probs[~is_cal], labels[~is_cal]
            sets = fitted.sets(probs)["perlabel"]
            flagged = sets[:, 1] & ~sets[:, 0]
            tally.setdefault(name, []).append(
                float(labels[flagged].mean()) if flagged.any() else float("nan")
            )
    return {
        "note": (
            "Among tracings given the MI label alone by class-conditional "
            "calibration, the share that carry it. Deferred tracings are not counted."
        ),
        "n_draws": draws,
        "by_corpus": {
            name: {
                "name": CORPUS_NAMES[name],
                "positive_predictive_value": round(float(np.nanmean(values)), 4),
                "prevalence": round(float(np.mean(corpora[name]["labels"] == 1)), 4),
            }
            for name, values in tally.items()
        },
    }


def discrimination(corpora: dict[str, dict[str, NDArray[Any]]], seed: int) -> dict[str, Any]:
    """The baseline model's AUROC on each corpus, with a bootstrap interval by record.

    AUROC is a property of the scores alone, so it is read on the whole of each
    corpus, fold 10 included, without any calibration draw.
    """
    out: dict[str, Any] = {}
    for name, bundle in corpora.items():
        point, low, high = bootstrap_ci(
            lambda y, p: float(roc_auc_score(y, p)),
            bundle["labels"],
            bundle["probs"][:, 1],
            n_draws=AUROC_DRAWS,
            seed=seed,
        )
        out[name] = {
            "name": CORPUS_NAMES[name],
            "auroc": round(point, 4),
            "ci95": [round(low, 4), round(high, 4)],
        }
    return {"n_bootstrap": AUROC_DRAWS, "by_corpus": out}


def site_intervals(
    corpora: dict[str, dict[str, NDArray[Any]]], patients: NDArray[Any], draws: int, seed: int
) -> dict[str, Any]:
    """The interval each target site's own MI cases support, beside the spread between draws.

    The coverage is the mean over the calibration draws; the Wilson interval is
    on that coverage over the site's MI cases, which do not change from draw to
    draw.  The spread across draws is the calibration's, a different and much
    smaller quantity.
    """
    tally: dict[tuple[str, str], list[float]] = {}
    for is_cal in calibration_halves(patients, draws, seed):
        source = corpora["ptbxl"]
        fitted = fit_thresholds(source["probs"][is_cal], source["labels"][is_cal], ALPHA)
        for name in ("sph", "acs"):
            sets = fitted.sets(corpora[name]["probs"])
            for scheme in ("pooled", "perlabel"):
                tally.setdefault((name, scheme), []).append(
                    _mi_coverage(sets[scheme], corpora[name]["labels"])
                )
    out: dict[str, Any] = {}
    for name in ("sph", "acs"):
        n_mi = int((corpora[name]["labels"] == 1).sum())
        out[name] = {"name": CORPUS_NAMES[name], "n_mi_cases": n_mi}
        for scheme in ("pooled", "perlabel"):
            observed = float(np.mean(tally[(name, scheme)]))
            low, high = wilson_interval(int(round(observed * n_mi)), n_mi)
            out[name][scheme] = {
                "coverage": round(observed, 4),
                "wilson_ci95": [round(low, 4), round(high, 4)],
                "draw_sd": round(float(np.std(tally[(name, scheme)], ddof=1)), 4),
            }
    return {
        "note": (
            "MI coverage at each target site, mean over the calibration draws, with "
            "the 95% Wilson interval its own MI cases support."
        ),
        "n_draws": draws,
        "by_corpus": out,
    }


def build(draws: int, bootstrap: int, seed: int) -> dict[str, Any]:
    started = time.time()
    corpora = {
        "ptbxl": dict(np.load(RESULTS_DIR / "baseline/scores.npz")),
        "sph": dict(np.load(RESULTS_DIR / "external/sph.npz")),
        "acs": dict(np.load(RESULTS_DIR / "external/acs.npz")),
    }
    patients = np.asarray(ptbxl_patients(corpora["ptbxl"]["ids"], PTBXL_DIR))
    return {
        "question": (
            "The infarction quantities the report argues from beside its coverage "
            "tables, computed from the committed scores."
        ),
        "alpha": ALPHA,
        "n_draws": draws,
        "seed": seed,
        "provenance": provenance_block(
            [
                "scripts/auxiliary.py",
                "src/ecs/conformal.py",
                "src/ecs/ingest.py",
                "src/ecs/metrics.py",
                "src/ecs/splits.py",
            ]
        ),
        "seconds": 0.0,
        "differences": differences(corpora, patients, bootstrap, seed),
        "chow": chow(corpora, patients, draws, seed),
        "correction": correction(corpora["ptbxl"], patients, draws, seed),
        "sample_size": sample_size(corpora),
        "predictive_value": predictive_value(corpora, patients, draws, seed),
        "site_intervals": site_intervals(corpora, patients, draws, seed),
        "discrimination": discrimination(corpora, seed),
    } | {"seconds": round(time.time() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "auxiliary.json")
    args = parser.parse_args()
    table = build(args.draws, args.bootstrap, args.seed)
    args.out.write_text(json.dumps(table, indent=2) + "\n")
    print(f"wrote {args.out}")
    for name, row in table["differences"]["by_corpus"].items():
        low, high = row["difference"]["ci95"]
        gap = row["difference"]["mean"] * 100
        print(f"  gap {name:<6s} {gap:+6.2f} [{low * 100:+6.2f},{high * 100:+6.2f}]")
    for name, row in table["chow"]["by_corpus"].items():
        for rule in ("conformal_perlabel", "chow_per_class", "chow_symmetric"):
            cell = row[rule]
            print(
                f"  chow {name:<6s} {rule:<19s} MI {cell['mi'] * 100:6.2f}  "
                f"non-MI {cell['non_mi'] * 100:6.2f}  deferred {cell['deferred'] * 100:6.2f}"
            )
    c = table["correction"]
    print(
        f"  (n+1) worth {c['correction_worth_points']:+.2f} pts, "
        f"between-draw sd {c['between_draw_sd_points']:.2f} pts"
    )
    size = table["sample_size"]
    print(f"  sample size: {size['mi_cases_needed']} MI cases -> {size['tracings_needed']}")
    for name, row in table["predictive_value"]["by_corpus"].items():
        ppv = row["positive_predictive_value"] * 100
        print(f"  ppv {name:<6s} {ppv:6.2f}  (prevalence {row['prevalence'] * 100:.1f}%)")
    for name, row in table["site_intervals"]["by_corpus"].items():
        for scheme in ("pooled", "perlabel"):
            cell = row[scheme]
            print(
                f"  {name:<4s} {scheme:<9s} MI {cell['coverage']:.4f} Wilson {cell['wilson_ci95']}"
            )


if __name__ == "__main__":
    main()
