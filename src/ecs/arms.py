"""One linear head per encoder arm, and the paired test that compares two of them.

The four arms differ in their weights and in nothing else: the same tracings go
through each, the representations are cached once (``results/embeddings/``), and
what happens after the encoder is identical.  That is what makes the difference
between two arms readable as a difference between two pre-trainings rather than
between two pipelines.

**The head.**  The encoder is frozen and a linear probe is trained on its
output.  This is the convergent protocol for evaluating a frozen biosignal
encoder -- standardise the embedding, fit an L2-regularised logistic regression,
choose the regularisation strength on a validation split (arXiv:2601.21830,
"Looking Beyond Accuracy: A Holistic Benchmark of ECG Foundation Models", and
arXiv:2509.25095, "Benchmarking ECG FMs", both read 2026-08-26).  A deeper head
would measure the head; a linear one measures the representation, which is the
question.

**The split is the baseline's, not a new one.**  PTB-XL folds 1-8 fit the
standardisation and the probe, fold 9 chooses ``C``, fold 10 is scored once.
Fold 9 is where the supervised baseline stops its training, so the arms and the
baseline consume the same folds in the same roles and no arm sees a record the
baseline was allowed to see and it was not.

**Comparing two arms.**  Two AUROCs a hundredth apart with overlapping intervals
are not a difference, and two *independent* intervals overlapping does not mean
the difference is zero either -- the arms scored the same records, so their
errors are correlated and the paired difference is far tighter than the two
intervals suggest.  So an arm-versus-arm claim is made on a paired bootstrap:
one set of resampled record indices per draw, applied to both arms, difference
taken inside the draw (C-18).  DeLong's test is the closed-form standard for
paired AUROC but tests AUROC only; the paired bootstrap covers AUPRC on the same
footing, so both metrics are compared the same way here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]

__all__ = [
    "REGULARISATION_GRID",
    "Head",
    "Probe",
    "auprc",
    "auroc",
    "fit_probe",
    "fitting_statistics",
    "paired_difference",
    "standardise",
]

# The regularisation strengths fold 9 chooses between.  Logarithmic and wide,
# because the arms' embeddings differ by a factor of four in width (256 for the
# random ResNet1d, 1024 for ECGFounder) and the strength that suits one need not
# suit another; fixing one value for all arms would hand the widest embedding a
# handicap that would read as a fact about its pre-training.
REGULARISATION_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

# The probe's optimiser is deterministic (lbfgs on a convex objective), so the
# seed does not move the fit; it is carried anyway so that anything downstream
# that does resample has one place to read it from.
MAX_ITER = 2000


@dataclass(frozen=True)
class Head:
    """A fitted probe and the standardisation it was fitted under.

    Both come from PTB-XL folds 1-8 only.  Keeping them together is what stops a
    target corpus from being scored under statistics computed on itself, which
    would quietly re-calibrate the arm on the corpus it is being tested against.
    """

    model: LogisticRegression
    centre: Array
    scale: Array
    c: float

    def probabilities(self, embeddings: Array) -> Array:
        """One probability row per record, in the order given."""
        z = standardise(np.asarray(embeddings, dtype=np.float64), self.centre, self.scale)
        return np.asarray(self.model.predict_proba(z), dtype=np.float64)


@dataclass(frozen=True)
class Probe:
    """A fitted head with the validation trace that chose its strength."""

    head: Head
    validation_auroc: dict[float, float]

    @property
    def chosen(self) -> float:
        return self.head.c


def standardise(x: Array, centre: Array, scale: Array) -> Array:
    """``x`` centred and scaled per dimension, by statistics fitted elsewhere."""
    return (x - centre) / scale


def fitting_statistics(x: Array) -> tuple[Array, Array]:
    """Per-dimension mean and spread, with dead dimensions left alone.

    A dimension that is constant across the training fold has zero spread, and
    dividing by it would produce a NaN that propagates through the whole row.
    Such a dimension carries no signal, so it is scaled by one and centred to
    zero, which is the same as dropping it without changing the matrix width.
    """
    centre = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0.0] = 1.0
    return centre, scale


def fit_probe(
    train_x: Array,
    train_y: IntArray,
    validation_x: Array,
    validation_y: IntArray,
    grid: Sequence[float] = REGULARISATION_GRID,
    seed: int = 0,
) -> Probe:
    """The probe at the strength fold 9 preferred, and what every strength scored.

    The standardisation is fitted on ``train_x`` alone and applied unchanged to
    the validation fold, exactly as it will later be applied to fold 10 and to
    the external corpora.
    """
    train_x = np.asarray(train_x, dtype=np.float64)
    validation_x = np.asarray(validation_x, dtype=np.float64)
    centre, scale = fitting_statistics(train_x)
    z_train = standardise(train_x, centre, scale)
    z_validation = standardise(validation_x, centre, scale)

    scored: dict[float, float] = {}
    best: tuple[float, LogisticRegression, float] | None = None
    for c in grid:
        model = LogisticRegression(C=c, max_iter=MAX_ITER, random_state=seed)
        model.fit(z_train, train_y)
        value = auroc(validation_y, model.predict_proba(z_validation)[:, 1])
        scored[c] = value
        if best is None or value > best[0]:
            best = (value, model, c)
    if best is None:
        raise ValueError("the regularisation grid is empty")
    return Probe(Head(best[1], centre, scale, best[2]), scored)


def auroc(labels: IntArray, scores: Array) -> float:
    return float(roc_auc_score(labels, scores))


def auprc(labels: IntArray, scores: Array) -> float:
    return float(average_precision_score(labels, scores))


def paired_difference(
    statistic: Callable[[IntArray, Array], float],
    labels: IntArray,
    left: Array,
    right: Array,
    n_draws: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """``statistic(left) - statistic(right)`` with a paired bootstrap interval.

    The two score vectors describe the same records in the same order, so each
    draw resamples record *indices* once and scores both arms on that one
    resample.  Whatever the draw does to the difficulty of the sample, it does
    to both arms, and the difference is left.  Resampling the two arms
    independently would add a variance that is not in the comparison and would
    widen the interval until no pair of arms ever differed.

    A draw holding one class only leaves AUROC undefined for both arms and is
    dropped; as in ``metrics.bootstrap_ci`` that is tolerated only while it
    stays rare, and an interval is refused rather than computed on the remainder
    when it does not.

    Returns the observed difference, the interval, and the share of draws on the
    same side as it -- which is the number that says whether the two arms are
    separated, without asking anyone to eyeball two overlapping intervals.
    """
    labels = np.asarray(labels)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if not len(labels) == len(left) == len(right):
        raise ValueError(
            f"paired comparison needs one label per score on both arms: "
            f"{len(labels)} labels, {len(left)} and {len(right)} scores"
        )
    rng = np.random.default_rng(seed)
    drawn: list[float] = []
    for _ in range(n_draws):
        index = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[index])) < 2:
            continue
        drawn.append(statistic(labels[index], left[index]) - statistic(labels[index], right[index]))
    if len(drawn) < 0.95 * n_draws:
        raise ValueError(
            f"only {len(drawn)} of {n_draws} draws held both classes; "
            f"the sample of {len(labels)} is too small for a paired interval"
        )
    observed = statistic(labels, left) - statistic(labels, right)
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(drawn, [100 * tail, 100 * (1 - tail)])
    array = np.asarray(drawn, dtype=np.float64)
    agreeing = float((array > 0).mean()) if observed > 0 else float((array < 0).mean())
    return {
        "difference": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "separated": bool(low > 0.0 or high < 0.0),
        "share_of_draws_on_the_same_side": agreeing,
        "n_draws": len(drawn),
        "n_points": int(len(labels)),
    }
