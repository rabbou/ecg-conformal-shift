"""The supervised baseline: our own ResNet1d saying "infarction or not" on PTB-XL.

This is the witness the rest of the week is read against.  If the coverage
guarantee breaks on another hospital's tracings, the first question is whether
the model was ever any good; a baseline that reproduces the published benchmark
answers it, and one that does not means the break has to be explained before it
is believed (C-19).

The split is the one PTB-XL ships for benchmarking, so the number is comparable
to the published one: folds 1-8 to train, fold 9 to decide when to stop, fold 10
scored once at the end and never looked at before.  Everything that could move
the number -- the seed, the learning rate, the batch size, the epoch budget, the
label definition, the standardisation -- is written to
``results/baseline/config.json`` beside the result.

Four files come out of a run, under ``results/baseline/``:

``config.json``   what was run, including the standardisation the training fold
                  fixed and the label spec;
``train.log``     one line per epoch: losses, validation AUROC, seconds;
``metrics.json``  fold-10 AUROC and AUPRC with 95% bootstrap intervals, the
                  epoch that was kept, the duration;
``scores.npz``    the fold-10 record identifiers, their true labels, and the
                  model's probability per class -- what conformal prediction
                  consumes next.

The checkpoint goes to ``model.pt`` beside them and is not committed.

Usage: .venv/bin/python scripts/train_baseline.py [--epochs 15] [--max-records N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn

from ecs.config import PTBXL_DIR, RESULTS_DIR
from ecs.encoders import machine_info
from ecs.ingest import load_ptbxl
from ecs.labels import MILabelSpec, ptbxl_mi_label
from ecs.metrics import bootstrap_ci
from ecs.models import ResNet1d
from ecs.splits import ptbxl_benchmark_split

CHUNK = 500  # records read from disk at once, so no fold is read twice


@dataclass(frozen=True)
class Settings:
    """Everything that changes the number, in one place and written out with it."""

    epochs: int = 15
    learning_rate: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-2
    patience: int = 3  # epochs without a better validation AUROC before stopping
    seed: int = 0
    bootstrap_draws: int = 1000


@dataclass
class Fold:
    """One part of the split, held in memory in canonical form."""

    name: str
    ids: list[str]
    x: NDArray[np.float32]
    y: NDArray[np.int_]


def read_fold(
    database: pd.DataFrame,
    labels: pd.Series,
    name: str,
    wanted: pd.Index,
    root: Path = PTBXL_DIR,
) -> Fold:
    """The records of one part, read in chunks so the corpus is never held twice."""
    ids: list[str] = []
    blocks: list[NDArray[np.float32]] = []
    for start in range(0, len(wanted), CHUNK):
        chunk = load_ptbxl(database, root=root, ids=[int(i) for i in wanted[start : start + CHUNK]])
        ids.extend(chunk.ids)
        blocks.append(chunk.x)
        print(f"  {name}: {len(ids):>6} / {len(wanted)}", flush=True)
    x = np.concatenate(blocks) if blocks else np.empty((0, 12, 5000), dtype=np.float32)
    y = labels.loc[[int(i) for i in ids]].to_numpy().astype(int)
    return Fold(name, ids, x, y)


def standardisation(x: NDArray[np.float32]) -> tuple[float, float]:
    """One mean and one spread, taken from the training fold only.

    Per-lead statistics were the alternative; one pair over all leads keeps the
    relative amplitude between leads, which is part of what an infarct pattern
    is, rather than flattening it away.
    """
    return float(x.mean()), float(x.std()) or 1.0


def probabilities(model: nn.Module, x: NDArray[np.float32], centre: float, scale: float) -> Tensor:
    """Class probabilities for every row, in batches, in the same order."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 128):
            batch = torch.from_numpy((x[i : i + 128] - centre) / scale)
            out.append(torch.softmax(model(batch), dim=1))
    return torch.cat(out) if out else torch.empty((0, 2))


def train(
    train_fold: Fold, validation: Fold, settings: Settings, log: Path
) -> tuple[nn.Module, float, float, int]:
    """The model at its best validation AUROC, with the standardisation it was
    trained under and the epoch that produced it."""
    torch.manual_seed(settings.seed)
    model = ResNet1d(n_classes=2)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    loss_of = nn.CrossEntropyLoss()
    centre, scale = standardisation(train_fold.x)
    rng = np.random.default_rng(settings.seed)

    best_auroc, best_epoch = -1.0, 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    with log.open("w") as journal:
        for epoch in range(1, settings.epochs + 1):
            started = time.perf_counter()
            model.train()
            order = rng.permutation(len(train_fold.x))
            total = 0.0
            for i in range(0, len(order), settings.batch_size):
                index = order[i : i + settings.batch_size]
                batch = torch.from_numpy((train_fold.x[index] - centre) / scale)
                target = torch.from_numpy(train_fold.y[index]).long()
                optimiser.zero_grad()
                loss = loss_of(model(batch), target)
                loss.backward()
                optimiser.step()
                total += float(loss) * len(index)
            scores = probabilities(model, validation.x, centre, scale)[:, 1].numpy()
            auroc = float(roc_auc_score(validation.y, scores))
            line = (
                f"epoch {epoch:>2}  train loss {total / len(order):.4f}  "
                f"validation AUROC {auroc:.4f}  {time.perf_counter() - started:.0f} s"
            )
            print(line, flush=True)
            journal.write(line + "\n")
            journal.flush()
            if auroc > best_auroc:
                best_auroc, best_epoch = auroc, epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= settings.patience:
                stop = f"stopped: {settings.patience} epochs without a better validation AUROC"
                print(stop, flush=True)
                journal.write(stop + "\n")
                break
    model.load_state_dict(best_state)
    return model, centre, scale, best_epoch


@dataclass
class Scored:
    """What fold 10 gave: one probability row per record, and the numbers on top."""

    probs: NDArray[np.float64]
    metrics: dict[str, object]


def score(model: nn.Module, test: Fold, centre: float, scale: float, settings: Settings) -> Scored:
    """Fold 10, scored once, with the interval that makes the number arguable."""
    probs: NDArray[np.float64] = probabilities(model, test.x, centre, scale).numpy()
    auroc, auroc_low, auroc_high = bootstrap_ci(
        lambda y, s: float(roc_auc_score(y, s)),
        test.y,
        probs[:, 1],
        n_draws=settings.bootstrap_draws,
        seed=settings.seed,
    )
    auprc, auprc_low, auprc_high = bootstrap_ci(
        lambda y, s: float(average_precision_score(y, s)),
        test.y,
        probs[:, 1],
        n_draws=settings.bootstrap_draws,
        seed=settings.seed,
    )
    return Scored(
        probs,
        {
            "auroc": auroc,
            "auroc_ci95": [auroc_low, auroc_high],
            "auprc": auprc,
            "auprc_ci95": [auprc_low, auprc_high],
            "n_test": len(test.y),
            "n_test_positive": int(test.y.sum()),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=Settings.epochs)
    parser.add_argument("--seed", type=int, default=Settings.seed)
    parser.add_argument("--batch-size", type=int, default=Settings.batch_size)
    parser.add_argument("--learning-rate", type=float, default=Settings.learning_rate)
    parser.add_argument("--bootstrap-draws", type=int, default=Settings.bootstrap_draws)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="use only the first N records of each fold; for checking the run, not for a result",
    )
    parser.add_argument("--out", default=str(RESULTS_DIR / "baseline"))
    parser.add_argument("--ptbxl-dir", default=str(PTBXL_DIR))
    args = parser.parse_args(argv)
    settings = Settings(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ptbxl = Path(args.ptbxl_dir)
    database = pd.read_csv(ptbxl / "ptbxl_database.csv", index_col="ecg_id")
    statements = pd.read_csv(ptbxl / "scp_statements.csv", index_col=0)
    spec = MILabelSpec()
    labels = ptbxl_mi_label(database, statements, spec)
    part = ptbxl_benchmark_split(database)

    started = time.perf_counter()
    folds = {}
    for name in ("train", "validation", "test"):
        wanted = part.index[part == name]
        if args.max_records is not None:
            wanted = wanted[: args.max_records]
        folds[name] = read_fold(database, labels, name, wanted, ptbxl)
        print(
            f"{name}: {len(folds[name].ids)} records, {int(folds[name].y.sum())} MI",
            flush=True,
        )

    model, centre, scale, best_epoch = train(
        folds["train"], folds["validation"], settings, out / "train.log"
    )
    result = score(model, folds["test"], centre, scale, settings)
    seconds = round(time.perf_counter() - started, 1)

    (out / "config.json").write_text(
        json.dumps(
            {
                **asdict(settings),
                "label_spec": asdict(spec),
                "split": "PTB-XL strat_fold: 1-8 train, 9 validation, 10 test",
                "standardisation": {"centre_mv": centre, "scale_mv": scale},
                "n_train": len(folds["train"].ids),
                "n_validation": len(folds["validation"].ids),
                "n_test": len(folds["test"].ids),
                "max_records": args.max_records,
                "machine": machine_info(),
            },
            indent=2,
        )
        + "\n"
    )
    (out / "metrics.json").write_text(
        json.dumps(
            {**result.metrics, "epoch_kept": best_epoch, "seconds": seconds},
            indent=2,
        )
        + "\n"
    )
    np.savez(
        out / "scores.npz",
        ids=np.asarray(folds["test"].ids, dtype=str),
        labels=folds["test"].y,
        probs=result.probs,
    )
    torch.save(
        {"state_dict": model.state_dict(), "centre": centre, "scale": scale},
        out / "model.pt",
    )
    print(json.dumps(result.metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
