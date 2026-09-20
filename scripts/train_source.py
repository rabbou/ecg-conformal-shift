"""One model per source corpus: the same ResNet, five diagnoses, five hospitals.

The rotation needs a model that belongs to each source, so that "spend this
source's thresholds elsewhere" means something.  This script trains one, on one
corpus, and writes it beside what it took to get there.

The architecture, the optimiser, the schedule, the standardisation and the
label set are identical across the five runs, and the training set is capped at
a size every corpus can reach, so the only thing that differs between two runs
is the hospital the records came from.  The head has five sigmoid outputs, one
per diagnosis, because an ECG carries several at once: a record can be sinus
rhythm and right bundle-branch block together.

Four files per source, under ``results/rotation/<source>/``:

``config.json``   everything that moves the number, including the split sizes,
                  the standardisation the training part fixed, and the machine;
``train.log``     one line per epoch;
``metrics.json``  validation AUROC per class, the epoch kept, the duration;
``model.pt``      the checkpoint, not committed.

Usage: .venv/bin/python scripts/train_source.py --source georgia [--epochs 8]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from torch import nn

from ecs.config import N_LEADS, RESULTS_DIR, WINDOW_SAMPLES
from ecs.encoders import machine_info
from ecs.models import ResNet1d
from ecs.rotation import (
    CAL_CAP,
    SOURCES,
    TEST_CAP,
    TRAIN_CAP,
    VAL_CAP,
    CorpusIndex,
    class_keys,
    corpus_index,
    load_waveforms,
    usable_classes,
)

CHUNK = 500

# The name a checkpoint carries while its run is still going. A run killed
# mid-training -- the memory guard on a shared machine does that -- leaves this
# behind and no model.pt, so nothing downstream mistakes an interrupted run for
# a finished one.
PARTIAL = "model.pt.partial"


@dataclass(frozen=True)
class Settings:
    """Everything that changes the number, in one place and written out with it."""

    epochs: int = 8
    learning_rate: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-2
    patience: int = 3
    seed: int = 0


def read_part(
    index: CorpusIndex, part: str, scratch: Path
) -> tuple[NDArray[np.float32], NDArray[np.int_], list[str]]:
    """One part of a corpus in canonical form, with its labels, read in chunks.

    The array is written once to ``scratch`` and read back through a memory map.
    The values are the same float32 the chunks produced, so nothing about the
    training changes; what changes is that a training part of three and a half
    thousand tracings stops holding eight hundred megabytes resident on a machine
    several sessions share, and the kernel may drop the pages instead. The file
    belongs to this run and is deleted with it, so no transformed waveform
    outlives the process (C-14b).
    """
    wanted = index.ids(part)
    kept: list[str] = []
    block_of: NDArray[np.float32] | None = None
    for start in range(0, len(wanted), CHUNK):
        block, ids = load_waveforms(index, wanted[start : start + CHUNK])
        if block_of is None:
            block_of = np.lib.format.open_memmap(
                scratch,
                mode="w+",
                dtype=np.float32,
                shape=(len(wanted), N_LEADS, WINDOW_SAMPLES),
            )
        block_of[len(kept) : len(kept) + len(ids)] = block
        kept.extend(ids)
        print(f"  {index.name}/{part}: {len(kept):>6} / {len(wanted)}", flush=True)
    if block_of is None:
        return np.empty((0, N_LEADS, WINDOW_SAMPLES), np.float32), index.labels([]), []
    # open_memmap returns a memmap, whose flush() the ndarray stub does not know.
    block_of.flush()  # type: ignore[attr-defined]
    del block_of
    x = np.lib.format.open_memmap(scratch, mode="r")[: len(kept)]
    return x, index.labels(kept), kept


def standardisation(x: NDArray[np.float32]) -> tuple[float, float]:
    """One mean and one spread, from the training part of this source alone."""
    return float(x.mean()), float(x.std()) or 1.0


def probabilities(
    model: nn.Module, x: NDArray[np.float32], centre: float, scale: float, batch: int = 64
) -> NDArray[np.float64]:
    """Per-class probability for every row, in the same order."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            block = torch.from_numpy((x[i : i + batch] - centre) / scale)
            out.append(torch.sigmoid(model(block)))
    if not out:
        return np.empty((0, len(class_keys())), dtype=np.float64)
    return torch.cat(out).numpy().astype(np.float64)


def per_class_auroc(
    y: NDArray[np.int_], p: NDArray[np.float64], keys: list[str]
) -> dict[str, float]:
    """AUROC of each class that has both labels present; the others are absent."""
    out: dict[str, float] = {}
    for i, key in enumerate(class_keys()):
        if key not in keys:
            continue
        column = y[:, i]
        if column.min() == column.max():
            continue
        out[key] = float(roc_auc_score(column, p[:, i]))
    return out


def train(source: str, settings: Settings, results: Path) -> dict[str, Any]:
    started = time.time()
    torch.manual_seed(settings.seed)

    index = corpus_index(source)
    keys = usable_classes(source)
    scratch = Path(tempfile.mkdtemp(prefix="ecs-train-"))
    x_train, y_train, ids_train = read_part(index, "train", scratch / "train.npy")
    x_val, y_val, ids_val = read_part(index, "val", scratch / "val.npy")
    centre, scale = standardisation(x_train)

    model = ResNet1d(n_classes=len(class_keys()))
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    # Only the classes this corpus can carry contribute to the loss: a refused
    # class has no labels here, and training the head on all-zero targets would
    # teach the model that the diagnosis never happens.
    mask = torch.tensor([1.0 if key in keys else 0.0 for key in class_keys()])
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    directory = results / "rotation" / source
    directory.mkdir(parents=True, exist_ok=True)
    log = (directory / "train.log").open("w")
    best = -1.0
    best_epoch = -1
    since_best = 0
    order = np.arange(len(x_train))
    rng = np.random.default_rng(settings.seed)
    history: list[dict[str, Any]] = []

    for epoch in range(settings.epochs):
        epoch_started = time.time()
        model.train()
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), settings.batch_size):
            rows = order[start : start + settings.batch_size]
            batch = torch.from_numpy((x_train[rows] - centre) / scale)
            target = torch.from_numpy(y_train[rows]).float()
            optimiser.zero_grad()
            loss = (loss_fn(model(batch), target) * mask).sum() / (mask.sum() * len(rows))
            loss.backward()
            optimiser.step()
            total += float(loss) * len(rows)
        p_val = probabilities(model, x_val, centre, scale, settings.batch_size)
        auroc = per_class_auroc(y_val, p_val, keys)
        macro = float(np.mean(list(auroc.values()))) if auroc else float("nan")
        row = {
            "epoch": epoch,
            "train_loss": round(total / max(len(order), 1), 5),
            "val_auroc": {k: round(v, 4) for k, v in auroc.items()},
            "val_macro_auroc": round(macro, 4),
            "seconds": round(time.time() - epoch_started, 1),
        }
        history.append(row)
        line = json.dumps(row)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        if macro > best:
            best, best_epoch, since_best = macro, epoch, 0
            torch.save(model.state_dict(), directory / PARTIAL)
        else:
            since_best += 1
            if since_best >= settings.patience:
                break
    log.close()

    config = {
        **asdict(settings),
        "source": source,
        "classes": class_keys(),
        "usable_classes": keys,
        "caps": {"train": TRAIN_CAP, "val": VAL_CAP, "cal": CAL_CAP, "test": TEST_CAP},
        "split_sizes": {
            part: int((index.frame["part"] == part).sum())
            for part in ("train", "val", "cal", "test", "unused")
        },
        "n_train": len(ids_train),
        "n_validation": len(ids_val),
        "standardisation": {"centre_mv": centre, "scale_mv": scale},
        "deviations": index.deviations,
        "machine": machine_info(),
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    metrics = {
        "epoch_kept": best_epoch,
        "val_macro_auroc": round(best, 4),
        "history": history,
        "seconds": round(time.time() - started, 1),
    }
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    # Last, and only now: a directory holding model.pt holds a finished run.
    (directory / PARTIAL).replace(directory / "model.pt")
    print(
        f"{source}: kept epoch {best_epoch}, macro AUROC {best:.4f}, {metrics['seconds']:.0f} s",
        flush=True,
    )
    del x_train, x_val
    shutil.rmtree(scratch, ignore_errors=True)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=list(SOURCES))
    parser.add_argument("--epochs", type=int, default=Settings.epochs)
    parser.add_argument("--threads", type=int, default=0, help="torch intra-op threads")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    train(args.source, Settings(epochs=args.epochs), Path(RESULTS_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
