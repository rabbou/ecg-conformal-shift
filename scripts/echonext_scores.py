"""Score every validation and test ECG of EchoNext with one arm, on the twelve labels.

Arms that run here:

``resnet``       the study's ResNet1d with twelve outputs, trained from scratch
                 on the train split (a tenth of its patients held out to pick
                 the epoch), input the stored z-scored tracing at 250 Hz.
``random_init``  the study's ResNet1d frozen at its seeded random
                 initialisation, one logistic probe per label on its 256-d
                 vectors, fitted on the train split.  It is the floor a
                 pre-trained encoder has to clear, and it runs the whole
                 frozen-encoder path: tracing digest, embedding store, probe.

Neither calibration nor any threshold is touched here: the validation and test
records are only scored.  Scores are derived restricted data and go to
``$ECS_ECHONEXT_DERIVED/scores/<arm>.npz`` with a ``.json`` describing the run.

Usage: PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/echonext_scores.py --arm resnet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ecs.echonext import (
    DERIVED_DIR,
    ECHONEXT_DIR,
    LABELS,
    UNIT,
    canonical,
    read_metadata,
    read_rows,
    transfer_cohorts,
    waveform_path,
)
from ecs.embedding_store import EmbeddingStore
from ecs.encoders import RANDOM_INIT_SEED, machine_info
from ecs.models import ResNet1d
from ecs.splits import patient_split

SCORES_DIR = DERIVED_DIR / "scores"
SCORED_SPLITS = ("val", "test")
BATCH = 128
EPOCHS = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HOLDOUT_SHARE = 0.1
SEED = 0
# Written after every epoch, so a run cut short resumes at the next epoch.
CHECKPOINT = DERIVED_DIR / "models/resnet-training.pt"
RANDOM_INIT_VERSION = f"resnet1d-seed{RANDOM_INIT_SEED}-echonext-250hz-zscore"

Float32Array = NDArray[np.float32]


def device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class Tracings:
    """Canonical float32 tracings for any metadata positions, read from the mapped files."""

    def __init__(self, meta: pd.DataFrame, root: Path = ECHONEXT_DIR) -> None:
        self.split = meta["split"].to_numpy()
        self.row = meta["row"].to_numpy()
        self.paths = {s: waveform_path(root, s) for s in ("train", "val", "test")}

    def __call__(self, positions: NDArray[np.int_]) -> Float32Array:
        out = np.empty((len(positions), 12, 2500), dtype=np.float32)
        for split in np.unique(self.split[positions]):
            inside = np.flatnonzero(self.split[positions] == split)
            rows = self.row[positions[inside]]
            out[inside] = canonical(read_rows(self.paths[split], rows))
        return out


def scored_positions(meta: pd.DataFrame) -> NDArray[np.int_]:
    return np.flatnonzero(meta["split"].isin(SCORED_SPLITS).to_numpy())


def labels_of(meta: pd.DataFrame, positions: NDArray[np.int_]) -> Float32Array:
    return meta[list(LABELS)].to_numpy(dtype=np.float32)[positions]


def mean_auroc(y: Float32Array, p: Float32Array) -> float:
    return float(np.mean([roc_auc_score(y[:, k], p[:, k]) for k in range(y.shape[1])]))


def predict(model: ResNet1d, read: Tracings, positions: NDArray[np.int_]) -> Float32Array:
    dev = device()
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(positions), 512):
            x = torch.from_numpy(read(positions[start : start + 512])).to(dev)
            out.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(out)


def train_resnet(meta: pd.DataFrame, read: Tracings) -> tuple[Float32Array, dict[str, Any]]:
    cohorts = transfer_cohorts(meta)
    train = cohorts.training
    part = patient_split(
        meta["patient_key"].iloc[train], {"fit": 1 - HOLDOUT_SHARE, "holdout": HOLDOUT_SHARE}, SEED
    ).to_numpy()
    fit, holdout = train[part == "fit"], train[part == "holdout"]
    y_fit, y_hold = labels_of(meta, fit), labels_of(meta, holdout)

    torch.manual_seed(SEED)
    dev = device()
    model = ResNet1d(n_classes=len(LABELS)).to(dev)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history: list[dict[str, Any]] = []
    best, best_state = -1.0, None
    if CHECKPOINT.exists():
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        best_state = saved["best"]
        history = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        best = max(h["holdout_mean_auroc"] for h in history)
    for epoch in range(len(history), EPOCHS):
        started = time.time()
        model.train()
        order = np.random.default_rng(SEED + epoch).permutation(len(fit))
        losses = []
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            x = torch.from_numpy(read(fit[batch])).to(dev)
            y = torch.from_numpy(y_fit[batch]).to(dev)
            optimiser.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            loss.backward()
            optimiser.step()
            losses.append(float(loss))
        score = mean_auroc(y_hold, predict(model, read, holdout))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "holdout_mean_auroc": score,
                "seconds": round(time.time() - started, 1),
            }
        )
        print(history[-1], flush=True)
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "optimiser": optimiser.state_dict(), "best": best_state},
            CHECKPOINT,
        )
        CHECKPOINT.with_suffix(".json").write_text(json.dumps(history, indent=2) + "\n")
    if best_state is None:
        raise RuntimeError("no epoch was trained")
    model.load_state_dict(best_state)
    torch.save(best_state, DERIVED_DIR / "models/resnet.pt")
    info = {
        "arm": "resnet",
        "description": "study ResNet1d, 12 sigmoid outputs, trained from scratch on EchoNext train",
        "input": f"(12, 2500) at 250 Hz, unit {UNIT}, as stored",
        "fit_records": len(fit),
        "holdout_records": len(holdout),
        "epochs": EPOCHS,
        "batch": BATCH,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "history": history,
        "chosen_epoch": int(np.argmax([h["holdout_mean_auroc"] for h in history]) + 1),
    }
    return predict(model, read, scored_positions(meta)), info


def random_init_probe(meta: pd.DataFrame, read: Tracings) -> tuple[Float32Array, dict[str, Any]]:
    provenance = pd.read_csv(
        DERIVED_DIR / "provenance.csv.gz", usecols=["ecg_key", "tracing_sha256"]
    )
    digest_of = dict(zip(provenance["ecg_key"], provenance["tracing_sha256"], strict=True))
    torch.manual_seed(RANDOM_INIT_SEED)
    dev = device()
    model = ResNet1d().eval().to(dev)
    store = EmbeddingStore("random_init", RANDOM_INIT_VERSION)
    calls = 0

    def vectors(positions: NDArray[np.int_]) -> Float32Array:
        digests = [digest_of[k] for k in meta["ecg_key"].to_numpy()[positions]]

        def compute(inner: NDArray[np.int_]) -> Float32Array:
            nonlocal calls
            calls += 1
            out = []
            with torch.no_grad():
                for start in range(0, len(inner), 512):
                    x = torch.from_numpy(read(positions[inner[start : start + 512]])).to(dev)
                    out.append(model.embed(x).float().cpu().numpy())
            return np.concatenate(out)

        return store.embed(digests, compute)

    started = time.time()
    train = transfer_cohorts(meta).training
    x_train = vectors(train)
    scored = scored_positions(meta)
    x_scored = vectors(scored)
    embed_seconds = round(time.time() - started, 1)

    scaler = StandardScaler().fit(x_train)
    y_train = labels_of(meta, train)
    probs = np.empty((len(scored), len(LABELS)), dtype=np.float32)
    for k in range(len(LABELS)):
        probe = LogisticRegression(max_iter=2000).fit(scaler.transform(x_train), y_train[:, k])
        probs[:, k] = probe.predict_proba(scaler.transform(x_scored))[:, 1]
    info = {
        "arm": "random_init",
        "description": "study ResNet1d frozen at its seeded initialisation, one probe per label",
        "encoder_version": RANDOM_INIT_VERSION,
        "input": f"(12, 2500) at 250 Hz, unit {UNIT}, as stored",
        "probe": "sklearn LogisticRegression(max_iter=2000), StandardScaler fitted on train",
        "train_records": len(train),
        "embedding_seconds": embed_seconds,
        "encoder_batches_computed": calls,
        "store_size": len(store),
    }
    return probs, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("resnet", "random_init"), required=True)
    args = parser.parse_args()
    meta = read_metadata()
    read = Tracings(meta)
    started = time.time()
    probs, info = (train_resnet if args.arm == "resnet" else random_init_probe)(meta, read)
    info |= {
        "seconds": round(time.time() - started, 1),
        "device": str(device()),
        "machine": machine_info(),
    }
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    keys = meta["ecg_key"].to_numpy()[scored_positions(meta)]
    np.savez(SCORES_DIR / f"{args.arm}.npz", ecg_key=keys, probs=probs, labels=np.array(LABELS))
    (SCORES_DIR / f"{args.arm}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "machine"}, indent=2))


if __name__ == "__main__":
    main()
