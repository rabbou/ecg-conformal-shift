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
``ecgfounder``   ECGFounder's published 12-lead Net1D, frozen, the same probes on
                 its 1024-d vectors.  The tracing is brought from 250 to 500 Hz
                 (the rate its config.json asks for) and keeps its z-score unit.
``echonext_mini`` the published Columbia mini-model on its own weights, with the
                 seven tabular features of the distribution, nothing refitted.

Neither calibration nor any threshold is touched here: the validation and test
records are only scored.  Scores are derived restricted data and go to
``$ECS_ECHONEXT_DERIVED/scores/<arm>.npz`` with a ``.json`` describing the run.

Usage: PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/echonext_scores.py --arm resnet
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.signal import resample_poly
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
from ecs.echonext_mini import EchoNextMini
from ecs.embedding_store import EmbeddingStore
from ecs.encoders import (
    RANDOM_INIT_SEED,
    allowlisted_load,
    ecgfounder_net,
    machine_info,
    verified,
)
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
ECGFOUNDER_VERSION = "ecgfounder-12lead-echonext-250to500hz-zscore"
ECGFOUNDER_BATCH = 64
ARM_NAMES = ("resnet", "random_init", "ecgfounder", "echonext_mini")

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


Embed = Callable[[torch.Tensor], torch.Tensor]


def frozen_vectors(
    meta: pd.DataFrame,
    read: Tracings,
    store: EmbeddingStore,
    embed: Embed,
    batch: int,
) -> tuple[Callable[[NDArray[np.int_]], Float32Array], Callable[[], int]]:
    """A function from metadata positions to filed vectors, and the count of encoder passes."""
    provenance = pd.read_csv(
        DERIVED_DIR / "provenance.csv.gz", usecols=["ecg_key", "tracing_sha256"]
    )
    digest_of = dict(zip(provenance["ecg_key"], provenance["tracing_sha256"], strict=True))
    dev = device()
    calls = 0

    def vectors(positions: NDArray[np.int_]) -> Float32Array:
        digests = [digest_of[k] for k in meta["ecg_key"].to_numpy()[positions]]

        def compute(inner: NDArray[np.int_]) -> Float32Array:
            nonlocal calls
            calls += 1
            return embed_positions(read, positions[inner], embed, batch, dev)

        return store.embed(digests, compute)

    return vectors, lambda: calls


def embed_positions(
    read: Tracings, positions: NDArray[np.int_], embed: Embed, batch: int, dev: torch.device
) -> Float32Array:
    out = []
    with torch.no_grad():
        for start in range(0, len(positions), batch):
            x = torch.from_numpy(read(positions[start : start + batch])).to(dev)
            out.append(embed(x).float().cpu().numpy())
    return np.concatenate(out)


def fit_probes(
    x_train: Float32Array, y_train: Float32Array, x_scored: Float32Array
) -> Float32Array:
    """One logistic probe per label on standardised vectors; probabilities for x_scored."""
    scaler = StandardScaler().fit(x_train)
    train, scored = scaler.transform(x_train), scaler.transform(x_scored)
    probs = np.empty((len(x_scored), y_train.shape[1]), dtype=np.float32)
    for k in range(y_train.shape[1]):
        probe = LogisticRegression(max_iter=2000).fit(train, y_train[:, k])
        probs[:, k] = probe.predict_proba(scored)[:, 1]
    return probs


def probe_arm(
    meta: pd.DataFrame, read: Tracings, store: EmbeddingStore, embed: Embed, batch: int
) -> tuple[Float32Array, dict[str, Any]]:
    vectors, calls = frozen_vectors(meta, read, store, embed, batch)
    started = time.time()
    train = transfer_cohorts(meta).training
    x_train = vectors(train)
    x_scored = vectors(scored_positions(meta))
    embed_seconds = round(time.time() - started, 1)
    started = time.time()
    probs = fit_probes(x_train, labels_of(meta, train), x_scored)
    info = {
        "probe": "sklearn LogisticRegression(max_iter=2000), StandardScaler fitted on train",
        "train_records": len(train),
        "embedding_seconds": embed_seconds,
        "probe_seconds": round(time.time() - started, 1),
        "encoder_batches_computed": calls(),
        "store_size": len(store),
    }
    return probs, info


def random_init_encoder() -> Embed:
    torch.manual_seed(RANDOM_INIT_SEED)
    model = ResNet1d().eval().to(device())
    return model.embed


def random_init_probe(meta: pd.DataFrame, read: Tracings) -> tuple[Float32Array, dict[str, Any]]:
    store = EmbeddingStore("random_init", RANDOM_INIT_VERSION)
    probs, info = probe_arm(meta, read, store, random_init_encoder(), 512)
    return probs, {
        "arm": "random_init",
        "description": "study ResNet1d frozen at its seeded initialisation, one probe per label",
        "encoder_version": RANDOM_INIT_VERSION,
        "input": f"(12, 2500) at 250 Hz, unit {UNIT}, as stored",
        **info,
    }


def ecgfounder_encoder() -> Embed:
    # The checkpoint holds one numpy scalar beside its tensors, which
    # weights_only=True refuses on torch 2.2.2; the digest is checked first and
    # the file is then read through the allowlist, never unpickled in full.
    state = allowlisted_load(verified("ecgfounder/12_lead_ECGFounder.pth"))["state_dict"]
    model = ecgfounder_net()
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device())

    def embed(x: torch.Tensor) -> torch.Tensor:
        # 250 Hz to the 500 Hz the model card asks for, by polyphase filtering.
        at_500 = resample_poly(x.cpu().numpy(), 2, 1, axis=-1).astype(np.float32)
        features: torch.Tensor = model(torch.from_numpy(at_500).to(x.device))[1]
        return features

    return embed


def ecgfounder_probe(meta: pd.DataFrame, read: Tracings) -> tuple[Float32Array, dict[str, Any]]:
    store = EmbeddingStore("ecgfounder", ECGFOUNDER_VERSION)
    probs, info = probe_arm(meta, read, store, ecgfounder_encoder(), ECGFOUNDER_BATCH)
    return probs, {
        "arm": "ecgfounder",
        "description": "ECGFounder 12-lead Net1D frozen, one logistic probe per label",
        "encoder_version": ECGFOUNDER_VERSION,
        "weights": "huggingface.co/PKUDigitalHealth/ECGFounder 12_lead_ECGFounder.pth (MIT)",
        "input": (
            f"(12, 5000): the stored (12, 2500) at 250 Hz, unit {UNIT}, resampled to 500 Hz "
            "by scipy.signal.resample_poly(2, 1); no millivolt scale exists to restore"
        ),
        "features": "1024-d mean over time before the dense head",
        **info,
    }


class Tabular:
    """The distribution's seven preprocessed tabular features for any metadata positions."""

    def __init__(self, meta: pd.DataFrame, root: Path = ECHONEXT_DIR) -> None:
        self.split = meta["split"].to_numpy()
        self.row = meta["row"].to_numpy()
        self.arrays = {
            s: np.load(root / f"EchoNext_{s}_tabular_features.npy") for s in SCORED_SPLITS
        }

    def __call__(self, positions: NDArray[np.int_]) -> Float32Array:
        out = np.empty((len(positions), 7), dtype=np.float32)
        for split in np.unique(self.split[positions]):
            inside = np.flatnonzero(self.split[positions] == split)
            out[inside] = self.arrays[split][self.row[positions[inside]]]
        return out


def echonext_mini_model() -> EchoNextMini:
    path = verified("echonext_mini/weights.pt")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = EchoNextMini()
    model.load_state_dict(saved["model"], strict=True)
    return model.eval().to(device())


def mini_predict(
    model: EchoNextMini, read: Tracings, tabular: Tabular, positions: NDArray[np.int_]
) -> Float32Array:
    dev = device()
    out = []
    with torch.no_grad():
        for start in range(0, len(positions), 512):
            chunk = positions[start : start + 512]
            x = torch.from_numpy(read(chunk)).to(dev)
            t = torch.from_numpy(tabular(chunk)).to(dev)
            out.append(torch.sigmoid(model(x, t)).float().cpu().numpy())
    return np.concatenate(out)


def echonext_mini(meta: pd.DataFrame, read: Tracings) -> tuple[Float32Array, dict[str, Any]]:
    probs = mini_predict(echonext_mini_model(), read, Tabular(meta), scored_positions(meta))
    return probs, {
        "arm": "echonext_mini",
        "description": "the published EchoNext mini-model on its own weights, nothing refitted",
        "weights": (
            "github.com/PierreElias/IntroECG commit 15233e93, 7-EchoNext Minimodel weights.pt "
            "(non-commercial), read with torch.load(weights_only=True)"
        ),
        "architecture": "ecs.echonext_mini.EchoNextMini, written from the checkpoint shapes",
        "input": (
            f"(12, 2500) at 250 Hz, unit {UNIT}, as stored, and the 7 preprocessed tabular "
            "features of the distribution (sex, rates, intervals, age)"
        ),
    }


SCORERS: dict[str, Callable[[pd.DataFrame, Tracings], tuple[Float32Array, dict[str, Any]]]] = {
    "resnet": train_resnet,
    "random_init": random_init_probe,
    "ecgfounder": ecgfounder_probe,
    "echonext_mini": echonext_mini,
}


def peak_memory() -> dict[str, int]:
    """Peak resident set of this process, and what the Apple GPU driver holds now."""
    out = {"peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    if torch.backends.mps.is_available():
        out["mps_driver_bytes"] = int(torch.mps.driver_allocated_memory())
    return out


def trial(arm: str, meta: pd.DataFrame, read: Tracings, n: int) -> None:
    """Time the arm's forward pass on n test records and project the full run; writes nothing."""
    positions = scored_positions(meta)[:n]
    if arm == "ecgfounder":
        embed = ecgfounder_encoder()
        started = time.time()
        embed_positions(read, positions, embed, ECGFOUNDER_BATCH, device())
        records = len(transfer_cohorts(meta).training) + len(scored_positions(meta))
    elif arm == "echonext_mini":
        model, tabular = echonext_mini_model(), Tabular(meta)
        started = time.time()
        mini_predict(model, read, tabular, positions)
        records = len(scored_positions(meta))
    else:
        raise SystemExit(f"no trial for {arm}")
    seconds = time.time() - started
    print(
        json.dumps(
            {
                "arm": arm,
                "trial_records": n,
                "trial_seconds": round(seconds, 1),
                "records_to_encode": records,
                "expected_encoder_hours": round(seconds / n * records / 3600, 2),
                **peak_memory(),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--trial", type=int, help="time this many records, write nothing")
    args = parser.parse_args()
    meta = read_metadata()
    read = Tracings(meta)
    if args.trial:
        trial(args.arm, meta, read, args.trial)
        return
    started = time.time()
    probs, info = SCORERS[args.arm](meta, read)
    info |= {
        "seconds": round(time.time() - started, 1),
        "device": str(device()),
        **peak_memory(),
        "machine": machine_info(),
    }
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    keys = meta["ecg_key"].to_numpy()[scored_positions(meta)]
    np.savez(SCORES_DIR / f"{args.arm}.npz", ecg_key=keys, probs=probs, labels=np.array(LABELS))
    (SCORES_DIR / f"{args.arm}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "machine"}, indent=2))


if __name__ == "__main__":
    main()
