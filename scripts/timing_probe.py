"""Seconds per hundred PTB-XL records through each encoder arm, on this machine.

The number sizes the week: one forward pass over ~65,000 records on a CPU is
the cost of adding an arm, and no training run starts before it is known
(PLAN.md, day 1).  One row per arm goes to results/timing.json; an arm that
cannot be loaded gets ``seconds_per_100: null`` and a note saying exactly why,
so the next session does not rediscover the blocker.

Usage: .venv/bin/python scripts/timing_probe.py [--arm NAME ...] [--n 100]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from scipy.signal import decimate

from ecs.config import PTBXL_DIR, REPO_ROOT, RESULTS_DIR
from ecs.ingest import load_ptbxl
from ecs.models import ResNet1d

WEIGHTS = REPO_ROOT / "data/weights"
BATCH = 20

# What each arm's source says it was pre-trained on, with where that is stated.
PRETRAINING = {
    "random_init": "none (random initialisation, frozen)",
    "ecgfounder": (
        "Harvard-Emory ECG Database, >10 million recordings; no public corpus named "
        "(arXiv:2410.04133; HF model card PKUDigitalHealth/ECGFounder)"
    ),
    "ecgfm": (
        "MIMIC-IV-ECG v1.0 and PhysioNet/CinC 2021 v1.0.3, the latter containing PTB-XL "
        "(github.com/bowang-lab/ECG-FM README, 'Pretrained on')"
    ),
    "hubert_ecg": (
        "Ribeiro/CODE, CPSC and CPSC-Extra, PTB and PTB-XL, Georgia, Chapman-Shaoxing, "
        "Ningbo, Tianchi (Hefei), Shandong Provincial Hospital (SPH), MIMIC-IV ECG "
        "(medRxiv 10.1101/2024.11.14.24317328v3, Methods - Data and Preprocessing, "
        "read 2026-08-23)"
    ),
}


def records(n: int) -> np.ndarray:
    database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    return load_ptbxl(database, ids=list(database.index[:n])).x


# ---- arms: each returns (embed function over a (B, 12, 5000) float32 batch, meta) ----


def random_init() -> tuple[Callable[[np.ndarray], torch.Tensor], dict[str, object]]:
    model = ResNet1d().eval()
    return (
        lambda x: model.embed(torch.from_numpy(x)),
        {
            "n_params": sum(p.numel() for p in model.parameters()),
            "weights_source": "torch default initialisation, seed unset",
            "notes": "ResNet1d from src/ecs/models.py, input (B, 12, 5000) mV, 256-d embedding",
        },
    )


def ecgfounder() -> tuple[Callable[[np.ndarray], torch.Tensor], dict[str, object]]:
    sys.path.insert(0, str(REPO_ROOT / "third_party/ecgfounder"))
    from net1d import Net1D  # vendored, MIT; architecture as in finetune_model.py

    model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=150,
        return_features=True,
    )
    path = WEIGHTS / "ecgfounder/12_lead_ECGFounder.pth"
    state = torch.load(path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return (
        lambda x: model(torch.from_numpy(x))[1],
        {
            "n_params": sum(p.numel() for p in model.parameters()),
            "weights_source": (
                "huggingface.co/PKUDigitalHealth/ECGFounder 12_lead_ECGFounder.pth (MIT)"
            ),
            "notes": (
                "Net1D per the authors' finetune_model.py; 1024-d features before the dense "
                "head; input (B, 12, 5000) at 500 Hz per config.json; the authors' fine-tuning "
                "code band-passes and z-scores each record, not applied here; "
                f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}"
            ),
        },
    )


def hubert_ecg() -> tuple[Callable[[np.ndarray], torch.Tensor], dict[str, object]]:
    from transformers import AutoModel

    path = WEIGHTS / "hubert-ecg-base"
    model = AutoModel.from_pretrained(path, trust_remote_code=True).eval()

    def embed(x: np.ndarray) -> torch.Tensor:
        # The authors' dataset.py: first five seconds, the 12 leads flattened into
        # one sequence, decimated 500 -> 100 Hz.  Band-pass and [-1, 1] rescaling
        # are also in the paper's chain and are not applied here.
        flat = x[:, :, :2500].reshape(len(x), -1).astype(np.float64)
        inputs = torch.from_numpy(decimate(flat, 5, axis=-1).astype(np.float32))
        return model(input_values=inputs).last_hidden_state.mean(dim=1)

    return (
        embed,
        {
            "n_params": sum(p.numel() for p in model.parameters()),
            "weights_source": (
                "huggingface.co/Edoardo-Coppola/hubert-ecg-base model.safetensors (CC BY-NC 4.0)"
            ),
            "notes": (
                "transformers AutoModel with the repo's hubert_ecg.py (custom code); input "
                "(B, 6000): 5 s x 12 leads flattened then decimated by 5; 768-d mean-pooled "
                "last hidden state"
            ),
        },
    )


def ecgfm() -> tuple[Callable[[np.ndarray], torch.Tensor], dict[str, object]]:
    # The weights only load through the authors' fairseq fork (HF card: "cannot be
    # loaded using transformers").  It is not in this project's environment: it
    # needs a C++/Cython build and its own hydra/omegaconf pins, so the attempt
    # lives in a scratch venv and this arm is run with that interpreter.
    from fairseq_signals.utils import checkpoint_utils

    path = WEIGHTS / "ecgfm/mimic_iv_ecg_physionet_pretrained.pt"
    model, cfg, _task = checkpoint_utils.load_model_and_task(str(path))
    model.eval()

    def embed(x: np.ndarray) -> torch.Tensor:
        out = model.extract_features(source=torch.from_numpy(x), padding_mask=None)
        return out["x"].mean(dim=1)

    return (
        embed,
        {
            "n_params": sum(p.numel() for p in model.parameters()),
            "weights_source": (
                "huggingface.co/wanglab/ecg-fm mimic_iv_ecg_physionet_pretrained.pt (MIT)"
            ),
            "notes": (
                f"fairseq_signals {cfg.model._name}; input (B, 12, 5000) at 500 Hz; "
                "mean-pooled extract_features output"
            ),
        },
    )


ARMS: dict[str, Callable[[], tuple[Callable[[np.ndarray], torch.Tensor], dict[str, object]]]] = {
    "random_init": random_init,
    "ecgfounder": ecgfounder,
    "ecgfm": ecgfm,
    "hubert_ecg": hubert_ecg,
}


def time_arm(name: str, x: np.ndarray) -> dict[str, object]:
    row: dict[str, object] = {
        "arm": name,
        "seconds_per_100": None,
        "n_params": None,
        "weights_source": None,
        "pretraining_corpora": PRETRAINING[name],
        "notes": "",
        "n_records": len(x),
        "batch_size": BATCH,
        "torch_threads": torch.get_num_threads(),
        "load_average_1min": round(os.getloadavg()[0], 1),  # other work on the machine
    }
    try:
        embed, meta = ARMS[name]()
        row.update(meta)
        with torch.no_grad():
            embed(x[:BATCH])  # warm-up, not timed
            started = time.perf_counter()
            dims = {tuple(embed(x[i : i + BATCH]).shape[1:]) for i in range(0, len(x), BATCH)}
            seconds = time.perf_counter() - started
        row["seconds_per_100"] = round(seconds * 100 / len(x), 2)
        row["embedding_shape"] = sorted(dims)[0]
    except Exception as error:  # noqa: BLE001 -- the blocker IS the result
        row["notes"] = f"{row['notes']} | not timed: {type(error).__name__}: {error}".strip(" |")
        traceback.print_exc()
    if name == "ecgfm" and row["n_params"] is None:
        row["n_params"] = _ecgfm_param_count()
    return row


def _ecgfm_param_count() -> int | None:
    """Parameters in the ECG-FM checkpoint, countable without fairseq."""
    path = WEIGHTS / "ecgfm/mimic_iv_ecg_physionet_pretrained.pt"
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu")
    return int(sum(v.numel() for v in checkpoint["model"].values()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", nargs="*", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    out = RESULTS_DIR / "timing.json"
    report = json.loads(out.read_text()) if out.exists() else {"machine": {}, "arms": []}
    cpu = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
    ).stdout.strip()
    report["machine"] = {
        "cpu": cpu or platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    x = records(args.n)
    print(f"{len(x)} PTB-XL records in canonical form", flush=True)
    for name in args.arm:
        row = time_arm(name, x)
        report["arms"] = [r for r in report["arms"] if r["arm"] != name] + [row]
        report["arms"].sort(key=lambda r: list(ARMS).index(str(r["arm"])))
        RESULTS_DIR.mkdir(exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
