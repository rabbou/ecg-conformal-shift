"""The four encoder arms, in one place, with the chain each one demands.

An *arm* is a way of turning a canonical (12, 5000) tracing into one vector.
Three are pre-trained encoders published by other groups; the fourth is our own
ResNet1d left at its random initialisation, the floor the others have to clear
(C-17).

The common ingestion chain (``ecs.ingest``) is identical for every corpus and
stops at the canonical form.  What each encoder needs on top of that -- a
different sampling rate, a band-pass, a different input shape -- is *not*
common, and it is applied here, after ingestion, per arm.  Each arm therefore
reports a ``preprocessing`` string naming every step it applies and every step
its authors document that we do not, so a difference between arms can be
attributed rather than guessed at.

Loading an arm is expensive and its weights are large; nothing is loaded until
the arm's factory is called.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, decimate, filtfilt

from .config import REPO_ROOT
from .models import ResNet1d

__all__ = ["ARMS", "PRETRAINING", "WEIGHTS", "Arm", "machine_info"]

WEIGHTS = REPO_ROOT / "data/weights"

# One arm: the function that embeds a (B, 12, 5000) float32 batch, and what has
# to be said about it in any result file that uses it.
Arm = tuple[Callable[[np.ndarray], torch.Tensor], dict[str, object]]

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

NO_EXTRA_PREPROCESSING = (
    "none beyond the common ingestion: (12, 5000) float32 millivolts at 500 Hz "
    "goes into the network unchanged"
)


# The untrained control has to be one network, not one per call.  Every other
# arm loads the same weights from disk whenever it is built; this one draws
# them, and extract_embeddings.py builds the arm once per corpus.  Unseeded,
# each corpus went through a different random network and the probe fitted on
# PTB-XL was spent in someone else's feature space.
RANDOM_INIT_SEED = 0


def random_init() -> Arm:
    torch.manual_seed(RANDOM_INIT_SEED)
    model = ResNet1d().eval()
    return (
        lambda x: model.embed(torch.from_numpy(x)),
        {
            "n_params": sum(p.numel() for p in model.parameters()),
            "weights_source": (
                f"torch default initialisation, torch.manual_seed({RANDOM_INIT_SEED})"
            ),
            "preprocessing": NO_EXTRA_PREPROCESSING,
            "notes": "ResNet1d from src/ecs/models.py, input (B, 12, 5000) mV, 256-d embedding",
        },
    )


def ecgfounder() -> Arm:
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
            "preprocessing": (
                f"{NO_EXTRA_PREPROCESSING}; the model card's config.json asks for 500 Hz, "
                "which the canonical form already is. Not applied: the band-pass and "
                "per-record z-scoring in the authors' fine-tuning code, which their model "
                "card does not require for feature extraction"
            ),
            "notes": (
                "Net1D per the authors' finetune_model.py; 1024-d features before the dense "
                "head; input (B, 12, 5000) at 500 Hz per config.json; "
                f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}"
            ),
        },
    )


# HuBERT-ECG's published chain: a band-pass, then five seconds at 100 Hz with the
# twelve leads laid end to end into one sequence.
HUBERT_BAND_HZ = (0.05, 47.0)
HUBERT_FILTER_ORDER = 3
HUBERT_SECONDS = 5
HUBERT_TARGET_HZ = 100


def hubert_ecg() -> Arm:
    from transformers import AutoModel

    path = WEIGHTS / "hubert-ecg-base"
    model = AutoModel.from_pretrained(path, trust_remote_code=True).eval()
    b, a = butter(HUBERT_FILTER_ORDER, HUBERT_BAND_HZ, btype="bandpass", fs=500.0)

    def embed(x: np.ndarray) -> torch.Tensor:
        # Band-pass each lead at 500 Hz, keep the first five seconds, then follow
        # the authors' dataset.py: flatten the twelve leads into one sequence and
        # decimate 500 -> 100 Hz.
        filtered = filtfilt(b, a, x.astype(np.float64), axis=-1)
        flat = filtered[:, :, : HUBERT_SECONDS * 500].reshape(len(x), -1)
        step = 500 // HUBERT_TARGET_HZ
        inputs = torch.from_numpy(decimate(flat, step, axis=-1).astype(np.float32))
        return model(input_values=inputs).last_hidden_state.mean(dim=1)

    return (
        embed,
        {
            "n_params": sum(p.numel() for p in model.parameters()),
            "weights_source": (
                "huggingface.co/Edoardo-Coppola/hubert-ecg-base model.safetensors (CC BY-NC 4.0)"
            ),
            "preprocessing": (
                f"zero-phase Butterworth band-pass of order {HUBERT_FILTER_ORDER} between "
                f"{HUBERT_BAND_HZ[0]} and {HUBERT_BAND_HZ[1]} Hz applied per lead at 500 Hz; "
                f"first {HUBERT_SECONDS} seconds kept; the twelve leads flattened into one "
                f"sequence and decimated to {HUBERT_TARGET_HZ} Hz, in that order, as the "
                "authors' dataset.py does. Not applied: the per-record rescaling to [-1, 1] "
                "the paper also describes, which would change the scale the other arms see"
            ),
            "notes": (
                "transformers AutoModel with the repo's hubert_ecg.py (custom code); input "
                "(B, 6000): 5 s x 12 leads flattened then decimated by 5; 768-d mean-pooled "
                "last hidden state"
            ),
        },
    )


def ecgfm() -> Arm:
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
            "preprocessing": (
                f"{NO_EXTRA_PREPROCESSING}; the checkpoint carries its own normalisation "
                "inside fairseq_signals' extract_features, which is applied there and not here"
            ),
            "notes": (
                f"fairseq_signals {cfg.model._name}; input (B, 12, 5000) at 500 Hz; "
                "mean-pooled extract_features output"
            ),
        },
    )


ARMS: dict[str, Callable[[], Arm]] = {
    "random_init": random_init,
    "ecgfounder": ecgfounder,
    "ecgfm": ecgfm,
    "hubert_ecg": hubert_ecg,
}


def machine_info() -> dict[str, object]:
    """What machine a measurement was taken on, named well enough to compare two."""
    return {
        "cpu": _cpu_name() or platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def _cpu_name() -> str:
    """The processor's own brand string, however this operating system states it."""
    if platform.system() == "Darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return ""
