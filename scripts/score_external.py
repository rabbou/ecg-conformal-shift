"""The PTB-XL baseline, spent once on Shandong and once on Chongqing.

The model is the one PTB-XL trained (``results/baseline/model.pt``), loaded with
the standardisation its training fold fixed, and it is not touched again: no
fine-tuning, no re-thresholding, no per-corpus preprocessing.  Each external
corpus is read through the same canonical chain as PTB-XL, scored, and the
probability rows written out with the commit, the seed and the checkpoint's
configuration beside them, so the conformal step downstream reads numbers whose
provenance is on the file.

The corpus is never held in memory whole: records are read in chunks, scored,
and the waveforms dropped, because 25,770 tracings in canonical form are 6 GB
and the machine has 15.

Two files come out per corpus, under ``results/external/``:

``<corpus>.npz``    the record identifiers, their labels, and the probability row;
``<corpus>.json``   what was run, what was kept, and what had to give (C-14).

Usage: .venv/bin/python scripts/score_external.py [--corpus sph acs]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from ecs.config import ACS_DIR, ACS_LABELLED_SPLIT, RESULTS_DIR, SPH_DIR
from ecs.encoders import machine_info
from ecs.ingest import SPH_DEVIATION, Corpus, load_acs, load_sph
from ecs.labels import MILabelSpec, acs_mi_label, sph_mi_label
from ecs.models import ResNet1d

CHUNK = 500  # records read from disk at once, so the corpus is never held whole


@dataclass
class Scored:
    """One corpus after the model has seen it, and what it cost to get there."""

    ids: list[str]
    labels: NDArray[np.int_]
    probs: NDArray[np.float64]
    excluded: dict[str, str]
    n_resampled: int
    n_cropped: int
    notes: list[str]

    @property
    def deviations(self) -> list[str]:
        """Every step the chain could not make identical across corpora (C-14)."""
        n = len(self.ids) + len(self.excluded)
        out = list(self.notes)
        if self.n_resampled:
            out.append(f"{self.n_resampled} of {n} records resampled to 500 Hz")
        if self.n_cropped:
            out.append(f"{self.n_cropped} of {n} records longer than ten seconds, cropped")
        if self.excluded:
            out.append(f"{len(self.excluded)} of {n} records excluded: see the excluded field")
        return out


def load_model(checkpoint: Path) -> tuple[ResNet1d, float, float]:
    """The trained baseline and the standardisation it was trained under."""
    saved = torch.load(checkpoint, map_location="cpu")
    model = ResNet1d(n_classes=2)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, float(saved["centre"]), float(saved["scale"])


def probabilities(
    model: ResNet1d, x: NDArray[np.float32], centre: float, scale: float
) -> NDArray[np.float64]:
    """Class probabilities for every row, in batches, in the same order."""
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 128):
            batch = torch.from_numpy((x[i : i + 128] - centre) / scale)
            out.append(torch.softmax(model(batch), dim=1))
    stacked = torch.cat(out) if out else torch.empty((0, 2))
    return stacked.numpy().astype(np.float64)


def score_in_chunks(
    name: str,
    wanted: list[str],
    read: Callable[[list[str]], Corpus],
    labels: pd.Series,
    model: ResNet1d,
    centre: float,
    scale: float,
    notes: list[str],
) -> Scored:
    """Read ``wanted`` in chunks, score each, and keep only the probability rows."""
    ids: list[str] = []
    rows: list[NDArray[np.float64]] = []
    excluded: dict[str, str] = {}
    resampled = cropped = 0
    for start in range(0, len(wanted), CHUNK):
        block = wanted[start : start + CHUNK]
        corpus = read(block)
        excluded.update(corpus.excluded)
        resampled += corpus.n_resampled
        cropped += corpus.n_cropped
        if corpus.ids:
            ids.extend(corpus.ids)
            rows.append(probabilities(model, corpus.x, centre, scale))
        print(f"  {name}: {len(ids):>6} / {len(wanted)}", flush=True)
    probs = np.concatenate(rows) if rows else np.empty((0, 2), dtype=np.float64)
    y = labels.loc[ids].to_numpy().astype(int)
    return Scored(ids, y, probs, excluded, resampled, cropped, notes)


def sph_job(model: ResNet1d, centre: float, scale: float, root: Path, limit: int | None) -> Scored:
    metadata = pd.read_csv(root / "metadata.csv")
    labels = sph_mi_label(metadata).set_axis(metadata["ECG_ID"].astype(str))
    wanted = [str(i) for i in metadata["ECG_ID"]][:limit]
    return score_in_chunks(
        "sph",
        wanted,
        lambda block: load_sph(metadata, root, ids=block),
        labels,
        model,
        centre,
        scale,
        [
            SPH_DEVIATION,
            "the label is AHA category M (160, 161, 165, 166); 233 of its 260 positives "
            "carry the 'old' modifier, so the class is a chronic infarct pattern as PTB-XL's is",
        ],
    )


def acs_job(model: ResNet1d, centre: float, scale: float, root: Path, limit: int | None) -> Scored:
    table = pd.read_csv(root / ACS_LABELLED_SPLIT)
    names = [str(f).removesuffix(".dat") for f in table["ecg_row_record"]]
    labels = acs_mi_label(table, MILabelSpec()).set_axis(pd.Index(names))
    return score_in_chunks(
        "acs",
        names[:limit],
        lambda block: load_acs(table, root, ids=block),
        labels,
        model,
        centre,
        scale,
        [
            f"only the {len(table)} records of {ACS_LABELLED_SPLIT} carry labels; the published "
            "test split withholds them, so the remaining waveforms on disk are not scored",
            "the label is the corpus's own AMI column, read from a coronary angiogram rather "
            "than from the tracing, and acute by construction; PTB-XL's and Shandong's "
            "positives are dominated by the chronic infarct pattern, so the two label sets "
            "name different clinical events and no re-coding here makes them the same",
        ],
    )


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", default=["sph", "acs"], choices=["sph", "acs"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(RESULTS_DIR / "baseline/model.pt"))
    parser.add_argument("--out", default=str(RESULTS_DIR / "external"))
    parser.add_argument("--sph-dir", default=str(SPH_DIR))
    parser.add_argument("--acs-dir", default=str(ACS_DIR))
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="score only the first N records of each corpus; for checking the run, not a result",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    model, centre, scale = load_model(checkpoint)
    baseline_config = json.loads((checkpoint.parent / "config.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    jobs = {
        "sph": (sph_job, Path(args.sph_dir), "Shandong Provincial Hospital"),
        "acs": (acs_job, Path(args.acs_dir), "Chongqing ACS-ECG"),
    }
    for name in args.corpus:
        job, root, title = jobs[name]
        started = time.perf_counter()
        scored = job(model, centre, scale, root, args.max_records)
        seconds = round(time.perf_counter() - started, 1)
        np.savez_compressed(
            out / f"{name}.npz",
            ids=np.asarray(scored.ids, dtype=str),
            labels=scored.labels,
            probs=scored.probs,
        )
        (out / f"{name}.json").write_text(
            json.dumps(
                {
                    "corpus": name,
                    "title": title,
                    "model": "the PTB-XL supervised baseline, frozen: no fine-tuning, "
                    "no per-corpus preprocessing, one forward pass",
                    "checkpoint": str(checkpoint.relative_to(RESULTS_DIR.parent)),
                    "baseline_config": baseline_config,
                    "standardisation": {"centre_mv": centre, "scale_mv": scale},
                    "seed": args.seed,
                    "max_records": args.max_records,
                    "git_commit": commit,
                    "n_scored": len(scored.ids),
                    "n_positive": int(scored.labels.sum()),
                    "prevalence": float(scored.labels.mean()) if len(scored.ids) else float("nan"),
                    "n_excluded": len(scored.excluded),
                    "excluded": scored.excluded,
                    "deviations": scored.deviations,
                    "seconds": seconds,
                    "machine": machine_info(),
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"{name}: {len(scored.ids)} scored, {int(scored.labels.sum())} infarction "
            f"({scored.labels.mean():.4f}), {len(scored.excluded)} excluded, {seconds} s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
