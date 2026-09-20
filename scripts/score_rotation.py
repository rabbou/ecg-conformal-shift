"""Score every corpus with every source's model, once.

Twenty ordered pairs plus five at home.  Each corpus is read once, in chunks,
and every chunk goes through all five models before the next one is read: the
waveforms are the expensive part and they do not depend on which model is
looking at them.

Each model carries its own standardisation, fitted on its own training part, and
applies it to target records unchanged.  Nothing of a target corpus reaches a
model's parameters or its scaling; the target is read, scored and put down.

Output, under ``results/rotation/<source>/scores/``:

``<corpus>_test.npz``   record ids, labels (n, 5), probabilities (n, 5)
``<source>_cal.npz``    the same for the source's own calibration part, which is
                        the only part a threshold is ever fitted on

Usage: .venv/bin/python scripts/score_rotation.py [--sources ptbxl,georgia]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ecs.config import RESULTS_DIR
from ecs.models import ResNet1d
from ecs.rotation import SOURCES, CorpusIndex, class_keys, corpus_index, load_waveforms

CHUNK = 500


class Scorer:
    """One source's trained model with the standardisation it was fitted under."""

    def __init__(self, source: str, results: Path) -> None:
        directory = results / "rotation" / source
        config = json.loads((directory / "config.json").read_text())
        self.source = source
        self.centre = float(config["standardisation"]["centre_mv"])
        self.scale = float(config["standardisation"]["scale_mv"])
        self.usable = list(config["usable_classes"])
        self.model = ResNet1d(n_classes=len(class_keys()))
        self.model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu"))
        self.model.eval()

    def probabilities(self, x: NDArray[np.float32], batch: int = 64) -> NDArray[np.float64]:
        out = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                block = torch.from_numpy((x[i : i + batch] - self.centre) / self.scale)
                out.append(torch.sigmoid(self.model(block)))
        if not out:
            return np.empty((0, len(class_keys())), dtype=np.float64)
        return torch.cat(out).numpy().astype(np.float64)


def score_part(
    index: CorpusIndex, part: str, scorers: dict[str, Scorer], results: Path
) -> dict[str, int]:
    """Read one part of one corpus in chunks and score it with every model."""
    wanted = index.ids(part)
    kept: list[str] = []
    probabilities: dict[str, list[NDArray[np.float64]]] = {s: [] for s in scorers}
    started = time.time()
    for start in range(0, len(wanted), CHUNK):
        x, ids = load_waveforms(index, wanted[start : start + CHUNK])
        kept.extend(ids)
        for source, scorer in scorers.items():
            probabilities[source].append(scorer.probabilities(x))
        print(
            f"  {index.name}/{part}: {len(kept):>6} / {len(wanted)}  "
            f"{time.time() - started:6.0f} s",
            flush=True,
        )
    y = index.labels(kept)
    for source in scorers:
        directory = results / "rotation" / source / "scores"
        directory.mkdir(parents=True, exist_ok=True)
        stacked = (
            np.concatenate(probabilities[source])
            if probabilities[source]
            else np.empty((0, len(class_keys())))
        )
        np.savez_compressed(
            directory / f"{index.name}_{part}.npz",
            ids=np.asarray(kept),
            y=y,
            p=stacked,
            classes=np.asarray(class_keys()),
            scored_by=source,
            corpus=index.name,
            part=part,
        )
    return {"n": len(kept), "seconds": int(time.time() - started)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        default=",".join(SOURCES),
        help="comma-separated models to score with (default: all five)",
    )
    parser.add_argument(
        "--corpora",
        default=",".join(SOURCES),
        help="comma-separated corpora to score (default: all five)",
    )
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    results = Path(RESULTS_DIR)
    sources = [s for s in args.sources.split(",") if s]
    scorers = {source: Scorer(source, results) for source in sources}
    for corpus in [c for c in args.corpora.split(",") if c]:
        index = corpus_index(corpus)
        done = score_part(index, "test", scorers, results)
        print(f"{corpus}/test: {done}", flush=True)
        if corpus in scorers:
            done = score_part(index, "cal", {corpus: scorers[corpus]}, results)
            print(f"{corpus}/cal: {done}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
