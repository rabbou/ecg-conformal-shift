"""One encoder arm over one corpus, computed once and kept.

A *representation* is the vector an encoder produces for a tracing.  Computing
it costs the same whether it is used once or twenty times, and the timing probe
put a full pass over the three corpora at hours rather than minutes, so it is
computed once per (arm, corpus) pair and read from disk afterwards.

For each pair this writes two files under ``results/embeddings/<arm>/``:

``<corpus>.npz``
    ``ids``       the tracing identifiers, in the order the loader yields them,
                  with the records the ingestion chain excluded absent;
    ``embedding`` float32, one row per id.

``<corpus>.json``
    everything needed to say what that array is: the arm, where its weights
    came from, what it was pre-trained on, the corpus, the commit of this
    repository, the deviations the common ingestion chain had to name, the
    records it excluded and why, the extra chain this arm demands on top of the
    ingestion, and how long the pass took.

A pass that is interrupted keeps the chunks it finished, in a ``.partial``
directory beside the output, and a rerun picks up from the first unfinished
chunk.  A pair whose ``.npz`` already exists is left alone; ``--overwrite``
recomputes it.

Usage:
    .venv/bin/python scripts/extract_embeddings.py --arm random_init --corpus ptbxl sph acs
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecs.config import ACS_DIR, EMBEDDINGS_DIR, PTBXL_DIR, SPH_DIR
from ecs.encoders import ARMS, PRETRAINING, machine_info
from ecs.ingest import Corpus, load_acs, load_ptbxl, load_sph

CHUNK = 200  # records read and embedded before anything is written
BATCH = 20  # records through the network at once

CorpusLoader = Callable[[Sequence[str]], Corpus]


def corpus_loaders() -> dict[str, tuple[list[str], CorpusLoader]]:
    """Every corpus: its record ids in a fixed order, and how to load a slice."""
    database = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    metadata = pd.read_csv(SPH_DIR / "metadata.csv")
    train = pd.read_csv(ACS_DIR / "CSV/train.csv")
    test = pd.read_csv(ACS_DIR / "CSV/test.csv")
    acs = pd.concat([train, test], ignore_index=True)
    acs_ids = [str(f).removesuffix(".dat") for f in acs["ecg_row_record"]]
    return {
        "ptbxl": (
            [str(i) for i in database.index],
            lambda ids: load_ptbxl(database, ids=[int(i) for i in ids]),
        ),
        "sph": (list(metadata["ECG_ID"]), lambda ids: load_sph(metadata, ids=ids)),
        "acs": (acs_ids, lambda ids: load_acs(acs, ids=ids)),
    }


def embed_chunk(embed: Callable[[np.ndarray], torch.Tensor], x: np.ndarray) -> np.ndarray:
    """The chunk's rows through the network, in batches, in the same order."""
    if len(x) == 0:
        return np.empty((0, 0), dtype=np.float32)
    with torch.no_grad():
        parts = [embed(x[i : i + BATCH]).numpy() for i in range(0, len(x), BATCH)]
    return np.concatenate(parts).astype(np.float32)


def extract(
    arm: str,
    corpus: str,
    ids: Sequence[str],
    load: CorpusLoader,
    out_dir: Path,
    overwrite: bool = False,
) -> Path | None:
    """One (arm, corpus) pair.  Returns the output path, or None if it was skipped."""
    out = out_dir / arm / f"{corpus}.npz"
    if out.exists() and not overwrite:
        print(f"{arm}/{corpus}: already there, left alone ({out})", flush=True)
        return None
    partial = out.with_suffix(".partial")
    if overwrite and partial.exists():
        shutil.rmtree(partial)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir(exist_ok=True)

    started = time.perf_counter()
    embed, meta = ARMS[arm]()
    excluded: dict[str, str] = {}
    deviations: list[str] = []
    for start in range(0, len(ids), CHUNK):
        shard = partial / f"{start:08d}.npz"
        if shard.exists():
            with np.load(shard, allow_pickle=False) as done:
                excluded.update(json.loads(str(done["excluded"])))
                deviations = list(done["deviations"])
            continue
        chunk = load(list(ids[start : start + CHUNK]))
        vectors = embed_chunk(embed, chunk.x)
        excluded.update(chunk.excluded)
        deviations = chunk.deviations
        _write(
            shard,
            ids=np.array(chunk.ids, dtype=object).astype(str),
            embedding=vectors,
            excluded=json.dumps(chunk.excluded),
            deviations=np.array(chunk.deviations, dtype=str),
        )
        done_n = min(start + CHUNK, len(ids))
        print(
            f"  {arm}/{corpus}: {done_n:>6} / {len(ids)}  {time.perf_counter() - started:6.0f} s",
            flush=True,
        )

    all_ids, embedding = _merge(partial)
    _write(out, ids=all_ids, embedding=embedding)
    seconds = round(time.perf_counter() - started, 1)
    (out_dir / arm / f"{corpus}.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "corpus": corpus,
                "weights_source": meta["weights_source"],
                "pretraining_corpora": PRETRAINING[arm],
                "preprocessing": meta["preprocessing"],
                "notes": meta["notes"],
                "n_params": meta["n_params"],
                "git_commit": git_commit(),
                "n_records": len(ids),
                "n_kept": len(all_ids),
                "n_excluded": len(excluded),
                "excluded": excluded,
                "ingestion_deviations": deviations,
                "embedding_dim": int(embedding.shape[1]) if len(embedding) else 0,
                "seconds": seconds,
                "machine": machine_info(),
            },
            indent=2,
        )
        + "\n"
    )
    shutil.rmtree(partial)
    print(f"{arm}/{corpus}: {len(all_ids)} vectors in {seconds} s -> {out}", flush=True)
    return out


def _write(path: Path, **arrays: np.ndarray | str) -> None:
    """A .npz written whole: to a neighbour first, then moved into place, so an
    interrupted write never leaves a half file that a rerun would trust."""
    tmp = path.with_suffix(".writing.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def _merge(partial: Path) -> tuple[np.ndarray, np.ndarray]:
    """The shards in chunk order, concatenated."""
    ids: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    for shard in sorted(partial.glob("[0-9]*.npz")):
        with np.load(shard, allow_pickle=False) as data:
            ids.append(data["ids"])
            vectors.append(data["embedding"])
    kept = [v for v in vectors if v.size]
    return (
        np.concatenate(ids) if ids else np.array([], dtype=str),
        np.concatenate(kept) if kept else np.empty((0, 0), dtype=np.float32),
    )


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--corpus", nargs="+", default=["ptbxl", "sph", "acs"])
    parser.add_argument("--out", default=str(EMBEDDINGS_DIR))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    loaders = corpus_loaders()
    for corpus in args.corpus:
        ids, load = loaders[corpus]
        extract(args.arm, corpus, ids, load, Path(args.out), args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
