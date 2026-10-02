"""Encoder vectors computed once and filed by encoder, encoder version and tracing digest.

A vector is keyed by the SHA-256 of the tracing's stored bytes, not by a record
identifier: two records that are copies of one tracing share a vector, and a
record whose bytes change gets a new one instead of a stale hit.  The encoder
version is part of the directory, so a changed preprocessing or checkpoint
writes beside the old vectors rather than over them.

The store holds derived data of whatever corpus it embeds, so it lives outside
the repository (``ECS_EMBEDDING_STORE``, default ``~/data/ecg-embeddings``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["STORE_DIR", "EmbeddingStore"]

STORE_DIR = Path(os.environ.get("ECS_EMBEDDING_STORE", Path.home() / "data/ecg-embeddings"))

Vectors = NDArray[np.float32]


class EmbeddingStore:
    """Shards of (digest, vector) pairs under ``root/encoder/version``."""

    def __init__(self, encoder: str, version: str, root: Path = STORE_DIR) -> None:
        self.directory = root / encoder / version
        self.directory.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, tuple[Path, int]] = {}
        for shard in sorted(self.directory.glob("shard-*.npz")):
            with np.load(shard) as data:
                for i, digest in enumerate(data["digests"].tolist()):
                    self._index[str(digest)] = (shard, i)

    def __contains__(self, digest: str) -> bool:
        return digest in self._index

    def __len__(self) -> int:
        return len(self._index)

    def embed(
        self,
        digests: Sequence[str],
        compute: Callable[[NDArray[np.int_]], Vectors],
        chunk: int = 4096,
    ) -> Vectors:
        """The vector of each digest, in order, computing only those not yet filed.

        ``compute`` receives positions into ``digests`` and returns one row per
        position.  Each chunk it computes is written as its own shard before the
        next starts, so an interrupted pass resumes where it stopped.
        """
        missing: list[int] = []
        seen: set[str] = set()
        for i, d in enumerate(digests):
            if d not in self._index and d not in seen:
                missing.append(i)
                seen.add(d)
        for start in range(0, len(missing), chunk):
            positions = np.array(missing[start : start + chunk], dtype=int)
            vectors = np.asarray(compute(positions), dtype=np.float32)
            if vectors.shape[0] != len(positions):
                raise ValueError(f"compute returned {vectors.shape[0]} rows for {len(positions)}")
            self._write([digests[i] for i in positions], vectors)
        return self._read(digests)

    def _write(self, digests: list[str], vectors: Vectors) -> None:
        shard = self.directory / f"shard-{len(list(self.directory.glob('shard-*.npz'))):05d}.npz"
        np.savez(shard, digests=np.array(digests), vectors=vectors)
        for i, digest in enumerate(digests):
            self._index[digest] = (shard, i)
        meta = self.directory / "store.json"
        if not meta.exists():
            meta.write_text(json.dumps({"dimension": int(vectors.shape[1])}, indent=2) + "\n")

    def _read(self, digests: Sequence[str]) -> Vectors:
        by_shard: dict[Path, list[tuple[int, int]]] = {}
        for out_row, digest in enumerate(digests):
            shard, i = self._index[digest]
            by_shard.setdefault(shard, []).append((out_row, i))
        out: Vectors | None = None
        for shard, pairs in by_shard.items():
            with np.load(shard) as data:
                vectors = data["vectors"]
                if out is None:
                    out = np.empty((len(digests), vectors.shape[1]), dtype=np.float32)
                rows, inner = zip(*pairs, strict=True)
                out[list(rows)] = vectors[list(inner)]
        if out is None:
            return np.empty((0, 0), dtype=np.float32)
        return out
