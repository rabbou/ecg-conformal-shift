"""The embedding store returns a filed vector without calling the encoder again."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ecs.embedding_store import EmbeddingStore


class CountingEncoder:
    """Embeds a position as a vector derived from its digest, and counts the calls."""

    def __init__(self, digests: list[str]) -> None:
        self.digests = digests
        self.calls = 0
        self.rows = 0

    def __call__(self, positions: NDArray[np.int_]) -> NDArray[np.float32]:
        self.calls += 1
        self.rows += len(positions)
        return np.array(
            [[len(self.digests[i]), int(self.digests[i][-1], 16)] for i in positions],
            dtype=np.float32,
        )


DIGESTS = ["aa01", "bb02", "cc03", "aa01", "dd0f"]  # the fourth is a copy of the first


def test_a_second_pass_calls_the_encoder_zero_times(tmp_path: Path) -> None:
    store = EmbeddingStore("toy", "v1", root=tmp_path)
    encoder = CountingEncoder(DIGESTS)
    first = store.embed(DIGESTS, encoder, chunk=2)
    calls_after_first = encoder.calls
    second = store.embed(DIGESTS, encoder, chunk=2)
    assert calls_after_first == 2  # four distinct digests in chunks of two
    assert encoder.calls == calls_after_first
    assert np.array_equal(first, second)


def test_a_fresh_store_reads_the_vectors_from_disk(tmp_path: Path) -> None:
    EmbeddingStore("toy", "v1", root=tmp_path).embed(DIGESTS, CountingEncoder(DIGESTS))
    encoder = CountingEncoder(DIGESTS)
    vectors = EmbeddingStore("toy", "v1", root=tmp_path).embed(DIGESTS, encoder)
    assert encoder.calls == 0
    assert vectors[:, 1].tolist() == [1, 2, 3, 1, 15]


def test_copies_are_computed_once_and_share_their_vector(tmp_path: Path) -> None:
    encoder = CountingEncoder(DIGESTS)
    vectors = EmbeddingStore("toy", "v1", root=tmp_path).embed(DIGESTS, encoder)
    assert encoder.rows == 4
    assert np.array_equal(vectors[0], vectors[3])


def test_only_the_missing_digests_are_computed(tmp_path: Path) -> None:
    store = EmbeddingStore("toy", "v1", root=tmp_path)
    store.embed(DIGESTS[:2], CountingEncoder(DIGESTS[:2]))
    encoder = CountingEncoder(DIGESTS)
    store.embed(DIGESTS, encoder)
    assert encoder.rows == 2  # cc03 and dd0f


def test_another_encoder_version_does_not_hit_the_old_vectors(tmp_path: Path) -> None:
    EmbeddingStore("toy", "v1", root=tmp_path).embed(DIGESTS, CountingEncoder(DIGESTS))
    encoder = CountingEncoder(DIGESTS)
    EmbeddingStore("toy", "v2", root=tmp_path).embed(DIGESTS, encoder)
    assert encoder.rows == 4
