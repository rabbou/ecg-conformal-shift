"""What the cached representations promise: the right rows, in the right order,
with the record they belong to, and a rerun that costs nothing.

The stub arm here returns each tracing's mean amplitude.  The synthetic records
are written so that record ``k`` is flat at ``(k + 1) / 1000`` millivolts, so
the vector every row must hold is known from the row's identifier alone,
without reading the extractor.  A row landing against the wrong identifier
fails here rather than surfacing as a shifted label a week later.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import extract_embeddings as ee
import numpy as np
import pandas as pd
import pytest
import torch
import wfdb

from ecs.ingest import CANONICAL_LEADS, Corpus, load_acs

N_RECORDS = 40
MISSING_ID = "99999"


def _write_records(root: Path, n: int) -> list[str]:
    """``n`` flat WFDB records; record k sits at (k + 1) / 1000 mV in every lead."""
    directory = root / "row_data"
    directory.mkdir(parents=True)
    ids = []
    for k in range(n):
        name = f"{k:05d}"
        wfdb.wrsamp(
            name,
            fs=500,
            units=["mV"] * 12,
            sig_name=list(CANONICAL_LEADS),
            d_signal=np.full((5000, 12), k + 1, dtype=np.int16),
            fmt=["16"] * 12,
            adc_gain=[1000.0] * 12,
            baseline=[0] * 12,
            write_dir=str(directory),
        )
        ids.append(name)
    return ids


class _StubArm:
    """An encoder whose output is the tracing's mean amplitude, and which counts
    how many records it was actually asked to look at."""

    def __init__(self) -> None:
        self.n_seen = 0

    def __call__(self) -> tuple[Callable[[np.ndarray], torch.Tensor], dict[str, object]]:
        def embed(x: np.ndarray) -> torch.Tensor:
            self.n_seen += len(x)
            return torch.from_numpy(x.mean(axis=(1, 2), dtype=np.float64)[:, None])

        return embed, {
            "n_params": 0,
            "weights_source": "none, a stub",
            "preprocessing": "none",
            "notes": "mean amplitude, one number per tracing",
        }


@dataclass
class Fixture:
    """A synthetic corpus, the stub arm reading it, and where the output goes."""

    ids: list[str]
    load: Callable[[Sequence[str]], Corpus]
    stub: _StubArm
    out: Path


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    ids = _write_records(tmp_path, N_RECORDS) + [MISSING_ID]
    table = pd.DataFrame({"ecg_row_record": [f"{i}.dat" for i in ids]})
    stub = _StubArm()
    monkeypatch.setitem(ee.ARMS, "stub", stub)
    monkeypatch.setitem(ee.PRETRAINING, "stub", "none")
    monkeypatch.setattr(ee, "CHUNK", 10)

    def load(wanted: Sequence[str]) -> Corpus:
        return load_acs(table, root=tmp_path, ids=list(wanted))

    return Fixture(ids, load, stub, tmp_path / "embeddings")


def _run(corpus: Fixture, overwrite: bool = False) -> Path:
    ee.extract("stub", "acs", corpus.ids, corpus.load, corpus.out, overwrite=overwrite)
    return corpus.out / "stub/acs.npz"


class TestTheArrayItWrites:
    def test_every_row_carries_the_amplitude_its_identifier_implies(self, corpus: Fixture) -> None:
        with np.load(_run(corpus), allow_pickle=False) as data:
            ids, embedding = list(data["ids"]), data["embedding"]
        assert embedding.shape == (N_RECORDS, 1)
        assert embedding.dtype == np.float32
        expected = np.array([(int(i) + 1) / 1000 for i in ids], dtype=np.float32)
        np.testing.assert_allclose(embedding[:, 0], expected, rtol=1e-5)

    def test_identifiers_keep_the_order_the_loader_yields_them_in(self, corpus: Fixture) -> None:
        with np.load(_run(corpus), allow_pickle=False) as data:
            assert list(data["ids"]) == [f"{k:05d}" for k in range(N_RECORDS)]

    def test_a_record_that_cannot_be_read_is_absent_and_named_in_the_sidecar(
        self, corpus: Fixture
    ) -> None:
        out = _run(corpus)
        with np.load(out, allow_pickle=False) as data:
            assert MISSING_ID not in list(data["ids"])
        sidecar = json.loads((out.with_suffix(".json")).read_text())
        assert sidecar["n_records"] == N_RECORDS + 1
        assert sidecar["n_kept"] == N_RECORDS
        assert sidecar["n_excluded"] == 1
        assert "unreadable" in sidecar["excluded"][MISSING_ID]


class TestTheSidecar:
    def test_it_carries_every_field_a_result_has_to_be_able_to_state(self, corpus: Fixture) -> None:
        sidecar = json.loads(_run(corpus).with_suffix(".json").read_text())
        assert set(sidecar) >= {
            "arm",
            "corpus",
            "weights_source",
            "pretraining_corpora",
            "preprocessing",
            "git_commit",
            "n_records",
            "n_kept",
            "n_excluded",
            "excluded",
            "ingestion_deviations",
            "embedding_dim",
            "seconds",
            "machine",
        }
        assert sidecar["arm"] == "stub"
        assert sidecar["corpus"] == "acs"
        assert sidecar["embedding_dim"] == 1
        assert sidecar["machine"]["torch"] == torch.__version__


class TestRerunning:
    def test_an_existing_output_is_left_untouched_and_nothing_is_recomputed(
        self, corpus: Fixture
    ) -> None:
        out = _run(corpus)
        before, stamp = out.read_bytes(), out.stat().st_mtime_ns
        seen = corpus.stub.n_seen
        assert seen == N_RECORDS

        _run(corpus)
        assert out.read_bytes() == before
        assert out.stat().st_mtime_ns == stamp
        assert corpus.stub.n_seen == seen

    def test_overwrite_recomputes_the_same_array(self, corpus: Fixture) -> None:
        out = _run(corpus)
        before = out.read_bytes()
        _run(corpus, overwrite=True)
        assert corpus.stub.n_seen == 2 * N_RECORDS
        with (
            np.load(out, allow_pickle=False) as data,
            np.load(io.BytesIO(before), allow_pickle=False) as first,
        ):
            np.testing.assert_array_equal(data["embedding"], first["embedding"])
            np.testing.assert_array_equal(data["ids"], first["ids"])

    def test_a_finished_chunk_survives_an_interruption_and_is_not_recomputed(
        self, corpus: Fixture, tmp_path: Path
    ) -> None:
        # The first chunk, already on disk from a run that died before the end,
        # carrying a value the encoder would never produce.
        partial = corpus.out / "stub/acs.partial"
        partial.mkdir(parents=True)
        sentinel = np.full((10, 1), -7.0, dtype=np.float32)
        np.savez(
            partial / "00000000.npz",
            ids=np.array([f"{k:05d}" for k in range(10)], dtype=str),
            embedding=sentinel,
            excluded=json.dumps({}),
            deviations=np.array([], dtype=str),
        )

        with np.load(_run(corpus), allow_pickle=False) as data:
            embedding, ids = data["embedding"], list(data["ids"])
        assert corpus.stub.n_seen == N_RECORDS - 10
        np.testing.assert_array_equal(embedding[:10], sentinel)
        assert ids == [f"{k:05d}" for k in range(N_RECORDS)]
        expected = np.array([(k + 1) / 1000 for k in range(10, N_RECORDS)], dtype=np.float32)
        np.testing.assert_allclose(embedding[10:, 0], expected, rtol=1e-5)

    def test_the_partial_directory_is_gone_once_the_output_is_whole(self, corpus: Fixture) -> None:
        out = _run(corpus)
        assert out.exists()
        assert not out.with_suffix(".partial").exists()
