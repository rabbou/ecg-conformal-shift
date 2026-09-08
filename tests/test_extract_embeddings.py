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
import re
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


class TestTheUntrainedControlIsOneNetwork:
    """The random-init arm is the only one that draws its weights instead of
    loading them, and ``extract`` builds an arm once per corpus.  If the draw
    is not pinned, every corpus goes through a different network and the probe
    fitted on one corpus is spent in another corpus's feature space -- which is
    how that arm came to score below chance on a corpus it had never seen.
    """

    def test_two_builds_give_the_same_weights(self) -> None:
        from ecs.encoders import ARMS

        # Whatever consumed the global generator before a build must not reach
        # the weights: the two builds below are seeded differently on purpose.
        torch.manual_seed(12345)
        embed_a, meta_a = ARMS["random_init"]()
        torch.manual_seed(999)
        embed_b, meta_b = ARMS["random_init"]()

        x = np.zeros((2, 12, 5000), dtype=np.float32)
        x[1] = 1.0
        with torch.no_grad():
            np.testing.assert_array_equal(embed_a(x).numpy(), embed_b(x).numpy())
        assert meta_a["weights_source"] == meta_b["weights_source"]
        assert "seed unset" not in str(meta_a["weights_source"])


class TestNoThirdPartyWeightOpensUnchecked:
    """A .pth is a pickle, so opening one runs what it says to run. Every
    third-party weight file is checked against the digest it had when it was
    fetched, and the ECGFounder checkpoint really does carry the opcodes that
    make that matter (GLOBAL resolves a name the file chooses, REDUCE calls it).
    """

    def test_no_load_site_opens_a_file_that_was_neither_checked_nor_restricted(self) -> None:
        """Either torch is told to read weights only, or the digest was checked
        first. ECGFounder needs the second: weights_only=True refuses that
        checkpoint on torch 2.2.2, so the manifest is what guards it."""
        for module in (
            "src/ecs/encoders.py",
            "scripts/score_external.py",
            "scripts/timing_probe.py",
        ):
            source = (Path(__file__).resolve().parents[1] / module).read_text()
            for match in re.finditer(r"torch\.load\(([^)]*)\)", source):
                call = match.group(1)
                preceding = source[max(0, match.start() - 400) : match.start()]
                assert "weights_only=True" in call or "verified(" in preceding, (module, call)

    def test_the_file_transformers_executes_is_in_the_manifest(self) -> None:
        from ecs import encoders

        for member in encoders.HUBERT_FILES:
            assert member in encoders.WEIGHT_SHA256
        assert "hubert-ecg-base/hubert_ecg.py" in encoders.HUBERT_FILES

    def test_a_changed_file_is_refused_rather_than_opened(self, tmp_path: Path) -> None:
        from ecs import encoders

        name = "ecgfounder/12_lead_ECGFounder.pth"
        planted = tmp_path / name
        planted.parent.mkdir(parents=True)
        planted.write_bytes(b"not the published checkpoint")
        monkey = encoders.WEIGHTS
        try:
            encoders.WEIGHTS = tmp_path
            with pytest.raises(RuntimeError, match="expected"):
                encoders.verified(name)
        finally:
            encoders.WEIGHTS = monkey
