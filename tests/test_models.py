"""The random-init encoder: shapes, and that freezing it leaves it deterministic."""

from __future__ import annotations

import torch

from ecs.models import ResNet1d


def test_embedding_and_logit_shapes_on_a_canonical_batch() -> None:
    model = ResNet1d(n_classes=2).eval()
    x = torch.zeros(3, 12, 5000)
    with torch.no_grad():
        assert model.embed(x).shape == (3, 256)
        assert model(x).shape == (3, 2)
    assert model.embedding_size == 256


def test_parameter_count_is_the_one_timing_json_reports() -> None:
    assert sum(p.numel() for p in ResNet1d().parameters()) == 4_082_306


def test_frozen_encoder_gives_the_same_embedding_twice() -> None:
    model = ResNet1d().eval()
    x = torch.randn(2, 12, 5000, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        torch.testing.assert_close(model.embed(x), model.embed(x))
