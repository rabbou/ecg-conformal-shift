"""The random-init encoder: shapes, and that freezing it leaves it deterministic."""

from __future__ import annotations

import torch

from ecs.echonext_mini import EchoNextMini
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


def test_the_mini_model_takes_tracing_and_tabular_and_gives_twelve_logits() -> None:
    model = EchoNextMini().eval()
    with torch.no_grad():
        assert model(torch.zeros(3, 12, 2500), torch.zeros(3, 7)).shape == (3, 12)


def test_the_mini_model_names_as_many_tensors_as_its_checkpoint() -> None:
    """The published weights.pt holds 218 tensors, buffers included."""
    assert len(EchoNextMini().state_dict()) == 218
