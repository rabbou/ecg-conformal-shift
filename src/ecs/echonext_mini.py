"""The published EchoNext mini-model, rebuilt here to run its own checkpoint."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import N_LEADS

__all__ = ["EchoNextMini"]


class _MiniBlock(nn.Module):
    """A residual block named as the published checkpoint names its tensors."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))
        skip = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + skip)


class EchoNextMini(nn.Module):
    """The published EchoNext mini-model, for inference on its own checkpoint.

    Written here from the shapes its checkpoint holds, not copied: a ResNet-34
    layout (3, 4, 6, 3 blocks, kernel 7) over (N, 12, 2500) at 250 Hz, average
    and maximum pooling side by side, and the seven tabular features of the
    distribution concatenated before one linear layer with twelve outputs.
    Tensor names match the checkpoint so it loads with ``strict=True``.
    """

    def __init__(self, width: int = 16, n_tabular: int = 7, n_classes: int = 12) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(N_LEADS, width, 15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        stages = []
        channels = width
        for i, depth in enumerate((3, 4, 6, 3)):
            out = width * 2**i
            blocks = [_MiniBlock(channels, out, 1 if i == 0 else 2)]
            blocks += [_MiniBlock(out, out, 1) for _ in range(depth - 1)]
            stages.append(nn.Sequential(*blocks))
            channels = out
        self.layer1, self.layer2, self.layer3, self.layer4 = stages
        self.output = nn.Linear(2 * channels + n_tabular, n_classes)

    def forward(self, x: Tensor, tabular: Tensor) -> Tensor:
        x = self.maxpool(torch.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        return self.output(torch.cat([pooled, tabular], dim=1))
