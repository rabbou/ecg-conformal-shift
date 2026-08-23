"""The supervised encoder: a small one-dimensional ResNet over 12-lead ECG.

Frozen at its random initialisation it is the floor every pre-trained arm has
to clear (C-17); trained from scratch on PTB-XL it is the supervised baseline
whose MI AUROC is checked against the published benchmark (C-19).  The shape
follows the time-series ResNet of Wang et al. 2017 that the PTB-XL benchmark
uses: a wide stem, four stages of two residual blocks, global average pooling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import N_LEADS

__all__ = ["ResNet1d", "ResidualBlock"]


class ResidualBlock(nn.Module):
    """Two convolutions with a skip connection; the first may downsample."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.body(x) + self.skip(x))


class ResNet1d(nn.Module):
    """Input (N, 12, T) in millivolts; ``embed`` gives (N, channels[-1]),
    ``forward`` gives class logits (N, n_classes)."""

    def __init__(
        self,
        n_classes: int = 2,
        channels: tuple[int, ...] = (64, 128, 256, 256),
        kernel_size: int = 7,
        in_channels: int = N_LEADS,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        blocks: list[nn.Module] = []
        width = channels[0]
        for i, out in enumerate(channels):
            stride = 1 if i == 0 else 2
            blocks.append(ResidualBlock(width, out, stride, kernel_size))
            blocks.append(ResidualBlock(out, out, 1, kernel_size))
            width = out
        self.stages = nn.Sequential(*blocks)
        self.head = nn.Linear(width, n_classes)
        self.embedding_size = width

    def embed(self, x: Tensor) -> Tensor:
        return self.stages(self.stem(x)).mean(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.embed(x))
