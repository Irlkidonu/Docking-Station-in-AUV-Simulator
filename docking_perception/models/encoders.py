"""Shared encoders. Both baselines and the Transformer use these.

Sharing them is what makes the comparison about *fusion* rather than about who
got the better feature extractor. B1, B2, B4 and the Transformer all see the
same visual and acoustic representations; only what happens afterwards differs.

On the deployed perception path -- pixels and ranges in, features out.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["VisualEncoder", "AcousticEncoder", "VISUAL_TOKENS", "ACOUSTIC_TOKENS"]

VISUAL_TOKENS = 48          # 8 x 6 grid
ACOUSTIC_TOKENS = 16


class VisualEncoder(nn.Module):
    """160x120 RGB -> (B, 48, dim) tokens, or a pooled vector.

    Four stride-2 stages take 120x160 to 8x10, then an adaptive pool fixes the
    grid at 6x8 = 48 tokens. Small on purpose: the scene is a dark tank with a
    lit dock, and a heavier backbone would spend capacity on texture that
    turbidity removes anyway.
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        widths = (dim // 4, dim // 2, dim, dim)
        layers: list[nn.Module] = []
        channels = 3
        for width in widths:
            layers += [nn.Conv2d(channels, width, 3, stride=2, padding=1),
                       nn.GroupNorm(min(8, width), width), nn.GELU()]
            channels = width
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((6, 8))
        self.dim = dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.net(image))                 # (B, dim, 6, 8)
        return features.flatten(2).transpose(1, 2)            # (B, 48, dim)

    def pooled(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward(image).mean(dim=1)                # (B, dim)


class AcousticEncoder(nn.Module):
    """(B, 3, 128) FLS -> (B, 16, dim) tokens, or a pooled vector.

    Three stride-2 stages take 128 rays to 16 tokens, so each token spans eight
    adjacent rays -- about 0.065 rad, comparable to the angular extent of the
    dock's funnel edge at mid range.
    """

    def __init__(self, dim: int = 128, in_channels: int = 3) -> None:
        super().__init__()
        widths = (dim // 4, dim // 2, dim)
        layers: list[nn.Module] = []
        channels = in_channels
        for width in widths:
            layers += [nn.Conv1d(channels, width, 5, stride=2, padding=2),
                       nn.GroupNorm(min(8, width), width), nn.GELU()]
            channels = width
        self.net = nn.Sequential(*layers)
        self.dim = dim

    def forward(self, scan: torch.Tensor) -> torch.Tensor:
        return self.net(scan).transpose(1, 2)                 # (B, 16, dim)

    def pooled(self, scan: torch.Tensor) -> torch.Tensor:
        return self.forward(scan).mean(dim=1)                 # (B, dim)
