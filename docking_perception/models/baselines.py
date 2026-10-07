"""B1, B2 and B4 — the learned baselines.

All three reuse the encoders in ``encoders.py``, so any difference between them
and the Transformer is a difference in *fusion*, not in feature extraction.

B4 is the one that matters. It is given the same encoders, the same head
capacity and the same training budget as the Transformer, and is deliberately
not handicapped: if simple concatenation is enough, that is the finding.
"""

from __future__ import annotations

import torch
from torch import nn

from .encoders import AcousticEncoder, VisualEncoder
from .heads import PoseHead

__all__ = ["OpticalOnly", "AcousticOnly", "LateFusionMLP", "build"]


class OpticalOnly(nn.Module):
    """B1 — degraded RGB only."""

    name = "B1_optical"

    def __init__(self, dim: int = 128, with_uncertainty: bool = False) -> None:
        super().__init__()
        self.visual = VisualEncoder(dim)
        self.head = PoseHead(dim, with_uncertainty=with_uncertainty)

    def forward(self, image, scan=None):
        return self.head(self.visual.pooled(image))


class AcousticOnly(nn.Module):
    """B2 — 128-ray FLS only."""

    name = "B2_fls"

    def __init__(self, dim: int = 128, with_uncertainty: bool = False) -> None:
        super().__init__()
        self.acoustic = AcousticEncoder(dim)
        self.head = PoseHead(dim, with_uncertainty=with_uncertainty)

    def forward(self, image, scan):
        return self.head(self.acoustic.pooled(scan))


class LateFusionMLP(nn.Module):
    """B4 — pooled features concatenated, then an MLP.

    The critical comparator. Same encoders as the Transformer; the only
    difference is that fusion happens once, on pooled vectors, with no
    token-level interaction.
    """

    name = "B4_late_fusion"

    def __init__(self, dim: int = 128, hidden: int = 256,
                 with_uncertainty: bool = False) -> None:
        super().__init__()
        self.visual = VisualEncoder(dim)
        self.acoustic = AcousticEncoder(dim)
        self.fuse = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim), nn.GELU())
        self.head = PoseHead(dim, hidden=hidden, with_uncertainty=with_uncertainty)

    def forward(self, image, scan):
        joint = torch.cat([self.visual.pooled(image),
                           self.acoustic.pooled(scan)], dim=1)
        return self.head(self.fuse(joint))


def build(name: str, **kwargs) -> nn.Module:
    table = {"B1_optical": OpticalOnly, "B2_fls": AcousticOnly,
             "B4_late_fusion": LateFusionMLP}
    if name not in table:
        raise KeyError(f"unknown baseline {name}; have {sorted(table)}")
    return table[name](**kwargs)
