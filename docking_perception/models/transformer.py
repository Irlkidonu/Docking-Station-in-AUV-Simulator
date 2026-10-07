"""P3 — the compact multimodal fusion Transformer.

64 tokens total (48 visual + 16 acoustic) plus a learned ``[DOCK]`` query, at
dimension 128 over 4 layers with 4 heads. Deliberately small: the inputs are a
160x120 image of a dark tank and a 128-element range vector, and a larger model
would be testing optimisation rather than fusion.

Two pieces of structure the baselines do not have:

* **modality embeddings** -- a learned vector added to every token of a modality,
  so attention can condition on where information came from;
* **angular positional encoding for the acoustic tokens** -- each token spans
  eight rays with a known bearing, and that bearing is what makes a range
  discontinuity interpretable as a direction.

Everything else -- encoders, head capacity, loss, normalisation, splits,
training budget -- is identical to B4, so the comparison isolates token-level
cross-modal attention.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .encoders import ACOUSTIC_TOKENS, VISUAL_TOKENS, AcousticEncoder, VisualEncoder
from .heads import PoseHead

__all__ = ["DockingTransformer"]


class DockingTransformer(nn.Module):

    name = "T_transformer"

    def __init__(self, dim: int = 128, depth: int = 4, heads: int = 4,
                 hidden: int = 256, with_uncertainty: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.visual = VisualEncoder(dim)
        self.acoustic = AcousticEncoder(dim)

        self.visual_pos = nn.Parameter(torch.zeros(1, VISUAL_TOKENS, dim))
        self.modality = nn.Parameter(torch.zeros(2, dim))
        self.dock_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.visual_pos, std=0.02)
        nn.init.trunc_normal_(self.modality, std=0.02)
        nn.init.trunc_normal_(self.dock_token, std=0.02)

        self.register_buffer("acoustic_pos",
                             self._angular_encoding(ACOUSTIC_TOKENS, dim),
                             persistent=False)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=4 * dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = PoseHead(dim, hidden=hidden, with_uncertainty=with_uncertainty)

    @staticmethod
    def _angular_encoding(count: int, dim: int) -> torch.Tensor:
        """Sinusoidal encoding of each acoustic token's mean bearing.

        Fixed rather than learned: the ray geometry is known exactly from the
        sensor definition, so there is nothing to infer.
        """
        angles = torch.linspace(-0.52, 0.52, count).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, dim, 2).float()
                              * (-math.log(10000.0) / dim))
        encoding = torch.zeros(1, count, dim)
        encoding[0, :, 0::2] = torch.sin(angles * frequency * 10.0)
        encoding[0, :, 1::2] = torch.cos(angles * frequency * 10.0)
        return encoding

    def forward(self, image: torch.Tensor, scan: torch.Tensor):
        batch = image.shape[0]
        visual = self.visual(image) + self.visual_pos + self.modality[0]
        acoustic = self.acoustic(scan) + self.acoustic_pos + self.modality[1]

        query = self.dock_token.expand(batch, -1, -1)
        tokens = torch.cat([query, visual, acoustic], dim=1)
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded[:, 0]))

    def attention_share(self, image: torch.Tensor, scan: torch.Tensor) -> dict:
        """Fraction of the [DOCK] query's final-layer attention per modality.

        ``nn.TransformerEncoder`` discards attention weights, so the last layer
        is re-run by hand with ``need_weights=True``. Only the weights are taken;
        the layer's output is not used, so this cannot perturb a prediction.

        **Supporting evidence only.** Attention mass is not proof of reliance --
        a head can attend to tokens it then ignores. Modality dropout is the
        causal test; this shows whether the mechanism moves in the same
        direction.
        """
        batch = image.shape[0]
        visual = self.visual(image) + self.visual_pos + self.modality[0]
        acoustic = self.acoustic(scan) + self.acoustic_pos + self.modality[1]
        tokens = torch.cat([self.dock_token.expand(batch, -1, -1),
                            visual, acoustic], dim=1)

        for layer in self.encoder.layers[:-1]:
            tokens = layer(tokens)

        last = self.encoder.layers[-1]
        normed = last.norm1(tokens)          # norm_first=True
        _, weights = last.self_attn(normed, normed, normed,
                                    need_weights=True, average_attn_weights=True)

        query_row = weights[:, 0, :]         # what [DOCK] attends to
        visual_share = query_row[:, 1:1 + VISUAL_TOKENS].sum(dim=1)
        acoustic_share = query_row[:, 1 + VISUAL_TOKENS:].sum(dim=1)
        total = visual_share + acoustic_share + 1e-9
        return {"visual": float((visual_share / total).mean()),
                "acoustic": float((acoustic_share / total).mean()),
                "self": float(query_row[:, 0].mean())}
