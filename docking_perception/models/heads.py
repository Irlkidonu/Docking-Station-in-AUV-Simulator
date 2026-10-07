"""Prediction heads and the yaw-aware loss.

Yaw is periodic, so it is predicted as ``(cos psi, sin psi)`` and recovered with
``atan2``. Regressing the angle directly makes -179 deg and +179 deg look two
radians apart, which pulls the network toward the mean of a discontinuity and is
exactly the naive treatment the plan forbids.

The output is therefore five numbers -- x, y, z, cos, sin -- and four
uncertainties, one per physical degree of freedom.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["PoseHead", "decode_pose", "gaussian_nll", "huber_pose_loss",
           "LOG_VAR_MIN", "LOG_VAR_MAX"]

#: Variance clamps. Without a floor the NLL rewards driving sigma to zero on
#: easy samples and the loss diverges; without a ceiling a hard sample can be
#: excused entirely. Both bounds are wide enough not to bind in practice.
LOG_VAR_MIN = -8.0
LOG_VAR_MAX = 4.0


class PoseHead(nn.Module):
    """Feature vector -> (mu_xyz, cos, sin) and optionally log-variance."""

    def __init__(self, dim: int, hidden: int = 256,
                 with_uncertainty: bool = False) -> None:
        super().__init__()
        self.with_uncertainty = with_uncertainty
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.mean = nn.Linear(hidden, 5)          # x, y, z, cos psi, sin psi
        self.log_var = nn.Linear(hidden, 4) if with_uncertainty else None
        if self.log_var is not None:
            nn.init.zeros_(self.log_var.weight)
            nn.init.zeros_(self.log_var.bias)

    def forward(self, features: torch.Tensor):
        hidden = self.trunk(features)
        mean = self.mean(hidden)
        if self.log_var is None:
            return mean, None
        return mean, self.log_var(hidden).clamp(LOG_VAR_MIN, LOG_VAR_MAX)


def decode_pose(mean: torch.Tensor) -> torch.Tensor:
    """(B, 5) -> (B, 4) with psi recovered from the cos/sin pair."""
    psi = torch.atan2(mean[:, 4], mean[:, 3])
    return torch.stack([mean[:, 0], mean[:, 1], mean[:, 2], psi], dim=1)


def _angle_delta(predicted_psi: torch.Tensor, target_psi: torch.Tensor):
    """Signed difference wrapped to (-pi, pi]."""
    return torch.atan2(torch.sin(predicted_psi - target_psi),
                       torch.cos(predicted_psi - target_psi))


def huber_pose_loss(mean: torch.Tensor, target: torch.Tensor,
                    yaw_weight: float = 1.0) -> torch.Tensor:
    """Deterministic loss for the baselines.

    Position uses Huber, which keeps a far-range outlier from dominating the
    gradient. Yaw is supervised on the unit circle: the target is turned into
    (cos, sin) and matched directly, which is well-behaved across the wrap.
    """
    position = nn.functional.huber_loss(mean[:, :3], target[:, :3],
                                        delta=0.5, reduction="mean")
    target_cos, target_sin = torch.cos(target[:, 3]), torch.sin(target[:, 3])
    norm = torch.sqrt(mean[:, 3] ** 2 + mean[:, 4] ** 2 + 1e-8)
    angle = ((mean[:, 3] / norm - target_cos) ** 2
             + (mean[:, 4] / norm - target_sin) ** 2).mean()
    return position + yaw_weight * angle


def gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor,
                 target: torch.Tensor, yaw_weight: float = 1.0) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL, one variance per physical DOF.

        L = 0.5 * sum_d [ exp(-s_d) * e_d^2 + s_d ]

    with ``s = log sigma^2``. The yaw residual is the wrapped angular error, so
    a prediction just across the +/-pi boundary is scored as close, not as
    maximally wrong.
    """
    predicted = decode_pose(mean)
    error = torch.empty_like(predicted)
    error[:, :3] = predicted[:, :3] - target[:, :3]
    error[:, 3] = _angle_delta(predicted[:, 3], target[:, 3]) * yaw_weight

    precision = torch.exp(-log_var)
    return 0.5 * (precision * error ** 2 + log_var).sum(dim=1).mean()
