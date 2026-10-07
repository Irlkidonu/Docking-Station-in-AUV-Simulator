"""SE(2)-consistent dock-relative target representation.

The historical label ``(x_D, y_D, z_D, psi)`` expresses vehicle position in
the dock frame and vehicle yaw relative to the approach heading.  V3 expresses
the horizontal translation in the vehicle/approach-relative frame:

    r = R(-psi) t_D,       t_D = R(psi) r.

This keeps longitudinal, lateral and relative-yaw geometry coupled by a rigid
SE(2) transform.  Vertical displacement is unchanged.
"""
from __future__ import annotations

import numpy as np


def wrap_angle(angle):
    """Wrap scalar or array angles to [-pi, pi)."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def dock_to_body_pose(pose):
    """Map ``[..., x_D, y_D, z_D, psi]`` to ``[..., r_x, r_y, z_D, psi]``."""
    p = np.asarray(pose)
    if p.shape[-1] != 4:
        raise ValueError("pose must have final dimension 4")
    psi = wrap_angle(p[..., 3])
    c, s = np.cos(psi), np.sin(psi)
    out = np.empty_like(p, dtype=np.result_type(p, np.float64))
    out[..., 0] = c * p[..., 0] + s * p[..., 1]
    out[..., 1] = -s * p[..., 0] + c * p[..., 1]
    out[..., 2] = p[..., 2]
    out[..., 3] = psi
    return out


def body_to_dock_pose(pose):
    """Inverse map ``[..., r_x, r_y, z_D, psi]`` to dock-frame pose."""
    p = np.asarray(pose)
    if p.shape[-1] != 4:
        raise ValueError("pose must have final dimension 4")
    psi = wrap_angle(p[..., 3])
    c, s = np.cos(psi), np.sin(psi)
    out = np.empty_like(p, dtype=np.result_type(p, np.float64))
    out[..., 0] = c * p[..., 0] - s * p[..., 1]
    out[..., 1] = s * p[..., 0] + c * p[..., 1]
    out[..., 2] = p[..., 2]
    out[..., 3] = psi
    return out


def horizontal_covariance_to_dock(body_pose, covariance):
    """Transform covariance of ``[r_x,r_y,psi]`` into ``[x_D,y_D,psi]``.

    Uses the exact first-order Jacobian of ``t_D=R(psi)r``.  ``covariance``
    may be a single 3x3 matrix or a batch matching ``body_pose``.
    """
    p = np.asarray(body_pose, dtype=float)
    q = np.asarray(covariance, dtype=float)
    if p.shape[-1] != 4 or q.shape[-2:] != (3, 3):
        raise ValueError("expected pose [...,4] and covariance [...,3,3]")
    rx, ry, psi = p[..., 0], p[..., 1], wrap_angle(p[..., 3])
    c, s = np.cos(psi), np.sin(psi)
    j = np.zeros(p.shape[:-1] + (3, 3), dtype=float)
    j[..., 0, 0], j[..., 0, 1] = c, -s
    j[..., 0, 2] = -s * rx - c * ry
    j[..., 1, 0], j[..., 1, 1] = s, c
    j[..., 1, 2] = c * rx - s * ry
    j[..., 2, 2] = 1.0
    return j @ q @ np.swapaxes(j, -1, -2)
