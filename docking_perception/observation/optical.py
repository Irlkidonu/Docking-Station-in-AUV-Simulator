"""Optical observation: rendered frame -> degraded network input.

On the deployed perception path. Takes pixels and a scalar standoff, returns
pixels. It has no access to vehicle or dock pose and must never acquire any --
``privileged.PERCEPTION_PATH`` lists this module, so a privileged import here
fails the leakage test.

The water-column model itself is the simulator's, unchanged. Turbidity is
applied at *load* time rather than at capture time so one physics run supports
every water condition, including the held-out extreme that was never generated.
"""

from __future__ import annotations

import numpy as np

from uuv_sim_physics.sensors import WaterColumn, degrade

__all__ = ["degrade_frame", "to_network_input", "TURBIDITY_POINTS",
           "HELD_OUT_C"]

#: Reporting operating points. Continuous values are used for training; these
#: are the bins results are read out in.
TURBIDITY_POINTS = {"clear": 0.20, "moderate": 0.60, "high": 1.20}

#: Never generated, never trained on, never used for selection.
HELD_OUT_C = 2.00


def degrade_frame(camera_raw: np.ndarray, turbidity_c: float,
                  standoff_m: float) -> np.ndarray:
    """Apply the water column to one rendered frame.

    ``standoff_m`` is a scalar taken at capture time from the nearest central
    FLS return. The simulator exposes no depth buffer, so degradation is
    uniform across the frame rather than per-pixel. This is a documented
    approximation and must not be described as depth-aware rendering.
    """
    return degrade(camera_raw, WaterColumn(beam_attenuation=float(turbidity_c)),
                   range_m=float(standoff_m))


def to_network_input(camera_raw: np.ndarray, turbidity_c: float,
                     standoff_m: float) -> np.ndarray:
    """Degraded frame as CHW float32 in [0, 1]."""
    degraded = degrade_frame(camera_raw, turbidity_c, standoff_m)
    return np.transpose(degraded.astype(np.float32) / 255.0, (2, 0, 1))
