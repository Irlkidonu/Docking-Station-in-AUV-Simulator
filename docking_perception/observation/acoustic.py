"""Acoustic observation: 128 raw FLS ranges -> network input.

On the deployed perception path; sees ranges and nothing else.

The sensor is a **128 x 1 ray-cast forward-looking acoustic proxy**, not an
imaging sonar, so it is encoded as a 1-D signal. Reshaping it into a 2-D
"sonar image" would invent structure the sensor never measured; the earlier
prototype did that and its results do not carry over.

Three channels, because a bare range vector loses two things a model would
otherwise have to rediscover:

* **normalised range** -- the measurement;
* **validity** -- a non-return is informative (no surface within 12 m), and
  substituting a number would make absence indistinguishable from a far echo;
* **range gradient** -- the discontinuity at a funnel edge is what carries
  bearing information, and finite differences make it directly available.
"""

from __future__ import annotations

import numpy as np

__all__ = ["encode", "FLS_CHANNELS", "RANGE_MAX_M"]

RANGE_MAX_M = 12.0
FLS_CHANNELS = 3


def encode(fls_range: np.ndarray, fls_valid: np.ndarray) -> np.ndarray:
    """(128,) ranges + validity -> (3, 128) float32.

    Invalid rays are filled at the maximum range for the range channel and
    flagged in the validity channel, so the network can tell "nothing out there"
    from "something at 12 m" instead of guessing.
    """
    ranges = np.asarray(fls_range, dtype=np.float32)
    valid = np.asarray(fls_valid, dtype=bool)

    filled = np.where(valid & np.isfinite(ranges), ranges, RANGE_MAX_M)
    filled = np.clip(filled, 0.0, RANGE_MAX_M)
    normalised = filled / RANGE_MAX_M

    gradient = np.gradient(normalised).astype(np.float32)

    return np.stack([normalised.astype(np.float32),
                     valid.astype(np.float32),
                     gradient], axis=0)


def ray_angles(count: int = 128, angle_min: float = -0.52,
               angle_max: float = 0.52) -> np.ndarray:
    """Bearing of each ray. Used for positional encoding, not as a feature."""
    return np.linspace(angle_min, angle_max, count).astype(np.float32)
