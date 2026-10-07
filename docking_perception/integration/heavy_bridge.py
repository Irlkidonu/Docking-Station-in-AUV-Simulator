"""Observable calibration bridge from the Heavy sensors to frozen A/B tensors.

This module contains calibration and deterministic sensor preprocessing only.  It has
no simulator-state, pose, label, scenario, or truth input.  Heavy images already carry
their rendered water/lighting effects, so the bridge never accepts a turbidity value
and never applies the A/B post-render degradation a second time.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from docking_perception.observation.acoustic import encode
from uuv_sim_physics.sensors.oculus_m750d import AcousticImage

__all__ = ["CameraCalibration", "AB_CAMERA", "HEAVY_CAMERA",
           "HeavyObservationBridge", "rectification_maps",
           "oculus_image_to_network", "oculus_image_to_network_v2"]


def oculus_image_to_network(image: AcousticImage) -> np.ndarray:
    """Reduce an M750d acoustic image to the deployed 3x128 interface.

    Only quantised intensity, its detection mask, and declared image geometry
    are consumed.  The central 60-degree field matches the original model's
    angular support. Within each output angular cell an intensity-weighted
    median detection range is used; cells without detections remain invalid.
    Renderer ray intersections and privileged target state are unavailable.
    """
    intensity = np.asarray(image.intensity)
    detected = np.asarray(image.detection, bool)
    if intensity.shape != detected.shape or intensity.ndim != 2:
        raise ValueError("Oculus intensity/detection arrays must be 2-D and aligned")
    if intensity.shape[0] != 512:
        raise ValueError("frozen Heavy Oculus interface requires 512 beams")
    bearings = image.angle_min_rad + np.arange(512) * image.angle_step_rad
    edges = np.linspace(-np.deg2rad(30), np.deg2rad(30), 129)
    ranges = np.full(128, np.nan, np.float32)
    valid = np.zeros(128, bool)
    for output in range(128):
        beam_mask = ((bearings >= edges[output]) &
                     (bearings < edges[output + 1] if output < 127
                      else bearings <= edges[output + 1]))
        beams, bins = np.nonzero(detected[beam_mask])
        if bins.size == 0:
            continue
        source_beams = np.flatnonzero(beam_mask)[beams]
        weights = intensity[source_beams, bins].astype(np.float64) + 1.0
        observed_ranges = bins.astype(np.float64) * image.range_resolution_m
        order = np.argsort(observed_ranges)
        cumulative = np.cumsum(weights[order])
        pick = order[np.searchsorted(cumulative, 0.5 * cumulative[-1])]
        ranges[output] = observed_ranges[pick]
        valid[output] = True
    return encode(ranges, valid)


def oculus_image_to_network_v2(image: AcousticImage) -> np.ndarray:
    """Intensity-preserving Heavy-domain Oculus representation.

    The shape remains 3x128, so the fundamental acoustic encoder and fusion
    architecture are unchanged. Channels are normalized detected range,
    normalized peak return intensity (zero means unavailable), and the spatial
    gradient of range. All values derive from the quantized acoustic image.
    """
    intensity = np.asarray(image.intensity)
    detected = np.asarray(image.detection, bool)
    if intensity.shape != detected.shape or intensity.ndim != 2 or intensity.shape[0] != 512:
        raise ValueError("V2 Oculus observation must be aligned 512-beam intensity/detection")
    bearings = image.angle_min_rad + np.arange(512) * image.angle_step_rad
    edges = np.linspace(-np.deg2rad(30), np.deg2rad(30), 129)
    ranges = np.full(128, 12.0, np.float32)
    strength = np.zeros(128, np.float32)
    for output in range(128):
        beam_mask = ((bearings >= edges[output]) &
                     (bearings < edges[output + 1] if output < 127
                      else bearings <= edges[output + 1]))
        beams, bins = np.nonzero(detected[beam_mask])
        if bins.size == 0:
            continue
        source_beams = np.flatnonzero(beam_mask)[beams]
        weights = intensity[source_beams, bins].astype(np.float64) + 1.0
        observed_ranges = bins.astype(np.float64) * image.range_resolution_m
        order = np.argsort(observed_ranges); cumulative = np.cumsum(weights[order])
        ranges[output] = observed_ranges[order[np.searchsorted(cumulative, .5*cumulative[-1])]]
        strength[output] = float(intensity[source_beams, bins].max()) / 255.0
    normalized = np.clip(ranges / 12.0, 0.0, 1.0)
    return np.stack([normalized, strength,
                     np.gradient(normalized).astype(np.float32)]).astype(np.float32)


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    horizontal_fov_rad: float
    pitch_rad: float

    @property
    def focal_px(self) -> float:
        return 0.5 * self.width / math.tan(0.5 * self.horizontal_fov_rad)

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.width - 1.0) / 2.0, (self.height - 1.0) / 2.0)


# Values come directly from the two simulator vehicle configurations.  AB_CAMERA is
# expressed at the network resolution; its focal length is the native 640x480 focal
# length scaled by four in both axes.
AB_CAMERA = CameraCalibration(160, 120, 1.047, 0.05)
HEAVY_CAMERA = CameraCalibration(1920, 1080, 1.9199, 0.0)


def rectification_maps(source: CameraCalibration = HEAVY_CAMERA,
                       target: CameraCalibration = AB_CAMERA) -> tuple[np.ndarray,
                                                                       np.ndarray]:
    """Return output-to-source maps matching the trained A/B ray geometry.

    The cameras share the declared x/y/z mount translation.  Their only extrinsic
    difference is pitch, so each target pixel is converted to an angular ray, rotated
    by the declared pitch difference, and projected into the Heavy image.
    """
    u, v = np.meshgrid(np.arange(target.width, dtype=np.float64),
                       np.arange(target.height, dtype=np.float64))
    tcx, tcy = target.centre
    scx, scy = source.centre
    tx = np.arctan((u - tcx) / target.focal_px)
    ty = np.arctan((v - tcy) / target.focal_px) + (target.pitch_rad - source.pitch_rad)
    map_x = scx + source.focal_px * np.tan(tx)
    map_y = scy + source.focal_px * np.tan(ty)
    return map_x.astype(np.float32), map_y.astype(np.float32)


class HeavyObservationBridge:
    """Calibrate Heavy camera/FLS observations for the frozen Transformer."""

    heavy_fls_rays = 128
    heavy_fls_angle_min_rad = -0.52
    heavy_fls_angle_max_rad = 0.52
    heavy_fls_range_max_m = 12.0

    def __init__(self) -> None:
        self._map_x, self._map_y = rectification_maps()

    def camera_uint8(self, heavy_rgb: np.ndarray) -> np.ndarray:
        if heavy_rgb.shape != (HEAVY_CAMERA.height, HEAVY_CAMERA.width, 3):
            raise ValueError(f"Heavy camera shape {heavy_rgb.shape} does not match "
                             f"{(HEAVY_CAMERA.height, HEAVY_CAMERA.width, 3)}")
        if heavy_rgb.dtype != np.uint8:
            raise ValueError("Heavy camera must be uint8 RGB")
        return cv2.remap(heavy_rgb, self._map_x, self._map_y,
                         interpolation=cv2.INTER_AREA,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def network_image(self, heavy_rgb: np.ndarray) -> np.ndarray:
        calibrated = self.camera_uint8(heavy_rgb)
        return np.transpose(calibrated.astype(np.float32) / 255.0, (2, 0, 1))

    def network_fls(self, ranges: np.ndarray, angle_min_rad: float,
                    angle_max_rad: float) -> np.ndarray:
        ranges = np.asarray(ranges, dtype=np.float32)
        if ranges.shape != (self.heavy_fls_rays,):
            raise ValueError("Heavy FLS must contain exactly 128 rays")
        if not math.isclose(float(angle_min_rad), self.heavy_fls_angle_min_rad,
                            abs_tol=1e-6) or not math.isclose(
                                float(angle_max_rad), self.heavy_fls_angle_max_rad,
                                abs_tol=1e-6):
            raise ValueError("Heavy FLS bearings do not match the frozen A/B fan")
        valid = np.isfinite(ranges) & (ranges > 0.2) & (
            ranges < self.heavy_fls_range_max_m)
        return encode(ranges, valid)
