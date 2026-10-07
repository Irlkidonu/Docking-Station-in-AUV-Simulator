"""V4 fusion with V3 SE(2) translation resolved by observable IMU yaw."""
from __future__ import annotations

import math

import numpy as np

from docking_perception.integration import heavy_v5_protocol as P
from docking_perception.integration.heavy_bridge import oculus_image_to_network_v2
from docking_perception.integration.heavy_v4_terminal_provider import HeavyV4TerminalProvider
from uuv_sim_physics.sensors.oculus_m750d import extract_observable_features


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def decode_with_imu_yaw(se2, sigma_se2, yaw_absolute,
                        approach_yaw=math.pi):
    """Rotate native SE(2) translation using onboard IMU attitude.

    ``yaw_absolute`` is obtained from the operational IMU quaternion in the
    lock-step runner.  The network yaw remains a diagnostic output only.
    """
    psi = wrap(float(yaw_absolute) - float(approach_yaw))
    c, s = math.cos(psi), math.sin(psi)
    rotation = np.array([[c, -s], [s, c]])
    translation = rotation @ np.asarray(se2[:2], dtype=float)
    covariance = (rotation
                  @ np.diag(np.asarray(sigma_se2[:2], dtype=float) ** 2)
                  @ rotation.T)
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return translation, sigma, psi


class HeavyV5TerminalProvider(HeavyV4TerminalProvider):
    def __init__(self, socket_path=P.DEFAULT_SOCKET, connect_timeout_s=60.0):
        super().__init__(socket_path, connect_timeout_s)

    def _observe(self, obs):
        if obs.image is None or obs.fls_ranges is None:
            return None, "no_synchronized_observation"
        fan = np.asarray(obs.fls_ranges, np.float32)
        if fan.size != 512 * 9:
            raise ValueError(
                f"V5 requires 512x9 Oculus intersections, got {fan.size}")
        acquisition_t = float(obs.fls_acquisition_t
                              if obs.fls_acquisition_t is not None else obs.t)
        acoustic = self.renderer.render(
            fan.reshape(9, 512), acquisition_t=acquisition_t)
        # Non-operational instrumentation for the interactive demonstrator.
        # The array is never read by estimation or control.
        self._last_acoustic_image = acoustic
        support = extract_observable_features(acoustic)
        if not support["valid"]:
            return None, "no_oculus_support"
        scan = oculus_image_to_network_v2(acoustic)
        camera = self.bridge.camera_uint8(obs.image)
        if self._sock is None:
            self._connect()
        self._sock.sendall(P.pack_request(camera, scan))
        se2, sigma = P.unpack_response(P.recv_exact(self._sock,
                                                    P.RESPONSE_BYTES))
        z, sg, psi_imu = decode_with_imu_yaw(se2, sigma, obs.yaw)
        if not np.isfinite(z).all() or not np.isfinite(sg).all():
            return None, "nonfinite_v5"
        if np.any(sg > self.SIGMA_LIMIT_M):
            return None, "v3_sigma_above_limit"
        support = {
            **support,
            "network_yaw_rad": float(se2[3]),
            "imu_relative_yaw_rad": psi_imu,
            "translation_rotation_source": "operational_imu_quaternion",
        }
        return (z, sg, support), "candidate"
