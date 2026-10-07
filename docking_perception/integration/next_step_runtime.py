"""Truth-free command bridge and synchronized recorder for the rich-world demo.

This module is engineering infrastructure, not a scientific campaign runner.
It wraps the qualified V6 provider without changing its estimator/controller
logic and stores generated data only beneath the registered external root.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from docking_perception.integration.heavy_v6_return_provider import HeavyV6ReturnProvider
from uuv_sim_physics.docking.autonomy import ObservableDockingInput
from uuv_sim_physics.docking.state_providers import ProviderEstimate
from uuv_sim_physics.sensors.particulate_water import (
    MINERAL_SILT, ParticulateWaterMedium, degrade_from_medium)


class FailSafeLogger:
    """Logging whose stdout failure can never terminate vehicle execution."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, message: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(str(message).rstrip() + "\n")
        try:
            print(message, flush=True)
        except (BrokenPipeError, OSError, ValueError):
            pass


class LiveEnvironmentCommands:
    """Thread-safe, policy-hidden environmental controls from ROS/UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._dock = False
        self._modeled_turbidity = 0.0
        self._current = (0.0, 0.0, 0.0)
        self._capture_beam_engaged = False

    def request_dock(self):
        with self._lock:
            self._dock = True

    def set_modeled_turbidity(self, value: float):
        with self._lock:
            self._modeled_turbidity = max(0.0, float(value))

    def set_current(self, xyz):
        with self._lock:
            self._current = tuple(float(x) for x in xyz)

    def set_capture_beam(self, engaged: bool):
        with self._lock:
            self._capture_beam_engaged = bool(engaged)

    def capture_beam_engaged(self):
        with self._lock:
            return self._capture_beam_engaged

    def snapshot(self):
        with self._lock:
            dock = self._dock
            self._dock = False
            return dock, self._modeled_turbidity, self._current


class CaptureConfirmationDebouncer:
    """Require persistent collar break-beam engagement before confirmation."""
    REQUIRED_CONSECUTIVE = 3  # 3 controller ticks, approximately 0.10 s

    def __init__(self):
        self.count = 0

    def observe(self, engaged: bool) -> bool:
        self.count = self.count + 1 if engaged else 0
        return self.count >= self.REQUIRED_CONSECUTIVE


class SynchronizedMultimodalRecorder:
    """Write observable streams and scoring truth into separated namespaces."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        allowed = Path(os.environ.get("DOCK_OUTPUT_ROOT", "~/.cache/docking_station")).expanduser().resolve()
        if allowed not in self.root.parents:
            raise ValueError(f"unsafe recording destination: {self.root}")
        self.camera = self.root / "observable/camera"
        self.fls = self.root / "observable/realistic_fls"
        self.camera.mkdir(parents=True, exist_ok=False)
        self.fls.mkdir(parents=True, exist_ok=False)
        (self.root / "scoring_truth").mkdir()
        self.observable_log = self.root / "observable/synchronized.jsonl"
        self.truth_log = self.root / "scoring_truth/trajectory.jsonl"
        self.count = 0

    @staticmethod
    def _append(path: Path, payload: dict):
        def safe(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [safe(item) for item in value]
            return value
        with path.open("a") as stream:
            stream.write(json.dumps(safe(payload), sort_keys=True,
                                    allow_nan=False) + "\n")

    def record_observation(self, obs, estimate, acoustic_image, environment):
        index = self.count
        self.count += 1
        camera_name = fls_name = None
        if obs.image is not None:
            camera_name = f"{index:06d}.npy"
            np.save(self.camera / camera_name, np.asarray(obs.image, np.uint8), allow_pickle=False)
        if acoustic_image is not None:
            fls_name = f"{index:06d}.npy"
            intensity = getattr(acoustic_image, "intensity", acoustic_image)
            np.save(self.fls / fls_name, np.asarray(intensity, np.float32), allow_pickle=False)
        self._append(self.observable_log, {
            "index": index, "acquisition_time_s": float(obs.t),
            "camera": camera_name, "realistic_fls": fls_name,
            "imu": {"rpy_rad": [float(obs.roll), float(obs.pitch), float(obs.yaw)],
                    "angular_rate": np.asarray(obs.angular_rate).tolist(),
                    "linear_acceleration": np.asarray(obs.linear_acceleration).tolist()},
            "depth_m": float(obs.depth_measured),
            "dvl": {"velocity": (None if obs.dvl_velocity is None else
                                  np.asarray(obs.dvl_velocity).tolist()),
                    "valid": bool(obs.dvl_valid), "beams_locked": int(obs.dvl_beams_locked),
                    "acquisition_time_s": obs.dvl_acquisition_t},
            "environment_controls": environment,
            "transformer_estimator": {"xy_m": np.asarray(estimate.xy).tolist(),
                                      "valid": bool(estimate.valid),
                                      "mode": estimate.mode,
                                      "diagnostics": estimate.detail},
        })

    def record_scoring_truth(self, row: dict):
        self._append(self.truth_log, {"t": row["t"],
                                     "dock_relative_m": row["dock_rel_true"],
                                     "capture": row["capture"]})

    def finalize(self):
        files = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.json":
                files[str(path.relative_to(self.root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.root / "MANIFEST.json").write_text(json.dumps(files, indent=2) + "\n")


class InteractiveV6Provider:
    """Apply UI controls to sensors/environment, then invoke unchanged V6."""

    hardware_realisable = True
    requires = HeavyV6ReturnProvider.requires

    def __init__(self, socket_path, initial_nav_xy, commands, recorder,
                 telemetry_callback=None, trigger="external"):
        self.delegate = HeavyV6ReturnProvider(socket_path, initial_nav_xy, trigger)
        self.commands = commands
        self.recorder = recorder
        self.telemetry_callback = telemetry_callback
        self.capture_confirmation = CaptureConfirmationDebouncer()

    def reset(self, seed):
        self.delegate.reset(seed)

    def update(self, obs):
        dock, modeled_turbidity, current = self.commands.snapshot()
        if dock:
            self.delegate.request_dock("operator")
        # The control changes the rendered/observed camera only. Range is the
        # onboard V6 navigation estimate, never scoring truth.
        visible = obs
        if obs.image is not None and modeled_turbidity > 0.0:
            distance = max(0.05, float(abs(self.delegate.nav_xy[0])))
            medium = ParticulateWaterMedium(MINERAL_SILT, modeled_turbidity)
            visible = replace(obs, image=degrade_from_medium(obs.image, medium, distance))
        estimate = self.delegate.update(visible)
        beam_engaged = self.commands.capture_beam_engaged()
        capture_confirmed = self.capture_confirmation.observe(beam_engaged)
        if capture_confirmed:
            sigma = float(np.sqrt(np.max(np.diag(
                self.delegate.nav_covariance))))
            dock_pose = tuple(np.asarray(estimate.xy, float)) + (
                float(obs.depth_measured), float(obs.yaw))
            command = self.delegate.supervisor.step(ObservableDockingInput(
                nav_position_m=tuple(self.delegate.nav_xy) + (0.0,),
                nav_uncertainty_m=sigma,
                dock_relative_pose=dock_pose,
                dock_fix_valid=bool(estimate.valid),
                physically_captured=True))
            detail = {**estimate.detail,
                      "capture_beam_engaged": beam_engaged,
                      "capture_confirmation_debounced": True,
                      "autonomy_state": command.state.value,
                      "autonomy_behavior": command.behavior,
                      "autonomy_reason": command.reason}
            estimate = ProviderEstimate(
                np.asarray(estimate.xy).copy(), estimate.valid, detail,
                velocity_xy=estimate.velocity_xy,
                mode="DONE" if command.state.value == "DOCKED" else estimate.mode)
        acoustic = getattr(self.delegate.terminal, "_last_acoustic_image", None)
        self.recorder.record_observation(
            visible, estimate, acoustic,
            {"modeled_turbidity": modeled_turbidity,
             "simulated_nephelometric_index": modeled_turbidity,
             "current_mps": list(current),
             "capture_beam_engaged": beam_engaged,
             "capture_confirmation_debounced": capture_confirmed})
        if self.telemetry_callback is not None:
            self.telemetry_callback(estimate)
        return estimate
