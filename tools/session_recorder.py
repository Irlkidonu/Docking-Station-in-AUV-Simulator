#!/usr/bin/env python3
"""Record a playground docking session to the configured output directory (DOCK_OUTPUT_ROOT).

Deliberately a SEPARATE process from the supervisor. It is the only component
that touches ground truth, and it only ever writes it to disk: keeping it out
of the supervisor's address space is what makes "operational control uses
onboard-observable data only" checkable rather than merely intended.

Every row carries both a wall stamp and the ROS clock stamp, so the telemetry
stream, the break-beam trace and the state timeline line up on one axis.

Nothing here is scientific evidence. It is the demonstrator's flight recorder:
the frozen Heavy campaign writes its own results and does not go through this.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32, String

ROOT = Path(os.environ.get("DOCK_OUTPUT_ROOT", "~/.cache/docking_station")).expanduser()
RATE_HZ = 10.0


class Recorder(Node):
    def __init__(self) -> None:
        super().__init__("playground_docking_recorder")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run = ROOT / "playground_docking" / stamp
        self.run.mkdir(parents=True, exist_ok=True)
        self.telemetry = (self.run / "telemetry.jsonl").open("a", buffering=1)
        self.camera_dir = self.run / "camera"; self.camera_dir.mkdir()
        self.fls_dir = self.run / "fls"; self.fls_dir.mkdir()
        self._sensor_lock = threading.Lock()
        self._camera = self._fls = None
        self._camera_written_t = self._fls_written_t = -1e9
        self.events: list[dict] = []
        self.state = "MISSION"
        self.started = time.time()
        self.beam_first_break: float | None = None
        self.docked_at: float | None = None

        #: Onboard-observable, and therefore also what the supervisor steers on.
        self.value = {
            "estimate": None, "attitude": None, "cov": None, "altitude": None,
            "modeled_turbidity": None, "simulated_nephelometric_index": None,
            "current_speed": None, "nav_mode": None,
            "dvl": None, "controller_command": None, "control_owner": "NORMAL",
            #: Scoring-side only. Written, never consumed by any controller.
            "truth_scoring_only": None, "position_error_scoring_only": None,
        }

        def vec(topic, key):
            self.create_subscription(
                Vector3, topic,
                lambda m, k=key: self.value.__setitem__(k, [m.x, m.y, m.z]), 10)

        def scalar(topic, key):
            self.create_subscription(
                Float32, topic,
                lambda m, k=key: self.value.__setitem__(k, float(m.data)), 10)

        vec("/uuv/estimated_position", "estimate")
        vec("/uuv/attitude_rpy", "attitude")
        vec("/uuv/dvl_velocity", "dvl")
        vec("/uuv/set_current", "commanded_environment_current")
        vec("/uuv/true_position", "truth_scoring_only")
        scalar("/uuv/position_covariance_trace", "cov")
        scalar("/uuv/position_error", "position_error_scoring_only")
        scalar("/uuv/altitude", "altitude")
        self.create_subscription(
            Float32, "/uuv/simulated_nephelometric_index",
            lambda m: self.value.update({
                "modeled_turbidity": float(m.data),
                "simulated_nephelometric_index": float(m.data)}), 10)
        scalar("/uuv/current_speed", "current_speed")
        self.create_subscription(
            String, "/uuv/nav_mode",
            lambda m: self.value.__setitem__("nav_mode", m.data), 10)
        self.create_subscription(String, "/uuv/autonomy_state", self._state, 10)
        self.create_subscription(String, "/uuv/control_owner",
                                 lambda m: self.value.__setitem__("control_owner", m.data), 10)
        self.create_subscription(Twist, "/uuv/docking_cmd", self._command, 10)
        self._multimodal_streams()
        self._break_beam()
        self.create_timer(1.0 / RATE_HZ, self._row)
        self.get_logger().info(f"recording to {self.run}")

    def _command(self, msg: Twist) -> None:
        self.value["controller_command"] = {
            "linear": [msg.linear.x, msg.linear.y, msg.linear.z],
            "angular": [msg.angular.x, msg.angular.y, msg.angular.z]}

    def _multimodal_streams(self) -> None:
        """Capture bounded-rate camera/FLS evidence off the operational bus."""
        try:
            from gz.msgs10.image_pb2 import Image
            from gz.msgs10.laserscan_pb2 import LaserScan
            from gz.transport13 import Node as GzNode
        except Exception as exc:
            self.get_logger().warning(f"multimodal recording unavailable: {exc}")
            return
        import numpy as np
        self.sensor_gz = GzNode()
        def camera(msg):
            if msg.step != msg.width * 3:
                return
            a = np.frombuffer(msg.data, np.uint8)
            if a.size == msg.height * msg.width * 3:
                with self._sensor_lock:
                    self._camera = (time.time(), a.reshape(msg.height, msg.width, 3).copy())
        def fls(msg):
            with self._sensor_lock:
                self._fls = (time.time(), np.asarray(msg.ranges, np.float32).copy(),
                             int(msg.count), int(msg.vertical_count))
        self._camera_cb = camera; self._fls_cb = fls
        self._Image = Image
        self._camera_recording = False
        self.sensor_gz.subscribe(LaserScan,
                                 "/bluerov2_heavy/fls_oculus/intersections", fls)

    def _break_beam(self) -> None:
        """The dock's own confirmation ray, straight off the Gazebo transport.

        It is a sensor reading, not a truth query: the same signal a real dock's
        break beam produces when a hull interrupts it.
        """
        self.beam_range = None
        try:
            from gz.msgs10.laserscan_pb2 import LaserScan
            from gz.transport13 import Node as GzNode
        except Exception as exc:  # noqa: BLE001 - recorder must not take the run down
            self.get_logger().warning(f"break-beam trace unavailable: {exc}")
            return
        self.gz = GzNode()

        def beam(msg):
            # NaN is "no return this scan"; +inf is "nothing within range".
            # Neither is a distance, and neither is valid JSON, so both become
            # null rather than a number a later reader would take literally.
            ranges = [r for r in msg.ranges if math.isfinite(r)]
            self.beam_range = min(ranges) if ranges else None
            if self.beam_range is not None and self.beam_range < 0.40:
                if self.beam_first_break is None:
                    self.beam_first_break = time.time()
            else:
                self.beam_first_break = None

        self.gz.subscribe(LaserScan, "/dock/capture_break_beam", beam)

    def _state(self, msg: String) -> None:
        if msg.data == self.state:
            return
        event = {"wall": time.time(), "elapsed_s": time.time() - self.started,
                 "from": self.state, "to": msg.data}
        self.state = msg.data
        if hasattr(self, "sensor_gz"):
            should_record_camera = msg.data not in {"MISSION", "DOCKED"}
            if should_record_camera and not self._camera_recording:
                self.sensor_gz.subscribe(
                    self._Image, "/bluerov2_heavy/camera", self._camera_cb)
                self._camera_recording = True
            elif not should_record_camera and self._camera_recording:
                self.sensor_gz.unsubscribe("/bluerov2_heavy/camera")
                self._camera_recording = False
        self.events.append(event)
        self.get_logger().info(f"{event['from']} -> {event['to']}")
        if msg.data == "DOCKED" and self.docked_at is None:
            self.docked_at = time.time()
        self._summary()

    def _row(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        row = {"wall": time.time(), "sim": now,
               "elapsed_s": time.time() - self.started,
               "autonomy_state": self.state,
               "break_beam_m": self.beam_range}
        row.update(self.value)
        self.telemetry.write(json.dumps(row) + "\n")
        self._write_sensor_samples()

    def _write_sensor_samples(self) -> None:
        import cv2
        import numpy as np
        with self._sensor_lock:
            camera, fls = self._camera, self._fls
        # Recording is bounded independently of render/inference rate: 5 Hz
        # images and 10 Hz FLS retain synchronized engineering evidence without
        # turning GUI performance into a disk benchmark.
        if camera is not None and camera[0] - self._camera_written_t >= 0.19:
            name = f"{camera[0]:.6f}.jpg"
            cv2.imwrite(str(self.camera_dir / name),
                        cv2.cvtColor(camera[1], cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            self._camera_written_t = camera[0]
        if fls is not None and fls[0] - self._fls_written_t >= 0.09:
            name = f"{fls[0]:.6f}.npz"
            np.savez_compressed(self.fls_dir / name, ranges=fls[1],
                                count=fls[2], vertical_count=fls[3])
            self._fls_written_t = fls[0]

    def _summary(self) -> None:
        truth = self.value["truth_scoring_only"]
        summary = {
            "run": self.run.name,
            "started_utc": datetime.fromtimestamp(
                self.started, timezone.utc).isoformat(),
            "state": self.state,
            "events": self.events,
            "docked": self.docked_at is not None,
            "time_to_dock_s": (None if self.docked_at is None
                               else self.docked_at - self.events[0]["wall"]),
            "final_estimate": self.value["estimate"],
            "final_truth_scoring_only": truth,
            "final_position_error_scoring_only":
                self.value["position_error_scoring_only"],
            "break_beam_m": self.beam_range,
        }
        (self.run / "summary.json").write_text(json.dumps(summary, indent=2))

    def destroy_node(self):
        self._summary()
        self.telemetry.close()
        files = {}
        for path in sorted(self.run.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.json":
                files[str(path.relative_to(self.run))] = {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (self.run / "MANIFEST.json").write_text(json.dumps({
            "title": "Synchronized multimodal docking recording",
            "scientific_campaign": False,
            "scoring_truth_is_separate": True,
            "files": files}, indent=2) + "\n")
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
