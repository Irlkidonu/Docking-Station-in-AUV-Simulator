#!/usr/bin/env python3
"""Feed the Gazebo window's camera panel the onboard view after the water.

The panel in the rich playground's GUI block is an ImageDisplay on
``/uuv/camera_degraded``: the vehicle's own camera as it looks through the
water column, not the clean render. In navigation-demonstrator that topic came from an optical
feedback node and an image bridge. This demonstrator runs neither, so the panel
had no publisher at all and stayed blank; pointing it straight at the docking
camera instead gave a live but wrong picture -- a 110 degree, 1920x1080 frame
with no medium applied, which looked nothing like the playground and held the
Transformer's camera rendering from launch.

This node closes that gap using the sensor path and nothing else:

* it reads the display camera composed onto the Heavy for exactly this purpose
  (``playground_world.OPERATOR_CAMERA_TOPIC``, 640x480 at 60 degrees, the
  playground's own optical format);
* it applies the same particulate-water model, at the same modeled turbidity,
  that the supervisor applies on the way into the Transformer, so zero
  turbidity is the clear rich-world view and raising the index degrades the
  panel exactly as it degrades the network's input;
* it republishes on gz transport under the topic the GUI already names.

It is display only. It never publishes a command, never touches the docking
camera, and the Transformer's frames do not pass through it.
"""
from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node as RosNode
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32

import os
REPO = Path(__file__).resolve().parents[1]
P3 = REPO
SIM = Path(os.environ.get("AUV_SIM_ROOT", "")).expanduser()
sys.path[:0] = [str(P3), str(SIM / "src/uuv_sim_physics"),
                str(SIM / "src/uuv_mode_aware_navigation")]

from docking_perception.integration.playground_world import (  # noqa: E402
    OPERATOR_CAMERA_TOPIC)
from uuv_sim_physics.sensors.particulate_water import (  # noqa: E402
    MINERAL_SILT, ParticulateWaterMedium, degrade_from_medium)

#: Where the GUI's ImageDisplay listens, as authored in the playground world.
DISPLAY_TOPIC = "/uuv/camera_degraded"
#: Dock position, to size the path length light travels through the medium.
DOCK = np.array([0.0, 0.0, -17.0])
#: Path length the medium is integrated over. The supervisor uses the range to
#: the dock, because for the Transformer the dock IS the subject. A wide-angle
#: operator view is not that: its nearest content is whatever is ahead and its
#: bulk is seabed and scenery metres further out. The range is therefore held
#: in a representative band -- it still clears as the vehicle closes on the
#: dock, but it does not collapse to a few centimetres at capture and report a
#: pristine image in water the operator has just made opaque. This is a display
#: path length, not a per-pixel range map, and it is stated rather than implied.
RANGE_LIMITS_M = (2.0, 6.0)


class OperatorCamera(RosNode):
    def __init__(self) -> None:
        super().__init__("playground_operator_camera")
        from gz.transport13 import Node as GzNode
        from gz.msgs10.image_pb2 import Image

        self.Image = Image
        self.gz = GzNode()
        self.lock = threading.Lock()
        self.turbidity = 0.2
        self.position = np.array([-10.0, -9.0, -17.0])
        self.frames = 0
        self.create_subscription(Float32, "/uuv/simulated_nephelometric_index",
                                 self._turbidity, 10)
        self.create_subscription(Vector3, "/uuv/estimated_position",
                                 self._position, 10)
        self.publisher = self.gz.advertise(DISPLAY_TOPIC, Image)
        if not self.gz.subscribe(Image, OPERATOR_CAMERA_TOPIC, self._frame):
            self.get_logger().error(
                f"could not subscribe to {OPERATOR_CAMERA_TOPIC}")
        self.get_logger().info(
            f"operator camera: {OPERATOR_CAMERA_TOPIC} -> {DISPLAY_TOPIC}")
        self.create_timer(10.0, self._report)

    def _turbidity(self, m) -> None:
        with self.lock:
            self.turbidity = max(float(m.data), 0.0)

    def _position(self, m) -> None:
        with self.lock:
            self.position = np.array([m.x, m.y, m.z], float)

    def _report(self) -> None:
        with self.lock:
            frames, c = self.frames, self.turbidity
        self.get_logger().info(
            f"{frames} frames republished, modeled index {c:.2f}")

    def _frame(self, message) -> None:
        height, width = message.height, message.width
        # Same guard the supervisor uses on the docking camera: only accept a
        # tightly packed RGB8 frame, so a format change shows up as no panel
        # rather than as a reshaped smear.
        if message.step != width * 3:
            return
        buffer = np.frombuffer(message.data, np.uint8)
        if buffer.size != height * width * 3:
            return
        frame = buffer.reshape(height, width, 3)
        with self.lock:
            c = self.turbidity
            standoff = float(np.linalg.norm((self.position - DOCK)[:2]))
        standoff = float(np.clip(standoff, *RANGE_LIMITS_M))
        degraded = degrade_from_medium(
            frame, ParticulateWaterMedium(MINERAL_SILT, c), standoff)
        out = self.Image()
        out.header.CopyFrom(message.header)
        out.width, out.height = width, height
        out.step = message.step
        out.pixel_format_type = message.pixel_format_type
        out.data = np.ascontiguousarray(
            np.clip(degraded, 0, 255).astype(np.uint8)).tobytes()
        self.publisher.publish(out)
        with self.lock:
            self.frames += 1


def main() -> int:
    rclpy.init()
    node = OperatorCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
