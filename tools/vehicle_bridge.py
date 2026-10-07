#!/usr/bin/env python3
"""ROS/Gazebo bridge for the interactive powered Heavy demonstrator.

It converts operator/autonomy velocity requests to eight physical thruster
commands.  Navigation is propagated from DVL and IMU from a declared launch
pose; Gazebo pose is published on a scoring-only diagnostic topic and never
enters the controller or operational navigation estimate.
"""
from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

import os

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, Vector3
from rclpy.node import Node as RosNode
from std_msgs.msg import Float32, String

REPO = Path(__file__).resolve().parents[1]
SIM = Path(os.environ.get("AUV_SIM_ROOT", "")).expanduser()
sys.path[:0] = [str(SIM / "src/uuv_sim_physics")]

from uuv_sim_physics import world_builder
from uuv_sim_physics.control.heavy_allocation import HeavyAllocator


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


class PoweredPlaygroundBridge(RosNode):
    #: Dead-reckoning origin. There is no absolute position fix in this
    #: demonstrator, so the DVL integration can only be right if it starts
    #: where the vehicle actually is: this must track the world spawn or the
    #: whole approach flies to a standoff that is wrong by the difference.
    SPAWN = np.array([float(v) for v in os.environ.get(
        "DOCK_HEAVY_SPAWN", "-10 -9 -17 0 0 0").split()[:3]])

    def __init__(self):
        super().__init__("powered_heavy_playground_bridge")
        self.lock = threading.Lock()
        self.owner = "NORMAL"
        self.teleop = Twist(); self.docking = Twist()
        self.velocity = np.zeros(3); self.world_velocity = np.zeros(3)
        self.yaw = 0.0; self.rate = np.zeros(3)
        self.position = self.SPAWN.copy(); self.last_dvl = None
        self.last_dvl_sim = None
        self.truth = None
        self.create_subscription(Twist, "/uuv/teleop_cmd", self._teleop, 10)
        self.create_subscription(Twist, "/uuv/docking_cmd", self._docking, 10)
        self.create_subscription(String, "/uuv/control_owner", self._owner, 10)
        # Ocean current. The same setting the panel and the recorder use is
        # forwarded to the plant, so what the operator sets is what the water
        # actually does -- it is not merely logged alongside the run.
        self.create_subscription(Vector3, "/uuv/set_current", self._current, 10)
        self.position_pub = self.create_publisher(Vector3, "/uuv/estimated_position", 10)
        self.attitude_pub = self.create_publisher(Vector3, "/uuv/attitude_rpy", 10)
        self.dvl_pub = self.create_publisher(Vector3, "/uuv/dvl_velocity", 10)
        self.depth_pub = self.create_publisher(Float32, "/uuv/depth", 10)
        self.cov_pub = self.create_publisher(Float32, "/uuv/position_covariance_trace", 10)
        # Modeled turbidity. The panel's slider publishes /uuv/set_modeled_turbidity
        # and everything downstream -- supervisor, recorder, stalker, operator
        # camera -- reads /uuv/simulated_nephelometric_index, which in the navigation-demonstrator
        # playground was republished by that project's vehicle node. This launch
        # does not run it, so nothing carried the value across and the slider
        # moved nothing: the supervisor sat on its 0.2 default for the whole
        # session. The bridge already owns vehicle-side state, so it forwards it.
        # The name stays a declared simulation index, never a claimed NTU/FNU.
        self.turbidity_pub = self.create_publisher(
            Float32, "/uuv/simulated_nephelometric_index", 10)
        self.modeled_turbidity = 0.2
        self.create_subscription(Float32, "/uuv/set_modeled_turbidity",
                                 self._modeled_turbidity, 10)
        self.create_timer(0.5, lambda: self.turbidity_pub.publish(
            Float32(data=float(self.modeled_turbidity))))
        self.truth_pub = self.create_publisher(Vector3, "/uuv/scoring_truth_position", 10)
        # Infrastructure-only observability.  This status contains no hidden
        # state used by autonomy; it lets launchers prove that Gazebo's
        # powered command path and sensor clock are alive before accepting an
        # attempt as a trial.
        self.plant_status_pub = self.create_publisher(
            String, "/uuv/powered_plant_status", 10)
        self.imu_messages = 0; self.dvl_messages = 0; self.pose_messages = 0
        self.last_command_norm = 0.0; self.last_thrust_norm = 0.0
        self._gz()
        self.last = time.monotonic()
        self.create_timer(0.02, self._step)

    def _gz(self):
        from gz.msgs10.double_pb2 import Double
        from gz.msgs10.dvl_velocity_tracking_pb2 import DVLVelocityTracking
        from gz.msgs10.altimeter_pb2 import Altimeter
        from gz.msgs10.imu_pb2 import IMU
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node
        from gz.msgs10.vector3d_pb2 import Vector3d
        self.Vector3d = Vector3d
        self.Double = Double; self.gz = Node()
        # gz-sim's Hydrodynamics system reads the flow field from this topic and
        # evaluates its drag terms on the WATER-RELATIVE velocity, so a current
        # is a real force on the hull rather than a number added to a reported
        # state. A vehicle with no propulsion is carried downstream by it,
        # because the relative velocity -- and therefore the drag -- is nonzero
        # exactly when the vehicle is not already moving with the water.
        self.current_pub = self.gz.advertise("/ocean_current", Vector3d)
        self.current = np.zeros(3)
        cfg = world_builder.load_config(True, "bluerov2_heavy")["vehicle_bluerov2_phys"]
        self.allocator = HeavyAllocator(cfg)
        self.joints = [u["joint"] for u in cfg["thrusters"]["units"]]
        self.thrust = {j: self.gz.advertise(
            f"/model/bluerov2_heavy/joint/{j}/cmd_thrust", Double) for j in self.joints}

        def imu(m):
            with self.lock:
                self.imu_messages += 1
                self.yaw = _yaw(m.orientation)
                self.rate = np.array([m.angular_velocity.x,
                                      m.angular_velocity.y,
                                      m.angular_velocity.z])
        def dvl(m):
            locked = sum(1 for b in m.beams if b.locked)
            if m.target.type == 1 and locked >= 3:
                stamp = float(m.header.stamp.sec) + 1e-9 * float(m.header.stamp.nsec)
                measured = np.array([m.velocity.mean.x,
                                     m.velocity.mean.y,
                                     m.velocity.mean.z])
                with self.lock:
                    self.dvl_messages += 1
                    # DVL velocity is body-frame metres per SIMULATION second.
                    # Integrating it on the ROS wall timer scales navigation by
                    # 1/RTF whenever rendering slows Gazebo. Integrate each
                    # acquisition exactly once using its sensor timestamp.
                    c, s = math.cos(self.yaw), math.sin(self.yaw)
                    world = np.array([c * measured[0] - s * measured[1],
                                      s * measured[0] + c * measured[1],
                                      measured[2]])
                    if self.last_dvl_sim is not None:
                        dt_sim = stamp - self.last_dvl_sim
                        if 0.0 < dt_sim <= 0.25:
                            self.position += world * dt_sim
                    self.velocity = measured
                    self.world_velocity = world
                    self.last_dvl_sim = stamp
                    self.last_dvl = time.monotonic()
        def depth(m):
            # Altimeter vertical_position is relative to vertical_reference;
            # both are operational sensor fields, not scoring pose.
            with self.lock:
                self.position[2] = float(m.vertical_reference + m.vertical_position)
        def pose(m):
            for p in m.pose:
                if p.name == "bluerov2_heavy":
                    self.pose_messages += 1
                    self.truth = np.array([p.position.x, p.position.y, p.position.z])
        self.gz.subscribe(IMU, "/bluerov2_heavy/imu", imu)
        self.gz.subscribe(DVLVelocityTracking, "/bluerov2_heavy/dvl", dvl)
        self.gz.subscribe(Altimeter, "/bluerov2_heavy/depth", depth)
        self.gz.subscribe(Pose_V, "/world/mode_aware_survey/dynamic_pose/info", pose)

    def _modeled_turbidity(self, m):
        self.modeled_turbidity = float(m.data)
        self.turbidity_pub.publish(Float32(data=self.modeled_turbidity))
        self.get_logger().info(
            f"modeled turbidity index {self.modeled_turbidity:.2f}")

    def _teleop(self, m): self.teleop = m
    def _docking(self, m): self.docking = m
    def _owner(self, m): self.owner = m.data.upper()

    def _current(self, m):
        """Forward the commanded flow to the plant's hydrodynamics."""
        with self.lock:
            self.current = np.array([m.x, m.y, m.z], float)
        if self.current_pub is not None:
            self.current_pub.publish(
                self.Vector3d(x=float(m.x), y=float(m.y), z=float(m.z)))
        self.get_logger().info(
            f"ocean current set to ({m.x:.3f}, {m.y:.3f}, {m.z:.3f}) m/s")

    def _zero(self):
        for pub in self.thrust.values(): pub.publish(self.Double(data=0.0))

    def _step(self):
        now = time.monotonic(); dt = min(max(now - self.last, 0.001), 0.1); self.last = now
        with self.lock:
            vel = self.velocity.copy(); world_v = self.world_velocity.copy()
            yaw = self.yaw; rate = self.rate.copy(); position = self.position.copy()
        cmd = self.docking if self.owner == "DOCKING" else self.teleop
        if self.owner in {"DOCKED", "CRASHED"}:
            self._zero(); desired = np.zeros(3); yaw_rate = 0.0
        else:
            desired = np.array([cmd.linear.x, cmd.linear.y, cmd.linear.z], float)
            yaw_rate = float(cmd.angular.z)
            force = np.clip(65.0 * (desired - vel), -55.0, 55.0)
            moment = np.array([-3.0 * rate[0], -3.0 * rate[1],
                               8.0 * (yaw_rate - rate[2])])
            report = self.allocator(np.r_[force, moment])
            self.last_command_norm = float(np.linalg.norm(
                np.r_[desired, yaw_rate]))
            self.last_thrust_norm = float(np.linalg.norm(report.thrusts))
            for value, joint in zip(report.thrusts, self.joints):
                self.thrust[joint].publish(self.Double(data=float(value)))
        self.position_pub.publish(Vector3(x=float(position[0]), y=float(position[1]), z=float(position[2])))
        self.attitude_pub.publish(Vector3(x=0.0, y=0.0, z=float(yaw)))
        self.dvl_pub.publish(Vector3(x=float(world_v[0]), y=float(world_v[1]), z=float(world_v[2])))
        self.depth_pub.publish(Float32(data=float(-position[2])))
        age = 10.0 if self.last_dvl is None else now - self.last_dvl
        self.cov_pub.publish(Float32(data=float(0.01 + 0.02 * age)))
        if self.truth is not None:
            self.truth_pub.publish(Vector3(x=float(self.truth[0]), y=float(self.truth[1]), z=float(self.truth[2])))
        status = {
            "owner": self.owner,
            "command_norm": self.last_command_norm,
            "thrust_norm": self.last_thrust_norm,
            "thruster_publishers_valid": all(p.valid() for p in self.thrust.values()),
            "thruster_publishers_connected": sum(
                bool(p.has_connections()) for p in self.thrust.values()),
            "thruster_count": len(self.thrust),
            "imu_messages": self.imu_messages,
            "dvl_messages": self.dvl_messages,
            "pose_messages": self.pose_messages,
        }
        import json
        self.plant_status_pub.publish(String(data=json.dumps(status, sort_keys=True)))

    def destroy_node(self):
        self._zero()
        return super().destroy_node()


def main():
    rclpy.init(); node = PoweredPlaygroundBridge()
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__": main()
