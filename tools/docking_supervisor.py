#!/usr/bin/env python3
"""Opt-in V6 docking supervisor for the free-running AUV playground."""
from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
P3 = REPO
SIM = Path(os.environ.get("AUV_SIM_ROOT", "")).expanduser()
sys.path[:0] = [str(P3), str(SIM / "src/uuv_sim_physics"),
                str(SIM / "src/uuv_mode_aware_navigation")]

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, Vector3
from rclpy.node import Node as RosNode
from std_msgs.msg import Bool, Float32, String

from docking_perception.integration.heavy_v6_terminal_provider import HeavyV6TerminalProvider
import docking_perception.integration.heavy_v5_terminal_provider as _v5mod

#: Diagnostics only. Records the network's decoded measurement BEFORE the
#: filter sees it, so a trace can separate raw-network error from filtered
#: error. It returns the original values untouched and changes no behaviour.
_RAW = {}
_orig_decode = _v5mod.decode_with_imu_yaw
def _decode_recording(se2, sigma_se2, yaw_absolute, *args, **kwargs):
    z, sg, psi = _orig_decode(se2, sigma_se2, yaw_absolute, *args, **kwargs)
    try:
        _RAW["raw_se2"] = [float(v) for v in np.asarray(se2, float)]
        _RAW["decoded_range_m"] = float(z[0]); _RAW["decoded_lateral_m"] = float(z[1])
        _RAW["decoded_sigma"] = [float(v) for v in np.asarray(sg, float)]
    except Exception:
        pass
    return z, sg, psi
_v5mod.decode_with_imu_yaw = _decode_recording
from docking_perception.integration.next_step_runtime import CaptureConfirmationDebouncer
from uuv_sim_physics.docking.state_providers import ProviderEstimate, SensorObservation
from uuv_sim_physics.docking.terminal_estimator import TerminalEstimator
from uuv_sim_physics.sensors.oculus_m750d import (
    OculusM750dRenderer, extract_observable_features)
from uuv_sim_physics.process_group import terminate_owned_process

#: The rich-world Transformer the demonstrator flies by default. Promoted
#: 2026-09-12; provenance, acceptance results and the one recorded static
#: failure are in its SELECTION_RECORD_V2.json alongside it.
#: Dock states in which touching the funnel is permitted guidance rather than a
#: collision. A funnel exists to receive a slightly off-axis vehicle and walk it
#: to the throat, and this one is a real 0.60 m mouth narrowing to 0.28 m with
#: real collision geometry.
GUIDED_STATES = ("DOCK_APPROACH", "DOCK_TERMINAL", "CAPTURE_PENDING")


def is_forbidden_contact(contact_link, state):
    """True when this contact is a collision rather than permitted guidance.

    The collar rim and the support post are structural: touching either is
    never a valid capture, in any phase, and they are what a genuine
    mis-approach hits. The funnel is a guidance surface, forbidden only before
    terminal alignment has begun.

    Hoisted out of the supervisor tick so the rule can be exercised directly.
    It previously existed only as an inline condition, and no test covered the
    CRASHED branch at all -- while integrated runs cannot reach it, because the
    approach either docks or aborts to RETURN_TO_DOCK long before striking
    structure, even at five times the tested current.
    """
    if contact_link in ("support_collision", "collar_collision"):
        return True
    return contact_link == "funnel_collision" and state not in GUIDED_STATES


PROMOTED_CHECKPOINT = (Path(__file__).resolve().parents[1]
                       / "models" / "rich_domain_transformer_v2.pt")


DOCK = np.array([0.0, 0.0, -17.0])
#: The initial condition the terminal estimator was validated from. The frozen
#: Heavy scenario places the vehicle 4 m in front of the dock, on the approach
#: axis, facing it (`world_builder`: bluerov2_heavy pose "4 0 -15 0 0 3.14159"
#: against a dock at the origin). The return leg's job is to REPRODUCE that
#: condition here -- not to hand over from wherever the survey happened to end.
# The frozen Transformer was qualified for 0.65--1.25 m.  The old 4 m Heavy
# start belonged to return/navigation qualification, not the learned terminal
# operating envelope; asking the network for a pose there produced the observed
# 2.3 m alias.  Return navigation therefore brings the vehicle to the declared
# 1.2 m acquisition boundary before the camera/Transformer is engaged.
STANDOFF = DOCK + np.array([1.20, 0.0, 0.0])
#: REMEDY 1 -- qualified-band seeding.
#:
#: V6's terminal filter seeds directly from its first accepted visual fix
#: (`heavy_v4_terminal_provider`: `if not self.have: self.x = z.copy()`), and
#: whatever error that one fix carries becomes a constant offset for the whole
#: run: correlation between seed error and run bias is +0.996 over the four
#: runs with ground truth, with a within-run spread of 0.6-3.0 mm across a
#: 1.1 m range sweep.
#:
#: The handover puts the vehicle at an onboard range of about 1.29 m, so that
#: seeding fix is taken ABOVE the 0.65-1.25 m band the frozen network was
#: qualified over -- where its lateral disagreement with navigation is a median
#: 65 mm against 15 mm inside the band, pooled over 38 runs.
#:
#: So no visual measurement is offered to the filter until the vehicle is
#: inside that band. Nothing inside V6 changes; it is simply not asked for a
#: fix from a place it was never qualified to give one. The range used for the
#: test is the onboard navigation range, not V6's own, because V6's range
#: estimate is itself biased high by +85 to +277 mm -- it would believe itself
#: in band while the true range was 0.37 m.
V6_SEED_BAND_M = (0.65, 1.25)
#: Where inside that band the seed is actually taken.
#:
#: Measured directly, uncontaminated by the filter: the RAW error of the
#: seeding fix against ground truth, by the true range it was taken at --
#:
#:     1.286-1.298 m   -62, -84, -94, -113 mm
#:     1.237-1.240 m   -26.0, -25.7 mm
#:     0.816 m         -158 mm
#:
#: The error is not monotonic in range. It has a minimum near 1.24 m and gets
#: worse in BOTH directions, so seeding deeper inside the band is not better --
#: a re-seed forced at 0.89 m produced a -130 mm run bias and 0/22
#: corroboration, worse than doing nothing. An earlier reading of the pooled
#: per-range statistics suggested the opposite; those were confounded, because
#: they measured a filtered estimate that already carried an earlier seed bias
#: rather than raw network accuracy.
#:
#: The window is therefore the whole qualified band. In practice the vehicle
#: crosses 1.25 m a second after handover, so the seed lands at ~1.24 m, at the
#: measured minimum.
V6_SEED_WINDOW_M = (0.65, 1.25)
#: How far V6's lateral estimate may differ from the onboard cross-track before
#: it is treated as unusable for steering. The two measure the same quantity in
#: the same frame; a disagreement larger than the offsets being corrected means
#: one of them is wrong, and it is not the navigation solution -- logged against
#: ground truth it tracked to a mean 5.4 mm, while V6 carried a constant per-run
#: bias of -58 to -119 mm inherited from its seeding fix.
V6_NAV_AGREEMENT_M = 0.025
CORRIDOR_ENTRY = DOCK + np.array([3.0, 0.0, 0.0])
APPROACH_YAW = math.pi
#: Where the collar break beam sits and which way it points, in the dock frame.
#: Diagnostics only -- the beam itself is configured in
#: docking_perception.integration.playground_world, and these must agree with it or
#: (the y origin now sits on the widened collar's inner surface, radius 0.265)
#: the log describes a beam that is not the one being read. The trace used to
#: carry the beam's previous plane (x=-0.01) long after it had moved to +0.10.
#: test_playground_docking_diagnostics.py pins the two together.
BEAM_ORIGIN_DOCK = (0.05, -0.265, 0.0)
#: The world the demonstrator runs in, needed to address the dock's contact
#: sensors. Matches the <world name> the launch composes.
WORLD = os.environ.get("DOCK_PLAYGROUND_WORLD", "mode_aware_survey")
#: The model whose presence in a contact pair makes that contact a docking
#: collision rather than dock structure resting against itself.
VEHICLE_MODEL = "bluerov2_heavy"


def select_terminal_cross_track(mode: str, filtered_m: float,
                                navigation_m: float, quarantined: bool):
    """Select the lateral signal for the registered mechanism comparison.

    ``raw`` and ``filter`` deliberately receive no navigation fallback.  The
    operational ``full`` path retains the existing 25 mm corroboration rule.
    """
    corroborated = (abs(float(filtered_m) - float(navigation_m))
                    <= V6_NAV_AGREEMENT_M and not bool(quarantined))
    if mode == "raw":
        return float(filtered_m), "raw_transformer", corroborated
    if mode == "filter":
        return float(filtered_m), "estimator_filter", corroborated
    if mode == "handover":
        return float(filtered_m), "geometric_terminal", corroborated
    if mode == "full":
        return ((float(filtered_m), "estimator_filter", corroborated)
                if corroborated else
                (float(navigation_m), "navigation_fallback", corroborated))
    raise ValueError(f"unknown terminal comparison mode: {mode}")


def _is_vehicle(collision_name: str) -> bool:
    """Is this scoped collision name the vehicle's?

    Gazebo scopes contact names as `model::link::collision`, so the model is
    compared exactly rather than by substring: a dock part that merely had the
    vehicle's name inside its own must not read as the vehicle.
    """
    return collision_name.split("::")[0] == VEHICLE_MODEL


def contact_topic(link: str, sensor: str) -> str:
    """Where a dock contact sensor ACTUALLY publishes.

    gz-sim's Contact system ignores the <topic> element on a contact sensor and
    publishes on the scoped default name instead. The dock's sensors declare
    `/dock/contact/<link>`; nothing ever appears there, and a subscription to
    it is silently dead -- `gz topic -i` shows the supervisor as the only party
    on the topic, with no publisher, while the sensor publishes here.

    That is why a forbidden dock contact never reached the supervisor and the
    CRASHED branch was unreachable: the collision was detected by Gazebo and
    published every tick, addressed to a name no one was listening to.
    """
    return (f"/world/{WORLD}/model/docking_station/link/{link}"
            f"/sensor/{sensor}/contact")
BEAM_DIRECTION_DOCK = (0.0, 1.0, 0.0)


@dataclass
class BoundedLateralPI:
    """Terminal sway regulator with bounded integral disturbance rejection.

    ``kp`` preserves the executed proportional law.  ``ki = kp / 4 s`` gives
    the integral action a four-second time constant: slow relative to the
    proportional correction, but fast enough to reject a persistent current
    during the terminal approach.  Both the total command and the integral
    contribution are bounded.  Conditional integration prevents windup when
    the total command is saturated, while still allowing the state to unwind.
    """

    kp: float = 0.8
    ki: float = 0.2
    output_limit_mps: float = 0.18
    integral_output_limit_mps: float = 0.10
    integral_error_s: float = 0.0

    def reset(self) -> None:
        self.integral_error_s = 0.0

    @property
    def integral_output_mps(self) -> float:
        return self.ki * self.integral_error_s

    def update(self, error_m: float, dt_s: float,
               integral_error_m: float | None = None) -> float:
        error = float(error_m)
        # The fast proportional term follows V6's dock-relative estimate.  The
        # slow integral term may use the vehicle navigation estimator's mapped
        # cross-track error: unlike a near-field visual residual, it retains
        # the accumulated displacement caused by a steady current.  Both are
        # onboard observations; neither is evaluator truth.
        integral_error = (error if integral_error_m is None
                          else float(integral_error_m))
        dt = max(float(dt_s), 0.0)
        integral_limit = self.integral_output_limit_mps / self.ki
        candidate = float(np.clip(
            self.integral_error_s + integral_error * dt,
            -integral_limit, integral_limit))
        candidate_unbounded = self.kp * error + self.ki * candidate
        at_upper = candidate_unbounded > self.output_limit_mps
        at_lower = candidate_unbounded < -self.output_limit_mps
        # Integrate unless doing so drives an already-saturated command farther
        # into saturation. Opposite-sign error always unwinds the integrator.
        if at_upper and integral_error > 0.0:
            # Land exactly on the actuator boundary when this step first
            # reaches it; subsequent same-sign error is then ignored.
            boundary = (self.output_limit_mps - self.kp * error) / self.ki
            if self.integral_error_s < boundary:
                self.integral_error_s = min(candidate, boundary)
        elif at_lower and integral_error < 0.0:
            boundary = (-self.output_limit_mps - self.kp * error) / self.ki
            if self.integral_error_s > boundary:
                self.integral_error_s = max(candidate, boundary)
        else:
            self.integral_error_s = candidate
        command = self.kp * error + self.ki * self.integral_error_s
        return float(np.clip(command, -self.output_limit_mps,
                             self.output_limit_mps))


class PlaygroundDockingSupervisor(RosNode):
    """Leaves the playground untouched until the shared DOCK request edge."""
    def __init__(self):
        super().__init__("playground_v6_docking_supervisor")
        # Development-only mechanism comparison.  The default is the
        # operational demonstrator path and is deliberately unchanged.
        self.comparison_mode = os.environ.get(
            "DOCK_TERMINAL_COMPARISON_MODE", "full").strip().lower()
        if self.comparison_mode not in {"raw", "handover", "filter", "full"}:
            raise ValueError(
                "DOCK_TERMINAL_COMPARISON_MODE must be raw, handover, filter, or full")
        self.raw_comparison_xy = None
        self.raw_comparison_at = None
        # Scenario-B replication state.  This is the already-defined 0.60 m
        # geometric/kinematic terminal estimator, adapted only at the sensor
        # interface from the rich world's 512x9 Oculus fan.
        self.scenario_b_terminal = TerminalEstimator()
        self.scenario_b_renderer = OculusM750dRenderer(seed=79091)
        self.scenario_b_handover = None
        self.scenario_b_last_fls = None
        self.lock = threading.Lock(); self.requested = False; self.state = "MISSION"
        self.position = self.attitude = None; self.covariance = float("inf")
        self.image = self.fls = self.dvl = None; self.last_image_stamp = None
        self.beam = False; self.debounce = CaptureConfirmationDebouncer()
        self.contact_link = None; self.contact_at = None
        #: Which link caused the crash. Kept after the contact itself ends, so
        #: the record of WHY the run stopped outlives the transient touch.
        self.crashed_on = None
        self.latch_confirmed = False
        self.capture_pending_since = None
        #: Change-detection state for the settled-terminal-state heartbeat.
        self.last_terminal_key = None; self.last_terminal_trace = None
        self.modeled_turbidity = 0.2
        self.last_camera_hash = self.last_fls_hash = None
        #: Latched at handover. Terminal control drives TOWARDS the dock, so a
        #: distance test would immediately read as "left the handover point"
        #: and give control back to the return leg, which drags the vehicle
        #: out again: the vehicle sits beside the station yawing left and
        #: right and never converges. Only losing the dock altogether reopens
        #: the return leg.
        self.terminal = False
        self.corridor_entered = False
        #: Where to sit if the terminal estimator loses the dock mid-approach.
        self.hold_point = None
        self.origin_checked = False; self.origin_retries = 0
        #: Remedy-1 instrumentation, diagnostics only.
        self.seed_deferred = False; self.seed_defer_count = 0
        self.seed_range_m = None
        self.nav_range_m = None; self.steering_source = None
        self.v6_nav_disagreement_m = None
        self.lateral_controller = BoundedLateralPI()
        self.camera_started_at = None
        self.provider = self.server = None
        root=Path(os.environ.get("DOCK_RUNTIME_ROOT",
                                 os.path.expanduser("~/.cache/docking_station/runtime")))
        try:
            root.mkdir(parents=True,exist_ok=True)
            self.trace=(root/"supervisor_trace.jsonl").open("a",buffering=1)
        except OSError as exc:
            self.trace=None;self.get_logger().warning(f"no trace: {exc}")
        self.cmd = self.create_publisher(Twist, "/uuv/docking_cmd", 10)
        self.owner = self.create_publisher(String, "/uuv/control_owner", 10)
        self.control = self.create_publisher(String, "/uuv/control_mode", 10)
        self.status = self.create_publisher(String, "/uuv/autonomy_state", 10)
        self.create_subscription(String, "/uuv/autonomy_request", self._request, 10)
        self.create_subscription(Bool, "/uuv/mission_complete", self._mission_complete, 10)
        self.create_subscription(Float32, "/uuv/soc_fraction", self._soc, 10)
        self.create_subscription(Vector3, "/uuv/estimated_position", self._position, 10)
        self.create_subscription(Vector3, "/uuv/attitude_rpy", self._attitude, 10)
        # The vehicle's own DVL, not Gazebo's. Gazebo's instrument reports
        # 0.000 m/s with four beams locked on this vehicle, because the model
        # is teleported by SetEntityPose rather than moved by DART and a set
        # pose carries no velocity. Fed to the terminal filter, that confident
        # zero is what made the approach crab: the filter dead-reckoned as if
        # the vehicle were stationary, the visual updates disagreed in one
        # consistent direction, V6 quarantined the lateral channel, and the
        # sway command then chased an estimate that had stopped tracking.
        #
        # /uuv/dvl_velocity is the playground's own bottom-track measurement:
        # sensor model, noise and dropouts included, no truth. World frame, so
        # it is rotated into the body frame the provider documents.
        self.create_subscription(Vector3, "/uuv/dvl_velocity", self._dvl, 10)
        self.create_subscription(Float32, "/uuv/position_covariance_trace",
                                 lambda m: setattr(self, "covariance", float(m.data)), 10)
        self.create_subscription(
            Float32, "/uuv/simulated_nephelometric_index",
            lambda m: setattr(self, "modeled_turbidity", float(m.data)), 10)
        self._gz_subscriptions()
        self.last_tick = time.monotonic(); self.create_timer(0.05, self._step)
        self.create_timer(0.10, self._publish_owner)

    #: THE CLOCK. Every stamp in this file, and dt, is the wall clock, because
    #: that is the clock the playground vehicle moves in.
    #:
    #: The navigation-demonstrator vehicle integrates its own motion on a plain ROS timer with
    #: no use_sim_time: it is a real-time kinematic animation, and a commanded
    #: 0.25 m/s is 0.25 m per second of WALL time. Gazebo stamps its sensors in
    #: SIMULATION time, and the two only agree while RTF is exactly 1.
    #:
    #: Measured on this machine with the docking camera streaming: RTF 0.37,
    #: commanded surge 0.250 m/s, true travel 0.244 m/s wall = 0.652 m/s sim,
    #: filter dead reckoning 0.200 m/s. Propagating on sim-time dt therefore
    #: under-counted the distance travelled by 1/RTF, so the vehicle flew
    #: through the collar before the estimate ever reached capture range, and
    #: the lateral correction was computed against an estimate running about
    #: three times slow -- which is exactly the drift to starboard and the
    #: fly-past that were observed.
    #:
    #: So sensor arrival time is used in place of the Gazebo header stamp.
    #: Ordering is preserved (the transport delivers in order), freshness is
    #: preserved, and every age, every dt and every velocity is now in one
    #: clock. Nothing in the frozen V6 logic changes: it is handed a
    #: consistent observation stream, which is what it always assumed.
    @staticmethod
    def _stamp(_header): return time.monotonic()

    #: Gazebo renders a camera only while something is subscribed to it. The
    #: docking camera is the validated 1920x1080 30 Hz sensor the frozen
    #: perception model was trained and evaluated on, and rendering it costs
    #: about half of real time on this machine: measured RTF 0.999 with no
    #: subscriber against 0.456 with one, in the same world. The sonar, DVL
    #: and break beam together cost nothing measurable (1.00, 1.02).
    #:
    #: So the camera is subscribed only for the terminal approach, which is
    #: the only phase that consumes it. Nothing about the sensor changes --
    #: same resolution, same rate, same optics -- so the perception the model
    #: sees is the validated one; the simulator simply does not render frames
    #: no one is reading. That is what keeps the pre-DOCK playground running
    #: at the same speed as the untouched navigation-demonstrator world.
    CAMERA_TOPIC = "/bluerov2_heavy/camera"

    def _gz_subscriptions(self):
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        from gz.msgs10.laserscan_pb2 import LaserScan
        from gz.msgs10.contacts_pb2 import Contacts
        from gz.msgs10.stringmsg_pb2 import StringMsg
        from gz.msgs10.empty_pb2 import Empty
        self.gz = Node(); self.Image = Image
        def image(m):
            if m.step == m.width * 3:
                a=np.frombuffer(m.data,np.uint8)
                if a.size == m.height*m.width*3:
                    with self.lock:
                        self.image=(self._stamp(m.header),a.reshape(m.height,m.width,3).copy())
                        self.image_count+=1
        def fls(m):
            with self.lock:self.fls=(self._stamp(m.header),np.asarray(m.ranges,float),float(m.angle_min),float(m.angle_max))
        def beam(m):
            with self.lock:self.beam=bool(m.ranges and min(m.ranges)<0.40)
        def contact(region):
            def receive(m):
                # A contact on a dock link is only a docking collision if the
                # VEHICLE is one of the two parties. The dock's own support
                # post rests permanently against the grabber's mount collar
                # (docking_station::support_collision::col_post against
                # dock_grabber::base_link::physical_mount_collar), so an
                # unfiltered handler reported a forbidden collision from world
                # load onwards and crashed every approach the moment the
                # terminal phase began -- with the vehicle still 1.3 m away at
                # the standoff, touching nothing.
                if any(_is_vehicle(c.collision1.name) or _is_vehicle(c.collision2.name)
                       for c in m.contact):
                    with self.lock:
                        self.contact_link=region; self.contact_at=time.monotonic()
            return receive
        def latch(m):
            # The DetachableJoint publishes its state as a StringMsg carrying
            # "attached" or "detached" -- not a Boolean. Subscribing with the
            # wrong type made gz-transport drop every message silently, so the
            # latch could never be confirmed and CAPTURE_PENDING never became
            # DOCKED. bool(m.data) is wrong twice over: "detached" is a
            # non-empty string and would have read as latched.
            with self.lock:self.latch_confirmed=(m.data=="attached")
        self._image_cb = image
        self.gz.subscribe(LaserScan,"/bluerov2_heavy/fls_oculus/intersections",fls)
        self.gz.subscribe(LaserScan,"/dock/capture_break_beam",beam)
        for region in ("support_collision", "funnel_collision", "collar_collision"):
            self.gz.subscribe(Contacts, contact_topic(region, f"{region}_contact"),
                              contact(region))
        self.gz.subscribe(StringMsg, "/dock/latch/state", latch)
        self.Empty = Empty
        self.latch_attach = self.gz.advertise("/dock/latch/attach", Empty)
        self.latch_detach = self.gz.advertise("/dock/latch/detach", Empty)
        self.camera_live = False; self.image_count = 0
        try:
            from gz.msgs10.double_pb2 import Double
            self.Double = Double
            self.arm_port = self.gz.advertise("/dock_grabber/port", Double)
            self.arm_stbd = self.gz.advertise("/dock_grabber/stbd", Double)
        except Exception as exc:  # noqa: BLE001 - arms are cosmetic, not control
            self.arm_port = self.arm_stbd = None
            self.get_logger().warning(f"capture arms unavailable: {exc}")
        # Open them once the world is listening, and leave them open.
        self.create_timer(2.0, self._present_arms)
        self.create_timer(1.0, self._ensure_unlatched)

    def _ensure_unlatched(self):
        if not self.requested and self.latch_detach is not None:
            self.latch_detach.publish(self.Empty())

    #: Grabber shoulder angles. The arms rest OPEN, which is what a dock
    #: waiting for a vehicle looks like, and close on confirmed capture. The
    #: joint limits are -0.2..1.2 (port) and -1.2..0.2 (starboard), so these
    #: are the open extremes; zero is the closed position the model is
    #: authored in.
    #: Open is now joint coordinate zero, because the arms are composed into
    #: the world already open (playground_world._remount_capture_arms). The
    #: previous pair drove both shoulders to their outward limits, 2.0 rad from
    #: the clamp: not a choice, but the only place the old hinge geometry was
    #: clear of the vehicle, and the reason the arms looked flung open. With
    #: the hinges moved in front of the collar the parked pose already clears
    #: the Heavy's approach corridor by 129 mm.
    ARMS_OPEN = (0.0, 0.0)
    #: Closed is a real -1.30 rad rotation back to the authored clamp pose,
    #: plus 0.06 rad of overtravel. At the clamp pose the jaw faces are 2 mm
    #: inside the hull's 0.169 m half width; the overtravel is what the servo
    #: converts into clamping force once the pads are on the plating, and the
    #: joint limits sit a further 0.09 rad out so contact, not the end stop, is
    #: what arrests the arm. Retention remains the detachable joint; the arms
    #: are the mechanical grip that must be seen and felt to close on the hull.
    ARMS_CLOSED = (-1.36, 1.36)
    #: How long capture waits before asking the latch to engage.
    #:
    #: This is the window in which the jaws do their real work. They travel
    #: 1.30 rad at a limited 1.5 rad/s -- about 0.9 s of SIMULATION time -- and
    #: this clock, like every other in this file, is the WALL clock, so the
    #: budget has to cover that travel at the worst real-time factor the
    #: session runs at. The jaws were slowed to 0.5 rad/s so they push the
    #: vehicle into line instead of smacking it there, which makes the travel
    #: about 2.6 s of sim time; 9 s of wall clock covers that down to RTF 0.3.
    #:
    #: Why it matters that the arms finish first. Once the detachable joint
    #: engages, the vehicle becomes part of the dock's kinematic tree and jaw
    #: against hull stops being an enforced contact: the position servo then
    #: drives the jaw straight into the hull it can no longer feel. Measured
    #: with the old 0.10 s, and still with 1.2 s at RTF 0.53 -- 0.64 s of sim
    #: time, less than the travel -- the latch closed first and the port jaw
    #: finished 41 mm inside the vehicle while the starboard one hung 13 mm
    #: clear of it.
    #:
    #: Given the time, the mechanism does the right thing on its own. The jaws
    #: reach a vehicle that is still FREE, the first one to touch pushes it --
    #: 24 N.m over a 0.226 m arm is about 106 N against 55 N of station keeping
    #: -- and the vehicle settles between them. That motion shows up as speed,
    #: which the capture gate already refuses to latch on, so the latch waits
    #: by itself until the vehicle has stopped being pushed. Nothing else about
    #: capture moves: the beam, the speed gate and the debounce are unchanged.
    ARM_CLOSE_TRAVEL_S = 9.0

    def _arms(self, angles) -> None:
        """Command the dock's capture arms.

        Dock-side actuation, not a vehicle command: the arms are part of the
        station and moving them does not touch the playground's own control
        path, so opening them at startup leaves the pre-DOCK session exactly
        as it was.
        """
        if self.arm_port is None: return
        port, stbd = angles
        self.arm_port.publish(self.Double(data=float(port)))
        self.arm_stbd.publish(self.Double(data=float(stbd)))

    def _camera(self, live: bool) -> None:
        if live == self.camera_live: return
        if live:
            with self.lock: self.image_count = 0
            self.gz.subscribe(self.Image, self.CAMERA_TOPIC, self._image_cb)
        else:
            self.gz.unsubscribe(self.CAMERA_TOPIC)
            with self.lock: self.image = None
            self.last_image_stamp = None
        self.camera_live = live
        self.camera_started_at = time.monotonic() if live else None
        self.get_logger().info(
            f"docking camera {'streaming' if live else 'idle'}")

    def _present_arms(self):
        """Hold the arms open until capture. Repeated because the controller
        only has a setpoint once something has published one, and the world
        may not have been up when the first command went out."""
        if self.state not in ("CAPTURE_PENDING", "DOCKED"):
            self._arms(self.ARMS_OPEN)

    def _position(self,m): self.position=np.array([m.x,m.y,m.z],float)
    def _dvl(self,m):
        if self.attitude is None:return
        yaw=float(self.attitude[2]);c,s=math.cos(yaw),math.sin(yaw)
        body=np.array([c*m.x+s*m.y,-s*m.x+c*m.y,float(m.z)])
        # A message IS the acquisition: the vehicle publishes only on the ticks
        # its DVL actually returns a velocity, so a gap here is a real outage.
        with self.lock:self.dvl=(time.monotonic(),body,4,True)
    def _attitude(self,m): self.attitude=np.array([m.x,m.y,m.z],float)
    def _request(self,m):
        request=m.data.upper()
        if request=="ABORT" and self.requested:
            # The driver takes the vehicle back. The perception server stays up
            # so a second DOCK is immediate; only its estimator is re-seeded.
            self.requested=False; self.state="MISSION"; self.cmd.publish(Twist())
            self.owner.publish(String(data="NORMAL"))
            self.debounce=CaptureConfirmationDebouncer(); self.terminal=False
            self.lateral_controller.reset()
            self._camera(False); self._arms(self.ARMS_OPEN)
            self.get_logger().info("docking aborted by the operator")
            return
        if request=="DOCK" and not self.requested:
            self._activate("operator")

    def _mission_complete(self, message):
        if message.data and not self.requested:
            self._activate("mission_complete")

    def _soc(self, message):
        if float(message.data) <= 0.20 and not self.requested:
            self._activate("low_soc")

    def _activate(self, reason):
            self.requested=True; self.state="RETURN_TO_DOCK"; self.terminal=False
            self.lateral_controller.reset()
            self.scenario_b_terminal.reset()
            self.scenario_b_renderer = OculusM750dRenderer(seed=79091)
            self.scenario_b_handover = None
            self.scenario_b_last_fls = None
            with self.lock: self.contact_link=None; self.contact_at=None
            self.reason = reason
            self.owner.publish(String(data="DOCKING"))
            self.control.publish(String(data="manual"))
            try:
                self._start_v6()
            except Exception as exc:
                self.state="SAFE_HOLD"; self.cmd.publish(Twist())
                self.get_logger().error(f"docking perception unavailable: {exc}")

    def _publish_owner(self):
        # CRASHED has to survive this timer. The crash branch publishes it
        # once, and without a case here the next tick overwrote it with
        # DOCKING (the request is still latched), so the bridge never entered
        # its `owner in {DOCKED, CRASHED}` explicit zero-output path and the
        # owner topic reported a crashed vehicle as still docking.
        owner = ("CRASHED" if self.state == "CRASHED" else
                 "DOCKED" if self.state == "DOCKED" else
                 "DOCKING" if self.requested else "NORMAL")
        self.owner.publish(String(data=owner))
    def _start_v6(self):
        if self.provider is not None:
            # Re-arming after an abort: reuse the running server, discard the
            # estimator state that belongs to the abandoned approach.
            self.provider.reset(79091); self.last_image_stamp=None; return
        # Per-process endpoint: a previous demonstrator can never silently
        # answer a new session.  Only this owned path is removed.
        socket=f"/tmp/docking_station_playground_v6_{os.getpid()}_{uuid.uuid4().hex}.sock"
        env=os.environ.copy();env["PYTHONPATH"]=":".join([
            str(P3/"src"), str(SIM/"src/uuv_sim_physics"),
            str(SIM/"src/uuv_mode_aware_navigation")])
        log_root=Path(os.environ.get("DOCK_RUNTIME_ROOT",
                                    os.path.expanduser("~/.cache/docking_station/runtime")))
        log_root.mkdir(parents=True,exist_ok=True)
        self.server_log=(log_root/"playground_transformer_server.log").open("a")
        # PERCEPTION CHECKPOINT SELECTION.
        #
        # The demonstrator now defaults to the PROMOTED rich-world checkpoint
        # (see PROMOTED_CHECKPOINT). The frozen checkpoint cannot fly this
        # world: its lateral slope is -0.21..-0.35, the sign is inverted, and
        # V6 never corroborates. The promoted checkpoint drives 100% of
        # terminal lateral steering at 0, 0.06 and 0.12 m/s across three runs
        # each.
        #
        # Overrides, both explicit:
        #   DOCK_DEV_CHECKPOINT=<path>  serve some other development
        #                                 checkpoint, as the comparison
        #                                 experiments do;
        #   DOCK_USE_FROZEN=1           serve the frozen checkpoint through
        #                                 the frozen server, which hard-codes
        #                                 its own file and refuses to start
        #                                 unless it matches
        #                                 SELECTION_FREEZE.json.
        #
        # Nothing else changes with any of these: not the 25 mm gate, not the
        # V6 filter, not a gain, not the fallback. The frozen checkpoint and
        # the frozen server are untouched on every path.
        dev_ckpt=os.environ.get("DOCK_DEV_CHECKPOINT") or (
            None if os.environ.get("DOCK_USE_FROZEN") else str(PROMOTED_CHECKPOINT))
        if dev_ckpt:
            command=[os.environ.get("DOCK_TORCH_PYTHON", sys.executable),str(P3/"tools/perception_server.py"),
                     "--socket",socket,"--checkpoint",dev_ckpt,"--device","cpu"]
            if os.environ.get("DOCK_DEV_CHECKPOINT"):
                self.get_logger().warning(
                    f"EXPERIMENT: perception overridden to {dev_ckpt}")
            else:
                self.get_logger().info(
                    f"perception: promoted rich-world checkpoint {dev_ckpt}")
        else:
            command=[os.environ.get("DOCK_TORCH_PYTHON", sys.executable),"-m",
                     "docking_perception.integration.heavy_v5_perception_server",
                     "--socket",socket,"--device","cpu"]
        self.server=subprocess.Popen(command,
            env=env,stdout=self.server_log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+30
        while not Path(socket).exists() and time.monotonic()<deadline:
            if self.server.poll() is not None: raise RuntimeError("Transformer server exited")
            time.sleep(.05)
        if not Path(socket).exists(): raise RuntimeError("Transformer socket unavailable")
        self.provider=HeavyV6TerminalProvider(socket);self.provider.reset(79091)

    def _publish_velocity(self,world_velocity,yaw,yaw_target):
        c,s=math.cos(yaw),math.sin(yaw);v=np.asarray(world_velocity,float)
        msg=Twist();msg.linear.x=float(c*v[0]+s*v[1]);msg.linear.y=float(-s*v[0]+c*v[1]);msg.linear.z=float(v[2])
        err=(yaw_target-yaw+math.pi)%(2*math.pi)-math.pi;msg.angular.z=float(np.clip(1.2*err,-.5,.5));self.cmd.publish(msg)
    #: Handover tolerances. Arrival is generous because the estimator's own
    #: position error is a few tenths of a metre and the funnel exists to take
    #: up exactly that; alignment is tight because the terminal estimator was
    #: validated facing straight down the axis.
    ARRIVED_M = 0.35
    ALIGNED_RAD = 0.09
    #: Frames to let through before handover, so the filter's origin is not
    #: taken from a camera that has only just started rendering.
    WARMUP_FRAMES = 8
    #: How far the terminal filter's first fix may disagree with the vehicle's
    #: own navigation estimate of the range to the dock. Two independent
    #: onboard sources: the survey estimator (metre-level, and it has been
    #: tracking all mission) and the terminal perception (centimetre-level
    #: when healthy). A metre apart means the visual origin is wrong, and
    #: since the filter cannot revise its own origin, the answer is to reseed
    #: it rather than to fly on it. No truth is involved.
    ORIGIN_TOLERANCE_M = 1.0
    ORIGIN_RETRIES = 5
    #: How far the vehicle has to stray from the dock before the return leg is
    #: allowed to take it back. Comfortably beyond the standoff, so ordinary
    #: terminal manoeuvring never trips it.
    LOST_M = 6.0

    def _hold_station(self, target, yaw, yaw_target):
        """Sit on a point while turning to face somewhere else."""
        delta = np.asarray(target, float) - self.position
        v = np.clip(.35 * delta, -.2, .2)
        c, s = math.cos(yaw), math.sin(yaw)
        msg = Twist()
        msg.linear.x = float(c*v[0] + s*v[1]); msg.linear.y = float(-s*v[0] + c*v[1])
        msg.linear.z = float(v[2])
        err = (yaw_target - yaw + math.pi) % (2*math.pi) - math.pi
        msg.angular.z = float(np.clip(1.2*err, -.5, .5))
        self.cmd.publish(msg)
        return err

    def _return_leg(self, yaw):
        """Fly to the standoff, then turn onto the approach axis."""
        # The survey spawn lies behind the dock.  Enter the clear positive-x
        # approach corridor before flying to STANDOFF; the former direct line
        # crossed the physical funnel flare and jammed the Heavy.
        if not self.corridor_entered:
            corridor_delta = CORRIDOR_ENTRY - self.position
            if float(np.linalg.norm(corridor_delta)) > self.ARRIVED_M:
                v = np.clip(.45 * corridor_delta /
                            max(np.linalg.norm(corridor_delta), 1e-6), -.45, .45)
                self._publish_velocity(
                    v, yaw, math.atan2(corridor_delta[1], corridor_delta[0]))
                return
            self.corridor_entered = True
        delta = STANDOFF - self.position
        if self.state == "RETURN_TO_DOCK":
            if float(np.linalg.norm(delta)) > self.ARRIVED_M:
                v = np.clip(.45*delta/max(np.linalg.norm(delta), 1e-6), -.45, .45)
                # Point along the track while there is track left to fly, then
                # start swinging onto the approach axis for the last metre.
                heading = (math.atan2(delta[1], delta[0])
                           if float(np.linalg.norm(delta[:2])) > 1.0
                           else APPROACH_YAW)
                self._publish_velocity(v, yaw, heading)
                return
            self.state = "ALIGN_APPROACH"
        error = self._hold_station(STANDOFF, yaw, APPROACH_YAW)
        if not (abs(error) < self.ALIGNED_RAD
                and float(np.linalg.norm(STANDOFF - self.position)) < 0.6):
            return
        # On station and pointed at the dock. Start the camera and let the
        # stream settle BEFORE handing over.
        #
        # The terminal filter adopts its first visual fix unconditionally and
        # NIS-gates everything after it, so one bad frame at handover is not a
        # transient -- it becomes the filter's origin, every later fix is then
        # rejected as an outlier, and the approach dead-reckons to a stop
        # outside the funnel. That is precisely what the first frame off a
        # freshly subscribed camera produced: 2.39 m when the dock was 3.92 m
        # away, followed by 185 consecutive rejections with NIS above 1000.
        self._camera(True)
        with self.lock: warm = self.image_count
        if warm < self.WARMUP_FRAMES:
            if (self.camera_started_at is not None and
                    time.monotonic() - self.camera_started_at > 10.0):
                self.state = "SAFE_HOLD"
                self.get_logger().error(
                    "docking readiness failed: no live camera frames")
            return
        with self.lock: fls_ready = self.fls is not None
        if not fls_ready:
            if (self.camera_started_at is not None and
                    time.monotonic() - self.camera_started_at > 10.0):
                self.state = "SAFE_HOLD"
                self.get_logger().error(
                    "docking readiness failed: realistic FLS unavailable")
            return
        self.provider.reset(79091); self.last_image_stamp = None
        self.hold_point = STANDOFF.copy()
        self.origin_checked = False; self.origin_retries = 0
        self.lateral_controller.reset()
        self.seed_defer_count = 0; self.seed_range_m = None
        self.terminal = True; self.state = "DOCK_ACQUIRE"

    def _step(self):
        now=time.monotonic();dt=min(max(now-self.last_tick,.01),.2);self.last_tick=now
        if not self.requested: self.status.publish(String(data="MISSION")); return
        # Terminal states hold the vehicle instead of re-entering the approach.
        # SAFE_HOLD in particular has no perception provider to call, and
        # DOCKED must not creep if the beam flickers.
        if self.state in ("SAFE_HOLD","DOCKED","CRASHED"):
            self.lateral_controller.reset()
            hold=Twist(); self.cmd.publish(hold)
            self.status.publish(String(data=self.state))
            self._trace_terminal(now,dt,hold); return
        if self.position is None or self.attitude is None:return
        if not self.terminal:
            self._return_leg(float(self.attitude[2]))
            self.status.publish(String(data=self.state)); return
        if float(np.linalg.norm(self.position-DOCK)) > self.LOST_M:
            # The approach has genuinely failed rather than merely closed in.
            self.terminal=False; self.state="RETURN_TO_DOCK"; self._camera(False)
            self.lateral_controller.reset()
            self.cmd.publish(Twist()); self.status.publish(String(data=self.state)); return
        yaw=float(self.attitude[2])
        with self.lock:
            image,fls,dvl,beam=self.image,self.fls,self.dvl,self.beam
            contact_link,latch_confirmed=self._current_contact(now),self.latch_confirmed
        # Support/collar contact is never a valid capture. Funnel contact is a
        # permitted guiding contact only after terminal alignment has begun.
        # A funnel is a guidance surface. Its entire purpose is to receive a
        # vehicle that is slightly off axis and walk it to the throat, and this
        # one is a real 0.60 m mouth narrowing to 0.28 m with real collision
        # geometry. Contact with it was already permitted once the terminal
        # phase had begun; it was still a crash during DOCK_APPROACH, which is
        # the phase in which the vehicle is deliberately flying into the mouth.
        #
        # That mattered because the approach does not arrive perfectly centred.
        # The terminal lateral loop is handed the vehicle about 1.5 s from
        # capture with 30-75 mm of cross-track -- the accumulated dead-reckoning
        # error over a 13 m transit, which is 0.3-0.6% of distance run and good
        # DVL performance -- and it cannot null all of that in the time it has.
        # Runs that touched the cone at 75 mm aborted, where the cone would
        # have guided them in.
        #
        # Structural contact is untouched: the collar rim and the support post
        # are still forbidden in every phase, and they are what a genuine
        # mis-approach hits.
        if is_forbidden_contact(contact_link, self.state):
            self.state="CRASHED"; self.crashed_on=contact_link
            hold=Twist(); self.cmd.publish(hold)
            self.owner.publish(String(data="CRASHED"))
            self.status.publish(String(data=f"CRASHED — DOCK COLLISION: {contact_link}"))
            self.get_logger().error(f"forbidden dock contact: {contact_link}")
            self._trace(now,dt,command=hold); return
        if self.state == "CAPTURE_PENDING":
            hold=Twist(); self.cmd.publish(hold)
            speed = float(np.linalg.norm(dvl[1])) if dvl is not None else float("inf")
            attach_sent=False
            if (beam and speed <= 0.05 and self.capture_pending_since is not None
                    and now-self.capture_pending_since >= self.ARM_CLOSE_TRAVEL_S):
                self.latch_attach.publish(self.Empty()); attach_sent=True
            if latch_confirmed and beam and speed <= 0.05:
                self.state="DOCKED"; self._camera(False)
                self.owner.publish(String(data="DOCKED"))
            self.status.publish(String(data=self.state))
            # Traced AFTER the promotion so the row carries the state this tick
            # ended in. This is the phase a failed capture has to be explained
            # from, so it is recorded every tick, not sampled.
            self._trace(now,dt,command=hold,
                        capture={"attach_sent":attach_sent,"speed_mps":speed,
                                 "capture_pending_s":None if self.capture_pending_since
                                 is None else now-self.capture_pending_since})
            return
        fresh=None if image is None or image[0]==self.last_image_stamp else image[1]
        if image is not None:self.last_image_stamp=image[0]
        if fresh is not None:
            # Same common-medium camera degradation used by the frozen rich-
            # world compatibility qualification.  The slider is a modeled
            # simulation index, never a claimed NTU/FNU measurement.
            from uuv_sim_physics.sensors.particulate_water import (
                MINERAL_SILT, ParticulateWaterMedium, degrade_from_medium)
            observable_range = float(np.linalg.norm((self.position - DOCK)[:2]))
            fresh = degrade_from_medium(
                fresh, ParticulateWaterMedium(
                    MINERAL_SILT, max(self.modeled_turbidity, 0.0)),
                observable_range)
            self.last_camera_hash = hashlib.sha256(fresh.tobytes()).hexdigest()
        if fls is not None:
            self.last_fls_hash = hashlib.sha256(
                np.asarray(fls[1], np.float32).tobytes()).hexdigest()
        # Remedy 1: withhold the seeding fix until the vehicle is inside the
        # band V6 was qualified over. Only the SEED is gated -- once the filter
        # holds a state, every subsequent frame is offered as before.
        #
        # A re-seed forced deeper inside the band was tried instead and was
        # worse (-158 mm seed error at 0.82 m, 0/22 corroboration), and
        # withholding the first seed until deeper still deadlocks: the terminal
        # controller only issues surge while the estimate is valid, so the
        # vehicle parks at the 1.20 m standoff and never reaches the range it
        # is waiting for -- 115 rows, 0 valid fixes, stuck in DOCK_ACQUIRE.
        nav_range=float(np.linalg.norm((self.position-DOCK)[:2]))
        seeded=bool(getattr(self.provider,"have",False))
        in_seed_band=(V6_SEED_WINDOW_M[0]<=nav_range<=V6_SEED_WINDOW_M[1])
        # Raw-network control has no filter origin to seed.  The qualified-band
        # delay belongs only to the estimator/filter paths.
        self.seed_deferred=(self.comparison_mode in {"filter", "full"}
                            and fresh is not None
                            and not seeded and not in_seed_band)
        if self.seed_deferred:
            if self.seed_defer_count==0:
                self.get_logger().info(
                    f"V6 seed deferred at {nav_range:.3f} m; waiting for the "
                    f"{V6_SEED_WINDOW_M[0]:.2f}-{V6_SEED_WINDOW_M[1]:.2f} m band")
            self.seed_defer_count+=1
            fresh=None
        elif fresh is not None and not seeded:
            self.seed_range_m=nav_range
            self.get_logger().info(
                f"V6 seeding at onboard range {nav_range:.3f} m "
                f"(deferred {self.seed_defer_count} frames)")
        self.nav_range_m=nav_range
        obs=SensorObservation(now,dt,fresh,0.,0.,yaw,np.zeros(3),np.zeros(3),float(self.position[2]),
            None if fls is None else fls[1],None if fls is None else fls[0],0 if fls is None else fls[2],0 if fls is None else fls[3],
            None if dvl is None else dvl[1],None if dvl is None else dvl[0],0 if dvl is None else dvl[2],False if dvl is None else dvl[3])
        # Clear the diagnostic capture before each update so a raw comparison
        # can never mistake an older network response for the current frame.
        _RAW.clear()
        # Scenario B uses raw Transformer perception until the fixed 0.60 m
        # handover.  After that instant the network is not queried again: the
        # existing terminal estimator propagates with live onboard DVL and may
        # accept a geometrically available Oculus longitudinal correction.
        if (self.comparison_mode == "handover"
                and self.scenario_b_terminal.state is not None):
            fls_x = self._scenario_b_fls_x(fls)
            velocity = (np.zeros(3) if dvl is None else dvl[1])
            state_b = self.scenario_b_terminal.step(
                velocity_body=velocity, yaw_absolute=yaw,
                approach_yaw=APPROACH_YAW, dt=dt,
                depth_measured=float(self.position[2]),
                dock_z_world=float(DOCK[2]), fls_range_estimate=fls_x)
            self.scenario_b_last_fls = fls_x
            est = ProviderEstimate(
                np.array([state_b.x, state_b.y]), True,
                {"source": "scenario_b_terminal", "dvl_live": dvl is not None,
                 "fls_fixes": state_b.fls_fixes,
                 "fls_rejected": state_b.fls_rejected,
                 "dead_reckoned_m": state_b.dead_reckoned_m},
                mode="TERMINAL")
        else:
            est=self.provider.update(obs)
        if (fresh is not None and "decoded_range_m" in _RAW
                and "decoded_lateral_m" in _RAW):
            self.raw_comparison_xy = np.array(
                [_RAW["decoded_range_m"], _RAW["decoded_lateral_m"]], float)
            self.raw_comparison_at = now
        if (self.comparison_mode == "handover"
                and self.scenario_b_terminal.state is None
                and self.raw_comparison_xy is not None
                and self.raw_comparison_at is not None
                and now - self.raw_comparison_at <= 0.5
                and float(np.linalg.norm(self.raw_comparison_xy)) <= 0.60):
            raw_x, raw_y = map(float, self.raw_comparison_xy)
            state_b = self.scenario_b_terminal.initialise(
                [raw_x, raw_y, float(self.position[2] - DOCK[2]),
                 float((yaw - APPROACH_YAW + math.pi) % (2*math.pi) - math.pi)],
                dock_z_world=float(DOCK[2]),
                depth_measured=float(self.position[2]))
            self.scenario_b_handover = {
                "wall": time.time(), "estimated_range_m": float(
                    np.linalg.norm(self.raw_comparison_xy)),
                "estimated_xy": [raw_x, raw_y],
                "dvl_live": dvl is not None,
            }
            est = ProviderEstimate(
                np.array([state_b.x, state_b.y]), True,
                {"source": "scenario_b_terminal", "dvl_live": dvl is not None,
                 "fls_fixes": 0, "fls_rejected": 0,
                 "dead_reckoned_m": 0.0}, mode="TERMINAL")
        if (self.comparison_mode in {"filter", "full"} and est.valid
                and not self.origin_checked):
            # The filter has just adopted an origin. Cross-check it against the
            # vehicle's own navigation estimate before flying on it.
            expected=float(np.linalg.norm((self.position-DOCK)[:2]))
            got=float(est.xy[0])
            if abs(got-expected)>self.ORIGIN_TOLERANCE_M and self.origin_retries<self.ORIGIN_RETRIES:
                self.origin_retries+=1
                self.get_logger().warning(
                    f"terminal origin {got:.2f} m disagrees with the onboard "
                    f"estimate {expected:.2f} m; reseeding "
                    f"({self.origin_retries}/{self.ORIGIN_RETRIES})")
                self.provider.reset(79091); self.last_image_stamp=None
                self.lateral_controller.reset()
                station=self.hold_point if self.hold_point is not None else self.position
                self._hold_station(station,yaw,APPROACH_YAW)
                self.state="DOCK_ACQUIRE";self.status.publish(String(data=self.state));return
            self.origin_checked=True
            self.get_logger().info(
                f"terminal filter seeded at {got:.2f} m "
                f"(onboard estimate {expected:.2f} m)")
        raw_live = (self.raw_comparison_xy is not None
                    and self.raw_comparison_at is not None
                    and now - self.raw_comparison_at <= 0.5)
        before_b_handover = (self.comparison_mode == "handover"
                             and self.scenario_b_terminal.state is None)
        control_valid = (raw_live if self.comparison_mode == "raw"
                         or before_b_handover else est.valid)
        if control_valid:
            self.hold_point=None
            if self.comparison_mode == "raw" or before_b_handover:
                x, y = map(float, self.raw_comparison_xy)
            else:
                x, y = map(float, est.xy)
            surge=float(np.clip(.65*(x-.2285),0,.25))
            # V6's lateral output carries a constant per-run bias. That, not
            # a sign or a frame, is why the vehicle used to dock off axis.
            #
            # Measured against Gazebo ground truth over full approaches:
            # est_y tracks the true cross-track with the SAME sign and a fixed
            # offset -- est_y - truth = -58.3 mm with a standard deviation of
            # 2.5 mm across true ranges from 1.27 m down to 0.15 m. The offset
            # differs between runs (-58, -80, -94, -119 mm) and is set by ONE
            # thing: the error in the single visual fix the filter seeds from.
            # Correlation between seed error and run bias is +0.996 over the
            # four runs with ground truth.
            #
            # The seed is taken at a true range of about 1.29 m, above the
            # 0.65-1.25 m band V6 was qualified over, where its lateral
            # disagreement with navigation is a median 65 mm against 15 mm in
            # band (pooled over 38 runs). HeavyV6TerminalProvider seeds
            # `self.x = z` directly, the gain is then too small to work that
            # error out, and if the innovations turn persistently same-sign --
            # which is exactly what a seed bias produces -- the lateral monitor
            # reads it as a biased visual stream and quarantines the channel,
            # freezing the error rather than correcting it.
            #
            # The controller then nulls a biased estimate, so the vehicle parks
            # at +bias. Everything observed follows from that.
            #
            # NOTE. An earlier reading of this as a body-vs-world frame error
            # was wrong, and negating V6 on that reading ran the vehicle off
            # axis three times. `y` is already dock-frame cross-track and needs
            # no sign change. The evidence, including those failures, is under
            # internal records
            #
            # Until the seeding is fixed, V6 steers only when the onboard
            # cross-track corroborates it; otherwise the onboard cross-track
            # steers. Measured over two validation runs V6 was corroborated on
            # 0 of 41 terminal samples, disagreeing by 93-122 mm, so lateral
            # guidance is currently coming from navigation and the trace says
            # so on every row. No gain, bound or anti-windup rule changes, and
            # nothing inside V6 changes: its range estimate still drives surge
            # and capture.
            nav_cross_track=float(self.position[1]-DOCK[1])
            v6_cross_track=float(y)
            selection_mode = ("raw" if before_b_handover
                              else self.comparison_mode)
            cross_track_m, self.steering_source, corroborated = (
                select_terminal_cross_track(
                    selection_mode, v6_cross_track, nav_cross_track,
                    getattr(self.provider,"lateral_quarantined",False)))
            self.v6_nav_disagreement_m=float(abs(v6_cross_track-nav_cross_track))
            sway=self.lateral_controller.update(cross_track_m,dt,nav_cross_track)
            msg=Twist();msg.linear.x=surge;msg.linear.y=sway
            msg.angular.z=float(np.clip(1.2*((APPROACH_YAW-yaw+math.pi)%(2*math.pi)-math.pi),-.35,.35))
            self.cmd.publish(msg);self._trace(now,dt,est,fresh is not None,msg)
            self.state="DOCK_TERMINAL" if x<.60 else "DOCK_APPROACH"
        else:
            # Hold the station where acquisition was lost. A zero command is
            # not a stop: the water is still moving, and the vehicle would
            # drift off the axis while the estimator tries to re-acquire.
            if self.hold_point is None:self.hold_point=self.position.copy()
            self.lateral_controller.reset()
            self._hold_station(self.hold_point,yaw,APPROACH_YAW);self.state="DOCK_ACQUIRE"
            self._trace(now,dt,est,fresh is not None)
        speed = float(np.linalg.norm(dvl[1])) if dvl is not None else float("inf")
        if self.state != "CAPTURE_PENDING" and self.debounce.observe(beam):
            self.lateral_controller.reset()
            self.cmd.publish(Twist()); self.state="CAPTURE_PENDING"
            self.capture_pending_since=now; self._arms(self.ARMS_CLOSED)
        if self.state == "CAPTURE_PENDING":
            self.cmd.publish(Twist())
            # The break beam is an operational collar sensor and DVL supplies
            # speed. Only after both remain valid is the mechanical joint asked
            # to engage; the joint's own state is the final confirmation.
            if beam and speed <= 0.05 and now-self.capture_pending_since >= self.ARM_CLOSE_TRAVEL_S:
                self.latch_attach.publish(self.Empty())
            if latch_confirmed and beam and speed <= 0.05:
                self.state="DOCKED";self._camera(False)
                self.owner.publish(String(data="DOCKED"))
        self.status.publish(String(data=self.state))

    def _scenario_b_fls_x(self, fls):
        """Observable Oculus range expressed as vehicle-origin dock x.

        The range is measured from the declared +0.24 m FLS origin to the
        declared +0.07 m front collar plane while the vehicle faces dock -x;
        hence vehicle x = measured range + 0.24 + 0.07.  No simulator pose or
        evaluator truth enters this conversion.
        """
        if fls is None:
            return None
        fan = np.asarray(fls[1], np.float32)
        if fan.size != 512 * 9:
            return None
        acoustic = self.scenario_b_renderer.render(
            fan.reshape(9, 512), acquisition_t=float(fls[0]))
        support = extract_observable_features(acoustic)
        if not support["valid"]:
            return None
        return float(support["range_m"] + 0.24 + 0.07)
    #: How long a dock contact stays current. The sensors republish at 100 Hz
    #: for as long as the surfaces touch, so anything older than this is a
    #: contact that has ended.
    CONTACT_FRESH_S = 0.5

    def _current_contact(self, now):
        """The contact state of THIS moment, not a latch.

        contact_link used to be set once and never cleared, so a touch from an
        earlier phase -- or an earlier docking attempt entirely -- crashed the
        next approach the instant it entered the terminal phase, with the
        vehicle metres away and touching nothing. Caller holds the lock.
        """
        if self.contact_link is None:
            return None
        if self.contact_at is None or now - self.contact_at > self.CONTACT_FRESH_S:
            self.contact_link = None; self.contact_at = None
            return None
        return self.contact_link

    def _sonar_range(self):
        image=getattr(self.provider,"_last_acoustic_image",None)
        if image is None: return None
        try:
            from uuv_sim_physics.sensors.oculus_m750d import extract_observable_features
            support=extract_observable_features(image)
        except Exception:  # noqa: BLE001 - diagnostics must not stop an approach
            return None
        return None if not support["valid"] else float(support["range_m"])

    #: Heartbeat for a settled terminal state. DOCKED can last for hours, and
    #: a 20 Hz record of a vehicle that is not moving is not evidence, it is
    #: volume. Every CHANGE is still written the tick it happens -- state,
    #: beam, latch, contact and debounce are all in the change key -- so a
    #: beam flicker or a latch dropout cannot be sampled away.
    TERMINAL_HEARTBEAT_S = 0.5

    def _terminal_key(self):
        return (self.state, bool(self.beam), bool(self.latch_confirmed),
                self.contact_link, int(self.debounce.count))

    def _trace_terminal(self,now,dt,command):
        key=self._terminal_key()
        changed = key != self.last_terminal_key
        due = (self.last_terminal_trace is None
               or now-self.last_terminal_trace >= self.TERMINAL_HEARTBEAT_S)
        if not (changed or due): return
        self.last_terminal_key=key; self.last_terminal_trace=now
        self._trace(now,dt,command=command,
                    capture={"reason":"change" if changed else "heartbeat"})

    def _trace(self,now,dt,est=None,fresh=False,command=None,capture=None):
        """What the terminal filter believed, tick by tick.

        Onboard values only -- this is the controller's own view, written so a
        failed approach can be diagnosed without guessing. Truth lives in the
        recorder's telemetry and is joined afterwards on the sim clock.

        `est` is optional because the terminal states do not run the provider:
        CAPTURE_PENDING and DOCKED hold the vehicle and return early. Tracing
        used to stop at exactly that point, so the log went silent on the tick
        the interesting part began -- a capture that failed left no record of
        its beam, its debounce or its latch. Those ticks are now recorded with
        the filter columns null.
        """
        if self.trace is None:return
        # The terminal branches trace before the position/attitude guard, and
        # a diagnostic must never be able to stop an approach it is only
        # watching.
        if self.position is None or self.attitude is None:return
        xy=getattr(est,"xy",None)
        self.trace.write(json.dumps({
            # Both clocks. dt and the filter run on sim time; the recorder
            # stamps wall time, and the two logs have to be joinable.
            "mono":now,"wall":time.time(),"dt":dt,"state":self.state,
            "fresh_image":bool(fresh),
            "surge":None if command is None else float(command.linear.x),
            "sway":None if command is None else float(command.linear.y),
            "sway_integral_mps":float(getattr(
                getattr(self, "lateral_controller", None),
                "integral_output_mps", 0.0)),
            "yaw_rate":None if command is None else float(command.angular.z),
            "valid":None if est is None else bool(est.valid),
            "est_x":None if xy is None else float(xy[0]),
            "est_y":None if xy is None else float(xy[1]),
            "mode":getattr(est,"mode",None),
            "detail":{k:v for k,v in getattr(est,"detail",{}).items()
                      if k in ("visual_update","visual_accepted","innovation_nis",
                               "standardized_lateral_innovation",
                               "lateral_quarantine_triggered",
                               "reported_sigma_xy","filter_sigma_xy","dvl_age_s",
                               "dvl_live","absolute_misses","reason",
                               "lateral_state_reseeded",
                               "visual_motion_consistent",
                               "visual_motion_residual_m")},
            "quarantined":getattr(self.provider,"lateral_quarantined",None),
            # Remedy-1 instrumentation. Diagnostics only: nothing here is read
            # by the controller.
            "nav_range_m":getattr(self,"nav_range_m",None),
            "v6_seeded":bool(getattr(self.provider,"have",False)),
            "seed_deferred":bool(getattr(self,"seed_deferred",False)),
            "seed_defer_count":int(getattr(self,"seed_defer_count",0)),
            "seed_range_m":getattr(self,"seed_range_m",None),
            # Raw network measurement before filtering, diagnostics only.
            "raw_decoded_lateral_m":_RAW.get("decoded_lateral_m"),
            "raw_decoded_range_m":_RAW.get("decoded_range_m"),
            "raw_decoded_sigma":_RAW.get("decoded_sigma"),
            "steering_source":getattr(self,"steering_source",None),
            "terminal_comparison_mode":getattr(self,"comparison_mode","full"),
            "scenario_b_handover":getattr(self,"scenario_b_handover",None),
            "scenario_b_fls_x_m":getattr(self,"scenario_b_last_fls",None),
            "scenario_b_terminal_state":None if getattr(
                self,"scenario_b_terminal",None) is None or
                self.scenario_b_terminal.state is None else {
                    "dead_reckoned_m":self.scenario_b_terminal.state.dead_reckoned_m,
                    "fls_fixes":self.scenario_b_terminal.state.fls_fixes,
                    "fls_rejected":self.scenario_b_terminal.state.fls_rejected},
            "v6_nav_disagreement_m":getattr(self,"v6_nav_disagreement_m",None),
            # The sonar's own detected range, independent of the network. If
            # the acoustic image says 4 m while the network says 2.4 m, the
            # fault is in the network's input; if both say 2.4 m, the vehicle
            # really is where the network thinks it is.
            "sonar_range_m":self._sonar_range(),
            "camera_sha256":self.last_camera_hash,
            "fls_sha256":self.last_fls_hash,
            "modeled_turbidity":self.modeled_turbidity,
            "break_beam":bool(self.beam),
            "break_beam_hit_identity":"bluerov2_heavy::capture_beam_target" if self.beam else None,
            "break_beam_debounce_count":int(self.debounce.count),
            "break_beam_origin_dock":list(BEAM_ORIGIN_DOCK),
            "break_beam_direction_dock":list(BEAM_DIRECTION_DOCK),
            "dock_contact":self.contact_link,
            "crashed_on":self.crashed_on,
            "latch_confirmed":bool(self.latch_confirmed),
            "estimated_position":[float(v) for v in self.position],
            "yaw":float(self.attitude[2]),
            # Capture-phase fields: whether the attach was commanded this tick,
            # the speed the gate saw, and how long CAPTURE_PENDING has been
            # held. Null outside the capture phase.
            "capture":capture,
        })+"\n")

    def destroy_node(self):
        try:
            self.cmd.publish(Twist()); self.owner.publish(String(data="NORMAL"))
        except Exception:
            pass
        if self.server is not None:terminate_owned_process(self.server)
        if hasattr(self,"server_log"):self.server_log.close()
        return super().destroy_node()

def main():
    rclpy.init();n=PlaygroundDockingSupervisor()
    try:rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally:n.destroy_node();rclpy.try_shutdown()
if __name__=="__main__":main()
