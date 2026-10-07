"""Continuous DVL return navigation followed by qualified V6 docking."""
from __future__ import annotations

import math

import numpy as np

from docking_perception.integration.heavy_v6_terminal_provider import (
    HeavyV6TerminalProvider,
)
from uuv_mode_aware_navigation.energy import (
    EnergyBreakdown,
    FullPlatformEnergyModel,
)
from uuv_sim_physics.docking.autonomy import (
    DockingAutonomySupervisor,
    ObservableDockingInput,
    SupervisorConfig,
)
from uuv_sim_physics.docking.state_providers import ProviderEstimate


class HeavyV6ReturnProvider:
    """Truth-free mapped return, local acquisition, and V6 terminal estimate."""

    hardware_realisable = True
    requires = ("camera", "realistic_fls", "dvl", "imu", "depth")
    ACQUISITION_RADIUS_M = 1.25  # V3's predeclared operating-envelope boundary
    DVL_NOISE_MPS = 0.0005
    PROCESS_VELOCITY_MPS = 0.002
    LOW_SOC_START = 0.2001
    OPTICAL_POWER_W = 12.0
    ACOUSTIC_POWER_W = 4.0
    ACTUATION_POWER_W = 2.0

    def __init__(self, socket_path, initial_nav_xy, trigger):
        if trigger not in {"operator", "mission_complete", "low_soc", "external"}:
            raise ValueError(f"unsupported trigger {trigger!r}")
        self.terminal = HeavyV6TerminalProvider(socket_path)
        self.initial_nav_xy = np.asarray(initial_nav_xy, dtype=float)
        self.trigger = trigger

    def request_dock(self, reason="operator"):
        """Inject the shared DOCK request received through an external UI.

        This is intentionally only a request edge.  Return, acquisition,
        terminal estimation, control and capture remain owned by the same
        supervisor/path used by the qualified headless triggers.
        """
        if self.request_emitted or self.external_reason is not None:
            return False
        self.external_reason = str(reason)
        return True

    def reset(self, seed):
        self.terminal.reset(seed)
        self.nav_xy = self.initial_nav_xy.copy()
        self.nav_covariance = np.eye(2) * 0.05 ** 2
        self.velocity = np.zeros(2)
        self.last_dvl_stamp = None
        self.request_emitted = False
        self.external_reason = None
        self.local_active = False
        self.supervisor = DockingAutonomySupervisor(SupervisorConfig(
            acquisition_radius_m=self.ACQUISITION_RADIUS_M))
        self.supervisor.reset(mission_active=True)
        initial_used = (1.0 - self.LOW_SOC_START) * 2.4e6
        self.energy = FullPlatformEnergyModel(
            breakdown=EnergyBreakdown(propulsion_j=initial_used))

    def _update_navigation(self, obs):
        fresh = (obs.dvl_acquisition_t is not None
                 and (self.last_dvl_stamp is None
                      or obs.dvl_acquisition_t > self.last_dvl_stamp))
        if fresh:
            self.last_dvl_stamp = obs.dvl_acquisition_t
        if fresh and obs.dvl_valid and obs.dvl_velocity is not None:
            body = np.asarray(obs.dvl_velocity, dtype=float)
            c, s = math.cos(obs.yaw), math.sin(obs.yaw)
            self.velocity = np.array([
                c * body[0] - s * body[1],
                s * body[0] + c * body[1],
            ])
        if obs.dvl_valid:
            self.nav_xy += self.velocity * obs.dt
            q = ((self.DVL_NOISE_MPS * obs.dt) ** 2
                 + self.PROCESS_VELOCITY_MPS ** 2 * obs.dt)
            self.nav_covariance += np.eye(2) * q

    def _shared_request(self, obs):
        speed = float(np.linalg.norm(self.velocity))
        self.energy.step(
            speed, self.OPTICAL_POWER_W, self.ACOUSTIC_POWER_W,
            self.ACTUATION_POWER_W, obs.dt)
        if self.request_emitted:
            return None, None
        reason = None
        if self.trigger == "external" and self.external_reason is not None:
            reason = self.external_reason
        elif self.trigger == "operator":
            reason = "operator"
        elif self.trigger == "mission_complete":
            reason = "mission_complete"
        elif self.energy.state_of_charge <= self.supervisor.config.low_soc_fraction:
            reason = "low_soc"
        if reason is not None:
            self.request_emitted = True
            return "DOCK", reason
        return None, None

    def update(self, obs):
        self._update_navigation(obs)
        request, trigger_reason = self._shared_request(obs)
        if request is not None:
            self.supervisor.request(request)

        within_envelope = self.nav_xy[0] <= self.ACQUISITION_RADIUS_M
        terminal_estimate = None
        if self.request_emitted and within_envelope:
            terminal_estimate = self.terminal.update(obs)
            if terminal_estimate.valid:
                self.local_active = True

        sigma = float(np.sqrt(np.max(np.diag(self.nav_covariance))))
        dock_fix = bool(terminal_estimate is not None
                        and terminal_estimate.valid)
        dock_pose = None
        if dock_fix:
            dock_pose = (float(terminal_estimate.xy[0]),
                         float(terminal_estimate.xy[1]),
                         float(obs.depth_measured),
                         float(obs.yaw))
        command = self.supervisor.step(ObservableDockingInput(
            # Horizontal return uses the dock-depth navigation datum; depth is
            # controlled independently from the Bar30 path in the runner.
            nav_position_m=tuple(self.nav_xy) + (0.0,),
            nav_uncertainty_m=sigma,
            dock_relative_pose=dock_pose,
            dock_fix_valid=dock_fix,
        ))

        common = {
            "source": "v6_integrated_return",
            "trigger": self.trigger,
            "shared_request": request,
            "trigger_reason": trigger_reason,
            "request_emitted": self.request_emitted,
            "autonomy_state": command.state.value,
            "autonomy_behavior": command.behavior,
            "autonomy_reason": command.reason,
            "navigation_sigma_m": sigma,
            "navigation_xy_m": self.nav_xy.tolist(),
            "local_v6_active": self.local_active,
            "soc_fraction": self.energy.state_of_charge,
            "dvl_valid": bool(obs.dvl_valid),
        }
        if self.local_active and terminal_estimate is not None:
            return ProviderEstimate(
                terminal_estimate.xy.copy(), terminal_estimate.valid,
                {**terminal_estimate.detail, **common},
                velocity_xy=terminal_estimate.velocity_xy,
                mode=terminal_estimate.mode)

        # Before a request the vehicle holds. Once requested, the declared
        # initial estimate and DVL drive the mapped return continuously.
        xy = self.nav_xy.copy() if self.request_emitted else np.zeros(2)
        velocity = self.velocity.copy() if self.request_emitted else np.zeros(2)
        return ProviderEstimate(
            xy, True, common, velocity_xy=velocity,
            mode=("RETURN_TO_DOCK" if self.request_emitted else "MISSION_HOLD"))
