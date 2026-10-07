"""V5 decoding with protection against correlated terminal lateral bias."""
from __future__ import annotations

import numpy as np

from docking_perception.integration.heavy_v5_terminal_provider import (
    HeavyV5TerminalProvider,
)


class LateralInnovationMonitor:
    """Detect a genuinely biased visual lateral channel.

    Quarantine holds the lateral state and ignores the visual channel once the
    standardized lateral innovation has stayed above STANDARDIZED_LIMIT with
    the same sign for REQUIRED_CONSECUTIVE samples.

    Motion consistency: a refuted hypothesis, kept as instrumentation
    -------------------------------------------------------------------
    A persistent same-sign innovation was thought to have two separable causes:
    a biased VISUAL channel, or a wrong FILTER STATE with a healthy visual. The
    supporting observation was that across four integrated trials the final
    filtered lateral error equalled the SEEDING error almost exactly -- +12.6
    -> +10.0, +11.8 -> +9.0, +18.5 -> +13.1 and, under 0.06 m/s cross-current,
    +94.7 -> +89.5 mm -- which reads as a state that was seeded badly and never
    recovered, with quarantine freezing out the measurements that would have
    fixed it.

    The proposed discriminator asked whether the visual channel still tracked
    MOTION the DVL independently measured, comparing the visual's change since
    a standing anchor against the DVL-integrated displacement over the same
    window. On that evidence the monitor would waive quarantine, and a diverged
    state would be re-seeded from the "verified healthy" sensor.

    It was implemented, unit-tested, and measured over eight integrated runs.
    IT IS WRONG, for a reason that is worth keeping:

      A motion check establishes that a sensor tracks CHANGE. It cannot
      establish ABSOLUTE accuracy, and a constant offset produces no motion
      inconsistency whatsoever.

    Every re-seed it authorised made the state worse, because the measurement
    it adopted was itself biased -- at 0.96 m it took +51.5 mm against a truth
    of +11.8; at 0.57 m it took +72.5 against -0.1, which alone cost a
    zero-current run its 100% V6 primary rate; and in the fault-injection run
    it adopted +99.6 mm, which WAS the deliberately injected 0.12 m bias. The
    earlier argument that a constant offset cannot hide (the filter seeds from
    the visual, so the offset is inherited and produces no innovation) fails
    whenever the offset appears after seeding, which is exactly the injected
    fault.

    The premise was wrong too. The raw candD lateral channel in this world does
    not have the assumed accuracy: above 0.4 m its median error is +11 mm at
    zero current, +40 mm at 0.06 m/s and +41..+86 mm at 0.12 m/s, and below
    0.4 m it collapses entirely (median 176-234 mm, max 504 mm). The filtered
    V6 error under current, 47-80 mm, simply tracks that. The state was not
    diverging from a healthy sensor; the sensor was biased from the seed
    onward, and the state faithfully reproduced it. Quarantine firing under
    current is therefore CORRECT containment of a genuinely biased channel, and
    the navigation fallback that takes over is the system working as designed.

    So the gating was reverted. The measurement is still computed and reported
    in the detail dict (visual_motion_consistent, visual_motion_residual_m)
    because it is informative diagnostics, but it does not influence
    quarantine, the filter state, or the controller. Any future use of it must
    first establish absolute accuracy by some means a motion check cannot
    provide.

    No truth is used. The only inputs are the visual measurement, its reported
    sigma, and DVL-derived motion already present in the filter.
    """

    #: Standardized lateral innovation beyond which a sample counts toward a
    #: fault, as a two-sided 99% point of the standard normal (scipy:
    #: norm.ppf(0.995)).
    #:
    #: This matches the confidence the filter ALREADY uses elsewhere. The NIS
    #: gate a few lines below admits measurements at NIS_LIMIT = 9.21034, which
    #: is chi-square(0.99) on two degrees of freedom -- a 1% false-positive
    #: design point. The lateral monitor previously sat at 1.0 sigma, roughly a
    #: 32% false-positive rate per sample, so the two gates inside one filter
    #: disagreed about what "unlikely" means by a factor of about thirty.
    #:
    #: The cost was measured, not assumed. Across twelve integrated candU runs,
    #: 18.5% of terminal samples exceeded 1.0 sigma, and every one of the three
    #: quarantine events fired at the bare threshold (|z| = 1.00, 1.01, 1.09)
    #: after which the innovations returned to normal -- textbook false alarms.
    #: Because the flag latches for the rest of the approach, each cost V6 the
    #: majority of terminal steering in a run whose estimate was in fact
    #: agreeing with navigation to 11-14 mm, well inside the 25 mm gate.
    #:
    #: This is a calibration correction, authorised on 2026-09-12, not a
    #: relaxation to make a run pass: the value is derived from the filter's
    #: own existing confidence level rather than tuned against results, and
    #: containment is re-verified against injected biases at 30, 50, 80 and
    #: 120 mm. Detection of a genuinely biased channel is unchanged in kind --
    #: a real bias produces innovations far beyond 2.576 sigma, and the NIS
    #: gate and the 25 mm navigation corroboration gate are both untouched.
    STANDARDIZED_LIMIT = 2.576
    REQUIRED_CONSECUTIVE = 3
    #: Envelope for the reported motion residual, in multiples of the combined
    #: measurement and dead-reckoning noise. Two measurements contribute (the
    #: anchor and the current sample), hence the factor of two on the
    #: measurement variance. Diagnostics only -- nothing gates on it.
    MOTION_CONSISTENCY_SIGMAS = 3.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.sign = 0
        self.count = 0
        self.motion_consistent = None

    def observe(self, standardized, motion_consistent=None):
        """Return (trigger, started).

        ``motion_consistent`` is True when the visual channel's change since
        the motion anchor matched the DVL-measured displacement over the same
        window, False when it did not, and None when it could not be evaluated
        (no anchor yet, or no live DVL). Unknown is treated as the original
        behaviour, so the containment path is never weakened by missing
        information.
        """
        if abs(float(standardized)) <= self.STANDARDIZED_LIMIT:
            self.reset()
            return False, False
        sign = 1 if standardized > 0.0 else -1
        started = sign != self.sign
        if started:
            self.sign = sign
            self.count = 1
        else:
            self.count += 1
        # Recorded, deliberately not acted on. See the class docstring: the
        # motion evidence was tested as a quarantine exemption and refuted.
        self.motion_consistent = motion_consistent
        return self.count >= self.REQUIRED_CONSECUTIVE, started


class HeavyV6TerminalProvider(HeavyV5TerminalProvider):

    """Retain V5/V4 behavior while isolating persistent lateral visual bias."""

    def reset(self, seed):
        super().reset(seed)
        self.lateral_monitor = LateralInnovationMonitor()
        self.lateral_quarantined = False
        self._lateral_snapshot = None
        #: Motion-consistency anchor: a past visual lateral measurement, and
        #: the DVL-integrated lateral displacement and elapsed time since it
        #: was taken. The comparison is made against the ANCHOR rather than
        #: against the previous sample, because a per-sample check cannot see
        #: a slow drift: at 0.02 m/s each 0.12 s step moves 2.4 mm, far inside
        #: a ~28 mm measurement noise, so every individual step looks healthy
        #: while the channel walks away. Differencing against a fixed anchor
        #: telescopes the visual noise -- the residual is
        #: (y_now - y_anchor) - dvl_displacement regardless of how many
        #: samples separate them -- so a real drift accumulates linearly while
        #: the tolerance grows only with the DVL and process noise.
        self._anchor_visual_y = None
        self._anchor_dvl_lateral = 0.0
        self._anchor_elapsed = 0.0
        self._motion_consistent = None
        self._motion_residual = None

    def _set_motion_anchor(self, z):
        """Re-anchor the motion check on the current visual measurement.

        Called when there is nothing left to explain -- the state and the
        sensor agree, or the state has just been taken from the sensor -- so
        the integration window stays short during healthy tracking and only
        grows while a disagreement is actually outstanding.
        """
        self._anchor_visual_y = float(z[1])
        self._anchor_dvl_lateral = 0.0
        self._anchor_elapsed = 0.0

    def _propagate_snapshot(self, obs, dvl_live):
        if self._lateral_snapshot is None or not dvl_live:
            return
        y, variance = self._lateral_snapshot
        q = ((self.DVL_NOISE_MPS * obs.dt) ** 2
             + self.PROCESS_VELOCITY_MPS ** 2 * obs.dt)
        self._lateral_snapshot = (y + self.vel[1] * obs.dt, variance + q)

    def update(self, obs):
        from uuv_sim_physics.docking.state_providers import ProviderEstimate

        dvl_live, _ = self._predict_dvl(obs)
        self._propagate_snapshot(obs, dvl_live)
        if dvl_live:
            # World-frame lateral displacement the DVL says has happened since
            # the motion anchor. self.vel is already the DVL velocity rotated
            # into the world frame by _predict_dvl.
            self._anchor_dvl_lateral += float(self.vel[1]) * obs.dt
            self._anchor_elapsed += obs.dt
        else:
            # Without the DVL the displacement since the anchor is unknown, so
            # the window is abandoned rather than compared against an
            # under-counted integral. The next measurement re-anchors and the
            # check reports "unknown" until then, which the monitor treats as
            # the original, containment-preserving behaviour.
            self._anchor_visual_y = None
        candidate, reason = self._observe(obs)
        accepted = False
        nis = None
        sigma = None
        standardized_lateral = None
        triggered = False

        if candidate is not None:
            z, sg, support = candidate
            measurement_covariance = np.diag(sg ** 2)
            sigma = sg.tolist()
            # Does the visual channel still track motion the DVL measures?
            self._motion_consistent = None
            self._motion_residual = None
            if self._anchor_visual_y is None:
                self._set_motion_anchor(z)
            elif dvl_live and self._anchor_elapsed > 0.0:
                residual = ((float(z[1]) - self._anchor_visual_y)
                            - self._anchor_dvl_lateral)
                # Two measurement noises (anchor and current), plus the
                # velocity error integrated over the window: the DVL's own
                # noise and the unmodelled velocity the filter already budgets
                # for in its process noise.
                tolerance = (self.lateral_monitor.MOTION_CONSISTENCY_SIGMAS
                             * float(np.sqrt(
                                 2.0 * float(sg[1]) ** 2
                                 + (self.DVL_NOISE_MPS
                                    * self._anchor_elapsed) ** 2
                                 + (self.PROCESS_VELOCITY_MPS
                                    * self._anchor_elapsed) ** 2)))
                self._motion_residual = float(residual)
                self._motion_consistent = bool(abs(residual) <= tolerance)
            if not self.have:
                self.x = z.copy()
                self.P = measurement_covariance.copy()
                self.have = True
                accepted = True
                nis = 0.0
            else:
                innovation = z - self.x
                innovation_covariance = self.P + measurement_covariance
                nis = float(innovation @ np.linalg.solve(
                    innovation_covariance, innovation))
                standardized_lateral = float(
                    innovation[1] / np.sqrt(innovation_covariance[1, 1]))
                if nis <= self.NIS_LIMIT:
                    if not self.lateral_quarantined:
                        triggered, started = self.lateral_monitor.observe(
                            standardized_lateral, self._motion_consistent)
                        if started:
                            self._lateral_snapshot = (
                                float(self.x[1]), float(self.P[1, 1]))

                    gain = self.P @ np.linalg.inv(innovation_covariance)
                    updated = self.x + gain @ innovation
                    identity = np.eye(2)
                    updated_covariance = (
                        (identity - gain) @ self.P @ (identity - gain).T
                        + gain @ measurement_covariance @ gain.T)

                    if self.lateral_quarantined or triggered:
                        if triggered:
                            self.lateral_quarantined = True
                        updated[1], updated_covariance[1, 1] = (
                            self._lateral_snapshot)
                        updated_covariance[0, 1] = 0.0
                        updated_covariance[1, 0] = 0.0
                    self.x = updated
                    self.P = updated_covariance
                    accepted = True
                    # Deliberately NOT re-anchored here. Agreement between the
                    # state and the sensor is not evidence that the sensor is
                    # right: when the channel drifts slowly the filter simply
                    # follows it, and agreement is the fault's own signature.
                    # Only the DVL can referee that, and it does so over the
                    # whole window, so the anchor is left standing.
                reason = "accepted" if accepted else "innovation_rejected"

        self.absolute_misses = 0 if accepted else self.absolute_misses + 1
        if self.safe:
            if accepted or (dvl_live and self.recovery_live >= 3):
                self.safe = False
        elif self.absolute_misses >= 3 and not dvl_live:
            self.safe = True
            self.recovery_live = 0

        detail = {
            "source": "v6",
            "visual_update": reason,
            "visual_accepted": accepted,
            "innovation_nis": nis,
            "standardized_lateral_innovation": standardized_lateral,
            "lateral_visual_quarantined": self.lateral_quarantined,
            "lateral_quarantine_triggered": triggered,
            "visual_motion_consistent": self._motion_consistent,
            "visual_motion_residual_m": self._motion_residual,
            "reported_sigma_xy": sigma,
            "filter_sigma_xy": np.sqrt(
                np.maximum(np.diag(self.P), 0.0)).tolist(),
            "dvl_age_s": self.dvl_age,
            "dvl_live": dvl_live,
            "dvl_locked": obs.dvl_beams_locked,
            "absolute_misses": self.absolute_misses,
        }
        if not self.have:
            return ProviderEstimate(
                self.x.copy(), False, detail, velocity_xy=self.vel.copy(),
                mode="ACQUIRE")
        if self.safe:
            mode = "SAFE_HOLD"
        elif accepted:
            mode = "OVERLAP" if self.x[0] <= 0.8 and dvl_live else "TRACK"
        elif dvl_live:
            mode = "DONE" if self.x[0] <= 0.05 else "COMMIT"
        else:
            mode = "TRACK"
        return ProviderEstimate(
            self.x.copy(), not self.safe, detail, velocity_xy=self.vel.copy(),
            mode=mode)
