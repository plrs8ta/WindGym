"""Two-layer Python transcription of the reference's normal automatic yaw path.

POU order: TrackingNacelle -> YawAutomaticMode/TriggerYawMotor -> startup stop.
Program inputs and FB memory survive scans. Diagnostics are not control inputs.
"""

from .config import PLCInputs, PLCYawConfig
from .timers import TON, TP, Edge, milliseconds


class YawTime:
    """One direction's FB; stored vane sign is independent of motor direction."""

    def __init__(self, config):
        self.config = config
        self.start_edge = Edge()
        self.tp_fall = Edge(falling=True)
        self.physical_fall = Edge(falling=True)
        self.auto_edge = Edge()
        self.pulse = TP()
        self.absolute_stop = TON()
        self.output = False
        self.positive = self.negative = self.extension = False
        self.t = self.h = self.t_max = self.h_max = 0
        self.phase = "idle"
        self.accepted = False
        self.accepted_t = self.accepted_h = 0
        self.stop_reason = ""

    def scan(self, now, start, theta1, theta30, motor_active, automatic):
        self.accepted = False
        self.stop_reason = ""
        if self.start_edge(start) and not self.pulse.q and not self.output:
            self.positive = theta30 > 0
            self.negative = not self.positive
            self.extension = False
            speed = self.config.yaw_speed_deg_min
            self.t_max = milliseconds(180 * 60 / speed)
            self.h_max = milliseconds(90 * 60 / speed)
            self.t = min(milliseconds(abs(theta30) * 60 / speed), self.t_max)
            self.h = min(milliseconds(abs(theta30) * 30 / speed), self.h_max)
            self.accepted = True
            self.accepted_t, self.accepted_h = self.t, self.h
        self.output = self.pulse(now, start, self.t)
        half_elapsed = self.pulse.et >= self.h and self.pulse.q
        if self.tp_fall(self.pulse.q):
            self.extension = True
        # Intentionally AFTER half_elapsed, exactly as the ST source.
        if self.auto_edge(automatic):
            self.h = self.h_max
        absolute = self.absolute_stop(now, self.extension, self.h)
        self.phase = "supervised" if half_elapsed else "minimum"
        if (
            self.pulse.q
            and half_elapsed
            and ((self.positive and theta1 <= 1) or (self.negative and theta1 >= -1))
        ):
            self.output = False
            self.stop_reason = "supervised_alignment"
        if self.extension:
            self.phase = "extension"
            # Equality continues. Last assignment (absolute stop) takes priority.
            self.output = (self.positive and theta1 >= 1) or (
                self.negative and theta1 <= -1
            )
            if not self.output:
                self.stop_reason = "extension_alignment"
            if absolute:
                self.output = False
                self.stop_reason = "absolute_timeout"
        physical_fall = self.physical_fall(motor_active)
        if not self.output or physical_fall:
            self.positive = self.negative = self.extension = False
            self.t = self.h = 0
            # The ST resets FB memory here; it does NOT force bStartYaw false
            # on this scan when the physical motor falls.
        if not self.output:
            self.phase = "idle"
        return self.output


class MotorSequence:
    """PRG_TriggerYawMotor, including retained request inputs and stop TP."""

    def __init__(self, config):
        self.config = config
        self.request_cw = self.request_ccw = self.request_stop = False
        self.cw = self.ccw = self.electrical_open = False
        self.valves = (False, False, False)
        self.stop_finished = False
        self.stop_edge = Edge()
        self.stop_tp = TP()
        self.open_delay = TON()
        self.start_delay = TON()
        self.stop_delay = TON()
        self.close_delay = TON()
        self.pressure_pulse = TP()
        self.phase = "stopped"
        self.stop_condition = ""
        self.wind_stop_seconds = 1.0
        self.pressure_target = 0.0

    def scan(self, now, inputs, automatic, *, cw=None, ccw=None, stop=None):
        c = self.config
        for name, value in [
            ("request_cw", cw),
            ("request_ccw", ccw),
            ("request_stop", stop),
        ]:
            if value is not None:
                setattr(self, name, value)
        if self.stop_edge(self.request_stop):
            self.request_cw = self.request_ccw = False
            self.stop_finished = False
        self.stop_tp(now, self.request_stop, 22_000)
        for direction in ("cw", "ccw"):
            if getattr(self, "request_" + direction) and not self.stop_tp.q:
                self.stop_delay(now, False)
                self.close_delay(now, False)
                self.pressure_pulse(now, False)
                self.stop_finished = False
                self.stop_condition = ""
                self.valves = (True, not automatic, automatic)
                self.phase = "brake_opening"
                if self.open_delay(now, True, milliseconds(c.brake_open_seconds)):
                    self.electrical_open = True
                    self.phase = "motor_start_delay"
                    if self.start_delay(now, True, milliseconds(c.motor_start_seconds)):
                        self.open_delay(now, False)
                        self.start_delay(now, False)
                        setattr(self, direction, True)
                        setattr(self, "request_" + direction, False)
                        self.phase = "running_" + direction
        constant, lower, upper = c.wind_time_constant, c.wind_time_min, c.wind_time_max
        if not constant or not lower or not upper:
            constant, lower, upper = 5, 5, 13  # Explicit fallback in ST.
        wind_time = min(upper, max(lower, (inputs.wind_5s_mps - 5) * 2 + constant))
        # REAL_TO_TIME(WindTime) * 1000 rounds BEFORE multiplication.
        self.wind_stop_seconds = (
            milliseconds(wind_time / 1000) if c.wind_time_enabled else 1
        )
        pressure_limits = {
            1: (40, 90),
            2: (30, 75),
            3: (25, 50),
            4: (c.pressure_low, c.pressure_high),
        }
        lo, hi = pressure_limits[c.pressure_mode]
        self.pressure_target = min(
            hi,
            max(
                lo,
                lo
                + (hi - lo)
                * (inputs.wind_5s_mps - c.pressure_wind_low)
                / (c.pressure_wind_high - c.pressure_wind_low),
            ),
        )
        if self.stop_tp.q and not self.stop_finished:
            self.valves = (False, False, False)
            self.open_delay(now, False)
            self.start_delay(now, False)
            self.phase = "stop_pressure"
            delay = {1: 1.0, 2: self.wind_stop_seconds, 3: c.pressure_safety_seconds}[
                c.stop_mode
            ]
            elapsed = self.stop_delay(now, True, milliseconds(delay))
            if c.stop_mode == 3:
                exceeded = inputs.pressure_1s > self.pressure_target  # Strict >.
                stop_ready = self.pressure_pulse(
                    now,
                    (exceeded or elapsed) and self.request_stop,
                    milliseconds(c.brake_close_seconds + 0.5),
                )
                if stop_ready and not self.stop_condition:
                    self.stop_condition = "pressure" if exceeded else "safety_timeout"
            else:
                stop_ready = elapsed
                if stop_ready:
                    self.stop_condition = (
                        "fixed_delay" if c.stop_mode == 1 else "wind_delay"
                    )
            if stop_ready:
                self.cw = self.ccw = False
                self.phase = "brake_closing"
                if self.close_delay(now, True, milliseconds(c.brake_close_seconds)):
                    self.electrical_open = False
                    self.stop_delay(now, False)
                    self.close_delay(now, False)
                    if c.stop_mode == 3:
                        self.pressure_pulse(now, False)
                    self.request_stop = False
                    self.stop_finished = True
                    self.phase = "stopped"
        # These FB calls exist at the bottom of the ST. Same timestamp, not +dt.
        for timer in (
            self.start_delay,
            self.open_delay,
            self.stop_delay,
            self.close_delay,
            self.pressure_pulse,
        ):
            timer(now)


class PLCYawController:
    """Deterministic scan executor for automatic tracking; inactive modes stop.

    Numeric mode constants and task declarations are absent from the XML.
    auto_mode is therefore an explicit selector, not an invented giYPLevel.
    Slip/frozen/manual/emergency/untwist/typhoon programs are not emulated.
    """

    def __init__(self, config=None):
        self.config = config or PLCYawConfig()
        self.motor = MotorSequence(self.config)
        self.cw_time = YawTime(self.config)
        self.ccw_time = YawTime(self.config)
        self.release = TON()
        self.logical_inhibit = TP()
        self.logical_fall = Edge(falling=True)
        self.wind_rise = Edge()
        self.auto_gate = TON()
        self.start_cw = Edge()
        self.start_ccw = Edge()
        self.stop_cw = Edge(falling=True)
        self.stop_ccw = Edge(falling=True)
        self.automatic = self.logical_cw = self.logical_ccw = False
        self.previous_auto_selected = True
        self.scan_index = 0
        self.threshold = 0.0
        self.division_by_zero = False
        self.request_paths = ""
        self.events = []

    def _track(self, now, i):
        c, m = self.config, self.motor
        released = self.release(
            now, not (m.cw or m.ccw) and self.automatic, milliseconds(c.restart_seconds)
        )
        factor = 2 if i.brake_program_51 else 1
        lo, hi = c.reaction_low_deg * factor, c.reaction_high_deg * factor
        if c.wind_high_mps == c.wind_low_mps:
            self.threshold = hi
            self.division_by_zero = True
        else:
            a = lo + (hi - lo) * (i.wind_30s_mps - c.wind_low_mps) / (
                c.wind_high_mps - c.wind_low_mps
            )
            # Preserve XML MAX(low), MIN(high) order, even for decreasing values.
            self.threshold = min(hi, max(a, lo))
        a = self.threshold
        active = self.automatic and i.active_wind
        return_edge = self.wind_rise(i.active_wind)
        restart_ccw = (
            return_edge
            and self.automatic
            and released
            and i.cable_twist_deg >= 270
            and i.theta_30s_deg >= 90
        )
        restart_cw = (
            return_edge
            and self.automatic
            and released
            and i.cable_twist_deg <= -270
            and i.theta_30s_deg <= -90
        )
        cw_fast = active and released and i.theta_30s_deg > a
        ccw_fast = active and released and i.theta_30s_deg < -a
        standby_cw = (
            i.standby and i.start_misalignment and active and i.theta_30s_deg > 0
        )
        standby_ccw = (
            i.standby and i.start_misalignment and active and i.theta_30s_deg < 0
        )
        cw_fast = (cw_fast or standby_cw) and i.theta_1s_deg > a
        ccw_fast = (ccw_fast or standby_ccw) and i.theta_1s_deg < -a
        slow_allowed = active and released and not (self.logical_cw or self.logical_ccw)
        cw_slow = (
            slow_allowed
            and min(i.theta_5min_deg, i.theta_30s_deg, i.theta_1s_deg) > a / 2
        )
        ccw_slow = (
            slow_allowed
            and max(i.theta_5min_deg, i.theta_30s_deg, i.theta_1s_deg) < -a / 2
        )
        if restart_ccw:
            cw_fast = cw_slow = restart_cw = False
        if restart_cw:
            ccw_fast = ccw_slow = restart_ccw = False
        paths = []
        for name, value in [
            ("cw_fast", cw_fast),
            ("ccw_fast", ccw_fast),
            ("cw_slow", cw_slow),
            ("ccw_slow", ccw_slow),
            ("cw_wind_return", restart_cw),
            ("ccw_wind_return", restart_ccw),
        ]:
            if value:
                paths.append(name)
        if (standby_cw and cw_fast) or (standby_ccw and ccw_fast):
            paths.append("standby_override")
        self.request_paths = ";".join(paths)
        self.logical_inhibit(
            now, self.logical_fall(self.logical_cw or self.logical_ccw), 5000
        )
        ccw_cmd = (
            (ccw_fast or ccw_slow or restart_ccw)
            and self.automatic
            and not m.cw
            and not self.logical_cw
            and not self.logical_inhibit.q
        )
        self.logical_ccw = self.ccw_time.scan(
            now, ccw_cmd, i.theta_1s_deg, i.theta_30s_deg, m.cw or m.ccw, self.automatic
        )
        # CCW FB runs first; CW sees its newly assigned output.
        cw_cmd = (
            (cw_fast or cw_slow or restart_cw)
            and self.automatic
            and not m.ccw
            and not self.logical_ccw
            and not self.logical_inhibit.q
        )
        self.logical_cw = self.cw_time.scan(
            now, cw_cmd, i.theta_1s_deg, i.theta_30s_deg, m.cw or m.ccw, self.automatic
        )
        if i.nacelle_open_electrical:
            self.logical_cw = self.logical_ccw = False
        self.release(now)
        for direction, fb in [("cw", self.cw_time), ("ccw", self.ccw_time)]:
            if fb.accepted:
                self.events.append(f"logical_start_{direction}")
            if fb.stop_reason:
                self.events.append(f"{direction}_{fb.stop_reason}")

    def scan(self, inputs):
        if not isinstance(inputs, PLCInputs):
            raise TypeError("scan expects PLCInputs")
        now = self.scan_index * self.config.scan_ms
        self.events = []
        self._track(now, inputs)  # Uses previous scan's gbYawAutomaticMode.
        m = self.motor
        before = (m.cw, m.ccw, m.electrical_open, m.phase)
        if inputs.auto_mode and inputs.active_wind:
            # Other modes are represented by a stop-only selector. Entering from
            # it models gbInitYawMode; low-wind return while mode is unchanged
            # retains the auto TON, matching the assignment in PRG_YawControl.
            if not self.previous_auto_selected:
                self.start_cw(False)
                self.start_ccw(False)
                m.scan(now, inputs, self.automatic, cw=False, ccw=False, stop=True)
                self.auto_gate(now, False)
            gate = self.auto_gate(now, True, 10_000)
            m.scan(now, inputs, self.automatic)
            rising_cw = self.start_cw(self.logical_cw and gate)
            rising_ccw = self.start_ccw(self.logical_ccw and gate)
            falling_cw = self.stop_cw(self.logical_cw)
            falling_ccw = self.stop_ccw(self.logical_ccw)
            if rising_cw:
                m.scan(now, inputs, self.automatic, cw=True, ccw=False, stop=False)
            if rising_ccw:
                m.scan(now, inputs, self.automatic, cw=False, ccw=True, stop=False)
            if falling_cw or falling_ccw:
                m.scan(now, inputs, self.automatic, cw=False, ccw=False, stop=True)
            self.automatic = True
        else:
            m.scan(now, inputs, self.automatic, stop=True)
            self.automatic = False
        if not inputs.startup_ok or inputs.stop_request:
            m.scan(now, inputs, self.automatic, stop=True)
        self.previous_auto_selected = inputs.auto_mode
        if m.cw and m.ccw:
            raise RuntimeError(
                "Conflicting physical motor outputs; invalid reference trace"
            )
        if (m.cw or m.ccw) and not m.electrical_open:
            raise RuntimeError("Motor output without electrical brake release")
        if (m.cw, m.ccw, m.electrical_open, m.phase) != before:
            self.events.append("physical_" + m.phase)
        snapshot = self.snapshot(now)
        self.scan_index += 1
        return snapshot

    def snapshot(self, now=None):
        if now is None:
            now = self.scan_index * self.config.scan_ms
        m = self.motor
        result = {
            "time_s": now / 1000,
            "gbYawAutomaticMode": self.automatic,
            "gbStartYawMotorCW": self.logical_cw,
            "gbStartYawMotorCCW": self.logical_ccw,
            "DQ_NacelleStartYawCW": m.cw,
            "DQ_NacelleStartYawCCW": m.ccw,
            "DQ_NacelleOpenElectricalYawBrake": m.electrical_open,
            "DQ_NacelleValveOpenYawBrake1": m.valves[0],
            "DQ_NacelleValveOpenYawBrake2": m.valves[1],
            "DQ_NacelleValveOpenYawBrake3": m.valves[2],
            "physical_phase": m.phase,
            "logical_phase": self.cw_time.phase
            if self.logical_cw
            else self.ccw_time.phase
            if self.logical_ccw
            else "idle",
            "A_deg": self.threshold,
            "restart_release": self.release.q,
            "restart_release_et_s": self.release.et / 1000,
            "logical_inhibit_5s": self.logical_inhibit.q,
            "logical_inhibit_et_s": self.logical_inhibit.et / 1000,
            "physical_inhibit_22s": m.stop_tp.q,
            "physical_inhibit_et_s": m.stop_tp.et / 1000,
            "auto_gate_10s": self.auto_gate.q,
            "auto_gate_et_s": self.auto_gate.et / 1000,
            "pressure_target": m.pressure_target,
            "wind_stop_seconds": m.wind_stop_seconds,
            "stop_condition": m.stop_condition,
            "request_paths": self.request_paths,
            "division_by_zero_14": self.division_by_zero,
            "events": ";".join(self.events),
        }
        for direction, fb in [("cw", self.cw_time), ("ccw", self.ccw_time)]:
            result.update(
                {
                    f"{direction}_T_s": fb.t / 1000,
                    f"{direction}_H_s": fb.h / 1000,
                    f"{direction}_elapsed_s": fb.pulse.et / 1000,
                    f"{direction}_extension_elapsed_s": fb.absolute_stop.et / 1000,
                    f"{direction}_stored_sign": int(fb.positive) - int(fb.negative),
                    f"{direction}_accepted_T_s": fb.accepted_t / 1000,
                    f"{direction}_accepted_H_s": fb.accepted_h / 1000,
                }
            )
        return result
