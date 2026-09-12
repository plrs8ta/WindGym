"""Gymnasium wrapper around the PLC reference controller and a kinematic plant."""

import math
from collections import deque
from dataclasses import asdict, dataclass, fields, replace
from typing import ClassVar

import gymnasium as gym
import numpy as np

from .config import SIGNAL_NAMES, PLCInputs, PLCYawConfig
from .controller import PLCYawController


def wrap180(value):
    return (value + 180) % 360 - 180


class RollingMean:
    """Causal fixed windows, prefilled with a stationary prehistory at reset."""

    def __init__(self, count, initial, circular=False):
        self.circular = circular
        value = self._encode(initial)
        self.samples = deque([value] * count, maxlen=count)
        self.total = value * count

    def _encode(self, value):
        return (
            complex(math.cos(math.radians(value)), math.sin(math.radians(value)))
            if self.circular
            else float(value)
        )

    def push(self, value):
        value = self._encode(value)
        self.total += value - self.samples[0]
        self.samples.append(value)
        return self.mean

    @property
    def mean(self):
        if self.circular:
            return math.degrees(math.atan2(self.total.imag, self.total.real))
        return self.total / len(self.samples)


@dataclass(frozen=True)
class PLCYawScenario:
    initial_error_deg: float = 18.0
    wind_direction_deg: float = 270.0
    wind_speed_mps: float = 8.0
    vane_bias_deg: float = 0.0
    initial_cable_twist_deg: float = 0.0
    wind_amplitude_deg: float = 0.0
    wind_period_seconds: float = 120.0
    initial_pressure: float = 90.0
    startup_protection: bool = True

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type is bool:
                if not isinstance(value, bool):
                    raise ValueError(f"{f.name} must be boolean")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{f.name} must be finite numeric")
        if (
            self.wind_speed_mps < 0
            or self.initial_pressure < 0
            or self.wind_period_seconds <= 0
        ):
            raise ValueError("Invalid wind speed, pressure or wind period")
        if abs(self.initial_error_deg) > 180:
            raise ValueError("Initial error must be within +/-180 degrees")


# Engineering scaling only; controller memory and hidden truth do not enter obs.
FEATURE_NAMES = (
    "sin_theta1",
    "cos_theta1",
    "sin_theta30",
    "cos_theta30",
    "sin_theta5min",
    "cos_theta5min",
    "wind30_div25",
    "wind5_div25",
    "pressure_div150",
    "cable_twist_div720",
    "logical_cw",
    "logical_ccw",
    "motor_cw",
    "motor_ccw",
    "electrical_brake_open",
    "valve1",
    "valve2",
    "valve3",
    "restart_release",
    "logical_inhibit",
    "physical_inhibit",
    "auto_gate",
    "automatic",
    "active_wind",
    "startup_ok",
    "nacelle_open_electrical",
    "standby",
    "start_misalignment",
    "brake_program_51",
    "external_stop",
    "previous_correction_divmax",
)


class PLCYawEnv(gym.Env):
    """Zero action reproduces reference tracking with simulated measurements.

    action[0] in [-1,1] adds a bounded vane correction BEFORE the 1s/30s/5min
    averages. It never commands a contactor or bypasses the PLC execution layer.
    Reward is an experimental cost, not a value present in the PLC source.
    Use PLCYawController/CSV replay for externally averaged real PLC signals.
    """

    metadata: ClassVar[dict] = {"render_modes": []}
    feature_names = FEATURE_NAMES

    def __init__(self, config=None):
        super().__init__()
        self.config = config or PLCYawConfig()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -2.0, 2.0, shape=(len(FEATURE_NAMES),), dtype=np.float32
        )
        self._ready = False

    def reset(self, *, seed=None, options=None):
        self._ready = False
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"scenario", "events"}:
            raise ValueError("reset options accepts scenario and events only")
        self.scenario = options.get("scenario", PLCYawScenario())
        if isinstance(self.scenario, dict):
            self.scenario = PLCYawScenario(**self.scenario)
        if not isinstance(self.scenario, PLCYawScenario):
            raise TypeError("scenario must be PLCYawScenario or a dictionary")
        self.controller = PLCYawController(self.config)
        s, c = self.scenario, self.config
        self.heading = s.wind_direction_deg - s.initial_error_deg
        self.cable = s.initial_cable_twist_deg
        self.pressure = s.initial_pressure
        initial = wrap180(s.initial_error_deg + s.vane_bias_deg)
        self.vane_windows = [
            RollingMean(round(t * 1000 / c.scan_ms), initial, True)
            for t in (1, 30, 300)
        ]
        self.wind_windows = [
            RollingMean(round(t * 1000 / c.scan_ms), s.wind_speed_mps) for t in (5, 30)
        ]
        self.pressure_window = RollingMean(1000 // c.scan_ms, self.pressure)
        self._signals = {}
        self._events = []
        allowed = {f.name for f in fields(PLCInputs)} - {
            "theta_1s_deg",
            "theta_30s_deg",
            "theta_5min_deg",
            "wind_30s_mps",
            "wind_5s_mps",
            "pressure_1s",
            "cable_twist_deg",
        }
        for event in options.get("events", []):
            if not isinstance(event, dict) or set(event) != {"time_s", "signals"}:
                raise ValueError("Each event requires time_s and signals")
            time = event["time_s"]
            if (
                isinstance(time, bool)
                or not isinstance(time, (int, float))
                or not math.isfinite(time)
                or time < 0
                or not math.isclose(
                    time * 1000 / c.scan_ms, round(time * 1000 / c.scan_ms)
                )
            ):
                raise ValueError("Event time must be a nonnegative scan boundary")
            values = event["signals"]
            if not isinstance(values, dict) or set(values) - allowed:
                raise ValueError("Events may change discrete PLC input signals only")
            PLCInputs(**values)
            self._events.append((round(time * 1000), dict(values)))
        self._events.sort(key=lambda item: item[0])
        self._event_index = 0
        self.elapsed_steps = 0
        self.previous_correction = 0.0
        self._totals = {
            "reward": 0.0,
            "yaw_travel_deg": 0.0,
            "motor_starts": 0,
            "error_integral_deg_s": 0.0,
            "moving_seconds": 0.0,
        }
        self._apply_events(0)
        self.last_inputs = self._inputs(0)
        self.last_snapshot = self.controller.snapshot(0)
        self._ready = True
        obs, clipped = self._observation()
        return obs, {
            "scenario": asdict(s),
            "config": c.to_dict(),
            "plc": self.last_snapshot.copy(),
            "observation_clip_count": clipped,
            "plant_model": "ideal yaw-rate and synthetic pressure; no aerodynamic model",
        }

    def _apply_events(self, now_ms):
        while (
            self._event_index < len(self._events)
            and self._events[self._event_index][0] <= now_ms
        ):
            self._signals.update(self._events[self._event_index][1])
            self._event_index += 1

    def _inputs(self, now_s):
        values = {
            "theta_1s_deg": self.vane_windows[0].mean,
            "theta_30s_deg": self.vane_windows[1].mean,
            "theta_5min_deg": self.vane_windows[2].mean,
            "wind_30s_mps": self.wind_windows[1].mean,
            "wind_5s_mps": self.wind_windows[0].mean,
            "pressure_1s": self.pressure_window.mean,
            "cable_twist_deg": self.cable,
            "startup_ok": not self.scenario.startup_protection or now_s >= 30,
        }
        values.update(self._signals)
        return PLCInputs(**values)

    def _observation(self):
        i, p = self.last_inputs, self.last_snapshot
        values = []
        for theta in (i.theta_1s_deg, i.theta_30s_deg, i.theta_5min_deg):
            values += [math.sin(math.radians(theta)), math.cos(math.radians(theta))]
        values += [
            i.wind_30s_mps / 25,
            i.wind_5s_mps / 25,
            i.pressure_1s / 150,
            i.cable_twist_deg / 720,
        ]
        values += [
            p[name]
            for name in (
                "gbStartYawMotorCW",
                "gbStartYawMotorCCW",
                "DQ_NacelleStartYawCW",
                "DQ_NacelleStartYawCCW",
                "DQ_NacelleOpenElectricalYawBrake",
                "DQ_NacelleValveOpenYawBrake1",
                "DQ_NacelleValveOpenYawBrake2",
                "DQ_NacelleValveOpenYawBrake3",
                "restart_release",
                "logical_inhibit_5s",
                "physical_inhibit_22s",
                "auto_gate_10s",
            )
        ]
        values += [
            i.auto_mode,
            i.active_wind,
            i.startup_ok,
            i.nacelle_open_electrical,
            i.standby,
            i.start_misalignment,
            i.brake_program_51,
            i.stop_request,
            self.previous_correction / self.config.max_correction_deg,
        ]
        raw = np.asarray(values, dtype=np.float32)
        if not np.isfinite(raw).all():
            raise FloatingPointError("Non-finite observation")
        clipped = np.clip(raw, -2, 2)
        return clipped, int(np.count_nonzero(raw != clipped))

    def step(self, action):
        if not self._ready:
            raise gym.error.ResetNeeded(
                "Call reset before stepping or after episode end"
            )
        action = np.asarray(action)
        if (
            action.shape != (1,)
            or not np.isfinite(action).all()
            or np.any(np.abs(action) > 1)
        ):
            raise ValueError("Action must contain one finite correction in [-1,1]")
        self.previous_correction = float(action[0]) * self.config.max_correction_deg
        try:
            return self._advance()
        except Exception:
            self._ready = False
            raise

    def _advance(self):
        c, s = self.config, self.scenario
        dt = c.scan_ms / 1000
        trace = []
        travel = error_integral = moving_seconds = 0.0
        starts = 0
        for _ in range(round(c.decision_seconds / dt)):
            now_ms = self.controller.scan_index * c.scan_ms
            now = now_ms / 1000
            self._apply_events(now_ms)
            wind = s.wind_direction_deg + s.wind_amplitude_deg * math.sin(
                2 * math.pi * now / s.wind_period_seconds
            )
            error = wrap180(wind - self.heading)
            measured = wrap180(
                error + s.vane_bias_deg + self.np_random.normal(0, c.vane_noise_std_deg)
            )
            for window in self.vane_windows:
                window.push(wrap180(measured + self.previous_correction))
            for window in self.wind_windows:
                window.push(s.wind_speed_mps)
            self.pressure_window.push(self.pressure)
            i = self._inputs(now)
            was_running = self.controller.motor.cw or self.controller.motor.ccw
            p = self.controller.scan(i)
            direction = int(p["DQ_NacelleStartYawCW"]) - int(p["DQ_NacelleStartYawCCW"])
            displacement = direction * c.plant_yaw_rate_deg_s * dt
            started = int(bool(direction) and not was_running)
            # Outputs from this scan are held through [time_s, interval_end_s).
            end_heading = self.heading + displacement
            end_wind = s.wind_direction_deg + s.wind_amplitude_deg * math.sin(
                2 * math.pi * (now + dt) / s.wind_period_seconds
            )
            end_error = wrap180(end_wind - end_heading)
            # Trapezoidal integration includes ongoing motion during stop pressure.
            error_integral += (abs(error) + abs(end_error)) / 2 * dt
            travel += abs(displacement)
            moving_seconds += dt if direction else 0
            starts += started
            row = dict(p)
            row.update({plc: getattr(i, field) for plc, field in SIGNAL_NAMES.items()})
            row.update(
                interval_end_s=round(now + dt, 6),
                wind_direction_deg=wind % 360,
                heading_deg=self.heading % 360,
                end_heading_deg=end_heading % 360,
                true_error_deg=error,
                end_true_error_deg=end_error,
                raw_vane_deg=measured,
                correction_deg=self.previous_correction,
                movement_deg=displacement,
                motor_start=started,
                pressure=self.pressure,
            )
            trace.append(row)
            self.heading = end_heading
            self.cable += displacement
            if p["DQ_NacelleValveOpenYawBrake1"]:
                self.pressure = max(
                    0, self.pressure - c.pressure_release_per_second * dt
                )
            else:
                self.pressure = min(
                    c.pressure_ceiling, self.pressure + c.pressure_rise_per_second * dt
                )
            self.last_inputs, self.last_snapshot = i, p
        mean_error = error_integral / c.decision_seconds
        terms = {
            "alignment": -error_integral / 10,
            "travel": -c.travel_cost * travel,
            "starts": -c.start_cost * starts,
        }
        reward = sum(terms.values())
        for key, value in {
            "reward": reward,
            "yaw_travel_deg": travel,
            "motor_starts": starts,
            "error_integral_deg_s": error_integral,
            "moving_seconds": moving_seconds,
        }.items():
            self._totals[key] += value
        self.elapsed_steps += 1
        elapsed = self.elapsed_steps * c.decision_seconds
        truncated = self.elapsed_steps >= round(c.episode_seconds / c.decision_seconds)
        if truncated:
            self._ready = False
        obs, clipped = self._observation()
        metrics = dict(
            self._totals,
            mean_abs_error_deg=self._totals["error_integral_deg_s"] / elapsed,
        )
        info = {
            "plc": self.last_snapshot.copy(),
            "trace": trace,
            "reward_terms": terms,
            "mean_abs_error_deg": mean_error,
            "yaw_travel_deg": travel,
            "motor_starts": starts,
            "elapsed_seconds": elapsed,
            "episode_metrics": metrics,
            "observation_clip_count": clipped,
        }
        return obs, float(reward), False, truncated, info

    def close(self):
        self._ready = False
