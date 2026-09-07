"""Gymnasium experiment for alignment with biased sensors and physical yaw motion."""

from collections import deque
from dataclasses import asdict
from typing import ClassVar

import gymnasium as gym
import numpy as np

from .config import AlignmentConfig, Scenario
from .plant import PyWakeYawPlant, YawActuator
from .sensors import FEATURE_HIGH, FEATURE_LOW, FEATURE_NAMES, SensorModel


def alignment_reward(errors_deg, travel_deg, starts, config):
    loss = np.mean(
        (
            np.maximum(np.abs(errors_deg) - config.tolerance_deg, 0)
            / config.error_scale_deg
        )
        ** 2
    )
    terms = {
        "alignment": -float(loss),
        "travel": -config.travel_penalty * travel_deg / config.yaw_step_deg,
        "starts": -config.start_penalty * starts,
    }
    return float(sum(terms.values())), terms


class YawAlignmentEnv(gym.Env):
    """Single-turbine PyWake lab. Actions 0/1/2 mean CCW/hold/CW.

    Sensor history is the entire policy input. Simulator truth is returned only
    in info and used for the training reward, never added to observation features.
    This owns physical heading instead of using WindFarmEnv's wind-relative yaw
    controller. Both environments reuse the same PyWake physics adapter.
    """

    metadata: ClassVar[dict] = {"render_modes": []}
    feature_names = FEATURE_NAMES

    def __init__(self, config=None, *, turbine=None):
        super().__init__()
        self.config = config or AlignmentConfig()
        self.plant = PyWakeYawPlant(self.config, turbine)
        self.actuator = YawActuator(self.config)
        self.sensors = SensorModel(self.config, self.plant.rated_power_w)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            np.tile(FEATURE_LOW, self.config.history_windows),
            np.tile(FEATURE_HIGH, self.config.history_windows),
        )
        self.history = deque(maxlen=self.config.history_windows)
        self._ready = False

    def reset(self, *, seed=None, options=None):
        self._ready = False
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"scenario"}:
            raise ValueError("reset options only accepts 'scenario'")
        sampled = Scenario.sample(self.config, self.np_random)
        scenario = options.get("scenario", sampled)
        if isinstance(scenario, dict):
            scenario = Scenario(**scenario)
        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario or its dictionary")
        if scenario.wind_speed_mps <= self.config.wind_speed_amplitude_mps:
            raise ValueError("Scenario wind-speed trace must remain positive")
        self.scenario = scenario
        self.plant.reset()
        self.actuator.reset(
            scenario.wind_direction_deg - scenario.initial_error_deg,
            scenario.yaw_enabled,
        )
        sensor_seed = int(self.np_random.integers(0, 2**32))
        self.sensors.reset(
            scenario, np.random.default_rng(sensor_seed), self.actuator.heading_deg
        )
        self.time_seconds = -self.config.history_windows * self.config.decision_seconds
        self.history.clear()
        for _ in range(self.config.history_windows):
            frame, _, info = self._advance_interval(0)
            self.history.append(frame)
        self.time_seconds = 0.0
        self.elapsed_steps = 0
        self._totals = {
            k: 0.0
            for k in (
                "reward",
                "energy_wh",
                "yaw_travel_deg",
                "yaw_starts",
                "mean_abs_error_deg",
                "within_tolerance_fraction",
                "sensor_out_of_range_count",
                "observation_clip_count",
            )
        }
        self._ready = True
        info["scenario"] = asdict(scenario)
        info["scored_seconds"] = 0.0
        return self._observation(), info

    def _observation(self):
        observation = np.concatenate(self.history).astype(np.float32)
        if not np.isfinite(observation).all():
            raise FloatingPointError("Non-finite observation; episode is invalid")
        return observation

    def _advance_interval(self, direction):
        c = self.config
        self.actuator.command(direction)
        measured = []
        trace = []
        travel = 0.0
        starts = 0
        moving_seconds = 0.0
        for _ in range(c.substeps):
            movement = self.actuator.advance(c.simulation_seconds)
            self.time_seconds += c.simulation_seconds
            wd, ws = self.scenario.wind(self.time_seconds, c)
            error, speed, power = self.plant.sample(
                wd, ws, self.actuator.heading_deg, c.simulation_seconds
            )
            sensor_sample = self.sensors.sample(
                wd, self.actuator.heading_deg, speed, power
            )
            measured.append(sensor_sample)
            travel += movement.travel_deg
            starts += movement.starts
            moving_seconds += movement.moving_seconds
            trace.append(
                (
                    self.time_seconds,
                    wd,
                    self.actuator.heading_deg,
                    error,
                    power,
                    sensor_sample[0],
                    movement.displacement_deg,
                )
            )
        measured = np.asarray(measured)
        trace = np.asarray(trace)
        if not np.isfinite(measured).all() or not np.isfinite(trace).all():
            raise FloatingPointError(
                "Non-finite physical/sensor trace; episode is invalid"
            )
        reward, terms = alignment_reward(trace[:, 3], travel, starts, c)
        frame = self.sensors.window(
            measured, direction, self.actuator.rest_remaining, self.actuator.enabled
        )
        if not np.isfinite(frame).all():
            raise FloatingPointError("Non-finite sensor window; episode is invalid")
        clipped = np.clip(frame, FEATURE_LOW, FEATURE_HIGH)
        clip_count = int(np.count_nonzero(frame != clipped))
        frame = clipped
        outside = (
            (measured[:, 2] < 0)
            | (measured[:, 2] > c.wind_speed_scale_mps)
            | (measured[:, 3] < 0)
            | (measured[:, 3] > self.plant.rated_power_w)
        )
        info = {
            "true_error_deg": float(trace[-1, 3]),
            "measured_error_deg": float(measured[-1, 0]),
            "heading_deg": self.actuator.heading_deg % 360,
            "direction_bias_deg": self.scenario.direction_bias_deg,
            "requested_direction": direction,
            "applied_movement_deg": float(trace[:, 6].sum()),
            "yaw_travel_deg": travel,
            "yaw_starts": starts,
            "moving_seconds": moving_seconds,
            "energy_wh": float(trace[:, 4].sum() * c.simulation_seconds / 3600),
            "mean_abs_error_deg": float(np.abs(trace[:, 3]).mean()),
            "within_tolerance_fraction": float(
                (np.abs(trace[:, 3]) <= c.tolerance_deg).mean()
            ),
            "sensor_out_of_range_count": int(outside.sum()),
            "observation_clip_count": clip_count,
            "reward_terms": terms,
            "trace": {
                name: trace[:, i].copy()
                for i, name in enumerate(
                    (
                        "time_seconds",
                        "wind_direction_deg",
                        "heading_deg",
                        "true_error_deg",
                        "power_w",
                        "measured_error_deg",
                        "movement_deg",
                    )
                )
            },
        }
        return frame, reward, info

    def step(self, action):
        if not self._ready:
            raise gym.error.ResetNeeded(
                "Call reset() before stepping or after episode end"
            )
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be an integer in {{0,1,2}}, got {action!r}")
        try:
            frame, reward, info = self._advance_interval(int(action) - 1)
        except Exception:
            self._ready = False
            raise
        self.history.append(frame)
        self.elapsed_steps += 1
        for key in self._totals:
            self._totals[key] += reward if key == "reward" else info[key]
        metrics = self._totals.copy()
        for key in ("mean_abs_error_deg", "within_tolerance_fraction"):
            metrics[key] /= self.elapsed_steps
        info["episode_metrics"] = metrics
        info["scored_seconds"] = self.elapsed_steps * self.config.decision_seconds
        truncated = self.elapsed_steps >= self.config.episode_steps
        if truncated:
            self._ready = False
        return self._observation(), reward, False, truncated, info

    def close(self):
        self.plant.close()
        self._ready = False
