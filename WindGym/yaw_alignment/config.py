"""Serializable settings and independent episode scenarios for the yaw lab."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class AlignmentConfig:
    decision_seconds: float = 10.0
    simulation_seconds: float = 1.0
    episode_seconds: float = 1800.0
    history_windows: int = 12
    yaw_step_deg: float = 1.0
    yaw_rate_deg_s: float = 0.3
    yaw_rest_seconds: float = 5.0
    travel_limit_deg: float = 180.0
    tolerance_deg: float = 2.0
    error_scale_deg: float = 10.0
    travel_penalty: float = 0.01
    start_penalty: float = 0.02
    wind_speed_min: float = 8.0
    wind_speed_max: float = 8.0
    initial_error_limit_deg: float = 15.0
    bias_limit_deg: float = 10.0
    encoder_bias_limit_deg: float = 0.0
    direction_noise_std_deg: float = 0.0
    encoder_noise_std_deg: float = 0.0
    speed_noise_std_mps: float = 0.0
    power_noise_std_fraction: float = 0.0
    wind_direction_amplitude_deg: float = 0.0
    wind_speed_amplitude_mps: float = 0.0
    wind_period_seconds: float = 300.0
    turbulence_intensity: float = 0.05
    wind_speed_scale_mps: float = 25.0

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{f.name} must be a finite number")
        positive = (
            "decision_seconds",
            "simulation_seconds",
            "episode_seconds",
            "yaw_step_deg",
            "yaw_rate_deg_s",
            "travel_limit_deg",
            "error_scale_deg",
            "wind_speed_min",
            "wind_speed_max",
            "wind_period_seconds",
            "wind_speed_scale_mps",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for f in fields(self):
            if getattr(self, f.name) < 0:
                raise ValueError(f"{f.name} must be nonnegative")
        if not isinstance(self.history_windows, int) or self.history_windows < 1:
            raise ValueError("history_windows must be a positive integer")
        for large, small in [
            (self.decision_seconds, self.simulation_seconds),
            (self.episode_seconds, self.decision_seconds),
        ]:
            if not math.isclose(large / small, round(large / small)) or large < small:
                raise ValueError(
                    "Decision and episode durations must contain whole simulation/decision steps"
                )
        if self.wind_speed_min > self.wind_speed_max:
            raise ValueError("wind_speed_min must not exceed wind_speed_max")
        if self.wind_speed_min - self.wind_speed_amplitude_mps <= 0:
            raise ValueError("Wind-speed variation must remain positive")
        if (
            self.yaw_step_deg >= 180
            or self.yaw_rate_deg_s * self.simulation_seconds >= 180
        ):
            raise ValueError(
                "A movement sample must be less than 180 degrees for encoder unwrapping"
            )
        if self.initial_error_limit_deg >= 180 or self.bias_limit_deg >= 180:
            raise ValueError(
                "Initial error and sensor bias limits must be below 180 degrees"
            )

    @property
    def substeps(self):
        return round(self.decision_seconds / self.simulation_seconds)

    @property
    def episode_steps(self):
        return round(self.episode_seconds / self.decision_seconds)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise TypeError("Config must be a YAML/JSON mapping")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown alignment settings: {sorted(unknown)}")
        return cls(**data)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


@dataclass(frozen=True)
class Scenario:
    wind_direction_deg: float
    wind_speed_mps: float
    initial_error_deg: float
    direction_bias_deg: float
    encoder_bias_deg: float = 0.0
    yaw_enabled: bool = True

    def __post_init__(self):
        for f in fields(self):
            if f.name != "yaw_enabled" and not math.isfinite(getattr(self, f.name)):
                raise ValueError(f"Scenario {f.name} must be finite")
        if self.wind_speed_mps <= 0 or abs(self.initial_error_deg) >= 180:
            raise ValueError(
                "Scenario requires positive wind speed and error below 180 degrees"
            )
        if not isinstance(self.yaw_enabled, bool):
            raise TypeError("yaw_enabled must be boolean")

    @classmethod
    def sample(cls, config, rng: np.random.Generator):
        return cls(
            wind_direction_deg=float(rng.uniform(0, 360)),
            wind_speed_mps=float(
                rng.uniform(config.wind_speed_min, config.wind_speed_max)
            ),
            initial_error_deg=float(
                rng.uniform(
                    -config.initial_error_limit_deg, config.initial_error_limit_deg
                )
            ),
            direction_bias_deg=float(
                rng.uniform(-config.bias_limit_deg, config.bias_limit_deg)
            ),
            encoder_bias_deg=float(
                rng.uniform(
                    -config.encoder_bias_limit_deg, config.encoder_bias_limit_deg
                )
            ),
        )

    def wind(self, time_seconds, config):
        # Prehistory is steady; variations start continuously at scored time zero.
        phase = 2 * np.pi * max(time_seconds, 0.0) / config.wind_period_seconds
        return (
            (
                self.wind_direction_deg
                + config.wind_direction_amplitude_deg * np.sin(phase)
            )
            % 360,
            self.wind_speed_mps + config.wind_speed_amplitude_mps * np.sin(phase),
        )
