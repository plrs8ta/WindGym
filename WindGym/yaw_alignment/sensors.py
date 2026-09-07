"""Causal sensor samples; hidden truth is never part of the policy features."""

import numpy as np

from .plant import wrap180

FEATURE_NAMES = (
    "error_sin",
    "error_cos",
    "heading_sin",
    "heading_cos",
    "wind_speed",
    "power",
    "encoder_displacement",
    "previous_direction",
    "inhibit_time",
    "yaw_enabled",
)
# Explicit input saturation: speed/power can exceed nominal scales, and noisy
# encoder increments can exceed the physical request. Raw data remain in logs.
FEATURE_LOW = np.array([-1, -1, -1, -1, -2, -2, -3, -1, 0, 0], dtype=np.float32)
FEATURE_HIGH = np.array([1, 1, 1, 1, 2, 2, 3, 1, 1, 1], dtype=np.float32)


def circular_mean_deg(values):
    radians = np.deg2rad(values)
    return float(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())))


class SensorModel:
    def __init__(self, config, rated_power_w):
        self.config = config
        self.rated_power_w = rated_power_w

    def reset(self, scenario, rng, heading_deg):
        self.scenario = scenario
        self.rng = rng
        # Consume the same innovations regardless of applied controller actions.
        self.previous_encoder = self._encoder(heading_deg, rng.normal())

    def _encoder(self, heading, innovation):
        return (
            heading
            + self.scenario.encoder_bias_deg
            + self.config.encoder_noise_std_deg * innovation
        ) % 360

    def sample(self, true_direction, heading, speed, power):
        c = self.config
        z = self.rng.normal(size=4)
        direction = (
            true_direction
            + self.scenario.direction_bias_deg
            + c.direction_noise_std_deg * z[0]
        )
        encoder = self._encoder(heading, z[1])
        displacement = float(wrap180(encoder - self.previous_encoder))
        self.previous_encoder = encoder
        return np.array(
            [
                float(wrap180(direction - encoder)),
                encoder,
                speed + c.speed_noise_std_mps * z[2],
                power + c.power_noise_std_fraction * self.rated_power_w * z[3],
                displacement,
            ]
        )

    def window(self, samples, direction, rest_remaining, enabled):
        c = self.config
        error = np.deg2rad(circular_mean_deg(samples[:, 0]))
        heading = np.deg2rad(circular_mean_deg(samples[:, 1]))
        return np.array(
            [
                np.sin(error),
                np.cos(error),
                np.sin(heading),
                np.cos(heading),
                samples[:, 2].mean() / c.wind_speed_scale_mps,
                samples[:, 3].mean() / self.rated_power_w,
                samples[:, 4].sum() / c.yaw_step_deg,
                direction,
                rest_remaining / max(c.yaw_rest_seconds, c.simulation_seconds),
                float(enabled),
            ],
            dtype=np.float32,
        )
