"""Baselines that receive the same observations as PPO, with no simulator access."""

import numpy as np

from .plant import wrap180


class HoldController:
    def reset(self):
        pass

    def predict(self, observation, deterministic=True):
        return 1, None


class FilteredYawController:
    def __init__(
        self, decision_seconds=10, filter_seconds=20, deadband_deg=2, calibration_deg=0
    ):
        if not np.isfinite(
            [decision_seconds, filter_seconds, deadband_deg, calibration_deg]
        ).all():
            raise ValueError("Controller settings must be finite")
        if filter_seconds < 0 or deadband_deg < 0 or decision_seconds <= 0:
            raise ValueError("Invalid controller filter, deadband or timing")
        self.alpha = (
            1.0
            if filter_seconds == 0
            else 1 - np.exp(-decision_seconds / filter_seconds)
        )
        self.deadband_deg = deadband_deg
        # Fixed externally supplied calibration, never the hidden episode bias.
        self.calibration_deg = calibration_deg
        self.reset()

    def reset(self):
        self.filtered_error = None

    def predict(self, observation, deterministic=True):
        frame = np.asarray(observation).reshape(-1, 10)[-1]
        error = float(
            wrap180(np.rad2deg(np.arctan2(frame[0], frame[1])) - self.calibration_deg)
        )
        if self.filtered_error is None:
            self.filtered_error = error
        else:
            self.filtered_error = float(
                wrap180(
                    self.filtered_error
                    + self.alpha * wrap180(error - self.filtered_error)
                )
            )
        direction = (
            int(np.sign(self.filtered_error))
            if abs(self.filtered_error) > self.deadband_deg
            else 0
        )
        if not frame[9]:
            direction = 0
        return direction + 1, None
