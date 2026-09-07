"""Physical heading and yaw mechanics, separate from the RL interface."""

from dataclasses import dataclass

import numpy as np

from WindGym.backend.pywake_adapter import PyWakeFlowSimulationAdapter


def wrap180(angle):
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Movement:
    displacement_deg: float
    travel_deg: float
    starts: int
    moving_seconds: float


class YawActuator:
    """Rate-limited unwrapped physical bearing with stops, rest and travel limits."""

    def __init__(self, config):
        self.config = config
        self.reset(0.0)

    def reset(self, heading_deg, enabled=True):
        self.heading_deg = float(heading_deg)
        self.home_deg = self.heading_deg
        self.target_deg = self.heading_deg
        self.enabled = enabled
        self.moving_direction = 0
        self.rest_remaining = 0.0

    def command(self, direction):
        c = self.config
        self.target_deg = float(
            np.clip(
                self.heading_deg + direction * c.yaw_step_deg,
                self.home_deg - c.travel_limit_deg,
                self.home_deg + c.travel_limit_deg,
            )
        )

    def _stop(self):
        if self.moving_direction:
            self.rest_remaining = self.config.yaw_rest_seconds
        self.moving_direction = 0

    def advance(self, dt):
        distance = self.target_deg - self.heading_deg
        direction = int(np.sign(distance)) if abs(distance) > 1e-10 else 0
        if not self.enabled or not direction:
            self._stop()
            self.rest_remaining = max(0.0, self.rest_remaining - dt)
            return Movement(0.0, 0.0, 0, 0.0)
        if self.moving_direction and self.moving_direction != direction:
            self._stop()
        wait = min(dt, self.rest_remaining)
        self.rest_remaining -= wait
        available = dt - wait
        if available <= 1e-12:
            return Movement(0.0, 0.0, 0, 0.0)
        starts = int(self.moving_direction == 0)
        self.moving_direction = direction
        travel = min(abs(distance), self.config.yaw_rate_deg_s * available)
        move_time = travel / self.config.yaw_rate_deg_s
        displacement = direction * travel
        self.heading_deg += displacement
        if abs(self.target_deg - self.heading_deg) < 1e-10:
            self._stop()
            self.rest_remaining = max(
                0.0, self.rest_remaining - (available - move_time)
            )
        return Movement(displacement, travel, starts, move_time)


class PyWakeYawPlant:
    """One turbine using WindGym's actual PyWake backend, with an explicit bearing.

    PyWake's plotted rotor orientation is 90-wd+yaw. Therefore yaw=wd-heading
    when wind direction and nacelle heading use clockwise compass bearings.
    Wind is used here only to express physical orientation in backend coordinates.
    """

    def __init__(self, config, turbine=None):
        if turbine is None:
            from py_wake.examples.data.hornsrev1 import V80

            turbine = V80()
        self.turbine = turbine
        self.rated_power_w = float(np.max(turbine.power(np.linspace(0, 30, 301))))
        if not np.isfinite(self.rated_power_w) or self.rated_power_w <= 0:
            raise ValueError("Turbine must have positive finite rated power")
        self.flow = PyWakeFlowSimulationAdapter(
            x=[0],
            y=[0],
            windTurbine=turbine,
            ws=8,
            wd=270,
            ti=config.turbulence_intensity,
            dt=config.simulation_seconds,
        )
        self._last_conditions = None

    def reset(self):
        self.flow.time = 0.0
        self._last_conditions = None

    def sample(self, wind_direction_deg, wind_speed_mps, heading_deg, dt):
        error = float(wrap180(wind_direction_deg - heading_deg))
        conditions = (float(wind_direction_deg), float(wind_speed_mps), error)
        self.flow.wd = self.flow.wind_direction = float(wind_direction_deg)
        self.flow.ws = float(wind_speed_mps)
        self.flow.windTurbines.yaw = np.array([error])
        # A steady-state hold has the same flow solution; avoid solving it again.
        if conditions != self._last_conditions:
            self.flow.run(dt)
            self._last_conditions = conditions
        else:
            self.flow.time += dt
        power = float(self.flow.windTurbines.power()[0])
        speed = float(self.flow.windTurbines.rotor_avg_windspeed[0, 0])
        if not np.isfinite([power, speed, error]).all():
            raise FloatingPointError("Non-finite PyWake sample; episode is invalid")
        return error, speed, power

    def close(self):
        # The steady-state adapter owns no subprocesses or external resources.
        pass
