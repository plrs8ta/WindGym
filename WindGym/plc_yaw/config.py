"""Explicit example parameters; the supplied PLCopen snapshot has no GVL values."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class PLCYawConfig:
    scan_ms: int = 100
    decision_seconds: float = 1.0
    episode_seconds: float = 600.0
    yaw_speed_deg_min: float = 18.0
    reaction_low_deg: float = 8.0
    reaction_high_deg: float = 15.0
    wind_low_mps: float = 3.0
    wind_high_mps: float = 15.0
    restart_seconds: float = 30.0
    brake_open_seconds: float = 2.0
    motor_start_seconds: float = 1.0
    brake_close_seconds: float = 1.0
    stop_mode: int = 1
    wind_time_enabled: bool = True
    wind_time_constant: float = 5.0
    wind_time_min: float = 5.0
    wind_time_max: float = 13.0
    pressure_mode: int = 1
    pressure_low: float = 20.0
    pressure_high: float = 25.0
    pressure_wind_low: float = 5.0
    pressure_wind_high: float = 15.0
    pressure_safety_seconds: float = 15.0
    # Simulator-only settings, NOT values recovered from the PLC.
    plant_yaw_rate_deg_s: float = 0.3
    pressure_rise_per_second: float = 15.0
    pressure_release_per_second: float = 30.0
    pressure_ceiling: float = 120.0
    max_correction_deg: float = 10.0
    vane_noise_std_deg: float = 0.0
    travel_cost: float = 0.01
    start_cost: float = 0.02

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "wind_time_enabled":
                if not isinstance(value, bool):
                    raise ValueError("wind_time_enabled must be boolean")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{f.name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{f.name} must be finite and nonnegative")
        for name in ("scan_ms", "stop_mode", "pressure_mode"):
            if not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be an integer")
        for name in (
            "scan_ms",
            "decision_seconds",
            "episode_seconds",
            "yaw_speed_deg_min",
            "plant_yaw_rate_deg_s",
            "max_correction_deg",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.scan_ms > 1000 or 1000 % self.scan_ms:
            raise ValueError("scan_ms must divide one second (default: 100 ms)")
        for big, small in [
            (self.decision_seconds * 1000, self.scan_ms),
            (self.episode_seconds, self.decision_seconds),
        ]:
            if big < small or not math.isclose(big / small, round(big / small)):
                raise ValueError(
                    "Decision/episode lengths must contain whole scans/decisions"
                )
        if self.stop_mode not in (1, 2, 3) or self.pressure_mode not in (1, 2, 3, 4):
            raise ValueError("Supported stop modes: 1/2/3; pressure modes: 1/2/3/4")
        if self.wind_low_mps > self.wind_high_mps:
            raise ValueError("wind_low_mps must not exceed wind_high_mps")
        if self.pressure_wind_low >= self.pressure_wind_high:
            raise ValueError(
                "Pressure wind limits must increase; XML divides by their difference"
            )
        if self.pressure_low > self.pressure_high:
            raise ValueError("Pressure limits must increase")
        if self.wind_time_min > self.wind_time_max:
            raise ValueError("Wind-time limits must increase")
        # XML only processes a stop while its 22 s TP is active. Reject
        # configurations that cannot finish that sequence instead of fixing it.
        wind_max = self.wind_time_max
        if not self.wind_time_constant or not self.wind_time_min or not wind_max:
            wind_max = 13.0
        wind_max = math.floor(wind_max + 0.5)
        stop_time = {
            1: 1.0,
            2: wind_max if self.wind_time_enabled else 1.0,
            3: self.pressure_safety_seconds,
        }[self.stop_mode]
        if stop_time + self.brake_close_seconds + 2 * self.scan_ms / 1000 >= 22:
            raise ValueError(
                "Stop and brake-close sequence must finish inside the 22 s TP"
            )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise TypeError("Config must be a JSON object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**data)


@dataclass(frozen=True)
class PLCInputs:
    """Already averaged PLC signals. Replay feeds these directly, without filtering."""

    theta_1s_deg: float = 0.0
    theta_30s_deg: float = 0.0
    theta_5min_deg: float = 0.0
    wind_30s_mps: float = 8.0
    wind_5s_mps: float = 8.0
    pressure_1s: float = 0.0
    cable_twist_deg: float = 0.0
    auto_mode: bool = True
    active_wind: bool = True
    standby: bool = False
    start_misalignment: bool = False
    brake_program_51: bool = False
    nacelle_open_electrical: bool = False
    startup_ok: bool = True
    stop_request: bool = False
    slip_yaw: bool = False

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if f.type is bool:
                if not isinstance(v, bool):
                    raise ValueError(f"{f.name} must be boolean")
            elif (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
            ):
                raise ValueError(f"{f.name} must be finite numeric")
        if min(self.wind_30s_mps, self.wind_5s_mps, self.pressure_1s) < 0:
            raise ValueError("Wind speed and pressure must be nonnegative")
        if self.slip_yaw:
            raise ValueError("Slip-yaw control is outside the automatic-tracking scope")


SIGNAL_NAMES = {
    "grVaneDirection_1sec": "theta_1s_deg",
    "grVaneDirection_30sec": "theta_30s_deg",
    "grVaneDirection_5min": "theta_5min_deg",
    "grWindSpeed_30sec": "wind_30s_mps",
    "grWindSpeed_5sec": "wind_5s_mps",
    "grNacelleYawLoopPressure_1sec": "pressure_1s",
    "grCableTwistTotal": "cable_twist_deg",
    "automatic_mode_selected": "auto_mode",
    "OC_MeanWindSpeedAboveActivatingYaw": "active_wind",
    "standby_selected": "standby",
    "OC_YawMisalignmentForStartProcedure": "start_misalignment",
    "brake_program_at_least_51": "brake_program_51",
    "PAR_bNacelleOpenElectrical": "nacelle_open_electrical",
    "gbFirst30secAfterPLCStartOK": "startup_ok",
    "external_stop_request": "stop_request",
    "gbSlipYaw_flag": "slip_yaw",
}
