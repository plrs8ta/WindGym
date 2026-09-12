"""Document-derived boundary cases and scan timelines, not plant field validation."""

import csv
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from WindGym.plc_yaw import (
    PLCInputs,
    PLCYawConfig,
    PLCYawController,
    PLCYawEnv,
    PLCYawScenario,
)
from WindGym.plc_yaw.controller import MotorSequence, YawTime
from WindGym.plc_yaw.env import RollingMean
from WindGym.plc_yaw.replay import load_input_csv, replay
from WindGym.plc_yaw.timers import TON, TP, Edge


def run_scans(controller, seconds, inputs):
    rows = []
    for _ in range(round(seconds * 1000 / controller.config.scan_ms)):
        row = controller.scan(inputs)
        rows.append(row)
    return rows


def test_timers_repeat_calls_at_same_scan_and_hold_inputs():
    ton = TON()
    assert not ton(100, True, 1000)
    for _ in range(10):
        assert not ton(100)
        assert ton.et == 0
    assert not ton(1099)
    assert ton(1100)
    assert not ton(1100, False)
    assert ton.et == 0
    pulse = TP()
    assert pulse(0, True, 1000)
    assert pulse(200, False)
    assert pulse(300, True)  # Active pulse ignores another edge.
    assert not pulse(1000, True)
    assert not pulse(1500, True)  # Held high does not retrigger.
    assert not pulse(1500, False)
    assert pulse(1500, True)
    assert not pulse(1500, pt=0)  # Explicit live-PT reset assumption.
    fall = Edge(falling=True)
    assert not fall(False)
    assert not fall(True)
    assert fall(False)


def running_fb(theta=2, speed=60):
    fb = YawTime(PLCYawConfig(yaw_speed_deg_min=speed))
    fb.scan(0, False, theta, theta, False, True)  # Consume automatic-mode edge.
    assert fb.scan(100, True, theta, theta, False, True)
    return fb


@pytest.mark.parametrize("sign", [-1, 1])
def test_alignment_cannot_stop_before_half_time_but_equality_stops_at_half(sign):
    fb = running_fb(2 * sign)
    assert (fb.t, fb.h) == (2000, 1000)
    assert fb.scan(1099, True, sign, 100 * sign, False, True)
    assert not fb.scan(1100, True, sign, 100 * sign, False, True)
    assert fb.stop_reason == "supervised_alignment"
    assert fb.accepted_t == 2000  # No recalculation from the new theta30.
    assert not fb.scan(1200, True, sign, 100 * sign, False, True)
    assert not fb.accepted


@pytest.mark.parametrize("sign", [-1, 1])
def test_extension_equality_continues_strict_crossing_stops(sign):
    fb = running_fb(2 * sign)
    assert fb.scan(2000, True, 2 * sign, 2 * sign, False, True)
    assert fb.scan(2100, True, sign, 2 * sign, False, True)
    assert fb.phase == "extension"
    assert not fb.scan(2200, True, 0.999 * sign, 2 * sign, False, True)
    assert fb.stop_reason == "extension_alignment"


@pytest.mark.parametrize("sign", [-1, 1])
def test_absolute_timeout_wins_even_when_still_misaligned(sign):
    fb = running_fb(2 * sign)
    assert fb.scan(2100, True, 20 * sign, 2 * sign, False, True)
    assert fb.scan(3099, True, 20 * sign, 2 * sign, False, True)
    assert not fb.scan(3100, True, 20 * sign, 2 * sign, False, True)
    assert fb.stop_reason == "absolute_timeout"


def test_independent_time_caps_rounding_and_auto_edge_override():
    fb = running_fb(270)
    assert (fb.t, fb.h) == (180000, 90000)
    fb = running_fb(0.001, speed=40)
    assert (fb.t, fb.h) == (2, 1)  # Separate REAL-to-time rounding.
    fb = YawTime(PLCYawConfig(yaw_speed_deg_min=60))
    assert fb.scan(0, True, 2, 2, False, True)
    assert fb.t == 2000
    assert fb.h == 90000
    # Zero theta30 stores the negative sign even though T=0 stops immediately.
    fb = YawTime(PLCYawConfig())
    assert not fb.scan(0, True, 0, 0, False, True)


def test_physical_fall_resets_memory_without_forcing_same_scan_logical_output():
    fb = running_fb(20)
    assert fb.scan(200, False, 20, 20, True, True)
    assert fb.scan(300, False, 20, 20, False, True)
    assert fb.t == fb.h == 0
    assert not fb.positive and not fb.negative
    assert not fb.scan(400, False, 20, 20, False, True)


@pytest.mark.parametrize(
    "angle,direction,path",
    [(12, "cw", "fast"), (-12, "ccw", "fast"), (6, "cw", "slow"), (-6, "ccw", "slow")],
)
def test_fast_and_slow_start_paths(angle, direction, path):
    c = PLCYawController(
        PLCYawConfig(reaction_low_deg=10, reaction_high_deg=10, restart_seconds=0)
    )
    i = PLCInputs(theta_1s_deg=angle, theta_30s_deg=angle, theta_5min_deg=angle)
    c.scan(i)
    row = c.scan(i)
    assert row[f"gbStartYawMotor{direction.upper()}"]
    assert f"{direction}_{path}" in row["request_paths"]


@pytest.mark.parametrize(
    "theta1,theta30,theta5",
    [
        (10, 10, 0),
        (-10, -10, 0),
        (5, 6, 6),
        (6, 5, 6),
        (6, 6, 5),
        (-5, -6, -6),
        (12, -12, 0),
    ],
)
def test_equality_and_inconsistent_signs_do_not_start(theta1, theta30, theta5):
    c = PLCYawController(
        PLCYawConfig(reaction_low_deg=10, reaction_high_deg=10, restart_seconds=0)
    )
    i = PLCInputs(theta_1s_deg=theta1, theta_30s_deg=theta30, theta_5min_deg=theta5)
    rows = run_scans(c, 2, i)
    assert not any(r["gbStartYawMotorCW"] or r["gbStartYawMotorCCW"] for r in rows)


def test_threshold_interpolation_factor_and_literal_decreasing_clamp():
    cfg = PLCYawConfig(
        reaction_low_deg=10, reaction_high_deg=20, wind_low_mps=5, wind_high_mps=15
    )
    c = PLCYawController(cfg)
    assert c.scan(PLCInputs(wind_30s_mps=10))["A_deg"] == 15
    assert c.scan(PLCInputs(wind_30s_mps=10, brake_program_51=True))["A_deg"] == 30
    c = PLCYawController(replace(cfg, reaction_low_deg=20, reaction_high_deg=10))
    assert c.scan(PLCInputs(wind_30s_mps=5))["A_deg"] == 10  # Preserve source order.
    c = PLCYawController(replace(cfg, wind_low_mps=15))
    row = c.scan(PLCInputs())
    assert row["A_deg"] == 20 and row["division_by_zero_14"]


def test_standby_bypasses_restart_and_theta30_magnitude_not_five_second_inhibit():
    c = PLCYawController(
        PLCYawConfig(reaction_low_deg=10, reaction_high_deg=10, restart_seconds=100)
    )
    i = PLCInputs(
        theta_1s_deg=12, theta_30s_deg=1, standby=True, start_misalignment=True
    )
    c.scan(i)
    row = c.scan(i)
    assert not row["restart_release"] and row["gbStartYawMotorCW"]
    assert "standby_override" in row["request_paths"]
    assert row["cw_accepted_T_s"] == pytest.approx(3.333)
    # Physical gate has not opened; logical clock is already active.
    assert not row["DQ_NacelleStartYawCW"] and not row["auto_gate_10s"]


@pytest.mark.parametrize("direction", ["cw", "ccw"])
def test_opposite_physical_motor_and_five_second_inhibit_block_standby(direction):
    c = PLCYawController(PLCYawConfig(restart_seconds=100))
    c.automatic = True
    theta = 20 if direction == "cw" else -20
    i = PLCInputs(
        theta_1s_deg=theta, theta_30s_deg=theta, standby=True, start_misalignment=True
    )
    setattr(c.motor, "ccw" if direction == "cw" else "cw", True)
    c._track(0, i)
    assert not c.logical_cw and not c.logical_ccw
    c.motor.cw = c.motor.ccw = False
    c.logical_inhibit(100, True, 5000)
    c._track(100, i)
    assert not c.logical_cw and not c.logical_ccw
    c._track(5099, i)
    assert not c.logical_cw and not c.logical_ccw
    c._track(5100, i)
    assert getattr(c, "logical_" + direction)


@pytest.mark.parametrize("direction", ["cw", "ccw"])
def test_open_electrical_forces_global_output_off_without_resetting_yawtime(direction):
    c = PLCYawController(PLCYawConfig(restart_seconds=0))
    theta = 20 if direction == "cw" else -20
    i = PLCInputs(theta_1s_deg=theta, theta_30s_deg=theta)
    c.scan(i)
    c.scan(i)
    fb = getattr(c, direction + "_time")
    assert fb.output
    c.scan(replace(i, nacelle_open_electrical=True))
    assert not c.logical_cw and not c.logical_ccw
    assert fb.output and fb.t > 0


@pytest.mark.parametrize(
    "twist,theta,direction,sign", [(270, 90, "ccw", 1), (-270, -90, "cw", -1)]
)
def test_wind_return_direction_can_oppose_stored_vane_sign(
    twist, theta, direction, sign
):
    c = PLCYawController(PLCYawConfig(restart_seconds=0))
    # Isolate PRG_TrackingNacelle with its entering global context. The complete
    # external mode arbiter is absent; do not invent it to create this edge.
    c.automatic = True
    c._track(0, PLCInputs(active_wind=False))
    c._track(
        100, PLCInputs(theta_1s_deg=theta, theta_30s_deg=theta, cable_twist_deg=twist)
    )
    row = c.snapshot(100)
    assert row[f"gbStartYawMotor{direction.upper()}"]
    assert row[f"{direction}_stored_sign"] == sign
    assert f"{direction}_wind_return" in row["request_paths"]


@pytest.mark.parametrize("direction", ["cw", "ccw"])
def test_motor_brake_sequence_exact_times_and_pending_start_during_22s(direction):
    m = MotorSequence(PLCYawConfig())
    i = PLCInputs()
    m.scan(0, i, True, **{direction: True}, stop=False)
    assert m.valves == (True, False, True) and not m.electrical_open
    for _ in range(4):
        m.scan(0, i, True)
        assert m.open_delay.et == 0
    m.scan(2000, i, True)
    assert m.electrical_open and not (m.cw or m.ccw)
    m.scan(3000, i, True)
    assert getattr(m, direction)
    m.scan(4000, i, True, stop=True)
    assert getattr(m, direction) and m.valves == (False, False, False)
    m.scan(4999, i, True)
    assert getattr(m, direction)
    m.scan(5000, i, True)
    assert not (m.cw or m.ccw) and m.electrical_open
    m.scan(6000, i, True)
    assert not m.electrical_open and m.stop_finished
    m.scan(10_000, i, True, cw=True, ccw=False, stop=False)
    assert m.phase == "stopped" and m.stop_tp.q
    m.scan(25_999, i, True)
    assert m.phase == "stopped"
    m.scan(26_000, i, True)
    assert m.phase == "brake_opening"  # Pending input is not discarded.


def started_motor(config):
    m = MotorSequence(replace(config, brake_open_seconds=0, motor_start_seconds=0))
    m.scan(0, PLCInputs(), True, cw=True, stop=False)
    assert m.cw
    return m


def test_wind_time_stop_rounds_before_seconds_conversion():
    m = started_motor(PLCYawConfig(stop_mode=2))
    i = PLCInputs(wind_5s_mps=8.25)  # (8.25-5)*2 + 5 = 11.5 -> 12 s.
    m.scan(100, i, True, stop=True)
    assert m.wind_stop_seconds == 12
    m.scan(12_099, i, True)
    assert m.cw
    m.scan(12_100, i, True)
    assert not m.cw
    disabled = started_motor(PLCYawConfig(stop_mode=2, wind_time_enabled=False))
    disabled.scan(100, i, True, stop=True)
    disabled.scan(1100, i, True)
    assert not disabled.cw


def test_zero_wind_time_fallback_and_stop_budget_use_effective_parameters():
    m = started_motor(PLCYawConfig(stop_mode=2, wind_time_constant=0))
    m.scan(100, PLCInputs(wind_5s_mps=30), True, stop=True)
    assert m.wind_stop_seconds == 13
    with pytest.raises(ValueError, match="22 s"):
        PLCYawConfig(
            stop_mode=2,
            wind_time_constant=0,
            wind_time_max=0,
            wind_time_min=0,
            brake_close_seconds=10,
        )


def test_pressure_stop_strict_threshold_pulse_and_safety_timeout():
    cfg = PLCYawConfig(stop_mode=3, pressure_wind_low=5, pressure_wind_high=15)
    m = started_motor(cfg)
    equal = PLCInputs(wind_5s_mps=8, pressure_1s=55)
    m.scan(100, equal, True, stop=True)
    assert m.pressure_target == 55 and m.cw
    m.scan(200, replace(equal, pressure_1s=55.01), True)
    assert not m.cw and m.electrical_open and m.stop_condition == "pressure"
    m.scan(1200, replace(equal, pressure_1s=0), True)
    assert not m.electrical_open  # Latched pressure TP survives a falling pressure.
    m = started_motor(cfg)
    m.scan(100, equal, True, stop=True)
    m.scan(15_099, equal, True)
    assert m.cw
    m.scan(15_100, equal, True)
    assert not m.cw and m.stop_condition == "safety_timeout"


def test_auto_gate_delays_motor_not_logic_and_uses_gated_rising_edge():
    c = PLCYawController(
        PLCYawConfig(restart_seconds=0, brake_open_seconds=0, motor_start_seconds=0)
    )
    i = PLCInputs(theta_1s_deg=20, theta_30s_deg=20)
    rows = run_scans(c, 10.1, i)
    assert rows[1]["gbStartYawMotorCW"]
    assert all(not r["DQ_NacelleStartYawCW"] for r in rows[:100])
    assert rows[100]["DQ_NacelleStartYawCW"]


@pytest.mark.parametrize(
    "stop_signal",
    [
        "active_wind",
        "auto_mode",
        "startup_ok",
        "nacelle_open_electrical",
        "stop_request",
    ],
)
def test_each_documented_stop_request_reaches_physical_sequence(stop_signal):
    cfg = PLCYawConfig(restart_seconds=0, brake_open_seconds=0, motor_start_seconds=0)
    c = PLCYawController(cfg)
    i = PLCInputs(theta_1s_deg=20, theta_30s_deg=20)
    run_scans(c, 11, i)
    assert c.motor.cw
    value = stop_signal in {"stop_request", "nacelle_open_electrical"}
    stopped = replace(i, **{stop_signal: value})
    row = c.scan(stopped)
    assert row["physical_phase"] == "stop_pressure"
    assert row["DQ_NacelleStartYawCW"]  # Stop request is not instantaneous motor-off.
    rows = run_scans(c, 1.1, stopped)
    assert not rows[-1]["DQ_NacelleStartYawCW"]
    if stop_signal == "nacelle_open_electrical":
        assert not row["gbStartYawMotorCW"]
    elif stop_signal in {"startup_ok", "active_wind", "auto_mode", "stop_request"}:
        assert row["gbStartYawMotorCW"]  # YawTime is not directly reset by this check.


def test_stop_restart_timers_overlap_and_logical_inhibit_is_one_scan_later():
    cfg = PLCYawConfig(
        restart_seconds=2,
        brake_open_seconds=0,
        motor_start_seconds=0,
        brake_close_seconds=4,
    )
    c = PLCYawController(cfg)
    i = PLCInputs(theta_1s_deg=20, theta_30s_deg=20)
    run_scans(c, 40, i)  # H elapsed, physical motor running.
    aligned = replace(i, theta_1s_deg=0)
    rows = [c.scan(aligned)] + run_scans(c, 23, aligned)
    first = rows[0]
    assert not first["gbStartYawMotorCW"]
    assert first["physical_inhibit_22s"] and not first["logical_inhibit_5s"]
    assert rows[1]["logical_inhibit_5s"]
    released = next(r for r in rows if r["restart_release"])
    assert released["time_s"] == 43.1
    assert released["physical_phase"] == "brake_closing"
    assert released["physical_inhibit_22s"]


def test_circular_sensor_average_and_causal_action_history():
    window = RollingMean(2, 179, circular=True)
    assert abs(window.push(-179)) == pytest.approx(180)
    e = PLCYawEnv(PLCYawConfig(episode_seconds=2))
    try:
        e.reset(seed=1, options={"scenario": PLCYawScenario(initial_error_deg=10)})
        obs, _, _, _, info = e.step([-1.0])
        first, last = info["trace"][0], info["trace"][-1]
        assert first["grVaneDirection_1sec"] > 8.9
        assert abs(last["grVaneDirection_1sec"]) < 1e-10
        assert last["grVaneDirection_30sec"] > 9
        assert e.observation_space.contains(obs)
    finally:
        e.close()


def test_environment_check_seed_reset_bounds_and_finite_episode():
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check

    e = PLCYawEnv(PLCYawConfig(episode_seconds=3, vane_noise_std_deg=0.5))
    try:
        check_env(e, skip_render_check=True)
        sb3_check(e)
        a, _ = e.reset(seed=4)
        out_a, _, _, _, info_a = e.step([0.0])
        b, _ = e.reset(seed=4)
        out_b, _, _, _, info_b = e.step([0.0])
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(out_a, out_b)
        assert info_a["trace"] == info_b["trace"]
        assert a.shape == (31,)
        for bad in ([2], [np.nan], [0, 1], []):
            with pytest.raises(ValueError):
                e.step(bad)
        e.step([0.0])
        _, _, terminated, truncated, _ = e.step([0.0])
        assert not terminated and truncated
        with pytest.raises(gym.error.ResetNeeded):
            e.step([0.0])
    finally:
        e.close()


def test_error_truth_not_policy_input_and_zero_action_preserves_measurements():
    e = PLCYawEnv(PLCYawConfig(episode_seconds=2))
    try:
        obs_a, _ = e.reset(
            seed=1,
            options={
                "scenario": PLCYawScenario(initial_error_deg=10, vane_bias_deg=-10)
            },
        )
        a, reward_a, _, _, _ = e.step([0.0])
        obs_b, _ = e.reset(
            seed=1,
            options={"scenario": PLCYawScenario(initial_error_deg=0, vane_bias_deg=0)},
        )
        b, reward_b, _, _, _ = e.step([0.0])
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(a, b)
        assert reward_a < reward_b
    finally:
        e.close()


def test_closed_loop_movement_follows_actual_outputs_including_stop_delay():
    e = PLCYawEnv(PLCYawConfig(episode_seconds=100))
    try:
        e.reset(seed=1)
        rows = []
        for _ in range(100):
            _, _, _, _, info = e.step([0.0])
            rows.extend(info["trace"])
        commanded = [
            r for r in rows if r["gbStartYawMotorCW"] and not r["DQ_NacelleStartYawCW"]
        ]
        assert commanded and all(r["movement_deg"] == 0 for r in commanded)
        tail = [
            r for r in rows if not r["gbStartYawMotorCW"] and r["DQ_NacelleStartYawCW"]
        ]
        assert len(tail) == 10
        assert sum(r["movement_deg"] for r in tail) == pytest.approx(0.3)
        assert info["episode_metrics"]["motor_starts"] == 1
        assert info["episode_metrics"]["yaw_travel_deg"] == pytest.approx(
            sum(abs(r["movement_deg"]) for r in rows)
        )
        integral = sum(
            (abs(r["true_error_deg"]) + abs(r["end_true_error_deg"])) / 2 * 0.1
            for r in rows
        )
        assert info["episode_metrics"]["mean_abs_error_deg"] == pytest.approx(
            integral / 100
        )
    finally:
        e.close()


def test_events_apply_on_scan_boundary_and_episode_rejects_nonfinite_state():
    e = PLCYawEnv(PLCYawConfig(episode_seconds=2))
    try:
        e.reset(
            options={"events": [{"time_s": 0.5, "signals": {"active_wind": False}}]}
        )
        _, _, _, _, info = e.step([0.0])
        assert all(r["OC_MeanWindSpeedAboveActivatingYaw"] for r in info["trace"][:5])
        assert not any(
            r["OC_MeanWindSpeedAboveActivatingYaw"] for r in info["trace"][5:]
        )
        e.heading = float("nan")
        with pytest.raises(ValueError):
            e.step([0.0])
        with pytest.raises(gym.error.ResetNeeded):
            e.step([0.0])
        with pytest.raises(ValueError):
            e.reset(
                options={
                    "events": [{"time_s": 0.05, "signals": {"active_wind": False}}]
                }
            )
    finally:
        e.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scan_ms": 0},
        {"scan_ms": 30},
        {"yaw_speed_deg_min": 0},
        {"pressure_wind_high": 5},
        {"stop_mode": 0},
        {"pressure_mode": 0},
        {"wind_time_min": 15},
        {"episode_seconds": 1.5},
        {"brake_close_seconds": 21},
        {"vane_noise_std_deg": float("inf")},
    ],
)
def test_invalid_and_unsupported_parameters_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        PLCYawConfig(**kwargs)


def test_slip_input_and_unknown_configuration_rejected(tmp_path):
    with pytest.raises(ValueError, match="Slip"):
        PLCInputs(slip_yaw=True)
    path = tmp_path / "config.json"
    path.write_text('{"typo": 2}')
    with pytest.raises(ValueError, match="Unknown"):
        PLCYawConfig.load(path)


def test_replay_csv_validates_and_preserves_supplied_averages(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text(
        "time_s,grVaneDirection_1sec,grVaneDirection_30sec,grVaneDirection_5min,grWindSpeed_30sec,grWindSpeed_5sec,grNacelleYawLoopPressure_1sec,grCableTwistTotal\n"
        "0,20,12,6,8,8,90,0\n1,0,,,,,,\n"
    )
    cfg = PLCYawConfig(restart_seconds=0)
    events = load_input_csv(path, cfg)
    assert events[1][1].theta_1s_deg == 0 and events[1][1].theta_30s_deg == 12
    out = replay(path, tmp_path / "replay", cfg)
    with (out / "trace.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 11
    assert all(float(r["grVaneDirection_30sec"]) == 12 for r in rows)
    assert (out / "execution.svg").stat().st_size > 1000
    with pytest.raises(FileExistsError):
        replay(path, out, cfg)
    path.write_text(path.read_text().replace("1,0,", "0,0,"))
    with pytest.raises(ValueError, match="strictly increasing"):
        load_input_csv(path, cfg)


def test_sb3_can_train_on_new_environment_without_contactor_actions():
    from stable_baselines3 import PPO

    e = PLCYawEnv(PLCYawConfig(episode_seconds=5))
    try:
        model = PPO(
            "MlpPolicy", e, n_steps=8, batch_size=8, n_epochs=1, seed=3, device="cpu"
        )
        model.learn(16)
        obs, _ = e.reset(seed=3)
        action, _ = model.predict(obs, deterministic=True)
        assert e.action_space.contains(action)
        _, _, _, _, info = e.step(action)
        assert not any(
            r["DQ_NacelleStartYawCW"] or r["DQ_NacelleStartYawCCW"]
            for r in info["trace"]
        )
    finally:
        e.close()
