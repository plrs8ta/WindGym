"""Control invariants, information boundaries, and paired evaluation for the lab."""

import csv
import json
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from WindGym.yaw_alignment import (
    AlignmentConfig,
    FilteredYawController,
    Scenario,
    YawAlignmentEnv,
)
from WindGym.yaw_alignment.env import alignment_reward
from WindGym.yaw_alignment.experiment import summarize
from WindGym.yaw_alignment.plant import YawActuator, wrap180


@pytest.fixture
def env_factory():
    envs = []

    def make(**kwargs):
        env = YawAlignmentEnv(
            AlignmentConfig(episode_seconds=30, history_windows=2, **kwargs)
        )
        envs.append(env)
        return env

    yield make
    for env in envs:
        env.close()


def test_default_interface_and_seed_replay():
    env = YawAlignmentEnv(AlignmentConfig(episode_seconds=30))
    try:
        from gymnasium.utils.env_checker import check_env
        from stable_baselines3.common.env_checker import check_env as sb3_check

        check_env(env, skip_render_check=True)
        sb3_check(env)
        obs, _ = env.reset(seed=12)
        assert obs.shape == (120,)
        assert env.observation_space.contains(obs)
    finally:
        env.close()


@pytest.mark.parametrize("error,action,expected", [(10, 2, 9), (-10, 0, -9)])
def test_turn_sign_and_real_pywake_power_improvement(
    env_factory, error, action, expected
):
    env = env_factory()
    _, before = env.reset(seed=2, options={"scenario": Scenario(270, 8, error, 0)})
    _, _, _, _, after = env.step(action)
    assert after["true_error_deg"] == pytest.approx(expected)
    assert after["trace"]["power_w"][-1] > before["trace"]["power_w"][-1]
    assert after["yaw_travel_deg"] == pytest.approx(1)
    assert after["moving_seconds"] == pytest.approx(1 / 0.3)
    assert after["yaw_starts"] == 1
    assert np.abs(after["trace"]["movement_deg"]).max() <= 0.3 + 1e-10


def test_hold_preserves_heading_through_north_and_wind_shift(env_factory):
    env = env_factory(wind_direction_amplitude_deg=5, wind_period_seconds=40)
    _, initial = env.reset(seed=2, options={"scenario": Scenario(359, 8, 0, 0)})
    _, _, _, _, info = env.step(1)
    assert initial["heading_deg"] == 359
    assert info["heading_deg"] == 359
    assert info["true_error_deg"] == pytest.approx(5)
    assert np.all(info["trace"]["heading_deg"] == 359)
    assert info["yaw_travel_deg"] == 0
    assert float(wrap180(1 - 359)) == 2


def test_reward_and_energy_cover_movement_and_hold(env_factory):
    env = env_factory()
    env.reset(seed=2, options={"scenario": Scenario(270, 8, 10, 0)})
    _, reward, _, _, info = env.step(2)
    errors = np.array([9.7, 9.4, 9.1, 9, 9, 9, 9, 9, 9, 9])
    expected_reward = -np.mean(((errors - 2) / 10) ** 2) - 0.01 - 0.02
    expected_energy = env.plant.turbine.power(np.full(10, 8), yaw=errors).sum() / 3600
    assert reward == pytest.approx(expected_reward)
    assert info["energy_wh"] == pytest.approx(expected_energy)
    assert (
        reward < -(0.7**2) - 0.03
    )  # Endpoint-only loss would miss early misalignment.


def test_motor_rest_and_continuous_motion_starts():
    config = AlignmentConfig(yaw_rate_deg_s=1, yaw_rest_seconds=2)
    actuator = YawActuator(config)
    actuator.command(1)
    assert actuator.advance(0.5).starts == 1
    assert actuator.advance(0.5).starts == 0
    actuator.command(1)
    assert actuator.advance(1).travel_deg == 0
    resumed = actuator.advance(2)
    assert resumed.travel_deg == 1
    assert resumed.moving_seconds == 1
    assert resumed.starts == 1
    slow = YawActuator(replace(config, yaw_rate_deg_s=0.03))
    slow.command(1)
    assert slow.advance(10).starts == 1
    slow.command(1)
    assert slow.advance(10).starts == 0
    slow.command(-1)
    assert slow.advance(1).travel_deg == 0  # Reversal requires a stop and rest.


def test_physical_travel_limit_and_disabled_motor(env_factory):
    env = env_factory(travel_limit_deg=0.5)
    env.reset(seed=1, options={"scenario": Scenario(270, 8, 10, 0)})
    _, _, _, _, info = env.step(2)
    assert info["yaw_travel_deg"] == pytest.approx(0.5)
    _, _, _, _, info = env.step(2)
    assert info["yaw_travel_deg"] == 0
    assert info["yaw_starts"] == 0
    obs, _ = env.reset(
        seed=1, options={"scenario": Scenario(270, 8, 10, 0, yaw_enabled=False)}
    )
    assert obs[-1] == 0
    _, _, _, _, info = env.step(2)
    assert info["applied_movement_deg"] == 0


def test_reward_truth_never_enters_policy_history(env_factory, monkeypatch):
    env = env_factory()
    # Construct indistinguishable sensors with different hidden error. Flat power
    # makes this explicitly unobservable, as it can be under power curtailment.
    monkeypatch.setattr(
        env.plant,
        "sample",
        lambda wd, ws, heading, dt: (float(wrap180(wd - heading)), ws, 1000.0),
    )
    obs_a, _ = env.reset(seed=10, options={"scenario": Scenario(10, 8, 10, -10)})
    next_a, reward_a, _, _, _ = env.step(1)
    obs_b, _ = env.reset(seed=10, options={"scenario": Scenario(0, 8, 0, 0)})
    next_b, reward_b, _, _, _ = env.step(1)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(next_a, next_b)
    assert reward_a == pytest.approx(-0.64)
    assert reward_b == 0


def test_persistent_bias_and_causal_history(env_factory):
    env = env_factory()
    obs, _ = env.reset(
        seed=4, options={"scenario": Scenario(270, 8, 10, 5, encoder_bias_deg=2)}
    )
    errors = np.rad2deg(
        np.arctan2(obs.reshape(-1, 10)[:, 0], obs.reshape(-1, 10)[:, 1])
    )
    np.testing.assert_allclose(errors, 13, atol=1e-5)
    next_obs, _, _, _, _ = env.step(2)
    np.testing.assert_array_equal(next_obs[:-10], obs[10:])


def test_paired_noise_is_identical_despite_different_actions(env_factory):
    env = env_factory(direction_noise_std_deg=1, encoder_noise_std_deg=0.1)
    scenario = Scenario(270, 8, 10, 5)
    obs_a, _ = env.reset(seed=4, options={"scenario": scenario})
    _, _, _, _, a = env.step(2)
    obs_b, _ = env.reset(seed=4, options={"scenario": scenario})
    _, _, _, _, b = env.step(0)
    np.testing.assert_array_equal(obs_a, obs_b)
    noise_a = a["trace"]["measured_error_deg"] - a["trace"]["true_error_deg"]
    noise_b = b["trace"]["measured_error_deg"] - b["trace"]["true_error_deg"]
    np.testing.assert_allclose(noise_a, noise_b, atol=1e-10)


def test_time_limit_and_reset_clear_history(env_factory):
    env = env_factory()
    with pytest.raises(gym.error.ResetNeeded):
        env.step(1)
    obs, _ = env.reset(seed=3, options={"scenario": Scenario(270, 8, 0, 0)})
    for step in range(3):
        _, _, terminated, truncated, info = env.step(1)
        assert not terminated
        assert truncated == (step == 2)
    assert info["scored_seconds"] == 30
    assert info["episode_metrics"]["energy_wh"] == pytest.approx(696000 * 30 / 3600)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(1)
    reset_obs, _ = env.reset(seed=3, options={"scenario": Scenario(270, 8, 0, 0)})
    np.testing.assert_array_equal(obs, reset_obs)
    with pytest.raises(ValueError):
        env.step(4)


def test_default_scenarios_do_not_memorize_one_heading(env_factory):
    env = env_factory()
    rows = [env.reset(seed=seed)[1]["scenario"] for seed in range(8)]
    assert np.ptp([row["wind_direction_deg"] for row in rows]) > 180
    assert not np.allclose(
        [r["initial_error_deg"] for r in rows], [r["direction_bias_deg"] for r in rows]
    )


def test_numerical_failure_invalidates_episode(env_factory, monkeypatch):
    env = env_factory()
    env.reset(seed=1)
    monkeypatch.setattr(env.plant, "sample", lambda *args: (float("nan"), 8, 1000))
    with pytest.raises(FloatingPointError):
        env.step(1)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(1)


def test_sensor_saturation_is_explicit(env_factory, monkeypatch):
    env = env_factory()
    env.reset(seed=1)
    monkeypatch.setattr(
        env.sensors, "sample", lambda *args: np.array([0, 0, 1000, 1e9, 10])
    )
    obs, _, _, _, info = env.step(1)
    assert env.observation_space.contains(obs)
    assert info["observation_clip_count"] == 3
    assert info["sensor_out_of_range_count"] == env.config.substeps


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decision_seconds": 0},
        {"simulation_seconds": 3},
        {"history_windows": 1.5},
        {"episode_seconds": 21},
        {"direction_noise_std_deg": -1},
        {"yaw_rate_deg_s": float("nan")},
        {"wind_speed_min": 10, "wind_speed_max": 8},
        {"wind_speed_amplitude_mps": 9},
    ],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        AlignmentConfig(**kwargs)


def test_unknown_config_and_baseline_calibration(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("yaw_step_typo: 1\n")
    with pytest.raises(ValueError, match="Unknown"):
        AlignmentConfig.load(path)
    observation = np.zeros(120, dtype=np.float32)
    observation[-10:-8] = [np.sin(np.deg2rad(10)), np.cos(np.deg2rad(10))]
    observation[-1] = 1
    assert FilteredYawController().predict(observation)[0] == 2
    assert FilteredYawController(calibration_deg=10).predict(observation)[0] == 1


def test_summary_uses_paired_scenarios_and_reproducible_uncertainty():
    rows = []
    for label, error, energy in [("filtered_yaw", 10, 100), ("ppo", 5, 105)]:
        for seed in range(3):
            rows.append(
                {
                    "controller": label,
                    "seed": seed,
                    "mean_abs_error_deg": error,
                    "p95_abs_error_deg": error,
                    "energy_wh": energy,
                    "yaw_travel_deg": 1,
                    "yaw_starts": 1,
                    "within_tolerance_fraction": 0,
                    "reward": -error,
                }
            )
    result = summarize(rows)["controllers"]["ppo"]
    assert result["paired_error_delta_deg"]["ci95"] == [-5, -5]
    assert result["paired_energy_gain_pct"]["mean"] == pytest.approx(5)


def test_training_checkpoint_and_frozen_evaluation(tmp_path):
    from stable_baselines3 import PPO

    from WindGym.yaw_alignment.experiment import evaluate, train

    config = AlignmentConfig(episode_seconds=20, history_windows=2)
    run = train(
        config,
        tmp_path / "run",
        steps=64,
        n_envs=1,
        rollout_steps=32,
        eval_every=32,
        validation_episodes=1,
    )
    metadata = json.loads((run / "training.json").read_text())
    with (run / "validation.csv").open() as f:
        validation = list(csv.DictReader(f))
    assert metadata["actual_timesteps"] == 64
    assert metadata["status"] == "complete"
    assert len(metadata["source_sha256"]) == 8
    assert [int(row["ppo_epochs"]) for row in validation] == [10, 20]
    assert (run / "model.zip").is_file()
    checkpoint_before = (run / "best_model.zip").read_bytes()
    result = evaluate(run, tmp_path / "evaluation", episodes=2)
    assert set(result["controllers"]) == {"ppo", "filtered_yaw", "hold"}
    assert (run / "best_model.zip").read_bytes() == checkpoint_before
    assert PPO.load(run / "best_model.zip").observation_space.shape == (20,)
    with pytest.raises(ValueError, match="overlap"):
        evaluate(run, tmp_path / "invalid-test", episodes=1, seed=1_000_000)
    with pytest.raises(FileExistsError):
        train(config, run, steps=64)
