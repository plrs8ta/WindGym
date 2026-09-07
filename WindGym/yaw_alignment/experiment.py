"""Reproducible training and paired evaluation without external tracking services."""

import csv
import hashlib
import json
import platform
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .config import AlignmentConfig, Scenario
from .controllers import FilteredYawController, HoldController
from .env import YawAlignmentEnv


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluation_scenario(config, seed, index):
    scenario = Scenario.sample(config, np.random.default_rng(seed))
    # A quarter of the cases check an already calibrated direction sensor.
    if index % 4 == 0:
        scenario = replace(scenario, direction_bias_deg=0.0, encoder_bias_deg=0.0)
    return scenario


def evaluate_controller(
    controller, config, seeds, *, label="controller", trace_first=False
):
    """Policy sees obs only. Truth is consumed strictly by this evaluator."""
    episodes, traces = [], []
    env = YawAlignmentEnv(config)
    try:
        for index, seed in enumerate(seeds):
            scenario = evaluation_scenario(config, seed, index)
            observation, _ = env.reset(seed=seed, options={"scenario": scenario})
            if isinstance(controller, (FilteredYawController, HoldController)):
                controller.reset()
            errors = []
            for step in range(config.episode_steps):
                action, _ = controller.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(
                    int(np.asarray(action).item())
                )
                errors.extend(np.abs(info["trace"]["true_error_deg"]).tolist())
                if trace_first and index == 0:
                    for j in range(config.substeps):
                        traces.append(
                            {
                                "controller": label,
                                "seed": seed,
                                "decision": step,
                                "requested_action": int(action),
                                "interval_reward": reward,
                                **{k: float(v[j]) for k, v in info["trace"].items()},
                            }
                        )
                if terminated or truncated:
                    break
            episodes.append(
                {
                    "controller": label,
                    "seed": seed,
                    "case": "unbiased" if index % 4 == 0 else "sampled_bias",
                    "wind_direction_deg": scenario.wind_direction_deg,
                    "wind_speed_mps": scenario.wind_speed_mps,
                    "initial_error_deg": scenario.initial_error_deg,
                    "direction_bias_deg": scenario.direction_bias_deg,
                    **info["episode_metrics"],
                    "p95_abs_error_deg": float(np.percentile(errors, 95)),
                }
            )
    finally:
        env.close()
    return episodes, traces


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "ci95": None}
    rng = np.random.default_rng(2026)
    means = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return {"mean": mean, "ci95": np.percentile(means, [2.5, 97.5]).tolist()}


def summarize(episodes):
    labels = list(dict.fromkeys(row["controller"] for row in episodes))
    baseline = {
        row["seed"]: row for row in episodes if row["controller"] == "filtered_yaw"
    }
    metrics = (
        "mean_abs_error_deg",
        "p95_abs_error_deg",
        "energy_wh",
        "yaw_travel_deg",
        "yaw_starts",
        "within_tolerance_fraction",
        "reward",
    )
    summary = {}
    for label in labels:
        rows = [row for row in episodes if row["controller"] == label]
        summary[label] = {k: float(np.mean([row[k] for row in rows])) for k in metrics}
        if baseline:
            summary[label]["paired_error_delta_deg"] = _mean_ci(
                [
                    row["mean_abs_error_deg"]
                    - baseline[row["seed"]]["mean_abs_error_deg"]
                    for row in rows
                ]
            )
            gains = [
                100 * (row["energy_wh"] / baseline[row["seed"]]["energy_wh"] - 1)
                for row in rows
                if baseline[row["seed"]]["energy_wh"] > 0
            ]
            summary[label]["paired_energy_gain_pct"] = (
                _mean_ci(gains) if gains else None
            )
    return {
        "controllers": summary,
        "uncertainty": "Paired bootstrap over scenario means, conditional on this trained policy; not training-seed uncertainty.",
        "alignment_status": "Metrics only: engineering acceptance thresholds have not been set.",
    }


def train(
    config,
    output,
    *,
    steps=100_000,
    seed=42,
    n_envs=4,
    rollout_steps=512,
    eval_every=10_000,
    validation_episodes=5,
    validation_seed=1_000_000,
):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    settings = {
        "steps": steps,
        "n_envs": n_envs,
        "rollout_steps": rollout_steps,
        "eval_every": eval_every,
        "validation_episodes": validation_episodes,
    }
    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if seed < 0 or validation_seed < 0:
        raise ValueError("Seeds must be nonnegative")
    if rollout_steps * n_envs < 2:
        raise ValueError("PPO needs at least two rollout transitions")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config.save(output / "config.json")
    torch.set_num_threads(1)
    validation_seeds = list(
        range(validation_seed, validation_seed + validation_episodes)
    )
    rows = []

    class ValidationCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = eval_every
            self.best_score = -np.inf

        def evaluate(self):
            result, _ = evaluate_controller(self.model, config, validation_seeds)
            score = float(np.mean([r["reward"] for r in result]))
            rows.append(
                {
                    "timesteps": self.num_timesteps,
                    "ppo_epochs": self.model._n_updates,
                    "mean_reward": score,
                    "mean_abs_error_deg": float(
                        np.mean([r["mean_abs_error_deg"] for r in result])
                    ),
                }
            )
            write_csv(output / "validation.csv", rows)
            if score > self.best_score:
                self.best_score = score
                self.model.save(output / "best_model")
            print(
                f"Validation at {self.num_timesteps} decisions: reward={score:.3f}, "
                f"error={rows[-1]['mean_abs_error_deg']:.3f} deg",
                flush=True,
            )

        def _on_rollout_start(self):
            # The previous rollout has now been optimized. on_step/on_rollout_end
            # run before that update and would select a stale (initially untrained) policy.
            if self.num_timesteps >= self.next_eval:
                self.evaluate()
                self.next_eval = self.num_timesteps + eval_every

        def _on_step(self):
            return True

        def _on_training_end(self):
            # No next rollout exists to trigger validation of the final update.
            self.evaluate()

    batch_size = min(64, rollout_steps * n_envs)
    while (rollout_steps * n_envs) % batch_size:
        batch_size -= 1
    metadata = {
        "format_version": 1,
        "status": "running",
        "backend": "WindGym PyWake",
        "turbine": "V80",
        "seed": seed,
        "requested_timesteps": steps,
        "n_envs": n_envs,
        "rollout_steps": rollout_steps,
        "validation_seeds": validation_seeds,
        "selection_metric": "mean validation episode reward",
        "observation_features": 10,
        "history_windows": config.history_windows,
        "normalization": "fixed engineering scales",
        "hyperparameters": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "batch_size": batch_size,
            "n_epochs": 10,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "actor_layers": [64, 64],
            "critic_layers": [64, 64],
            "activation": "tanh",
            "device": "cpu",
        },
        "versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("stable_baselines3", "gymnasium", "numpy", "py_wake")
            },
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    write_json(output / "training.json", metadata)
    env = None
    started = time.perf_counter()
    try:
        env = DummyVecEnv(
            [lambda: Monitor(YawAlignmentEnv(config)) for _ in range(n_envs)]
        )
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            device="cpu",
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            n_steps=rollout_steps,
            batch_size=batch_size,
            n_epochs=10,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs={
                "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                "activation_fn": torch.nn.Tanh,
            },
            verbose=0,
        )
        model.learn(total_timesteps=steps, callback=ValidationCallback())
        model.save(output / "model")
        metadata.update(status="complete", actual_timesteps=model.num_timesteps)
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["elapsed_wall_seconds"] = time.perf_counter() - started
        write_json(output / "training.json", metadata)
        if env is not None:
            env.close()
    return output


def evaluate(
    run,
    output,
    *,
    episodes=20,
    seed=2_000_000,
    calibration_deg=None,
    filter_seconds=20.0,
    deadband_deg=2.0,
):
    import torch
    from stable_baselines3 import PPO

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    run = Path(run)
    config = AlignmentConfig.load(run / "config.json")
    metadata = json.loads((run / "training.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError(
            "Training run is incomplete; finish a run before final evaluation"
        )
    seeds = list(range(seed, seed + episodes))
    if set(seeds) & set(metadata["validation_seeds"]):
        raise ValueError(
            "Test seeds overlap the saved validation set; choose a different seed"
        )
    model_path = run / "best_model.zip"
    if not model_path.exists():
        model_path = run / "model.zip"
    torch.set_num_threads(1)
    model = PPO.load(model_path, device="cpu")
    controllers = {
        "ppo": model,
        "filtered_yaw": FilteredYawController(
            config.decision_seconds, filter_seconds, deadband_deg
        ),
        "hold": HoldController(),
    }
    if calibration_deg is not None:
        controllers["calibrated_yaw"] = FilteredYawController(
            config.decision_seconds, filter_seconds, deadband_deg, calibration_deg
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    all_episodes, all_traces = [], []
    for label, controller in controllers.items():
        rows, traces = evaluate_controller(
            controller, config, seeds, label=label, trace_first=True
        )
        all_episodes.extend(rows)
        all_traces.extend(traces)
        print(f"Evaluated {label}: {len(rows)} scenarios", flush=True)
    write_csv(output / "episodes.csv", all_episodes)
    write_csv(output / "trajectory.csv", all_traces)
    summary = summarize(all_episodes)
    summary.update(
        model=str(model_path.resolve()),
        training_seed=metadata["seed"],
        test_seeds=seeds,
        baseline={
            "filter_seconds": filter_seconds,
            "deadband_deg": deadband_deg,
            "calibration_deg": calibration_deg,
        },
    )
    write_json(output / "summary.json", summary)
    return summary
