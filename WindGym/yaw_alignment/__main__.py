"""Run with python -m WindGym.yaw_alignment {check,train,evaluate}."""

import argparse
import json

from .config import AlignmentConfig
from .env import YawAlignmentEnv
from .experiment import evaluate, train


def main():
    parser = argparse.ArgumentParser(description="Single-turbine RL yaw-alignment lab")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "check", help="Check the Gymnasium interface with the real PyWake plant"
    )
    check.add_argument("--config")
    training = sub.add_parser(
        "train", help="Train PPO and save validation-selected and final models"
    )
    training.add_argument("--config")
    training.add_argument("--output", required=True)
    training.add_argument("--steps", type=int, default=100_000)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--n-envs", type=int, default=4)
    training.add_argument("--rollout-steps", type=int, default=512)
    training.add_argument("--eval-every", type=int, default=10_000)
    training.add_argument("--validation-episodes", type=int, default=5)
    training.add_argument("--validation-seed", type=int, default=1_000_000)
    evaluation = sub.add_parser(
        "evaluate", help="Compare the frozen policy against equally informed baselines"
    )
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--episodes", type=int, default=20)
    evaluation.add_argument("--seed", type=int, default=2_000_000)
    evaluation.add_argument("--calibration-deg", type=float)
    evaluation.add_argument("--filter-seconds", type=float, default=20)
    evaluation.add_argument("--deadband-deg", type=float, default=2)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "evaluate":
        result = evaluate(**args)
        print(json.dumps(result["controllers"], indent=2))
    else:
        path = args.pop("config")
        config = AlignmentConfig.load(path) if path else AlignmentConfig()
        if command == "train":
            print(f"Models saved in {train(config, **args)}")
        else:
            from gymnasium.utils.env_checker import check_env
            from stable_baselines3.common.env_checker import check_env as sb3_check

            env = YawAlignmentEnv(config)
            try:
                check_env(env, skip_render_check=True)
                sb3_check(env)
                print(
                    f"Gymnasium and SB3 checks passed: obs={env.observation_space.shape}, actions=3"
                )
            finally:
                env.close()


if __name__ == "__main__":
    main()
